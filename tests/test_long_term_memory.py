# tests/test_long_term_memory.py
# 长期记忆的确定性测试：不调用大模型，直接验证存储、检索、遗忘与工具层
import json
import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from core import memory_manager
from core.memory_tools import build_memory_tools
from server.main import app

TEST_USER = "test_mem_main"
OTHER_USER = "test_mem_other"

client = TestClient(app)


def setup_function():
    """每条用例前清空两个测试用户的画像与记忆，避免互相污染"""
    memory_manager.clear_user_data(TEST_USER)
    memory_manager.clear_user_data(OTHER_USER)


def test_add_and_list_memories():
    memory_manager.add_user_memory(TEST_USER, "喜欢周末上课", kind="note", importance=0.6)
    memory_manager.add_user_memory(TEST_USER, "准备转行做后端", kind="note", importance=0.8)

    memories = memory_manager.list_user_memories(TEST_USER)

    assert len(memories) == 2
    assert {item["content"] for item in memories} == {"喜欢周末上课", "准备转行做后端"}


def test_duplicate_content_is_merged():
    memory_manager.add_user_memory(TEST_USER, "喜欢周末上课", importance=0.4)
    memory_manager.add_user_memory(TEST_USER, "喜欢周末上课", importance=0.9)

    memories = memory_manager.list_user_memories(TEST_USER)

    assert len(memories) == 1
    # 重复写入不新增行，但重要性取更高的一次
    assert memories[0]["importance"] == pytest.approx(0.9)


def test_search_returns_relevant_memory_first():
    memory_manager.add_user_memory(TEST_USER, "订单 ORD_001 退款成功，金额 1999 元", kind="refund")
    memory_manager.add_user_memory(TEST_USER, "喜欢周末上课", kind="note")
    memory_manager.add_user_memory(TEST_USER, "正在准备面试", kind="note")

    results = memory_manager.search_user_memories(TEST_USER, "退款", top_k=3)

    assert results
    assert "退款" in results[0]["content"]
    assert results[0]["match_ratio"] > 0


def test_search_without_overlap_returns_empty():
    memory_manager.add_user_memory(TEST_USER, "喜欢周末上课", kind="note")

    assert memory_manager.search_user_memories(TEST_USER, "今天天气", top_k=3) == []


def test_users_are_isolated():
    memory_manager.add_user_memory(TEST_USER, "订单 ORD_001 退款成功", kind="refund")

    assert memory_manager.search_user_memories(OTHER_USER, "退款", top_k=3) == []
    assert memory_manager.list_user_memories(OTHER_USER) == []


def test_delete_memory_checks_owner():
    memory_id = memory_manager.add_user_memory(TEST_USER, "喜欢周末上课", kind="note")

    # 别的用户不能删
    assert memory_manager.delete_user_memory(OTHER_USER, memory_id) is False
    # 本人可以删
    assert memory_manager.delete_user_memory(TEST_USER, memory_id) is True
    assert memory_manager.list_user_memories(TEST_USER) == []


def test_prune_forgets_old_low_importance_memory():
    old_low = memory_manager.add_user_memory(TEST_USER, "旧的低重要性记忆", importance=0.3)
    old_high = memory_manager.add_user_memory(TEST_USER, "旧的高重要性记忆", importance=0.9)

    # 直接把时间改到 2020 年，模拟很久以前写入
    conn = sqlite3.connect(memory_manager.MEM_DB_PATH)
    conn.execute(
        "UPDATE user_memories SET created_at = '2020-01-01 00:00:00' WHERE id IN (?, ?)",
        (old_low, old_high),
    )
    conn.commit()
    conn.close()

    deleted = memory_manager.prune_user_memories(TEST_USER, days=90, min_importance=0.8)

    assert deleted == 1
    remaining = memory_manager.list_user_memories(TEST_USER)
    assert [item["id"] for item in remaining] == [old_high]


def test_profile_change_writes_audit_memory():
    memory_manager.save_user_long_term_memory(TEST_USER, {"skill_level": "零基础"})
    memory_manager.save_user_long_term_memory(TEST_USER, {"skill_level": "有基础"})

    changes = memory_manager.list_user_memories(TEST_USER, kind="profile_change")

    assert len(changes) == 1
    assert "零基础" in changes[0]["content"]
    assert "有基础" in changes[0]["content"]


def test_format_user_profile():
    text = memory_manager.format_user_profile({
        "learning_goal": "转行就业",
        "skill_level": "零基础",
        "preferences": ["周末上课", "中文答疑"],
    })

    assert "转行就业" in text
    assert "零基础" in text
    assert "周末上课" in text


def test_memory_tools_roundtrip():
    tools = {tool.name: tool for tool in build_memory_tools(TEST_USER)}

    saved = json.loads(tools["save_memory"].invoke({"content": "希望周末上课", "importance": 0.7}))
    assert saved["ok"] is True

    found = json.loads(tools["search_memory"].invoke({"query": "周末", "top_k": 3}))
    assert found["ok"] is True
    assert found["memories"][0]["content"] == "希望周末上课"

    profile = json.loads(tools["get_user_profile"].invoke({}))
    assert profile["ok"] is True
    assert profile["profile"]["learning_goal"] == "未知"


def test_memory_api_roundtrip():
    user_id = f"api_mem_{uuid.uuid4().hex[:6]}"
    memory_id = memory_manager.add_user_memory(user_id, "喜欢周末上课", kind="note", importance=0.6)

    response = client.get(f"/api/memory/{user_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["profile"]["learning_goal"] == "未知"
    assert len(data["memories"]) == 1

    deleted = client.delete(f"/api/memory/{user_id}/items/{memory_id}")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "DELETED"

    missing = client.delete(f"/api/memory/{user_id}/items/{memory_id}")
    assert missing.status_code == 404
