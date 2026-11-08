# tests/test_memory_rewrite.py
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)

def test_multi_turn_query_rewrite():
    print("\n" + "="*65)
    print("🧠 多轮上下文指代消解 (Query Rewrite) 自动化实测")
    print("="*65)

    thread_id = "test_memory_session_888"

    # 第一轮提问：建立主题实体
    q1 = "请问 Python 高级架构师训练营售价是多少？"
    print(f"\n👤 第 1 轮提问: {q1}")
    resp1 = client.post("/api/chat", json={
        "thread_id": thread_id,
        "user_id": "USER_A",
        "message": q1
    }).json()

    print(f"🤖 第 1 轮回答: {resp1['response'][:60]}...")
    assert "1999" in resp1["response"]

    # 第二轮追问：使用模糊代词“它”，不出现任何 Python 字眼！
    q2 = "它适合什么基础的人学？主要讲哪些技术栈？"
    print(f"\n👤 第 2 轮追问 (含代词): {q2}")
    resp2 = client.post("/api/chat", json={
        "thread_id": thread_id,
        "user_id": "USER_A",
        "message": q2
    }).json()

    print(f"✨【大模型自主改写的独立检索词】: 【{resp2.get('rewritten_query')}】")
    print(f"🤖 第 2 轮回答: {resp2['response']}")
    print(f"📚 来源页码: {resp2['citations']}")

    # 断言：改写后的 query 必须自动补齐主语“Python”
    assert "Python" in resp2.get("rewritten_query", ""), "断言失败: 未能将代词'它'成功消解补齐为主语！"
    # 断言：回答必须准确命中第一页中关于适合人群的大纲描述
    assert "1年" in resp2["response"] or "分布式" in resp2["response"]

if __name__ == "__main__":
    test_multi_turn_query_rewrite()