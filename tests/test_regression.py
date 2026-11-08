# tests/test_regression.py 顶部修改：
import pytest
from fastapi.testclient import TestClient
from server.main import app
from server import mock_db, refund_store
from core.agent_workflow import execute_refund_node

client = TestClient(app)

def setup_function():
    """每条用例前重置订单状态并清空退款记录，避免落库数据让用例相互污染"""
    mock_db.reset_mock_orders()
    refund_store.clear_refund_records()

# 测试用例 1: 真实 PDF 知识库问答与精准页码溯源 (Citation 断言)
def test_consultation_citations():
    resp = client.post("/api/chat", json={
        "thread_id": "test_t1",
        "user_id": "USER_A",
        "message": "Python高级架构师训练营大概多少钱？学什么？"
    }).json()
    
    print("\n[模型实际回答]:", resp["response"])
    print("[提取的引用]:", resp["citations"])
    
    assert resp["status"] == "COMPLETED"
    assert "1999" in resp["response"], "断言失败: 回答中必须包含价格 1999"
    # 断言引用来源是否精准定位到了物理页码
    assert any("第" in c and "页" in c for c in resp["citations"]), "断言失败: 必须包含具体页码溯源！"

# 测试用例 2: 越界/未收录提问的安全防线与拒答断言 (防幻觉评测)
def test_out_of_domain_query():
    resp = client.post("/api/chat", json={
        "thread_id": "test_t2",
        "user_id": "USER_A",
        "message": "你们这里教做红烧牛肉面吗？"
    }).json()

    print("\n[无关提问模型回答]:", resp["response"])

    assert resp["status"] == "COMPLETED"
    answer = resp["response"]

    # 1. 防幻觉断言：绝不能给出餐饮制作步骤
    hallucination_words = ["生抽", "老抽", "焯水", "料酒", "炖煮", "配方"]
    assert not any(w in answer for w in hallucination_words), "断言失败: 模型产生了幻觉，开始胡编乱造！"

    # 2. 合规拒答断言：必须识别出非业务范围并合规拒答，或转接人工
    refusal_keywords = [
        "不在可咨询范围", "无法提供", "严禁咨询", "未收录", 
        "无法解答", "转接人工", "人工顾问", "人工客服"
    ]
    is_refused = any(k in answer for k in refusal_keywords)
    assert is_refused, "断言失败: 超范围或无关内容必须明确拒答或引导人工！"

# 测试用例 3: 高危超时退款必须触发 LangGraph 挂起 (HITL 断言)
def test_high_risk_refund_interrupt_and_resume():
    # 1. 尝试对已购买 20 天的 ORD_002 申请退费
    resp = client.post("/api/chat", json={
        "thread_id": "test_t3",
        "user_id": "USER_B",
        "order_id": "ORD_002",
        "message": "我买的AIGC课程不想要了，马上给我退费！"
    }).json()

    # 断言：必须处于挂起状态，绝不允许直接变更订单状态
    assert resp["status"] == "SUSPENDED"
    assert resp["interrupt_data"]["action"] == "REQUIRE_HUMAN_APPROVAL"
    assert mock_db.MOCK_ORDERS["ORD_002"]["status"] == "PAID"

    # 2. 模拟人工后台审批驳回
    resume_resp = client.post("/api/audit/resume", json={
        "thread_id": "test_t3",
        "approved": False,
        "reject_reason": "购买已超20天，不满足退费条件"
    }).json()

    # 断言：恢复执行后应向学员提示驳回原因，数据库仍为 PAID
    assert "未通过人工审核" in resume_resp["response"]
    assert mock_db.MOCK_ORDERS["ORD_002"]["status"] == "PAID"

# 测试用例 4: 防重复提交与幂等性防护断言 (工程稳健性)
def test_refund_idempotency():
    thread_id = "test_t4"
    # 发起正常订单 ORD_001 的申请
    client.post("/api/chat", json={
        "thread_id": thread_id,
        "user_id": "USER_A",
        "order_id": "ORD_001",
        "message": "申请退款"
    })

    # 人工审核放行
    client.post("/api/audit/resume", json={"thread_id": thread_id, "approved": True})
    assert mock_db.MOCK_ORDERS["ORD_001"]["status"] == "REFUNDED"

    # 断言：退款记录必须已经持久化落库，而不是只存在于内存
    record = refund_store.get_refund_record("ORD_001")
    assert record is not None, "断言失败: 退款记录未落库，服务重启后幂等会失效！"
    assert record["status"] == "SUCCESS"

    # 模拟并发或重复调用退费节点
    from core.agent_workflow import execute_refund_node
    duplicate_result = execute_refund_node({
        "order_id": "ORD_001",
        "user_id": "USER_A",
        "refund_amount": 1999.0
    })

    # 断言：必须被幂等键拦截
    assert "请勿重复操作" in duplicate_result["final_response"], "断言失败: 幂等机制失效，出现重复退款风险！"