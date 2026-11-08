# tests/test_api.py
# 通过 FastAPI TestClient 在 HTTP 层验证：工具调用、HITL 挂起/恢复、调试接口。
# 全程使用脚本化的假模型，不产生真实大模型费用。
import core.agent_workflow as workflow_module
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from server import mock_db, refund_store
from server.main import app

client = TestClient(app)


def setup_function():
    """每条用例前重置订单与退款流水，避免互相污染"""
    mock_db.reset_mock_orders()
    refund_store.clear_refund_records()


class ScriptedLLM:
    """按顺序返回预设消息：第 1 次给 router 分类，后面几次给工具循环"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def build_refund_script(order_id, reason="用户申请退款"):
    return [
        AIMessage(content="refund"),
        AIMessage(content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": order_id}, "id": "call_query"}
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "submit_refund_request",
             "args": {"order_id": order_id, "reason": reason},
             "id": "call_submit"}
        ]),
        AIMessage(content="已提交退款申请。"),
    ]


def test_chat_auto_refund_returns_success(monkeypatch):
    monkeypatch.setattr(workflow_module, "llm", ScriptedLLM(build_refund_script("ORD_001")))

    response = client.post("/api/chat", json={
        "thread_id": "api-auto-refund",
        "user_id": "USER_A",
        "order_id": "ORD_001",
        "message": "帮我退款",
    }).json()

    assert response["status"] == "COMPLETED"
    assert "退款成功" in response["response"]
    assert [event["tool"] for event in response["tool_events"]] == [
        "query_order",
        "submit_refund_request",
    ]
    assert mock_db.MOCK_ORDERS["ORD_001"]["status"] == "REFUNDED"

    record = refund_store.get_refund_record("ORD_001")
    assert record is not None
    assert record["status"] == "SUCCESS"

    refunds = client.get("/api/debug/refunds").json()
    assert [item["order_id"] for item in refunds["refunds"]] == ["ORD_001"]


def test_chat_high_risk_suspend_then_reject(monkeypatch):
    monkeypatch.setattr(workflow_module, "llm", ScriptedLLM(build_refund_script("ORD_002")))

    response = client.post("/api/chat", json={
        "thread_id": "api-risk",
        "user_id": "USER_B",
        "order_id": "ORD_002",
        "message": "我要退费",
    }).json()

    assert response["status"] == "SUSPENDED"
    assert response["interrupt_data"]["order_id"] == "ORD_002"
    assert response["interrupt_data"]["refund_amount"] == 2999.0
    assert all(not event.get("fallback") for event in response["tool_events"])
    assert mock_db.MOCK_ORDERS["ORD_002"]["status"] == "PAID"

    # 挂起后可以通过调试接口看到当前停在哪
    snapshot = client.get("/api/debug/thread/api-risk").json()
    assert "human_approval" in snapshot["next"]
    assert snapshot["interrupt"]["order_id"] == "ORD_002"

    resumed = client.post("/api/audit/resume", json={
        "thread_id": "api-risk",
        "approved": False,
        "reject_reason": "已购买 20 天，超过退费期限",
    }).json()

    assert "未通过人工审核" in resumed["response"]
    assert mock_db.MOCK_ORDERS["ORD_002"]["status"] == "PAID"
    assert refund_store.get_refund_record("ORD_002") is None


def test_resume_without_suspended_thread_returns_400():
    response = client.post("/api/audit/resume", json={
        "thread_id": "thread-not-suspended",
        "approved": True,
    })
    assert response.status_code == 400


def test_debug_endpoints_and_reset():
    health = client.get("/api/health").json()
    assert health["status"] == "ok"

    orders = client.get("/api/debug/orders", params={"user_id": "USER_B"}).json()
    assert [item["order_id"] for item in orders["orders"]] == ["ORD_002"]

    reset = client.post("/api/debug/reset").json()
    assert reset["status"] == "ok"
    assert mock_db.MOCK_ORDERS["ORD_001"]["status"] == "PAID"
