# tests/test_tool_agent.py
# 用脚本化的假模型验证工具循环，不产生真实大模型调用
import core.agent_workflow as workflow_module
from langchain_core.messages import AIMessage, HumanMessage
from server import mock_db, refund_store


def setup_function():
    mock_db.reset_mock_orders()
    refund_store.clear_refund_records()


class ScriptedLLM:
    """按顺序返回预设消息，模拟模型的多轮工具调用"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def build_state(text):
    return {
        "messages": [HumanMessage(content=text)],
        "user_id": "USER_A",
        "order_id": "ORD_001",
        "citations": [],
        "need_human_approval": False,
        "refund_requested": False,
        "tool_events": [],
    }


def test_agent_calls_tools_then_submits(monkeypatch):
    script = [
        AIMessage(content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "ORD_001"}, "id": "call_1"}
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "submit_refund_request", "args": {"order_id": "ORD_001", "reason": "不想学了"}, "id": "call_2"}
        ]),
        AIMessage(content="已为您提交退款申请。"),
    ]
    fake = ScriptedLLM(script)
    monkeypatch.setattr(workflow_module, "llm", fake)

    result = workflow_module.after_sales_agent_node(build_state("我要退款"))

    assert [event["tool"] for event in result["tool_events"]] == ["query_order", "submit_refund_request"]
    assert result["refund_requested"] is True
    assert result["need_human_approval"] is False
    assert result["refund_amount"] == 1999.0
    assert result["final_response"]


def test_agent_falls_back_to_deterministic_submit(monkeypatch):
    # 模型只回了一句话、没有调用任何工具
    fake = ScriptedLLM([AIMessage(content="好的，我帮您看看。")])
    monkeypatch.setattr(workflow_module, "llm", fake)

    result = workflow_module.after_sales_agent_node(build_state("我要退款"))

    assert result["refund_requested"] is True
    assert result["tool_events"][-1]["tool"] == "submit_refund_request"
    assert result["tool_events"][-1].get("fallback") is True
    assert result["refund_amount"] == 1999.0


def test_agent_marks_high_risk_refund_for_approval(monkeypatch):
    # 高风险订单：模型提交成功后，节点必须带上 need_human_approval，交给 human_approval 挂起
    script = [
        AIMessage(content="", tool_calls=[
            {"name": "submit_refund_request",
             "args": {"order_id": "ORD_002", "reason": "用户要求退费"},
             "id": "call_risk"}
        ]),
        AIMessage(content="已提交退款申请，正在等待人工审核。"),
    ]
    fake = ScriptedLLM(script)
    monkeypatch.setattr(workflow_module, "llm", fake)

    state = build_state("我要退费")
    state["user_id"] = "USER_B"
    state["order_id"] = "ORD_002"

    result = workflow_module.after_sales_agent_node(state)

    assert result["refund_requested"] is True
    assert result["need_human_approval"] is True
    assert result["refund_amount"] == 2999.0
