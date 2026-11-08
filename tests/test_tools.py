# tests/test_tools.py
# 工具层的确定性测试：不调用大模型，直接验证订单查询、政策判定与提交申请
import json

from server import mock_db, refund_store
from core.tools import build_order_tools


def setup_function():
    """每条用例前重置订单与退款流水，避免落库数据互相污染"""
    mock_db.reset_mock_orders()
    refund_store.clear_refund_records()


def call_tool(user_id, name, args):
    tools = {tool.name: tool for tool in build_order_tools(user_id)}
    raw = tools[name].invoke(args)
    return json.loads(raw)


def test_query_own_order_returns_policy():
    result = call_tool("USER_A", "query_order", {"order_id": "ORD_001"})

    assert result["ok"] is True
    assert result["order"]["price"] == 1999.0
    assert result["order"]["refund_policy"]["need_human_approval"] is False


def test_query_other_users_order_is_rejected():
    result = call_tool("USER_A", "query_order", {"order_id": "ORD_002"})

    assert result["ok"] is False
    assert "未查询到订单" in result["error"]


def test_check_policy_marks_high_risk_order():
    result = call_tool("USER_B", "check_refund_policy", {"order_id": "ORD_002"})

    assert result["ok"] is True
    assert result["need_human_approval"] is True
    assert result["refund_amount"] == 2999.0
    assert result["reasons"], "高风险订单必须给出原因，供人工审核参考"


def test_submit_refund_and_duplicate():
    first = call_tool("USER_A", "submit_refund_request", {"order_id": "ORD_001", "reason": "不想要了"})
    assert first["ok"] is True
    assert first["need_human_approval"] is False

    # 模拟执行阶段已经落库，此时再次申请必须被幂等挡住
    refund_store.reserve_refund("ORD_001", "USER_A", 1999.0)
    second = call_tool("USER_A", "submit_refund_request", {"order_id": "ORD_001"})

    assert second["ok"] is False
    assert "请勿重复操作" in second["error"]


def test_list_orders_only_returns_own():
    result = call_tool("USER_B", "list_my_orders", {})

    assert result["ok"] is True
    assert [order["order_id"] for order in result["orders"]] == ["ORD_002"]
