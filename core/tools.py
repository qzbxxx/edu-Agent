# core/tools.py
# 售后场景的工具集：模型通过 function calling 自己决定查订单、核政策、提交退款申请。
# 关键安全点：user_id 由后端注入，模型只能传 order_id / reason，伪造别人的订单号也查不到。
import json

from langchain_core.tools import tool

from core import refund_service


def _to_json(payload: dict) -> str:
    """工具统一返回 JSON 字符串，模型好读，节点层也好解析"""
    return json.dumps(payload, ensure_ascii=False)


def build_order_tools(user_id: str):
    """为当前登录用户构造工具列表"""

    @tool
    def list_my_orders() -> str:
        """列出当前登录用户全部订单的订单号、课程、金额和状态。用户不知道自己订单号时先调用它。"""
        return _to_json({"ok": True, "orders": refund_service.list_orders(user_id)})

    @tool
    def query_order(order_id: str) -> str:
        """查询指定订单的详情：课程、金额、支付时间、学习进度、订单状态，并返回退费政策判定。"""
        return _to_json(refund_service.query_order(user_id, order_id))

    @tool
    def check_refund_policy(order_id: str) -> str:
        """按 7 天无理由与学习进度 10% 的规则，判断该订单能否自动退款、是否需要人工审批、可退金额。"""
        return _to_json(refund_service.check_refund_policy(user_id, order_id))

    @tool
    def submit_refund_request(order_id: str, reason: str = "") -> str:
        """
        提交退款申请（只登记申请，不会立即打款）。
        仅在用户明确要求退款时调用；同一个订单不要重复提交。
        """
        return _to_json(refund_service.submit_refund_request(user_id, order_id, reason))

    return [list_my_orders, query_order, check_refund_policy, submit_refund_request]
