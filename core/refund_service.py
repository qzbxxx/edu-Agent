# core/refund_service.py
# 退费业务规则的唯一实现，供工具函数和状态机节点共同调用：
# 1) 订单查询（带归属校验）
# 2) 7 天 / 10% 政策判定
# 3) 登记退款申请（只登记，不执行打款）
# 这样规则不会在工具层和节点层各写一份。
from datetime import datetime
from typing import Any, Dict, List, Optional

from server import mock_db, refund_store

REFUND_WINDOW_DAYS = 7        # 7 个自然日内可以无理由退款
MAX_LEARNED_PROGRESS = 0.10   # 学习进度超过 10% 不能自动退款


def find_order(order_id: Optional[str], user_id: str) -> Optional[Dict[str, Any]]:
    """按订单号查订单并校验归属；查不到和不是本人都返回 None，由调用方给同一句提示"""
    order = mock_db.MOCK_ORDERS.get(order_id)
    if not order:
        return None
    if order.get("user_id") != user_id:
        return None
    return order


def order_brief(order: Dict[str, Any]) -> Dict[str, Any]:
    """订单的对外展示字段"""
    return {
        "order_id": order["order_id"],
        "course_name": order["course_name"],
        "price": order["price"],
        "status": order["status"],
        "pay_time": order["pay_time"].strftime("%Y-%m-%d %H:%M"),
    }


def list_orders(user_id: str) -> List[Dict[str, Any]]:
    """列出某个用户名下的全部订单"""
    orders = []
    for order in mock_db.MOCK_ORDERS.values():
        if order.get("user_id") == user_id:
            orders.append(order_brief(order))
    return orders


def evaluate_refund(order: Dict[str, Any], now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    按知识库第 3、4 条判定这笔订单能否自动退款

    - 支付超过 7 个自然日 → 必须人工审核
    - 学习进度超过 10% → 必须人工审核
    两条都不命中，才允许系统自动退款。
    """
    now = now or datetime.now()
    days_since_payment = (now - order["pay_time"]).days
    learned_progress = order["learned_progress"]

    overdue = days_since_payment > REFUND_WINDOW_DAYS
    progress_high = learned_progress > MAX_LEARNED_PROGRESS

    reasons = []
    if overdue:
        reasons.append(f"已购买 {days_since_payment} 天，超过 {REFUND_WINDOW_DAYS} 天无理由退款期限")
    if progress_high:
        reasons.append(f"学习进度 {learned_progress:.0%}，超过 {MAX_LEARNED_PROGRESS:.0%} 的上限")

    return {
        "refund_amount": float(order["price"]),
        "days_since_payment": days_since_payment,
        "learned_progress": learned_progress,
        "need_human_approval": overdue or progress_high,
        "reasons": reasons,
    }


def query_order(user_id: str, order_id: Optional[str]) -> Dict[str, Any]:
    """查询订单详情，同时带上政策判定，方便模型直接解释给用户"""
    order = find_order(order_id, user_id)
    if not order:
        return {"ok": False, "error": f"未查询到订单 {order_id}，请核对后重试。"}

    brief = order_brief(order)
    brief["learned_progress"] = order["learned_progress"]
    brief["refund_policy"] = evaluate_refund(order)
    return {"ok": True, "order": brief}


def check_refund_policy(user_id: str, order_id: Optional[str]) -> Dict[str, Any]:
    """只做政策判定，不产生任何写操作"""
    order = find_order(order_id, user_id)
    if not order:
        return {"ok": False, "error": f"未查询到订单 {order_id}，请核对后重试。"}

    policy = evaluate_refund(order)
    return {
        "ok": True,
        "order_id": order["order_id"],
        "course_name": order["course_name"],
        "refund_amount": policy["refund_amount"],
        "days_since_payment": policy["days_since_payment"],
        "learned_progress": policy["learned_progress"],
        "need_human_approval": policy["need_human_approval"],
        "reasons": policy["reasons"],
    }


def submit_refund_request(user_id: str, order_id: Optional[str], reason: str = "") -> Dict[str, Any]:
    """
    登记一笔退款申请

    注意：这里只生成申请，不扣款、不改订单状态。
    真正的打款在 execute_refund_node 里执行，靠数据库唯一约束保证幂等。
    """
    order = find_order(order_id, user_id)
    if not order:
        return {"ok": False, "error": f"未查询到订单 {order_id}，请核对后重试。"}
    if order.get("status") == "REFUNDED":
        return {"ok": False, "error": "该订单已经退款，无需重复申请。"}
    if refund_store.get_refund_record(order["order_id"]):
        return {"ok": False, "error": "该订单已提交过退款申请，请勿重复操作"}

    policy = evaluate_refund(order)
    amount = policy["refund_amount"]

    if policy["need_human_approval"]:
        message = (
            f"已提交退款申请：订单 {order['order_id']}（{order['course_name']}），"
            f"金额 {amount:.0f} 元。因{'；'.join(policy['reasons'])}，需人工审核后处理。"
        )
    else:
        message = (
            f"已提交退款申请：订单 {order['order_id']}（{order['course_name']}），"
            f"金额 {amount:.0f} 元，符合 7 天无理由退款条件，系统将自动处理。"
        )

    return {
        "ok": True,
        "order_id": order["order_id"],
        "course_name": order["course_name"],
        "refund_amount": amount,
        "need_human_approval": policy["need_human_approval"],
        "days_since_payment": policy["days_since_payment"],
        "learned_progress": policy["learned_progress"],
        "reasons": policy["reasons"],
        "reason": reason,
        "message": message,
    }
