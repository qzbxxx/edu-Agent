# server/main.py
import os
import sys

# 自动定位项目根目录并加入 Python 寻路列表
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.messages import HumanMessage
from core.agent_workflow import app_graph
from langgraph.types import Command
from server import mock_db, refund_store
import uvicorn

app = FastAPI(title="Education Agent Engine")

class ChatRequest(BaseModel):
    thread_id: str
    user_id: str
    message: str
    order_id: Optional[str] = None

class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool
    reject_reason: Optional[str] = None

@app.post("/api/chat")
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    
    # 将用户输入包装为标准的 HumanMessage 传入，状态机会自动通过 checkpointer 累加
    result = app_graph.invoke({
        "messages": [HumanMessage(content=req.message)],
        "user_id": req.user_id,
        "order_id": req.order_id,
        "citations": [],
        "need_human_approval": False,
        "refund_requested": False,
        "tool_events": []
    }, config=config)
    
    # 检查状态机是否因触发风控而被挂起 (HITL)
    state = app_graph.get_state(config)
    if state.next:
        interrupt_info = state.tasks[0].interrupts[0].value
        return {
            "status": "SUSPENDED",
            "message": "当前操作触发风险控制，已挂起等待人工审核",
            "interrupt_data": interrupt_info,
            "tool_events": result.get("tool_events", [])
        }

    return {
        "status": "COMPLETED",
        "response": result.get("final_response"),
        "rewritten_query": result.get("rewritten_query"),
        "citations": result.get("citations", []),
        "tool_events": result.get("tool_events", [])
    }

@app.post("/api/audit/resume")
def resume_audit(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    state = app_graph.get_state(config)
    
    if not state.next:
        raise HTTPException(status_code=400, detail="该会话未处于挂起等待状态，无法恢复")
    
    result = app_graph.invoke(
        Command(resume={"approved": req.approved, "reject_reason": req.reject_reason}),
        config=config
    )
    
    return {
        "status": "COMPLETED",
        "response": result.get("final_response")
    }

# ---------- 调试接口：方便在 /docs 里直接测工具调用与 HITL ----------
# 默认开启；部署到线上时设置 ENABLE_DEBUG_API=0 关闭
DEBUG_API_ENABLED = os.getenv("ENABLE_DEBUG_API", "1") == "1"


def require_debug_api():
    if not DEBUG_API_ENABLED:
        raise HTTPException(status_code=404, detail="调试接口已关闭（ENABLE_DEBUG_API=0）")


@app.get("/api/health")
def health():
    """健康检查：确认服务已启动"""
    return {"status": "ok", "orders": len(mock_db.MOCK_ORDERS)}


@app.get("/api/debug/orders", dependencies=[Depends(require_debug_api)])
def debug_orders(user_id: Optional[str] = None):
    """查看模拟订单；可按 user_id 过滤"""
    orders = []
    for order in mock_db.MOCK_ORDERS.values():
        if user_id and order["user_id"] != user_id:
            continue
        orders.append({
            "order_id": order["order_id"],
            "user_id": order["user_id"],
            "course_name": order["course_name"],
            "price": order["price"],
            "status": order["status"],
            "learned_progress": order["learned_progress"],
            "pay_time": order["pay_time"].strftime("%Y-%m-%d %H:%M"),
        })
    return {"count": len(orders), "orders": orders}


@app.get("/api/debug/refunds", dependencies=[Depends(require_debug_api)])
def debug_refunds(limit: int = 50):
    """查看落库的退款流水（不是内存）"""
    records = refund_store.list_refund_records(limit)
    return {"count": len(records), "refunds": records}


@app.post("/api/debug/reset", dependencies=[Depends(require_debug_api)])
def debug_reset():
    """重置演示数据：订单恢复初始状态，退款记录清空"""
    mock_db.reset_mock_orders()
    refund_store.clear_refund_records()
    return {
        "status": "ok",
        "message": "订单已重置，退款记录已清空；已挂起的会话请换新的 thread_id",
    }


@app.get("/api/debug/thread/{thread_id}", dependencies=[Depends(require_debug_api)])
def debug_thread(thread_id: str):
    """查看某个会话的当前状态、挂起信息和本轮工具调用"""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = app_graph.get_state(config)
    values = snapshot.values or {}

    interrupt_value = None
    if snapshot.next and snapshot.tasks:
        interrupts = snapshot.tasks[0].interrupts
        if interrupts:
            interrupt_value = interrupts[0].value

    return {
        "thread_id": thread_id,
        "next": list(snapshot.next),
        "interrupt": interrupt_value,
        "state": {
            "intent": values.get("intent"),
            "rewritten_query": values.get("rewritten_query"),
            "refund_requested": values.get("refund_requested"),
            "need_human_approval": values.get("need_human_approval"),
            "approval_granted": values.get("approval_granted"),
            "refund_amount": values.get("refund_amount"),
            "citations": values.get("citations", []),
            "tool_events": values.get("tool_events", []),
            "final_response": values.get("final_response"),
            "message_count": len(values.get("messages", [])),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)