# scripts/interactive_cli.py
import sys
import uuid
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from langchain_core.messages import HumanMessage
from core.agent_workflow import app_graph
from langgraph.types import Command
from server import mock_db

def run_chat():
    if not os.path.exists("./chroma_db"):
        print("❌ 错误：未检测到本地向量库 ./chroma_db！请先运行 python scripts/ingest.py")
        return

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    current_user_id = "USER_B"
    current_order_id = "ORD_002"

    print("\n" + "="*65)
    print("🎓 智学在线教育 - 生产级多轮上下文智能客服终端 (CLI)")
    print(f"会话 Thread_ID: {thread_id[:8]}... | 绑定订单: {current_order_id}")
    print("💡 体验亮点：支持连贯追问代词（'它学什么？'、'多少钱？'），自动上下文消解")
    print("="*65 + "\n")

    while True:
        try:
            user_input = input("\n👤 学员提问 > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("👋 感谢咨询，再见！")
                break

            # 1. 传入 HumanMessage
            app_graph.invoke({
                "messages": [HumanMessage(content=user_input)],
                "user_id": current_user_id,
                "order_id": current_order_id,
                "citations": [],
                "need_human_approval": False,
                "refund_requested": False,
                "tool_events": []
            }, config=config)

            # 2. 检查状态机是否因触发风控而被挂起 (HITL)
            state = app_graph.get_state(config)
            
            if state.next:
                interrupt_info = state.tasks[0].interrupts[0].value
                print("\n" + "!"*60)
                print("⚠️ [安全风控拦截] 该操作属于资金高危行为，已挂起暂停！")
                print(f"拦截原因: {interrupt_info.get('reason')}")
                print(f"目标订单: {interrupt_info.get('order_id')}")
                print("!"*60)

                # 模拟进入人工客服/主管审核后台
                print("\n--- 👨‍💼 进入人工审批控制台 ---")
                while True:
                    decision = input("主管请审批 [输入 1 同意退款 / 输入 2 驳回申请]: ").strip()
                    if decision == "1":
                        print(">>> 已同意，正在恢复 Agent 执行退费操作...")
                        res = app_graph.invoke(
                            Command(resume={"approved": True}),
                            config=config
                        )
                        print(f"\n🤖 Agent 执行结果 > {res.get('final_response')}")
                        break
                    elif decision == "2":
                        reject_reason = input("请输入驳回原因 (如: 已超7天且进度超标): ").strip()
                        if not reject_reason:
                            reject_reason = "不满足退费条件"
                        print(">>> 已驳回，正在恢复 Agent 告知学员...")
                        res = app_graph.invoke(
                            Command(resume={"approved": False, "reject_reason": reject_reason}),
                            config=config
                        )
                        print(f"\n🤖 Agent 回复学员 > {res.get('final_response')}")
                        break
                    else:
                        print("输入无效，请输入 1 或 2！")
            else:
                final_answer = state.values.get("final_response")
                rewritten_q = state.values.get("rewritten_query")
                citations = state.values.get("citations", [])
                tool_events = state.values.get("tool_events", [])

                if rewritten_q and rewritten_q != user_input:
                    print(f"✨ [意图消解重写]: 【{rewritten_q}】")

                # 展示模型这一轮到底调了哪些工具，方便观察与排错
                if tool_events:
                    print("\n🔧 [工具调用]")
                    for event in tool_events:
                        ok = event.get("result", {}).get("ok")
                        mark = "✅" if ok else "⚠️"
                        fallback = "（兜底）" if event.get("fallback") else ""
                        print(f"   {mark} {event.get('tool')}{fallback} {event.get('args')}")

                print(f"\n🤖 Agent 回复 >\n{final_answer}")
                if citations:
                    print(f"\n📚 [真实文档溯源]: {', '.join(citations)}")

        except Exception as e:
            print(f"\n❌ 系统发生异常: {e}")

if __name__ == "__main__":
    run_chat()