# core/agent_workflow.py
import os
import json
import sqlite3
from typing import TypedDict, Annotated, Optional, List, Dict, Any

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 强制让 .env 文件里的新 Key 覆盖终端残留的旧环境变量！
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, trim_messages
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
from core.rag_engine import search_knowledge_base
from core.memory_manager import get_user_long_term_memory, save_user_long_term_memory
from core.tools import build_order_tools
from core import refund_service
from server import mock_db, refund_store

# 导入 Langfuse 回调（可选）
try:
    from langfuse.langchain import CallbackHandler
    langfuse_handler = CallbackHandler()
    callbacks = [langfuse_handler]
except ImportError:
    callbacks = []

llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0,
    callbacks=callbacks
)

DB_PATH = os.path.join(BASE_DIR, "agent_state.db")

# 1. 定义标准状态机 State（整合长短期记忆）
class AgentState(TypedDict):
    # 短期记忆：add_messages 自动实现同 thread 多轮历史追加
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    order_id: Optional[str]
    intent: Optional[str]           # "consult" | "refund"
    rewritten_query: Optional[str]  # 意图改写后的独立检索问句
    citations: List[str]            # 溯源引用
    refund_amount: float
    need_human_approval: bool
    approval_granted: bool
    refund_requested: bool          # 工具是否已成功登记退款申请
    refund_reason: Optional[str]    # 申请时填写的原因
    tool_events: List[Dict[str, Any]]  # 本轮的调用记录，供前端展示与排查
    final_response: str
    # ----- 长期记忆槽位 -----
    long_term_memory: Dict[str, Any] # 跨会话沉淀的用户画像 (技能、偏好、目标等)

# 2. 长期记忆装载节点（入口第 1 站）
def load_memory_node(state: AgentState):
    user_id = state.get("user_id", "default_user")
    ltm = get_user_long_term_memory(user_id)
    # tool_events 是本轮的过程记录，入口统一清空，避免把上一轮的调用记到这一轮
    return {"long_term_memory": ltm, "tool_events": []}

# 3. 意图路由节点
def router_node(state: AgentState):
    last_user_msg = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
    prompt = f"""分析用户输入意图，只返回一个单词: 'consult' 或 'refund'。
判定规则：
- 用户咨询课程内容、费用标准、退款政策、询问“能不能/是否允许/规则是什么” -> 返回 'consult'
- 用户明确提出“我要退款/申请退款/帮我退钱/取消订单”等具体售后操作 -> 返回 'refund'

用户输入: {last_user_msg}
意图:"""
    res = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    intent = "refund" if "refund" in res else "consult"
    return {"intent": intent}

# 4. 短期上下文滑动窗口修剪 + 意图重写节点
def query_rewrite_node(state: AgentState):
    messages = state["messages"]
    last_user_msg = [m for m in messages if isinstance(m, HumanMessage)][-1].content

    # 如果是首轮提问，无需进行上下文重写
    if len(messages) <= 1:
        return {"rewritten_query": last_user_msg}

    # 滑动窗口机制：提取最近消息，阻断历史冗余
    recent_msgs = trim_messages(
        messages,
        max_tokens=1000,
        strategy="last",
        token_counter=len,
        start_on="human",
        end_on="human"
    )

    formatted_history = []
    for m in recent_msgs[:-1]:
        role = "学员" if isinstance(m, HumanMessage) else "客服AI"
        formatted_history.append(f"{role}: {m.content}")
    history_str = "\n".join(formatted_history)

    rewrite_prompt = f"""你是一个智能问答系统的 Query 改写助手。
结合以下对话历史，判断用户的最新提问是否包含代词（如“它”、“这个课”、“那门训练营”）或省略了主语。
如果包含指代或不完整，请将其改写为一个独立的、包含完整主语的问句，以便用于向量检索。
如果不包含代词，或者提问本身已经很完整，请原样输出用户提问。
严禁回答问题，只输出改写后的问句！

【对话历史】:
{history_str}

【用户最新提问】:
{last_user_msg}

改写后的独立检索词:"""

    rewritten = llm.invoke([HumanMessage(content=rewrite_prompt)]).content.strip()
    return {"rewritten_query": rewritten}

# 5. 咨询问答节点（结合 短期上下文 + 长期画像 + 知识库切片）
def consult_rag_node(state: AgentState):
    search_query = state.get("rewritten_query") or state["messages"][-1].content
    matched_chunks = search_knowledge_base(search_query, top_n=2)
    
    if not matched_chunks:
        fallback_msg = "抱歉，关于该问题，官方文档与知识库中未检索到明确记录，已为您转接人工顾问处理。"
        return {
            "final_response": fallback_msg,
            "citations": [],
            "messages": [AIMessage(content=fallback_msg)]
        }
    
    context_list = []
    citations = []
    for c in matched_chunks:
        cite_label = f"《{c['source']}》第{c['page']}页"
        citations.append(cite_label)
        context_list.append(f"[{cite_label}] 内容: {c['content']}")
        
    context = "\n\n".join(context_list)
    
    # 注入长期记忆画像
    ltm = state.get("long_term_memory", {})
    ltm_str = f"学员画像: 目标={ltm.get('learning_goal', '未知')}, 基础={ltm.get('skill_level', '未知')}, 偏好={ltm.get('preferences', [])}"

    # 提取短期记忆历史对话（最近 2 轮）
    #  补上 token_counter=len
    recent_dialogue = trim_messages(
    state["messages"][-5:], 
    max_tokens=600, 
    token_counter=len, 
    strategy="last"
    )
    short_term_context = "\n".join([
        f"{'学员' if isinstance(m, HumanMessage) else '客服'}: {m.content}" 
        for m in recent_dialogue[:-1]
    ])

    rag_prompt = f"""请严格依据以下参考文档回答问题。
{ltm_str}

【参考文档】:
{context}

【本轮对话短期上下文】:
{short_term_context}

用户当前问题: {state['messages'][-1].content}

要求：
1. 答案必须完全基于参考文档，严禁自行发挥和捏造。
2. 结合学员的画像背景给出贴合语境的建议。
3. 在引用的每一个关键事实语句后方，必须打上引用来源标记，例如 [《文件名》第X页]。"""

    res = llm.invoke([HumanMessage(content=rag_prompt)]).content
    
    return {
        "final_response": res,
        "citations": list(set(citations)),
        "messages": [AIMessage(content=res)]
    }

# 6. 长期记忆抽取节点（在咨询完成后，自动分析用户是否有新画像特征需要持久化）
def memory_extractor_node(state: AgentState):
    user_id = state.get("user_id")
    # 取得本轮用户的输入
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_msgs or not user_id:
        return {}

    last_user_msg = human_msgs[-1].content
    extract_prompt = f"""分析学员这句提问是否透露了长期的技术基础、职业目标或学习偏好（例如零基础、转行、准备面试等）？
如果包含，请提取并只返回 JSON 格式，如: {{"skill_level": "零基础", "learning_goal": "转行就业"}}；
如果没有任何有价值的长期特征，请直接返回: {{}}。
学员提问: "{last_user_msg}"
JSON:"""
    try:
        res = llm.invoke([HumanMessage(content=extract_prompt)]).content.strip()
        clean_json = res.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(clean_json)
        if isinstance(extracted, dict) and extracted:
            save_user_long_term_memory(user_id, extracted)
    except Exception:
        pass
    return {}

# 7. 售后工具调用节点（模型自己决定查订单、核政策、提交退款申请）
MAX_TOOL_STEPS = 4  # 最多连续调用几轮工具，防止模型来回打转

EXPLICIT_REFUND_WORDS = ("退款", "退钱", "退费", "退课", "取消订单", "不想要了")


def _last_human_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content or ""
    return ""


def _is_explicit_refund(text: str) -> bool:
    return any(word in text for word in EXPLICIT_REFUND_WORDS)


def _read_ai_text(message: BaseMessage) -> str:
    """取模型回复的文本；部分模型返回分段内容，这里统一拼成字符串"""
    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts).strip()
    return str(content or "").strip()


def _run_tool(tools_by_name: Dict[str, Any], name: str, args: dict) -> dict:
    """执行一个工具并把结果统一成字典；工具内部异常也转成结构化结果"""
    tool = tools_by_name.get(name)
    if tool is None:
        return {"ok": False, "error": f"未知工具：{name}"}

    try:
        raw = tool.invoke(args or {})
    except Exception as error:
        return {"ok": False, "error": f"工具执行失败：{error}"}

    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"ok": True, "text": raw}
    return raw


def _after_sales_system_prompt(user_id: str, order_id: Optional[str]) -> str:
    order_text = order_id or "未提供"
    return f"""你是「智学在线教育」的售后助手，可以调用工具查询订单、核对退费政策、提交退款申请。

当前用户：{user_id}
当前订单号：{order_text}

规则：
1. 只依据工具返回的数据回答，禁止编造金额、天数、学习进度。
2. 用户只是询问政策或订单状态时，查完直接解释，不要提交退款。
3. 用户明确要求退款时，先 query_order 核实订单，再调用 submit_refund_request 提交；同一订单只提交一次。
4. 即使 query_order 显示 need_human_approval=true（超 7 天或进度超 10%），也必须调用 submit_refund_request 登记申请，后续流程会转人工审批；不要因为“需要审核”就跳过提交。
5. 提交成功后告诉用户“已提交退款申请”；需要审核时补充“已转人工审核”，不要承诺到账时间。
6. 工具返回 ok=false 时，如实转述原因，不要反复重试。
7. 用中文简洁回答。"""


def after_sales_agent_node(state: AgentState):
    """
    售后节点：让模型自主选择工具

    执行方式：模型提议工具调用 -> 本地执行 -> 结果作为 ToolMessage 回填 -> 模型继续，
    直到它给出最终答复（最多 MAX_TOOL_STEPS 轮）。
    资金动作只做到“提交申请”，真正的退款仍由 human_approval / execute_refund 把关。
    """
    user_id = state.get("user_id") or "default_user"
    order_id = state.get("order_id")
    user_text = _last_human_text(state)

    tools = build_order_tools(user_id)
    tools_by_name = {tool.name: tool for tool in tools}

    conversation = [SystemMessage(content=_after_sales_system_prompt(user_id, order_id))]
    conversation.extend(state.get("messages", [])[-6:])

    tool_events = []
    submitted = None
    answer = ""

    try:
        llm_with_tools = llm.bind_tools(tools)
    except Exception:
        # 模型或网关不支持工具调用时降级；后面还有确定性兜底
        llm_with_tools = None

    if llm_with_tools is not None:
        try:
            for step in range(MAX_TOOL_STEPS):
                response = llm_with_tools.invoke(conversation)
                conversation.append(response)

                tool_calls = list(getattr(response, "tool_calls", None) or [])
                if not tool_calls:
                    answer = _read_ai_text(response)
                    break

                for call in tool_calls:
                    name = call.get("name", "")
                    args = call.get("args") or {}
                    payload = _run_tool(tools_by_name, name, args)
                    tool_events.append({"tool": name, "args": args, "result": payload})

                    if name == "submit_refund_request" and payload.get("ok"):
                        submitted = payload
                    if name == "submit_refund_request" and not payload.get("ok"):
                        answer = payload.get("error") or answer

                    conversation.append(ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False),
                        tool_call_id=call.get("id") or f"call_{step}",
                    ))
        except Exception as error:
            # 模型或网关异常时降级：已提交的退款继续走流程，没提交的交给下面的兜底
            tool_events.append({
                "tool": "llm",
                "args": {},
                "result": {"ok": False, "error": f"大模型调用失败：{error}"},
            })

    # 模型没走工具，但用户确实明确要求退款：用确定性逻辑兜底，资金链路不依赖模型自觉
    attempted_submit = any(event["tool"] == "submit_refund_request" for event in tool_events)
    if submitted is None and not attempted_submit and order_id and _is_explicit_refund(user_text):
        payload = refund_service.submit_refund_request(user_id, order_id, reason="用户主动申请退款")
        tool_events.append({
            "tool": "submit_refund_request",
            "args": {"order_id": order_id},
            "result": payload,
            "fallback": True,
        })
        if payload.get("ok"):
            submitted = payload
        else:
            answer = payload.get("error") or answer

    # 每轮先清空上一轮的资金标记，避免旧的 refund_requested 让本轮误进审批
    updates = {
        "tool_events": tool_events,
        "refund_requested": False,
        "need_human_approval": False,
        "refund_amount": 0.0,
    }

    if submitted is not None:
        updates["refund_requested"] = True
        updates["order_id"] = submitted.get("order_id") or order_id
        updates["refund_amount"] = float(submitted.get("refund_amount") or 0.0)
        updates["need_human_approval"] = bool(submitted.get("need_human_approval"))
        updates["refund_reason"] = submitted.get("reason") or ""
        answer = submitted.get("message") or answer or "已提交退款申请，正在处理中。"

    if not answer:
        answer = "抱歉，暂时无法处理这个售后请求，已为您转接人工客服。"

    updates["final_response"] = answer
    updates["messages"] = [AIMessage(content=answer)]
    return updates

# 8. 高危挂起节点 (HITL)
def human_approval_node(state: AgentState):
    # 没有成功登记退款申请（例如订单无效、重复提交）时，直接结束
    if not state.get("refund_requested"):
        return {"approval_granted": False}

    # 低风险订单：政策校验通过，自动放行
    if not state.get("need_human_approval"):
        return {"approval_granted": True}

    approval_decision = interrupt({
        "action": "REQUIRE_HUMAN_APPROVAL",
        "order_id": state.get("order_id"),
        "refund_amount": state.get("refund_amount", 0.0),
        "reason": "订单已超出无条件退款期限或已学进度超标，需人工核验。"
    })

    is_approved = approval_decision.get("approved", False)
    if not is_approved:
        reason = approval_decision.get("reject_reason") or "不符合退费要求"
        reject_msg = f"您的退费申请未通过人工审核，原因: {reason}"
        return {
            "approval_granted": False,
            "final_response": reject_msg,
            "messages": [AIMessage(content=reject_msg)]
        }
    return {"approval_granted": True}

# 9. 幂等退费执行节点
def execute_refund_node(state: AgentState):
    order_id = state.get("order_id")
    user_id = state.get("user_id", "default_user")
    order = refund_service.find_order(order_id, user_id)

    # 这里再校验一次归属：即使有人绕过图、直接调本函数，也不能退款别人的订单
    if not order:
        msg = f"错误：订单 {order_id} 不存在或不属于当前账号，无法执行退款。"
        return {"final_response": msg, "messages": [AIMessage(content=msg)]}

    amount = state.get("refund_amount") or order["price"]

    # 幂等交给数据库的唯一约束：并发请求、消息重放、服务重启都拦得住
    reserved, reason = refund_store.reserve_refund(order_id, user_id, amount)
    if not reserved:
        msg = f"错误：{reason}！"
        return {"final_response": msg, "messages": [AIMessage(content=msg)]}

    mock_db.MOCK_ORDERS[order_id]["status"] = "REFUNDED"
    # 预占之后再置为成功态，中间挂掉会留下 PROCESSING 记录供对账
    refund_store.mark_refund_success(order_id)

    succ_msg = f"退款成功！退款金额 {amount} 元已原路退回。"
    return {"final_response": succ_msg, "messages": [AIMessage(content=succ_msg)]}

# 10. 构建完整工作流图
workflow = StateGraph(AgentState)

# 注册所有节点
workflow.add_node("load_memory", load_memory_node)
workflow.add_node("router", router_node)
workflow.add_node("query_rewrite", query_rewrite_node)
workflow.add_node("consult_rag", consult_rag_node)
workflow.add_node("memory_extractor", memory_extractor_node)
workflow.add_node("after_sales_agent", after_sales_agent_node)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("execute_refund", execute_refund_node)

# 设置首节点为读取长期记忆
workflow.set_entry_point("load_memory")
workflow.add_edge("load_memory", "router")

def route_decision(state: AgentState):
    return "query_rewrite" if state["intent"] == "consult" else "after_sales_agent"

def after_agent_decision(state: AgentState):
    # 只有工具成功登记了退款申请，才进入审批/执行；否则本轮直接结束
    return "human_approval" if state.get("refund_requested") else END

def after_approval_decision(state: AgentState):
    return "execute_refund" if state.get("approval_granted") else END

workflow.add_conditional_edges("router", route_decision)
workflow.add_edge("query_rewrite", "consult_rag")
workflow.add_edge("consult_rag", "memory_extractor")
workflow.add_edge("memory_extractor", END)

workflow.add_conditional_edges("after_sales_agent", after_agent_decision)
workflow.add_conditional_edges("human_approval", after_approval_decision)
workflow.add_edge("execute_refund", END)

# 初始化 SQLite 持久化层
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
app_graph = workflow.compile(checkpointer=checkpointer)
