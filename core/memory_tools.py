# core/memory_tools.py
# 记忆工具：让模型在售后链路里也能读画像、写记忆、按问题检索历史。
# 与订单工具一样，user_id 由后端注入，模型无法指定别人。
import json

from langchain_core.tools import tool

from core import memory_manager


def build_memory_tools(user_id: str, source_thread: str = ""):
    """按当前用户构造记忆工具"""

    @tool
    def get_user_profile() -> str:
        """读取当前用户的长期画像：学习目标、基础水平、偏好。需要个性化回答时先调用它。"""
        profile = memory_manager.get_user_long_term_memory(user_id)
        return json.dumps({"ok": True, "profile": profile}, ensure_ascii=False)

    @tool
    def save_memory(content: str, importance: float = 0.6) -> str:
        """
        把用户透露的长期有效信息写入记忆，例如转行目标、基础水平、上课时间偏好。
        不要记录订单号、手机号、身份证等敏感信息。
        """
        text = (content or "").strip()
        if len(text) < 2:
            return json.dumps({"ok": False, "error": "记忆内容太短"}, ensure_ascii=False)

        memory_id = memory_manager.add_user_memory(
            user_id,
            text,
            kind="note",
            importance=importance,
            source_thread=source_thread or None,
        )
        return json.dumps({"ok": True, "memory_id": memory_id}, ensure_ascii=False)

    @tool
    def search_memory(query: str, top_k: int = 3) -> str:
        """按关键词检索该用户的历史记忆，需要了解用户背景或之前发生过什么时使用。"""
        memories = memory_manager.search_user_memories(user_id, query, top_k=top_k)
        return json.dumps({"ok": True, "memories": memories}, ensure_ascii=False)

    return [get_user_profile, save_memory, search_memory]
