# core/memory_manager.py
# 长期记忆的两层存储：
# 1) user_profiles   结构化画像（学习目标 / 基础 / 偏好），每个用户一条
# 2) user_memories   情景记忆（退款、画像变化、用户提到的事实），按时间追加
# 读取时画像整体注入 Prompt，情景记忆按当前问题检索 Top-K，避免把全部历史塞进上下文。
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jieba

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEM_DB_PATH = os.path.join(BASE_DIR, "long_term_memory.db")

# 检索时丢掉的高频虚词，避免“这个/什么”把匹配率算虚高
STOP_WORDS = {
    "什么", "怎么", "可以", "我们", "你们", "这个", "那个", "一下", "就是",
    "还是", "然后", "因为", "所以", "但是", "已经", "一个", "没有", "如果",
    "需要", "问题", "请问", "以及", "并且", "或者", "自己", "现在",
}


def _connect():
    return sqlite3.connect(MEM_DB_PATH, timeout=5)


def _utc_now() -> datetime:
    # SQLite 的 CURRENT_TIMESTAMP 是 UTC，比较时间时保持一致，避免时区差
    return datetime.now(timezone.utc).replace(tzinfo=None)


def init_memory_db():
    conn = _connect()
    cursor = conn.cursor()
    # 存储用户长期结构化画像与偏好事实
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            profile_data TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # 情景记忆：同一条内容只存一次，重复写入只刷新时间与重要性
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'event',
            content TEXT NOT NULL,
            importance REAL NOT NULL DEFAULT 0.5,
            source_thread TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, content)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_memories_user "
        "ON user_memories(user_id, created_at DESC)"
    )
    conn.commit()
    conn.close()


init_memory_db()


def get_user_long_term_memory(user_id: str) -> Dict[str, Any]:
    """读取用户长期画像与背景"""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT profile_data FROM user_profiles WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return {"learning_goal": "未知", "skill_level": "未知", "preferences": []}


def save_user_long_term_memory(user_id: str, new_memory: Dict[str, Any]):
    """更新用户长期画像；如果覆盖了旧值，额外记一条变更，方便回溯"""
    current_mem = get_user_long_term_memory(user_id)

    changes = []
    for key, value in new_memory.items():
        old_value = current_mem.get(key)
        # 默认占位值不算“旧信息”，避免第一次写入也记一条变更
        if old_value in (None, "", "未知", []):
            continue
        if old_value == value:
            continue
        changes.append(f"画像更新：{key} 由「{old_value}」变为「{value}」")

    current_mem.update(new_memory)

    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_profiles (user_id, profile_data, updated_at) 
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET 
            profile_data = excluded.profile_data,
            updated_at = CURRENT_TIMESTAMP
    """, (user_id, json.dumps(current_mem, ensure_ascii=False)))
    conn.commit()
    conn.close()

    # 画像被改写时留痕，避免新的值静默覆盖旧值
    for change in changes:
        add_user_memory(user_id, change, kind="profile_change", importance=0.5)


def format_user_profile(profile: Dict[str, Any]) -> str:
    """把画像整理成一行文字，供 Prompt 使用"""
    goal = profile.get("learning_goal") or "未知"
    level = profile.get("skill_level") or "未知"

    preferences = profile.get("preferences") or []
    if isinstance(preferences, list):
        preferences_text = "、".join(str(item) for item in preferences)
    else:
        preferences_text = str(preferences)
    if not preferences_text:
        preferences_text = "无"

    return f"学习目标：{goal}；基础：{level}；偏好：{preferences_text}"


def add_user_memory(
    user_id: str,
    content: str,
    kind: str = "event",
    importance: float = 0.5,
    source_thread: Optional[str] = None,
) -> int:
    """写入一条情景记忆，返回记忆 id；内容为空直接返回 0"""
    text = (content or "").strip()[:1000]
    if not text:
        return 0

    importance = max(0.0, min(1.0, float(importance or 0.0)))

    init_memory_db()
    conn = _connect()
    conn.execute("""
        INSERT INTO user_memories (user_id, kind, content, importance, source_thread)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id, content) DO UPDATE SET
            created_at = CURRENT_TIMESTAMP,
            importance = MAX(user_memories.importance, excluded.importance)
    """, (user_id, kind, text, importance, source_thread))
    conn.commit()

    row = conn.execute(
        "SELECT id FROM user_memories WHERE user_id = ? AND content = ?",
        (user_id, text),
    ).fetchone()
    conn.close()

    return int(row[0]) if row else 0


def list_user_memories(user_id: str, limit: int = 20, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """按时间倒序列出用户的记忆，供管理接口和调试使用"""
    limit = max(1, min(200, int(limit)))

    init_memory_db()
    conn = _connect()
    conn.row_factory = sqlite3.Row

    sql = (
        "SELECT id, user_id, kind, content, importance, source_thread, created_at "
        "FROM user_memories WHERE user_id = ?"
    )
    params: List[Any] = [user_id]
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _terms(text: str) -> set:
    """分词并丢掉单字与高频虚词"""
    terms = set()
    for token in jieba.lcut(str(text or "")):
        token = token.strip().lower()
        if len(token) > 1 and token not in STOP_WORDS:
            terms.add(token)
    return terms


def _age_days(created_at: Any, now: datetime) -> float:
    try:
        created = datetime.fromisoformat(str(created_at))
    except (TypeError, ValueError):
        return 999.0
    return max(0.0, (now - created).total_seconds() / 86400)


def search_user_memories(user_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    轻量检索：关键词重合度 65% + 时间新鲜度 25% + 重要性 10%

    为什么不用向量：项目里的 embedding 是英文模型，中文记忆召回反而不如 jieba 关键词；
    这里保留固定打分接口，后续换 bge/embedding 只需替换这一处。
    与问题没有任何词重合的记忆直接丢弃，避免“最近记忆”污染上下文。
    """
    top_k = max(1, min(10, int(top_k)))
    query_terms = _terms(query)
    if not query_terms:
        return []

    now = _utc_now()
    scored = []
    for item in list_user_memories(user_id, limit=200):
        content_terms = _terms(item["content"])
        if not content_terms:
            continue
        overlap = len(query_terms & content_terms) / len(query_terms)
        if overlap <= 0:
            continue

        recency = 1.0 / (1.0 + _age_days(item["created_at"], now) / 30.0)
        score = 0.65 * overlap + 0.25 * recency + 0.10 * float(item["importance"])
        item["match_ratio"] = round(overlap, 4)
        item["score"] = round(score, 4)
        scored.append(item)

    scored.sort(key=lambda item: (item["score"], item["id"]), reverse=True)
    return scored[:top_k]


def delete_user_memory(user_id: str, memory_id: int) -> bool:
    """删除一条记忆；带上 user_id 防止越权删除别人的记忆"""
    init_memory_db()
    conn = _connect()
    cursor = conn.execute(
        "DELETE FROM user_memories WHERE id = ? AND user_id = ?",
        (int(memory_id), user_id),
    )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted > 0


def prune_user_memories(
    user_id: Optional[str] = None,
    days: int = 90,
    min_importance: float = 0.8,
) -> int:
    """遗忘策略：删除超过 days 天且重要性低于 min_importance 的记忆"""
    days = max(1, int(days))
    cutoff = (_utc_now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

    init_memory_db()
    conn = _connect()
    sql = "DELETE FROM user_memories WHERE created_at < ? AND importance < ?"
    params: List[Any] = [cutoff, float(min_importance)]
    if user_id:
        sql += " AND user_id = ?"
        params.append(user_id)

    cursor = conn.execute(sql, params)
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def clear_user_memories(user_id: str) -> int:
    """清空某个用户的情景记忆（测试与用户主动删除用），画像保留"""
    init_memory_db()
    conn = _connect()
    cursor = conn.execute("DELETE FROM user_memories WHERE user_id = ?", (user_id,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def clear_user_data(user_id: str) -> int:
    """删除某个用户的全部长期记忆（画像 + 情景记忆）"""
    init_memory_db()
    conn = _connect()
    cursor = conn.execute("DELETE FROM user_memories WHERE user_id = ?", (user_id,))
    deleted = cursor.rowcount
    conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return deleted
