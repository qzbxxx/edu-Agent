# core/memory_manager.py
import sqlite3
import os
import json
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEM_DB_PATH = os.path.join(BASE_DIR, "long_term_memory.db")

def init_memory_db():
    conn = sqlite3.connect(MEM_DB_PATH)
    cursor = conn.cursor()
    # 存储用户长期结构化画像与偏好事实
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            profile_data TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_memory_db()

def get_user_long_term_memory(user_id: str) -> Dict[str, Any]:
    """读取用户长期画像与背景"""
    conn = sqlite3.connect(MEM_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT profile_data FROM user_profiles WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return {"learning_goal": "未知", "skill_level": "未知", "preferences": []}

def save_user_long_term_memory(user_id: str, new_memory: Dict[str, Any]):
    """更新用户长期画像与背景"""
    current_mem = get_user_long_term_memory(user_id)
    current_mem.update(new_memory)
    
    conn = sqlite3.connect(MEM_DB_PATH)
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