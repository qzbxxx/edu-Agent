# server/refund_store.py
# 退款记录的持久化存储：用 SQLite 的唯一约束做幂等，替代原先内存里的 set
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFUND_DB_PATH = os.path.join(BASE_DIR, "refund_records.db")


def init_refund_db():
    """建表：order_id 做主键，一个订单最多只能有一条退款记录"""
    conn = sqlite3.connect(REFUND_DB_PATH, timeout=5)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS refund_records (
            order_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def reserve_refund(order_id: str, user_id: str, amount: float):
    """
    抢占退款资格，返回 (是否抢到, 提示语)

    这里不查后用，而是直接 INSERT，让 order_id 主键的唯一约束来判重。
    好处是：即使两个请求/两个进程同时到达，SQLite 也只会让其中一个插入成功，
    不需要应用层加锁；服务重启后记录仍在，不依赖内存。
    """
    init_refund_db()
    conn = sqlite3.connect(REFUND_DB_PATH, timeout=5)
    try:
        conn.execute(
            "INSERT INTO refund_records (order_id, user_id, amount, status) "
            "VALUES (?, ?, ?, 'PROCESSING')",
            (order_id, user_id, amount),
        )
        conn.commit()
        return True, ""
    except sqlite3.IntegrityError:
        conn.rollback()
        return False, "该订单已提交过退款申请，请勿重复操作"
    finally:
        conn.close()


def mark_refund_success(order_id: str):
    """
    把预占记录改成成功态
    如果进程在占位之后、这一步之前挂了，记录会停在 PROCESSING，
    正好留作对账线索，而不是被当成"已经退好了"。
    """
    conn = sqlite3.connect(REFUND_DB_PATH, timeout=5)
    conn.execute(
        "UPDATE refund_records SET status = 'SUCCESS' WHERE order_id = ?",
        (order_id,),
    )
    conn.commit()
    conn.close()


def get_refund_record(order_id: str):
    """按订单号查退款记录，查不到返回 None"""
    init_refund_db()
    conn = sqlite3.connect(REFUND_DB_PATH, timeout=5)
    cursor = conn.execute(
        "SELECT order_id, user_id, amount, status, created_at "
        "FROM refund_records WHERE order_id = ?",
        (order_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "order_id": row[0],
        "user_id": row[1],
        "amount": row[2],
        "status": row[3],
        "created_at": row[4],
    }


def list_refund_records(limit: int = 100):
    """按创建时间倒序列出退款记录（调试、对账用）"""
    init_refund_db()
    conn = sqlite3.connect(REFUND_DB_PATH, timeout=5)
    rows = conn.execute(
        "SELECT order_id, user_id, amount, status, created_at "
        "FROM refund_records ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    records = []
    for row in rows:
        records.append({
            "order_id": row[0],
            "user_id": row[1],
            "amount": row[2],
            "status": row[3],
            "created_at": row[4],
        })
    return records


def clear_refund_records():
    """清空退款记录（仅供测试使用，生产环境不要调用）"""
    init_refund_db()
    conn = sqlite3.connect(REFUND_DB_PATH, timeout=5)
    conn.execute("DELETE FROM refund_records")
    conn.commit()
    conn.close()
