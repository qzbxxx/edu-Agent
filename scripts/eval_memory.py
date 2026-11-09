# scripts/eval_memory.py
# 长期记忆检索的离线评测：Recall@1 / Recall@3 / MRR / 无关问题误召回 / 跨用户隔离。
# 用法：python scripts/eval_memory.py
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
# Windows 终端默认 GBK，统一成 UTF-8，避免中文输出乱码
sys.stdout.reconfigure(encoding="utf-8")

from core import memory_manager

EVAL_USER = "eval_memory_user"
OTHER_USER = "eval_memory_other"
DATA_PATH = os.path.join(BASE_DIR, "data", "memory_eval_set.jsonl")


def load_cases():
    cases = []
    with open(DATA_PATH, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def add_memories(memories):
    """支持两种写法：纯字符串，或 {"content": ..., "importance": ...}"""
    for memory in memories:
        if isinstance(memory, str):
            content, importance = memory, 0.5
        else:
            content = memory.get("content", "")
            importance = memory.get("importance", 0.5)
        memory_manager.add_user_memory(EVAL_USER, content, kind="eval", importance=importance)


def run_case(case):
    memory_manager.clear_user_data(EVAL_USER)
    add_memories(case.get("memories", []))

    results = memory_manager.search_user_memories(EVAL_USER, case["query"], top_k=3)
    expected = case.get("expect")

    if not expected:
        ok = not results
        detail = "无召回" if ok else f"误召回 {len(results)} 条"
        return {"id": case["id"], "query": case["query"], "ok": ok, "detail": detail,
                "rank": None, "negative": True}

    rank = None
    for index, item in enumerate(results, start=1):
        if expected in item["content"]:
            rank = index
            break

    ok = rank is not None
    detail = f"rank={rank}" if ok else "未命中"
    return {"id": case["id"], "query": case["query"], "ok": ok, "detail": detail,
            "rank": rank, "negative": False}


def check_isolation():
    """其他用户的记忆不能出现在当前用户的检索结果里"""
    memory_manager.clear_user_data(EVAL_USER)
    memory_manager.clear_user_data(OTHER_USER)
    memory_manager.add_user_memory(OTHER_USER, "订单 ORD_999 退款成功", kind="eval", importance=1.0)
    leaked = memory_manager.search_user_memories(EVAL_USER, "退款成功", top_k=3)
    memory_manager.clear_user_data(OTHER_USER)
    return not leaked


def main():
    cases = load_cases()
    rows = []
    positive_total = 0
    hit1 = 0
    hit3 = 0
    reciprocal_rank = 0.0
    negative_total = 0
    false_recall = 0

    for case in cases:
        row = run_case(case)
        rows.append(row)
        if row["negative"]:
            negative_total += 1
            if not row["ok"]:
                false_recall += 1
            continue

        positive_total += 1
        if row["rank"] == 1:
            hit1 += 1
        if row["rank"] and row["rank"] <= 3:
            hit3 += 1
            reciprocal_rank += 1.0 / row["rank"]

    isolation_ok = check_isolation()
    memory_manager.clear_user_data(EVAL_USER)

    print("长期记忆检索评测")
    print("-" * 72)
    for row in rows:
        mark = "PASS" if row["ok"] else "FAIL"
        print(f"{row['id']:<4} {mark:<4} {row['detail']:<12} {row['query']}")

    print("-" * 72)
    print(f"正例数: {positive_total}    负例数: {negative_total}")
    print(f"Recall@1: {hit1}/{positive_total} = {hit1 / positive_total:.1%}")
    print(f"Recall@3: {hit3}/{positive_total} = {hit3 / positive_total:.1%}")
    print(f"MRR: {reciprocal_rank / positive_total:.3f}")
    print(f"无关问题误召回: {false_recall}/{negative_total}")
    print(f"跨用户隔离: {'通过' if isolation_ok else '失败'}")

    passed = hit3 / positive_total >= 0.8 and false_recall == 0 and isolation_ok
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
