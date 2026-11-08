# scripts/eval_rag.py
# 知识库问答的离线评测脚本
#
# 用法：
#   python scripts/eval_rag.py                 只评检索层（免费，秒级出结果）
#   python scripts/eval_rag.py --with-llm      再评答案质量（会真实调用大模型，按条计费）
#   python scripts/eval_rag.py --limit 5       只跑前 5 条，调试时用
#   python scripts/eval_rag.py --output data/eval_report.md
import os
import sys
import json
import uuid
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Windows 控制台默认 GBK，直接打印 ✅/❌ 会抛 UnicodeEncodeError
# 统一切成 UTF-8 并允许替换字符，避免评测跑到一半被终端编码打断
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_SET_PATH = os.path.join(BASE_DIR, "data", "eval_set.jsonl")

# 应拒答的问题，答案里出现这些词就算合规拒答
REFUSAL_KEYWORDS = [
    "未检索到", "无法提供", "无法解答", "不在", "转接人工", "人工顾问", "严禁咨询"
]

# 出现这些词说明模型在对拒答类问题硬编答案（评测防幻觉用）
HALLUCINATION_WORDS = [
    "生抽", "老抽", "焯水", "料酒", "炖煮", "配方", "降雨", "多云", "气温"
]


def load_cases(path):
    """逐行读取 JSONL 标注集，坏行跳过并提示，不让一条脏数据中断整轮评测"""
    cases = []
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"⚠️ 第 {line_no} 行不是合法 JSON，已跳过：{error}")
    return cases


def default_retriever(query: str):
    """真实检索器；延迟 import，使指标计算部分不依赖向量库"""
    from core.rag_engine import search_knowledge_base
    return search_knowledge_base(query, top_n=2)


def default_answer_runner(query: str, thread_id: str):
    """真实问答执行器：走完整状态机，每条用例一个独立会话"""
    from langchain_core.messages import HumanMessage
    from core.agent_workflow import app_graph

    config = {"configurable": {"thread_id": thread_id}}
    result = app_graph.invoke({
        "messages": [HumanMessage(content=query)],
        "user_id": "eval_user",
        "order_id": None,
        "citations": [],
        "need_human_approval": False
    }, config=config)
    return result.get("final_response") or "", result.get("citations") or []


def evaluate_retrieval(cases, retriever):
    """
    检索层评测
    可答用例：召回的切片里必须包含期望的来源文件与页码
    应拒用例：理想情况是一条切片都不召回（宁可转人工，也不要给模型喂无关材料）
    """
    records = []
    for case in cases:
        chunks = retriever(case["query"])

        if case["expect"] == "answer":
            expect_source = case.get("expect_source")
            expect_page = case.get("expect_page")
            hit = False
            for chunk in chunks:
                if chunk.get("source") != expect_source:
                    continue
                if expect_page is not None and chunk.get("page") != expect_page:
                    continue
                hit = True
                break
        else:
            if len(chunks) == 0:
                hit = True
            else:
                # 有些"该拒答"的问题本身就写在政策条款里，比如第5条明确禁止代写作业。
                # 这时召回该条款是正确行为，只要召回的全都是这条政策切片也算通过。
                expect_source = case.get("expect_source")
                expect_page = case.get("expect_page")
                hit = False
                if expect_source:
                    hit = True
                    for chunk in chunks:
                        if chunk.get("source") != expect_source:
                            hit = False
                            break
                        if expect_page is not None and chunk.get("page") != expect_page:
                            hit = False
                            break

        records.append({
            "id": case["id"],
            "query": case["query"],
            "expect": case["expect"],
            "chunk_count": len(chunks),
            "top_score": chunks[0]["rerank_score"] if chunks else 0.0,
            "hit": hit,
            "note": case.get("note", "")
        })
    return records


def evaluate_answers(cases, answer_runner):
    """答案层评测：关键词是否覆盖、是否带引用页码、该拒的有没有拒"""
    records = []
    for case in cases:
        thread_id = f"eval_{case['id']}_{uuid.uuid4().hex[:6]}"
        answer, citations = answer_runner(case["query"], thread_id)

        if case["expect"] == "answer":
            missing = []
            for keyword in case.get("must_contain", []):
                if keyword not in answer:
                    missing.append(keyword)
            passed = len(missing) == 0
            reason = "" if passed else f"缺少关键词：{'、'.join(missing)}"
        else:
            refused = False
            for keyword in REFUSAL_KEYWORDS:
                if keyword in answer:
                    refused = True
                    break
            hallucinated = False
            for word in HALLUCINATION_WORDS:
                if word in answer:
                    hallucinated = True
                    break
            passed = refused and not hallucinated
            if hallucinated:
                reason = "出现幻觉内容"
            elif not refused:
                reason = "未合规拒答"
            else:
                reason = ""

        records.append({
            "id": case["id"],
            "passed": passed,
            "has_citation": len(citations) > 0,
            "answer": answer,
            "reason": reason
        })
    return records


def summarize(retrieval_records, answer_records=None):
    """汇总成一组可横向对比的数字"""
    answer_cases = [r for r in retrieval_records if r["expect"] == "answer"]
    refuse_cases = [r for r in retrieval_records if r["expect"] == "refuse"]

    answer_hit = 0
    for record in answer_cases:
        if record["hit"]:
            answer_hit += 1
    refuse_blocked = 0
    for record in refuse_cases:
        if record["hit"]:
            refuse_blocked += 1

    scores = [r["top_score"] for r in retrieval_records if r["top_score"] > 0]
    average_score = round(sum(scores) / len(scores), 3) if scores else 0.0

    summary = {
        "可答用例数": len(answer_cases),
        "可答命中数": answer_hit,
        "应拒用例数": len(refuse_cases),
        "应拒拦截数": refuse_blocked,
        "平均Top1置信度": average_score
    }

    if answer_records is not None:
        answer_passed = 0
        cited = 0
        for record in answer_records:
            if record["passed"]:
                answer_passed += 1
            if record["has_citation"]:
                cited += 1
        summary["答案通过数"] = answer_passed
        summary["带引用条数"] = cited

    return summary


def print_report(retrieval_records, answer_records=None):
    # 汇总放在这里算，避免调用方传错 summary 导致字段缺失
    summary = summarize(retrieval_records, answer_records)

    print("=" * 72)
    print("知识库问答评测报告")
    print("=" * 72)
    print(f"标注总数 {summary['可答用例数'] + summary['应拒用例数']} 条"
          f"（可答 {summary['可答用例数']} / 应拒 {summary['应拒用例数']}）")
    print(f"说明：应拒用例在检索层要求「一条切片都不召回」，比最终答案层更严格。")

    print("\n【检索层】")
    print(f"  可答命中率   : {summary['可答命中数']}/{summary['可答用例数']}")
    print(f"  应拒拦截率   : {summary['应拒拦截数']}/{summary['应拒用例数']}")
    print(f"  平均置信度   : {summary['平均Top1置信度']}")

    print("\n【检索明细】")
    for record in retrieval_records:
        mark = "✅" if record["hit"] else "❌"
        expect_label = "可答" if record["expect"] == "answer" else "应拒"
        line = (f"  {mark} {record['id']} {expect_label} "
                f"召回{record['chunk_count']}条 置信度{record['top_score']:.3f} "
                f"{record['query']}")
        if record["note"]:
            line += f"  ← {record['note']}"
        print(line)

    if answer_records is None:
        return

    print("\n【答案层】")
    total = len(answer_records)
    print(f"  答案通过数   : {summary['答案通过数']}/{total}")
    print(f"  带引用条数   : {summary['带引用条数']}/{total}")
    for record in answer_records:
        mark = "✅" if record["passed"] else "❌"
        line = f"  {mark} {record['id']} 引用{'有' if record['has_citation'] else '无'}"
        if record["reason"]:
            line += f"  ← {record['reason']}"
        print(line)
        print(f"      回答: {record['answer'][:70]}...")


def write_markdown(path, retrieval_records, answer_records=None):
    """把报告落成 Markdown，方便贴进 README 或做版本间对比"""
    summary = summarize(retrieval_records, answer_records)

    lines = []
    lines.append("# 知识库问答评测报告\n")
    lines.append(f"- 可答命中率：{summary['可答命中数']}/{summary['可答用例数']}")
    lines.append(f"- 应拒拦截率：{summary['应拒拦截数']}/{summary['应拒用例数']}")
    lines.append(f"- 平均置信度：{summary['平均Top1置信度']}")
    if answer_records is not None:
        lines.append(f"- 答案通过数：{summary['答案通过数']}/{len(answer_records)}")
        lines.append(f"- 带引用条数：{summary['带引用条数']}/{len(answer_records)}")
    lines.append("")
    lines.append("| 用例 | 期望 | 召回数 | 置信度 | 结果 | 备注 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for record in retrieval_records:
        mark = "✅" if record["hit"] else "❌"
        lines.append(f"| {record['id']} | {record['expect']} | {record['chunk_count']} "
                     f"| {record['top_score']:.3f} | {mark} | {record['note']} |")

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    print(f"\n📝 报告已写入：{path}")


def main():
    parser = argparse.ArgumentParser(description="知识库问答离线评测")
    parser.add_argument("--set", default=DEFAULT_SET_PATH, help="标注集路径（JSONL）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全部")
    parser.add_argument("--with-llm", action="store_true",
                        help="追加答案质量评测（会调用大模型，产生费用）")
    parser.add_argument("--output", default="", help="把报告写成 Markdown 到指定路径")
    args = parser.parse_args()

    cases = load_cases(args.set)
    if args.limit > 0:
        cases = cases[:args.limit]
    if not cases:
        print("标注集为空，退出")
        return

    print(f"载入标注集：{args.set}，共 {len(cases)} 条")

    retrieval_records = evaluate_retrieval(cases, default_retriever)

    answer_records = None
    if args.with_llm:
        print("开始答案层评测（每条都会真实调用大模型，请耐心等待）...")
        answer_records = evaluate_answers(cases, default_answer_runner)

    print_report(retrieval_records, answer_records)

    if args.output:
        write_markdown(args.output, retrieval_records, answer_records)


if __name__ == "__main__":
    main()
