# scripts/eval_rag.py
# 知识库问答的离线评测脚本：检索指标 + 答案质量 + 延迟。
#
# 用法：
#   python scripts/eval_rag.py                        只评检索层（免费，秒级）
#   python scripts/eval_rag.py --with-llm             再评答案质量（真实调用大模型，按条计费）
#   python scripts/eval_rag.py --limit 5              只跑前 5 条
#   python scripts/eval_rag.py --top-k 5              检索深度，影响 MRR 的计算范围
#   python scripts/eval_rag.py --mode dense           只走向量召回，用来量化 dense 单路贡献
#   python scripts/eval_rag.py --mode sparse          只走 BM25，用来量化关键词单路贡献
#   python scripts/eval_rag.py --set data/eval_set_hard.jsonl   跑更难的第二套题
#   python scripts/eval_rag.py --output data/eval_report.md
import argparse
import json
import os
import re
import sys
import time
import uuid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Windows 控制台默认 GBK，直接打印 ✅/❌ 会抛 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_SET_PATH = os.path.join(BASE_DIR, "data", "eval_set.jsonl")

# 对外口径：可答用例只看前 2 条切片
RECALL_K = 2

# 应拒答的问题，答案里出现这些词就算合规拒答
REFUSAL_KEYWORDS = [
    "未检索到", "无法提供", "无法解答", "不在", "转接人工", "人工顾问", "严禁咨询",
    "未说明", "未记载", "未收录", "未提及", "未涉及", "没有说明", "没有明确", "无法查询",
    "无法基于文档", "严禁", "无法保证", "不能保证", "无法承诺", "不能承诺", "没有相关",
    "未提供", "不支持",
]

# 模型拒答时措辞会变（无法确认 / 未包含 / 没有提到），固定词表容易漏判，补一组正则
REFUSAL_PATTERNS = [
    r"未(检索到|说明|记载|收录|提及|涉及|包含|提供|给出)",
    r"没有(说明|明确|相关|提到|给出|关于)",
    r"无法(提供|解答|查询|保证|承诺|确认|基于文档)",
    r"(不|无法)支持",
    r"转接人工|人工顾问|严禁咨询",
]


def is_refusal_answer(answer):
    """判断答案是否明确拒答或说明文档没有依据，用于应拒类用例的合规性检查。"""
    if any(word in answer for word in REFUSAL_KEYWORDS):
        return True
    return any(re.search(pattern, answer) for pattern in REFUSAL_PATTERNS)

# 判断“禁止词”时要能识别否定语境：
# “无法保证就业”里虽然出现了“保证就业”，但它不是承诺，不能算幻觉。
NEGATION_WORDS = ["不", "无法", "不能", "未", "没有", "严禁", "不可", "禁止", "非"]

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


def normalize_text(text):
    """去掉所有空白，避免 PDF 硬换行和模型排版影响关键词匹配"""
    return "".join(str(text).split())


def contains_keyword(answer, keyword):
    return normalize_text(keyword) in normalize_text(answer)


def contains_banned(answer, phrase):
    """
    答案里是否出现了“不该出现”的说法。

    同一个短语可能出现在否定句里，例如：
      “我们无法保证就业” —— 包含“保证就业”，但这是拒答，不是幻觉。
    所以逐次查找，只有短语前面没有否定词时才算命中。
    """
    text = normalize_text(answer)
    target = normalize_text(phrase)
    if not target:
        return False

    start = 0
    while True:
        index = text.find(target, start)
        if index == -1:
            return False
        prefix = text[max(0, index - 4):index]
        if not any(word in prefix for word in NEGATION_WORDS):
            return True
        start = index + len(target)


def matches_expected(chunk, source, page):
    """切片是否命中标注的来源文件与页码"""
    if chunk.get("source") != source:
        return False
    if page is not None and chunk.get("page") != page:
        return False
    return True


def average(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values, ratio):
    """简单的百分位：排序后按位置取，样本少时够用"""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = int(round((len(ordered) - 1) * ratio))
    return ordered[position]


def default_retriever(query, top_k=5):
    """真实检索器；延迟 import，使指标计算部分不依赖向量库"""
    from core.rag_engine import search_knowledge_base
    return search_knowledge_base(query, top_n=top_k)


def build_retriever(mode):
    """
    选择检索通道，用来做消融对比。

    - hybrid：生产链路，dense + sparse + RRF + 关键词覆盖率精排 + 阈值截断
    - dense ：只走向量召回，不经过阈值，能看到“纯语义”的上限和延迟
    - sparse：只走 BM25，不经过阈值，能看到“纯关键词”的上限和延迟
    """
    if mode == "dense":
        def dense_retriever(query, top_k=5):
            from core.rag_engine import dense_search
            chunks = dense_search(query, top_k=top_k)
            for chunk in chunks:
                # 距离越小越相似，转成 0~1 的分数只用于报告展示
                distance = chunk.get("dense_distance", 0.0)
                chunk["rerank_score"] = round(1.0 / (1.0 + distance), 4)
            return chunks
        return dense_retriever

    if mode == "sparse":
        def sparse_retriever(query, top_k=5):
            from core.rag_engine import sparse_search
            chunks = sparse_search(query, top_k=top_k)
            for chunk in chunks:
                chunk["rerank_score"] = round(chunk.get("bm25_score", 0.0), 4)
            return chunks
        return sparse_retriever

    return default_retriever


def default_answer_runner(query, thread_id):
    """真实问答执行器：走完整状态机，每条用例一个独立会话"""
    from langchain_core.messages import HumanMessage
    from core.agent_workflow import app_graph

    config = {"configurable": {"thread_id": thread_id}}
    result = app_graph.invoke({
        "messages": [HumanMessage(content=query)],
        "user_id": "eval_user",
        "order_id": None,
        "citations": [],
        "need_human_approval": False,
        "refund_requested": False,
        "tool_events": [],
    }, config=config)
    return result.get("final_response") or "", result.get("citations") or []


def evaluate_retrieval(cases, retriever, top_k):
    """
    检索层评测

    - 可答用例：前 RECALL_K 条里是否出现正确切片（Recall@2），并记录首次命中的排名（MRR）
    - 应拒用例：一条都不召回算通过；如果召回的首条是“政策本身”（例如禁止代写作业的条款），也算通过。
      这里只看首条：首条命中禁止条款就足以支撑拒答，低排名混入无关切片不应判失败。
    """
    records = []

    # 预热一次，避免第一条用例把模型加载时间算进延迟
    if cases:
        retriever("七天无理由退款", top_k)

    for case in cases:
        start = time.perf_counter()
        chunks = retriever(case["query"], top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        source = case.get("expect_source")
        page = case.get("expect_page")

        first_rank = None
        for rank, chunk in enumerate(chunks, start=1):
            if matches_expected(chunk, source, page):
                first_rank = rank
                break

        if case["expect"] == "answer":
            hit = first_rank is not None and first_rank <= RECALL_K
            reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
        else:
            if not chunks:
                hit = True
            elif source:
                hit = matches_expected(chunks[0], source, page)
            else:
                hit = False
            reciprocal_rank = None

        records.append({
            "id": case["id"],
            "query": case["query"],
            "expect": case["expect"],
            "chunk_count": len(chunks),
            "first_rank": first_rank,
            "reciprocal_rank": reciprocal_rank,
            "top_score": chunks[0]["rerank_score"] if chunks else 0.0,
            "latency_ms": latency_ms,
            "hit": hit,
            "note": case.get("note", ""),
        })

    return records


def evaluate_answers(cases, answer_runner):
    """答案层评测：关键词覆盖、引用页码命中、该拒的有没有拒"""
    records = []

    for case in cases:
        thread_id = f"eval_{case['id']}_{uuid.uuid4().hex[:6]}"
        start = time.perf_counter()
        answer, citations = answer_runner(case["query"], thread_id)
        latency_ms = (time.perf_counter() - start) * 1000

        banned = [word for word in case.get("must_not_contain", [])
                  if contains_banned(answer, word)]

        if case["expect"] == "answer":
            missing = []
            for keyword in case.get("must_contain", []):
                if not contains_keyword(answer, keyword):
                    missing.append(keyword)
            passed = len(missing) == 0 and not banned

            reason_parts = []
            if missing:
                reason_parts.append(f"缺少关键词：{'、'.join(missing)}")
            if banned:
                reason_parts.append(f"出现禁止词：{'、'.join(banned)}")
            reason = "；".join(reason_parts)

            citation_hit = None
            source = case.get("expect_source")
            page = case.get("expect_page")
            if source and page is not None:
                expected_label = f"《{source}》第{page}页"
                citation_hit = expected_label in citations
        else:
            refused = is_refusal_answer(answer)
            hallucinated = any(word in answer for word in HALLUCINATION_WORDS)
            passed = refused and not hallucinated and not banned
            citation_hit = None
            if hallucinated:
                reason = "出现幻觉内容"
            elif banned:
                reason = f"出现禁止词：{'、'.join(banned)}"
            elif not refused:
                reason = "未合规拒答"
            else:
                reason = ""

        records.append({
            "id": case["id"],
            "expect": case["expect"],
            "passed": passed,
            "has_citation": len(citations) > 0,
            "citation_hit": citation_hit,
            "latency_ms": latency_ms,
            "answer": answer,
            "reason": reason,
        })

    return records


def build_summary(retrieval_records, answer_records=None):
    """汇总成一组可横向对比的数字"""
    answer_cases = [r for r in retrieval_records if r["expect"] == "answer"]
    refuse_cases = [r for r in retrieval_records if r["expect"] == "refuse"]

    retrieval_latencies = [r["latency_ms"] for r in retrieval_records]
    summary = {
        "总数": len(retrieval_records),
        "可答数": len(answer_cases),
        "应拒数": len(refuse_cases),
        "Recall@2": average([1.0 if r["hit"] else 0.0 for r in answer_cases]),
        "MRR": average([r["reciprocal_rank"] for r in answer_cases]),
        "应拒拦截率": average([1.0 if r["hit"] else 0.0 for r in refuse_cases]),
        "检索P50": percentile(retrieval_latencies, 0.5),
        "检索P95": percentile(retrieval_latencies, 0.95),
    }

    if answer_records is not None:
        citation_cases = [r for r in answer_records if r["citation_hit"] is not None]
        refusal_cases = [r for r in answer_records if r["expect"] == "refuse"]
        answer_latencies = [r["latency_ms"] for r in answer_records]

        summary["答案通过率"] = average([1.0 if r["passed"] else 0.0 for r in answer_records])
        summary["引用命中率"] = average([1.0 if r["citation_hit"] else 0.0 for r in citation_cases])
        summary["拒答准确率"] = average([1.0 if r["passed"] else 0.0 for r in refusal_cases])
        summary["答案P50"] = percentile(answer_latencies, 0.5)
        summary["答案P95"] = percentile(answer_latencies, 0.95)

    return summary


def print_report(retrieval_records, answer_records, top_k, mode="hybrid"):
    summary = build_summary(retrieval_records, answer_records)
    score_label = "置信度" if mode == "hybrid" else "分数"

    print("=" * 72)
    print("知识库问答评测报告")
    print("=" * 72)
    print(f"用例 {summary['总数']} 条（可答 {summary['可答数']} / 应拒 {summary['应拒数']}）"
          f"  检索深度 top_k={top_k}  检索模式={mode}")

    print("\n【检索层】")
    print(f"  Recall@{RECALL_K}     : {summary['Recall@2']:.1%}")
    print(f"  MRR           : {summary['MRR']:.3f}")
    print(f"  应拒拦截率     : {summary['应拒拦截率']:.1%}")
    print(f"  检索延迟       : P50 {summary['检索P50']:.1f} ms / P95 {summary['检索P95']:.1f} ms")

    print("\n【检索明细】")
    for record in retrieval_records:
        mark = "✅" if record["hit"] else "❌"
        expect_label = "可答" if record["expect"] == "answer" else "应拒"
        rank = record["first_rank"] if record["first_rank"] else "-"
        rr = record["reciprocal_rank"]
        rr_text = f"{rr:.2f}" if rr is not None else "-"
        line = (f"  {mark} {record['id']} {expect_label} "
                f"召回{record['chunk_count']}条 首中{rank} RR{rr_text} "
                f"{score_label}{record['top_score']:.3f} {record['latency_ms']:.1f}ms "
                f"{record['query']}")
        if record["note"]:
            line += f"  ← {record['note']}"
        print(line)

    if answer_records is None:
        return

    print("\n【答案层】")
    print(f"  答案通过率     : {summary['答案通过率']:.1%}")
    print(f"  引用命中率     : {summary['引用命中率']:.1%}")
    print(f"  拒答准确率     : {summary['拒答准确率']:.1%}")
    print(f"  答案延迟       : P50 {summary['答案P50']:.1f} ms / P95 {summary['答案P95']:.1f} ms")

    print("\n【答案明细】")
    for record in answer_records:
        mark = "✅" if record["passed"] else "❌"
        if record["citation_hit"] is None:
            citation_text = "-"
        elif record["citation_hit"]:
            citation_text = "命中"
        else:
            citation_text = "未命中"
        line = f"  {mark} {record['id']} 引用{citation_text} {record['latency_ms']:.1f}ms"
        if record["reason"]:
            line += f"  ← {record['reason']}"
        print(line)
        print(f"      回答: {record['answer'][:70]}...")


def write_markdown(path, retrieval_records, answer_records, top_k, mode="hybrid"):
    """把报告落成 Markdown，方便贴进 README 或做版本间对比"""
    summary = build_summary(retrieval_records, answer_records)
    score_label = "置信度" if mode == "hybrid" else "分数"

    lines = []
    lines.append("# 知识库问答评测报告\n")
    lines.append(f"- 检索模式：{mode}")
    lines.append(f"- 用例总数：{summary['总数']}（可答 {summary['可答数']} / 应拒 {summary['应拒数']}）")
    lines.append(f"- Recall@{RECALL_K}：{summary['Recall@2']:.1%}")
    lines.append(f"- MRR（前 {top_k} 条）：{summary['MRR']:.3f}")
    lines.append(f"- 应拒拦截率：{summary['应拒拦截率']:.1%}")
    lines.append(f"- 检索 P50 / P95：{summary['检索P50']:.1f} ms / {summary['检索P95']:.1f} ms")
    if answer_records is not None:
        lines.append(f"- 答案通过率：{summary['答案通过率']:.1%}")
        lines.append(f"- 引用命中率：{summary['引用命中率']:.1%}")
        lines.append(f"- 拒答准确率：{summary['拒答准确率']:.1%}")
        lines.append(f"- 答案 P50 / P95：{summary['答案P50']:.1f} ms / {summary['答案P95']:.1f} ms")
    lines.append("")

    lines.append("## 检索明细\n")
    lines.append(f"| 用例 | 期望 | 召回数 | 首中排名 | RR | {score_label} | 延迟(ms) | 结果 | 备注 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for record in retrieval_records:
        mark = "✅" if record["hit"] else "❌"
        rank = record["first_rank"] if record["first_rank"] else "-"
        rr = record["reciprocal_rank"]
        rr_text = f"{rr:.2f}" if rr is not None else "-"
        lines.append(
            f"| {record['id']} | {record['expect']} | {record['chunk_count']} "
            f"| {rank} | {rr_text} | {record['top_score']:.3f} "
            f"| {record['latency_ms']:.1f} | {mark} | {record['note']} |"
        )

    if answer_records is not None:
        lines.append("\n## 答案明细\n")
        lines.append("| 用例 | 结果 | 引用命中 | 延迟(ms) | 备注 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for record in answer_records:
            mark = "✅" if record["passed"] else "❌"
            if record["citation_hit"] is None:
                citation_text = "-"
            elif record["citation_hit"]:
                citation_text = "命中"
            else:
                citation_text = "未命中"
            note = record["reason"] or normalize_text(record["answer"])[:40]
            lines.append(
                f"| {record['id']} | {mark} | {citation_text} "
                f"| {record['latency_ms']:.1f} | {note} |"
            )

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    print(f"\n📝 报告已写入：{path}")


def main():
    parser = argparse.ArgumentParser(description="知识库问答离线评测")
    parser.add_argument("--set", default=DEFAULT_SET_PATH, help="标注集路径（JSONL）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全部")
    parser.add_argument("--top-k", type=int, default=5, help="检索深度，用于 MRR 计算")
    parser.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid",
                        help="检索通道：hybrid=生产链路，dense=只走向量，sparse=只走 BM25")
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

    retriever = build_retriever(args.mode)
    retrieval_records = evaluate_retrieval(cases, retriever, args.top_k)

    answer_records = None
    if args.with_llm:
        if args.mode != "hybrid":
            print("⚠️ --mode 只影响检索层；答案层仍走生产 hybrid 链路，避免两套检索结果混淆")
        print("开始答案层评测（每条都会真实调用大模型，请耐心等待）...")
        answer_records = evaluate_answers(cases, default_answer_runner)

    print_report(retrieval_records, answer_records, args.top_k, args.mode)

    if args.output:
        write_markdown(args.output, retrieval_records, answer_records, args.top_k, args.mode)


if __name__ == "__main__":
    main()
