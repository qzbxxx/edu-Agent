# tests/test_hybrid.py 顶部修改：
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.rag_engine import dense_search, sparse_search, search_knowledge_base

def run_comparison():
    print("="*70)
    print("🔬 Hybrid RAG (双路召回 + RRF 融合精排) 效果实测对比")
    print("="*70)

    # 准备两个典型 Query：
    # 1. 编号精确查询（纯向量检索的软肋，BM25 的强项）
    # 2. 泛语义咨询（BM25 容易漏掉，纯向量的强项）
    queries = [
        "第三条七天无理由退款的核心条款是什么？",
        "想成为分布式和微服务架构师，推荐学哪个课？"
    ]

    for q in queries:
        print(f"\n🔍 测试提问: 【{q}】")
        print("-" * 65)

        # 1. 纯向量检索 (Chroma)
        d_res = dense_search(q, top_k=2)
        print("📌 [1. 纯 Dense 向量召回 Top 2]:")
        for idx, r in enumerate(d_res):
            print(f"   ({idx+1}) [第{r['page']}页] 距离: {round(r['dense_distance'], 4)} | 内容: {r['content'][:45]}...")

        # 2. 纯 BM25 关键词检索
        s_res = sparse_search(q, top_k=2)
        print("📌 [2. 纯 Sparse BM25 召回 Top 2]:")
        for idx, r in enumerate(s_res):
            print(f"   ({idx+1}) [第{r['page']}页] BM25分: {round(r['bm25_score'], 2)} | 内容: {r['content'][:45]}...")

        # 3. 最终 Hybrid RAG 输出 (融合 + 精排)
        final_res = search_knowledge_base(q, top_n=2)
        print("🌟 [3. Hybrid RAG 最终输出 (送给大模型的 Context)]:")
        for idx, r in enumerate(final_res):
            print(f"   🏆 排名 {idx+1} [第{r['page']}页 | 精排置信度: {r['rerank_score']} "
                  f"| 关键词覆盖率: {r['keyword_coverage']}] | 内容: {r['content'][:45]}...")
        if not final_res:
            print("   ⚠️ 全部低于阈值，走兜底话术（阈值可能偏严）")

if __name__ == "__main__":
    run_comparison()