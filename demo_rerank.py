# demo_rerank.py
# 检索链路演示脚本：双路召回 -> RRF 融合 -> 规则化置信度重排 -> 阈值截断
import os
import sys

# 脚本可能从任意目录启动，先把项目根目录加入寻路列表
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from core.rag_engine import (
    dense_search, sparse_search, rrf_fusion, apply_reranker, search_knowledge_base
)

query = "第七条无理由退款是什么要求？"
print(f"用户提问: {query}\n" + "=" * 60)

# 1. 粗排：两路各取 Top3，再用 RRF 融合
dense_res = dense_search(query, top_k=3)
sparse_res = sparse_search(query, top_k=3)
fused = rrf_fusion(dense_res, sparse_res)[:3]

print("【阶段一：RRF 粗排候选】:")
for idx, c in enumerate(fused):
    print(f"  {idx + 1}. [{c['source']} 第{c['page']}页] RRF分: {c['rrf_score']} "
          f"| 内容: {c['content'][:40]}...")

# 2. 精排：0.8 × 关键词覆盖率 + 0.2 × 排名分
#    注意：这里不是 Cross-Encoder 模型，而是零依赖的规则打分
reranked = apply_reranker(query, fused)

print("\n【阶段二：规则化置信度重排（覆盖率 0.8 + 排名 0.2）】:")
for idx, c in enumerate(reranked):
    print(f"  🌟 排名 {idx + 1}. 置信度: {c['rerank_score']} "
          f"| 关键词覆盖率: {c['keyword_coverage']} "
          f"| 第{c['page']}页 | 内容: {c['content'][:40]}...")

# 3. 对外统一接口：低于阈值直接丢弃，不交给大模型
final_chunks = search_knowledge_base(query, top_n=2)
print(f"\n【阶段三：阈值截断后真正交给大模型的切片（{len(final_chunks)} 条）】:")
if not final_chunks:
    print("  ⚠️ 全部低于阈值，触发兜底话术「未检索到，转接人工顾问」")
for idx, c in enumerate(final_chunks):
    print(f"  ✅ {idx + 1}. 置信度: {c['rerank_score']} "
          f"| 第{c['page']}页 | 内容: {c['content'][:40]}...")

# 4. 反向验证：与业务完全无关的提问必须被拦在模型之外
off_topic = "你们教做红烧牛肉面吗？"
kept = search_knowledge_base(off_topic, top_n=2)
print(f"\n【阶段四：无关提问必须被拦截】提问: {off_topic}")
print(f"  截断后剩余切片: {len(kept)} 条（预期 0 条，否则阈值未生效）")