# tests/test_rag_engine.py
# 回归：条款号查询不能被 jieba 切成单字后丢掉，否则覆盖率归零、阈值会把正确切片全部过滤。
from core.rag_engine import apply_reranker, extract_query_terms


def test_article_number_query_terms_are_kept():
    terms = extract_query_terms("第 3 条和第 4 条分别规定了什么？")
    assert "第3条" in terms
    assert "第4条" in terms


def test_article_number_hit_is_strong_evidence():
    candidates = [
        {"content": "第 3 条 【七天无理由全额退款政策】", "rrf_score": 0.03},
        {"content": "第 4 条 【高风险与特殊审批退款场景】", "rrf_score": 0.02},
        {"content": "第一章：精品课程体系与定价规范", "rrf_score": 0.01},
    ]
    reranked = apply_reranker("第 3 条和第 4 条分别规定了什么？", candidates)
    for c in reranked:
        if "第 3 条" in c["content"] or "第 4 条" in c["content"]:
            assert c["rerank_score"] >= 0.8
        else:
            assert c["rerank_score"] < 0.35

