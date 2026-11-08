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


def test_question_words_do_not_dilute_query_terms():
    # “哪个/多少”不携带业务信息，若留在覆盖率分母里，会把两个课程切片都压到阈值以下
    terms = extract_query_terms("Python 高级架构师和 AIGC 全栈营，哪个更贵？贵多少？")
    assert "Python" in terms
    assert "AIGC" in terms
    assert "哪个" not in terms
    assert "多少" not in terms
