# rag_engine.py
import os
import re
import jieba
import chromadb
from rank_bm25 import BM25Okapi
# 动态定位到项目根目录下的 chroma_db
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
# 条款号查询（"第 3 条"）在 jieba 里会被拆成单字，需要单独保留
ARTICLE_PATTERN = re.compile(r"第\s*\d+\s*条")


def tokenize(text: str):
    """BM25 统一分词：先去空白，再补回"第3条"这种条款号 token。"""
    compact = "".join(text.split())
    tokens = list(jieba.cut(compact))
    for match in ARTICLE_PATTERN.findall(text):
        term = "".join(match.split())
        if term not in tokens:
            tokens.append(term)
    return tokens

# 1. 连接本地持久化向量库 (Dense 稠密索引)
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name="edu_pdf_knowledge")

# 2. 从本地 Chroma 中读取所有切片，在内存中构建 BM25 词袋 (Sparse 稀疏索引)
ALL_DOCS = collection.get()
CHUNKS = []
TOKENIZED_CORPUS = []

if ALL_DOCS and ALL_DOCS["documents"]:
    for i in range(len(ALL_DOCS["documents"])):
        chunk_data = {
            "chunk_id": ALL_DOCS["ids"][i],
            "content": ALL_DOCS["documents"][i],
            "source": ALL_DOCS["metadatas"][i]["source"],
            "page": ALL_DOCS["metadatas"][i]["page"]
        }
        CHUNKS.append(chunk_data)
        tokens = tokenize(chunk_data["content"])
        TOKENIZED_CORPUS.append(tokens)

bm25 = BM25Okapi(TOKENIZED_CORPUS) if TOKENIZED_CORPUS else None

print(">>> RAG 引擎就绪：采用【BM25 + 向量双路召回 + RRF 融合动态精排】流水线（毫秒级免下载）")


def dense_search(query: str, top_k: int = 5):
    """【粗排阶段 1】Dense 稠密向量语义召回 (Chroma)"""
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=top_k)
    matched = []
    for i in range(len(results["ids"][0])):
        matched.append({
            "chunk_id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "dense_distance": results["distances"][0][i]
        })
    return matched


def sparse_search(query: str, top_k: int = 5):
    """【粗排阶段 2】Sparse 稀疏关键词精确召回 (BM25)"""
    if not bm25 or not CHUNKS:
        return []
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    matched = []
    for idx in top_indices:
        if scores[idx] > 0:
            item = dict(CHUNKS[idx])
            item["bm25_score"] = scores[idx]
            matched.append(item)
    return matched


def rrf_fusion(dense_results, sparse_results, k: int = 60):
    """
    【粗排融合阶段】RRF (倒数排名融合算法)
    Score = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
    抹平向量距离与 BM25 分数量纲差异，输出初步粗排列表
    """
    rrf_score_map = {}
    doc_map = {}

    for rank, doc in enumerate(dense_results):
        cid = doc["chunk_id"]
        doc_map[cid] = doc
        rrf_score_map[cid] = rrf_score_map.get(cid, 0.0) + (1.0 / (k + rank + 1))

    for rank, doc in enumerate(sparse_results):
        cid = doc["chunk_id"]
        doc_map[cid] = doc
        rrf_score_map[cid] = rrf_score_map.get(cid, 0.0) + (1.0 / (k + rank + 1))

    sorted_docs = sorted(rrf_score_map.items(), key=lambda x: x[1], reverse=True)
    fused = []
    for cid, score in sorted_docs:
        d = dict(doc_map[cid])
        d["rrf_score"] = round(score, 4)
        fused.append(d)
    return fused


def normalize_text(text: str):
    """
    去掉所有空白与换行
    PDF 抽出来的正文经常在词中间硬换行（例如"相关问\\n题"），
    不归一化就会让"问题"这种词精确匹配失败，覆盖率被算低，好切片被阈值误杀。
    """
    return "".join(text.split())


def extract_query_terms(query: str):
    """
    切出用于打分的查询词
    丢弃空白和单字：像"的/了/是/吗"这类单字在任何切片里都高频出现，
    留在分母里会把覆盖率算虚高，让无关切片也能轻松过阈值。

    额外保留"第 3 条""第4条"这类条款号：jieba 会把数字和"条"切成单字，
    再被上面的规则丢掉，导致"第 3 条和第 4 条分别规定了什么"覆盖率直接归零。
    """
    terms = []
    for token in jieba.cut(query):
        token = token.strip()
        if len(token) > 1:
            terms.append(token)

    for match in ARTICLE_PATTERN.findall(query):
        term = "".join(match.split())
        if term not in terms:
            terms.append(term)
    return terms


def apply_reranker(query: str, candidates: list):
    """
    【精排打分阶段】
    打分 = 0.8 × 关键词覆盖率(绝对证据) + 0.2 × RRF 排名分(相对排序)

    为什么不能只做 RRF 归一化：
    RRF 分数是纯排名信号（1/(k+rank)），Top1 归一化后恒等于约 0.95，
    第 2 名和第 1 名的分差又极小，所以"小于 threshold 就丢弃"实际上永远不触发，
    阈值形同虚设，防幻觉全靠后面的大模型自觉。
    引入关键词覆盖率后，分数才有绝对含义：
    覆盖率 0 的切片（与问题毫无交集）无论如何都过不了 0.35 这条线。
    """
    if not candidates:
        return []

    query_terms = extract_query_terms(query)
    article_terms = ["".join(match.split()) for match in ARTICLE_PATTERN.findall(query)]
    base_max = max(c.get("rrf_score", 0.01) for c in candidates)

    for candidate in candidates:
        # 归一化后再比对，避免 PDF 的硬换行把词切断
        content = normalize_text(candidate.get("content", ""))

        # 绝对证据：查询词有多少比例真的出现在这段切片里
        if query_terms:
            hit_count = 0
            for term in query_terms:
                if term in content:
                    hit_count += 1
            coverage = hit_count / len(query_terms)
        else:
            coverage = 0.0

        # 查询里明确写了“第 N 条”时，命中条款号就是强证据。
        # “分别规定了什么”这类词在条款正文里通常不会出现，只靠关键词覆盖率会把正确切片误杀。
        if article_terms and any(term in content for term in article_terms):
            coverage = max(coverage, 1.0)

        # 相对排序：保留双路召回的名次信息，只在排序上起辅助作用
        rank_score = min(candidate.get("rrf_score", 0) / (base_max * 1.05), 1.0)

        candidate["keyword_coverage"] = round(coverage, 4)
        candidate["rerank_score"] = round(0.8 * coverage + 0.2 * rank_score, 4)

    # 按精排置信度降序排列
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates


def search_knowledge_base(query: str, top_n: int = 2, threshold: float = 0.35):
    """
    【工业级对外统一接口】
    1. 双路并行召回 (Dense Top 8 + Sparse Top 8)
    2. RRF 倒数排名融合 (初步去重粗排)
    3. 精排打分器打分 (Rerank Scoring，含绝对关键词覆盖率与条款号强证据)
    4. 动态置信度截断 (低于 threshold 的切片直接丢弃，不交给大模型，从源头阻断幻觉)

    threshold 的含义：0.8 × 覆盖率 + 0.2 × 排名分。
    经验上 0.35 约等于"至少 20%~30% 的查询词能在切片里命中"，
    若你的业务问句口语化严重（同义词多），可以降到 0.30；反之可提到 0.45。
    """
    # 1. 双路粗排
    dense_candidates = dense_search(query, top_k=8)
    sparse_candidates = sparse_search(query, top_k=8)

    # 2. 粗排融合
    fused_candidates = rrf_fusion(dense_candidates, sparse_candidates)[:8]

    if not fused_candidates:
        return []

    # 3. 精排打分
    reranked_candidates = apply_reranker(query, fused_candidates)

    # 4. 置信度截断过滤
    valid_chunks = [c for c in reranked_candidates if c.get("rerank_score", 0) >= threshold]

    return valid_chunks[:top_n]