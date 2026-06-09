"""
检索模块 —— 加载 Embedding 模型 + 连接 ChromaDB，提供语义检索。

对外接口:
    search(query, n_results=5) -> list[dict]
"""

import os
from typing import Optional
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import chromadb
from config import CHROMA_DB_PATH, EMBEDDING_MODEL, DEFAULT_TOP_K, RERANK_MODEL, RERANK_TOP_N

# ============================================================
# 模块级单例 —— 整个进程生命周期内只加载一次模型和数据库
# 【底层原理】为什么用模块级变量做单例？
#   Python 的 import 机制保证模块只被加载一次，
#   所以模块级变量天然是单例，不需要写 __new__ 那种复杂的单例模式。
# ============================================================
_embedding_model = None   # SentenceTransformer 模型实例（懒加载）
_chroma_client = None      # ChromaDB 客户端实例
_collection = None         # ChromaDB Collection 实例
_reranker_model = None    # CrossEncoder Rerank 模型实例（懒加载，可选）


def _get_embedding_model():
    """懒加载 Embedding 模型，首次调用时加载，后续复用"""
    global _embedding_model
    if _embedding_model is None:
        print(f"正在加载 Embedding 模型: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding 模型加载完成")
    return _embedding_model


def _get_collection():
    """懒加载 ChromaDB 连接和 Collection"""
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = _chroma_client.get_or_create_collection(
            name="notes",
            metadata={"hnsw:space": "cosine"},  # 用余弦相似度（对归一化向量友好）
        )
        print(f"ChromaDB 连接成功，当前切片数: {_collection.count()}")
    return _collection


def _get_reranker_model() -> Optional[CrossEncoder]:
    """懒加载 Rerank 模型（如果配置了 RERANK_MODEL）"""
    global _reranker_model
    if not RERANK_MODEL:
        return None  # 未启用 Rerank
    if _reranker_model is None:
        print(f"正在加载 Rerank 模型: {RERANK_MODEL}")
        # CrossEncoder 是"交互式"模型：
        #   输入: (query, document) 拼接
        #   输出: 相关性分数（0~1，越大越相关）
        #   比 Bi-Encoder（向量检索）精度高，但速度慢
        _reranker_model = CrossEncoder(RERANK_MODEL)
        print("Rerank 模型加载完成")
    return _reranker_model


# ============================================================
# 公开接口
# ============================================================
def search(query: str, n_results: int = DEFAULT_TOP_K) -> list[dict]:
    """
    语义检索笔记（支持 Rerank 精排）。
    
    流程:
        1. 向量检索（Bi-Encoder）→ 取 Top-K
        2. （可选）Rerank 精排（Cross-Encoder）→ 重新打分，取 Top-N
        3. 返回格式化结果
    
    参数:
        query:      自然语言查询，如 "Django ForeignKey on_delete"
        n_results:  返回切片数，默认 5
    
    返回:
        formatted_results: list[dict]，每个元素包含:
            - "content":    切片原文（str）
            - "source":     源文件路径（str）
            - "header_path": 章节路径（str）
            - "similarity": 余弦相似度（float, 0~1）
    """
    model = _get_embedding_model()
    collection = _get_collection()
    reranker = _get_reranker_model()  # 如果未配置 RERANK_MODEL，返回 None

    # 1. 把查询文本转成向量（归一化，方便余弦相似度计算）
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    # 2. 在 ChromaDB 中检索最相似的 n_results 条（向量检索）
    # 向量检索是"双编码器"（Bi-Encoder），查询和文档分别编码，速度快但精度低；
    #  Rerank 是"交叉编码器"（Cross-Encoder），查询和文档拼接后一起编码，精度高但速度慢。
    #  工业级做法：先用向量检索取 Top-20，再用 Rerank 精排取 Top-5。
    results_raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # 3. 解析 ChromaDB 的返回格式（嵌套列表，取第一层）
    matched_documents = results_raw["documents"][0]   # List[str]
    matched_metadatas = results_raw["metadatas"][0]   # List[dict]
    matched_distances = results_raw["distances"][0]    # List[float]

    # 4. 构造基础结果列表（向量检索结果）
    base_results = []
    # zip() 语法: zip(iterable1, iterable2, ...)
    # 返回值: 一个迭代器，聚合每个可迭代对象的元素。
    # 例如: zip([1, 2], ['a', 'b']) -> (1, 'a'), (2, 'b')
    # 在此处用于并行遍历文档、元数据和距离列表
    for doc, meta, dist in zip(matched_documents, matched_metadatas, matched_distances):
        similarity = 1 - dist  # 余弦距离 → 相似度
        source = meta.get("source", "未知") if meta else "未知"
        header_path = meta.get("header_path", "") if meta else ""
        base_results.append({
            "content": doc,
            "source": source,
            "header_path": header_path,
            "similarity": round(similarity, 4),
        })

    # 5. （可选）Rerank 精排
    if reranker is not None:
        print(f"🔍 正在 Rerank 精排 {len(base_results)} 个结果...")
        # 构造 (query, document) 对
        # 构造 (query, document) 对列表，用于 CrossEncoder 输入
        rerank_pairs = [(query, r["content"]) for r in base_results]
        
        # reranker.predict 返回 numpy.ndarray，形状为 (N,)，元素为 float
        # 值域通常无严格限制，但一般越大表示相关性越高（类似 Logits）
        rerank_scores = reranker.predict(rerank_pairs)
        
        # 将分数绑定到结果对象，并按分数降序排列（越相关越靠前）
        for r, score in zip(base_results, rerank_scores):
            r["rerank_score"] = float(score)
        
        # 原地排序：reverse=True 表示从高分到最低分
        base_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        # 只保留 Top-N
        base_results = base_results[:RERANK_TOP_N]
        print(f"✅ Rerank 完成，返回 Top-{len(base_results)}")

    # 6. 拼接上下文信息（让 LLM 理解切片的来源）
    formatted_results = []
    for r in base_results:
        context_prefix = ""
        if r["header_path"] and r["header_path"] != "(无标题)":
            file_name = os.path.basename(r["source"])
            context_prefix = f"[来源: {file_name} | 章节: {r['header_path']}]\n\n"
        formatted_results.append({
            "content": context_prefix + r["content"],
            "source": r["source"],
            "header_path": r["header_path"],
            "similarity": r["similarity"],
        })

    return formatted_results
