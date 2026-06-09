"""
索引模块 —— 加载笔记 → 切分 → Embedding → 写入 ChromaDB。

对外接口:
    reindex_all()           全量重建索引
    index_file(file_path)   增量索引单个文件
    list_sources()          列出已索引的源文件
"""

import os
import re
import yaml
import glob
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import chromadb
from config import NOTES_DIRS, CHROMA_DB_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


# ============================================================
# 辅助函数
# ============================================================

def _load_model():
    """直接创建 SentenceTransformer 实例，参数是模型名。首次调用会从 HuggingFace 下载模型并缓存。"""
    return SentenceTransformer(EMBEDDING_MODEL)


def _load_collection():
    """打开（或创建）ChromaDB Collection"""
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)  # ChromaDB 的持久化客户端，数据写入 SQLite
    return client.get_or_create_collection(
        name="notes",
        metadata={"hnsw:space": "cosine"},  # 使用 HNSW 索引（高效近似最近邻搜索），距离函数为余弦
    )


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """
    解析 Markdown 文件顶部的 YAML Front Matter。
    
    格式:
        ---
        title: "文章标题"
        tags: [标签1, 标签2]
        ---
        正文内容...
    
    返回:
        (metadata, cleaned_text)
        - metadata:    解析出的 YAML 字典（如果没有则返回空 dict）
        - cleaned_text: 去掉 YAML Front Matter 后的正文
    """
    yaml_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
    match = yaml_pattern.match(text)
    
    if not match:
        return {}, text  # 没有 Front Matter，返回空元数据和原文
    
    yaml_text = match.group(1)
    try:
        metadata = yaml.safe_load(yaml_text) or {}  # 如果匹配成功，用 yaml.safe_load() 解析 YAML 内容
    except yaml.YAMLError as e:
        print(f"⚠️  YAML 解析失败: {e}")
        metadata = {}
    
    # 去掉 YAML Front Matter，只保留正文
    cleaned_text = yaml_pattern.sub('', text)
    return metadata, cleaned_text


def _split_markdown(file_path: str) -> list[dict]:
    """
    混合切分策略：
    1. 解析 YAML Front Matter
    2. 按 ## / ### 标题切分（保留结构）
    3. 再按 CHUNK_SIZE 二次切分（控制长度）
    
    返回: [{"content": "切片文本", "metadata": {"source": "源路径", "header_path": "标题路径", ...}}, ...]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ============================================================
    # 第 0 步：解析 YAML Front Matter（如果存在）
    # ============================================================
    front_matter, cleaned_text = _parse_front_matter(text)
    # front_matter 示例: {"title": "Django 教程", "tags": ["Python", "Web"]}

    # ============================================================
    # 第一步：按 Markdown 标题切分
    # ============================================================
    headers_to_split_on = [("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # 保留标题文本在切片里
    )
    # split_text 返回 List[Document]，其中 Document 是 LangChain 的核心数据类
    # 结构: Document(page_content="文本内容", metadata={"键": "值"})
    doc_slices = md_splitter.split_text(cleaned_text)

    # 兜底逻辑：如果 Markdown 中没有匹配到 ## 或 ### 标题，splitter 可能返回空列表
    # 此时将整个正文作为一个单独的 Document 对象，防止后续处理丢失内容
    if not doc_slices:
        doc_slices = [Document(page_content=cleaned_text, metadata={})]

    # ============================================================
    # 第二步：按长度二次切分（让 CHUNK_SIZE / CHUNK_OVERLAP 生效）
    # ============================================================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""],  # 优先在段落/句子边界切
    )

    # 整理成统一格式
    chunks_data = []
    for doc in doc_slices:
        content = doc.page_content.strip()
        if not content:
            continue

        # 构造层级标题路径（如 "Django 基础 > Model 字段"）
        # doc 是 LangChain 的 Document 对象，结构为:
        #   - doc.page_content: str, 当前切片的文本内容
        #   - doc.metadata: dict, 包含层级标题信息，例如 {"h2": "Django 基础", "h3": "Model 字段"}
        
        # header_path 是将层级标题拼接成的字符串，用于标识内容在文档中的位置
        # 例如: "Django 基础 > Model 字段"
        # 如果没有任何标题，则返回 "(无标题)"
        header_parts = []
        for level in ["h2", "h3"]:
            if level in doc.metadata:
                header_parts.append(doc.metadata[level])
        header_path = " > ".join(header_parts) if header_parts else "(无标题)"

        # 对该片段按长度二次切分
        sub_chunks = text_splitter.split_text(doc.page_content)
        for sub_text in sub_chunks:
            sub_text = sub_text.strip()
            if not sub_text:
                continue
            
            # 【关键】把 YAML Front Matter 的元数据也合并进来
            # 这样检索时可以看到 title、tags 等信息
            chunk_metadata = {
                "source": file_path,
                "header_path": header_path,
            }
            # 合并 YAML Front Matter（如 title, tags）
            if front_matter:
                for key, value in front_matter.items():
                    chunk_metadata[f"fm_{key}"] = value  # fm_ 前缀表示来自 Front Matter
            
            chunks_data.append({
                "content": sub_text,
                "metadata": chunk_metadata,
            })

    return chunks_data


def _find_md_files() -> list[str]:
    """扫描配置的笔记目录，收集所有 .md 文件"""
    files = []
    for directory in NOTES_DIRS:
        if os.path.isdir(directory):
            # 递归搜索所有 .md 文件
            # glob.glob 用于查找匹配特定模式的路径名
            # os.path.join(directory, "**", "*.md") 构造搜索模式：
            #   - directory: 基础目录
            #   - "**": 匹配任意子目录（包括当前目录）
            #   - "*.md": 匹配以 .md 结尾的文件
            # recursive=True: 启用递归搜索，使 "**" 生效
            # 返回值: 匹配到的文件绝对路径列表 (list[str])
            found = glob.glob(os.path.join(directory, "**", "*.md"), recursive=True)
            files.extend(found)
    return files


# ============================================================
# 公开接口
# ============================================================

def reindex_all():
    """清空现有索引，重新扫描所有笔记目录并全量入库。"""
    model = _load_model()
    collection = _load_collection()

    # 1. 清空旧数据
    # 先获取所有已有 id，逐个删除
    # collection.get() 返回一个字典，包含 "ids", "embeddings", "documents", "metadatas" 等键
    # 我们需要获取所有已存在的 ID，以便在删除时精准匹配
    existing = collection.get()
    if existing["ids"]:
        # ChromaDB 的 delete 方法需要通过 ids 列表来指定要删除的记录
        # 这些 ids 是在之前调用 collection.add() 时创建的（见下文 reindex_all 中的 ids 生成逻辑）
        collection.delete(ids=existing["ids"])

    # 2. 扫描所有 .md 文件
    md_files = _find_md_files()
    if not md_files:
        return f"未在 {NOTES_DIRS} 下找到任何 .md 文件"

    # 3. 逐文件处理
    total_chunks = 0
    for file_path in md_files:
        chunks = _split_markdown(file_path)
        if not chunks:
            continue

        # 4. 准备批量写入的数据
        # 从切分结果中提取文本内容、元数据和唯一ID，准备批量写入数据库
        # 列表推导式 (List Comprehension)：一种 Pythonic 的简洁语法，用于从现有列表创建新列表
        # 语法结构: [表达式 for 变量 in 可迭代对象]
        # 执行逻辑:
        #   1. 遍历 chunks 列表中的每一个元素，临时赋值给变量 chunk
        #   2. 对每个 chunk 执行表达式 chunk["content"]，提取其文本内容
        #   3. 将所有提取出的文本内容收集到一个新的列表中
        # 返回值:
        #   一个包含所有切片文本内容的字符串列表 (list[str])
        #   例如: ["第一段内容...", "第二段内容..."]
        texts = [chunk["content"] for chunk in chunks]       # 提取所有切片的文本内容
        metadatas = [chunk["metadata"] for chunk in chunks]  # 提取所有切片的元数据（如来源文件、标题路径等）
        # 为每个切片生成唯一ID：格式为 "文件路径__索引号"，确保同一文件的不同切片ID不重复
        # 生成唯一 ID 列表，格式为 "文件路径__索引号"
        # 
        # 1. idx 是什么值？
        #    idx 是列表推导式中的循环变量，代表当前切片在 chunks 列表中的下标（从 0 开始递增）。
        #    例如：如果有 3 个切片，idx 依次为 0, 1, 2。
        #
        # 2. 为什么要用 range(len(chunks))？
        #    range(n) 生成一个从 0 到 n-1 的整数序列。
        #    结合列表推导式，它为每个切片分配一个唯一的、连续的数字标识。
        #
        # 3. 会随机重复吗？
        #    不会。这里没有使用随机函数（如 random），而是使用确定性的计数。
        #    ID 的唯一性由两部分保证：
        #    - file_path: 确保不同文件的 ID 前缀不同。
        #    - idx: 确保同一文件内不同切片的 ID 后缀不同。
        ids = [f"{file_path}__{idx}" for idx in range(len(chunks))]

        # 5. Embedding + 写入 ChromaDB
        # model.encode() 是 SentenceTransformer 的核心方法，用于将文本列表转换为向量
        # 参数说明:
        #   - texts: 待编码的文本列表 (list[str])
        #   - normalize_embeddings=True: 对生成的向量进行 L2 归一化。
        #     归一化后，向量长度为 1。此时使用“点积”计算相似度等价于“余弦相似度”，
        #     且能加速后续 ChromaDB 的检索过程（因为配置了 cosine 空间）。
        # .tolist() 是将 NumPy 数组转换为 Python 原生列表 (list[list[float]])，
        # 因为 ChromaDB 的 add 方法通常期望接收原生的 Python 数据结构而非 NumPy 对象。
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        collection.add(
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )
        total_chunks += len(chunks)

    return f"全量索引完成：{len(md_files)} 个文件，{total_chunks} 个切片"


def index_file(file_path: str):
    """
    增量索引单个文件。如果该文件已存在于索引中，先删除旧切片再重新索引。

    参数:
        file_path: .md 文件的绝对路径
    """
    if not os.path.isfile(file_path):
        return f"文件不存在: {file_path}"
    if not file_path.endswith(".md"):
        return f"仅支持 .md 文件: {file_path}"

    model = _load_model()
    collection = _load_collection()

    # 1. 删除该文件的旧切片（如果存在）
    # ChromaDB 的 get 方法可以用 where 按 metadata 过滤
    old = collection.get(where={"source": file_path})
    if old["ids"]:
        collection.delete(ids=old["ids"])

    # 2. 切分 + Embedding + 写入
    chunks = _split_markdown(file_path)
    if not chunks:
        return f"文件无有效内容: {file_path}"

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{file_path}__{i}" for i in range(len(chunks))]

    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    collection.add(
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=ids,
    )

    return f"已索引: {file_path} → {len(chunks)} 个切片"


def list_sources() -> list[str]:
    """
    列出当前 ChromaDB 中所有已索引的源文件及其切片数量。
    
    返回:
        lines: list[str]，每行格式为 "  5 个切片 | 文件路径"
    """
    collection = _load_collection()
    data = collection.get(include=["metadatas"])
    if not data["metadatas"]:
        return ["(索引为空)"]

    # 统计每个源文件的切片数
    source_to_chunk_count = {}
    # data["metadatas"] 是一个列表，包含每个切片的元数据字典
    # meta 是当前遍历到的单个切片的元数据字典 (例如: {"source": "path/to/file.md", ...})
    # .get("source", "未知") 尝试从 meta 中获取 "source" 键的值；如果该键不存在，则返回默认值 "未知"
    # source_to_chunk_count.get(src, 0) 尝试从计数字典中获取当前文件 src 已有的切片数；如果该文件尚未记录，则返回默认值 0
    for meta in data["metadatas"]:
        src = meta.get("source", "未知")
        source_to_chunk_count[src] = source_to_chunk_count.get(src, 0) + 1

    # 按切片数降序排列（切片多的文件排前面）
    # .items() 将字典转换为 (键, 值) 元组列表，例如 [('file1.md', 5), ('file2.md', 3)]
    # key=lambda x: x[1] 指定排序依据为元组的第二个元素（即切片数量）
    # reverse=True 表示降序排列（数量多的排在前面）
    sorted_items = sorted(
        source_to_chunk_count.items(),
        key=lambda x: x[1],
        reverse=True
    )

    lines = []
    for src, count in sorted_items:
        lines.append(f"{count:3d} 个切片 | {src}")
    return lines
