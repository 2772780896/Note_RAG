---
title: "笔记 RAG 系统 —— 完整构建流程与代码解析"
tags: [RAG, Python, ChromaDB, Embedding, MCP, LangChain]
created: 2026-06-08
---

## 整体架构

```
用户在 CodeBuddy 中提问
        │
        ▼
┌──────────────────────┐
│   CodeBuddy (AI IDE)  │  ← MCP Client
└──────────┬───────────┘
           │ stdio (JSON-RPC)
           ▼
┌──────────────────────┐
│   rag_server.py       │  ← MCP Server（入口）
│   暴露 4 个 Tool       │
└──┬───────┬───────┬───┘
   │       │       │
   ▼       ▼       ▼
┌──────┐┌──────┐┌──────────┐
│indexer││retriever││ watch.py │
│ .py  ││ .py   ││(文件监控) │
└──┬───┘└──┬───┘└────┬─────┘
   │       │          │
   ▼       ▼          ▼
┌──────────────────────────┐
│  ChromaDB (本地向量数据库) │
│  SentenceTransformer     │
│  (Embedding/Rerank 模型) │
└──────────────────────────┘
```

### 两条核心数据流

| 阶段 | 数据流向 | 触发时机 |
|------|---------|---------|
| **索引** | `.md` 文件 → 切分 → Embedding 向量化 → 写入 ChromaDB | 首次部署 / 文件变动 |
| **检索** | 用户查询 → 查询向量化 → ChromaDB 向量检索 → (可选 Rerank) → 返回结果 | 用户提问时 |

---

## 核心概念速查

| 概念 | 解释 |
|------|------|
| **Embedding** | 把文本转成固定维度的浮点数向量（如 512 维），语义相近的文本向量距离近 |
| **Bi-Encoder** | 查询和文档**分别**编码成向量，用向量距离衡量相似度。速度快但精度低 |
| **Cross-Encoder (Rerank)** | 查询和文档**拼接**后一起编码，直接输出相关性分数。速度慢但精度高 |
| **HNSW** | 分层可导航小世界图，一种近似最近邻搜索 (ANN) 索引算法，查询复杂度 O(log n) |
| **Chunk Overlap** | 相邻切片的重叠区域，防止语义在切片边界处被截断，提高召回率 |
| **MCP** | Model Context Protocol，标准化 AI 工具调用协议，基于 JSON-RPC over stdio |
| **Front Matter** | Markdown 文件顶部 `---` 包裹的 YAML 元数据（title、tags 等） |
| **ANN** | Approximate Nearest Neighbor，近似最近邻搜索，牺牲少量精度换取数量级的速度提升 |

---

## 配置文件 config.py

全局参数集中管理，所有模块从这里读取配置。

```python
# 笔记源文件目录（支持多个，程序会递归扫描所有 .md 文件）
NOTES_DIRS = [
    r"E:\学习相关\学习笔记\python基础\py基础-md",  # r"" 原始字符串，防止 \ 被转义
    r"E:\学习相关\学习笔记\前后端-md",
]

# ChromaDB 持久化目录（向量数据库存本地磁盘的位置）
CHROMA_DB_PATH = r"E:\学习相关\实战项目\笔记RAG\rag_db"

# Embedding 模型（把文本转成向量的模型）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# 选型: 100MB, 512维, 中文优化, 轻量推荐

# 检索参数
DEFAULT_TOP_K = 5   # 向量检索返回的最相似切片数
RERANK_TOP_N = 3    # Rerank 精排后最终返回的结果数（必须 <= DEFAULT_TOP_K）

# Rerank 模型（可选，空字符串表示不启用）
RERANK_MODEL = ""
# RERANK_MODEL = "BAAI/bge-reranker-small"  # 取消注释以启用

# 切分参数
CHUNK_SIZE = 500     # 每个切片最大字符数
CHUNK_OVERLAP = 80   # 切片间重叠字符数（防止语义截断）
```

---

## 索引流程 indexer.py

### 模块职责

加载笔记 → 解析元数据 → 两阶段切分 → Embedding 向量化 → 写入 ChromaDB

### 导入的关键库

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import chromadb
```

| 库 | 用途 |
|----|------|
| `MarkdownHeaderTextSplitter` | 按 Markdown 标题（## / ###）切分文档 |
| `RecursiveCharacterTextSplitter` | 按长度递归切分，优先在段落/句子边界切 |
| `Document` | LangChain 的文档对象，包含 `page_content`（文本）和 `metadata`（元数据字典） |
| `SentenceTransformer` | 本地 Embedding 模型，把文本转成向量 |
| `chromadb` | 本地向量数据库，支持 HNSW 索引和持久化 |

### 辅助函数详解

#### _load_model()

```python
def _load_model():
    return SentenceTransformer(EMBEDDING_MODEL)
```

- **功能**: 创建 SentenceTransformer 模型实例
- **参数**: 模型名（如 `"BAAI/bge-small-zh-v1.5"`），首次运行自动从 HuggingFace 下载
- **返回**: `SentenceTransformer` 实例

#### _load_collection()

```python
def _load_collection():
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(
        name="notes",
        metadata={"hnsw:space": "cosine"},
    )
```

| 调用 | 说明 |
|------|------|
| `chromadb.PersistentClient(path=...)` | 创建持久化客户端，数据写入本地 SQLite 文件 |
| `client.get_or_create_collection(name, metadata)` | 获取或创建集合，已存在则直接返回 |
| `"hnsw:space": "cosine"` | 使用余弦相似度作为距离度量（对归一化向量，余弦相似度 = 内积） |
| **返回** | `Collection` 对象，提供 `.add()` / `.query()` / `.get()` / `.delete()` 方法 |

#### _parse_front_matter(text) —— YAML 元数据解析

```python
def _parse_front_matter(text: str) -> tuple[dict, str]:
    yaml_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
    match = yaml_pattern.match(text)
    if not match:
        return {}, text
    yaml_text = match.group(1)
    metadata = yaml.safe_load(yaml_text) or {}
    cleaned_text = yaml_pattern.sub('', text)
    return metadata, cleaned_text
```

- **正则解析**: `^---\s*\n(.*?)\n---\s*\n` 匹配文件开头的 `---...---` 块
- **`re.DOTALL`**: 让 `.` 也匹配换行符（默认不匹配）
- **`yaml.safe_load()`**: 安全解析 YAML 文本为 Python 字典
- **返回值**: `tuple[dict, str]`
  - `metadata`: 如 `{"title": "Python 装饰器", "tags": ["Python", "高级"]}`
  - `cleaned_text`: 去掉 YAML 后的正文

#### _split_markdown(file_path) —— 两阶段切分策略（核心）

```python
def _split_markdown(file_path: str) -> list[dict]:
```

**第 0 步：解析 YAML Front Matter**

```python
front_matter, cleaned_text = _parse_front_matter(text)
# front_matter 示例: {"title": "Django 教程", "tags": ["Python", "Web"]}
```

**第 1 步：按 Markdown 标题切分**

```python
headers_to_split_on = [("##", "h2"), ("###", "h3")]
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,  # 保留标题文本在切片里
)
doc_slices = md_splitter.split_text(cleaned_text)  # 返回 List[Document]
```

- **输入**:  cleaned_text（去掉 YAML 的正文，str）
- **返回**: `List[Document]`，每个 Document 包含:
  - `page_content`: 该标题下的文本内容（str）
  - `metadata`: 如 `{"h2": "Django 基础", "h3": "Model 字段"}`

**第 2 步：按长度二次切分**

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,       # 500 字符
    chunk_overlap=CHUNK_OVERLAP, # 80 字符
    separators=["\n\n", "\n", "。", "；", " ", ""],  # 优先级递减的分隔符
)
sub_chunks = text_splitter.split_text(doc.page_content)  # 返回 List[str]
```

- **输入**: 单段文本（str）
- **返回**: `List[str]`，按长度切分后的文本列表
- **separators 含义**: 优先在段落（`\n\n`）处切，其次是换行（`\n`）、句号（`。`）、分号（`；`）等
- **chunk_overlap**: 相邻切片有 80 字符重叠，防止语义截断

**元数据组装与最终返回格式**:

```python
# 返回: list[dict]，每个元素格式:
{
    "content": "切片文本内容...",
    "metadata": {
        "source": "E:\\笔记\\1.Flask.md",       # 源文件绝对路径
        "header_path": "Flask > 基本结构",        # 层级标题路径
        "fm_title": "Flask 教程",                 # 来自 YAML Front Matter（fm_ 前缀）
        "fm_tags": ["Python", "Web"],             # 来自 YAML Front Matter
    }
}
```

### 公开接口

#### reindex_all() —— 全量重建索引

```python
def reindex_all() -> str:
```

流程:
1. 清空 ChromaDB 中所有旧数据
2. 扫描 `NOTES_DIRS` 下所有 `.md` 文件
3. 逐文件切分 → Embedding → 写入

关键代码:

```python
# 清空旧数据
existing = collection.get()                    # 返回: {"ids": [...], "documents": [...], ...}
collection.delete(ids=existing["ids"])         # 按 id 列表批量删除

# Embedding 向量化
embeddings = model.encode(texts, normalize_embeddings=True).tolist()
# 输入: texts (List[str])，文本列表
# 输出: numpy.ndarray，形状 (n, 512)，归一化后的向量矩阵
# .tolist(): 转为 Python List[List[float]]（ChromaDB 要求）

# 批量写入 ChromaDB
ids = [f"{file_path}__{i}" for i in range(len(chunks))]
# id 格式: "E:\\笔记\\1.Flask.md__0"，文件路径+序号，保证唯一

collection.add(
    documents=texts,        # List[str]    原文列表
    metadatas=metadatas,    # List[dict]   元数据列表
    embeddings=embeddings,  # List[List[float]]  向量列表
    ids=ids,                # List[str]    唯一 ID 列表
)
```

- **返回值**: `str`，如 `"全量索引完成：15 个文件，128 个切片"`

#### index_file(file_path) —— 增量索引单文件

```python
def index_file(file_path: str) -> str:
```

流程:
1. 删除该文件的旧切片
2. 重新切分 → Embedding → 写入

```python
# 按 metadata 过滤删除旧数据
old = collection.get(where={"source": file_path})
# where 参数: ChromaDB 的 metadata 过滤条件，只返回 source 等于该路径的记录
if old["ids"]:
    collection.delete(ids=old["ids"])
```

- **返回值**: `str`，如 `"已索引: E:\\笔记\\1.Flask.md → 6 个切片"`

#### list_sources() —— 列出已索引源文件

```python
def list_sources() -> list[str]:
```

- **返回值**: `List[str]`，每行格式如 `"  6 个切片 | E:\\笔记\\1.Flask.md"`

---

## 检索流程 retriever.py

### 模块职责

加载 Embedding 模型 + 连接 ChromaDB，提供语义检索（支持可选 Rerank 精排）

### 模块级单例（懒加载）

```python
_embedding_model = None   # SentenceTransformer 实例
_chroma_client = None     # ChromaDB 客户端
_collection = None        # ChromaDB Collection
_reranker_model = None    # CrossEncoder Rerank 模型（可选）
```

**为什么用模块级变量做单例？** Python 的 `import` 机制保证模块只被加载一次，所以模块级变量天然是单例，不需要写 `__new__` 那种复杂的单例模式。

懒加载示例:

```python
def _get_embedding_model():
    global _embedding_model          # 声明使用全局变量
    if _embedding_model is None:     # 首次调用才加载
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model          # 后续调用直接返回已有实例
```

### search(query, n_results) —— 核心检索函数

```python
def search(query: str, n_results: int = DEFAULT_TOP_K) -> list[dict]:
```

#### 第 1 步：查询向量化

```python
query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
# 输入: ["Python装饰器"]（列表，即使只有一个查询）
# 输出: numpy.ndarray，形状 (1, 512)
# [0]: 取第一个向量，形状变为 (512,)
# .tolist(): 转为 List[float]
# 最终格式: [0.023, -0.156, 0.342, ...]  （512 个浮点数）
```

#### 第 2 步：ChromaDB 向量检索

```python
results_raw = collection.query(
    query_embeddings=[query_embedding],   # List[List[float]]，嵌套列表
    n_results=n_results,                   # 返回 Top-K（默认 5）
    include=["documents", "metadatas", "distances"],
)
```

**ChromaDB query 返回格式**:

```python
{
    "documents":  [["切片文本1", "切片文本2", ...]],   # List[List[str]]
    "metadatas":  [{"source": "...", ...}, {...}],    # List[List[dict]]  (注意这里少了外层列表)
    "distances":  [[0.503, 0.520, 0.551, ...]],       # List[List[float]] 余弦距离
    "ids":        [["id1", "id2", ...]],              # List[List[str]]
}
# 注意: 所有字段都是嵌套列表（外层是查询数，这里只有 1 个查询）
```

#### 第 3 步：解析结果

```python
matched_documents = results_raw["documents"][0]   # List[str]  取第一个查询的结果
matched_metadatas = results_raw["metadatas"][0]    # List[dict]
matched_distances = results_raw["distances"][0]    # List[float]

# 余弦距离 → 相似度
similarity = 1 - dist
# ChromaDB 的 cosine distance = 1 - cosine_similarity
# 所以 similarity = 1 - distance，范围 [0, 1]，越大越相似
```

#### 第 4 步（可选）：Rerank 精排

```python
if reranker is not None:
    # 构造 (query, document) 对
    rerank_pairs = [(query, r["content"]) for r in base_results]
    # 格式: [("Python装饰器", "切片文本1"), ("Python装饰器", "切片文本2"), ...]

    # CrossEncoder 打分
    rerank_scores = reranker.predict(rerank_pairs)
    # 输入: List[Tuple[str, str]]
    # 输出: numpy.ndarray，形状 (n,)，每个值是该对的相关性分数

    # 按 Rerank 分数重新排序，只保留 Top-N
    base_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    base_results = base_results[:RERANK_TOP_N]
```

**为什么向量检索后还要 Rerank？**
- 向量检索是 Bi-Encoder（双编码器）：查询和文档分别编码，速度快但精度低
- Rerank 是 Cross-Encoder（交叉编码器）：查询和文档拼接后一起编码，精度高但速度慢
- 工业级做法：先用向量检索取 Top-20（快筛），再用 Rerank 精排取 Top-5（精选）

#### 第 5 步：拼接上下文前缀

```python
context_prefix = f"[来源: {file_name} | 章节: {r['header_path']}]\n\n"
# 示例: "[来源: 1.Flask.md | 章节: Jinja2 模板语法 > 条件]\n\n"
```

#### 最终返回格式

```python
# 返回: List[dict]，每个元素:
{
    "content": "[来源: 1.Flask.md | 章节: 基本结构]\n\n## 基本结构\n...",
    "source": "E:\\笔记\\1.Flask.md",          # 源文件路径
    "header_path": "基本结构",                   # 章节路径
    "similarity": 0.4968,                        # 余弦相似度 (float, 0~1)
}
```

---

## MCP 通信层 rag_server.py

### 模块职责

作为 MCP Server，通过 stdio 与 CodeBuddy（MCP Client）通信，暴露 4 个 Tool。

### MCP 初始化

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("笔记 RAG Server")
```

- `FastMCP`: MCP Python SDK 的高层封装，自动处理 JSON-RPC 协议
- 参数 `"笔记 RAG Server"` 是 Server 名称，会出现在 Client 的工具列表中

### Tool 注册机制

```python
@mcp.tool()  # 装饰器语法：函数定义时自动注册为 MCP Tool
def search_notes(query: str, n_results: int = 5) -> str:
    """docstring 会被作为 Tool 的描述，LLM 据此决定何时调用"""
```

- **`@mcp.tool()`**: 装饰器，把函数注册到 MCP Server 的 Tool 列表
- **类型注解** `query: str`、`n_results: int = 5`: MCP 自动根据注解生成 JSON Schema
- **docstring**: 被提取为 Tool 描述，供 LLM 理解该工具的用途

### 4 个 Tool 一览

| Tool 函数 | 功能 | 参数 | 调用链 |
|-----------|------|------|--------|
| `search_notes(query, n_results)` | 语义检索笔记 | query: str, n_results: int | → `retriever.search()` |
| `index_file(path)` | 增量索引单文件 | path: str | → `indexer.index_file()` |
| `reindex_all_tool()` | 全量重建索引 | 无 | → `indexer.reindex_all()` |
| `list_sources_tool()` | 列出已索引源文件 | 无 | → `indexer.list_sources()` |

### stdio 通信流程

```python
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

`transport="stdio"` 使用标准输入/输出通信：

```
CodeBuddy (Client)                   rag_server.py (Server)
     │                                       │
     │── 启动子进程 python rag_server.py ──→  │
     │                                       │ (初始化 FastMCP，监听 stdin)
     │                                       │
     │──stdin──→  JSON-RPC 请求  ──────────→ │
     │           {"jsonrpc":"2.0",            │
     │            "method":"tools/list"}      │
     │                                       │
     │←──stdout── JSON-RPC 响应 ──────────── │
     │           {"result":{"tools":[...]}}   │
     │                                       │
     │──stdin──→  调用 Tool  ──────────────→ │
     │           {"method":"tools/call",      │
     │            "params":{                  │
     │              "name":"search_notes",    │
     │              "arguments":{             │
     │                "query":"装饰器原理"     │
     │              }                         │
     │            }}                          │
     │                                       │ → retriever.search("装饰器原理")
     │                                       │ → ChromaDB 向量检索
     │                                       │
     │←──stdout── 返回结果 ──────────────── │
     │           {"result":{                  │
     │             "content":[{               │
     │               "type":"text",           │
     │               "text":"搜索结果..."     │
     │             }]                         │
     │           }}                           │
```

---

## 文件监控 watch.py

### 模块职责

使用 watchdog 库监听笔记目录，当 `.md` 文件变动时自动调用 `index_file()` 增量索引。

### 为什么需要文件监控？

RAG 系统的索引和原文是"离线"的——原文更新后，向量库不会自动同步。如果不做监控，用户改了笔记，RAG 检索出来的还是旧内容，这叫**索引漂移**。

### 事件处理器

```python
class NotesFileHandler(FileSystemEventHandler):
```

继承 `watchdog.events.FileSystemEventHandler`，重写三个方法:

| 方法 | 触发时机 | 处理逻辑 |
|------|---------|---------|
| `on_modified(event)` | 文件被修改 | 重新索引该文件 |
| `on_created(event)` | 新文件被创建 | 索引新文件 |
| `on_moved(event)` | 文件被移动/重命名 | 索引新路径（旧索引会被覆盖） |

### 防抖机制

```python
self._debounce_seconds = 2  # 2 秒内同一文件只处理一次

def _should_process(self, file_path: str) -> bool:
    now = time.time()                                    # 当前时间戳（浮点数秒）
    last_time = self._last_processed.get(file_path, 0)   # 上次处理时间，默认 0
    if now - last_time < self._debounce_seconds:
        return False   # 间隔不足 2 秒，跳过
    self._last_processed[file_path] = now   # 更新为当前时间
    return True
```

- **为什么需要防抖？** 编辑器保存文件时可能在毫秒内触发多次 `on_modified` 事件
- **实现原理**: 用字典 `{文件路径: 上次处理时间戳}` 记录，间隔不足 2 秒则跳过

### 启动监控

```python
observer = Observer()                                           # 创建观察者
observer.schedule(event_handler, path=notes_dir, recursive=True) # 注册监控目录
observer.start()                                                # 启动后台线程
while True:
    time.sleep(1)   # 主线程阻塞，等待 Ctrl+C
```

- `Observer` 内部使用操作系统原生 API（Windows: `ReadDirectoryChangesW`）
- `recursive=True`: 递归监控所有子目录

---

## 关键数据格式汇总

### ChromaDB Collection.add() 参数格式

```python
collection.add(
    documents=["文本1", "文本2"],              # List[str]
    metadatas=[{"source": "路径", ...}, {}],   # List[dict]
    embeddings=[[0.1, 0.2, ...], [0.3, ...]], # List[List[float]]
    ids=["file__0", "file__1"],               # List[str]  唯一 ID
)
```

### ChromaDB Collection.query() 返回格式

```python
{
    "ids":       [["id1", "id2", "id3"]],           # List[List[str]]
    "documents": [["文本1", "文本2", "文本3"]],      # List[List[str]]
    "metadatas": [[{"source": "..."}, {...}, ...]],  # List[List[dict]]
    "distances": [[0.503, 0.520, 0.551]],           # List[List[float]] 余弦距离
}
# 外层列表对应查询数（支持批量查询），这里只有 1 个查询
```

### ChromaDB Collection.get() 返回格式

```python
{
    "ids": ["id1", "id2", ...],                    # List[str]
    "documents": ["文本1", "文本2", ...],           # List[str]（需 include=["documents"]）
    "metadatas": [{"source": "..."}, ...],          # List[dict]（需 include=["metadatas"]）
}
# 注意: get() 返回的是扁平列表（不像 query() 是嵌套列表）
```

### SentenceTransformer.encode() 输入输出

```python
# 输入
model.encode(
    ["文本1", "文本2"],           # List[str] 或 str
    normalize_embeddings=True     # L2 归一化，使向量模长为 1
)
# 输出: numpy.ndarray，形状 (n, dim)
#   n = 输入文本数
#   dim = 模型维度（bge-small-zh-v1.5 为 512）
# 示例: array([[ 0.023, -0.156,  0.342, ...],   # 文本1 的 512 维向量
#              [-0.087,  0.201, -0.113, ...]])  # 文本2 的 512 维向量
```

### CrossEncoder.predict() 输入输出（Rerank 模型）

```python
# 输入
reranker.predict([
    ("查询文本", "文档文本1"),    # List[Tuple[str, str]]
    ("查询文本", "文档文本2"),
])
# 输出: numpy.ndarray，形状 (n,)
# 示例: array([0.92, 0.45, 0.78])  # 每对的相关性分数，越大越相关
```

### search() 最终返回格式

```python
# List[dict]
[
    {
        "content": "[来源: 1.Flask.md | 章节: 基本结构]\n\n## 基本结构\nfrom flask import Flask...",
        "source": "E:\\笔记\\1.Flask.md",
        "header_path": "基本结构",
        "similarity": 0.4968,        # float, 0~1
    },
    {
        "content": "...",
        "source": "...",
        "header_path": "...",
        "similarity": 0.4800,
    },
]
```

---

## Embedding 模型 vs Rerank 模型对比

| | Embedding 模型 (Bi-Encoder) | Rerank 模型 (Cross-Encoder) |
|--|---|---|
| **输入方式** | query 和 document **分别**编码 | query 和 document **拼接**后一起编码 |
| **输出** | 固定维度向量 (如 512 维) | 一个相关性分数 (标量) |
| **比对方式** | 计算两个向量的余弦相似度 | 直接输出 0~1 的分数 |
| **速度** | 快（向量可预计算、缓存） | 慢（每次都要重新拼接计算） |
| **精度** | 较低（独立编码，丢失交互信息） | 较高（联合编码，捕捉细粒度匹配） |
| **用途** | 索引阶段 + 检索阶段的粗筛 | 检索阶段的精排（可选） |
| **本项目的模型** | `BAAI/bge-small-zh-v1.5` (100MB) | `BAAI/bge-reranker-small` (100MB, 可选) |

```
【Bi-Encoder 工作流程】
query: "Python装饰器"  → [0.1, 0.3, 0.5, ...]  ──┐
                                                    ├→ 余弦相似度 = 0.48
doc:   "装饰器原理..."  → [0.2, 0.4, 0.3, ...]  ──┘

【Cross-Encoder 工作流程】
"[CLS] Python装饰器 [SEP] 装饰器原理... [SEP]"  → 模型整体理解 → 分数 0.92
```

它们是同一系列（BGE）下针对**不同任务**训练的两个模型，参数不共享，不能互相替代。

---

## MCP 通信协议详解

### 什么是 MCP

MCP（Model Context Protocol）是一个标准化的 AI 工具调用协议，基于 **JSON-RPC 2.0 over stdio**。它让 LLM（大语言模型）能够调用外部工具，就像给 AI 装了"手"。

### JSON-RPC 2.0 消息格式

每条消息都是标准 JSON，分为三种类型：

| 类型 | 特征 | 是否需要响应 |
|------|------|-------------|
| **请求 (Request)** | 有 `id` + `method` | ✅ 需要 |
| **通知 (Notification)** | 有 `method`，**无 `id`** | ❌ 不需要 |
| **响应 (Response)** | 有 `id` + `result`/`error` | — |

```json
// 请求：需要响应
{"jsonrpc":"2.0", "id":1, "method":"tools/call", "params":{...}}

// 通知：不需要响应（单向告知）
{"jsonrpc":"2.0", "method":"notifications/initialized"}

// 响应：对应某个请求的 id
{"jsonrpc":"2.0", "id":1, "result":{...}}
```

- **`id`**：请求的唯一标识，响应中携带相同 id 用于匹配
- **`method`**：要调用的方法名
- **`params`**：方法参数

### 完整通信时序

#### 阶段一：握手（初始化）

```
Client (IDE)                              Server (rag_server.py)
    │                                          │
    │  ① initialize                            │
    │─────────────────────────────────────────→ │
    │ {"jsonrpc":"2.0","id":1,                 │
    │  "method":"initialize",                  │
    │  "params":{                              │
    │    "protocolVersion":"2024-11-05",       │
    │    "clientInfo":{"name":"Qoder",...}     │
    │  }}                                      │
    │                                          │
    │  ② 响应                                   │
    │ ←──────────────────────────────────────── │
    │ {"jsonrpc":"2.0","id":1,                 │
    │  "result":{                              │
    │    "protocolVersion":"2024-11-05",       │
    │    "serverInfo":{                        │
    │      "name":"笔记 RAG Server",            │
    │      "version":"1.27.2"                  │
    │    },                                    │
    │    "capabilities":{                      │
    │      "tools":{"listChanged":false},      │
    │      "prompts":{"listChanged":false},    │
    │      "resources":{...}                   │
    │    }                                     │
    │  }}                                      │
    │                                          │
    │  ③ initialized 通知（无 id，不需要响应）    │
    │─────────────────────────────────────────→ │
    │ {"jsonrpc":"2.0",                        │
    │  "method":"notifications/initialized"}   │
```

#### 阶段二：发现工具

```
    │  ④ tools/list                            │
    │─────────────────────────────────────────→ │
    │ {"jsonrpc":"2.0","id":2,                 │
    │  "method":"tools/list"}                  │
    │                                          │
    │  ⑤ 返回所有 Tool 的定义                    │
    │ ←──────────────────────────────────────── │
    │ {"jsonrpc":"2.0","id":2,                 │
    │  "result":{"tools":[                     │
    │    {                                     │
    │      "name":"search_notes",              │
    │      "description":"在本地笔记库中...",    │
    │      "inputSchema":{                     │
    │        "type":"object",                  │
    │        "properties":{                    │
    │          "query":{"type":"string",       │
    │            "description":"自然语言查询"},  │
    │          "n_results":{"type":"integer",  │
    │            "default":5}                  │
    │        },                                │
    │        "required":["query"]              │
    │      }                                   │
    │    },                                    │
    │    {"name":"index_file",...},            │
    │    {"name":"reindex_all_tool",...},      │
    │    {"name":"list_sources_tool",...}      │
    │  ]}}                                     │
```

**`inputSchema` 是从哪来的？** 就是 Python 代码中的类型注解 + docstring：

```python
@mcp.tool()
def search_notes(query: str, n_results: int = 5) -> str:
    """在本地笔记库中执行语义检索..."""
```

| Python 代码 | MCP JSON Schema |
|-------------|------------------|
| `query: str` | `"query": {"type": "string"}` |
| `n_results: int = 5` | `"n_results": {"type": "integer", "default": 5}` |
| docstring 内容 | `"description": "..."` |
| 无默认值的参数 | 自动加入 `"required"` 列表 |

#### 阶段三：调用工具

```
    │  ⑥ tools/call                            │
    │─────────────────────────────────────────→ │
    │ {"jsonrpc":"2.0","id":3,                 │
    │  "method":"tools/call",                  │
    │  "params":{                              │
    │    "name":"search_notes",                │
    │    "arguments":{                         │
    │      "query":"Python 装饰器原理",         │
    │      "n_results":3                       │
    │    }                                     │
    │  }}                                      │
    │                                          │ → search_notes() 执行
    │                                          │ → retriever.search()
    │                                          │ → ChromaDB 向量检索
    │                                          │
    │  ⑦ 返回结果                               │
    │ ←──────────────────────────────────────── │
    │ {"jsonrpc":"2.0","id":3,                 │
    │  "result":{                              │
    │    "content":[{                          │
    │      "type":"text",                      │
    │      "text":"搜索: \"Python 装饰器原理\"\n│
    │        返回 3 条结果:\n..."               │
    │    }],                                   │
    │    "isError":false                       │
    │  }}                                      │
```

---

## IDE 如何决定调用 MCP 工具

### 核心机制：LLM 自主决策

```
用户输入: "帮我查一下 Python 装饰器的笔记"
         │
         ▼
┌─────────────────────────────────────┐
│  LLM 收到的信息包含:                  │
│                                     │
│  1. 系统提示词 (System Prompt)       │
│  2. 用户消息                         │
│  3. 可用工具列表 ← tools/list 返回   │
│     的 name + description +          │
│     inputSchema                     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  LLM 推理过程:                       │
│                                     │
│  "用户问的是 Python 装饰器"           │
│  "我有一个 search_notes 工具"        │
│  "description 说它可以语义检索笔记"    │
│  → 我应该调用这个工具！              │
└──────────┬──────────────────────────┘
           │
           ▼
  LLM 返回 tool_call（不是普通文本）:
  {
    "name": "search_notes",
    "arguments": {"query": "Python 装饰器原理"}
  }
           │
           ▼
  IDE 收到 tool_call → 通过 MCP 发送 tools/call → 拿到结果
           │
           ▼
  IDE 把结果塞回对话 → LLM 基于检索结果生成最终回答
```

### 三种方案的权衡

| 方案 | 流程 | 优缺点 |
|------|------|--------|
| **不用 RAG** | 用户问 → LLM 直接答 | ❌ LLM 没看过你的笔记，答非所问 |
| **每次自动检索** | 用户问 → 无脑检索 → 拼到上下文 → LLM 答 | ❌ 不是每个问题都需要检索，浪费检索开销 |
| **LLM 决定（当前方案）** | 用户问 → LLM 判断需要检索 → 检索 → LLM 答 | ✅ 只在需要时才检索，多一次 LLM 调用但值得 |

**关键理解**：LLM 充当"智能路由器"——
- "今天天气怎么样" → 不调用工具，直接回答
- "帮我查 Flask 路由的笔记" → 调用 search_notes
- "重新索引所有笔记" → 调用 reindex_all_tool

多出的那次 LLM 调用是"思考成本"，通常只输出几百 token 的 JSON（tool_call），开销很小。

### 如何让 LLM 更准确地调用工具

**description 写得越清晰，LLM 调用得越准确。** 这就是代码中每个 Tool 都有详细 docstring 的原因。

如果需要强制 LLM 每次都检索，可以在系统提示词中写：
```
在回答问题之前，你必须先使用 search_notes 工具搜索用户问题中的关键词，
将检索结果作为参考来回答。如果检索无结果，再用自己的知识回答。
```

---

## MCP 配置与使用指南

### 配置文件位置

| 客户端 | 配置文件位置 |
|--------|-------------|
| **Qoder (CodeBuddy)** | 项目根目录 `.codebuddy/mcp_settings.json` |
| **Claude Desktop** | `%APPDATA%\Claude\claude_desktop_config.json` |
| **Cursor** | Settings → MCP → 添加 |

### 配置模板

```json
{
    "mcpServers": {
        "notes-rag": {
            "command": "E:\\学习相关\\实战项目\\笔记RAG\\venv\\Scripts\\python.exe",
            "args": ["rag_server.py"],
            "cwd": "E:\\学习相关\\实战项目\\笔记RAG"
        }
    }
}
```

| 字段 | 说明 | 注意事项 |
|------|------|----------|
| `command` | Python 解释器的**绝对路径** | 必须指向 venv 里的 python.exe，不能写 `"python"` |
| `args` | 脚本文件名 | `["rag_server.py"]` |
| `cwd` | 工作目录 | 脚本所在目录的绝对路径 |

### 为什么 command 不能用 "python"

MCP Client 启动子进程时**不会激活 venv**，直接用系统 PATH 查找可执行文件。如果系统 PATH 中没有 python，就会报：

```
spawn python ENOENT
```

`ENOENT` = Error NO ENTry，即"找不到这个可执行文件"。使用 venv 的绝对路径可以避免这个问题。

### 使用流程

1. **首次部署**：配置 MCP → 重启 IDE → 全量索引笔记
2. **日常使用**：在对话中自然提问，LLM 自动调用工具
3. **笔记更新**：
   - 方式 A：运行 `watch.py` 自动监控，文件变动自动增量索引
   - 方式 B：让 LLM 调用 `index_file` 工具手动索引指定文件
   - 方式 C：让 LLM 调用 `reindex_all_tool` 全量重建

### 手动验证 Server 是否正常

在终端运行：
```powershell
.\venv\Scripts\python.exe rag_server.py
```

Server 启动后会阻塞等待 stdin（看起来像"卡住了"，这是正常的）。粘贴以下 JSON 测试：
```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
```

如果返回包含 `serverInfo` 的 JSON 响应，说明 Server 工作正常。按 Ctrl+C 退出。

### rag_server.py 中的 4 个 Tool 速查

| Tool | 功能 | 参数 | 使用场景 |
|------|------|------|----------|
| `search_notes` | 语义检索笔记 | query: str, n_results: int=5 | 查找笔记内容 |
| `index_file` | 增量索引单文件 | path: str（.md 绝对路径） | 单个文件更新后重新索引 |
| `reindex_all_tool` | 全量重建索引 | 无 | 首次部署 / 大量文件变动 |
| `list_sources_tool` | 列出已索引文件 | 无 | 查看索引状态 |
