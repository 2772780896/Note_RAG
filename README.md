# 笔记 RAG 系统

基于本地 Embedding + ChromaDB 的 Markdown 笔记语义检索系统，通过 MCP 协议与 AI IDE 集成，让 LLM 能直接查阅你的个人知识库。

## 功能特性

- **语义检索**：基于向量的笔记内容搜索，支持自然语言查询
- **两阶段切分**：Markdown 标题切分 + 长度控制切分，保留文档结构
- **YAML Front Matter 解析**：自动提取笔记的 title、tags 等元数据
- **增量索引**：单文件更新时只重新索引该文件，无需全量重建
- **文件监控**：监听笔记目录变动，自动触发增量索引
- **可选 Rerank**：Cross-Encoder 精排，提升检索准确率
- **MCP 集成**：通过 MCP 协议暴露为 AI IDE 的 Tool，LLM 按需调用
- **纯本地运行**：无需 API Key，Embedding 模型和向量数据库均在本地

## 架构

```
用户提问 → AI IDE (MCP Client)
                │
                │ stdio JSON-RPC
                ▼
         rag_server.py (MCP Server)
           │          │
           ▼          ▼
      indexer.py   retriever.py
           │          │
           ▼          ▼
        ChromaDB   SentenceTransformer
       (向量存储)    (Embedding 模型)
```

### 核心模块

| 文件 | 职责 |
|------|------|
| `config.py` | 全局配置：笔记目录、模型选择、切分参数 |
| `indexer.py` | 索引模块：加载笔记 → 切分 → 向量化 → 写入 ChromaDB |
| `retriever.py` | 检索模块：查询向量化 → 向量检索 → 可选 Rerank |
| `rag_server.py` | MCP Server：暴露 4 个 Tool 供 AI IDE 调用 |
| `watch.py` | 文件监控：监听笔记目录变动，自动增量索引 |
| `test_rag.py` | 测试脚本：验证完整流程 |

### 数据流

**索引阶段**（写入）：
```
.md 文件 → 解析 YAML Front Matter → 按标题切分 → 按长度二次切分
→ Embedding 向量化 (512维) → 写入 ChromaDB
```

**检索阶段**（读取）：
```
用户查询 → Embedding 向量化 → ChromaDB 向量检索 (Top-K)
→ [可选] Rerank 精排 (Top-N) → 返回结果
```

## 部署

### 1. 环境要求

- Python 3.10+
- 约 200MB 磁盘空间（模型权重 ~100MB + 向量数据库）

### 2. 安装

```bash
# 克隆项目
git clone <repo-url>
cd 笔记RAG

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置

编辑 `config.py`：

```python
# 笔记目录（支持多个，递归扫描 .md 文件）
NOTES_DIRS = [
    r"C:\Users\你的用户名\Notes",
]

# ChromaDB 存储路径
CHROMA_DB_PATH = r"C:\Users\你的用户名\笔记RAG\rag_db"

# Embedding 模型（首次运行自动下载）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 切分参数
CHUNK_SIZE = 500       # 每片最大字符数
CHUNK_OVERLAP = 80     # 相邻片重叠字符数
```

### 4. 验证

```bash
python test_rag.py
```

测试脚本会自动验证：依赖导入 → 目录检查 → 单文件索引 → 语义检索 → 列表源文件。

### 5. 全量索引

```bash
python -c "from indexer import reindex_all; print(reindex_all())"
```

### 6. 接入 AI IDE（MCP 配置）

在 IDE 的 MCP 配置中添加：

```json
{
    "mcpServers": {
        "notes-rag": {
            "command": "项目绝对路径\\venv\\Scripts\\python.exe",
            "args": ["rag_server.py"],
            "cwd": "项目绝对路径"
        }
    }
}
```

> **注意**：`command` 必须使用 Python 解释器的绝对路径，不能使用 `"python"` 命令名，否则会出现 `spawn python ENOENT` 错误。

配置完成后重启 IDE，在对话中自然提问即可触发检索。

### 7.（可选）文件监控

```bash
python watch.py
```

启动后持续监听笔记目录，`.md` 文件变动时自动增量索引。`Ctrl+C` 停止。

## MCP 通信流程

本系统通过 MCP (Model Context Protocol) 与 AI IDE 通信，基于 JSON-RPC 2.0 over stdio。

```
IDE (Client)                         rag_server.py (Server)
    │                                       │
    │  ① initialize                         │  握手
    │──────────────────────────────────────→│
    │←──────────────────────────────────────│
    │                                       │
    │  ② tools/list                         │  发现可用工具
    │──────────────────────────────────────→│
    │←──────────────────────────────────────│
    │                                       │
    │  ③ tools/call {search_notes, ...}     │  LLM 决定调用
    │──────────────────────────────────────→│
    │                                       │  → 向量检索
    │←──────────────────────────────────────│  ← 返回结果
    │                                       │
    │  LLM 基于检索结果生成最终回答           │
```

### 暴露的 Tool

| Tool | 功能 | 参数 |
|------|------|------|
| `search_notes` | 语义检索笔记 | `query: str`, `n_results: int=5` |
| `index_file` | 增量索引单文件 | `path: str` |
| `reindex_all_tool` | 全量重建索引 | 无 |
| `list_sources_tool` | 列出已索引文件 | 无 |

LLM 根据 Tool 的 `description` 和 `inputSchema`（从 Python 类型注解自动生成）判断何时调用。

## 技术栈

| 组件 | 用途 |
|------|------|
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | 本地 Embedding 模型 (BAAI/bge-small-zh-v1.5) |
| [ChromaDB](https://github.com/chroma-core/chroma) | 本地向量数据库 (HNSW 索引) |
| [LangChain Text Splitters](https://python.langchain.com/) | Markdown 标题切分 + 递归字符切分 |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | MCP Server 框架 |
| [watchdog](https://github.com/gorakhargosh/watchdog) | 文件系统监控 |

## 项目结构

```
笔记RAG/
├── config.py          # 全局配置
├── indexer.py         # 索引模块
├── retriever.py       # 检索模块
├── rag_server.py      # MCP Server
├── watch.py           # 文件监控
├── test_rag.py        # 测试脚本
├── requirements.txt   # 依赖清单
├── rag_db/            # ChromaDB 持久化存储（自动生成）
└── venv/              # 虚拟环境
```
