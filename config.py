# ============================================================
# 笔记 RAG 配置文件
# ============================================================

# 笔记源文件目录（你的 md 笔记所在位置）
# 支持多个目录，程序会递归扫描所有 .md 文件
NOTES_DIRS = [
    r"E:\学习相关\学习笔记\python基础\py基础-md",
    r"E:\学习相关\学习笔记\前后端-md",
]

# ChromaDB 持久化目录（向量数据库存本地磁盘的位置）
CHROMA_DB_PATH = r"E:\学习相关\实战项目\笔记RAG\rag_db"

# Embedding 模型（把文本转成向量的模型）
# 选型说明：
#   - "BAAI/bge-m3"             2.2GB, 1024维, 多语言, 中英混合精度最高, 但体积大
#   - "BAAI/bge-small-zh-v1.5"  100MB, 512维,  中文优化, 轻量推荐（当前选用）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 检索默认参数
DEFAULT_TOP_K = 5  # 每次检索返回的最相似切片数
RERANK_TOP_N = 3    # Rerank 后最终返回的结果数（必须 <= DEFAULT_TOP_K）

# Rerank 模型（精排，提高检索准确率）
# 备选:
#   "BAAI/bge-reranker-v2-m3"  2.2GB, 1024维, 多语言, 精度最高
#   "BAAI/bge-reranker-small"     100MB, 384维,  中文优化, 轻量推荐（当前选用）
# RERANK_MODEL = ""  # 默认不启用，需要时在配置里填写模型名
RERANK_MODEL = "BAAI/bge-reranker-small"  # 取消注释以启用 Rerank

# 切分参数（控制每个切片的长度和重叠）
#  overlap
#   防止语义在边界处被截断。比如一句话的前半在切片 A，后半在切片 B，
#   两个切片单独检索时都可能匹配不上，overlap 让边界处的语义有重叠，提高召回率。
CHUNK_SIZE = 500      # 每个切片最大字符数（按中文字符算，约 250-500 个 token）
CHUNK_OVERLAP = 80    # 切片间重叠字符数（约 40-80 个 token）
