# 笔记 RAG 配置文件

# 笔记源文件目录（你的 md 笔记所在位置）
NOTES_DIRS = [
    r"E:\学习相关\学习笔记\python基础\py基础-md",
    r"E:\学习相关\学习笔记\前后端-md",
]

# ChromaDB 持久化目录
CHROMA_DB_PATH = r"E:\学习相关\实战项目\笔记RAG\rag_db"

# Embedding 模型
EMBEDDING_MODEL = "BAAI/bge-m3"

# 检索默认参数
DEFAULT_TOP_K = 5

# 切分参数
CHUNK_SIZE = 500      # 每个切片最大字符数
CHUNK_OVERLAP = 80    # 切片间重叠字符数
