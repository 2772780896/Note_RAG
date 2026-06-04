# 检索模块
# 负责：加载 Embedding 模型 + ChromaDB → 语义检索
#
# 对外接口：
#   search(query, n_results=5) → list[dict]
#       返回格式: [{"content": "切片文本", "source": "文件路径", "similarity": 0.92}, ...]
