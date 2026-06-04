# 索引模块
# 负责：加载笔记 → 切分 → Embedding → 写入 ChromaDB
#
# 对外接口：
#   reindex_all()      全量重建索引
#   index_file(path)   增量索引单个文件
#   list_sources()     列出已索引的源文件
