# RAG MCP Server 入口
# 基于 FastMCP，暴露以下 Tool 给 CodeBuddy：
#
#   search_notes(query, n_results=5)   语义检索笔记
#   index_file(file_path)              增量索引单个文件
#   reindex_all()                      全量重建索引
#   list_sources()                     列出已索引的源文件
#
# 启动方式（由 CodeBuddy 自动启动）：
#   python rag_server.py
