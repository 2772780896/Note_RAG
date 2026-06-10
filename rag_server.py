"""
RAG Server —— MCP 入口，通过 stdio 与 CodeBuddy 通信。

暴露 5 个 Tool:
    search_notes(query, n_results=5)
    index_file(path)
    reindex_all()
    list_sources()
    get_full_file_content(file_path)

CodeBuddy 配置写在 codebuddy_mcp_settings.json:
{
    "mcpServers": {
        "notes-rag": {
            "command": "E:\\学习相关\\实战项目\\笔记RAG\\venv\\Scripts\\python.exe",  # Python 解释器的绝对路径（指向 venv 里的）
            "args": ["rag_server.py"],  # 要运行的脚本文件名
            "cwd": "E:\\学习相关\\实战项目\\笔记RAG"  # 工作目录，即运行脚本时的当前目录
        }
    }
}
"""

import sys
import os

# 把当前目录加入 sys.path，确保 config / indexer / retriever 能被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from indexer import reindex_all as indexer_reindex_all
from indexer import index_file as indexer_index_file
from indexer import list_sources as indexer_list_sources
from retriever import search, get_full_file

# ------------------------------------------------------------
# 初始化 FastMCP 实例
# ------------------------------------------------------------
mcp = FastMCP("笔记 RAG Server")


# ============================================================
# Tool 1: 语义检索
# ============================================================
@mcp.tool()
def search_notes(query: str, n_results: int = 5) -> str:
    """
    在本地笔记库中执行语义检索，返回与查询最相关的笔记切片。

    :param query: 自然语言查询，例如 "Python 装饰器原理"
    :param n_results: 返回条数，默认 5
    :return: 格式化的检索结果文本
    """
    if not query or not query.strip():
        return "错误: 查询不能为空"

    try:
        results = search(query=query.strip(), n_results=n_results)
    except Exception as e:
        return f"检索失败: {e}"

    if not results:
        return f'未找到与 "{query}" 相关的笔记'

    lines = [f'搜索: "{query}"', f"返回 {len(results)} 条结果:", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"--- 结果 {i} (相似度: {r['similarity']}) ---")
        lines.append(f"来源: {r['source']}")
        # 如果有章节信息，也显示出来
        if r.get("header_path"):
            lines.append(f"章节: {r['header_path']}")
        lines.append(f"内容:\n{r['content']}")
        lines.append("")
    return "\n".join(lines)


# ============================================================
# Tool 2: 增量索引
# ============================================================
@mcp.tool()
def index_file(path: str) -> str:
    """
    增量索引单个 Markdown 笔记文件。如果该文件已存在旧索引，先删除旧数据再重新索引。

    :param path: .md 文件的绝对路径
    :return: 操作结果描述
    """
    if not path or not path.strip():
        return "错误: 文件路径不能为空"

    try:
        return indexer_index_file(file_path=path.strip())
    except Exception as e:
        return f"索引失败: {e}"


# ============================================================
# Tool 3: 全量重建
# ============================================================
@mcp.tool()
def reindex_all_tool() -> str:
    """
    清空现有索引，重新扫描笔记目录下的所有 .md 文件并全量入库。
    建议在首次部署或笔记目录有大量变动时执行。

    :return: 操作结果描述（文件数和切片数）
    """
    try:
        return indexer_reindex_all()
    except Exception as e:
        return f"全量重建失败: {e}"


# ============================================================
# Tool 4: 列表源文件
# ============================================================
@mcp.tool()
def list_sources_tool() -> str:
    """
    列出当前 ChromaDB 中所有已索引的源文件及其切片数量。
    当需要查找某个文件的完整路径时，先调用此工具获取文件列表。

    :return: 格式化的源文件列表
    """
    try:
        sources = indexer_list_sources()
    except Exception as e:
        return f"查询失败: {e}"

    if not sources:
        return "索引为空"

    lines = [f"已索引的源文件 ({len(sources)} 个):", ""]
    for s in sources:
        lines.append(s)
    return "\n".join(lines)


# ============================================================
# Tool 5: 获取完整文件内容
# ============================================================
@mcp.tool()
def get_full_file_content(file_path: str) -> str:
    """
    获取指定已索引文件的完整内容（所有切片按顺序拼接为全文）。
    
    适用场景:
        - 需要参考整个文件的完整内容（如最佳实践文档、接口规范、模板参考）
        - 语义检索（search_notes）返回的片段不够完整时，用此工具获取全文
    
    注意: 需先通过 list_sources_tool 获取文件路径，再传入此工具。

    :param file_path: .md 文件的绝对路径（从 list_sources_tool 获取）
    :return: 文件的完整内容
    """
    if not file_path or not file_path.strip():
        return "错误: 文件路径不能为空"

    try:
        results = get_full_file(file_path=file_path.strip())
    except Exception as e:
        return f"获取失败: {e}"

    if not results:
        return f"未找到已索引的文件: {file_path}，请先通过 index_file 索引该文件"

    r = results[0]
    header = f"文件: {r['source']}\n切片数: {r['chunk_count']}\n{'='*60}\n\n"
    return header + r["content"]


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    mcp.run(transport="stdio")
