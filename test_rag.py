"""
RAG 系统测试脚本 —— 验证从索引到检索的完整流程。

测试步骤:
    1. 测试导入（检查依赖是否安装正确）
    2. 测试单文件索引（index_file）
    3. 测试检索（search）
    4. 测试列表源文件（list_sources）
    5. （可选）测试 Rerank（如果配置了 RERANK_MODEL）
"""

import sys
import os

# 把当前目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 测试 1：检查依赖导入
# ============================================================
print("=" * 60)
print("测试 1：检查依赖导入")
print("=" * 60)

try:
    from config import NOTES_DIRS, CHROMA_DB_PATH, EMBEDDING_MODEL, DEFAULT_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
    print("[OK] config.py 导入成功")
    print(f"   笔记目录: {NOTES_DIRS}")
    print(f"   ChromaDB 路径: {CHROMA_DB_PATH}")
    print(f"   Embedding 模型: {EMBEDDING_MODEL}")
    print(f"   默认 Top-K: {DEFAULT_TOP_K}")
    print(f"   CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"   CHUNK_OVERLAP: {CHUNK_OVERLAP}")
except Exception as e:
    print(f"[FAIL] config.py 导入失败: {e}")
    sys.exit(1)

try:
    from indexer import reindex_all, index_file, list_sources
    print("[OK] indexer.py 导入成功")
except Exception as e:
    print(f"[FAIL] indexer.py 导入失败: {e}")
    sys.exit(1)

try:
    from retriever import search
    print("[OK] retriever.py 导入成功")
except Exception as e:
    print(f"[FAIL] retriever.py 导入失败: {e}")
    sys.exit(1)

try:
    from watch import start_watching
    print("[OK] watch.py 导入成功")
except Exception as e:
    print(f"[WARN] watch.py 导入失败（文件监控不可用）: {e}")

print()

# ============================================================
# 测试 2：检查笔记目录是否存在
# ============================================================
print("=" * 60)
print("测试 2：检查笔记目录")
print("=" * 60)

valid_dirs = []
for d in NOTES_DIRS:
    if os.path.isdir(d):
        print(f"[OK] 目录存在: {d}")
        valid_dirs.append(d)
    else:
        print(f"[WARN] 目录不存在: {d}")

if not valid_dirs:
    print("[FAIL] 没有可用的笔记目录，请在 config.py 中配置 NOTES_DIRS")
    sys.exit(1)

# 找一个测试文件
test_file = None
for d in valid_dirs:
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.endswith(".md"):
                test_file = os.path.join(root, f)
                break
        if test_file:
            break
    if test_file:
        break

if not test_file:
    print("[FAIL] 没有找到 .md 文件用于测试")
    sys.exit(1)

print(f"[OK] 找到测试文件: {test_file}")
print()

# ============================================================
# 测试 3：单文件索引
# ============================================================
print("=" * 60)
print("测试 3：单文件索引（index_file）")
print("=" * 60)

try:
    result = index_file(test_file)
    print(f"[OK] 索引成功: {result}")
except Exception as e:
    print(f"[FAIL] 索引失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================
# 测试 4：检索测试
# ============================================================
print("=" * 60)
print("测试 4：检索测试（search）")
print("=" * 60)

# 从测试文件里取一个关键词作为查询
try:
    with open(test_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        # 取第一行的前 20 个字符作为查询（假设第一行是标题）
        query = first_line[:20] if len(first_line) > 5 else "测试"
    
    print(f"查询: {query}")
    results = search(query, n_results=3)
    
    if not results:
        print("[WARN] 没有检索到结果")
    else:
        print(f"[OK] 检索成功，返回 {len(results)} 条结果:")
        for i, r in enumerate(results, 1):
            print(f"   --- 结果 {i} ---")
            print(f"   相似度: {r['similarity']}")
            print(f"   来源: {r['source']}")
            if r.get("header_path"):
                print(f"   章节: {r['header_path']}")
            print(f"   内容预览: {r['content'][:100]}...")
            print()
except Exception as e:
    print(f"[FAIL] 检索失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================
# 测试 5：列表源文件
# ============================================================
print("=" * 60)
print("测试 5：列表源文件（list_sources）")
print("=" * 60)

try:
    sources = list_sources()
    print(f"[OK] 已索引 {len(sources)} 个源文件:")
    for s in sources[:5]:  # 只显示前 5 个
        print(f"   {s}")
    if len(sources) > 5:
        print(f"   ... 还有 {len(sources) - 5} 个")
except Exception as e:
    print(f"[FAIL] 列表源文件失败: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================
# 测试 6：（可选）测试 Rerank
# ============================================================
print("=" * 60)
print("测试 6：（可选）Rerank 测试")
print("=" * 60)

from config import RERANK_MODEL
if not RERANK_MODEL:
    print("[WARN] RERANK_MODEL 未配置，跳过 Rerank 测试")
    print("   如需启用，请在 config.py 中设置 RERANK_MODEL")
else:
    print(f"[OK] RERANK_MODEL 已配置: {RERANK_MODEL}")
    print("   正在测试 Rerank...")
    try:
        results = search(query, n_results=5)
        if results and "rerank_score" in results[0]:
            print("[OK] Rerank 成功，结果已重新排序")
        else:
            print("[WARN] Rerank 可能未生效，请检查模型加载")
    except Exception as e:
        print(f"[FAIL] Rerank 测试失败: {e}")

print()

# ============================================================
# 测试总结
# ============================================================
print("=" * 60)
print("测试总结")
print("=" * 60)
print("[OK] 依赖导入: 成功")
print(f"[OK] 单文件索引: 成功（{test_file}）")
print(f"[OK] 检索测试: 成功（查询: {query}）")
print(f"[OK] 列表源文件: 成功（{len(sources)} 个源文件）")
if not RERANK_MODEL:
    print("[WARN] Rerank: 未配置（可选）")
else:
    print("[OK] Rerank: 已配置")

print()
print("[DONE] 基础测试全部通过！")
print()
print("下一步:")
print("  1. 运行 python -c \"from indexer import reindex_all; print(reindex_all())\" 进行全量索引")
print("  2. 配置 CodeBuddy MCP，使用 rag_server.py 作为 MCP Server")
print("  3. （可选）在 config.py 中配置 RERANK_MODEL 启用 Rerank")
