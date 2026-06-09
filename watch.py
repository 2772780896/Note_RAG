"""
文件监控模块 —— 使用 watchdog 监听笔记目录，
当 .md 文件发生变动时，自动调用 indexer.index_file() 重新索引。

用法:
    python watch.py

【面试必考】为什么需要文件监控？
    答：RAG 系统的索引和原文是"离线"的，原文更新后，向量库不会自动同步。
    如果不做监控，用户改了笔记，RAG 检索出来的还是旧内容，这叫"索引漂移"。
"""

import time
import os
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 把当前目录加入 sys.path，确保能导入 indexer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indexer import index_file, _find_md_files

# ============================================================
# 配置：要监控的笔记目录（从 config.py 读取）
# ============================================================
from config import NOTES_DIRS

class NotesFileHandler(FileSystemEventHandler):
    """
    文件系统事件处理器：
    - 当 .md 文件被创建/修改/移动时，自动重新索引
    - 忽略非 .md 文件，忽略目录事件
    """
    def __init__(self):
        super().__init__()
        # 防抖：同一文件在短时间内可能被触发多次，用字典记录最后处理时间
        self._last_processed = {}
        self._debounce_seconds = 2  # 2 秒内同一文件只处理一次

    def _should_process(self, file_path: str) -> bool:
        """防抖检查：避免文件保存时触发多次事件"""
        now = time.time()
        last_time = self._last_processed.get(file_path, 0)
        if now - last_time < self._debounce_seconds:
            return False
        self._last_processed[file_path] = now
        return True

    def _handle_md_file(self, file_path: str):
        """处理 .md 文件的通用逻辑"""
        if not file_path.endswith(".md"):
            return
        if not self._should_process(file_path):
            return

        print(f"🔍 检测到笔记变动: {file_path}")
        try:
            # 等待文件写入完成（Obsidian 保存时可能触发多次）
            time.sleep(0.5)
            result = index_file(file_path)
            print(f"✅ {result}")
        except Exception as e:
            print(f"❌ 索引失败: {e}")

    def on_modified(self, event):
        """文件被修改时触发"""
        if not event.is_directory:
            self._handle_md_file(event.src_path)

    def on_created(self, event):
        """新文件被创建时触发"""
        if not event.is_directory:
            self._handle_md_file(event.src_path)

    def on_moved(self, event):
        """文件被移动/重命名时触发"""
        if not event.is_directory:
            # 旧路径删除索引，新路径创建索引
            self._handle_md_file(event.dest_path)


def start_watching():
    """启动文件监控"""
    event_handler = NotesFileHandler()
    observer = Observer()

    # 为每个笔记目录启动一个监控
    watched_dirs = []
    for notes_dir in NOTES_DIRS:
        if os.path.isdir(notes_dir):
            observer.schedule(event_handler, path=notes_dir, recursive=True)
            watched_dirs.append(notes_dir)
            print(f"👀 正在监控: {notes_dir}")
        else:
            print(f"⚠️  目录不存在，跳过: {notes_dir}")

    if not watched_dirs:
        print("❌ 没有可监控的目录，请检查 config.py 中的 NOTES_DIRS")
        return

    observer.start()
    print("🚀 文件监控已启动！按 Ctrl+C 停止")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 正在停止监控...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start_watching()
