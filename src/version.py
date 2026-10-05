"""MusicMate 版本與建置資訊模組"""

import os
import sys
import datetime
from pathlib import Path

VERSION = "1.1.0"
AUTHOR = "William Chang"

# 當在 PyInstaller 打包環境或開發環境時自動取得修改/建置時間
def get_build_time_str() -> str:
    """取得程式碼最後修改時間或執行檔建置時間"""
    try:
        # 如果是 PyInstaller 打包的單一執行檔
        if getattr(sys, "frozen", False):
            exe_path = Path(sys.executable)
            mtime = exe_path.stat().st_mtime
            dt = datetime.datetime.fromtimestamp(mtime)
            return dt.strftime("%Y-%m-%d %H:%M")

        # 開發環境：找出 src/ 目錄下所有 .py 檔最新的修改時間
        src_dir = Path(__file__).resolve().parent
        py_files = list(src_dir.rglob("*.py"))
        if py_files:
            latest_mtime = max(f.stat().st_mtime for f in py_files)
            dt = datetime.datetime.fromtimestamp(latest_mtime)
            return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    # 兜底返回當前日期時間
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def get_window_title() -> str:
    """取得包含修改/建置時間的主視窗標題"""
    build_time = get_build_time_str()
    return f"練琴寶 MusicMate v{VERSION} ({build_time})"
