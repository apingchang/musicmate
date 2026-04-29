"""日誌設定

根據 Section 9.1.1 規格：
- 路徑：%APPDATA%/MusicMate/logs/musicmate.log
- 格式：[YYYY-MM-DD HH:MM:SS] [LEVEL] [模組] 訊息
- 輪轉：上限 10MB，保留 5 個備份
"""

import logging
import logging.handlers
import platform
from pathlib import Path


def setup_logger(name: str = "musicmate") -> logging.Logger:
    """設定並回傳 logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # 已經設定過了

    # 日誌層級
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 主控台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 檔案 Handler（輪轉）
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "musicmate.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "musicmate") -> logging.Logger:
    """取得 logger（若未設定會自動設定）"""
    return logging.getLogger(name)


def _get_log_dir() -> Path:
    """取得平台對應的日誌目錄"""
    if platform.system() == "Windows":
        base = Path.home() / "AppData" / "Roaming"
    else:
        base = Path.home() / ".config"

    return base / "MusicMate" / "logs"