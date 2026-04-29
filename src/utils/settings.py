"""設定檔管理

根據 Section 8.7：
- 設定檔：%APPDATA%/MusicMate/config/settings.json
- 最近檔案：%APPDATA%/MusicMate/config/recent_files.json
- 快取：%APPDATA%/MusicMate/cache/
"""

import json
import platform
from pathlib import Path
from typing import Any, Optional


class Settings:
    """設定檔管理類別"""

    DEFAULT = {
        "general": {
            "language": "zh-TW",
            "theme": "system",  # light / dark / system
            "check_update_on_startup": True
        },
        "omr": {
            "dpi": 300,
            "timeout_seconds": 60
        },
        "playback": {
            "default_timbre": "鋼琴",
            "default_speed": 1.0,
            "loop_enabled": False
        },
        "metronome": {
            "default_bpm": 120,
            "default_time_sig": "4/4",
            "default_sound": "Click",
            "default_volume": 70
        },
        "tuner": {
            "default_a4": 440,
            "default_instrument": "鋼琴"
        }
    }

    def __init__(self):
        self._path = self._get_settings_path()
        self._data: dict = {}
        self.load()

    def _get_settings_path(self) -> Path:
        if platform.system() == "Windows":
            base = Path.home() / "AppData" / "Roaming"
        else:
            base = Path.home() / ".config"
        config_dir = base / "MusicMate" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "settings.json"

    def load(self):
        """載入設定檔（若不存在則使用預設值）"""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                # 合併預設值（避免新欄位遺漏）
                self._data = self._merge_defaults(self._data, self.DEFAULT)
            except Exception:
                self._data = dict(self.DEFAULT)
        else:
            self._data = dict(self.DEFAULT)

    def save(self):
        """儲存設定檔（使用原子寫入）"""
        tmp_path = self._path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            tmp_path.replace(self._path)  # 原子 rename
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise e

    def get(self, *keys, default=None) -> Any:
        """取得設定值（巢狀 key）"""
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
                if val is None:
                    return default
            else:
                return default
        return val

    def set(self, value: Any, *keys):
        """設定設定值（巢狀 key）"""
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    @staticmethod
    def _merge_defaults(data: dict, defaults: dict) -> dict:
        """遞迴合併預設值"""
        result = dict(defaults)
        for k, v in data.items():
            if k in result and isinstance(v, dict) and isinstance(result[k], dict):
                result[k] = Settings._merge_defaults(v, result[k])
            else:
                result[k] = v
        return result