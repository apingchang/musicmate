"""MIDI 播放器核心

使用 mido 處理 MIDI 訊息 + pygame 播放
支援 GM 音色、速度調整、移調、循環播放
"""

from typing import Callable, Optional

import mido
from mido import Message, MidiFile, MidiTrack


# TODO: 實作 pygame simpleaudio MIDI 播放
# TODO: 實作 GM 音色映射（Program Change）
# TODO: 實作速度調整、移調、循環


class MidiPlayer:
    """MIDI 播放引擎"""

    def __init__(self):
        self._is_playing = False
        self._current_position = 0.0
        self._speed = 1.0  # 0.5x ~ 2.0x
        self._transpose = 0  # -12 ~ +12 semitones
        self._loop_start = None
        self._loop_end = None
        self._loop_enabled = False
        self._tempo = 120.0
        self._progress_callback: Optional[Callable[[float, float], None]] = None

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float):
        self._speed = max(0.5, min(2.0, value))

    @property
    def transpose(self) -> int:
        return self._transpose

    @transpose.setter
    def transpose(self, value: int):
        self._transpose = max(-12, min(12, value))

    def set_progress_callback(self, callback: Callable[[float, float], None]):
        """設定播放進度回調（已播放秒數，總秒數）"""
        self._progress_callback = callback

    def play(self, json_notes: dict):
        """播放樂譜（JSON 音符結構）"""
        # TODO: 實作
        # 1. 將 JSON 轉換為 mido MIDI 訊息
        # 2. 套用速度調整、移調
        # 3. 播放
        raise NotImplementedError("MIDI 播放待實作")

    def stop(self):
        """停止播放"""
        self._is_playing = False

    def pause(self):
        """暫停播放"""
        self._is_playing = False

    def set_loop(self, start: int, end: int, enabled: bool):
        """設定循環播放範圍（小節）"""
        self._loop_start = start
        self._loop_end = end
        self._loop_enabled = enabled

    def load_musicxml(self, path: str) -> dict:
        """從 MusicXML 檔案載入樂譜"""
        # TODO: 使用 music21 解析 MusicXML
        raise NotImplementedError("MusicXML 載入待實作")