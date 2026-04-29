"""節拍器核心邏輯

音效來源優先順序：
1. Windows MIDI Synthesizer（主機內建，預設）
2. 預錄 WAV 樣本（備援）

支援音色：Click / 木魚 / Digital / 小鼓 / 叮叮聲 / 狗吠
"""

import time
import threading
from typing import Callable, Optional


# TODO: 實作 Windows MIDI Synth 音效（winrt 或 winsound）
# TODO: 實作 WAV 樣本播放（pygame 或 simpleaudio）


class Metronome:
    """節拍器引擎"""

    def __init__(self):
        self._bpm = 120
        self._beats_per_measure = 4
        self._current_beat = 0
        self._is_playing = False
        self._thread: Optional[threading.Thread] = None
        self._tick_callback: Optional[Callable[[int, bool], None]] = None
        # 0=Click, 1=木魚, 2=Digital, 3=小鼓, 4=叮叮聲, 5=狗吠
        self._sound = 0

    @property
    def bpm(self) -> int:
        return self._bpm

    @bpm.setter
    def bpm(self, value: int):
        self._bpm = max(40, min(240, value))

    @property
    def beats_per_measure(self) -> int:
        return self._beats_per_measure

    @beats_per_measure.setter
    def beats_per_measure(self, value: int):
        self._beats_per_measure = value

    @property
    def current_beat(self) -> int:
        return self._current_beat

    def set_sound(self, sound_index: int):
        """設定節拍音效
        0=Click, 1=木魚, 2=Digital, 3=小鼓, 4=叮叮聲, 5=狗吠
        """
        self._sound = sound_index

    def set_tick_callback(self, callback: Callable[[int, bool], None]):
        """設定每拍觸發的回調（用於更新 UI）
        callback(beat_index, is_accent)
        """
        self._tick_callback = callback

    def start(self):
        """啟動節拍器"""
        if self._is_playing:
            return
        self._is_playing = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """停止節拍器"""
        self._is_playing = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._current_beat = 0

    def _run(self):
        """節拍器執行緒"""
        while self._is_playing:
            interval_ms = 60000.0 / self._bpm
            is_accent = (self._current_beat == 0)

            # 播放音效
            self._play_sound(is_accent)

            # 通知 UI
            if self._tick_callback:
                self._tick_callback(self._current_beat, is_accent)

            # 等待下一拍
            time.sleep(interval_ms / 1000.0)
            self._current_beat = (self._current_beat + 1) % self._beats_per_measure

    def _play_sound(self, is_accent: bool):
        """播放節拍音效"""
        # TODO: 實作主機內建音效或 WAV 樣本播放
        pass