"""調音器核心邏輯

音高偵測演算法：FFT + 自相關（Autocorrelation）混合
參考音 A4 預設 440 Hz（可調 430~450 Hz）
"""

import numpy as np
import sounddevice as sd
from typing import Callable, Optional


# TODO: 實作 FFT + 自相關音高偵測
# TODO: 實作麥克風收音串流


class Tuner:
    """調音器引擎"""

    def __init__(self):
        self._ref_a4 = 440.0  # Hz
        self._is_listening = False
        self._stream: Optional[sd.InputStream] = None
        self._pitch_callback: Optional[Callable[[str, float, float], None]] = None

    @property
    def ref_a4(self) -> float:
        return self._ref_a4

    @ref_a4.setter
    def ref_a4(self, value: float):
        self._ref_a4 = max(430.0, min(450.0, value))

    def set_pitch_callback(self, callback: Callable[[str, float, float], None]):
        """設定偵測到音高時的回調
        callback(note_name, cents, frequency)
        """
        self._pitch_callback = callback

    def start(self):
        """啟動麥克風收音"""
        if self._is_listening:
            return
        self._is_listening = True

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio input error: {status}")
                return
            # TODO: 實作音高偵測
            pass

        self._stream = sd.InputStream(
            samplerate=44100,
            channels=1,
            dtype='float32',
            blocksize=4096,
            callback=audio_callback
        )
        self._stream.start()

    def stop(self):
        """停止收音"""
        self._is_listening = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @staticmethod
    def frequency_to_note(freq: float, ref_a4: float = 440.0) -> tuple:
        """將頻率轉換為音名與 cents 偏差
        返回 (note_name, cents)
        """
        if freq <= 0:
            return "--", 0

        # 計算相對於 A4 的半音數
        semitones = 12 * np.log2(freq / ref_a4)
        nearest = round(semitones)
        cents = round((semitones - nearest) * 100)

        note_names = ["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"]
        note_index = (nearest + 12 * 100) % 12
        note_name = note_names[note_index]

        return note_name, cents