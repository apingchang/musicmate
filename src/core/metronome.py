"""節拍器核心邏輯

跨平台支援（Linux & Windows）：
- 採用高精度波形合成（NumPy）生成節拍音色
- 支援音色：0=Click, 1=木魚, 2=Digital, 3=小鼓, 4=叮叮聲, 5=狗吠
- 區分重音（第 1 拍）與輕音
- 支援音量控制與即時切換
- 基於 time.perf_counter 的無漂移精準計時迴圈
"""

import math
import threading
import time
from typing import Callable, Dict, Optional, Tuple
import numpy as np

# 音訊播放後端初始化（優先 pygame.mixer，次選 sounddevice）
_audio_backend = None
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    _audio_backend = "pygame"
except Exception:
    try:
        import sounddevice as sd
        _audio_backend = "sounddevice"
    except Exception:
        _audio_backend = None


class Metronome:
    """跨平台節拍器引擎"""

    def __init__(self):
        self._bpm = 120
        self._beats_per_measure = 4
        self._current_beat = 0
        self._is_playing = False
        self._thread: Optional[threading.Thread] = None
        self._tick_callback: Optional[Callable[[int, bool], None]] = None
        # 0=Click, 1=木魚, 2=Digital, 3=小鼓, 4=叮叮聲, 5=狗吠
        self._sound = 0
        self._volume = 0.7  # 0.0 ~ 1.0
        self._sample_rate = 44100

        # 快取的音訊波形與 sound 物件: (sound_idx, is_accent) -> Sound/Array
        self._sound_cache: Dict[Tuple[int, bool], any] = {}
        self._pregenerate_sounds()

    @property
    def bpm(self) -> int:
        return self._bpm

    @bpm.setter
    def bpm(self, value: int):
        self._bpm = max(40, min(240, int(value)))

    @property
    def beats_per_measure(self) -> int:
        return self._beats_per_measure

    @beats_per_measure.setter
    def beats_per_measure(self, value: int):
        self._beats_per_measure = max(1, int(value))

    @property
    def current_beat(self) -> int:
        return self._current_beat

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float):
        self._volume = max(0.0, min(1.0, float(value)))
        if _audio_backend == "pygame":
            for snd in self._sound_cache.values():
                if hasattr(snd, "set_volume"):
                    snd.set_volume(self._volume)

    def set_sound(self, sound_index: int):
        """設定節拍音效
        0=Click, 1=木魚, 2=Digital, 3=小鼓, 4=叮叮聲, 5=狗吠
        """
        self._sound = max(0, min(5, sound_index))

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
        self._current_beat = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """停止節拍器"""
        self._is_playing = False
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None
        self._current_beat = 0

    def _run(self):
        """節拍器無漂移精準計時執行緒"""
        interval = 60.0 / self._bpm
        next_tick = time.perf_counter()

        while self._is_playing:
            now = time.perf_counter()
            if now < next_tick:
                # 睡眠剩餘時間的一半，保持低 CPU 佔用且具高精準度
                sleep_time = (next_tick - now) * 0.5
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
                continue

            # 觸發此拍
            is_accent = (self._current_beat == 0)
            self._play_sound(is_accent)

            if self._tick_callback:
                try:
                    self._tick_callback(self._current_beat, is_accent)
                except Exception:
                    pass

            # 準備下一拍
            self._current_beat = (self._current_beat + 1) % self._beats_per_measure
            interval = 60.0 / self._bpm
            next_tick += interval

            # 如果落後太多（例如系統暫停），重新校正基準時間
            if time.perf_counter() - next_tick > interval:
                next_tick = time.perf_counter() + interval

    def _play_sound(self, is_accent: bool):
        """播放節拍音效"""
        key = (self._sound, is_accent)
        snd = self._sound_cache.get(key)
        if snd is None:
            return

        try:
            if _audio_backend == "pygame":
                snd.set_volume(self._volume)
                snd.play()
            elif _audio_backend == "sounddevice":
                import sounddevice as sd
                sd.play(snd * self._volume, samplerate=self._sample_rate, blocking=False)
        except Exception:
            pass

    def _pregenerate_sounds(self):
        """預先生成所有音色（6 種音色 × 重/輕音）之音訊波形"""
        for sound_idx in range(6):
            for is_accent in (True, False):
                raw_wave = self._generate_waveform(sound_idx, is_accent)
                if _audio_backend == "pygame":
                    try:
                        import pygame
                        # 轉換為 16-bit 有號整數 PCM
                        pcm16 = (raw_wave * 32767).astype(np.int16)
                        snd = pygame.sndarray.make_sound(pcm16)
                        self._sound_cache[(sound_idx, is_accent)] = snd
                    except Exception:
                        self._sound_cache[(sound_idx, is_accent)] = raw_wave
                else:
                    self._sound_cache[(sound_idx, is_accent)] = raw_wave.astype(np.float32)

    def _generate_waveform(self, sound_idx: int, is_accent: bool) -> np.ndarray:
        """合成不同音色的短促音訊波形（歸一化 -1.0 ~ 1.0）"""
        sr = self._sample_rate

        if sound_idx == 0:
            # 0: Click（經典節拍器短脈衝）
            duration = 0.03 if is_accent else 0.02
            t = np.linspace(0, duration, int(sr * duration), False)
            freq = 2400.0 if is_accent else 1600.0
            env = np.exp(-t * (120.0 if is_accent else 160.0))
            wave = np.sin(2 * np.pi * freq * t) * env
            return wave

        elif sound_idx == 1:
            # 1: 木魚 (Woodblock - 中頻共振與諧波)
            duration = 0.06 if is_accent else 0.04
            t = np.linspace(0, duration, int(sr * duration), False)
            f0 = 920.0 if is_accent else 680.0
            env = np.exp(-t * 90.0)
            wave = (np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * f0 * 2.2 * t)) * env
            return wave / 1.3

        elif sound_idx == 2:
            # 2: Digital (電子嗶聲)
            duration = 0.05 if is_accent else 0.035
            t = np.linspace(0, duration, int(sr * duration), False)
            freq = 1760.0 if is_accent else 880.0  # A6 / A5
            env = np.minimum(t / 0.005, 1.0) * np.exp(-t * 40.0)
            wave = np.sin(2 * np.pi * freq * t) * env
            return wave

        elif sound_idx == 3:
            # 3: 小鼓 (Snare / Drum)
            duration = 0.08 if is_accent else 0.05
            t = np.linspace(0, duration, int(sr * duration), False)
            f_tone = 200.0 if is_accent else 150.0
            tone = np.sin(2 * np.pi * f_tone * np.exp(-t * 30.0) * t) * np.exp(-t * 40.0)
            noise = np.random.uniform(-1.0, 1.0, len(t)) * np.exp(-t * 60.0)
            wave = 0.6 * tone + 0.4 * noise
            return wave

        elif sound_idx == 4:
            # 4: 叮叮聲 (Bell / Triangle)
            duration = 0.12 if is_accent else 0.08
            t = np.linspace(0, duration, int(sr * duration), False)
            f0 = 2093.0 if is_accent else 1567.0  # C7 / G6
            env = np.exp(-t * 35.0)
            wave = (np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * f0 * 1.5 * t)) * env
            return wave / 1.4

        else:
            # 5: 狗吠 (Bark / 短促降調調頻)
            duration = 0.10 if is_accent else 0.07
            t = np.linspace(0, duration, int(sr * duration), False)
            f_start = 450.0 if is_accent else 350.0
            f_end = 180.0 if is_accent else 150.0
            freq = f_start + (f_end - f_start) * (t / duration)
            phase = 2 * np.pi * np.cumsum(freq) / sr
            env = np.sin(np.pi * t / duration) ** 1.5
            noise = np.random.uniform(-0.2, 0.2, len(t))
            wave = (np.sin(phase) + noise) * env
            return wave