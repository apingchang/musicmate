"""跨平台即時樂音合成器 (NoteSynthesizer)

使用 NumPy 即時合成高品質樂器音色，透過 pygame.mixer 播放（與節拍器共用後端）：
- 支援音色：鋼琴 (piano)、薩克斯風 (sax)、長笛 (flute)
- 支援任意 MIDI 音高（36 ~ 96）與動態時長、ADSR 包絡線、防爆音淡出
- 零外部 SoundFont / MIDI 驅動依賴，Windows 與 Linux 皆可開箱即用
"""

import math
from typing import Dict, Optional, Tuple
import numpy as np

_audio_backend = None
try:
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    pygame.mixer.set_num_channels(16)
    _audio_backend = "pygame"
except Exception:
    _audio_backend = None


class NoteSynthesizer:
    """純 Python / NumPy 跨平台樂音合成播放器"""

    TIMBRES = {
        "鋼琴": "piano",
        "薩克斯風": "sax",
        "長笛": "flute",
    }

    def __init__(self, sample_rate: int = 44100):
        self._sample_rate = sample_rate
        self._timbre = "piano"
        self._volume = 0.75
        self._enabled = True
        self._cache: Dict[Tuple[str, int, int], any] = {}
        self._active_channel = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = bool(value)
        if not self._enabled:
            self.stop_all()

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float):
        self._volume = max(0.0, min(1.0, float(value)))

    @property
    def timbre(self) -> str:
        return self._timbre

    @timbre.setter
    def timbre(self, name: str):
        key = self.TIMBRES.get(name, name)
        if key in ("piano", "sax", "flute"):
            self._timbre = key

    @staticmethod
    def midi_to_freq(midi_note: int, ref_a4: float = 440.0) -> float:
        """將 MIDI 編號轉為頻率 (Hz)"""
        return float(ref_a4 * (2.0 ** ((midi_note - 69) / 12.0)))

    def synthesize_wave(self, midi_note: int, duration_sec: float, timbre: Optional[str] = None) -> np.ndarray:
        """合成指定音高、時長與音色的 16-bit PCM 波形陣列"""
        t_key = timbre or self._timbre
        dur = max(0.08, min(2.5, float(duration_sec)))
        n_samples = int(self._sample_rate * dur)
        t = np.linspace(0.0, dur, n_samples, endpoint=False, dtype=np.float32)
        freq = self.midi_to_freq(midi_note)

        if t_key == "sax":
            # 中音薩克斯風：豐富奇偶次泛音 + 微微溫暖的起音與氣息顫音
            vibrato = 1.0 + 0.003 * np.sin(2.0 * np.pi * 5.2 * t)
            phase = 2.0 * np.pi * freq * np.cumsum(vibrato) / self._sample_rate
            wave = (
                1.00 * np.sin(phase)
                + 0.65 * np.sin(2.0 * phase)
                + 0.42 * np.sin(3.0 * phase)
                + 0.22 * np.sin(4.0 * phase)
                + 0.14 * np.sin(5.0 * phase)
            )
            attack = min(0.035, dur * 0.20)
            release = min(0.06, dur * 0.25)
            env = np.ones(n_samples, dtype=np.float32)
            a_n = max(1, int(attack * self._sample_rate))
            r_n = max(1, int(release * self._sample_rate))
            env[:a_n] = np.linspace(0.0, 1.0, a_n, dtype=np.float32)
            env[-r_n:] = np.linspace(1.0, 0.0, r_n, dtype=np.float32)
            wave *= env

        elif t_key == "flute":
            # 長笛：純淨基頻 + 柔和二次/三次泛音
            phase = 2.0 * np.pi * freq * t
            wave = (
                1.00 * np.sin(phase)
                + 0.25 * np.sin(2.0 * phase)
                + 0.08 * np.sin(3.0 * phase)
            )
            attack = min(0.04, dur * 0.22)
            release = min(0.07, dur * 0.25)
            env = np.ones(n_samples, dtype=np.float32)
            a_n = max(1, int(attack * self._sample_rate))
            r_n = max(1, int(release * self._sample_rate))
            env[:a_n] = np.linspace(0.0, 1.0, a_n, dtype=np.float32)
            env[-r_n:] = np.linspace(1.0, 0.0, r_n, dtype=np.float32)
            wave *= env

        else:
            # 鋼琴（預設）：清脆起音 + 自然指數衰減 + 諧波衰減
            phase = 2.0 * np.pi * freq * t
            wave = (
                1.00 * np.sin(phase) * np.exp(-2.2 * t)
                + 0.45 * np.sin(2.0 * phase) * np.exp(-3.5 * t)
                + 0.22 * np.sin(3.0 * phase) * np.exp(-5.0 * t)
                + 0.10 * np.sin(4.0 * phase) * np.exp(-6.5 * t)
            )
            a_n = max(1, int(0.008 * self._sample_rate))
            r_n = max(1, int(min(0.04, dur * 0.2) * self._sample_rate))
            wave[:a_n] *= np.linspace(0.0, 1.0, a_n, dtype=np.float32)
            wave[-r_n:] *= np.linspace(1.0, 0.0, r_n, dtype=np.float32)

        peak = float(np.max(np.abs(wave)))
        if peak > 1e-6:
            wave = wave / peak * 0.75

        return (wave * 32767.0).astype(np.int16)

    def play_note(self, midi_note: int, duration_sec: float = 0.35):
        """即時播放指定 MIDI 音符"""
        if not self._enabled or _audio_backend != "pygame":
            return
        if midi_note < 24 or midi_note > 108:
            return

        # 量化時長至 50ms 精度以便高效快取，避免頻繁重複運算
        dur_bucket = max(100, min(2000, int(round(duration_sec * 1000 / 50) * 50)))
        cache_key = (self._timbre, int(midi_note), dur_bucket)

        snd = self._cache.get(cache_key)
        if snd is None:
            pcm = self.synthesize_wave(midi_note, dur_bucket / 1000.0, self._timbre)
            try:
                # 確認 mixer 聲道數相容（單聲道或立體聲）
                init_info = pygame.mixer.get_init()
                if init_info and init_info[2] == 2:
                    pcm = np.column_stack((pcm, pcm))
                snd = pygame.sndarray.make_sound(pcm)
                # 限制快取上限避免記憶體無限成長
                if len(self._cache) > 256:
                    self._cache.clear()
                self._cache[cache_key] = snd
            except Exception:
                return

        try:
            # 單音樂器模式：淡出前一個殘音，讓旋律線條乾淨俐落
            if self._active_channel is not None and self._active_channel.get_busy():
                self._active_channel.fadeout(25)
            snd.set_volume(self._volume)
            self._active_channel = snd.play()
        except Exception:
            pass

    def stop_all(self):
        """立即停止目前播放中的樂音"""
        if _audio_backend == "pygame":
            try:
                if self._active_channel is not None:
                    self._active_channel.fadeout(20)
                    self._active_channel = None
            except Exception:
                pass
