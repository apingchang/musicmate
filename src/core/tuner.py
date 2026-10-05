"""調音器核心邏輯

高精度音高偵測與跨平台麥克風安全收音：
- 音高偵測演算法：Normalized Autocorrelation（自相關）搭配拋物線次取樣插值
- 支援靜音閘（RMS 門檻過濾環境噪音）
- 參考音 A4 可自訂（430~450 Hz，預設 440 Hz）
- 輸出：音名、八度、Cents 偏差（-50 ~ +50）、頻率（Hz）
- 支援多種樂器標準音對應（吉他、烏克麗麗、小提琴、大提琴等）
- 跨平台 PortAudio / sounddevice 容錯與安全串流管理
"""

import math
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None


class Tuner:
    """跨平台調音器核心引擎"""

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # 常見樂器預設弦音高 (Hz 與 音名八度)
    INSTRUMENTS_PRESETS = {
        "吉他": [
            ("6弦 (E2)", 82.41),
            ("5弦 (A2)", 110.00),
            ("4弦 (D3)", 146.83),
            ("3弦 (G3)", 196.00),
            ("2弦 (B3)", 246.94),
            ("1弦 (E4)", 329.63),
        ],
        "烏克麗麗": [
            ("4弦 (G4)", 392.00),
            ("3弦 (C4)", 261.63),
            ("2弦 (E4)", 329.63),
            ("1弦 (A4)", 440.00),
        ],
        "小提琴": [
            ("4弦 (G3)", 196.00),
            ("3弦 (D4)", 293.66),
            ("2弦 (A4)", 440.00),
            ("1弦 (E5)", 659.25),
        ],
        "大提琴": [
            ("4弦 (C2)", 65.41),
            ("3弦 (G2)", 97.99),
            ("2弦 (D3)", 146.83),
            ("1弦 (A3)", 220.00),
        ],
        "長笛": [
            ("最低音 (C4)", 261.63),
            ("基準音 (A4)", 440.00),
        ],
        "鋼琴": [
            ("全音域", 0.0),
        ],
    }

    def __init__(self, sample_rate: int = 44100, buffer_size: int = 4096):
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size
        self._ref_a4 = 440.0  # Hz
        self._rms_threshold = 0.008  # 靜音門檻
        self._is_listening = False
        self._stream = None
        self._pitch_callback: Optional[Callable[[str, int, float, float, bool], None]] = None
        self._error_callback: Optional[Callable[[str], None]] = None
        self._current_instrument = "吉他"
        self._pitch_history: list[float] = []  # 最近偵測頻率暫存，用於中值平滑防抖

    @property
    def ref_a4(self) -> float:
        return self._ref_a4

    @ref_a4.setter
    def ref_a4(self, value: float):
        self._ref_a4 = max(430.0, min(450.0, float(value)))

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    @property
    def instrument(self) -> str:
        return self._current_instrument

    @instrument.setter
    def instrument(self, name: str):
        if name in self.INSTRUMENTS_PRESETS:
            self._current_instrument = name

    def set_pitch_callback(self, callback: Callable[[str, int, float, float, bool], None]):
        """設定偵測到音高時的回調
        callback(note_name, octave, cents, frequency, is_in_tune)
        """
        self._pitch_callback = callback

    def set_error_callback(self, callback: Callable[[str], None]):
        """設定麥克風發生錯誤時的回調
        callback(error_message)
        """
        self._error_callback = callback

    def start(self) -> bool:
        """啟動麥克風即時收音（Linux 優先使用 arecord 原生串流，Windows 使用 sounddevice）"""
        if self._is_listening:
            return True

        import platform, shutil, subprocess, threading

        # Linux 平台：若有 arecord，優先使用原生 ALSA 串流（零外部 C 依賴，保證穩定）
        if platform.system() == "Linux" and shutil.which("arecord") is not None:
            return self._start_arecord_stream()

        # Windows 或其他平台：使用 sounddevice / PortAudio
        return self._start_sounddevice_stream()

    def _start_arecord_stream(self) -> bool:
        """Linux 原生 ALSA 麥克風串流"""
        import subprocess, threading
        try:
            cmd = ['arecord', '-f', 'S16_LE', '-r', str(self._sample_rate), '-c', '1', '-q', '-t', 'raw']
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._is_listening = True

            def arecord_loop():
                bytes_to_read = self._buffer_size * 2  # 16-bit mono PCM
                while self._is_listening and self._proc and self._proc.poll() is None:
                    try:
                        raw = self._proc.stdout.read(bytes_to_read)
                        if not raw or len(raw) < bytes_to_read:
                            continue
                        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                        self._process_audio_frame(samples)
                    except Exception:
                        pass

            self._thread = threading.Thread(target=arecord_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            self._is_listening = False
            if self._error_callback:
                self._error_callback(f"Linux 麥克風啟動失敗：{str(e)}")
            return False

    def _start_sounddevice_stream(self) -> bool:
        """Windows / 通用 sounddevice 麥克風串流"""
        if sd is None:
            if self._error_callback:
                self._error_callback("系統未安裝 sounddevice 或音訊驅動不可用")
            return False

        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
            if not input_devices:
                if self._error_callback:
                    self._error_callback("找不到可用的麥克風輸入裝置")
                return False

            self._is_listening = True

            def audio_callback(indata, frames, time_info, status):
                if not self._is_listening:
                    return
                try:
                    audio_data = indata[:, 0]
                    self._process_audio_frame(audio_data)
                except Exception:
                    pass

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self._buffer_size,
                callback=audio_callback
            )
            self._stream.start()
            return True

        except Exception as e:
            self._is_listening = False
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            if self._error_callback:
                self._error_callback(f"無法存取麥克風：{str(e)}")
            return False

    def stop(self):
        """停止麥克風收音"""
        self._is_listening = False
        if hasattr(self, '_proc') and self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=0.3)
            except Exception:
                pass
            self._proc = None

        if hasattr(self, '_stream') and self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


    def _process_audio_frame(self, frame: np.ndarray):
        """處理單一音訊緩衝區"""
        # 計算 RMS 音量
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms < self._rms_threshold:
            # 靜音狀態，重置平滑隊列
            self._pitch_history.clear()
            if self._pitch_callback:
                self._pitch_callback("--", 0, 0.0, 0.0, False)
            return

        # 計算基頻
        raw_freq = self.detect_pitch(frame, self._sample_rate)
        if raw_freq is None or raw_freq < 30.0 or raw_freq > 4500.0:
            if self._pitch_callback and len(self._pitch_history) == 0:
                self._pitch_callback("--", 0, 0.0, 0.0, False)
            return

        # 中值濾波防抖平滑（取最近 5 幀的中位數）
        self._pitch_history.append(raw_freq)
        if len(self._pitch_history) > 5:
            self._pitch_history.pop(0)

        # 排序取中位數，排除偶發諧波跳躍與噪聲尖峰
        sorted_history = sorted(self._pitch_history)
        stable_freq = sorted_history[len(sorted_history) // 2]

        # 轉換為音名與 cents
        note_name, octave, cents = self.frequency_to_note(stable_freq, self._ref_a4)
        is_in_tune = abs(cents) <= 5

        if self._pitch_callback:
            self._pitch_callback(note_name, octave, cents, stable_freq, is_in_tune)

    @classmethod
    def detect_pitch(cls, signal: np.ndarray, sample_rate: int = 44100, thresh: float = 0.15) -> Optional[float]:
        """使用 FFT 加速且具備倍頻保護的 YIN 演算法計算樂器基頻 (Hz)
        涵蓋鋼琴全音域：30 Hz (A0) ~ 4500 Hz (C8)
        """
        w_len = 2048
        if len(signal) < w_len * 2:
            return None

        # 減去直流偏移
        x = signal[:w_len * 2] - np.mean(signal[:w_len * 2])
        w = w_len

        # 1. 使用 FFT 高效精確計算差分函數 d(tau)
        # d(tau) = sum_{j=0}^{w-1} (x[j] - x[j+tau])^2
        x_part = x[:w]
        power = np.sum(x_part ** 2)
        conv = np.fft.irfft(np.fft.rfft(x, n=w * 2) * np.conj(np.fft.rfft(x_part, n=w * 2)))
        r = conv[:w]

        cumsum = np.concatenate(([0], np.cumsum(x[:w * 2] ** 2)))
        power_tau = cumsum[w: 2 * w] - cumsum[:w]
        d = power + power_tau - 2 * r
        d[0] = 0.0

        # 2. 累積均值歸一化差分 (CMNDF)
        d_prime = np.zeros(w, dtype=np.float32)
        d_prime[0] = 1.0
        cum_d = 0.0
        for tau in range(1, w):
            cum_d += d[tau]
            d_prime[tau] = d[tau] / (cum_d / tau) if cum_d > 0 else 1.0

        # 3. 搜尋範圍限制在 30Hz ~ 4500Hz
        min_tau = max(3, int(sample_rate / 4500.0))  # 4500Hz 對應約 9 samples
        max_tau = min(w - 1, int(sample_rate / 30.0)) # 30Hz 對應約 1470 samples

        # 尋找第一個低於 thresh 的局部極小值（週期最短、頻率最高之候選）
        tau_chosen = None
        for tau in range(min_tau, max_tau):
            if d_prime[tau] < thresh:
                while tau + 1 < max_tau and d_prime[tau + 1] < d_prime[tau]:
                    tau += 1
                tau_chosen = tau
                break

        if tau_chosen is None:
            # 若無低於門檻值者，取有效區間內的最小值
            tau_chosen = min_tau + int(np.argmin(d_prime[min_tau:max_tau]))
            if d_prime[tau_chosen] > 0.35:
                # 非週期性噪音，過濾掉
                return None

        # 4. 拋物線插值求取亞取樣精準週期
        p = tau_chosen
        if 0 < p < w - 1:
            s0, s1, s2 = d_prime[p - 1], d_prime[p], d_prime[p + 1]
            denom = 2 * (s0 - 2 * s1 + s2)
            delta = (s0 - s2) / denom if abs(denom) > 1e-9 else 0.0
            exact_tau = p + delta
        else:
            exact_tau = float(p)

        if exact_tau <= 0:
            return None

        freq = float(sample_rate / exact_tau)
        return freq


    @classmethod
    def frequency_to_note(cls, freq: float, ref_a4: float = 440.0) -> Tuple[str, int, float]:
        """將頻率轉換為 (音名, 八度, cents偏差)"""
        if freq <= 0:
            return "--", 0, 0.0

        # 計算相對於 A4 的半音數
        semitones = 12.0 * math.log2(freq / ref_a4)
        midi_note = round(69 + semitones)
        exact_semitones = semitones - (midi_note - 69)
        cents = round(exact_semitones * 100.0, 1)

        note_idx = midi_note % 12
        octave = (midi_note // 12) - 1
        note_name = cls.NOTE_NAMES[note_idx]

        return note_name, octave, cents

    def get_closest_preset_target(self, current_freq: float) -> Tuple[str, float, float]:
        """取得當前樂器預設中最接近的目標音
        返回: (目標名稱, 目標頻率, 偏差Hz)
        """
        targets = self.INSTRUMENTS_PRESETS.get(self._current_instrument, [])
        if not targets or targets[0][1] <= 0 or current_freq <= 0:
            return "--", 0.0, 0.0

        best_target = None
        min_diff = float('inf')
        for name, target_freq in targets:
            diff = abs(current_freq - target_freq)
            if diff < min_diff:
                min_diff = diff
                best_target = (name, target_freq, current_freq - target_freq)

        return best_target