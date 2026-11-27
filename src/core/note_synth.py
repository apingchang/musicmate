"""跨平台真實樂器 SoundFont 合成器 (NoteSynthesizer)

優先使用 TinySoundFont (C 單檔 SF2 引擎，零外部系統套件依賴) 載入 GeneralUser GS
真實取樣音色庫 (assets/soundfonts/GeneralUser-GS.sf2)，透過 pygame.mixer 立體聲播放：
- 內建 17 種真實取樣交響／管樂／打擊樂器：
  鋼琴、薩克斯風、長笛、豎笛、雙簧管、低音管、小提琴、大提琴、低音提琴、
  小號、法國號、管風琴、馬林巴、定音鼓、三角鐵、小鼓、大鼓
- 採用 2048 緩衝區防欠載、固定樂器校準增益防多音疊加破音（anti-clipping）、
  餘弦平滑起音與長尾音釋放包絡線（raised-cosine fade），徹底消除音符銜接雜音與截斷爆音
- 若 SoundFont 或 tinysoundfont 不可用，自動無縫降級為純 NumPy 波形合成
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np

_audio_backend = None
try:
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    pygame.mixer.set_num_channels(24)
    _audio_backend = "pygame"
except Exception:
    _audio_backend = None

try:
    import tinysoundfont as _tsf
except Exception:
    _tsf = None


@dataclass(frozen=True)
class InstrumentSpec:
    """單一樂器之 SoundFont 音色與無雜音增益／尾音設定"""
    key: str
    display_name: str
    bank: int
    program: int
    gain: float = 1.0                    # 固定樂器校準增益（避免逐音峰值放大抽吸底噪與多音疊加破音）
    fixed_key: Optional[int] = None      # 無音高打擊樂器固定 GM 鼓組鍵位（如三角鐵/小鼓/大鼓）
    tail_sec: float = 0.38               # 放鍵後完整自然殘響尾音長度（秒）
    fade_out_sec: float = 0.12           # 尾端餘弦平滑收尾長度（秒），徹底消除截斷雜音
    overlap: bool = False                # 是否允許相鄰音符完全自由重疊殘響（鋼琴/馬林巴/打擊樂）
    octave_shift_above: Optional[int] = None  # 若高音譜號音符過高，自動移低八度以符合樂器真實音域


def _find_default_soundfont() -> Optional[Path]:
    """自動尋找內建 GeneralUser-GS.sf2（相容開發環境與 PyInstaller 打包環境）"""
    candidates: List[Path] = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "assets" / "soundfonts" / "GeneralUser-GS.sf2")
    root = Path(__file__).resolve().parent.parent.parent
    candidates.append(root / "assets" / "soundfonts" / "GeneralUser-GS.sf2")
    candidates.append(Path.cwd() / "assets" / "soundfonts" / "GeneralUser-GS.sf2")
    for p in candidates:
        if p.is_file():
            return p
    return None


class NoteSynthesizer:
    """跨平台真實取樣 SoundFont / NumPy 備援樂音合成播放器"""

    # 各樂器依 GeneralUser-GS 實測振幅校準固定增益，並內建樣本級餘弦釋放尾音，免除 SDL_mixer 強制切斷音軌之階梯雜音！
    INSTRUMENT_SPECS: Tuple[InstrumentSpec, ...] = (
        InstrumentSpec("piano", "鋼琴", 0, 0, gain=1.22, tail_sec=0.58, fade_out_sec=0.22, overlap=True),
        InstrumentSpec("sax", "薩克斯風", 0, 65, gain=1.42, tail_sec=0.18, fade_out_sec=0.14, overlap=True),
        InstrumentSpec("flute", "長笛", 0, 73, gain=1.20, tail_sec=0.18, fade_out_sec=0.14, overlap=True),
        InstrumentSpec("clarinet", "豎笛", 0, 71, gain=1.68, tail_sec=0.18, fade_out_sec=0.14, overlap=True),
        InstrumentSpec("oboe", "雙簧管", 0, 68, gain=1.32, tail_sec=0.18, fade_out_sec=0.14, overlap=True),
        InstrumentSpec("bassoon", "低音管", 0, 70, gain=1.68, tail_sec=0.20, fade_out_sec=0.15, overlap=True, octave_shift_above=64),
        InstrumentSpec("violin", "小提琴", 0, 40, gain=1.02, tail_sec=0.22, fade_out_sec=0.16, overlap=True),
        InstrumentSpec("cello", "大提琴", 0, 42, gain=1.05, tail_sec=0.24, fade_out_sec=0.17, overlap=True, octave_shift_above=68),
        InstrumentSpec("contrabass", "低音提琴", 0, 43, gain=1.20, tail_sec=0.26, fade_out_sec=0.18, overlap=True, octave_shift_above=60),
        InstrumentSpec("trumpet", "小號", 0, 56, gain=0.95, tail_sec=0.18, fade_out_sec=0.14, overlap=True),
        InstrumentSpec("french_horn", "法國號", 0, 60, gain=0.88, tail_sec=0.22, fade_out_sec=0.16, overlap=True),
        InstrumentSpec("organ", "管風琴", 0, 19, gain=2.10, tail_sec=0.20, fade_out_sec=0.15, overlap=True),
        InstrumentSpec("marimba", "馬林巴", 0, 12, gain=1.65, tail_sec=0.52, fade_out_sec=0.20, overlap=True),
        InstrumentSpec("timpani", "定音鼓", 0, 47, gain=1.85, tail_sec=0.68, fade_out_sec=0.24, overlap=True, octave_shift_above=58),
        InstrumentSpec("triangle", "三角鐵", 128, 0, gain=2.20, fixed_key=81, tail_sec=0.78, fade_out_sec=0.26, overlap=True),
        InstrumentSpec("snare", "小鼓", 128, 0, gain=1.40, fixed_key=38, tail_sec=0.45, fade_out_sec=0.16, overlap=True),
        InstrumentSpec("bass_drum", "大鼓", 128, 0, gain=0.82, fixed_key=36, tail_sec=0.52, fade_out_sec=0.18, overlap=True),
    )

    TIMBRES: Dict[str, str] = {spec.display_name: spec.key for spec in INSTRUMENT_SPECS}
    _SPECS_BY_KEY: Dict[str, InstrumentSpec] = {spec.key: spec for spec in INSTRUMENT_SPECS}

    def __init__(self, sample_rate: int = 44100, soundfont_path: Optional[str] = None):
        self._sample_rate = sample_rate
        self._timbre = "piano"
        self._volume = 0.85
        self._enabled = True
        self._cache: Dict[Tuple[str, int, int], any] = {}
        self._active_channel = None
        self._lock = threading.Lock()

        self._tsf_synth = None
        self._sfid: Optional[int] = None
        self._sf_path: Optional[Path] = None
        self._load_soundfont(soundfont_path)

    def _load_soundfont(self, soundfont_path: Optional[str] = None):
        """載入 SoundFont (.sf2) 音色庫"""
        if _tsf is None:
            return
        sf_file = Path(soundfont_path) if soundfont_path else _find_default_soundfont()
        if sf_file is None or not sf_file.is_file():
            return
        try:
            synth = _tsf.Synth(samplerate=self._sample_rate)
            sfid = synth.sfload(str(sf_file))
            self._tsf_synth = synth
            self._sfid = sfid
            self._sf_path = sf_file
        except Exception:
            self._tsf_synth = None
            self._sfid = None
            self._sf_path = None

    @property
    def using_soundfont(self) -> bool:
        return self._tsf_synth is not None and self._sfid is not None

    @property
    def soundfont_name(self) -> str:
        return self._sf_path.name if self._sf_path is not None else "NumPy Fallback"

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
        if key in self._SPECS_BY_KEY:
            if key != self._timbre:
                self.stop_all()
            self._timbre = key

    @staticmethod
    def midi_to_freq(midi_note: int, ref_a4: float = 440.0) -> float:
        """將 MIDI 編號轉為頻率 (Hz)"""
        return float(ref_a4 * (2.0 ** ((midi_note - 69) / 12.0)))

    def _synthesize_sf2(self, spec: InstrumentSpec, midi_note: int, dur: float) -> Optional[np.ndarray]:
        """透過 TinySoundFont 渲染無截斷雜音之真實立體聲 16-bit PCM 波形 (N, 2)"""
        if not self.using_soundfont:
            return None

        play_key = spec.fixed_key if spec.fixed_key is not None else int(midi_note)
        if spec.fixed_key is None and spec.octave_shift_above is not None:
            while play_key > spec.octave_shift_above and play_key - 12 >= 24:
                play_key -= 12
        play_key = max(21, min(108, play_key))

        chan = 9 if spec.bank == 128 else 0
        hold_n = max(1024, int(self._sample_rate * dur))
        tail_n = max(2048, int(self._sample_rate * spec.tail_sec))

        with self._lock:
            try:
                # 徹底清空前次內部濾波器與殘響狀態，確保起音零殘留相位跳變
                self._tsf_synth.sounds_off(chan)
                self._tsf_synth.program_select(chan, self._sfid, spec.bank, spec.program)
                self._tsf_synth.noteon(chan, play_key, 98)
                hold_raw = self._tsf_synth.generate(hold_n)
                self._tsf_synth.noteoff(chan, play_key)
                tail_raw = self._tsf_synth.generate(tail_n)
                self._tsf_synth.sounds_off(chan)
            except Exception:
                return None

        hold_arr = np.frombuffer(hold_raw, dtype=np.float32).reshape(-1, 2)
        tail_arr = np.frombuffer(tail_raw, dtype=np.float32).reshape(-1, 2)
        stereo = np.concatenate((hold_arr, tail_arr), axis=0).copy()

        # 消除極微弱直流偏移（DC offset），避免切換音符時喇叭振膜跳動產生咔噠聲
        stereo -= np.mean(stereo, axis=0, keepdims=True)

        # 起音前 3ms 餘弦平滑淡入（斜率由 0 起步，徹底消除取樣起點非零爆音）
        fade_in_n = min(stereo.shape[0] // 4, max(16, int(self._sample_rate * 0.003)))
        if fade_in_n > 1:
            t_in = np.linspace(0.0, np.pi, fade_in_n, dtype=np.float32)
            env_in = 0.5 * (1.0 - np.cos(t_in))
            stereo[:fade_in_n] *= env_in[:, None]

        # 尾端長餘弦平滑收尾（120~250ms raised-cosine 漸弱至 0，絕不生硬切斷樂器共鳴！）
        fade_out_n = min(stereo.shape[0] // 2, max(64, int(self._sample_rate * spec.fade_out_sec)))
        if fade_out_n > 1:
            t_out = np.linspace(0.0, np.pi, fade_out_n, dtype=np.float32)
            env_out = 0.5 * (1.0 + np.cos(t_out))
            stereo[-fade_out_n:] *= env_out[:, None]

        # 套用固定樂器校準增益 + 柔性雙曲正切軟限幅（保留多音疊加動態餘裕，杜絕 SDL_mixer 破音！）
        stereo *= spec.gain
        stereo = 0.62 * np.tanh(stereo / 0.62)

        return (stereo * 32767.0).astype(np.int16)

    def _synthesize_numpy_fallback(self, t_key: str, midi_note: int, dur: float) -> np.ndarray:
        """無 SoundFont 時之純 NumPy 備援合成（回傳單聲道 16-bit PCM）"""
        n_samples = int(self._sample_rate * dur)
        t = np.linspace(0.0, dur, n_samples, endpoint=False, dtype=np.float32)
        freq = self.midi_to_freq(midi_note)

        if t_key in ("triangle", "snare", "bass_drum"):
            rng = np.random.default_rng(42)
            noise = rng.standard_normal(n_samples).astype(np.float32)
            if t_key == "triangle":
                wave = (np.sin(2 * np.pi * 2600 * t) + 0.6 * np.sin(2 * np.pi * 4120 * t)) * np.exp(-6.0 * t)
            elif t_key == "snare":
                wave = (0.65 * noise + 0.35 * np.sin(2 * np.pi * 185 * t)) * np.exp(-14.0 * t)
            else:
                wave = np.sin(2 * np.pi * (85.0 * np.exp(-6.0 * t) + 42.0) * t) * np.exp(-8.0 * t)
        elif t_key in ("sax", "clarinet", "oboe", "bassoon", "trumpet", "french_horn"):
            vibrato = 1.0 + 0.003 * np.sin(2.0 * np.pi * 5.2 * t)
            phase = 2.0 * np.pi * freq * np.cumsum(vibrato) / self._sample_rate
            wave = (
                1.00 * np.sin(phase)
                + 0.60 * np.sin(2.0 * phase)
                + 0.38 * np.sin(3.0 * phase)
                + 0.20 * np.sin(4.0 * phase)
            )
            a_n = max(1, int(min(0.035, dur * 0.20) * self._sample_rate))
            r_n = max(1, int(min(0.06, dur * 0.25) * self._sample_rate))
            wave[:a_n] *= np.linspace(0.0, 1.0, a_n, dtype=np.float32)
            wave[-r_n:] *= np.linspace(1.0, 0.0, r_n, dtype=np.float32)
        elif t_key in ("flute", "organ"):
            phase = 2.0 * np.pi * freq * t
            wave = 1.00 * np.sin(phase) + 0.28 * np.sin(2.0 * phase) + 0.10 * np.sin(3.0 * phase)
            a_n = max(1, int(min(0.04, dur * 0.22) * self._sample_rate))
            r_n = max(1, int(min(0.07, dur * 0.25) * self._sample_rate))
            wave[:a_n] *= np.linspace(0.0, 1.0, a_n, dtype=np.float32)
            wave[-r_n:] *= np.linspace(1.0, 0.0, r_n, dtype=np.float32)
        else:
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
            wave = wave / peak * 0.50
        return (wave * 32767.0).astype(np.int16)

    def synthesize_wave(self, midi_note: int, duration_sec: float, timbre: Optional[str] = None) -> np.ndarray:
        """合成指定音高、時長與音色的 16-bit PCM 波形陣列（SF2 模式回傳立體聲 (N, 2)）"""
        t_key = self.TIMBRES.get(timbre, timbre) if timbre else self._timbre
        spec = self._SPECS_BY_KEY.get(t_key, self._SPECS_BY_KEY["piano"])
        dur = max(0.05, min(2.5, float(duration_sec)))

        sf2_wave = self._synthesize_sf2(spec, midi_note, dur)
        if sf2_wave is not None:
            return sf2_wave
        return self._synthesize_numpy_fallback(spec.key, midi_note, dur)

    def play_note(self, midi_note: int, duration_sec: float = 0.35):
        """即時無縫播放指定 MIDI 音符"""
        if not self._enabled or _audio_backend != "pygame":
            return
        if midi_note < 21 or midi_note > 108:
            return

        spec = self._SPECS_BY_KEY.get(self._timbre, self._SPECS_BY_KEY["piano"])
        # 量化時長至 25ms 精度（最低 50ms 以準確表現短促裝飾音，不再拖長互撞！）
        dur_bucket = max(50, min(2200, int(round(duration_sec * 1000 / 25) * 25)))
        cache_key = (spec.key, int(midi_note), dur_bucket)

        snd = self._cache.get(cache_key)
        if snd is None:
            pcm = self.synthesize_wave(midi_note, dur_bucket / 1000.0, spec.key)
            try:
                init_info = pygame.mixer.get_init()
                mixer_channels = init_info[2] if init_info else 2
                if mixer_channels == 2 and pcm.ndim == 1:
                    pcm = np.column_stack((pcm, pcm))
                elif mixer_channels == 1 and pcm.ndim == 2:
                    pcm = np.mean(pcm.astype(np.float32), axis=1).astype(np.int16)
                snd = pygame.sndarray.make_sound(np.ascontiguousarray(pcm))
                if len(self._cache) > 512:
                    self._cache.clear()
                self._cache[cache_key] = snd
            except Exception:
                return

        try:
            # 直接在獨立音軌播放已內建樣本級餘弦釋放尾音之波形（完全不呼叫 SDL Mix_FadeOutChannel，100% 杜絕階梯截斷雜音！）
            snd.set_volume(self._volume)
            ch = pygame.mixer.find_channel(True)
            if ch is not None:
                ch.play(snd)
                self._active_channel = ch
            else:
                self._active_channel = snd.play()
        except Exception:
            pass

    def stop_all(self):
        """平滑淡出停止目前播放中的樂音"""
        if _audio_backend == "pygame":
            try:
                pygame.mixer.fadeout(60)
                self._active_channel = None
            except Exception:
                pass
