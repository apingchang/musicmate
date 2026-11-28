"""符號樂譜載入與高解析大譜表渲染引擎 (SymbolicScoreLoader)

原生支援 MuseScore 網站匯出的全部符號格式，免外部軟體依賴即可 100% 精準解析雙手大譜表、調號、拍號、低音譜號與和弦：
- MusicXML (.musicxml, .xml, .mxl)
- MuseScore (.mscz, .mscx)
- MIDI (.mid, .midi)

並透過 PyQt6 QPainter 抗鋸齒向量繪製高解析 A4 鋼琴雙手大譜表 / 單行樂譜頁面，
直接產生含精準歸一化座標與雙手同步時間軸之 ScoreAnalysis，無縫銜接自動捲動、紅線游標與 SoundFont 真實樂器發聲。
"""

from dataclasses import dataclass, field
import math
import os
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QBrush

from src.core.score_analyzer import (
    MeasureInfo,
    NoteEvent,
    PageAnalysis,
    PlaybackMeasure,
    ScoreAnalysis,
    StaffInfo,
)

SUPPORTED_SYMBOLIC_EXTS = (".musicxml", ".xml", ".mxl", ".mscz", ".mscx", ".mid", ".midi")

# 調號升降順序（以音名 C=0, D=1, E=2, F=3, G=4, A=5, B=6 表示）
_SHARP_ORDER = (3, 0, 4, 1, 5, 2, 6)  # F, C, G, D, A, E, B
_FLAT_ORDER = (6, 2, 5, 1, 4, 0, 3)   # B, E, A, D, G, C, F
_PC_TO_LETTER_SHARP = (
    (0, 0), (0, 1), (1, 0), (1, 1), (2, 0),
    (3, 0), (3, 1), (4, 0), (4, 1), (5, 0), (5, 1), (6, 0),
)
_PC_TO_LETTER_FLAT = (
    (0, 0), (1, -1), (1, 0), (2, -1), (2, 0),
    (3, 0), (4, -1), (4, 0), (5, -1), (5, 0), (6, -1), (6, 0),
)
_LETTER_NAMES = ("C", "D", "E", "F", "G", "A", "B")
_LETTER_SEMITONES = (0, 2, 4, 5, 7, 9, 11)


def key_signature_map(fifths: int) -> Dict[int, int]:
    """回傳指定調號 (-7..+7) 對應各音名 (0=C..6=B) 的預設升降值 (+1 / -1 / 0)"""
    m = {i: 0 for i in range(7)}
    f = max(-7, min(7, int(fifths)))
    if f > 0:
        for i in range(f):
            m[_SHARP_ORDER[i]] = 1
    elif f < 0:
        for i in range(-f):
            m[_FLAT_ORDER[i]] = -1
    return m


def format_key_signature_label(fifths: int) -> str:
    """將調號轉換為友善顯示標籤（如 '4♯ (E大調/C♯小調)'）"""
    f = max(-7, min(7, int(fifths)))
    names = {
        0: "C大調 / A小調",
        1: "1♯ (G大調 / E小調)",
        2: "2♯ (D大調 / B小調)",
        3: "3♯ (A大調 / F♯小調)",
        4: "4♯ (E大調 / C♯小調)",
        5: "5♯ (B大調 / G♯小調)",
        6: "6♯ (F♯大調 / D♯小調)",
        7: "7♯ (C♯大調 / A♯小調)",
        -1: "1♭ (F大調 / D小調)",
        -2: "2♭ (B♭大調 / G小調)",
        -3: "3♭ (E♭大調 / C小調)",
        -4: "4♭ (A♭大調 / F小調)",
        -5: "5♭ (D♭大調 / B♭小調)",
        -6: "6♭ (G♭大調 / E♭小調)",
        -7: "7♭ (C♭大調 / A♭小調)",
    }
    return names.get(f, "C大調")


def midi_to_note_name(midi_val: int, key_fifths: int = 0) -> str:
    """依調號偏好將 MIDI 音高轉為精準音名（如 C#4 / Bb3）"""
    octave = (int(midi_val) // 12) - 1
    pc = int(midi_val) % 12
    table = _PC_TO_LETTER_FLAT if key_fifths < 0 else _PC_TO_LETTER_SHARP
    letter_idx, acc = table[pc]
    acc_str = "#" if acc > 0 else ("b" if acc < 0 else "")
    return f"{_LETTER_NAMES[letter_idx]}{acc_str}{octave}"


def midi_to_staff_step(midi_val: int, clef: str, key_fifths: int = 0) -> Tuple[int, int]:
    """計算 MIDI 音高在指定譜號上的半間階數 (0 = 底線: 高音 E4 / 低音 G2) 與需顯示的臨時記號"""
    octave = (int(midi_val) // 12) - 1
    pc = int(midi_val) % 12
    table = _PC_TO_LETTER_FLAT if key_fifths < 0 else _PC_TO_LETTER_SHARP
    letter_idx, acc = table[pc]
    ks_map = key_signature_map(key_fifths)
    # 若目標音剛好吻合調號（例如 4♯ 中的 C#），不需額外臨時記號
    if acc != ks_map[letter_idx]:
        # 嘗試檢查相鄰等音是否剛好吻合調號
        for cand_l in range(7):
            cand_pc = (_LETTER_SEMITONES[cand_l] + ks_map[cand_l]) % 12
            if cand_pc == pc:
                # 處理跨八度邊界 (B# / Cb)
                cand_oct = octave
                if cand_l == 6 and ks_map[cand_l] == 1 and pc == 0:
                    cand_oct -= 1
                elif cand_l == 0 and ks_map[cand_l] == -1 and pc == 11:
                    cand_oct += 1
                letter_idx, acc, octave = cand_l, ks_map[cand_l], cand_oct
                break

    abs_diatonic = octave * 7 + letter_idx
    base_diatonic = (4 * 7 + 2) if clef == "treble" else (2 * 7 + 4)  # E4=30, G2=18
    step = abs_diatonic - base_diatonic
    disp_acc = 0 if acc == ks_map[letter_idx] else (2 if acc == 0 else acc)  # 2 表示還原記號 ♮
    return step, disp_acc


@dataclass
class _SymEvent:
    local_beat: float
    duration_beats: float
    midi_notes: List[int] = field(default_factory=list)
    is_rest: bool = False
    is_grace: bool = False
    is_dotted: bool = False
    beam_count: int = 0
    note_type: str = ""


@dataclass
class _SymPart:
    name: str = ""
    clef: str = "treble"
    hand: str = ""
    timbre: str = ""


@dataclass
class _SymMeasure:
    measure_number: int
    duration_beats: float
    rh_events: List[_SymEvent] = field(default_factory=list)
    lh_events: List[_SymEvent] = field(default_factory=list)
    part_events: List[List[_SymEvent]] = field(default_factory=list)


@dataclass
class _SymScore:
    title: str = ""
    composer: str = ""
    time_sig: str = "4/4"
    beats_per_measure: float = 4.0
    bpm: int = 96
    key_fifths: int = 0
    is_grand_staff: bool = False
    parts: List[_SymPart] = field(default_factory=list)
    measures: List[_SymMeasure] = field(default_factory=list)


def _guess_instrument_timbre(part_name: str) -> str:
    """依樂器聲部名稱自動對應內建 SoundFont 真實樂器音色"""
    low = (part_name or "").lower()
    mapping = (
        ("violin", "小提琴"), ("小提琴", "小提琴"),
        ("cello", "大提琴"), ("violoncello", "大提琴"), ("大提琴", "大提琴"),
        ("contrabass", "低音提琴"), ("double bass", "低音提琴"), ("低音提琴", "低音提琴"),
        ("flute", "長笛"), ("長笛", "長笛"),
        ("clarinet", "豎笛"), ("豎笛", "豎笛"), ("單簧管", "豎笛"),
        ("oboe", "雙簧管"), ("雙簧管", "雙簧管"),
        ("bassoon", "低音管"), ("低音管", "低音管"),
        ("sax", "薩克斯風"), ("薩克斯", "薩克斯風"),
        ("trumpet", "小號"), ("小號", "小號"),
        ("horn", "法國號"), ("法國號", "法國號"),
        ("organ", "管風琴"), ("管風琴", "管風琴"),
        ("marimba", "馬林巴"), ("馬林巴", "馬林巴"),
        ("timpani", "定音鼓"), ("定音鼓", "定音鼓"),
        ("piano", "鋼琴"), ("鋼琴", "鋼琴"),
    )
    for kw, zh in mapping:
        if kw in low:
            return zh
    return ""


def _duration_to_meta(dur_beats: float, is_rest: bool = False, is_grace: bool = False) -> Tuple[str, bool, int, bool]:
    """將拍數轉為 (中文音符名稱, is_hollow, beam_count, is_dotted)"""
    if is_grace:
        return "裝飾音 (倚音)", False, 2, False
    if is_rest:
        if dur_beats >= 3.5:
            return "全休止符", True, 0, False
        if dur_beats >= 1.75:
            return "二分休止符", True, 0, False
        if dur_beats >= 0.85:
            return "四分休止符", False, 0, False
        return "八分休止符", False, 1, False

    if dur_beats >= 3.5:
        return "全音符 (1/1)", True, 0, False
    if dur_beats >= 2.6:
        return "附點二分音符 (3/4)", True, 0, True
    if dur_beats >= 1.75:
        return "二分音符 (1/2)", True, 0, False
    if dur_beats >= 1.35:
        return "附點四分音符 (3/8)", False, 0, True
    if dur_beats >= 0.85:
        return "四分音符 (1/4)", False, 0, False
    if dur_beats >= 0.68:
        return "附點八分音符 (3/16)", False, 1, True
    if dur_beats >= 0.30:
        if 0.30 <= dur_beats <= 0.38:
            return "八分三連音 (1/3拍)", False, 1, False
        return "八分音符 (1/8)", False, 1, False
    return "十六分音符 (1/16)", False, 2, False


def _merge_simultaneous_events(raw_list: List[_SymEvent]) -> List[_SymEvent]:
    """將同一手於完全相同起拍點 (Δbeat <= 0.02) 之多顆音符整合為單一和弦事件（同時保留各聲部獨立起拍音）"""
    if not raw_list:
        return []
    raw_list.sort(key=lambda e: (round(e.local_beat, 4), 0 if e.is_grace else 1, e.is_rest))
    merged: List[_SymEvent] = []
    for ev in raw_list:
        if ev.is_rest:
            merged.append(ev)
            continue
        if (
            merged
            and not merged[-1].is_rest
            and merged[-1].is_grace == ev.is_grace
            and abs(merged[-1].local_beat - ev.local_beat) <= 0.025
        ):
            # 若兩音起拍完全相同，合併為和弦並取較長時值作為代表（讓長音與短音都能完整發聲）
            prev = merged[-1]
            prev.midi_notes = sorted(set(prev.midi_notes + ev.midi_notes))
            prev.duration_beats = max(prev.duration_beats, ev.duration_beats)
        else:
            merged.append(ev)
    # 若小節內已有實體音符，過濾與實體音符完全重疊的多聲部休止符
    has_notes = any(not e.is_rest for e in merged)
    if has_notes:
        note_beats = [e.local_beat for e in merged if not e.is_rest]
        merged = [
            e for e in merged
            if (not e.is_rest) or all(abs(e.local_beat - nb) > 0.05 for nb in note_beats)
        ]
    return merged


class SymbolicScoreLoader:
    """載入 MusicXML / MuseScore (.mscz/.mscx) / MIDI 並渲染為高解析大譜表樂譜頁面與 ScoreAnalysis"""

    PAGE_W = 1654
    PAGE_H = 2339

    @classmethod
    def is_symbolic_file(cls, file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower() in SUPPORTED_SYMBOLIC_EXTS

    @classmethod
    def load(cls, file_path: str) -> Tuple[List[np.ndarray], ScoreAnalysis]:
        """解析符號樂譜並回傳 (高解析 RGB 頁面陣列清單, 完整 ScoreAnalysis)"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".mscz", ".mscx"):
            sym = cls._parse_musescore(file_path)
        elif ext in (".musicxml", ".xml", ".mxl"):
            sym = cls._parse_musicxml(file_path)
        elif ext in (".mid", ".midi"):
            sym = cls._parse_midi(file_path)
        else:
            raise ValueError(f"不支援的符號樂譜格式：{ext}")

        if not sym.measures:
            raise ValueError("檔案中未找到任何可播放的小節或音符。")

        return cls._render_and_build_analysis(sym, source_format=ext.lstrip("."))

    # ------------------------------------------------------------------
    # 1. MusicXML (.musicxml / .xml / .mxl) 解析（使用 music21）
    # ------------------------------------------------------------------
    @classmethod
    def _parse_musicxml(cls, file_path: str) -> _SymScore:
        import music21

        score = music21.converter.parse(file_path)
        sym = _SymScore()
        if score.metadata:
            sym.title = str(score.metadata.title or "")
            sym.composer = str(score.metadata.composer or "")
        if not sym.title:
            sym.title = os.path.splitext(os.path.basename(file_path))[0]

        # 拍號、調號與速度
        ts_list = list(score.recurse().getElementsByClass(music21.meter.TimeSignature))
        if ts_list:
            ts0 = ts_list[0]
            sym.time_sig = f"{ts0.numerator}/{ts0.denominator}"
            sym.beats_per_measure = max(1.0, float(ts0.barDuration.quarterLength))
        ks_list = list(score.recurse().getElementsByClass(music21.key.KeySignature))
        if ks_list:
            sym.key_fifths = int(ks_list[0].sharps)
        mm_list = list(score.recurse().getElementsByClass(music21.tempo.MetronomeMark))
        if mm_list:
            bpm_val = mm_list[0].getQuarterBPM()
            if bpm_val and 25 <= bpm_val <= 280:
                sym.bpm = int(round(bpm_val))

        parts = list(score.parts)
        if not parts:
            parts = [score]
        parts = parts[:8]  # 單系統上限 8 行確保 A4 譜面清晰可讀

        n_parts = len(parts)
        sym.is_grand_staff = n_parts >= 2

        # 偵測各聲部名稱、譜號與樂器音色（支援 2 行鋼琴大譜表及 3~8 行多種樂器重奏總譜！）
        raw_part_names = [str(getattr(pt, "partName", "") or "").strip() for pt in parts]
        is_piano_two = (
            n_parts == 2
            and (
                not any(raw_part_names)
                or all(("piano" in nm.lower() or "rh" in nm.lower() or "lh" in nm.lower() or "鋼琴" in nm or nm == "") for nm in raw_part_names)
            )
        )
        sym_parts: List[_SymPart] = []
        part_measures_list: List[list] = []
        for p_i, pt in enumerate(parts):
            ms_p = list(pt.getElementsByClass(music21.stream.Measure))
            part_measures_list.append(ms_p)
            clefs_p = list(pt.recurse().getElementsByClass(music21.clef.Clef))
            clef_str = "bass" if (clefs_p and "Bass" in clefs_p[0].name) else ("bass" if (is_piano_two and p_i == 1) else "treble")
            if n_parts == 1:
                p_name, h_tag, imb = raw_part_names[0], "", _guess_instrument_timbre(raw_part_names[0])
            elif is_piano_two:
                p_name = "右手 RH" if p_i == 0 else "左手 LH"
                h_tag = "RH" if p_i == 0 else "LH"
                imb = "鋼琴"
            else:
                p_name = raw_part_names[p_i] or f"樂器 {p_i + 1}"
                h_tag = f"P{p_i + 1}"
                imb = _guess_instrument_timbre(p_name)
            sym_parts.append(_SymPart(name=p_name, clef=clef_str, hand=h_tag, timbre=imb))
        sym.parts = sym_parts

        n_m = max((len(ml) for ml in part_measures_list), default=0)

        def extract_measure_events(m_obj) -> List[_SymEvent]:
            if m_obj is None:
                return []
            evs: List[_SymEvent] = []
            voices = list(m_obj.voices)
            containers = voices if voices else [m_obj]
            for cont in containers:
                for el in cont.notesAndRests:
                    # 跳過連結線延續音（只發起音，避免重複發聲）
                    if getattr(el, "tie", None) is not None and el.tie.type in ("stop", "continue"):
                        continue
                    dur_q = float(el.quarterLength)
                    is_grace = dur_q <= 1e-6 or bool(getattr(el.duration, "isGrace", False))
                    if is_grace:
                        dur_q = 0.16
                    off_q = float(el.offset)
                    if el.isRest:
                        evs.append(_SymEvent(local_beat=off_q, duration_beats=dur_q, is_rest=True))
                    elif el.isChord:
                        mids = sorted({int(p.midi) for p in el.pitches if 21 <= p.midi <= 108})
                        if mids:
                            evs.append(_SymEvent(local_beat=off_q, duration_beats=dur_q, midi_notes=mids, is_grace=is_grace))
                    else:
                        mv = int(el.pitch.midi)
                        if 21 <= mv <= 108:
                            evs.append(_SymEvent(local_beat=off_q, duration_beats=dur_q, midi_notes=[mv], is_grace=is_grace))
            return _merge_simultaneous_events(evs)

        if n_m > 0:
            for idx in range(n_m):
                p_evs_per_part = [
                    extract_measure_events(part_measures_list[p_i][idx] if idx < len(part_measures_list[p_i]) else None)
                    for p_i in range(n_parts)
                ]
                m_dur = sym.beats_per_measure
                rm0 = part_measures_list[0][idx] if idx < len(part_measures_list[0]) else None
                if rm0 is not None and float(rm0.barDuration.quarterLength) > 0:
                    m_dur = float(rm0.barDuration.quarterLength)
                sym.measures.append(
                    _SymMeasure(
                        measure_number=idx + 1,
                        duration_beats=m_dur,
                        rh_events=p_evs_per_part[0] if n_parts >= 1 else [],
                        lh_events=p_evs_per_part[1] if n_parts >= 2 else [],
                        part_events=p_evs_per_part,
                    )
                )
        else:
            # 無小節容器時依拍數自動切小節
            sym = cls._slice_flat_music21_stream(score, sym)

        return sym

    @classmethod
    def _slice_flat_music21_stream(cls, score, sym: _SymScore) -> _SymScore:
        import music21

        parts = list(score.parts) or [score]
        rh_all: List[Tuple[float, float, List[int]]] = []
        lh_all: List[Tuple[float, float, List[int]]] = []
        if len(parts) >= 2:
            sym.is_grand_staff = True
            for el in parts[0].flatten().notes:
                mids = [int(p.midi) for p in el.pitches] if el.isChord else [int(el.pitch.midi)]
                rh_all.append((float(el.offset), max(0.12, float(el.quarterLength)), mids))
            for el in parts[1].flatten().notes:
                mids = [int(p.midi) for p in el.pitches] if el.isChord else [int(el.pitch.midi)]
                lh_all.append((float(el.offset), max(0.12, float(el.quarterLength)), mids))
        else:
            all_mids = []
            for el in parts[0].flatten().notes:
                mids = [int(p.midi) for p in el.pitches] if el.isChord else [int(el.pitch.midi)]
                all_mids.extend(mids)
            has_bass = any(m < 55 for m in all_mids) and any(m >= 62 for m in all_mids)
            sym.is_grand_staff = has_bass
            for el in parts[0].flatten().notes:
                mids = [int(p.midi) for p in el.pitches] if el.isChord else [int(el.pitch.midi)]
                off = float(el.offset)
                dur = max(0.12, float(el.quarterLength))
                if has_bass:
                    rh_m = [m for m in mids if m >= 60]
                    lh_m = [m for m in mids if m < 60]
                    if rh_m:
                        rh_all.append((off, dur, rh_m))
                    if lh_m:
                        lh_all.append((off, dur, lh_m))
                else:
                    rh_all.append((off, dur, mids))

        max_beat = 0.0
        for off, dur, _ in rh_all + lh_all:
            max_beat = max(max_beat, off + dur)
        bpm_len = max(1.0, sym.beats_per_measure)
        n_m = max(1, int(math.ceil(max_beat / bpm_len)))
        measures = [_SymMeasure(measure_number=i + 1, duration_beats=bpm_len) for i in range(n_m)]
        for off, dur, mids in rh_all:
            mi = min(n_m - 1, max(0, int(off // bpm_len)))
            measures[mi].rh_events.append(_SymEvent(local_beat=off - mi * bpm_len, duration_beats=dur, midi_notes=sorted(set(mids))))
        for off, dur, mids in lh_all:
            mi = min(n_m - 1, max(0, int(off // bpm_len)))
            measures[mi].lh_events.append(_SymEvent(local_beat=off - mi * bpm_len, duration_beats=dur, midi_notes=sorted(set(mids))))
        for m in measures:
            m.rh_events = _merge_simultaneous_events(m.rh_events)
            m.lh_events = _merge_simultaneous_events(m.lh_events)
        sym.measures = measures
        return sym

    # ------------------------------------------------------------------
    # 2. MIDI (.mid / .midi) 解析（使用 mido 精準解讀多軌鋼琴）
    # ------------------------------------------------------------------
    @classmethod
    def _parse_midi(cls, file_path: str) -> _SymScore:
        import mido

        mid = mido.MidiFile(file_path)
        tpb = max(1, int(mid.ticks_per_beat))
        sym = _SymScore(title=os.path.splitext(os.path.basename(file_path))[0])

        # 解析所有音軌之音符 (track_idx, start_beat, dur_beat, midi_pitch)
        track_notes: List[List[Tuple[float, float, int]]] = []
        found_tempo = False
        found_ts = False
        found_ks = False
        key_name_to_fifths = {
            "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
            "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
            "Am": 0, "Em": 1, "Bm": 2, "F#m": 3, "C#m": 4, "G#m": 5, "D#m": 6, "A#m": 7,
            "Dm": -1, "Gm": -2, "Cm": -3, "Fm": -4, "Bbm": -5, "Ebm": -6, "Abm": -7,
        }

        for tr in mid.tracks:
            abs_tick = 0
            active: Dict[Tuple[int, int], int] = {}
            notes_in_tr: List[Tuple[float, float, int]] = []
            for msg in tr:
                abs_tick += msg.time
                if msg.is_meta:
                    if msg.type == "set_tempo" and not found_tempo:
                        bpm_v = mido.tempo2bpm(msg.tempo)
                        if 25 <= bpm_v <= 280:
                            sym.bpm = int(round(bpm_v))
                            found_tempo = True
                    elif msg.type == "time_signature" and not found_ts:
                        sym.time_sig = f"{msg.numerator}/{msg.denominator}"
                        sym.beats_per_measure = max(1.0, float(msg.numerator) * (4.0 / float(msg.denominator)))
                        found_ts = True
                    elif msg.type == "key_signature" and not found_ks:
                        sym.key_fifths = key_name_to_fifths.get(str(msg.key), 0)
                        found_ks = True
                elif msg.type == "note_on" and msg.velocity > 0:
                    key_nc = (getattr(msg, "channel", 0), msg.note)
                    if key_nc not in active:
                        active[key_nc] = abs_tick
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    key_nc = (getattr(msg, "channel", 0), msg.note)
                    st_tick = active.pop(key_nc, None)
                    if st_tick is not None and 21 <= msg.note <= 108:
                        st_b = st_tick / float(tpb)
                        dur_b = max(0.10, (abs_tick - st_tick) / float(tpb))
                        notes_in_tr.append((st_b, dur_b, int(msg.note)))
            if notes_in_tr:
                track_notes.append(notes_in_tr)

        if not track_notes:
            raise ValueError("MIDI 檔案中沒有音符資料。")

        rh_raw: List[Tuple[float, float, int]] = []
        lh_raw: List[Tuple[float, float, int]] = []
        if len(track_notes) >= 2:
            # 依平均音高排序，較高音軌為右手、較低音軌為左手
            sorted_trs = sorted(track_notes, key=lambda lst: float(np.mean([n[2] for n in lst])), reverse=True)
            rh_raw = sorted_trs[0]
            for tr_l in sorted_trs[1:]:
                lh_raw.extend(tr_l)
            sym.is_grand_staff = True
        else:
            pitches = [n[2] for n in track_notes[0]]
            if min(pitches) < 55 and max(pitches) >= 62:
                sym.is_grand_staff = True
                for st_b, dur_b, p in track_notes[0]:
                    if p >= 60:
                        rh_raw.append((st_b, dur_b, p))
                    else:
                        lh_raw.append((st_b, dur_b, p))
            else:
                sym.is_grand_staff = False
                rh_raw = track_notes[0]

        max_b = max((st + du for st, du, _ in rh_raw + lh_raw), default=sym.beats_per_measure)
        bpm_len = max(1.0, sym.beats_per_measure)
        n_m = max(1, int(math.ceil(max_b / bpm_len)))
        measures = [_SymMeasure(measure_number=i + 1, duration_beats=bpm_len) for i in range(n_m)]

        for st_b, dur_b, p in rh_raw:
            mi = min(n_m - 1, max(0, int(st_b // bpm_len)))
            measures[mi].rh_events.append(_SymEvent(local_beat=st_b - mi * bpm_len, duration_beats=dur_b, midi_notes=[p]))
        for st_b, dur_b, p in lh_raw:
            mi = min(n_m - 1, max(0, int(st_b // bpm_len)))
            measures[mi].lh_events.append(_SymEvent(local_beat=st_b - mi * bpm_len, duration_beats=dur_b, midi_notes=[p]))

        for m in measures:
            m.rh_events = _merge_simultaneous_events(m.rh_events)
            m.lh_events = _merge_simultaneous_events(m.lh_events)

        sym.measures = measures
        return sym

    # ------------------------------------------------------------------
    # 3. MuseScore (.mscz / .mscx) 原生 XML 解析（相容 MuseScore 3 & 4，免安裝 MuseScore）
    # ------------------------------------------------------------------
    @classmethod
    def _parse_musescore(cls, file_path: str) -> _SymScore:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".mscz":
            with zipfile.ZipFile(file_path, "r") as zf:
                mscx_names = [n for n in zf.namelist() if n.lower().endswith(".mscx")]
                if not mscx_names:
                    raise ValueError("無效的 .mscz 檔案：壓縮檔內找不到 .mscx 樂譜 XML。")
                xml_bytes = zf.read(mscx_names[0])
            root = ET.fromstring(xml_bytes)
        else:
            root = ET.parse(file_path).getroot()

        score_el = root.find("Score")
        if score_el is None:
            score_el = root

        sym = _SymScore(title=os.path.splitext(os.path.basename(file_path))[0])
        for meta in score_el.findall("metaTag"):
            name = meta.attrib.get("name", "")
            val = (meta.text or "").strip()
            if name == "workTitle" and val:
                sym.title = val
            elif name == "composer" and val:
                sym.composer = val

        # 尋找實際含 <Measure> 的 <Staff id="...">（跳過 <Part> 內的結構宣告 <Staff>）
        music_staves = [st for st in score_el.findall("Staff") if st.find("Measure") is not None]
        if not music_staves:
            raise ValueError("MuseScore 檔案內未找到任何樂譜小節。")

        sym.is_grand_staff = len(music_staves) >= 2
        dur_map = {
            "long": 16.0, "breve": 8.0, "whole": 4.0, "half": 2.0,
            "quarter": 1.0, "eighth": 0.5, "16th": 0.25, "32nd": 0.125, "64th": 0.0625,
            "measure": -1.0,
        }

        def parse_staff_measures(st_el, is_primary: bool) -> List[List[_SymEvent]]:
            out_measures: List[List[_SymEvent]] = []
            for m_el in st_el.findall("Measure"):
                m_evs: List[_SymEvent] = []
                voices = m_el.findall("voice")
                if not voices:
                    voices = [m_el]
                for v_el in voices:
                    cur_b = 0.0
                    tuplet_ratio = 1.0
                    tuplet_remaining = 0.0
                    for child in list(v_el):
                        tag = child.tag
                        if tag == "KeySig" and is_primary:
                            acc_el = child.find("concertKey")
                            if acc_el is None:
                                acc_el = child.find("accidental")
                            if acc_el is not None and acc_el.text:
                                try:
                                    sym.key_fifths = int(acc_el.text.strip())
                                except ValueError:
                                    pass
                        elif tag == "TimeSig" and is_primary:
                            sig_n = child.findtext("sigN")
                            sig_d = child.findtext("sigD")
                            if sig_n and sig_d:
                                try:
                                    nn, dd = int(sig_n), int(sig_d)
                                    sym.time_sig = f"{nn}/{dd}"
                                    sym.beats_per_measure = max(1.0, float(nn) * (4.0 / float(dd)))
                                except ValueError:
                                    pass
                        elif tag == "Tempo" and is_primary:
                            t_txt = child.findtext("tempo")
                            if t_txt:
                                try:
                                    bpm_v = float(t_txt) * 60.0
                                    if 25 <= bpm_v <= 280:
                                        sym.bpm = int(round(bpm_v))
                                except ValueError:
                                    pass
                        elif tag == "Tuplet":
                            nn = child.findtext("normalNotes")
                            an = child.findtext("actualNotes")
                            bd = child.findtext("baseNote")
                            if nn and an:
                                try:
                                    n_f, a_f = float(nn), float(an)
                                    if a_f > 0:
                                        tuplet_ratio = n_f / a_f
                                        base_q = dur_map.get(bd or "eighth", 0.5)
                                        tuplet_remaining = a_f * base_q
                                except ValueError:
                                    pass
                        elif tag == "endTuplet":
                            tuplet_ratio = 1.0
                            tuplet_remaining = 0.0
                        elif tag in ("Chord", "Rest"):
                            dt_str = (child.findtext("durationType") or "quarter").strip()
                            base_d = dur_map.get(dt_str, 1.0)
                            if base_d < 0:
                                base_d = sym.beats_per_measure
                            dots = int(child.findtext("dots") or "0")
                            d_val = base_d
                            add_d = base_d * 0.5
                            for _ in range(dots):
                                d_val += add_d
                                add_d *= 0.5
                            if tuplet_remaining > 1e-6:
                                eff_d = d_val * tuplet_ratio
                                tuplet_remaining -= d_val
                                if tuplet_remaining <= 1e-4:
                                    tuplet_ratio = 1.0
                                    tuplet_remaining = 0.0
                            else:
                                eff_d = d_val

                            is_grace = (
                                child.find("appoggiatura") is not None
                                or child.find("acciaccatura") is not None
                                or child.find("grace8") is not None
                                or child.find("grace16") is not None
                            )
                            if tag == "Rest":
                                m_evs.append(_SymEvent(local_beat=cur_b, duration_beats=eff_d, is_rest=True))
                                cur_b += eff_d
                            else:
                                mids: List[int] = []
                                for n_el in child.findall("Note"):
                                    # 忽略連結線的結束端
                                    is_tied_stop = False
                                    for sp_el in n_el.findall(".//Spanner"):
                                        if sp_el.attrib.get("type") == "Tie" and sp_el.find("prev") is not None:
                                            is_tied_stop = True
                                    if is_tied_stop:
                                        continue
                                    p_txt = n_el.findtext("pitch")
                                    if p_txt:
                                        try:
                                            mv = int(p_txt.strip())
                                            if 21 <= mv <= 108:
                                                mids.append(mv)
                                        except ValueError:
                                            pass
                                if mids:
                                    m_evs.append(
                                        _SymEvent(
                                            local_beat=cur_b,
                                            duration_beats=0.16 if is_grace else eff_d,
                                            midi_notes=sorted(set(mids)),
                                            is_grace=is_grace,
                                            is_dotted=(dots > 0),
                                        )
                                    )
                                if not is_grace:
                                    cur_b += eff_d
                out_measures.append(_merge_simultaneous_events(m_evs))
            return out_measures

        # 讀取 <Part> 宣告的樂器名稱與多行總譜（相容 1 行獨奏、2 行鋼琴與 3~8 行重奏總譜）
        staff_part_names: List[str] = []
        for p_el in score_el.findall("Part"):
            t_name = (p_el.findtext("trackName") or p_el.findtext("Instrument/longName") or "").strip()
            n_st_in_p = max(1, len(p_el.findall("Staff")))
            for k_sp in range(n_st_in_p):
                staff_part_names.append(t_name)

        use_staves = music_staves[:8]
        n_st = len(use_staves)
        sym.is_grand_staff = n_st >= 2
        all_staff_lists = [parse_staff_measures(st_e, is_primary=(i == 0)) for i, st_e in enumerate(use_staves)]
        is_piano_two = (
            n_st == 2
            and (
                not any(staff_part_names[:2])
                or all(("piano" in nm.lower() or "鋼琴" in nm or nm == "") for nm in staff_part_names[:2])
            )
        )
        sym.parts = []
        for p_i in range(n_st):
            raw_nm = staff_part_names[p_i] if p_i < len(staff_part_names) else ""
            if n_st == 1:
                p_name, h_tag, clef_s, imb = raw_nm, "", "treble", _guess_instrument_timbre(raw_nm)
            elif is_piano_two:
                p_name = "右手 RH" if p_i == 0 else "左手 LH"
                h_tag = "RH" if p_i == 0 else "LH"
                clef_s = "treble" if p_i == 0 else "bass"
                imb = "鋼琴"
            else:
                p_name = raw_nm or f"樂器 {p_i + 1}"
                h_tag = f"P{p_i + 1}"
                # 依該聲部平均音高自動選高音或低音譜號
                p_mids = [m for ms in all_staff_lists[p_i] for ev in ms for m in ev.midi_notes]
                clef_s = "bass" if (p_mids and float(np.mean(p_mids)) < 57.0) else "treble"
                imb = _guess_instrument_timbre(p_name)
            sym.parts.append(_SymPart(name=p_name, clef=clef_s, hand=h_tag, timbre=imb))

        n_m = max((len(lst) for lst in all_staff_lists), default=0)
        for i in range(n_m):
            pe = [all_staff_lists[p_i][i] if i < len(all_staff_lists[p_i]) else [] for p_i in range(n_st)]
            sym.measures.append(
                _SymMeasure(
                    measure_number=i + 1,
                    duration_beats=sym.beats_per_measure,
                    rh_events=pe[0] if n_st >= 1 else [],
                    lh_events=pe[1] if n_st >= 2 else [],
                    part_events=pe,
                )
            )
        return sym

    # ------------------------------------------------------------------
    # 4. 高解析 A4 向量五線譜繪製 + 精準座標 ScoreAnalysis 建構
    # ------------------------------------------------------------------
    @classmethod
    def _render_and_build_analysis(cls, sym: _SymScore, source_format: str) -> Tuple[List[np.ndarray], ScoreAnalysis]:
        W, H = cls.PAGE_W, cls.PAGE_H
        # 若未建立 sym.parts（例如 MIDI 單/雙軌），自動補齊標準 1 行或 2 行鋼琴聲部定義
        if not sym.parts:
            if sym.is_grand_staff:
                sym.parts = [
                    _SymPart(name="右手 RH", clef="treble", hand="RH", timbre="鋼琴"),
                    _SymPart(name="左手 LH", clef="bass", hand="LH", timbre="鋼琴"),
                ]
            else:
                sym.parts = [_SymPart(name="", clef="treble", hand="", timbre="")]

        n_parts = len(sym.parts)
        sp = 18.0 if n_parts <= 2 else (15.5 if n_parts <= 4 else 13.0)
        staff_h = 4.0 * sp
        staff_gap = 7.0 * sp if n_parts == 2 else 5.2 * sp
        sys_block_h = n_parts * staff_h + max(0, n_parts - 1) * staff_gap
        margin_l = 155.0 if n_parts >= 3 else 125.0
        margin_r = 75.0

        def get_part_events(sm: _SymMeasure, p_idx: int) -> List[_SymEvent]:
            if sm.part_events and p_idx < len(sm.part_events):
                return sm.part_events[p_idx]
            if p_idx == 0:
                return sm.rh_events
            if p_idx == 1:
                return sm.lh_events
            return []

        avg_notes = float(np.mean([max([len(get_part_events(m, pi)) for pi in range(n_parts)] + [1]) for m in sym.measures]))
        measures_per_sys = 3 if avg_notes >= 9 else 4

        systems: List[List[_SymMeasure]] = [
            sym.measures[i:i + measures_per_sys]
            for i in range(0, len(sym.measures), measures_per_sys)
        ]

        sys_pitch_base = sys_block_h + (160.0 if n_parts <= 2 else 135.0)
        sys_per_page_p1 = max(1, int((H - 270.0) // sys_pitch_base))
        sys_per_page_pn = max(1, int((H - 180.0) // sys_pitch_base))

        pages_systems: List[List[List[_SymMeasure]]] = []
        idx_s = 0
        while idx_s < len(systems):
            cap = sys_per_page_p1 if len(pages_systems) == 0 else sys_per_page_pn
            pages_systems.append(systems[idx_s:idx_s + cap])
            idx_s += cap

        analysis = ScoreAnalysis(
            beats_per_measure=sym.beats_per_measure,
            enable_repeats=True,
            detected_time_sig=sym.time_sig,
            detected_bpm=sym.bpm,
            detected_key_fifths=sym.key_fifths,
            is_grand_staff=(n_parts >= 2),
            source_format=source_format,
        )
        rgb_pages: List[np.ndarray] = []

        cur_beat = 0.0
        cur_ev_idx = 0
        sys_global_idx = 0

        for page_idx, page_sys_list in enumerate(pages_systems):
            qimg = QImage(W, H, QImage.Format.Format_RGB888)
            qimg.fill(QColor(255, 255, 255))
            p = QPainter(qimg)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

            if page_idx == 0:
                title_font = QFont("Serif", 26)
                title_font.setBold(True)
                p.setFont(title_font)
                p.setPen(QColor(20, 24, 31))
                p.drawText(QRectF(80, 55, W - 160, 56), Qt.AlignmentFlag.AlignCenter, sym.title or "Untitled Score")

                sub_font = QFont("Serif", 13)
                p.setFont(sub_font)
                p.setPen(QColor(70, 78, 90))
                meta_line = f"{sym.composer + '   |   ' if sym.composer else ''}{format_key_signature_label(sym.key_fifths)}   •   拍號 {sym.time_sig}   •   ♩ = {sym.bpm}"
                p.drawText(QRectF(80, 118, W - 160, 32), Qt.AlignmentFlag.AlignCenter, meta_line)
                y_start = 220.0
            else:
                y_start = 120.0

            pg_font = QFont("Serif", 11)
            p.setFont(pg_font)
            p.setPen(QColor(110, 115, 125))
            p.drawText(QRectF(0, H - 55, W, 30), Qt.AlignmentFlag.AlignCenter, str(page_idx + 1))

            page_res = PageAnalysis(
                page_index=page_idx,
                width=W,
                height=H,
                skew_angle=0.0,
                detected_time_sig=sym.time_sig,
                detected_bpm=sym.bpm,
                detected_key_fifths=sym.key_fifths,
                is_grand_staff=(n_parts >= 2),
            )

            for local_s_idx, sys_measures in enumerate(page_sys_list):
                sys_top_y = y_start + local_s_idx * sys_pitch_base
                part_y_tops = [sys_top_y + pi * (staff_h + staff_gap) for pi in range(n_parts)]
                part_y_bots = [yt + staff_h for yt in part_y_tops]
                sys_bot_y = part_y_bots[-1]

                base_staff_idx = len(page_res.staves)
                sys_staff_indices = [base_staff_idx + pi for pi in range(n_parts)]

                # 繪製各行五線譜橫線與左側樂器名稱
                line_pen = QPen(QColor(35, 38, 45), 1.5)
                part_font = QFont("Sans Serif", 10)
                part_font.setBold(True)
                for pi in range(n_parts):
                    p.setPen(line_pen)
                    for k in range(5):
                        yk = part_y_tops[pi] + k * sp
                        p.drawLine(QPointF(margin_l, yk), QPointF(W - margin_r, yk))
                    if n_parts >= 3 and sym.parts[pi].name:
                        p.setFont(part_font)
                        p.setPen(QColor(45, 55, 72))
                        p.drawText(
                            QRectF(12.0, part_y_tops[pi], margin_l - 28.0, staff_h),
                            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                            sym.parts[pi].name[:12],
                        )

                # 左端系統貫穿直線與大括弧
                p.setPen(QPen(QColor(20, 22, 26), 2.4))
                p.drawLine(QPointF(margin_l, part_y_tops[0]), QPointF(margin_l, sys_bot_y))
                if n_parts >= 2:
                    cls._draw_grand_brace(p, margin_l - 10.0, part_y_tops[0], sys_bot_y)

                # 繪製每行譜頭（譜號、調號、拍號）
                header_w = 0.0
                for pi in range(n_parts):
                    hw_i = cls._draw_staff_header(
                        p, margin_l, part_y_tops[pi], sp, sym.parts[pi].clef, sym.key_fifths,
                        sym.time_sig if (page_idx == 0 and local_s_idx == 0) else None,
                    )
                    header_w = max(header_w, hw_i)

                content_x0 = margin_l + header_w
                content_w = (W - margin_r) - content_x0
                m_w = content_w / max(1, len(sys_measures))

                sys_staves: List[StaffInfo] = []
                for pi in range(n_parts):
                    st_i = base_staff_idx + pi
                    pair_idx = (base_staff_idx + 1 if pi == 0 else base_staff_idx) if n_parts == 2 else None
                    sys_staves.append(
                        StaffInfo(
                            page_index=page_idx,
                            staff_index=st_i,
                            y_top_norm=part_y_tops[pi] / H,
                            y_bot_norm=part_y_bots[pi] / H,
                            x_left_norm=margin_l / W,
                            x_right_norm=(W - margin_r) / W,
                            spacing_norm=sp / H,
                            clef=sym.parts[pi].clef,
                            key_fifths=sym.key_fifths,
                            system_index=sys_global_idx,
                            paired_staff_index=pair_idx,
                            system_staff_indices=list(sys_staff_indices),
                            part_name=sym.parts[pi].name,
                            instrument_timbre=sym.parts[pi].timbre,
                        )
                    )

                for m_pos, sm in enumerate(sys_measures):
                    mx1 = content_x0 + m_pos * m_w
                    mx2 = mx1 + m_w
                    mn_font = QFont("Sans Serif", 9)
                    p.setFont(mn_font)
                    p.setPen(QColor(95, 102, 115))
                    p.drawText(QPointF(mx1 + 4.0, part_y_tops[0] - 12.0), str(sm.measure_number))

                    p.setPen(QPen(QColor(25, 28, 34), 1.8))
                    p.drawLine(QPointF(mx2, part_y_tops[0]), QPointF(mx2, sys_bot_y))
                    for st_obj in sys_staves:
                        st_obj.barlines_norm.append(mx2 / W)

                    m_beat_start = cur_beat
                    m_beat_end = cur_beat + sm.duration_beats
                    pad_x = min(26.0, m_w * 0.08)
                    usable_mw = max(20.0, m_w - 2.0 * pad_x)
                    combined_m: List[NoteEvent] = []
                    lead_m_info = None

                    for pi in range(n_parts):
                        st_obj = sys_staves[pi]
                        part_meta = sym.parts[pi]
                        sy_top, sy_bot = part_y_tops[pi], part_y_bots[pi]
                        m_info = MeasureInfo(
                            page_index=page_idx,
                            staff_index=st_obj.staff_index,
                            measure_number=sm.measure_number,
                            x_start_norm=mx1 / W,
                            x_end_norm=mx2 / W,
                            y_top_norm=sy_top / H,
                            y_bot_norm=sy_bot / H,
                            beat_start=m_beat_start,
                            beat_end=m_beat_end,
                        )
                        if lead_m_info is None:
                            lead_m_info = m_info
                        ev_list = get_part_events(sm, pi)
                        if not ev_list:
                            cx = 0.5 * (mx1 + mx2)
                            cy = sy_top + sp * 1.25
                            p.setPen(Qt.PenStyle.NoPen)
                            p.setBrush(QBrush(QColor(25, 25, 25)))
                            p.drawRect(QRectF(cx - 8.0, sy_top + sp, 16.0, sp * 0.48))
                            ne = NoteEvent(
                                event_index=0,
                                page_index=page_idx,
                                staff_index=st_obj.staff_index,
                                measure_number=sm.measure_number,
                                note_index_in_measure=1,
                                notes_in_measure=1,
                                x_norm=cx / W,
                                y_norm=cy / H,
                                beat_offset=m_beat_start,
                                duration_beats=sm.duration_beats,
                                is_rest=True,
                                rest_measures=1,
                                note_name="整小節休止",
                                note_type="整小節休止",
                                clef=part_meta.clef,
                                hand=part_meta.hand,
                                system_index=sys_global_idx,
                                part_name=part_meta.name,
                                instrument_timbre=part_meta.timbre,
                            )
                            m_info.notes.append(ne)
                            combined_m.append(ne)
                        else:
                            total_e = len(ev_list)
                            for k_i, se in enumerate(ev_list, start=1):
                                frac_b = max(0.0, min(0.95, se.local_beat / max(0.25, sm.duration_beats)))
                                nx = mx1 + pad_x + frac_b * usable_mw
                                n_type, is_hollow, beams, is_dot = _duration_to_meta(se.duration_beats, se.is_rest, se.is_grace)
                                if se.is_rest or not se.midi_notes:
                                    cy = sy_top + 2.0 * sp
                                    cls._draw_rest(p, nx, sy_top, sp, se.duration_beats)
                                    ne = NoteEvent(
                                        event_index=0,
                                        page_index=page_idx,
                                        staff_index=st_obj.staff_index,
                                        measure_number=sm.measure_number,
                                        note_index_in_measure=k_i,
                                        notes_in_measure=total_e,
                                        x_norm=nx / W,
                                        y_norm=cy / H,
                                        beat_offset=m_beat_start + se.local_beat,
                                        duration_beats=se.duration_beats,
                                        is_rest=True,
                                        note_name="休止",
                                        note_type=n_type,
                                        clef=part_meta.clef,
                                        hand=part_meta.hand,
                                        system_index=sys_global_idx,
                                        part_name=part_meta.name,
                                        instrument_timbre=part_meta.timbre,
                                    )
                                    m_info.notes.append(ne)
                                    combined_m.append(ne)
                                    continue

                                ys_px: List[float] = []
                                names: List[str] = []
                                for mv in se.midi_notes:
                                    step, disp_acc = midi_to_staff_step(mv, part_meta.clef, sym.key_fifths)
                                    ny = sy_bot - step * (sp * 0.5)
                                    ys_px.append(ny)
                                    names.append(midi_to_note_name(mv, sym.key_fifths))
                                    cls._draw_notehead_and_ledgers(
                                        p, nx, ny, sy_top, sy_bot, sp,
                                        is_hollow=is_hollow, is_grace=se.is_grace,
                                        disp_acc=disp_acc, is_dotted=(is_dot or se.is_dotted),
                                    )
                                if se.duration_beats < 3.5:
                                    avg_y = float(np.mean(ys_px))
                                    stem_up = avg_y >= (sy_top + 2.0 * sp)
                                    cls._draw_stem_and_flag(
                                        p, nx, min(ys_px), max(ys_px), sp,
                                        stem_up=stem_up, beams=beams, is_grace=se.is_grace,
                                    )

                                rep_midi = se.midi_notes[-1]
                                rep_y = ys_px[-1]
                                disp_name = "+".join(names) if len(names) <= 3 else f"{names[0]}..{names[-1]}({len(names)}音)"
                                if se.is_grace:
                                    disp_name = f"({disp_name})"

                                ne = NoteEvent(
                                    event_index=0,
                                    page_index=page_idx,
                                    staff_index=st_obj.staff_index,
                                    measure_number=sm.measure_number,
                                    note_index_in_measure=k_i,
                                    notes_in_measure=total_e,
                                    x_norm=nx / W,
                                    y_norm=rep_y / H,
                                    beat_offset=m_beat_start + se.local_beat,
                                    duration_beats=se.duration_beats,
                                    is_rest=False,
                                    is_grace=se.is_grace,
                                    is_hollow=is_hollow,
                                    is_dotted=(is_dot or se.is_dotted),
                                    beam_count=beams,
                                    midi_note=rep_midi,
                                    note_name=disp_name,
                                    note_type=n_type,
                                    midi_notes=list(se.midi_notes),
                                    note_ys_norm=[y_v / H for y_v in ys_px],
                                    clef=part_meta.clef,
                                    hand=part_meta.hand,
                                    system_index=sys_global_idx,
                                    part_name=part_meta.name,
                                    instrument_timbre=part_meta.timbre,
                                )
                                m_info.notes.append(ne)
                                combined_m.append(ne)
                        st_obj.measures.append(m_info)

                    combined_m.sort(
                        key=lambda e: (round(e.beat_offset, 4), e.staff_index, e.is_rest),
                    )
                    for ne in combined_m:
                        cur_ev_idx += 1
                        ne.event_index = cur_ev_idx
                        page_res.events.append(ne)
                        analysis.events.append(ne)

                    if lead_m_info is not None:
                        analysis.playback_measures.append(
                            PlaybackMeasure(
                                measure=lead_m_info,
                                beat_start=m_beat_start,
                                beat_end=m_beat_end,
                                repeat_pass=1,
                            )
                        )
                    cur_beat = m_beat_end

                page_res.staves.extend(sys_staves)
                sys_global_idx += 1

            p.end()

            ptr = qimg.constBits()
            ptr.setsize(H * qimg.bytesPerLine())
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(H, qimg.bytesPerLine())[:, :W * 3].reshape(H, W, 3).copy()
            rgb_pages.append(arr)
            analysis.pages.append(page_res)

        analysis.total_beats = cur_beat
        analysis.total_measures = len(sym.measures)
        return rgb_pages, analysis

    # ------------------------------------------------------------------
    # 向量音樂符號繪製輔助函式
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_grand_brace(p: QPainter, x: float, y_top: float, y_bot: float):
        mid = 0.5 * (y_top + y_bot)
        path = QPainterPath()
        path.moveTo(x + 6.0, y_top)
        path.cubicTo(x - 14.0, y_top + 18.0, x + 6.0, mid - 14.0, x - 14.0, mid)
        path.cubicTo(x + 6.0, mid + 14.0, x - 14.0, y_bot - 18.0, x + 6.0, y_bot)
        p.setPen(QPen(QColor(20, 22, 26), 3.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    @staticmethod
    def _draw_staff_header(
        p: QPainter,
        x0: float,
        y_top: float,
        sp: float,
        clef: str,
        key_fifths: int,
        time_sig: Optional[str],
    ) -> float:
        x = x0 + 12.0
        # 譜號（高音 / 低音簡潔向量繪製，跨平台不依賴音樂字型）
        p.setPen(QPen(QColor(20, 22, 26), 2.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        if clef == "treble":
            # 簡潔高音譜號 G-clef
            cx = x + 14.0
            p.drawLine(QPointF(cx, y_top - sp * 0.9), QPointF(cx, y_top + sp * 4.7))
            p.drawEllipse(QPointF(cx, y_top + sp * 3.0), sp * 0.65, sp * 0.65)
            p.drawArc(QRectF(cx - sp * 0.45, y_top - sp * 0.9, sp * 0.9, sp * 1.8), -90 * 16, 180 * 16)
            p.setBrush(QBrush(QColor(20, 22, 26)))
            p.drawEllipse(QPointF(cx - 4.0, y_top + sp * 4.7), 3.2, 3.2)
            x += 38.0
        else:
            # 簡潔低音譜號 F-clef（弧線 + 兩點夾住第 4 線）
            cx = x + 12.0
            p.setBrush(QBrush(QColor(20, 22, 26)))
            p.drawEllipse(QPointF(cx - 2.0, y_top + sp * 1.0), 4.2, 4.2)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(cx - 10.0, y_top + sp * 0.15, sp * 1.45, sp * 2.6), -45 * 16, 180 * 16)
            p.setBrush(QBrush(QColor(20, 22, 26)))
            p.drawEllipse(QPointF(cx + sp * 1.15, y_top + sp * 0.55), 2.8, 2.8)
            p.drawEllipse(QPointF(cx + sp * 1.15, y_top + sp * 1.45), 2.8, 2.8)
            x += 38.0

        # 調號升降記號
        f = max(-7, min(7, int(key_fifths)))
        if f != 0:
            sharp_steps_treble = (8, 5, 9, 6, 3, 7, 4)
            sharp_steps_bass = (6, 3, 7, 4, 1, 5, 2)
            flat_steps_treble = (4, 7, 3, 6, 2, 5, 1)
            flat_steps_bass = (2, 5, 1, 4, 0, 3, -1)
            steps = (
                (sharp_steps_treble if clef == "treble" else sharp_steps_bass)
                if f > 0
                else (flat_steps_treble if clef == "treble" else flat_steps_bass)
            )
            acc_font = QFont("Sans Serif", 13)
            acc_font.setBold(True)
            p.setFont(acc_font)
            p.setPen(QColor(20, 22, 26))
            sym_ch = "#" if f > 0 else "b"
            for i in range(abs(f)):
                ay = (y_top + 4.0 * sp) - steps[i] * (sp * 0.5)
                p.drawText(QRectF(x - 4.0, ay - 12.0, 18.0, 24.0), Qt.AlignmentFlag.AlignCenter, sym_ch)
                x += 13.0
            x += 6.0

        # 拍號（僅首行）
        if time_sig and "/" in time_sig:
            num, den = time_sig.split("/", 1)
            ts_font = QFont("Serif", 17)
            ts_font.setBold(True)
            p.setFont(ts_font)
            p.setPen(QColor(20, 22, 26))
            p.drawText(QRectF(x, y_top - 2.0, 26.0, 2.0 * sp + 4.0), Qt.AlignmentFlag.AlignCenter, num)
            p.drawText(QRectF(x, y_top + 2.0 * sp - 2.0, 26.0, 2.0 * sp + 4.0), Qt.AlignmentFlag.AlignCenter, den)
            x += 34.0

        return (x - x0) + 10.0

    @staticmethod
    def _draw_notehead_and_ledgers(
        p: QPainter,
        nx: float,
        ny: float,
        sy_top: float,
        sy_bot: float,
        sp: float,
        is_hollow: bool,
        is_grace: bool,
        disp_acc: int,
        is_dotted: bool,
    ):
        # 上下加線
        p.setPen(QPen(QColor(35, 38, 45), 1.5))
        if ny < sy_top - sp * 0.45:
            ly = sy_top - sp
            while ly >= ny - sp * 0.25:
                p.drawLine(QPointF(nx - sp * 0.72, ly), QPointF(nx + sp * 0.72, ly))
                ly -= sp
        elif ny > sy_bot + sp * 0.45:
            ly = sy_bot + sp
            while ly <= ny + sp * 0.25:
                p.drawLine(QPointF(nx - sp * 0.72, ly), QPointF(nx + sp * 0.72, ly))
                ly += sp

        # 臨時記號
        if disp_acc != 0:
            af = QFont("Sans Serif", 11)
            af.setBold(True)
            p.setFont(af)
            p.setPen(QColor(20, 22, 26))
            ch = "#" if disp_acc == 1 else ("b" if disp_acc == -1 else "♮")
            p.drawText(QRectF(nx - sp * 1.45, ny - 10.0, 16.0, 20.0), Qt.AlignmentFlag.AlignCenter, ch)

        # 橢圓符頭
        rx = sp * (0.42 if is_grace else 0.58)
        ry = sp * (0.31 if is_grace else 0.43)
        p.save()
        p.translate(nx, ny)
        p.rotate(-18.0)
        if is_hollow:
            p.setPen(QPen(QColor(18, 20, 24), 2.3))
            p.setBrush(QBrush(QColor(255, 255, 255)))
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(18, 20, 24)))
        p.drawEllipse(QPointF(0.0, 0.0), rx, ry)
        p.restore()

        # 附點
        if is_dotted:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(18, 20, 24)))
            p.drawEllipse(QPointF(nx + rx + 5.0, ny - 2.0), 2.4, 2.4)

    @staticmethod
    def _draw_stem_and_flag(
        p: QPainter,
        nx: float,
        y_min: float,
        y_max: float,
        sp: float,
        stem_up: bool,
        beams: int,
        is_grace: bool,
    ):
        rx = sp * (0.40 if is_grace else 0.54)
        stem_len = sp * (2.3 if is_grace else 3.1)
        p.setPen(QPen(QColor(20, 22, 26), 1.6 if is_grace else 2.0))
        if stem_up:
            sx = nx + rx - 0.5
            y_end = y_min - stem_len
            p.drawLine(QPointF(sx, y_max), QPointF(sx, y_end))
            for b_i in range(beams):
                fy = y_end + b_i * 6.0
                p.drawLine(QPointF(sx, fy), QPointF(sx + sp * 0.55, fy + sp * 0.45))
        else:
            sx = nx - rx + 0.5
            y_end = y_max + stem_len
            p.drawLine(QPointF(sx, y_min), QPointF(sx, y_end))
            for b_i in range(beams):
                fy = y_end - b_i * 6.0
                p.drawLine(QPointF(sx, fy), QPointF(sx + sp * 0.55, fy - sp * 0.45))

    @staticmethod
    def _draw_rest(p: QPainter, nx: float, sy_top: float, sp: float, dur_beats: float):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(35, 38, 45)))
        if dur_beats >= 3.5:
            p.drawRect(QRectF(nx - 7.0, sy_top + sp, 14.0, sp * 0.45))
        elif dur_beats >= 1.75:
            p.drawRect(QRectF(nx - 7.0, sy_top + sp * 1.55, 14.0, sp * 0.45))
        else:
            f = QFont("Serif", 16)
            p.setFont(f)
            p.setPen(QColor(40, 44, 52))
            p.drawText(QRectF(nx - 8.0, sy_top + sp * 0.8, 16.0, sp * 2.4), Qt.AlignmentFlag.AlignCenter, "𝄽" if dur_beats >= 0.8 else "𝄾")
