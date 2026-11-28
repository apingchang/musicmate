"""數位向量 PDF 音樂字型結構解析引擎 (PDFVectorAnalyzer)

專為 MuseScore / Finale / Sibelius / LilyPond 匯出之向量 PDF 樂譜設計：
直接透過 PyMuPDF 讀取內嵌向量五線譜線 (`page.get_drawings()`) 與 SMuFL / Emmentaler (MScore) 音樂字型字元座標 (`page.get_text("rawdict")`)，
自動辨識：
1. 鋼琴雙手大譜表（Grand Staff）成對系統與跨行小節線
2. 高音譜號 (𝄞) / 低音譜號 (𝄢)
3. 調號升降記號數量（如 4♯ = F#, C#, G#, D#）與小節內臨時記號（♯ / ♭ / ♮ / 𝄪）
4. 拍號（C = 4/4、₵ = 2/2、2/4、3/4、4/4、6/8）
5. 全音符 / 二分音符 / 四分音符 / 八分與三連音 / 垂直多音和弦（如左手八度）

若 PDF 為掃描圖檔（不含內嵌音樂字型），`analyze_pdf_document` 會回傳 None，由上層無縫切換回 OpenCV 影像 OMR。
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.core.score_analyzer import (
    MeasureInfo,
    NoteEvent,
    PageAnalysis,
    PlaybackMeasure,
    ScoreAnalysis,
    StaffInfo,
)
from src.core.symbolic_score_loader import key_signature_map

# 同時支援標準 SMuFL (Bravura/Leland) 與 MuseScore 舊版 Emmentaler/MScore PUA 編碼
_TREBLE_CLEFS = {0xE050, 0xE19E, 0x1D11E}
_BASS_CLEFS = {0xE062, 0xE19C, 0x1D122}
_SHARPS = {0xE262, 0xE10E, 0x266F}
_FLATS = {0xE260, 0xE10D, 0xE111, 0x266D}
_NATURALS = {0xE261, 0xE113, 0x266E}
_DOUBLE_SHARPS = {0xE263, 0xE11C}
_DOUBLE_FLATS = {0xE264, 0xE11B}
_COMMON_TIME = {0xE08A, 0xE09A}
_CUT_TIME = {0xE08B, 0xE09B}
_NOTEHEAD_SOLID = {0xE0A4, 0xE12D}
_NOTEHEAD_HALF = {0xE0A3, 0xE12C}
_NOTEHEAD_WHOLE = {0xE0A2, 0xE12B}
_AUG_DOTS = {0xE1E7, 0xE127}
_ALL_NOTEHEADS = _NOTEHEAD_SOLID | _NOTEHEAD_HALF | _NOTEHEAD_WHOLE

# 拍號數字字型 (SMuFL 0xE080..0xE089 & Emmentaler 0xE080..0xE089 / ASCII '0'..'9')
_TS_DIGITS: Dict[int, int] = {}
for _d in range(10):
    _TS_DIGITS[0xE080 + _d] = _d
    _TS_DIGITS[0x0030 + _d] = _d

_LETTER_NAMES = ("C", "D", "E", "F", "G", "A", "B")
_LETTER_SEMITONES = (0, 2, 4, 5, 7, 9, 11)


@dataclass
class _Glyph:
    code: int
    x: float
    y: float
    x0: float
    y0: float
    x1: float
    y1: float
    font: str = ""


@dataclass
class _VecStaff:
    y_lines: List[float]
    x_left: float
    x_right: float
    sp: float
    clef: str = "treble"
    key_fifths: int = 0
    time_sig: Optional[str] = None
    header_end_x: float = 0.0
    barlines_x: List[float] = field(default_factory=list)
    system_index: int = -1
    paired_local_idx: Optional[int] = None

    @property
    def y_top(self) -> float:
        return self.y_lines[0]

    @property
    def y_bot(self) -> float:
        return self.y_lines[4]


def _step_to_pitch(
    step: int,
    clef: str,
    key_fifths: int,
    acc_override: Optional[int] = None,
    show_natural: bool = False,
) -> Tuple[int, str]:
    """將半間階數 (0 = 底線: 高音 E4 / 低音 G2) 轉為精準 (MIDI, 音名)，自動套用調號與臨時記號"""
    base_diatonic = (4 * 7 + 2) if clef == "treble" else (2 * 7 + 4)  # E4=30, G2=18
    abs_d = base_diatonic + int(step)
    octave = abs_d // 7
    deg = abs_d % 7
    ks = key_signature_map(key_fifths)
    eff_acc = ks[deg] if acc_override is None else int(acc_override)
    midi = (octave + 1) * 12 + _LETTER_SEMITONES[deg] + eff_acc
    midi = max(21, min(108, midi))

    if eff_acc == 2:
        acc_str = "##"
    elif eff_acc == 1:
        acc_str = "#"
    elif eff_acc == -1:
        acc_str = "b"
    elif eff_acc == -2:
        acc_str = "bb"
    elif show_natural:
        acc_str = "♮"
    else:
        acc_str = ""

    if acc_str == "♮":
        return midi, f"♮{_LETTER_NAMES[deg]}{octave}"
    return midi, f"{_LETTER_NAMES[deg]}{acc_str}{octave}"


class PDFVectorAnalyzer:
    """從含內嵌音樂字型的數位向量 PDF 直接萃取大譜表、調號、拍號與雙手精準音符"""

    @classmethod
    def analyze_pdf_document(
        cls,
        doc,
        beats_per_measure: float = 4.0,
        enable_repeats: bool = True,
    ) -> Optional[ScoreAnalysis]:
        """若 PDF 內含向量音樂字型，回傳完整 ScoreAnalysis；否則回傳 None（交由影像 OMR）"""
        if doc is None or doc.page_count <= 0:
            return None

        # 先快篩第一頁是否含內嵌音樂符頭字元
        p0_glyphs = cls._extract_glyphs(doc[0])
        n_heads_p0 = sum(1 for g in p0_glyphs if g.code in _ALL_NOTEHEADS)
        if n_heads_p0 < 6:
            return None

        analysis = ScoreAnalysis(
            beats_per_measure=beats_per_measure,
            enable_repeats=enable_repeats,
            source_format="pdf_vector",
        )
        cur_beat = 0.0
        cur_measure = 0
        cur_ev_idx = 0
        global_sys_idx = 0

        # 記錄第一行辨識到的調號/拍號供跨頁延續
        carry_key_fifths = 0
        carry_beats = beats_per_measure

        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            pw = float(page.rect.width)
            ph = float(page.rect.height)
            glyphs = p0_glyphs if page_idx == 0 else cls._extract_glyphs(page)
            h_segs, v_segs = cls._extract_line_segments(page)

            staves = cls._group_staff_lines(h_segs, pw, ph)
            if not staves:
                analysis.pages.append(PageAnalysis(page_idx, int(round(pw)), int(round(ph)), 0.0))
                continue

            # 偵測大譜表或多種樂器重奏總譜系統（2 行鋼琴大譜表或 N 行跨行貫穿小節線總譜）
            sys_groups = cls._group_systems(staves, v_segs)
            is_multi_page = any(len(g) >= 2 for g in sys_groups)
            if is_multi_page:
                analysis.is_grand_staff = True

            # 為每行五線譜識別譜號、調號、拍號與嚴格小節線（杜絕 M3 左手 A1+A2 長符幹誤判為小節線！）
            for grp in sys_groups:
                sys_top = staves[grp[0]].y_top
                sys_bot = staves[grp[-1]].y_bot
                is_multi_sys = len(grp) >= 2
                for s_i in grp:
                    st = staves[s_i]
                    cls._parse_staff_header(st, glyphs, carry_key_fifths)
                    if st.key_fifths != 0:
                        carry_key_fifths = st.key_fifths
                    if st.time_sig and analysis.detected_time_sig is None:
                        analysis.detected_time_sig = st.time_sig
                        ts_beats_map = {"2/4": 2.0, "3/4": 3.0, "4/4": 4.0, "2/2": 4.0, "6/8": 6.0}
                        if st.time_sig in ts_beats_map:
                            carry_beats = ts_beats_map[st.time_sig]
                            analysis.beats_per_measure = carry_beats
                    cls._find_staff_barlines(st, v_segs, sys_top=sys_top, sys_bot=sys_bot, is_multi_staff=is_multi_sys)

            if page_idx == 0:
                analysis.detected_key_fifths = carry_key_fifths

            # 將同一 System 的所有行小節線對齊
            cls._align_system_barlines(staves, sys_groups)

            # 將音符與臨時記號歸屬到對應五線譜
            staff_glyphs: List[List[_Glyph]] = [[] for _ in staves]
            for g in glyphs:
                if g.code in (_ALL_NOTEHEADS | _SHARPS | _FLATS | _NATURALS | _DOUBLE_SHARPS | _DOUBLE_FLATS | _AUG_DOTS):
                    best_i = cls._assign_glyph_to_staff(g, staves)
                    if best_i is not None:
                        staff_glyphs[best_i].append(g)

            page_res = PageAnalysis(
                page_index=page_idx,
                width=int(round(pw)),
                height=int(round(ph)),
                skew_angle=0.0,
                detected_time_sig=analysis.detected_time_sig,
                detected_key_fifths=carry_key_fifths,
                is_grand_staff=is_multi_page,
            )

            # 建立 StaffInfo 清單
            staff_infos: List[StaffInfo] = []
            for s_i, st in enumerate(staves):
                si = StaffInfo(
                    page_index=page_idx,
                    staff_index=s_i,
                    y_top_norm=st.y_top / ph,
                    y_bot_norm=st.y_bot / ph,
                    x_left_norm=st.x_left / pw,
                    x_right_norm=st.x_right / pw,
                    spacing_norm=st.sp / ph,
                    barlines_norm=[bx / pw for bx in st.barlines_x],
                    clef=st.clef,
                    key_fifths=st.key_fifths,
                    paired_staff_index=st.paired_local_idx,
                )
                staff_infos.append(si)

            # 依 System 群組迭代（支援單行、2 行鋼琴大譜表及 N 行多樂器重奏總譜）
            for grp in sys_groups:
                is_piano_pair = len(grp) == 2 and staves[grp[0]].clef == "treble" and staves[grp[1]].clef == "bass"
                for pos_in_sys, s_i in enumerate(grp):
                    staff_infos[s_i].system_index = global_sys_idx
                    staff_infos[s_i].system_staff_indices = list(grp)
                    if len(grp) == 1:
                        staff_infos[s_i].part_name = ""
                    elif is_piano_pair:
                        staff_infos[s_i].part_name = "右手 RH" if pos_in_sys == 0 else "左手 LH"
                    else:
                        clef_zh = "高音" if staves[s_i].clef == "treble" else "低音"
                        staff_infos[s_i].part_name = f"聲部 {pos_in_sys + 1} ({clef_zh})"

                edges_x = cls._measure_edges(staves[grp[0]])
                n_m = max(0, len(edges_x) - 1)
                for m_k in range(n_m):
                    cur_measure += 1
                    mx1, mx2 = edges_x[m_k], edges_x[m_k + 1]
                    m_beat_start = cur_beat
                    m_beat_end = cur_beat + carry_beats
                    combined: List[NoteEvent] = []
                    lead_m_info = None

                    for pos_in_sys, s_i in enumerate(grp):
                        if is_piano_pair:
                            hand_tag = "RH" if pos_in_sys == 0 else "LH"
                        elif len(grp) >= 2:
                            hand_tag = f"P{pos_in_sys + 1}"
                        else:
                            hand_tag = ""
                        m_info, evs = cls._build_measure_events(
                            staves[s_i], staff_glyphs[s_i], page_idx, s_i, cur_measure,
                            mx1, mx2, m_beat_start, carry_beats, pw, ph, hand_tag, global_sys_idx,
                            part_name=staff_infos[s_i].part_name,
                        )
                        staff_infos[s_i].measures.append(m_info)
                        if lead_m_info is None:
                            lead_m_info = m_info
                        combined.extend(evs)

                    combined.sort(
                        key=lambda e: (round(e.beat_offset, 4), e.staff_index, e.is_rest),
                    )
                    for ne in combined:
                        cur_ev_idx += 1
                        ne.event_index = cur_ev_idx
                        page_res.events.append(ne)
                        analysis.events.append(ne)

                    if lead_m_info is not None:
                        analysis.playback_measures.append(
                            PlaybackMeasure(measure=lead_m_info, beat_start=m_beat_start, beat_end=m_beat_end, repeat_pass=1)
                        )
                    cur_beat = m_beat_end
                global_sys_idx += 1

            page_res.staves = staff_infos
            analysis.pages.append(page_res)

        if not analysis.has_staves or not analysis.events:
            return None

        analysis.total_beats = cur_beat
        analysis.total_measures = cur_measure
        return analysis

    @staticmethod
    def _extract_glyphs(page) -> List[_Glyph]:
        out: List[_Glyph] = []
        raw = page.get_text("rawdict")
        for blk in raw.get("blocks", []):
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    font_name = str(span.get("font", ""))
                    for ch in span.get("chars", []):
                        c_str = ch.get("c", "")
                        if not c_str:
                            continue
                        code = ord(c_str[0])
                        bbox = ch.get("bbox")
                        origin = ch.get("origin")
                        if not bbox:
                            continue
                        x0, y0, x1, y1 = [float(v) for v in bbox]
                        cx = 0.5 * (x0 + x1)
                        # 對音樂字型而言，origin[1] 精準對應字形基準線（即符頭正中心 y 座標！）
                        cy = float(origin[1]) if (origin and code >= 0xE000) else 0.5 * (y0 + y1)
                        out.append(_Glyph(code=code, x=cx, y=cy, x0=x0, y0=y0, x1=x1, y1=y1, font=font_name))
        return out

    @staticmethod
    def _extract_line_segments(page) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """從 PDF 向量繪圖指令擷取水平線段 (y, x0, x1) 與垂直線段 (x, y0, y1)，相容按小節分段繪製的五線譜"""
        h_segs: List[Tuple[float, float, float]] = []
        v_segs: List[Tuple[float, float, float]] = []
        for dr in page.get_drawings():
            for item in dr.get("items", []):
                kind = item[0]
                if kind == "l":
                    p1, p2 = item[1], item[2]
                    dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
                    if dy <= 1.2 and dx >= 15.0:
                        h_segs.append((0.5 * (p1.y + p2.y), min(p1.x, p2.x), max(p1.x, p2.x)))
                    elif dx <= 1.2 and dy >= 12.0:
                        v_segs.append((0.5 * (p1.x + p2.x), min(p1.y, p2.y), max(p1.y, p2.y)))
                elif kind == "re":
                    r = item[1]
                    rw, rh = abs(r.width), abs(r.height)
                    if rh <= 2.5 and rw >= 15.0:
                        h_segs.append((0.5 * (r.y0 + r.y1), r.x0, r.x1))
                    elif rw <= 2.5 and rh >= 12.0:
                        v_segs.append((0.5 * (r.x0 + r.x1), r.y0, r.y1))
        return h_segs, v_segs

    @staticmethod
    def _group_staff_lines(
        h_segs: List[Tuple[float, float, float]],
        pw: float,
        ph: float,
    ) -> List[_VecStaff]:
        """先將同高度 y 的按小節分段水平線段串接還原成整行譜線，再群組為 5 條等距五線譜"""
        if not h_segs:
            return []
        sorted_h = sorted(h_segs, key=lambda s: s[0])
        y_groups: List[List[Tuple[float, float, float]]] = []
        for seg in sorted_h:
            if y_groups and abs(seg[0] - y_groups[-1][0][0]) <= 1.25:
                y_groups[-1].append(seg)
            else:
                y_groups.append([seg])

        merged_y: List[Tuple[float, float, float]] = []
        for grp in y_groups:
            # 合併同一 y 上的所有 [x0, x1] 區間，計算實際總覆蓋寬度（解決 MuseScore 逐小節畫譜線被 0.35*pw 濾掉的問題！）
            intervals = sorted((s[1], s[2]) for s in grp)
            union_iv: List[List[float]] = [[intervals[0][0], intervals[0][1]]]
            for ix0, ix1 in intervals[1:]:
                if ix0 <= union_iv[-1][1] + 3.0:
                    union_iv[-1][1] = max(union_iv[-1][1], ix1)
                else:
                    union_iv.append([ix0, ix1])
            total_cov = sum(iv[1] - iv[0] for iv in union_iv)
            xl_g = union_iv[0][0]
            xr_g = union_iv[-1][1]
            if total_cov >= pw * 0.35 and (xr_g - xl_g) >= pw * 0.38:
                avg_y = float(np.mean([s[0] for s in grp]))
                merged_y.append((avg_y, xl_g, xr_g))

        staves: List[_VecStaff] = []
        i = 0
        while i + 4 < len(merged_y):
            ys = [merged_y[i + k][0] for k in range(5)]
            diffs = np.diff(ys)
            med_sp = float(np.median(diffs))
            if 3.0 <= med_sp <= 18.0 and np.all(np.abs(diffs - med_sp) <= max(1.0, med_sp * 0.18)):
                xl = float(np.median([merged_y[i + k][1] for k in range(5)]))
                xr = float(np.median([merged_y[i + k][2] for k in range(5)]))
                staves.append(_VecStaff(y_lines=ys, x_left=xl, x_right=xr, sp=med_sp))
                i += 5
            else:
                i += 1
        return staves

    @staticmethod
    def _group_systems(
        staves: List[_VecStaff],
        v_segs: List[Tuple[float, float, float]],
    ) -> List[List[int]]:
        """將有跨行貫穿小節線或左端系統連線之相鄰五線譜合併為系統群組（支援 2 行鋼琴與 N 行多樂器重奏總譜）"""
        if not staves:
            return []
        groups: List[List[int]] = []
        i = 0
        while i < len(staves):
            grp = [i]
            while grp[-1] + 1 < len(staves):
                s_prev = staves[grp[-1]]
                s_next = staves[grp[-1] + 1]
                gap = s_next.y_top - s_prev.y_bot
                if not (
                    abs(s_prev.x_left - s_next.x_left) <= s_prev.sp * 2.5
                    and s_prev.sp * 1.2 <= gap <= s_prev.sp * 18.0
                ):
                    break
                # 檢查是否有垂直線（左端系統線或跨行小節線）連接 s_prev 與 s_next
                bridge_bars = [
                    vx for vx, vy0, vy1 in v_segs
                    if (s_prev.x_left - s_prev.sp * 2.0) <= vx <= (s_prev.x_right + s_prev.sp * 2.0)
                    and vy0 <= s_prev.y_top + s_prev.sp * 0.8
                    and vy1 >= s_next.y_bot - s_next.sp * 0.8
                ]
                if len(bridge_bars) >= 1:
                    grp.append(grp[-1] + 1)
                else:
                    break
            if len(grp) == 2:
                staves[grp[0]].paired_local_idx = grp[1]
                staves[grp[1]].paired_local_idx = grp[0]
            groups.append(grp)
            i = grp[-1] + 1
        return groups

    @staticmethod
    def _parse_staff_header(st: _VecStaff, glyphs: List[_Glyph], default_key_fifths: int):
        """辨識此行譜號 (高/低音)、調號升降記號數與拍號，並算出精準 header_end_x"""
        sp = st.sp
        hdr_max_x = st.x_left + sp * 20.0
        hdr_glyphs = [
            g for g in glyphs
            if (st.x_left - sp) <= g.x <= hdr_max_x
            and (st.y_top - sp * 1.6) <= g.y <= (st.y_bot + sp * 1.6)
        ]
        hdr_glyphs.sort(key=lambda g: g.x)

        clef_x1 = st.x_left + sp * 2.5
        for g in hdr_glyphs:
            if g.code in _TREBLE_CLEFS:
                st.clef = "treble"
                clef_x1 = max(clef_x1, g.x1)
                break
            if g.code in _BASS_CLEFS:
                st.clef = "bass"
                clef_x1 = max(clef_x1, g.x1)
                break

        # 找到第一顆音符的 x 座標，調號與拍號必位於 [clef_x1, first_note_x]
        first_note_x = st.x_right
        for g in hdr_glyphs:
            if g.code in _ALL_NOTEHEADS and g.x > clef_x1 + sp * 0.5:
                first_note_x = g.x0
                break

        sharps_hdr = [g for g in hdr_glyphs if g.code in _SHARPS and clef_x1 - 2.0 <= g.x <= first_note_x - sp * 0.8]
        flats_hdr = [g for g in hdr_glyphs if g.code in _FLATS and clef_x1 - 2.0 <= g.x <= first_note_x - sp * 0.8]

        # 過濾掉離譜號太遠（其實屬於第一個音符的臨時記號）
        def filter_key_cluster(items: List[_Glyph]) -> List[_Glyph]:
            if not items:
                return []
            items.sort(key=lambda g: g.x)
            # 若第一個記號距譜號超過 4.5*sp，多半是首音的臨時記號而非調號
            if items[0].x - clef_x1 > sp * 4.5:
                return []
            cluster = [items[0]]
            for it in items[1:]:
                if it.x - cluster[-1].x <= sp * 2.2:
                    cluster.append(it)
                else:
                    break
            return cluster

        ks_sharps = filter_key_cluster(sharps_hdr)
        ks_flats = filter_key_cluster(flats_hdr)
        if ks_sharps and len(ks_sharps) >= len(ks_flats):
            st.key_fifths = min(7, len(ks_sharps))
            last_hdr_x = ks_sharps[-1].x1
        elif ks_flats:
            st.key_fifths = -min(7, len(ks_flats))
            last_hdr_x = ks_flats[-1].x1
        else:
            st.key_fifths = default_key_fifths
            last_hdr_x = clef_x1

        # 拍號偵測（C / ₵ / 數字拍號）
        ts_x0 = last_hdr_x - 1.0
        ts_x1 = min(first_note_x - sp * 0.3, last_hdr_x + sp * 5.0)
        ts_glyphs = [g for g in hdr_glyphs if ts_x0 <= g.x <= ts_x1]
        for g in ts_glyphs:
            if g.code in _CUT_TIME:
                st.time_sig = "2/2"
                last_hdr_x = max(last_hdr_x, g.x1)
            elif g.code in _COMMON_TIME:
                st.time_sig = "4/4"
                last_hdr_x = max(last_hdr_x, g.x1)
        if st.time_sig is None:
            digs = [g for g in ts_glyphs if g.code in _TS_DIGITS]
            if len(digs) >= 2:
                digs.sort(key=lambda g: g.y)
                num_d = _TS_DIGITS[digs[0].code]
                den_d = _TS_DIGITS[digs[-1].code]
                if den_d in (2, 4, 8) and 2 <= num_d <= 12:
                    st.time_sig = f"{num_d}/{den_d}"
                    last_hdr_x = max(last_hdr_x, max(d.x1 for d in digs))

        st.header_end_x = min(first_note_x - sp * 0.35, last_hdr_x + sp * 0.65)
        st.header_end_x = max(st.header_end_x, clef_x1 + sp * 0.5)

    @staticmethod
    def _find_staff_barlines(
        st: _VecStaff,
        v_segs: List[Tuple[float, float, float]],
        sys_top: Optional[float] = None,
        sys_bot: Optional[float] = None,
        is_multi_staff: bool = False,
    ):
        """嚴格辨識小節線：端點必須精準起迄於五線譜上下界或整個系統上下界，徹底杜絕 M3 左手 A1+A2 長符幹誤判為小節線！"""
        sp = st.sp
        xs = []
        s_top = st.y_top if sys_top is None else sys_top
        s_bot = st.y_bot if sys_bot is None else sys_bot
        for vx, vy0, vy1 in v_segs:
            if not ((st.header_end_x + sp * 1.5) <= vx <= (st.x_right + sp * 0.8)):
                continue
            is_sys_bar = (
                is_multi_staff
                and abs(vy0 - s_top) <= sp * 0.65
                and abs(vy1 - s_bot) <= sp * 0.65
            )
            is_exact_staff_bar = (
                abs(vy0 - st.y_top) <= sp * 0.45
                and abs(vy1 - st.y_bot) <= sp * 0.45
            )
            if is_sys_bar or is_exact_staff_bar:
                xs.append(vx)
        xs.sort()
        dedup: List[float] = []
        for x in xs:
            if not dedup or (x - dedup[-1]) >= sp * 2.0:
                dedup.append(x)
        st.barlines_x = dedup

    @staticmethod
    def _align_system_barlines(staves: List[_VecStaff], sys_groups: List[List[int]]):
        for grp in sys_groups:
            if len(grp) < 2:
                continue
            sp0 = staves[grp[0]].sp
            combined = sorted(bx for idx in grp for bx in staves[idx].barlines_x)
            merged: List[float] = []
            for x in combined:
                if not merged or (x - merged[-1]) >= sp0 * 2.5:
                    merged.append(x)
            for idx in grp:
                staves[idx].barlines_x = list(merged)

    @staticmethod
    def _measure_edges(st: _VecStaff) -> List[float]:
        edges = [st.header_end_x] + [bx for bx in st.barlines_x if bx > st.header_end_x + st.sp * 2.0]
        if not edges or (st.x_right - edges[-1]) > st.sp * 3.0:
            edges.append(st.x_right)
        return edges

    @staticmethod
    def _assign_glyph_to_staff(g: _Glyph, staves: List[_VecStaff]) -> Optional[int]:
        best_i = None
        best_d = 1e9
        for i, st in enumerate(staves):
            if g.x < st.header_end_x - st.sp * 0.4 or g.x > st.x_right + st.sp * 1.2:
                continue
            mid_y = 0.5 * (st.y_top + st.y_bot)
            d = abs(g.y - mid_y)
            if d <= st.sp * 7.5 and d < best_d:
                best_d = d
                best_i = i
        return best_i

    @classmethod
    def _build_measure_events(
        cls,
        st: _VecStaff,
        glyphs: List[_Glyph],
        page_index: int,
        staff_index: int,
        measure_number: int,
        mx1: float,
        mx2: float,
        m_beat_start: float,
        beats_per_measure: float,
        pw: float,
        ph: float,
        hand: str,
        system_index: int,
        part_name: str = "",
    ) -> Tuple[MeasureInfo, List[NoteEvent]]:
        sp = st.sp
        m_info = MeasureInfo(
            page_index=page_index,
            staff_index=staff_index,
            measure_number=measure_number,
            x_start_norm=mx1 / pw,
            x_end_norm=mx2 / pw,
            y_top_norm=st.y_top / ph,
            y_bot_norm=st.y_bot / ph,
            beat_start=m_beat_start,
            beat_end=m_beat_start + beats_per_measure,
        )

        heads = [g for g in glyphs if g.code in _ALL_NOTEHEADS and (mx1 - 2.0) <= g.x <= (mx2 + 2.0)]
        accs = [g for g in glyphs if g.code in (_SHARPS | _FLATS | _NATURALS | _DOUBLE_SHARPS | _DOUBLE_FLATS) and (mx1 - sp * 1.5) <= g.x <= (mx2 + 2.0)]
        dots = [g for g in glyphs if g.code in _AUG_DOTS and mx1 <= g.x <= mx2 + sp]

        if not heads:
            cx = 0.5 * (mx1 + mx2)
            cy = 0.5 * (st.y_top + st.y_bot)
            ne = NoteEvent(
                event_index=0,
                page_index=page_index,
                staff_index=staff_index,
                measure_number=measure_number,
                note_index_in_measure=1,
                notes_in_measure=1,
                x_norm=cx / pw,
                y_norm=cy / ph,
                beat_offset=m_beat_start,
                duration_beats=beats_per_measure,
                is_rest=True,
                rest_measures=1,
                note_name="整小節休止",
                note_type="整小節休止",
                clef=st.clef,
                hand=hand,
                system_index=system_index,
                part_name=part_name,
            )
            m_info.notes.append(ne)
            return m_info, [ne]

        # 依水平座標群組垂直和弦（Δx <= 0.72*sp）
        heads.sort(key=lambda g: (g.x, -g.y))
        chords: List[List[_Glyph]] = []
        for h in heads:
            if chords and abs(h.x - chords[-1][0].x) <= sp * 0.72:
                chords[-1].append(h)
            else:
                chords.append([h])

        # 依小節內群組數與音符字形計算相對拍長並歸一化至 beats_per_measure
        raw_weights: List[float] = []
        for grp in chords:
            has_whole = any(g.code in _NOTEHEAD_WHOLE for g in grp)
            has_half = any(g.code in _NOTEHEAD_HALF for g in grp)
            gx = float(np.mean([g.x for g in grp]))
            gy_min = min(g.y for g in grp)
            gy_max = max(g.y for g in grp)
            has_dot = any(
                (sp * 0.45 <= (d.x - gx) <= sp * 2.2) and (gy_min - sp <= d.y <= gy_max + sp)
                for d in dots
            )
            w_val = 4.0 if has_whole else (2.0 if has_half else 1.0)
            if has_dot:
                w_val *= 1.5
            raw_weights.append(w_val)

        # 若小節內包含多個同時垂直聲部（如右手上方二分音符 + 下方三連音八分音符），以 X 位置比例估算起拍點更貼近真實演奏！
        total_w = sum(raw_weights)
        if len(chords) == 1:
            offsets = [0.0]
            durs = [beats_per_measure]
        elif all(abs(w - raw_weights[0]) < 1e-6 for w in raw_weights):
            step_b = beats_per_measure / len(chords)
            offsets = [k * step_b for k in range(len(chords))]
            durs = [step_b for _ in chords]
        else:
            # 依每個和弦中心 x 在小節內之相對位置對齊拍點（精準同步雙手！）
            x_first = float(np.mean([g.x for g in chords[0]]))
            span_x = max(sp * 2.0, mx2 - x_first)
            offsets = []
            for grp in chords:
                gx = float(np.mean([g.x for g in grp]))
                frac = max(0.0, min(0.95, (gx - x_first) / span_x))
                offsets.append(frac * beats_per_measure)
            durs = []
            for k in range(len(chords)):
                nxt_off = offsets[k + 1] if k + 1 < len(chords) else beats_per_measure
                base_d = max(0.15, nxt_off - offsets[k])
                if any(g.code in _NOTEHEAD_WHOLE for g in chords[k]):
                    base_d = max(base_d, beats_per_measure)
                elif any(g.code in _NOTEHEAD_HALF for g in chords[k]):
                    base_d = max(base_d, min(beats_per_measure, 2.0))
                durs.append(base_d)

        # 小節內臨時記號延續狀態 {step: eff_acc}
        measure_acc_state: Dict[int, int] = {}
        built_evs: List[NoteEvent] = []
        total_c = len(chords)

        for k_i, (grp, loc_off, dur_b) in enumerate(zip(chords, offsets, durs), start=1):
            gx = float(np.mean([g.x for g in grp]))
            mids: List[int] = []
            ys_n: List[float] = []
            names: List[str] = []
            is_hollow = any(g.code in (_NOTEHEAD_HALF | _NOTEHEAD_WHOLE) for g in grp)

            for g in sorted(grp, key=lambda item: -item.y):
                step = int(round((st.y_bot - g.y) / (sp * 0.5)))
                # 尋找緊鄰左側的臨時記號
                near_acc = None
                for a in accs:
                    if (sp * 0.35 <= (g.x - a.x) <= sp * 2.6) and abs(a.y - g.y) <= sp * 0.65:
                        near_acc = a
                        break
                show_nat = False
                if near_acc is not None:
                    if near_acc.code in _SHARPS:
                        measure_acc_state[step] = 1
                    elif near_acc.code in _FLATS:
                        measure_acc_state[step] = -1
                    elif near_acc.code in _NATURALS:
                        measure_acc_state[step] = 0
                        show_nat = True
                    elif near_acc.code in _DOUBLE_SHARPS:
                        measure_acc_state[step] = 2
                    elif near_acc.code in _DOUBLE_FLATS:
                        measure_acc_state[step] = -2

                acc_ov = measure_acc_state.get(step, None)
                mv, nm = _step_to_pitch(step, st.clef, st.key_fifths, acc_override=acc_ov, show_natural=show_nat)
                mids.append(mv)
                ys_n.append(g.y / ph)
                names.append(nm)

            rep_midi = mids[-1]
            rep_y = ys_n[-1]
            disp_name = "+".join(names) if len(names) <= 3 else f"{names[0]}..{names[-1]}({len(names)}音)"
            n_type = "全音符 (1/1)" if dur_b >= 3.5 else ("二分音符 (1/2)" if is_hollow else ("八分三連音" if 0.28 <= dur_b <= 0.39 else ("八分音符 (1/8)" if dur_b <= 0.6 else "四分音符 (1/4)")))

            ne = NoteEvent(
                event_index=0,
                page_index=page_index,
                staff_index=staff_index,
                measure_number=measure_number,
                note_index_in_measure=k_i,
                notes_in_measure=total_c,
                x_norm=gx / pw,
                y_norm=rep_y,
                beat_offset=m_beat_start + loc_off,
                duration_beats=dur_b,
                is_rest=False,
                is_hollow=is_hollow,
                midi_note=rep_midi,
                note_name=disp_name,
                note_type=n_type,
                midi_notes=mids,
                note_ys_norm=ys_n,
                clef=st.clef,
                hand=hand,
                system_index=system_index,
                part_name=part_name,
            )
            m_info.notes.append(ne)
            built_evs.append(ne)

        return m_info, built_evs
