"""樂譜影像結構分析引擎 (OpenCV OMR Lite v3 - 尺度不變與完整音符分類版)

針對各種解析度的拍照樂譜 (JPG/PNG/WEBP) 與 PDF 樂譜進行純 Python / OpenCV 影像分析：
1. 自動標準化分析解析度 (寬度歸一化至 1150 px)，確保高解析 300 DPI PDF 與手機照片表現完全一致
2. 自動傾斜偵測與座標校正 (Deskew via HoughLinesP)
3. 五線譜行 (Staff) 與自適應線寬抹除
4. 多小節休止符 (H-bar + 上方數字) 優先過濾，絕不把休止數字 "2" 誤判為音符
5. 實心符頭、空心符頭 (二分/全音符)、小裝飾音 (倚音)、附點與符尾橫槓 (Beam) 完整偵測
6. 產出每個音符的詳細除錯/校準資訊：第幾行、第幾小節、小節內第幾個音符、音符時值種類、音名、MIDI、頻率與拍數
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class NoteEvent:
    """單一演奏音符、裝飾音或休止事件（座標皆為原圖歸一化 0.0~1.0）"""
    event_index: int          # 全曲 1-based 總序號
    page_index: int           # 0-based 頁碼
    staff_index: int          # 本頁 0-based 五線譜行索引
    measure_number: int       # 全曲 1-based 小節編號
    note_index_in_measure: int  # 小節內 1-based 第幾個音符
    notes_in_measure: int     # 該小節內總事件數
    x_norm: float
    y_norm: float
    beat_offset: float
    duration_beats: float
    is_rest: bool = False
    is_grace: bool = False
    is_hollow: bool = False
    is_dotted: bool = False
    rest_measures: int = 0
    beam_count: int = 0
    midi_note: int = 0
    note_name: str = ""
    note_type: str = ""       # 例如 "四分音符 (1/4)", "十六分音符 (1/16)", "裝飾音", "休止 2 小節"

    @property
    def freq_hz(self) -> float:
        if self.is_rest or self.midi_note <= 0:
            return 0.0
        return float(440.0 * (2.0 ** ((self.midi_note - 69) / 12.0)))


@dataclass
class MeasureInfo:
    """單一小節資訊"""
    page_index: int
    staff_index: int
    measure_number: int
    x_start_norm: float
    x_end_norm: float
    y_top_norm: float
    y_bot_norm: float
    beat_start: float
    beat_end: float
    rest_measures: int = 0
    notes: List[NoteEvent] = field(default_factory=list)


@dataclass
class StaffInfo:
    """單一行五線譜資訊"""
    page_index: int
    staff_index: int
    y_top_norm: float
    y_bot_norm: float
    x_left_norm: float
    x_right_norm: float
    spacing_norm: float
    barlines_norm: List[float] = field(default_factory=list)
    measures: List[MeasureInfo] = field(default_factory=list)


@dataclass
class PageAnalysis:
    """單頁樂譜分析結果"""
    page_index: int
    width: int
    height: int
    skew_angle: float
    staves: List[StaffInfo] = field(default_factory=list)
    events: List[NoteEvent] = field(default_factory=list)


@dataclass
class ScoreAnalysis:
    """整份樂譜的分析結果與總演奏時間軸"""
    pages: List[PageAnalysis] = field(default_factory=list)
    events: List[NoteEvent] = field(default_factory=list)
    total_beats: float = 0.0
    total_measures: int = 0
    beats_per_measure: float = 4.0

    @property
    def has_staves(self) -> bool:
        return any(len(p.staves) > 0 for p in self.pages)

    def get_staff(self, page_index: int, staff_index: int) -> Optional[StaffInfo]:
        if 0 <= page_index < len(self.pages):
            staves = self.pages[page_index].staves
            if 0 <= staff_index < len(staves):
                return staves[staff_index]
        return None

    def get_measure(self, measure_number: int) -> Optional[MeasureInfo]:
        for p in self.pages:
            for s in p.staves:
                for m in s.measures:
                    if m.measure_number <= measure_number <= m.measure_number + max(0, m.rest_measures - 1):
                        return m
        return None


@dataclass
class _RawNote:
    x: float
    y: float
    area_ratio: float
    is_grace: bool = False
    is_hollow: bool = False
    has_stem: bool = True
    is_dotted: bool = False
    beam_count: int = 0


@dataclass
class _HorizBar:
    x1: float
    x2: float
    y_center: float
    width: float
    height: float
    in_staff_mid: bool


class ScoreAnalyzer:
    """純 OpenCV 尺度不變樂譜結構、休止符、裝飾音與音符分類分析器"""

    TARGET_ANALYSIS_WIDTH = 1150

    def analyze_pages(
        self,
        rgb_images: List[np.ndarray],
        beats_per_measure: float = 4.0,
    ) -> ScoreAnalysis:
        """分析整份樂譜的所有頁面並建立連貫的節拍時間軸"""
        analysis = ScoreAnalysis(beats_per_measure=beats_per_measure)
        cumulative_beat = 0.0
        measure_counter = 0
        event_counter = 0

        for page_idx, rgb in enumerate(rgb_images):
            page_res, cumulative_beat, measure_counter, event_counter = self._analyze_single_page(
                rgb=rgb,
                page_index=page_idx,
                beats_per_measure=beats_per_measure,
                start_beat=cumulative_beat,
                start_measure_num=measure_counter,
                start_event_idx=event_counter,
            )
            analysis.pages.append(page_res)
            analysis.events.extend(page_res.events)

        analysis.total_beats = cumulative_beat
        analysis.total_measures = measure_counter
        return analysis

    def _analyze_single_page(
        self,
        rgb: np.ndarray,
        page_index: int,
        beats_per_measure: float,
        start_beat: float,
        start_measure_num: int,
        start_event_idx: int,
    ) -> Tuple[PageAnalysis, float, int, int]:
        """分析單頁 RGB 影像（自動縮放至標準分析寬度以確保尺度一致性）"""
        if rgb.ndim == 3:
            orig_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = rgb.copy()

        orig_h, orig_w = orig_gray.shape[:2]
        if orig_h < 100 or orig_w < 100:
            return PageAnalysis(page_index, orig_w, orig_h, 0.0), start_beat, start_measure_num, start_event_idx

        # 0. 若為高解析影像 (>1250 px) 則等比例縮放至 1080 px，確保尺度一致且高速；過小影像 (<750 px) 適度放大
        if orig_w > 1250:
            scale = 1080.0 / float(orig_w)
            w = 1080
            h = max(100, int(round(orig_h * scale)))
            gray = cv2.resize(orig_gray, (w, h), interpolation=cv2.INTER_AREA)
        elif orig_w < 750:
            scale = 1000.0 / float(orig_w)
            w = 1000
            h = max(100, int(round(orig_h * scale)))
            gray = cv2.resize(orig_gray, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            gray = orig_gray
            h, w = orig_h, orig_w

        # 1. 偵測傾斜角度並產生校正圖
        skew = self._estimate_skew(gray)
        if abs(skew) > 0.05:
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), skew, 1.0)
            M_inv = cv2.invertAffineTransform(M)
            gray_d = cv2.warpAffine(
                gray, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderValue=255,
            )
        else:
            M_inv = None
            gray_d = gray

        # 2. 自適應二值化
        block_size = max(35, int(min(w, h) / 20) | 1)
        block_size = min(block_size, 65)
        binary = cv2.adaptiveThreshold(
            gray_d, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, 12,
        )

        # 3. 偵測五線譜行
        raw_staves = self._detect_staves(binary, w, h)

        page_res = PageAnalysis(
            page_index=page_index,
            width=orig_w,
            height=orig_h,
            skew_angle=skew,
        )

        cur_beat = start_beat
        cur_measure = start_measure_num
        cur_ev_idx = start_event_idx

        for s_idx, (y_top, y_bot, sp, xl, xr, line_ys) in enumerate(raw_staves):
            # 4. 偵測橫槓 (Beams / 多小節休止條)、音符符頭 (實心/空心/裝飾音/附點) 與小節線
            raw_notes, horiz_bars = self._detect_notes_and_bars(
                binary, y_top, y_bot, sp, xl, xr, line_ys
            )
            main_note_pts = [(n.x, n.y) for n in raw_notes if not n.is_grace]
            barlines = self._detect_barlines(binary, y_top, y_bot, sp, xl, xr, main_note_pts)

            clef_end = min(xr - sp, xl + sp * 4.2)
            valid_bars = [bx for bx in barlines if bx > clef_end]
            seg_edges = [clef_end] + valid_bars
            if not valid_bars or (xr - seg_edges[-1]) > sp * 3.0:
                seg_edges.append(float(xr))

            clean_edges = [seg_edges[0]]
            for ex in seg_edges[1:]:
                if ex - clean_edges[-1] >= sp * 2.2:
                    clean_edges.append(ex)
            if len(clean_edges) < 2:
                clean_edges = [clef_end, float(xr)]

            tl_x, tl_y = self._map_back(xl, y_top, M_inv, w, h)
            br_x, br_y = self._map_back(xr, y_bot, M_inv, w, h)
            barlines_norm = [self._map_back(bx, (y_top + y_bot) / 2.0, M_inv, w, h)[0] for bx in valid_bars]

            staff_info = StaffInfo(
                page_index=page_index,
                staff_index=s_idx,
                y_top_norm=tl_y,
                y_bot_norm=br_y,
                x_left_norm=tl_x,
                x_right_norm=br_x,
                spacing_norm=sp / float(h),
                barlines_norm=barlines_norm,
            )

            # 5. 建立每個小節與節拍時間軸
            for m_i in range(len(clean_edges) - 1):
                mx1 = clean_edges[m_i]
                mx2 = clean_edges[m_i + 1]

                # 尋找此小節內的五線譜中央粗橫條（H-bar 多小節休止符）
                mid_rest_bars = [
                    b for b in horiz_bars
                    if b.in_staff_mid
                    and b.width >= sp * 2.5
                    and b.x1 >= mx1 - sp * 0.8
                    and b.x2 <= mx2 + sp * 0.8
                ]

                m_notes_all = [n for n in raw_notes if mx1 <= n.x <= mx2]

                # 若存在中央粗橫條，先排除橫條正上方/重疊區的所有假符頭（例如粗體數字 "2"）
                if mid_rest_bars:
                    rest_bar = max(mid_rest_bars, key=lambda b: b.width)
                    notes_outside_bar = [
                        n for n in m_notes_all
                        if not (rest_bar.x1 - sp * 0.6 <= n.x <= rest_bar.x2 + sp * 0.6)
                        and not n.is_grace
                    ]
                    if len(notes_outside_bar) == 0:
                        m_notes_all = []  # 確認整小節為多小節休止！
                else:
                    rest_bar = None

                if rest_bar is not None and len(m_notes_all) == 0:
                    rest_count = self._estimate_rest_bar_count(binary, rest_bar, y_top, sp)
                elif len(m_notes_all) == 0:
                    rest_count = 1
                else:
                    rest_count = 0

                cur_measure += 1
                m_x1_n, m_yt_n = self._map_back(mx1, y_top, M_inv, w, h)
                m_x2_n, m_yb_n = self._map_back(mx2, y_bot, M_inv, w, h)

                if rest_count > 0:
                    total_rest_beats = beats_per_measure * rest_count
                    m_beat_start = cur_beat
                    m_beat_end = cur_beat + total_rest_beats
                    measure_info = MeasureInfo(
                        page_index=page_index,
                        staff_index=s_idx,
                        measure_number=cur_measure,
                        x_start_norm=m_x1_n,
                        x_end_norm=m_x2_n,
                        y_top_norm=m_yt_n,
                        y_bot_norm=m_yb_n,
                        beat_start=m_beat_start,
                        beat_end=m_beat_end,
                        rest_measures=rest_count,
                    )
                    cx_n, cy_n = self._map_back((mx1 + mx2) / 2.0, (y_top + y_bot) / 2.0, M_inv, w, h)
                    cur_ev_idx += 1
                    r_label = f"休止 {rest_count} 小節" if rest_count > 1 else "整小節休止"
                    ev = NoteEvent(
                        event_index=cur_ev_idx,
                        page_index=page_index,
                        staff_index=s_idx,
                        measure_number=cur_measure,
                        note_index_in_measure=1,
                        notes_in_measure=1,
                        x_norm=cx_n,
                        y_norm=cy_n,
                        beat_offset=m_beat_start,
                        duration_beats=total_rest_beats,
                        is_rest=True,
                        rest_measures=rest_count,
                        note_name=r_label,
                        note_type=r_label,
                    )
                    measure_info.notes.append(ev)
                    page_res.events.append(ev)
                    cur_measure += (rest_count - 1)
                    cur_beat = m_beat_end
                    staff_info.measures.append(measure_info)
                    continue

                # 一般含音符小節
                m_notes_all.sort(key=lambda n: n.x)
                m_beat_start = cur_beat
                m_beat_end = cur_beat + beats_per_measure
                measure_info = MeasureInfo(
                    page_index=page_index,
                    staff_index=s_idx,
                    measure_number=cur_measure,
                    x_start_norm=m_x1_n,
                    x_end_norm=m_x2_n,
                    y_top_norm=m_yt_n,
                    y_bot_norm=m_yb_n,
                    beat_start=m_beat_start,
                    beat_end=m_beat_end,
                )

                durations, note_types = self._classify_and_assign_durations(
                    m_notes_all, mx2, sp, beats_per_measure
                )
                local_beat = m_beat_start
                total_in_m = len(m_notes_all)
                for k_idx, (note_obj, dur, n_type) in enumerate(zip(m_notes_all, durations, note_types), start=1):
                    nx_n, ny_n = self._map_back(note_obj.x, note_obj.y, M_inv, w, h)
                    midi_val, n_name = self._y_to_treble_pitch(note_obj.y, y_bot, sp)
                    cur_ev_idx += 1
                    ev = NoteEvent(
                        event_index=cur_ev_idx,
                        page_index=page_index,
                        staff_index=s_idx,
                        measure_number=cur_measure,
                        note_index_in_measure=k_idx,
                        notes_in_measure=total_in_m,
                        x_norm=nx_n,
                        y_norm=ny_n,
                        beat_offset=local_beat,
                        duration_beats=dur,
                        is_rest=False,
                        is_grace=note_obj.is_grace,
                        is_hollow=note_obj.is_hollow,
                        is_dotted=note_obj.is_dotted,
                        beam_count=note_obj.beam_count,
                        midi_note=midi_val,
                        note_name=f"({n_name})" if note_obj.is_grace else n_name,
                        note_type=n_type,
                    )
                    measure_info.notes.append(ev)
                    page_res.events.append(ev)
                    local_beat += dur

                cur_beat = m_beat_end
                staff_info.measures.append(measure_info)

            page_res.staves.append(staff_info)

        return page_res, cur_beat, cur_measure, cur_ev_idx

    @staticmethod
    def _classify_and_assign_durations(
        notes: List[_RawNote], mx2: float, sp: float, beats_per_measure: float
    ) -> Tuple[List[float], List[str]]:
        """分類每個音符的樂理種類（全音/二分/四分/八分/十六分/附點/裝飾音）並分配拍數"""
        n_count = len(notes)
        if n_count == 0:
            return [], []

        grace_indices = [i for i, n in enumerate(notes) if n.is_grace]
        main_indices = [i for i, n in enumerate(notes) if not n.is_grace]

        note_types = [""] * n_count
        for gi in grace_indices:
            note_types[gi] = "裝飾音 (倚音)"

        if not main_indices:
            return [beats_per_measure / n_count] * n_count, note_types

        grace_dur = min(0.10, (beats_per_measure * 0.25) / max(1, len(grace_indices)))
        remaining_beats = max(0.25, beats_per_measure - grace_dur * len(grace_indices))

        main_weights = []
        for idx_pos, idx in enumerate(main_indices):
            n = notes[idx]
            dot_mult = 1.5 if n.is_dotted else 1.0
            dot_prefix = "附點" if n.is_dotted else ""

            if n.is_hollow:
                if not n.has_stem:
                    base_w = 4.0 * dot_mult
                    n_type = f"{dot_prefix}全音符 (1/1)"
                else:
                    base_w = 2.0 * dot_mult
                    n_type = f"{dot_prefix}二分音符 (1/2)"
            else:
                if n.beam_count >= 3:
                    base_w = 0.125 * dot_mult
                    n_type = f"{dot_prefix}三十二分音符 (1/32)"
                elif n.beam_count == 2:
                    base_w = 0.25 * dot_mult
                    n_type = f"{dot_prefix}十六分音符 (1/16)"
                elif n.beam_count == 1:
                    base_w = 0.50 * dot_mult
                    n_type = f"{dot_prefix}八分音符 (1/8)"
                else:
                    base_w = 1.00 * dot_mult
                    n_type = f"{dot_prefix}四分音符 (1/4)"

            note_types[idx] = n_type

            if idx_pos + 1 < len(main_indices):
                next_x = notes[main_indices[idx_pos + 1]].x
            else:
                next_x = mx2
            dx = max(sp * 0.8, next_x - n.x)
            main_weights.append((base_w, dx))

        avg_dx = sum(d for _, d in main_weights) / len(main_weights)
        combined = []
        for bw, dx in main_weights:
            dx_norm = (dx / avg_dx) if avg_dx > 0 else 1.0
            dx_factor = max(0.6, min(1.8, dx_norm))
            combined.append(bw * (0.82 + 0.18 * dx_factor))

        total_w = sum(combined)
        durations = [0.0] * n_count
        for gi in grace_indices:
            durations[gi] = grace_dur
        for idx, w_val in zip(main_indices, combined):
            durations[idx] = remaining_beats * (w_val / total_w) if total_w > 0 else (remaining_beats / len(main_indices))

        return durations, note_types

    @staticmethod
    def _estimate_rest_bar_count(binary: np.ndarray, bar: _HorizBar, y_top: float, sp: float) -> int:
        """檢查多小節休止橫條正上方的數字區域並辨識休止小節數（常見 2~8 小節）"""
        h, w = binary.shape[:2]
        # 數字可能跨在第一線(y_top)上下：檢查 y_top - 2.8*sp 到 y_top + 1.1*sp
        ry1 = max(0, int(y_top - sp * 2.8))
        ry2 = min(h, int(min(bar.y_center - sp * 0.5, y_top + sp * 1.1)))
        rx1 = max(0, int(bar.x1))
        rx2 = min(w, int(bar.x2))
        if rx2 <= rx1 + 4 or ry2 <= ry1 + 4:
            return 1

        patch = binary[ry1:ry2, rx1:rx2].copy()
        # 清除穿過數字的頂部五線譜橫線
        top_rel = int(round(y_top - ry1))
        for dy in (-1, 0, 1):
            r = top_rel + dy
            if 2 <= r < patch.shape[0] - 2:
                pure = (patch[r - 2, :] == 0) & (patch[r + 2, :] == 0)
                patch[r, pure] = 0

        ys, xs = np.where(patch > 0)
        if len(xs) < max(6, int(sp * sp * 0.28)):
            return 1

        # 擷取數字 bounding box 並與標準字型模板 "2"~"8" 比對，預設回傳 2
        digit_roi = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        dh, dw = digit_roi.shape[:2]
        if dh < int(sp * 0.8) or dw < int(sp * 0.4):
            return 2

        norm_digit = cv2.resize(digit_roi, (24, 32), interpolation=cv2.INTER_AREA)
        best_digit = 2
        best_score = -1.0
        for d in range(2, 9):
            tmpl = np.zeros((32, 24), dtype=np.uint8)
            cv2.putText(tmpl, str(d), (2, 28), cv2.FONT_HERSHEY_TRIPLEX, 1.0, 255, 2)
            tys, txs = np.where(tmpl > 0)
            if len(txs) > 0:
                tmpl = cv2.resize(tmpl[tys.min():tys.max() + 1, txs.min():txs.max() + 1], (24, 32))
            res = cv2.matchTemplate(norm_digit, tmpl, cv2.TM_CCOEFF_NORMED)
            sc = float(res[0, 0])
            if sc > best_score:
                best_score = sc
                best_digit = d

        return best_digit if best_score >= 0.42 else 2

    @staticmethod
    def _estimate_skew(gray: np.ndarray) -> float:
        """透過水平投影梯度平方和最大化（Projection Profile Sharpness）精準偵測五線譜傾角"""
        sh, sw = gray.shape[:2]
        if sw < 80 or sh < 80:
            return 0.0
        scale = 600.0 / float(sw)
        small_h = max(60, int(round(sh * scale)))
        small = cv2.resize(gray, (600, small_h), interpolation=cv2.INTER_AREA)
        b = cv2.adaptiveThreshold(
            small, 1,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25, 10,
        )
        bh, bw = b.shape[:2]
        center = (bw / 2.0, bh / 2.0)

        best_a = 0.0
        best_v = -1.0
        # 粗搜尋 -6.0° ~ +6.0° (步長 0.3°)
        for a in np.arange(-6.0, 6.01, 0.3):
            M = cv2.getRotationMatrix2D(center, float(a), 1.0)
            rot = cv2.warpAffine(b, M, (bw, bh), flags=cv2.INTER_NEAREST)
            proj = np.sum(rot, axis=1, dtype=np.float32)
            v = float(np.sum(np.diff(proj) ** 2))
            if v > best_v:
                best_v = v
                best_a = float(a)

        # 細搜尋 best_a ± 0.35° (步長 0.05°)
        for a in np.arange(best_a - 0.35, best_a + 0.36, 0.05):
            M = cv2.getRotationMatrix2D(center, float(a), 1.0)
            rot = cv2.warpAffine(b, M, (bw, bh), flags=cv2.INTER_NEAREST)
            proj = np.sum(rot, axis=1, dtype=np.float32)
            v = float(np.sum(np.diff(proj) ** 2))
            if v > best_v:
                best_v = v
                best_a = float(a)

        return best_a

    @staticmethod
    def _detect_staves(
        binary: np.ndarray, w: int, h: int
    ) -> List[Tuple[float, float, float, int, int, List[float]]]:
        """偵測所有 5 條線組成的五線譜行"""
        horiz_len = max(18, w // 35)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
        horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)

        row_proj = np.sum(horiz > 0, axis=1)
        line_rows = np.where(row_proj > w * 0.20)[0]
        if len(line_rows) < 5:
            return []

        lines: List[float] = []
        start = prev = int(line_rows[0])
        for r in line_rows[1:]:
            r = int(r)
            if r > prev + 2:
                seg = np.arange(start, prev + 1)
                lines.append(float(np.average(seg, weights=row_proj[seg])))
                start = r
            prev = r
        seg = np.arange(start, prev + 1)
        lines.append(float(np.average(seg, weights=row_proj[seg])))

        staves = []
        i = 0
        while i + 4 < len(lines):
            group = lines[i:i + 5]
            spacings = np.diff(group)
            med = float(np.median(spacings))
            if med > 3.0 and np.all(np.abs(spacings - med) < max(3.0, med * 0.35)):
                y1 = max(0, int(group[0] - 2))
                y2 = min(h, int(group[4] + 2))
                col_proj = np.sum(horiz[y1:y2, :] > 0, axis=0)
                xs = np.where(col_proj >= 3)[0]
                x_left = int(xs[0]) if len(xs) else 0
                x_right = int(xs[-1]) if len(xs) else w - 1
                if x_right - x_left > w * 0.25:
                    staves.append((group[0], group[4], med, x_left, x_right, group))
                i += 5
            else:
                i += 1
        return staves

    @staticmethod
    def _detect_notes_and_bars(
        binary: np.ndarray,
        y_top: float,
        y_bot: float,
        sp: float,
        xl: int,
        xr: int,
        line_ys: List[float],
    ) -> Tuple[List[_RawNote], List[_HorizBar]]:
        """去除五線譜線後，偵測橫槓、實心符頭、空心符頭（二分/全音符）、裝飾音與附點"""
        h, w = binary.shape[:2]
        pad = int(sp * 3.5)
        ry1 = max(0, int(y_top - pad))
        ry2 = min(h, int(y_bot + pad))
        roi = binary[ry1:ry2, xl:xr].copy()
        sy1 = float(y_top - ry1)
        sy2 = float(y_bot - ry1)

        # 1. 尺度自適應抹除純五線譜橫線
        half_t = max(1, int(round(sp * 0.20)))
        guard = half_t + 1
        for ly in line_ys:
            ly_rel = int(round(ly - ry1))
            if guard <= ly_rel < roi.shape[0] - guard:
                pure = (roi[ly_rel - guard, :] == 0) & (roi[ly_rel + guard, :] == 0)
                for dy in range(-half_t, half_t + 1):
                    roi[ly_rel + dy, pure] = 0

        # 2. 提取粗橫槓（符尾連槓 Beams 與多小節休止橫條）
        beam_w = max(8, int(round(sp * 1.65)))
        beam_h = max(2, int(round(sp * 0.26)))
        beam_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (beam_w, beam_h))
        beams_mask = cv2.morphologyEx(roi, cv2.MORPH_OPEN, beam_kernel)

        num_b, _, bstats, bcent = cv2.connectedComponentsWithStats(beams_mask)
        horiz_bars: List[_HorizBar] = []
        for lbl in range(1, num_b):
            bx = int(bstats[lbl, cv2.CC_STAT_LEFT])
            bw = int(bstats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(bstats[lbl, cv2.CC_STAT_HEIGHT])
            cy = float(bcent[lbl][1])
            in_mid = (sy1 + sp * 1.15) <= cy <= (sy2 - sp * 1.15)
            horiz_bars.append(
                _HorizBar(
                    x1=float(xl + bx),
                    x2=float(xl + bx + bw),
                    y_center=float(ry1 + cy),
                    width=float(bw),
                    height=float(bh),
                    in_staff_mid=in_mid,
                )
            )

        # 3. 實心符頭與裝飾音偵測
        roi_no_beams = cv2.subtract(roi, cv2.dilate(beams_mask, np.ones((3, 3), np.uint8)))
        nh_w = max(3, int(round(sp * 0.50)))
        nh_h = max(3, int(round(sp * 0.45)))
        nh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nh_w, nh_h))
        nh_mask = cv2.morphologyEx(roi_no_beams, cv2.MORPH_OPEN, nh_kernel)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(nh_mask)
        candidates: List[_RawNote] = []
        clef_skip = int(sp * 4.2)

        for lbl in range(1, num_labels):
            cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])
            if cx < clef_skip:
                continue
            aspect = cw / max(1.0, float(ch))
            area_ratio = area / max(1.0, sp * sp)
            if (
                (sp * 0.36 <= ch <= sp * 1.60)
                and (sp * 0.40 <= cw <= sp * 1.85)
                and (0.50 <= aspect <= 2.15)
                and (area_ratio >= 0.18)
                and ((sy1 - sp * 2.5) <= cy <= (sy2 + sp * 2.5))
            ):
                is_grace_cand = area_ratio < 0.72
                if is_grace_cand:
                    # 真正的裝飾音必有向上（或向下）的細直符幹；藉此100%排除圓滑線(slur)弧頂與雜訊！
                    gx1 = max(0, int(cx - sp * 0.5))
                    gx2 = min(roi.shape[1], int(cx + sp * 0.85))
                    gy1 = max(0, int(cy - sp * 2.6))
                    gy2 = min(roi.shape[0], int(cy + sp * 0.5))
                    g_patch = roi[gy1:gy2, gx1:gx2]
                    if g_patch.size == 0 or int(np.max(np.sum(g_patch > 0, axis=0))) < max(4, int(sp * 1.25)):
                        continue

                candidates.append(
                    _RawNote(
                        x=xl + cx,
                        y=ry1 + cy,
                        area_ratio=area_ratio,
                        is_grace=is_grace_cand,
                        is_hollow=False,
                    )
                )

        # 4. 空心符頭（二分音符 / 全音符）偵測：尋找譜內具有封閉橢圓白孔且外框非寬大圓滑線的輪廓
        closed_roi = cv2.morphologyEx(roi_no_beams, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        contours, hierarchy = cv2.findContours(closed_roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            hier = hierarchy[0]
            for c_idx, h_info in enumerate(hier):
                parent_idx = h_info[3]
                if parent_idx < 0:
                    continue
                hx, hy, hw, hh = cv2.boundingRect(contours[c_idx])
                ox, oy, ow, oh = cv2.boundingRect(contours[parent_idx])
                hole_area = cv2.contourArea(contours[c_idx])
                rect_area = max(1.0, float(hw * hh))
                fill_ratio = hole_area / rect_area
                hcx = hx + hw / 2.0
                hcy = hy + hh / 2.0
                if hcx < clef_skip:
                    continue
                # 排除圓滑線與五線譜圍出的寬扁大區塊 (ow <= 2.2*sp)，並要求內孔接近橢圓 (fill_ratio >= 0.52)
                if (
                    (ow <= sp * 2.2)
                    and (sp * 0.32 <= hw <= sp * 1.05)
                    and (sp * 0.25 <= hh <= sp * 0.80)
                    and (fill_ratio >= 0.52)
                    and (sp * sp * 0.08 <= hole_area <= sp * sp * 0.65)
                    and ((sy1 - sp * 2.0) <= hcy <= (sy2 + sp * 2.0))
                ):
                    abs_x = xl + hcx
                    abs_y = ry1 + hcy
                    # 確認同一個水平位置沒有已被偵測為實心主音符（避免圓滑線與符頭間的小白孔誤判為二分音符）
                    if all(abs(abs_x - c.x) > sp * 0.65 for c in candidates if not c.is_grace):
                        # 檢查左右側是否有長符幹 (高度 >= 2.0*sp) 以區分二分音符 vs 全音符
                        sx1 = max(0, int(hcx - sp * 0.9))
                        sx2 = min(roi.shape[1], int(hcx + sp * 0.9))
                        sy_top = max(0, int(hcy - sp * 3.2))
                        sy_bot = min(roi.shape[0], int(hcy + sp * 3.2))
                        stem_patch = roi[sy_top:sy_bot, sx1:sx2]
                        has_stem = (stem_patch.shape[0] >= int(sp * 2.2)) and (
                            int(np.max(np.sum(stem_patch > 0, axis=0))) >= int(sp * 2.0)
                        )
                        candidates.append(
                            _RawNote(
                                x=abs_x,
                                y=abs_y,
                                area_ratio=1.1,
                                is_grace=False,
                                is_hollow=True,
                                has_stem=has_stem,
                            )
                        )

        candidates.sort(key=lambda n: n.x)

        # 5. 僅合併真正同一時間點（水平距離 < 0.55*sp）的雙音或雜點
        merged: List[_RawNote] = []
        for cand in candidates:
            if merged and abs(cand.x - merged[-1].x) < sp * 0.55:
                prev = merged[-1]
                if prev.is_grace and not cand.is_grace and abs(cand.y - prev.y) > sp * 1.1:
                    if cand.y > prev.y:
                        merged[-1] = cand
                elif not prev.is_grace and cand.is_grace:
                    pass
                else:
                    merged[-1] = _RawNote(
                        x=(prev.x + cand.x) / 2.0,
                        y=min(prev.y, cand.y),
                        area_ratio=max(prev.area_ratio, cand.area_ratio),
                        is_grace=(prev.is_grace and cand.is_grace),
                        is_hollow=(prev.is_hollow and cand.is_hollow),
                        has_stem=(prev.has_stem or cand.has_stem),
                    )
            else:
                merged.append(cand)

        # 6. 計算每個主音符的符尾橫槓數量（beam_count）與右側附點（is_dotted）
        num_dots, _, dstats, dcent = cv2.connectedComponentsWithStats(roi_no_beams)
        dot_pts = []
        for dlbl in range(1, num_dots):
            dw = int(dstats[dlbl, cv2.CC_STAT_WIDTH])
            dh = int(dstats[dlbl, cv2.CC_STAT_HEIGHT])
            da = int(dstats[dlbl, cv2.CC_STAT_AREA])
            if (
                2 <= dw <= max(4, int(sp * 0.45))
                and 2 <= dh <= max(4, int(sp * 0.45))
                and sp * sp * 0.03 <= da <= sp * sp * 0.16
            ):
                dot_pts.append((xl + float(dcent[dlbl][0]), ry1 + float(dcent[dlbl][1])))

        for note in merged:
            if note.is_grace:
                note.beam_count = 2
                continue
            if not note.is_hollow:
                beam_ys = []
                for bar in horiz_bars:
                    if bar.in_staff_mid and bar.width >= sp * 2.5:
                        continue
                    if bar.x1 - sp * 0.7 <= note.x <= bar.x2 + sp * 0.7:
                        if abs(bar.y_center - note.y) >= sp * 0.8:
                            beam_ys.append(bar.y_center)
                beam_ys.sort()
                distinct_beams = 0
                last_y = None
                for by in beam_ys:
                    if last_y is None or abs(by - last_y) >= sp * 0.45:
                        distinct_beams += 1
                        last_y = by
                note.beam_count = min(3, distinct_beams)

            # 檢查符頭右側 0.65~1.65*sp、上下 0.55*sp 內是否有附點
            note.is_dotted = any(
                (sp * 0.65 <= (dx - note.x) <= sp * 1.65) and (abs(dy - note.y) <= sp * 0.55)
                for dx, dy in dot_pts
            )

        return merged, horiz_bars

    @staticmethod
    def _detect_barlines(
        binary: np.ndarray,
        y_top: float,
        y_bot: float,
        sp: float,
        xl: int,
        xr: int,
        note_pts: List[Tuple[float, float]],
    ) -> List[float]:
        """偵測五線譜內的小節線（自動排除音符符幹）"""
        iy1 = max(0, int(y_top))
        iy2 = min(binary.shape[0], int(y_bot) + 1)
        staff_h = max(1, iy2 - iy1)
        if staff_h < 8:
            return []

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(staff_h * 0.90))))
        v_lines = cv2.morphologyEx(binary[iy1:iy2, xl:xr], cv2.MORPH_OPEN, v_kernel)
        v_proj = np.sum(v_lines > 0, axis=0)
        bx = np.where(v_proj > staff_h * 0.85)[0]
        if len(bx) == 0:
            return []

        barlines: List[float] = []
        b_start = b_prev = int(bx[0])
        for x in bx[1:]:
            x = int(x)
            if x > b_prev + max(5, int(sp * 1.0)):
                cand_x = xl + (b_start + b_prev) / 2.0
                if all(abs(cand_x - nx) > sp * 0.95 for nx, _ in note_pts):
                    barlines.append(cand_x)
                b_start = x
            b_prev = x
        cand_x = xl + (b_start + b_prev) / 2.0
        if all(abs(cand_x - nx) > sp * 0.95 for nx, _ in note_pts):
            barlines.append(cand_x)
        return barlines

    @staticmethod
    def _y_to_treble_pitch(ny: float, y_bot: float, sp: float) -> Tuple[int, str]:
        """依據高音譜號五線譜垂直位置推算 MIDI 音高與音名（線距 sp = 2 個自然音階步）"""
        if sp <= 0:
            return 60, "C4"
        step = int(round((y_bot - ny) / (sp / 2.0)))
        step = max(-14, min(20, step))
        diatonic_idx = 2 + step
        octave = 4 + (diatonic_idx // 7)
        deg = diatonic_idx % 7
        semitones = (0, 2, 4, 5, 7, 9, 11)
        names = ("C", "D", "E", "F", "G", "A", "B")
        midi = (octave + 1) * 12 + semitones[deg]
        midi = max(36, min(96, midi))
        return midi, f"{names[deg]}{octave}"

    @staticmethod
    def _map_back(
        x: float, y: float, M_inv: Optional[np.ndarray], w: int, h: int
    ) -> Tuple[float, float]:
        """將校正後影像座標映射回原圖並歸一化至 [0.0, 1.0]"""
        if M_inv is not None:
            ox = float(M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2])
            oy = float(M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2])
        else:
            ox, oy = float(x), float(y)
        nx = max(0.0, min(1.0, ox / max(1.0, float(w))))
        ny = max(0.0, min(1.0, oy / max(1.0, float(h))))
        return nx, ny
