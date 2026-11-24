"""樂譜影像結構分析引擎 (OpenCV OMR Lite v4 - 去底色強化與精準符頭切分版)

核心處理流程：
1. 去底色與對比強化 (Division Illumination Normalization)：消除泛黃紙色、手機拍照陰影與背面透字
2. 尺度標準化 (寬度統一至 1200 px) 與水平投影梯度最大化傾斜校正 (Deskew)
3. 五線譜偵測與譜頭自動跳過 (自動辨識並略過譜號 Clef 與拍號如 2/4，絕不誤判為音符)
4. 全域細橫線抹除 (移除五線譜線與上/下加線翅膀，同時 100% 保留加線上的高音符頭如 A5/B5/C6)
5. 細直符幹剝離 (Stem Removal) + 跨雙幹短橫槓過濾 (Stem-Bridging Filter)：切出每顆獨立音符與雙裝飾音
6. 反覆記號雙點 (|: / :|) 與多小節休止符 (H-bar + 數字 2..8) 精準辨識
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class NoteEvent:
    """單一演奏音符、裝飾音或休止事件（座標皆為原圖歸一化 0.0~1.0）"""
    event_index: int
    page_index: int
    staff_index: int
    measure_number: int
    note_index_in_measure: int
    notes_in_measure: int
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
    note_type: str = ""

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


@dataclass
class _RawNote:
    x: float
    y: float
    width: float
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
    """純 OpenCV 去底色強化與精準音符切分引擎"""

    TARGET_WIDTH = 1200

    @staticmethod
    def enhance_score_gray(gray: np.ndarray) -> np.ndarray:
        """去底色與對比強化：消除泛黃紙張、拍照陰影漸層與背面透字，輸出純白底深黑墨跡灰階圖"""
        h, w = gray.shape[:2]
        ref_dim = max(w, h)
        # 膨脹核大於實心符頭與多小節休止橫條厚度（1200px 時約 15px），避免黑塊被當背景掏空
        k_dil = max(15, int(round(ref_dim / 80)) | 1)
        k_blur = max(51, int(round(ref_dim / 22)) | 1)
        bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil, k_dil)))
        bg = cv2.GaussianBlur(bg, (k_blur, k_blur), 0)
        bg = np.maximum(bg, 16)
        norm = cv2.divide(gray, bg, scale=255)
        # 壓制 >220 的紙張底色與背面透字至純白 255，同時將墨跡拉深至純黑
        enhanced = np.clip((norm.astype(np.float32) - 132.0) * (255.0 / (220.0 - 132.0)), 0.0, 255.0)
        return enhanced.astype(np.uint8)

    @classmethod
    def enhance_score_rgb(cls, rgb: np.ndarray) -> np.ndarray:
        """將輸入彩色樂譜去底色強化為清爽白底黑字 RGB 影像（供 UI 顯示）"""
        if rgb.ndim == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb
        clean = cls.enhance_score_gray(gray)
        return cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB)

    def analyze_pages(
        self,
        rgb_images: List[np.ndarray],
        beats_per_measure: float = 4.0,
    ) -> ScoreAnalysis:
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
        if rgb.ndim == 3:
            orig_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = rgb.copy()

        orig_h, orig_w = orig_gray.shape[:2]
        if orig_h < 100 or orig_w < 100:
            return PageAnalysis(page_index, orig_w, orig_h, 0.0), start_beat, start_measure_num, start_event_idx

        # 1. 標準化分析寬度至 1200 px，確保所有形態學核心尺度一致
        if abs(orig_w - self.TARGET_WIDTH) > 80:
            scale = self.TARGET_WIDTH / float(orig_w)
            w = self.TARGET_WIDTH
            h = max(100, int(round(orig_h * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            gray = cv2.resize(orig_gray, (w, h), interpolation=interp)
        else:
            gray = orig_gray
            h, w = orig_h, orig_w

        # 2. 先執行去底色與對比強化！消除紙色、陰影與透字
        clean_gray = self.enhance_score_gray(gray)

        # 3. 偵測傾斜角度並校正
        skew = self._estimate_skew(clean_gray)
        if abs(skew) > 0.05:
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), skew, 1.0)
            M_inv = cv2.invertAffineTransform(M)
            clean_d = cv2.warpAffine(
                clean_gray, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderValue=255,
            )
        else:
            M_inv = None
            clean_d = clean_gray

        # 去底色後 >220 之紙張底色與淡透印皆為純白 255；門檻 210 完整保留細五線譜線、十六分音符與裝飾音符頭
        binary = ((clean_d < 210).astype(np.uint8)) * 255

        # 4. 偵測五線譜行
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
            # 5. 自動偵測譜頭區（譜號 + 調號 + 拍號如 2/4）結束位置，並切分所有獨立音符與橫槓
            raw_notes, horiz_bars, header_end_x = self._detect_notes_and_bars(
                binary, y_top, y_bot, sp, xl, xr
            )
            main_note_pts = [(n.x, n.y) for n in raw_notes if not n.is_grace]
            barlines = self._detect_barlines(binary, y_top, y_bot, sp, xl, xr, main_note_pts)

            # 6. 過濾緊鄰小節線的反覆記號雙點 (|: / :|，寬度 < 0.85*sp 且距小節線 < 1.45*sp)
            if barlines:
                raw_notes = [
                    n for n in raw_notes
                    if not (
                        n.width < sp * 0.85
                        and any(abs(n.x - bx) < sp * 1.45 for bx in barlines)
                    )
                ]

            valid_bars = [bx for bx in barlines if bx > header_end_x + sp * 1.2]
            seg_edges = [header_end_x] + valid_bars
            if not valid_bars or (xr - seg_edges[-1]) > sp * 3.0:
                seg_edges.append(float(xr))

            clean_edges = [seg_edges[0]]
            for ex in seg_edges[1:]:
                if ex - clean_edges[-1] >= sp * 2.2:
                    clean_edges.append(ex)
            if len(clean_edges) < 2:
                clean_edges = [header_end_x, float(xr)]

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

            # 7. 建立每個小節與節拍時間軸
            for m_i in range(len(clean_edges) - 1):
                mx1 = clean_edges[m_i]
                mx2 = clean_edges[m_i + 1]

                # 尋找此小節內的五線譜中央粗橫條（H-bar 多小節休止符：須位於小節內部且非整條譜線）
                m_width = max(1.0, mx2 - mx1)
                mid_rest_bars = [
                    b for b in horiz_bars
                    if b.in_staff_mid
                    and sp * 2.2 <= b.width <= min(sp * 9.5, m_width * 0.80)
                    and b.x1 >= mx1 + sp * 0.5
                    and b.x2 <= mx2 - sp * 0.5
                ]

                m_notes_all = [n for n in raw_notes if mx1 <= n.x <= mx2]

                # 若存在中央休止粗橫條，僅排除橫條正上方且無符幹的休止數字（如跨在第一線間的 "2"），絕不刪除有符幹的真實音符！
                if mid_rest_bars:
                    rest_bar = max(mid_rest_bars, key=lambda b: b.width)
                    real_notes_in_m = [
                        n for n in m_notes_all
                        if n.has_stem
                        or n.y >= rest_bar.y_center
                        or not (rest_bar.x1 - sp * 0.6 <= n.x <= rest_bar.x2 + sp * 0.6)
                    ]
                    if len(real_notes_in_m) == 0:
                        m_notes_all = []
                    else:
                        rest_bar = None
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
        """分類每個音符的樂理種類並分配拍數"""
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

        grace_dur = min(0.10, (beats_per_measure * 0.24) / max(1, len(grace_indices)))
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
        """檢查多小節休止橫條正上方的數字區域並辨識休止小節數"""
        h, w = binary.shape[:2]
        ry1 = max(0, int(y_top - sp * 2.8))
        ry2 = min(h, int(min(bar.y_center - sp * 0.45, y_top + sp * 1.15)))
        rx1 = max(0, int(bar.x1))
        rx2 = min(w, int(bar.x2))
        if rx2 <= rx1 + 4 or ry2 <= ry1 + 4:
            return 1

        patch = binary[ry1:ry2, rx1:rx2].copy()
        pw = patch.shape[1]
        # 僅濾除橫跨休止橫條寬度 >= 72% 的五線譜細橫線，完整保留細體或襯線數字筆畫
        line_w = max(8, int(pw * 0.72))
        line_mask = cv2.morphologyEx(patch, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (line_w, 1)))
        vert_keep = cv2.morphologyEx(patch, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(round(sp * 0.30))))))
        vert_keep = cv2.dilate(vert_keep, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        patch_clean = cv2.subtract(patch, cv2.subtract(line_mask, vert_keep))

        ys, xs = np.where(patch_clean > 0)
        if len(xs) < max(5, int(sp * sp * 0.16)):
            return 1

        digit_roi = patch_clean[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        dh, dw = digit_roi.shape[:2]
        if dh < int(sp * 0.7) or dw < int(sp * 0.35):
            return 2

        # 音樂排版字體數字結構判別（避免 OpenCV 無襯線字型把粗襯線 "2" 誤認為 "8"）：
        # "2" 的特徵：底部 22% 為寬平實心橫底（填充率 >= 0.72），且下半部無封閉圓孔
        bot_band = digit_roi[max(0, int(dh * 0.78)):, :]
        bot_fill = float(np.mean(bot_band > 0)) if bot_band.size > 0 else 0.0

        # 計算內部封閉白孔數 (holes)：8 有 2 孔，6/4/0 有 1 孔，2/3/5/7 通常 0 孔（或頂部極小彎孔）
        padded = cv2.copyMakeBorder(digit_roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        _, hier = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes_bot = 0
        if hier is not None:
            contours, _ = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for c_i, h_i in enumerate(hier[0]):
                if h_i[3] >= 0 and cv2.contourArea(contours[c_i]) >= 4:
                    _, hy, _, hh = cv2.boundingRect(contours[c_i])
                    if (hy + hh / 2.0) > padded.shape[0] * 0.52:
                        holes_bot += 1

        if bot_fill >= 0.70 and holes_bot == 0:
            return 2

        norm_digit = cv2.resize(digit_roi, (24, 32), interpolation=cv2.INTER_AREA)
        best_digit = 2
        best_score = -1.0
        for d in range(2, 9):
            if d == 8 and holes_bot == 0:
                continue
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

        return best_digit if best_score >= 0.50 else 2

    @staticmethod
    def _estimate_skew(clean_gray: np.ndarray) -> float:
        """透過水平投影梯度平方和最大化精準偵測五線譜傾角"""
        sh, sw = clean_gray.shape[:2]
        if sw < 80 or sh < 80:
            return 0.0
        scale = 600.0 / float(sw)
        small_h = max(60, int(round(sh * scale)))
        small = cv2.resize(clean_gray, (600, small_h), interpolation=cv2.INTER_AREA)
        b = (small < 180).astype(np.uint8)
        bh, bw = b.shape[:2]
        center = (bw / 2.0, bh / 2.0)

        best_a = 0.0
        best_v = -1.0
        for a in np.arange(-6.0, 6.01, 0.3):
            M = cv2.getRotationMatrix2D(center, float(a), 1.0)
            rot = cv2.warpAffine(b, M, (bw, bh), flags=cv2.INTER_NEAREST)
            proj = np.sum(rot, axis=1, dtype=np.float32)
            v = float(np.sum(np.diff(proj) ** 2))
            if v > best_v:
                best_v = v
                best_a = float(a)

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
    ) -> Tuple[List[_RawNote], List[_HorizBar], float]:
        """移除細橫線與細直幹，自動略過譜頭拍號 (2/4)，切出每顆獨立音符與橫槓"""
        h, w = binary.shape[:2]
        pad = int(sp * 3.5)
        ry1 = max(0, int(y_top - pad))
        ry2 = min(h, int(y_bot + pad))
        roi = binary[ry1:ry2, xl:xr].copy()
        sy1 = float(y_top - ry1)
        sy2 = float(y_bot - ry1)

        # 1. 量測此行五線譜線實際粗細，全域抹除所有細橫線（五線譜線 + 加線翅膀 + 保持音短線），並以原圖遮罩回補 1px
        staff_band = roi[max(0, int(sy1 - 3)):min(roi.shape[0], int(sy2 + 4)), :]
        h_lines_only = cv2.morphologyEx(
            staff_band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, roi.shape[1] // 14), 1))
        )
        avg_line_t = float(np.sum(h_lines_only > 0)) / max(1.0, 5.0 * roi.shape[1])
        max_thin = max(2, min(int(round(sp * 0.36)), int(round(avg_line_t + 0.55))))
        vert_keep = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max_thin + 1)))
        vert_keep = cv2.bitwise_and(roi, cv2.dilate(vert_keep, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))))

        # 2. 自動偵測譜頭結束位置（略過高音譜號 Clef 與拍號如 2/4、3/4、4/4）
        header_skip = sp * 4.2
        scan_w = min(roi.shape[1], int(sp * 9.5))
        num_h, _, hstats, _ = cv2.connectedComponentsWithStats(vert_keep[:, :scan_w])
        for lbl in range(1, num_h):
            bx = int(hstats[lbl, cv2.CC_STAT_LEFT])
            by = int(hstats[lbl, cv2.CC_STAT_TOP])
            bw = int(hstats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(hstats[lbl, cv2.CC_STAT_HEIGHT])
            if bx < sp * 3.8 and bh > sp * 4.8:
                # 高音譜號
                header_skip = max(header_skip, bx + bw + sp * 0.6)
            elif sp * 2.8 <= bx <= sp * 8.5 and sp * 0.85 <= bw <= sp * 2.3:
                # 拍號（如 2/4）：完全落在五線譜上下界內，且上下半譜皆有高密度數字筆畫（沒有伸出譜外的細長符幹）
                if by >= sy1 - sp * 0.55 and (by + bh) <= sy2 + sp * 0.55:
                    x0, x1 = max(0, bx - 2), min(scan_w, bx + bw + 2)
                    up_ink = int(np.sum(vert_keep[max(0, int(sy1)):min(roi.shape[0], int(sy1 + 1.9 * sp)), x0:x1] > 0))
                    lo_ink = int(np.sum(vert_keep[max(0, int(sy1 + 2.1 * sp)):min(roi.shape[0], int(sy2)), x0:x1] > 0))
                    if up_ink >= sp * sp * 0.40 and lo_ink >= sp * sp * 0.40:
                        header_skip = max(header_skip, bx + bw + sp * 0.8)

        header_end_x = min(float(xr) - sp, float(xl) + header_skip)

        # 3. 提取細直符幹與小節線 (pure_stems)，以及長橫槓/多小節休止條 (beam_mask)
        v_stems = cv2.morphologyEx(vert_keep, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(6, int(round(sp * 1.4))))))
        horiz_thick = cv2.morphologyEx(vert_keep, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(round(sp * 0.50))), 1)))
        pure_stems = cv2.subtract(v_stems, horiz_thick)

        beam_w = max(8, int(round(sp * 1.65)))
        beam_h = max(max_thin + 1, int(round(sp * 0.30)))
        beams_mask = cv2.morphologyEx(vert_keep, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (beam_w, beam_h)))

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

        # 4. 同時減去長橫槓與細直符幹，讓每顆符頭 100% 獨立分離！
        roi_clean = cv2.subtract(vert_keep, cv2.dilate(beams_mask, np.ones((2, 2), np.uint8)))
        roi_clean = cv2.subtract(roi_clean, pure_stems)

        nh_w = max(3, int(round(sp * 0.50)))
        nh_h = max(3, int(round(sp * 0.44)))
        nh_mask = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nh_w, nh_h)))

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(nh_mask)
        candidates: List[_RawNote] = []
        short_hooks: List[Tuple[float, float]] = []

        for lbl in range(1, num_labels):
            nx = int(stats[lbl, cv2.CC_STAT_LEFT])
            ny = int(stats[lbl, cv2.CC_STAT_TOP])
            cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])
            if cx < header_skip:
                continue

            aspect = cw / max(1.0, float(ch))
            area_ratio = area / max(1.0, sp * sp)
            if not (
                (sp * 0.36 <= ch <= sp * 1.58)
                and (sp * 0.42 <= cw <= sp * 1.88)
                and (0.50 <= aspect <= 2.15)
                and (area_ratio >= 0.20)
                and ((sy1 - sp * 2.5) <= cy <= (sy2 + sp * 2.5))
            ):
                continue

            # 計算此候選區塊連接了幾根獨立符幹 (stems_touched)
            sx0, sx1 = max(0, nx - 2), min(roi.shape[1], nx + cw + 2)
            sy0, sy1_p = max(0, ny - int(sp * 0.8)), min(roi.shape[0], ny + ch + int(sp * 0.8))
            stem_cols = np.where(np.sum(pure_stems[sy0:sy1_p, sx0:sx1] > 0, axis=0) >= 3)[0]
            n_stems = (1 + int(np.sum(np.diff(stem_cols) > 3))) if len(stem_cols) > 0 else 0

            # 跨接 >=2 根符幹者必為短連槓（如雙裝飾音上方的短橫槓），記錄為橫槓而非音符！
            if n_stems >= 2:
                horiz_bars.append(
                    _HorizBar(
                        x1=float(xl + nx),
                        x2=float(xl + nx + cw),
                        y_center=float(ry1 + cy),
                        width=float(cw),
                        height=float(ch),
                        in_staff_mid=False,
                    )
                )
                continue

            is_grace_cand = area_ratio < 0.72
            # 小面積區塊 (area_ratio < 0.72) 若沒有碰到任何符幹，必為雜訊或文字殘片而非裝飾音！
            if is_grace_cand and n_stems == 0:
                continue

            candidates.append(
                _RawNote(
                    x=xl + cx,
                    y=ry1 + cy,
                    width=float(cw),
                    area_ratio=area_ratio,
                    is_grace=is_grace_cand,
                    is_hollow=False,
                )
            )

        # 5. 過濾單根符幹尾端的 16 分音符短勾槓 (partial hook) 與單顆裝飾音上方的斜線符尾 (grace flag)：
        #    注意：為解決 threshold 210 下短勾槓略大於 0.72 的情況，放寬短勾槓面積上限至 0.90
        main_cands = [c for c in candidates if c.area_ratio >= 0.90]
        grace_cands = [c for c in candidates if c.area_ratio < 0.90]
        filtered_cands: List[_RawNote] = []
        for c in candidates:
            if c.area_ratio < 0.90:
                # 5a. 若緊貼在某顆大主音符同一根符幹的另一端 -> 16 分短勾槓
                attached_main = [
                    m for m in main_cands
                    if abs(c.x - m.x) <= sp * 1.35 and abs(c.y - m.y) >= sp * 1.50
                ]
                if attached_main:
                    short_hooks.append((c.x, c.y))
                    continue
                # 5b. 若緊貼在另一顆裝飾音的正上方 (|dx| <= 0.85*sp 且在上方 >= 1.0*sp) -> 單顆倚音上方的斜線小旗標！
                lower_grace = [
                    g for g in grace_cands
                    if g is not c and abs(c.x - g.x) <= sp * 0.85 and (g.y - c.y) >= sp * 1.0
                ]
                if lower_grace:
                    continue
            filtered_cands.append(c)
        candidates = filtered_cands

        # 6. 空心符頭（二分音符 / 全音符）偵測
        closed_roi = cv2.morphologyEx(roi_clean, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        contours, hierarchy = cv2.findContours(closed_roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            hier = hierarchy[0]
            for c_idx, h_info in enumerate(hier):
                parent_idx = h_info[3]
                if parent_idx < 0:
                    continue
                hx, hy, hw, hh = cv2.boundingRect(contours[c_idx])
                _, _, ow, _ = cv2.boundingRect(contours[parent_idx])
                hole_area = cv2.contourArea(contours[c_idx])
                fill_ratio = hole_area / max(1.0, float(hw * hh))
                hcx = hx + hw / 2.0
                hcy = hy + hh / 2.0
                if hcx < header_skip:
                    continue
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
                    if all(abs(abs_x - c.x) > sp * 0.65 for c in candidates if not c.is_grace):
                        sx0 = max(0, int(hcx - sp * 0.9))
                        sx1 = min(roi.shape[1], int(hcx + sp * 0.9))
                        sy_t = max(0, int(hcy - sp * 3.2))
                        sy_b = min(roi.shape[0], int(hcy + sp * 3.2))
                        stem_patch = pure_stems[sy_t:sy_b, sx0:sx1]
                        has_stem = (stem_patch.size > 0) and (
                            int(np.max(np.sum(stem_patch > 0, axis=0))) >= int(sp * 1.4)
                        )
                        candidates.append(
                            _RawNote(
                                x=abs_x,
                                y=abs_y,
                                width=float(ow),
                                area_ratio=1.1,
                                is_grace=False,
                                is_hollow=True,
                                has_stem=has_stem,
                            )
                        )

        candidates.sort(key=lambda n: n.x)

        # 7. 僅合併真正垂直同軸（水平距離 < 0.50*sp）的雙音和弦
        merged: List[_RawNote] = []
        for cand in candidates:
            if merged and abs(cand.x - merged[-1].x) < sp * 0.50:
                prev = merged[-1]
                if not prev.is_grace and cand.is_grace:
                    continue
                if prev.is_grace and not cand.is_grace:
                    merged[-1] = cand
                    continue
                merged[-1] = _RawNote(
                    x=(prev.x + cand.x) / 2.0,
                    y=min(prev.y, cand.y),
                    width=max(prev.width, cand.width),
                    area_ratio=max(prev.area_ratio, cand.area_ratio),
                    is_grace=(prev.is_grace and cand.is_grace),
                    is_hollow=(prev.is_hollow and cand.is_hollow),
                    has_stem=(prev.has_stem or cand.has_stem),
                )
            else:
                merged.append(cand)

        # 8. 計算符尾橫槓數量（含長橫槓與 16 分短勾槓）與附點
        num_dots, _, dstats, dcent = cv2.connectedComponentsWithStats(roi_clean)
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
                    if bar.x1 - sp * 0.7 <= note.x <= bar.x2 + sp * 0.7:
                        if abs(bar.y_center - note.y) >= sp * 0.8:
                            beam_ys.append(bar.y_center)
                for hx, hy in short_hooks:
                    if abs(hx - note.x) <= sp * 1.25:
                        beam_ys.append(hy)
                beam_ys.sort()
                distinct_beams = 0
                last_y = None
                for by in beam_ys:
                    if last_y is None or abs(by - last_y) >= sp * 0.42:
                        distinct_beams += 1
                        last_y = by
                note.beam_count = min(3, distinct_beams)

            note.is_dotted = any(
                (sp * 0.65 <= (dx - note.x) <= sp * 1.65) and (abs(dy - note.y) <= sp * 0.55)
                for dx, dy in dot_pts
            )

        return merged, horiz_bars, header_end_x

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
        if M_inv is not None:
            ox = float(M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2])
            oy = float(M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2])
        else:
            ox, oy = float(x), float(y)
        nx = max(0.0, min(1.0, ox / max(1.0, float(w))))
        ny = max(0.0, min(1.0, oy / max(1.0, float(h))))
        return nx, ny
