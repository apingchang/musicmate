"""樂譜影像結構分析引擎 (OpenCV OMR Lite v2)

針對拍照樂譜 (JPG/PNG/WEBP) 與 PDF 樂譜進行純 Python / OpenCV 影像分析：
1. 自動傾斜偵測與座標校正 (Deskew via HoughLinesP)
2. 五線譜行 (Staff / System) 精準偵測
3. 小節線 (Barline) 偵測（自動排除音符符幹誤判）
4. 多小節休止符 (Multi-measure Rest) 偵測（辨識五線譜中央粗橫條與上方小節數字，避免誤判為音符）
5. 雙尺度符頭偵測：分離主音符與小裝飾音 (Grace Notes)，縮小合併門檻避免相鄰 16 分音符被誤併
6. 符尾橫槓 (Beam) 計數：區分四分音符 (無槓)、八分音符 (單槓)、十六分音符 (雙槓)，精準計算節拍時值
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class NoteEvent:
    """單一演奏音符、裝飾音或休止事件（座標皆為原圖歸一化 0.0~1.0）"""
    page_index: int
    staff_index: int
    measure_number: int
    x_norm: float
    y_norm: float
    beat_offset: float
    duration_beats: float
    is_rest: bool = False
    is_grace: bool = False
    rest_measures: int = 0
    beam_count: int = 0
    midi_note: int = 0
    note_name: str = ""


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
    area_ratio: float
    is_grace: bool
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
    """純 OpenCV 樂譜結構、休止符、裝飾音與音符節奏分析器"""

    def analyze_pages(
        self,
        rgb_images: List[np.ndarray],
        beats_per_measure: float = 4.0,
    ) -> ScoreAnalysis:
        """分析整份樂譜的所有頁面並建立連貫的節拍時間軸"""
        analysis = ScoreAnalysis(beats_per_measure=beats_per_measure)
        cumulative_beat = 0.0
        measure_counter = 0

        for page_idx, rgb in enumerate(rgb_images):
            page_res, cumulative_beat, measure_counter = self._analyze_single_page(
                rgb=rgb,
                page_index=page_idx,
                beats_per_measure=beats_per_measure,
                start_beat=cumulative_beat,
                start_measure_num=measure_counter,
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
    ) -> Tuple[PageAnalysis, float, int]:
        """分析單頁 RGB 影像"""
        if rgb.ndim == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb.copy()

        h, w = gray.shape[:2]
        if h < 100 or w < 100:
            return PageAnalysis(page_index, w, h, 0.0), start_beat, start_measure_num

        # 1. 偵測傾斜角度並產生校正圖（同時保留反矩陣以便映射回原圖座標）
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

        # 2. 自適應二值化（block_size 足夠大以防實心符頭中空）
        block_size = max(35, int(min(w, h) / 22) | 1)
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
            width=w,
            height=h,
            skew_angle=skew,
        )

        cur_beat = start_beat
        cur_measure = start_measure_num

        for s_idx, (y_top, y_bot, sp, xl, xr, line_ys) in enumerate(raw_staves):
            # 4. 偵測該行的橫槓 (Beams / 多小節休止條)、音符符頭 (含裝飾音) 與小節線
            raw_notes, horiz_bars = self._detect_notes_and_bars(
                binary, y_top, y_bot, sp, xl, xr, line_ys
            )
            main_note_pts = [(n.x, n.y) for n in raw_notes if not n.is_grace]
            barlines = self._detect_barlines(binary, y_top, y_bot, sp, xl, xr, main_note_pts)

            # 決定該行的有效小節切分點
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

                # 檢查是否為多小節休止符：小節內有位於五線譜中央的粗橫條，且沒有主音符
                m_notes_all = [n for n in raw_notes if mx1 <= n.x <= mx2]
                m_main_notes = [n for n in m_notes_all if not n.is_grace]
                mid_rest_bars = [
                    b for b in horiz_bars
                    if b.in_staff_mid
                    and b.width >= sp * 2.8
                    and b.x1 >= mx1 - sp
                    and b.x2 <= mx2 + sp
                ]

                is_multi_rest = (len(mid_rest_bars) > 0 and len(m_main_notes) == 0)
                if is_multi_rest:
                    # 檢查休止橫條正上方是否有小節數字（如 "2"），預設多小節休止計為 2 小節
                    bar = max(mid_rest_bars, key=lambda b: b.width)
                    has_digit_above = self._has_ink_above_bar(binary, bar, y_top, sp)
                    rest_count = 2 if has_digit_above else 1
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
                    ev = NoteEvent(
                        page_index=page_index,
                        staff_index=s_idx,
                        measure_number=cur_measure,
                        x_norm=cx_n,
                        y_norm=cy_n,
                        beat_offset=m_beat_start,
                        duration_beats=total_rest_beats,
                        is_rest=True,
                        rest_measures=rest_count,
                        note_name=f"休止 {rest_count} 小節",
                    )
                    measure_info.notes.append(ev)
                    page_res.events.append(ev)
                    # 多小節休止佔用額外小節計數
                    cur_measure += (rest_count - 1)
                    cur_beat = m_beat_end
                    staff_info.measures.append(measure_info)
                    continue

                # 一般含音符小節：依符尾橫槓數量（beam_count）+ 裝飾音（is_grace）精準推算拍數
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

                durations = self._assign_measure_durations(
                    m_notes_all, mx2, sp, beats_per_measure
                )
                local_beat = m_beat_start
                for note_obj, dur in zip(m_notes_all, durations):
                    nx_n, ny_n = self._map_back(note_obj.x, note_obj.y, M_inv, w, h)
                    midi_val, n_name = self._y_to_treble_pitch(note_obj.y, y_bot, sp)
                    ev = NoteEvent(
                        page_index=page_index,
                        staff_index=s_idx,
                        measure_number=cur_measure,
                        x_norm=nx_n,
                        y_norm=ny_n,
                        beat_offset=local_beat,
                        duration_beats=dur,
                        is_rest=False,
                        is_grace=note_obj.is_grace,
                        beam_count=note_obj.beam_count,
                        midi_note=midi_val,
                        note_name=f"({n_name})" if note_obj.is_grace else n_name,
                    )
                    measure_info.notes.append(ev)
                    page_res.events.append(ev)
                    local_beat += dur

                cur_beat = m_beat_end
                staff_info.measures.append(measure_info)

            page_res.staves.append(staff_info)

        return page_res, cur_beat, cur_measure

    @staticmethod
    def _assign_measure_durations(
        notes: List[_RawNote], mx2: float, sp: float, beats_per_measure: float
    ) -> List[float]:
        """結合符尾橫槓數量（beam_count）、裝飾音標記與視覺間距推算每個音符拍數"""
        n_count = len(notes)
        if n_count == 0:
            return []

        # 1. 為裝飾音保留極短時值（約 0.10 拍，總和不超過小節 30%）
        grace_indices = [i for i, n in enumerate(notes) if n.is_grace]
        main_indices = [i for i, n in enumerate(notes) if not n.is_grace]

        if not main_indices:
            # 全部都是小音符時均分
            return [beats_per_measure / n_count] * n_count

        grace_dur = min(0.10, (beats_per_measure * 0.28) / max(1, len(grace_indices)))
        total_grace_beats = grace_dur * len(grace_indices)
        remaining_beats = max(0.25, beats_per_measure - total_grace_beats)

        # 2. 為主音符依「符尾橫槓數 (beam_count)」與「水平間距」計算相對時值權重
        main_weights = []
        for idx_pos, idx in enumerate(main_indices):
            note = notes[idx]
            # 依橫槓數給予樂理基準權重：雙槓=16分(0.25)、單槓=8分(0.50)、無槓=4分(1.0)
            if note.beam_count >= 2:
                beam_w = 0.25
            elif note.beam_count == 1:
                beam_w = 0.50
            else:
                beam_w = 1.00

            # 輔助計算到下一個主音符（或小節線）的水平距離，處理附點音符或二分音符
            if idx_pos + 1 < len(main_indices):
                next_x = notes[main_indices[idx_pos + 1]].x
            else:
                next_x = mx2
            dx = max(sp * 0.8, next_x - note.x)
            main_weights.append((beam_w, dx))

        # 混合 75% 橫槓樂理權重 + 25% 水平間距權重（保留附點/長音拉長感，同時讓 16 分音符明確輕快）
        avg_dx = sum(d for _, d in main_weights) / len(main_weights)
        combined = []
        for bw, dx in main_weights:
            dx_norm = (dx / avg_dx) if avg_dx > 0 else 1.0
            # 限制間距影響範圍避免過大
            dx_factor = max(0.5, min(2.0, dx_norm))
            combined.append(bw * (0.75 + 0.25 * dx_factor))

        total_w = sum(combined)
        durations = [0.0] * n_count
        for i in grace_indices:
            durations[i] = grace_dur
        for idx, w_val in zip(main_indices, combined):
            durations[idx] = remaining_beats * (w_val / total_w) if total_w > 0 else (remaining_beats / len(main_indices))

        return durations

    @staticmethod
    def _has_ink_above_bar(binary: np.ndarray, bar: _HorizBar, y_top: float, sp: float) -> bool:
        """檢查多小節休止橫條正上方（五線譜上方 0.2~3.0 sp）是否有休止小節數字"""
        h, w = binary.shape[:2]
        ry1 = max(0, int(y_top - sp * 3.0))
        ry2 = max(ry1 + 1, int(y_top - sp * 0.15))
        rx1 = max(0, int(bar.x1))
        rx2 = min(w, int(bar.x2))
        if rx2 <= rx1 or ry2 <= ry1:
            return False
        ink = int(np.sum(binary[ry1:ry2, rx1:rx2] > 0))
        return ink >= int(sp * sp * 0.35)

    @staticmethod
    def _estimate_skew(gray: np.ndarray) -> float:
        """透過 Canny + HoughLinesP 偵測五線譜傾斜角度（度）"""
        h, w = gray.shape[:2]
        edges = cv2.Canny(gray, 50, 150)
        min_len = max(40, w // 8)
        hlines = cv2.HoughLinesP(
            edges, 1, np.pi / 1800.0,
            threshold=80,
            minLineLength=min_len,
            maxLineGap=15,
        )
        if hlines is None:
            return 0.0
        angles = []
        for x1, y1, x2, y2 in hlines.reshape(-1, 4):
            ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(ang) < 8.0:
                angles.append(ang)
        if len(angles) < 3:
            return 0.0
        return float(np.median(angles))

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
        """去除五線譜線後，同時偵測橫槓 (Beams / 多小節休止條) 與雙尺度音符符頭（主音符 + 裝飾音）"""
        h, w = binary.shape[:2]
        pad = int(sp * 3.5)
        ry1 = max(0, int(y_top - pad))
        ry2 = min(h, int(y_bot + pad))
        roi = binary[ry1:ry2, xl:xr].copy()
        sy1 = float(y_top - ry1)
        sy2 = float(y_bot - ry1)

        # 1. 移除純五線譜橫線（保留上下交會之音符與符幹筆畫）
        for ly in line_ys:
            ly_rel = int(round(ly - ry1))
            for dy in (-1, 0, 1):
                r = ly_rel + dy
                if 2 <= r < roi.shape[0] - 2:
                    pure = (roi[r - 2, :] == 0) & (roi[r + 2, :] == 0)
                    roi[r, pure] = 0

        # 2. 提取粗橫槓（符尾連槓 Beams 與多小節休止橫條）：寬 >= 1.65*sp，高 >= 0.26*sp
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

        # 3. 扣除粗橫槓後，以雙尺度橢圓核心偵測符頭（涵蓋一般實心符頭與較小的 16 分/裝飾音符頭）
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
                is_grace = area_ratio < 0.72
                candidates.append(
                    _RawNote(
                        x=xl + cx,
                        y=ry1 + cy,
                        area_ratio=area_ratio,
                        is_grace=is_grace,
                    )
                )

        candidates.sort(key=lambda n: n.x)

        # 4. 僅合併真正垂直堆疊（同一時間點）的雙音/和弦或裝飾音上方小符槓（水平距離 < 0.55 * sp）
        #    避免將相鄰的 16 分音符或裝飾音誤併成單一音符！
        merged: List[_RawNote] = []
        for cand in candidates:
            if merged and abs(cand.x - merged[-1].x) < sp * 0.55:
                prev = merged[-1]
                # 若一個是裝飾音符頭、另一個是上方小連槓，保留較靠五線譜內側的符頭
                if prev.is_grace and not cand.is_grace and abs(cand.y - prev.y) > sp * 1.1:
                    # 保留原本的裝飾音符頭（cand 可能是上方小粗塊），若 cand 明顯較低且在譜內則保留主音符
                    if cand.y > prev.y:
                        merged[-1] = cand
                elif not prev.is_grace and cand.is_grace:
                    # 主音符優先，忽略緊貼在同 x 的小雜點/裝飾符槓
                    pass
                else:
                    # 同質雙音和弦：取平均 x，保留較高音或較大面積
                    merged[-1] = _RawNote(
                        x=(prev.x + cand.x) / 2.0,
                        y=min(prev.y, cand.y),
                        area_ratio=max(prev.area_ratio, cand.area_ratio),
                        is_grace=(prev.is_grace and cand.is_grace),
                    )
            else:
                merged.append(cand)

        # 5. 為每個主音符計算其所屬的符尾橫槓數量（beam_count: 0=四分, 1=八分, 2+=十六分）
        for note in merged:
            if note.is_grace:
                note.beam_count = 2
                continue
            beam_ys = []
            for bar in horiz_bars:
                # 排除多小節休止長條，只看音符水平範圍內的符尾橫槓（位於符頭上方或下方）
                if bar.x1 - sp * 0.7 <= note.x <= bar.x2 + sp * 0.7:
                    if abs(bar.y_center - note.y) >= sp * 0.8:
                        beam_ys.append(bar.y_center)
            # 計算不同 y 層級的橫槓數量（同 y 差異 < 0.45*sp 視為同一條）
            beam_ys.sort()
            distinct_beams = 0
            last_y = None
            for by in beam_ys:
                if last_y is None or abs(by - last_y) >= sp * 0.45:
                    distinct_beams += 1
                    last_y = by
            note.beam_count = min(3, distinct_beams)

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
