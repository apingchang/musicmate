"""樂譜影像結構分析引擎 (OpenCV OMR Lite)

針對拍照樂譜 (JPG/PNG/WEBP) 與 PDF 樂譜進行純 Python / OpenCV 影像分析：
1. 自動傾斜偵測與座標校正 (Deskew via HoughLinesP)
2. 五線譜行 (Staff / System) 精準偵測
3. 小節線 (Barline) 偵測（自動排除音符符幹誤判）
4. 音符符頭 (Notehead) 偵測與和弦合併
5. 依小節與音符水平間距推算節拍時間軸 (Beat Timeline)，供演奏游標即時追蹤
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class NoteEvent:
    """單一演奏音符或休止事件（座標皆為原圖歸一化 0.0~1.0）"""
    page_index: int
    staff_index: int
    measure_number: int
    x_norm: float
    y_norm: float
    beat_offset: float
    duration_beats: float
    is_rest: bool = False
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


class ScoreAnalyzer:
    """純 OpenCV 樂譜結構與音符位置分析器"""

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

        # 2. 自適應二值化（適應手機拍照光線不均與透背雜訊，block_size 足夠大以防實心符頭中空）
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
            # 4. 偵測該行音符與小節線
            note_pts = self._detect_noteheads(binary, y_top, y_bot, sp, xl, xr, line_ys)
            barlines = self._detect_barlines(binary, y_top, y_bot, sp, xl, xr, note_pts)

            # 決定該行的有效小節切分點
            clef_end = min(xr - sp, xl + sp * 4.2)
            # 過濾掉太靠左（譜號區內）的小節線
            valid_bars = [bx for bx in barlines if bx > clef_end]
            # 決定起始 x：若第一個小節線很靠前（如反覆記號），仍保留前面空間作為第一小節（如多小節休止）
            seg_edges = [clef_end] + valid_bars
            if not valid_bars or (xr - seg_edges[-1]) > sp * 3.0:
                seg_edges.append(float(xr))

            # 移除過窄的假切分（小於 2.2 * sp）
            clean_edges = [seg_edges[0]]
            for ex in seg_edges[1:]:
                if ex - clean_edges[-1] >= sp * 2.2:
                    clean_edges.append(ex)
            if len(clean_edges) < 2:
                clean_edges = [clef_end, float(xr)]

            # 轉換五線譜框座標回原圖
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
                cur_measure += 1

                # 找出落在此小節內的音符
                m_notes = [pt for pt in note_pts if mx1 <= pt[0] <= mx2]
                m_notes.sort(key=lambda p: p[0])

                m_x1_n, m_yt_n = self._map_back(mx1, y_top, M_inv, w, h)
                m_x2_n, m_yb_n = self._map_back(mx2, y_bot, M_inv, w, h)

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

                if len(m_notes) == 0:
                    # 休止小節：在中央放置休止事件
                    cx_n, cy_n = self._map_back((mx1 + mx2) / 2.0, (y_top + y_bot) / 2.0, M_inv, w, h)
                    ev = NoteEvent(
                        page_index=page_index,
                        staff_index=s_idx,
                        measure_number=cur_measure,
                        x_norm=cx_n,
                        y_norm=cy_n,
                        beat_offset=m_beat_start,
                        duration_beats=beats_per_measure,
                        is_rest=True,
                    )
                    measure_info.notes.append(ev)
                    page_res.events.append(ev)
                else:
                    # 依水平視覺間距比例分配小節內的拍數
                    spans = []
                    for k, (nx, ny) in enumerate(m_notes):
                        if k + 1 < len(m_notes):
                            dx = max(sp * 0.5, m_notes[k + 1][0] - nx)
                        else:
                            dx = max(sp * 0.8, mx2 - nx)
                        spans.append(dx)
                    total_span = sum(spans)
                    local_beat = m_beat_start
                    for k, (nx, ny) in enumerate(m_notes):
                        dur = beats_per_measure * (spans[k] / total_span) if total_span > 0 else (beats_per_measure / len(m_notes))
                        nx_n, ny_n = self._map_back(nx, ny, M_inv, w, h)
                        midi_val, n_name = self._y_to_treble_pitch(ny, y_bot, sp)
                        ev = NoteEvent(
                            page_index=page_index,
                            staff_index=s_idx,
                            measure_number=cur_measure,
                            x_norm=nx_n,
                            y_norm=ny_n,
                            beat_offset=local_beat,
                            duration_beats=dur,
                            is_rest=False,
                            midi_note=midi_val,
                            note_name=n_name,
                        )
                        measure_info.notes.append(ev)
                        page_res.events.append(ev)
                        local_beat += dur

                cur_beat = m_beat_end
                staff_info.measures.append(measure_info)

            page_res.staves.append(staff_info)

        return page_res, cur_beat, cur_measure

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
    def _detect_noteheads(
        binary: np.ndarray,
        y_top: float,
        y_bot: float,
        sp: float,
        xl: int,
        xr: int,
        line_ys: List[float],
    ) -> List[Tuple[float, float]]:
        """去除五線譜線後偵測音符符頭中心座標 (x, y)"""
        h, w = binary.shape[:2]
        pad = int(sp * 3.0)
        ry1 = max(0, int(y_top - pad))
        ry2 = min(h, int(y_bot + pad))
        roi = binary[ry1:ry2, xl:xr].copy()
        sy1 = int(y_top - ry1)
        sy2 = int(y_bot - ry1)

        # 移除純五線譜橫線（保留上下交會之音符筆畫）
        for ly in line_ys:
            ly_rel = int(round(ly - ry1))
            for dy in (-1, 0, 1):
                r = ly_rel + dy
                if 2 <= r < roi.shape[0] - 2:
                    pure = (roi[r - 2, :] == 0) & (roi[r + 2, :] == 0)
                    roi[r, pure] = 0

        # 以橢圓核心型態學開運算提取實心符頭
        nh_h = max(3, int(round(sp * 0.62)))
        nh_w = max(3, int(round(sp * 0.72)))
        nh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nh_w, nh_h))
        nh_mask = cv2.morphologyEx(roi, cv2.MORPH_OPEN, nh_kernel)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(nh_mask)
        raw_notes: List[Tuple[float, float]] = []
        clef_skip = int(sp * 4.2)

        for lbl in range(1, num_labels):
            cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])
            if cx < clef_skip:
                continue
            aspect = cw / max(1.0, float(ch))
            if (
                (sp * 0.50 <= ch <= sp * 1.65)
                and (sp * 0.55 <= cw <= sp * 1.90)
                and (0.55 <= aspect <= 2.10)
                and (area >= sp * sp * 0.30)
                and ((sy1 - sp * 2.5) <= cy <= (sy2 + sp * 2.5))
            ):
                raw_notes.append((xl + cx, ry1 + cy))

        raw_notes.sort(key=lambda p: p[0])

        # 合併同時間點的和弦音符或重疊偵測（水平距離 < 0.95 * sp）
        merged: List[Tuple[float, float]] = []
        for nx, ny in raw_notes:
            if merged and abs(nx - merged[-1][0]) < sp * 0.95:
                px, py = merged[-1]
                merged[-1] = ((px + nx) / 2.0, min(py, ny))
            else:
                merged.append((nx, ny))
        return merged

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
        # 底線 y_bot = E4（step = 0），每往上 0.5 * sp 為 1 個自然音階步
        step = int(round((y_bot - ny) / (sp / 2.0)))
        step = max(-14, min(20, step))
        diatonic_idx = 2 + step  # C=0, D=1, E=2, F=3, G=4, A=5, B=6
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
