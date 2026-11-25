"""樂譜影像結構分析引擎 (OpenCV OMR Lite v5 - 分段追蹤逐行拉直與去底色精準切分版)

核心處理流程：
1. 去底色與對比強化 (Division Illumination Normalization)：消除泛黃紙色、手機拍照陰影與背面透字
2. 自適應高解析度縮放 (確保單行與整頁 A4 多行樂譜皆具備清晰五線譜間距 sp ≈ 12~16 px)
3. 垂直分片五線譜追蹤 (Multi-Slice Comb Tracking) + 逐行曲面/傾斜拉直 (Per-Staff Remap Flattening)：
   徹底解決手機翻拍書頁時「上行斜、下行平」的透視變形與書脊彎曲，杜絕斜線投影壓扁成假窄譜與假小節線
4. 譜頭自動跳過 (自動辨識並略過譜號 Clef 與拍號如 2/4，絕不誤判為音符)
5. 全域細橫線抹除 + 細直符幹剝離 + 跨雙幹短橫槓過濾：切出每顆獨立音符、十六分音符、裝飾音與高音加線音符
6. 反覆記號雙點 (|: / :|) 過濾與多小節休止符 (H-bar + 數字 2..8) 精準辨識
"""

import copy
import math
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
    repeat_pass: int = 1
    repeat_start: bool = False
    repeat_end: bool = False
    repeat_jump_to: int = 0

    @property
    def freq_hz(self) -> float:
        if self.is_rest or self.midi_note <= 0:
            return 0.0
        return float(440.0 * (2.0 ** ((self.midi_note - 69) / 12.0)))


@dataclass
class MeasureInfo:
    """單一小節資訊（對應譜面實體小節，含前/後反覆記號屬性）"""
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
    repeat_start: bool = False
    repeat_end: bool = False
    notes: List[NoteEvent] = field(default_factory=list)


@dataclass
class PlaybackMeasure:
    """實際播放順序中的單遍小節（支援第 1 遍與反覆第 2 遍精準跳轉與單小節播放）"""
    measure: MeasureInfo
    beat_start: float
    beat_end: float
    repeat_pass: int = 1
    jump_to_measure: int = 0


@dataclass
class StaffInfo:
    """單一行五線譜資訊（含隨 x 變化的傾斜/彎曲上下界軌跡）"""
    page_index: int
    staff_index: int
    y_top_norm: float
    y_bot_norm: float
    x_left_norm: float
    x_right_norm: float
    spacing_norm: float
    barlines_norm: List[float] = field(default_factory=list)
    measures: List[MeasureInfo] = field(default_factory=list)
    curve_xs_norm: List[float] = field(default_factory=list)
    curve_y_tops_norm: List[float] = field(default_factory=list)
    staff_height_norm: float = 0.0

    def y_range_at(self, x_norm: float) -> Tuple[float, float]:
        """取得此行五線譜在指定水平位置 x_norm 的精確 (y_top_norm, y_bot_norm)"""
        if self.curve_xs_norm and self.curve_y_tops_norm and self.staff_height_norm > 0:
            yt = float(np.interp(x_norm, self.curve_xs_norm, self.curve_y_tops_norm))
            return yt, yt + self.staff_height_norm
        return self.y_top_norm, self.y_bot_norm


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
    """整份樂譜的分析結果與總演奏時間軸（含反覆記號展開序列）"""
    pages: List[PageAnalysis] = field(default_factory=list)
    events: List[NoteEvent] = field(default_factory=list)
    playback_measures: List[PlaybackMeasure] = field(default_factory=list)
    repeat_sections: List[Tuple[int, int]] = field(default_factory=list)
    total_beats: float = 0.0
    total_measures: int = 0
    beats_per_measure: float = 4.0
    enable_repeats: bool = True

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
class _BarlineInfo:
    x_center: float
    x_left: float
    x_right: float
    repeat_end_left: bool = False
    repeat_start_right: bool = False


@dataclass
class _EdgeMark:
    x: float
    repeat_end_left: bool = False
    repeat_start_right: bool = False


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
    n_stems: int = 0


class ScoreAnalyzer:
    """純 OpenCV 去底色強化、逐行曲面拉直與精準音符切分引擎"""

    BASE_WIDTH = 1300

    @staticmethod
    def enhance_score_gray(gray: np.ndarray) -> np.ndarray:
        """去底色與對比強化：消除泛黃紙張、拍照陰影漸層與背面透字，輸出純白底深黑墨跡灰階圖"""
        h, w = gray.shape[:2]
        ref_dim = max(w, h)
        # 膨脹核大於實心符頭與多小節休止橫條厚度，避免實心黑塊被當成背景掏空
        k_dil = max(15, int(round(ref_dim / 75)) | 1)
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
        enable_repeats: bool = True,
    ) -> ScoreAnalysis:
        analysis = ScoreAnalysis(
            beats_per_measure=beats_per_measure,
            enable_repeats=enable_repeats,
        )
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

        analysis.total_measures = measure_counter
        self.rebuild_timeline(analysis, beats_per_measure=beats_per_measure, enable_repeats=enable_repeats)
        return analysis

    @staticmethod
    def rebuild_timeline(
        analysis: ScoreAnalysis,
        beats_per_measure: float,
        enable_repeats: bool = True,
    ) -> None:
        """依實體小節的前後反覆記號 (||: ... :||) 動態展開第 1 遍與第 2 遍演奏時間軸（無須重跑影像處理）"""
        old_bpm_beats = max(0.25, analysis.beats_per_measure)
        beat_scale = beats_per_measure / old_bpm_beats
        analysis.beats_per_measure = beats_per_measure
        analysis.enable_repeats = enable_repeats
        analysis.events.clear()
        analysis.playback_measures.clear()
        analysis.repeat_sections.clear()

        phys_measures: List[MeasureInfo] = []
        for p in analysis.pages:
            p.events.clear()
            for st in p.staves:
                for m in st.measures:
                    # 按新拍號等比例縮放小節內原始音符時長
                    if abs(beat_scale - 1.0) > 1e-6:
                        for ev0 in m.notes:
                            ev0.duration_beats *= beat_scale
                    phys_measures.append(m)

        if not phys_measures:
            analysis.total_beats = 0.0
            return

        # 解析反覆區段並產生播放小節序列 [(MeasureInfo, pass_num, jump_to_measure)]
        playback_plan: List[Tuple[MeasureInfo, int, int]] = []
        anchor_idx = 0
        for i, m in enumerate(phys_measures):
            if m.repeat_start:
                anchor_idx = i
            jump_target = 0
            if m.repeat_end:
                start_m = phys_measures[anchor_idx].measure_number
                end_m = m.measure_number + max(0, m.rest_measures - 1)
                analysis.repeat_sections.append((start_m, end_m))
                if enable_repeats:
                    jump_target = start_m

            playback_plan.append((m, 1, jump_target))
            if enable_repeats and m.repeat_end:
                for j in range(anchor_idx, i + 1):
                    playback_plan.append((phys_measures[j], 2, 0))
                anchor_idx = min(len(phys_measures) - 1, i + 1)

        cur_beat = 0.0
        cur_ev_idx = 0
        for m, pass_num, jump_target in playback_plan:
            m_dur = beats_per_measure * (m.rest_measures if m.rest_measures > 0 else 1)
            m_b_start = cur_beat
            m_b_end = cur_beat + m_dur

            if pass_num == 1:
                m.beat_start = m_b_start
                m.beat_end = m_b_end

            analysis.playback_measures.append(
                PlaybackMeasure(
                    measure=m,
                    beat_start=m_b_start,
                    beat_end=m_b_end,
                    repeat_pass=pass_num,
                    jump_to_measure=jump_target,
                )
            )

            local_beat = m_b_start
            for ev_orig in m.notes:
                cur_ev_idx += 1
                ev_play = copy.copy(ev_orig)
                ev_play.event_index = cur_ev_idx
                ev_play.beat_offset = local_beat
                ev_play.repeat_pass = pass_num
                ev_play.repeat_start = m.repeat_start
                ev_play.repeat_end = m.repeat_end
                ev_play.repeat_jump_to = jump_target
                analysis.events.append(ev_play)

                if pass_num == 1:
                    ev_orig.event_index = cur_ev_idx
                    ev_orig.beat_offset = local_beat
                    ev_orig.repeat_pass = 1
                    ev_orig.repeat_start = m.repeat_start
                    ev_orig.repeat_end = m.repeat_end
                    ev_orig.repeat_jump_to = jump_target
                    analysis.pages[m.page_index].events.append(ev_orig)

                local_beat += ev_play.duration_beats

            cur_beat = m_b_end

        analysis.total_beats = cur_beat

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
        if orig_h < 80 or orig_w < 100:
            return PageAnalysis(page_index, orig_w, orig_h, 0.0), start_beat, start_measure_num, start_event_idx

        # 1. 自適應標準化分析寬度（橫向截圖約 1300px，直向整頁 A4 自動放大至最高 1750px 確保譜線間距充足）
        aspect_hw = orig_h / max(1.0, float(orig_w))
        target_w = int(np.clip(self.BASE_WIDTH * max(1.0, aspect_hw * 0.92), 1300, 1750))
        if abs(orig_w - target_w) > 60:
            scale = target_w / float(orig_w)
            w = target_w
            h = max(80, int(round(orig_h * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            gray = cv2.resize(orig_gray, (w, h), interpolation=interp)
        else:
            gray = orig_gray
            h, w = orig_h, orig_w

        # 2. 先執行去底色與對比強化！消除紙色、陰影與透字
        clean_gray = self.enhance_score_gray(gray)

        # 3. 垂直分片追蹤每行五線譜（支援不同行各自傾斜、透視變形與書頁彎曲）
        staff_tracks = self._detect_curved_staves(clean_gray)

        slopes_deg = []
        for trk in staff_tracks:
            if len(trk) >= 2 and (trk[-1][0] - trk[0][0]) > 20:
                slopes_deg.append(math.degrees(math.atan2(trk[-1][1] - trk[0][1], trk[-1][0] - trk[0][0])))
        avg_skew = float(np.median(slopes_deg)) if slopes_deg else 0.0

        page_res = PageAnalysis(
            page_index=page_index,
            width=orig_w,
            height=orig_h,
            skew_angle=avg_skew,
        )

        cur_beat = start_beat
        cur_measure = start_measure_num
        cur_ev_idx = start_event_idx

        for s_idx, track in enumerate(staff_tracks):
            # 4. 將此行五線譜獨立沿正交曲線拉直為水平帶狀影像 strip_bin（橫線歸 0°、直立小節線與符幹同步歸 90°！）
            strip_bin, y_shift, slope_arr, center_dy, sy1, sy2, sp, xl, xr = self._straighten_staff_strip(clean_gray, track)
            if xr - xl < sp * 6.0:
                continue

            def to_page_norm(px: float, py_in_strip: float) -> Tuple[float, float]:
                ix = int(np.clip(round(px), 0, w - 1))
                ox = float(px) - float(slope_arr[ix]) * (float(py_in_strip) - center_dy)
                oy = float(y_shift[ix]) + float(py_in_strip)
                return (
                    float(np.clip(ox / max(1.0, float(w)), 0.0, 1.0)),
                    float(np.clip(oy / max(1.0, float(h)), 0.0, 1.0)),
                )

            # 5. 在 100% 拉直的五線譜帶上偵測譜頭結束位置、獨立音符、橫槓、小節線與反覆記號 (||: / :|| / :||:)
            raw_notes, horiz_bars, header_end_x = self._detect_notes_and_bars(
                strip_bin, sy1, sy2, sp, xl, xr
            )
            main_note_pts = [(n.x, n.y) for n in raw_notes if not n.is_grace]
            bar_infos = self._detect_barlines_and_repeats(
                strip_bin, sy1, sy2, sp, xl, xr, main_note_pts, horiz_bars
            )

            # 6. 過濾緊鄰小節線的反覆記號雙點 (|: / :|，寬度 < 0.85*sp 且距小節線 < 1.55*sp)，並清除誤吸反覆點的假附點
            if bar_infos:
                raw_notes = [
                    n for n in raw_notes
                    if not (
                        n.width < sp * 0.85
                        and any(b.x_left - sp * 1.50 <= n.x <= b.x_right + sp * 1.50 for b in bar_infos)
                    )
                ]
                for n in raw_notes:
                    if n.is_dotted and any(
                        b.repeat_end_left and 0.0 < (b.x_left - n.x) <= sp * 2.4 for b in bar_infos
                    ):
                        n.is_dotted = False

            # 檢查譜頭正後方是否緊接著前反覆記號 ||:
            header_rep_start = any(
                b.repeat_start_right and abs(b.x_center - header_end_x) <= sp * 1.25
                for b in bar_infos
            )
            valid_bars = [b for b in bar_infos if b.x_center > header_end_x + sp * 1.25]

            seg_edges: List[_EdgeMark] = [_EdgeMark(x=header_end_x, repeat_start_right=header_rep_start)]
            for b in valid_bars:
                seg_edges.append(
                    _EdgeMark(
                        x=b.x_center,
                        repeat_end_left=b.repeat_end_left,
                        repeat_start_right=b.repeat_start_right,
                    )
                )
            if not valid_bars or (xr - seg_edges[-1].x) > sp * 3.0:
                end_rep = any(
                    b.repeat_end_left and abs(b.x_center - float(xr)) <= sp * 3.0
                    for b in bar_infos
                )
                seg_edges.append(_EdgeMark(x=float(xr), repeat_end_left=end_rep))

            clean_edges: List[_EdgeMark] = [seg_edges[0]]
            for em in seg_edges[1:]:
                if em.x - clean_edges[-1].x >= sp * 2.4:
                    clean_edges.append(em)
                else:
                    clean_edges[-1].repeat_end_left = clean_edges[-1].repeat_end_left or em.repeat_end_left
                    clean_edges[-1].repeat_start_right = clean_edges[-1].repeat_start_right or em.repeat_start_right
            if len(clean_edges) < 2:
                clean_edges = [_EdgeMark(x=header_end_x), _EdgeMark(x=float(xr))]

            # 建立此行五線譜隨 x 變化的精確上下軌跡（供 UI 繪製貼合傾斜/彎曲譜面的高亮帶與游標）
            sample_xs = np.linspace(float(xl), float(xr), 17)
            curve_xs_norm = [float(x / w) for x in sample_xs]
            curve_y_tops_norm = [to_page_norm(float(x), sy1)[1] for x in sample_xs]
            staff_h_norm = float((sy2 - sy1) / max(1.0, float(h)))
            mean_yt_norm = float(np.mean(curve_y_tops_norm))

            staff_info = StaffInfo(
                page_index=page_index,
                staff_index=len(page_res.staves),
                y_top_norm=mean_yt_norm,
                y_bot_norm=min(1.0, mean_yt_norm + staff_h_norm),
                x_left_norm=float(xl) / float(w),
                x_right_norm=float(xr) / float(w),
                spacing_norm=sp / float(h),
                barlines_norm=[float(b.x_center) / float(w) for b in valid_bars],
                curve_xs_norm=curve_xs_norm,
                curve_y_tops_norm=curve_y_tops_norm,
                staff_height_norm=staff_h_norm,
            )
            real_s_idx = staff_info.staff_index

            # 7. 建立每個小節與節拍時間軸
            for m_i in range(len(clean_edges) - 1):
                e_left = clean_edges[m_i]
                e_right = clean_edges[m_i + 1]
                mx1 = e_left.x
                mx2 = e_right.x
                m_rep_start = e_left.repeat_start_right
                m_rep_end = e_right.repeat_end_left
                m_width = max(1.0, mx2 - mx1)

                # 尋找此小節內的五線譜中央粗橫條（H-bar 多小節休止符：位於小節內部且非整條譜線）
                mid_rest_bars = [
                    b for b in horiz_bars
                    if b.in_staff_mid
                    and sp * 2.2 <= b.width <= min(sp * 9.5, m_width * 0.82)
                    and b.x1 >= mx1 + sp * 0.45
                    and b.x2 <= mx2 - sp * 0.45
                ]

                m_notes_all = [n for n in raw_notes if mx1 <= n.x <= mx2]

                # 若存在中央休止粗橫條，排除橫條正上方的休止數字（如跨在第一線上的粗體 "2"，其與中央橫條間有明顯空白斷開，並非連槓音符！）
                if mid_rest_bars:
                    rest_bar = max(mid_rest_bars, key=lambda b: b.width)
                    # 檢查中央橫條上方是否有真正連通到橫條的長符幹（區分 M7 高音連槓音符 vs M1 懸空數字 "2"）
                    bx0 = max(0, int(round(rest_bar.x1)))
                    bx1 = min(strip_bin.shape[1], int(round(rest_bar.x2)))
                    gap_y1 = max(0, int(round(sy1 + sp * 0.55)))
                    gap_y2 = max(gap_y1 + 2, int(round(rest_bar.y_center - sp * 0.35)))
                    gap_patch = strip_bin[gap_y1:gap_y2, bx0:bx1]
                    # 真實下垂符幹會在 gap_patch 形成貫穿上下的細直柱；而懸空數字 "2" 下方與橫條之間是空白的！
                    has_connecting_stem = (
                        gap_patch.size > 0
                        and int(np.max(np.sum(gap_patch > 0, axis=0))) >= max(3, int((gap_y2 - gap_y1) * 0.85))
                    )
                    real_notes_in_m = [
                        n for n in m_notes_all
                        if not (
                            (rest_bar.x1 - sp * 0.65 <= n.x <= rest_bar.x2 + sp * 0.65)
                            and (n.y < rest_bar.y_center)
                            and (not has_connecting_stem or n.is_grace)
                        )
                        and not n.is_grace
                    ]
                    if len(real_notes_in_m) == 0:
                        m_notes_all = []
                    else:
                        rest_bar = None
                else:
                    rest_bar = None

                if rest_bar is not None and len(m_notes_all) == 0:
                    rest_count = self._estimate_rest_bar_count(strip_bin, rest_bar, sy1, sp)
                elif len(m_notes_all) == 0:
                    rest_count = 1
                else:
                    rest_count = 0

                cur_measure += 1
                mcx = 0.5 * (mx1 + mx2)
                _, m_yt_n = to_page_norm(mcx, sy1)
                _, m_yb_n = to_page_norm(mcx, sy2)
                m_x1_n = float(np.clip(mx1 / float(w), 0.0, 1.0))
                m_x2_n = float(np.clip(mx2 / float(w), 0.0, 1.0))

                if rest_count > 0:
                    total_rest_beats = beats_per_measure * rest_count
                    m_beat_start = cur_beat
                    m_beat_end = cur_beat + total_rest_beats
                    measure_info = MeasureInfo(
                        page_index=page_index,
                        staff_index=real_s_idx,
                        measure_number=cur_measure,
                        x_start_norm=m_x1_n,
                        x_end_norm=m_x2_n,
                        y_top_norm=m_yt_n,
                        y_bot_norm=m_yb_n,
                        beat_start=m_beat_start,
                        beat_end=m_beat_end,
                        rest_measures=rest_count,
                        repeat_start=m_rep_start,
                        repeat_end=m_rep_end,
                    )
                    cx_n, cy_n = to_page_norm(mcx, 0.5 * (sy1 + sy2))
                    cur_ev_idx += 1
                    r_label = f"休止 {rest_count} 小節" if rest_count > 1 else "整小節休止"
                    ev = NoteEvent(
                        event_index=cur_ev_idx,
                        page_index=page_index,
                        staff_index=real_s_idx,
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
                    staff_index=real_s_idx,
                    measure_number=cur_measure,
                    x_start_norm=m_x1_n,
                    x_end_norm=m_x2_n,
                    y_top_norm=m_yt_n,
                    y_bot_norm=m_yb_n,
                    beat_start=m_beat_start,
                    beat_end=m_beat_end,
                    repeat_start=m_rep_start,
                    repeat_end=m_rep_end,
                )

                durations, note_types = self._classify_and_assign_durations(
                    m_notes_all, mx2, sp, beats_per_measure
                )
                local_beat = m_beat_start
                total_in_m = len(m_notes_all)
                for k_idx, (note_obj, dur, n_type) in enumerate(zip(m_notes_all, durations, note_types), start=1):
                    nx_n, ny_n = to_page_norm(note_obj.x, note_obj.y)
                    midi_val, n_name = self._y_to_treble_pitch(note_obj.y, sy2, sp)
                    cur_ev_idx += 1
                    ev = NoteEvent(
                        event_index=cur_ev_idx,
                        page_index=page_index,
                        staff_index=real_s_idx,
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
    def _detect_curved_staves(clean_gray: np.ndarray) -> List[List[Tuple[float, float, float, float]]]:
        """垂直分片五線譜偵測：即使同一頁不同行傾斜角不同或書脊彎曲，仍能精準追蹤每一行 5 條線"""
        h, w = clean_gray.shape[:2]
        b = ((clean_gray < 212).astype(np.uint8)) * 255
        # 短水平結構核可容忍高達 ±6° 斜率而不斷裂
        hlen = max(10, w // 95)
        horiz = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1)))

        n_slices = 16
        slice_w = max(40, w // 10)
        xs_centers = np.linspace(slice_w // 2, w - slice_w // 2, n_slices)
        min_sp = max(5.5, w / 225.0)
        max_sp = max(20.0, w / 26.0)
        kernel_1d = np.array([0.2, 0.6, 1.0, 0.6, 0.2], dtype=np.float32)

        slice_combs: List[List[Tuple[float, float, float, float]]] = []
        for xc in xs_centers:
            x0 = max(0, int(xc - slice_w // 2))
            x1 = min(w, int(xc + slice_w // 2))
            sw = max(1, x1 - x0)
            proj = np.sum(horiz[:, x0:x1] > 0, axis=1).astype(np.float32)
            ps = np.convolve(proj, kernel_1d, mode="same")

            peaks: List[Tuple[float, float]] = []
            min_peak_dist = max(4, int(min_sp * 0.62))
            for r in range(3, h - 3):
                if (
                    ps[r] >= sw * 0.28
                    and ps[r] >= ps[r - 1]
                    and ps[r] >= ps[r + 1]
                    and ps[r] >= ps[r - 2]
                    and ps[r] >= ps[r + 2]
                ):
                    if not peaks or (r - peaks[-1][0]) >= min_peak_dist:
                        peaks.append((float(r), float(ps[r])))
                    elif ps[r] > peaks[-1][1]:
                        peaks[-1] = (float(r), float(ps[r]))

            p_ys = [p[0] for p in peaks]
            combs: List[Tuple[float, float, float, float]] = []
            idx = 0
            while idx + 4 < len(p_ys):
                grp = p_ys[idx:idx + 5]
                diffs = np.diff(grp)
                med = float(np.median(diffs))
                if min_sp <= med <= max_sp and np.all(np.abs(diffs - med) <= max(2.2, med * 0.22)):
                    combs.append((float(xc), grp[0], grp[4], med))
                    idx += 5
                else:
                    idx += 1
            slice_combs.append(combs)

        # 跨分片串聯同一行五線譜軌跡
        tracks: List[List[Tuple[float, float, float, float]]] = []
        for combs in slice_combs:
            for c in combs:
                xc, y0, y4, sp = c
                ymid = 0.5 * (y0 + y4)
                best_t = None
                best_dist = 1e9
                for t in tracks:
                    lx, ly0, ly4, lsp = t[-1]
                    if xc <= lx or (xc - lx) > w * 0.42:
                        continue
                    if len(t) >= 2:
                        slope = (
                            0.5 * (t[-1][1] + t[-1][2]) - 0.5 * (t[-2][1] + t[-2][2])
                        ) / max(1.0, t[-1][0] - t[-2][0])
                        ypred = 0.5 * (ly0 + ly4) + slope * (xc - lx)
                    else:
                        ypred = 0.5 * (ly0 + ly4)
                    if abs(ymid - ypred) <= lsp * 1.65 and abs(sp - lsp) <= lsp * 0.22:
                        d = abs(ymid - ypred)
                        if d < best_dist:
                            best_dist = d
                            best_t = t
                if best_t is not None:
                    best_t.append(c)
                else:
                    tracks.append([c])

        valid = [t for t in tracks if len(t) >= 3]
        if not valid:
            return []

        # 全頁譜線間距一致性約束（排除任何殘影假窄譜）
        page_sp = float(np.median([np.median([c[3] for c in t]) for t in valid]))
        valid = [t for t in valid if page_sp * 0.75 <= np.median([c[3] for c in t]) <= page_sp * 1.28]
        valid.sort(key=lambda t: float(np.mean([c[1] for c in t])))

        dedup: List[List[Tuple[float, float, float, float]]] = []
        for t in valid:
            ym = float(np.mean([0.5 * (c[1] + c[2]) for c in t]))
            if dedup and abs(ym - float(np.mean([0.5 * (c[1] + c[2]) for c in dedup[-1]]))) < page_sp * 3.0:
                if len(t) > len(dedup[-1]):
                    dedup[-1] = t
            else:
                dedup.append(t)
        return dedup

    @staticmethod
    def _straighten_staff_strip(
        clean_gray: np.ndarray,
        track: List[Tuple[float, float, float, float]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, int, int]:
        """對單行五線譜區域執行獨立斜率搜尋 + 5齒梳狀微彎正交拉直，讓橫線歸 0° 且小節線歸 90°"""
        h, w = clean_gray.shape[:2]
        sp_est = float(np.median([c[3] for c in track]))
        ymid_est = float(np.median([0.5 * (c[1] + c[2]) for c in track]))

        half_band = int(round(max(sp_est * 6.8, w * 0.085)))
        by1 = max(0, int(round(ymid_est - half_band)))
        by2 = min(h, int(round(ymid_est + half_band)))
        band_gray = clean_gray[by1:by2, :]
        bh = band_gray.shape[0]
        if bh < int(sp_est * 5):
            pad = int(round(sp_est * 3.8))
            z = np.zeros(w, dtype=np.float32)
            return np.zeros((32, w), dtype=np.uint8), z, z, float(pad + 2 * sp_est), float(pad), float(pad + 4 * sp_est), sp_est, 0, w - 1

        band_bin = ((band_gray < 212).astype(np.uint8))
        hlen = max(12, w // 70)
        band_horiz = cv2.morphologyEx(band_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1)))

        # 第一階段：全寬度垂直切變搜尋此行五線譜的真實獨立斜率 (5條全寬長線產生最高投影銳利度，不受短橫槓干擾)
        sw_small = min(w, 650)
        small_h_mat = cv2.resize(band_horiz.astype(np.float32), (sw_small, bh), interpolation=cv2.INTER_AREA)
        xs_norm = np.linspace(-0.5, 0.5, sw_small, dtype=np.float32)

        def eval_slope_sharpness(dy_total: float) -> float:
            shifts = np.round(dy_total * xs_norm).astype(np.int32)
            r_idx = np.arange(bh, dtype=np.int32)[:, None] + shifts[None, :]
            valid_m = (r_idx >= 0) & (r_idx < bh)
            warped = small_h_mat[np.clip(r_idx, 0, bh - 1), np.arange(sw_small)[None, :]] * valid_m
            proj = np.sum(warped, axis=1)
            return float(np.sum(np.diff(proj) ** 2))

        # 以分片追蹤點對的 Theil-Sen 中位數斜率作為穩健初值，在 ±1.2° 範圍內精搜全寬最佳斜率（完全免疫漸強弱記號 < > 干擾）
        tx_trk = np.array([c[0] for c in track], dtype=np.float64)
        ty_trk = np.array([0.5 * (c[1] + c[2]) for c in track], dtype=np.float64)
        pair_slopes = []
        for i_p in range(len(tx_trk)):
            for j_p in range(i_p + 1, len(tx_trk)):
                if tx_trk[j_p] - tx_trk[i_p] >= w * 0.15:
                    pair_slopes.append((ty_trk[j_p] - ty_trk[i_p]) / (tx_trk[j_p] - tx_trk[i_p]))
        init_dy = float(np.median(pair_slopes) * w) if pair_slopes else 0.0

        search_span = w * math.tan(math.radians(1.5))
        best_dy = init_dy
        best_score = -1.0
        for dy_cand in np.linspace(init_dy - search_span, init_dy + search_span, 33):
            sc = eval_slope_sharpness(float(dy_cand))
            if sc > best_score:
                best_score = sc
                best_dy = float(dy_cand)

        for dy_cand in np.linspace(best_dy - search_span / 12.0, best_dy + search_span / 12.0, 17):
            sc = eval_slope_sharpness(float(dy_cand))
            if sc > best_score:
                best_score = sc
                best_dy = float(dy_cand)

        x_all = np.arange(w, dtype=np.float64)
        linear_offset = best_dy * ((x_all - w / 2.0) / max(1.0, float(w)))

        pad = int(round(sp_est * 3.8))
        shifts_full = np.round(linear_offset).astype(np.int32)
        r_idx_f = np.arange(bh, dtype=np.int32)[:, None] + shifts_full[None, :]
        valid_f = (r_idx_f >= 0) & (r_idx_f < bh)
        aligned_horiz = band_horiz[np.clip(r_idx_f, 0, bh - 1), np.arange(w)[None, :]] * valid_f
        full_proj = np.sum(aligned_horiz > 0, axis=1).astype(np.float32)
        ps_full = np.convolve(full_proj, np.array([0.25, 0.6, 1.0, 0.6, 0.25], dtype=np.float32), mode="same")

        # 5 條線梳狀濾波器搜尋：找出總能量最高的 5 等距長譜線（完全免疫短橫槓與漸強弱記號！）
        best_comb_score = -1.0
        best_y0_rel = float(bh / 2.0 - 2.0 * sp_est)
        best_sp = sp_est
        for sp_try in np.linspace(sp_est * 0.88, sp_est * 1.12, 17):
            offsets_5 = np.round(np.arange(5) * sp_try).astype(np.int32)
            max_y0 = bh - 1 - int(offsets_5[-1])
            if max_y0 <= 1:
                continue
            y0_candidates = np.arange(1, max_y0)
            comb_vals = np.sum(ps_full[y0_candidates[:, None] + offsets_5[None, :]], axis=1)
            idx_max = int(np.argmax(comb_vals))
            val_max = float(comb_vals[idx_max])
            if val_max > best_comb_score:
                best_comb_score = val_max
                best_y0_rel = float(y0_candidates[idx_max])
                best_sp = float(sp_try)

        # 第二階段：細微書頁彎曲追蹤（限制局部微調量 |δy| <= 0.36*sp，絕不可能跳行或吸到橫槓）
        max_local_delta = max(2, int(round(best_sp * 0.36)))
        n_segs = 10
        seg_xs = []
        seg_deltas = []
        offsets_5 = np.round(np.arange(5) * best_sp).astype(np.int32)
        for s_i in range(n_segs):
            sx0 = int(round(s_i * w / n_segs))
            sx1 = int(round((s_i + 1) * w / n_segs))
            seg_proj = np.sum(aligned_horiz[:, sx0:sx1] > 0, axis=1).astype(np.float32)
            best_d = 0
            best_d_val = -1.0
            base_val = 0.0
            for d_try in range(-max_local_delta, max_local_delta + 1):
                rows_5 = int(round(best_y0_rel)) + d_try + offsets_5
                if rows_5[0] < 0 or rows_5[-1] >= bh:
                    continue
                v = float(np.sum(seg_proj[rows_5]))
                if d_try == 0:
                    base_val = v
                if v > best_d_val:
                    best_d_val = v
                    best_d = d_try
            if best_d_val >= (sx1 - sx0) * 1.1 and best_d_val > base_val * 1.03:
                seg_xs.append(0.5 * (sx0 + sx1))
                seg_deltas.append(float(best_d))
            else:
                seg_xs.append(0.5 * (sx0 + sx1))
                seg_deltas.append(0.0)

        smooth_delta = np.interp(x_all, seg_xs, seg_deltas)
        smooth_delta = cv2.GaussianBlur(smooth_delta.reshape(1, -1).astype(np.float32), (1, 91), 0).ravel()

        y0_curve = float(by1) + best_y0_rel + linear_offset + smooth_delta
        slope_arr = np.gradient(y0_curve).astype(np.float32)

        strip_h = max(32, int(round(4.0 * best_sp + 2 * pad)))
        center_dy = float(pad + 2.0 * best_sp)
        y_shift = (y0_curve - pad).astype(np.float32)

        dy_grid = np.arange(strip_h, dtype=np.float32)[:, None]
        x_grid = np.arange(w, dtype=np.float32)[None, :]
        # 正交展開：水平線拉平的同時，垂直小節線與符幹亦轉正為 90° 垂直
        map_x = x_grid - slope_arr[None, :] * (dy_grid - center_dy)
        map_y = y_shift[None, :] + dy_grid
        strip_gray = cv2.remap(
            clean_gray, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        strip_bin = ((strip_gray < 210).astype(np.uint8)) * 255

        horiz = cv2.morphologyEx(
            strip_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, w // 35), 1))
        )
        row_proj = np.sum(horiz > 0, axis=1).astype(np.float32)

        refined_lines = []
        for k_line in range(5):
            exp_y = pad + k_line * best_sp
            r0 = max(0, int(round(exp_y - max(2.0, best_sp * 0.25))))
            r1 = min(strip_h, int(round(exp_y + max(2.0, best_sp * 0.25))) + 1)
            w_seg = row_proj[r0:r1]
            if np.sum(w_seg) > w * 0.15:
                refined_lines.append(float(np.average(np.arange(r0, r1), weights=w_seg)))
            else:
                refined_lines.append(float(exp_y))
        sy1 = refined_lines[0]
        sy2 = refined_lines[4]
        sp = (sy2 - sy1) / 4.0

        col_proj = np.sum(horiz[max(0, int(sy1 - 3)):min(strip_h, int(sy2 + 4)), :] > 0, axis=0)
        xs_col = np.where(col_proj >= 2)[0]
        xl = int(xs_col[0]) if len(xs_col) > 0 else 0
        xr = int(xs_col[-1]) if len(xs_col) > 0 else w - 1

        return strip_bin, y_shift, slope_arr, center_dy, sy1, sy2, sp, xl, xr

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
    def _estimate_rest_bar_count(strip_bin: np.ndarray, bar: _HorizBar, sy1: float, sp: float) -> int:
        """檢查多小節休止橫條正上方的數字區域並辨識休止小節數"""
        sh, sw = strip_bin.shape[:2]
        ry1 = max(0, int(sy1 - sp * 2.8))
        ry2 = min(sh, int(min(bar.y_center - sp * 0.45, sy1 + sp * 1.15)))
        rx1 = max(0, int(bar.x1))
        rx2 = min(sw, int(bar.x2))
        if rx2 <= rx1 + 4 or ry2 <= ry1 + 4:
            return 1

        patch = strip_bin[ry1:ry2, rx1:rx2].copy()
        pw = patch.shape[1]
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

        bot_band = digit_roi[max(0, int(dh * 0.78)):, :]
        bot_fill = float(np.mean(bot_band > 0)) if bot_band.size > 0 else 0.0

        padded = cv2.copyMakeBorder(digit_roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        contours, hier = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes_bot = 0
        if hier is not None:
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
    def _detect_notes_and_bars(
        strip_bin: np.ndarray,
        sy1: float,
        sy2: float,
        sp: float,
        xl: int,
        xr: int,
    ) -> Tuple[List[_RawNote], List[_HorizBar], float]:
        """在已水平拉直的 strip_bin 上略過譜頭拍號 (2/4)，切出每顆獨立音符與橫槓"""
        roi = strip_bin[:, xl:xr].copy()

        # 1. 量測此行 5 條五線譜線實際第 85 百分位線寬，全域抹除所有細橫線（五線譜線 + 加線翅膀），並以原圖遮罩回補 1px
        h_lines_only = cv2.morphologyEx(
            roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, roi.shape[1] // 14), 1))
        )
        line_thicks = []
        for k_l in range(5):
            yc_l = int(round(sy1 + k_l * sp))
            band_l = h_lines_only[max(0, yc_l - 4):min(roi.shape[0], yc_l + 5), :]
            cnts = np.sum(band_l > 0, axis=0)
            nz = cnts[cnts > 0]
            if len(nz) > 10:
                line_thicks.append(float(np.percentile(nz, 96)))
        est_line_t = float(np.median(line_thicks)) if line_thicks else (sp * 0.25)
        max_thin = max(2, min(int(round(sp * 0.42)), int(round(est_line_t))))
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
                header_skip = max(header_skip, bx + bw + sp * 0.6)
            elif sp * 2.8 <= bx <= sp * 8.5 and sp * 0.85 <= bw <= sp * 2.3:
                if by >= sy1 - sp * 0.55 and (by + bh) <= sy2 + sp * 0.55:
                    x0, x1 = max(0, bx - 2), min(scan_w, bx + bw + 2)
                    up_ink = int(np.sum(vert_keep[max(0, int(sy1)):min(roi.shape[0], int(sy1 + 1.9 * sp)), x0:x1] > 0))
                    lo_ink = int(np.sum(vert_keep[max(0, int(sy1 + 2.1 * sp)):min(roi.shape[0], int(sy2)), x0:x1] > 0))
                    if up_ink >= sp * sp * 0.40 and lo_ink >= sp * sp * 0.40:
                        header_skip = max(header_skip, bx + bw + sp * 0.8)

        header_end_x = min(float(xr) - sp, float(xl) + header_skip)

        # 3. 提取細直符幹與小節線 (pure_stems)，以及長橫槓/多小節休止條 (beams_mask)
        v_stems = cv2.morphologyEx(vert_keep, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(6, int(round(sp * 1.4))))))
        horiz_thick = cv2.morphologyEx(vert_keep, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(round(sp * 0.50))), 1)))
        pure_stems = cv2.subtract(v_stems, horiz_thick)

        beam_w = max(8, int(round(sp * 1.65)))
        beam_h = max(max_thin + 1, int(round(sp * 0.30)))
        raw_beams = cv2.morphologyEx(vert_keep, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (beam_w, beam_h)))

        num_b, blabels, bstats, bcent = cv2.connectedComponentsWithStats(raw_beams)
        beams_mask = np.zeros_like(raw_beams)
        horiz_bars: List[_HorizBar] = []
        min_beam_stem_gap = max(5, int(round(sp * 0.65)))
        for lbl in range(1, num_b):
            bx = int(bstats[lbl, cv2.CC_STAT_LEFT])
            by = int(bstats[lbl, cv2.CC_STAT_TOP])
            bw = int(bstats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(bstats[lbl, cv2.CC_STAT_HEIGHT])
            cy = float(bcent[lbl][1])
            # 真實連槓必連接 >=2 根獨立符幹，或多小節休止橫條寬度 >= 2.25*sp；排除單顆符頭連帶殘留譜線被誤切！
            sx0, sx1 = max(0, bx - 2), min(roi.shape[1], bx + bw + 2)
            sy0, sy1_b = max(0, by - int(sp * 0.8)), min(roi.shape[0], by + bh + int(sp * 0.8))
            b_scols = np.where(np.sum(pure_stems[sy0:sy1_b, sx0:sx1] > 0, axis=0) >= 3)[0]
            b_nstems = (1 + int(np.sum(np.diff(b_scols) > min_beam_stem_gap))) if len(b_scols) > 0 else 0
            if bw < sp * 2.25 and b_nstems < 2:
                continue
            beams_mask[blabels == lbl] = 255
            in_mid = (sy1 + sp * 1.15) <= cy <= (sy2 - sp * 1.15)
            horiz_bars.append(
                _HorizBar(
                    x1=float(xl + bx),
                    x2=float(xl + bx + bw),
                    y_center=cy,
                    width=float(bw),
                    height=float(bh),
                    in_staff_mid=in_mid,
                    n_stems=b_nstems,
                )
            )

        # 4. 同時減去長橫槓與細直符幹，讓每顆符頭 100% 獨立分離
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
                and (sp * 0.42 <= cw <= sp * 2.25)
                and (0.50 <= aspect <= 2.30)
                and (area_ratio >= 0.20)
                and ((sy1 - sp * 2.5) <= cy <= (sy2 + sp * 2.5))
            ):
                continue

            sx0, sx1 = max(0, nx - 2), min(roi.shape[1], nx + cw + 2)
            sy0, sy1_p = max(0, ny - int(sp * 0.8)), min(roi.shape[0], ny + ch + int(sp * 0.8))
            stem_cols = np.where(np.sum(pure_stems[sy0:sy1_p, sx0:sx1] > 0, axis=0) >= 3)[0]
            min_stem_gap = max(5, int(round(sp * 0.65)))
            n_stems = (1 + int(np.sum(np.diff(stem_cols) > min_stem_gap))) if len(stem_cols) > 0 else 0

            if n_stems >= 2 and (cw >= sp * 1.15 or area_ratio < 0.68):
                horiz_bars.append(
                    _HorizBar(
                        x1=float(xl + nx),
                        x2=float(xl + nx + cw),
                        y_center=cy,
                        width=float(cw),
                        height=float(ch),
                        in_staff_mid=False,
                    )
                )
                continue

            is_grace_cand = area_ratio < 0.72
            if is_grace_cand and n_stems == 0:
                continue

            candidates.append(
                _RawNote(
                    x=xl + cx,
                    y=cy,
                    width=float(cw),
                    area_ratio=area_ratio,
                    is_grace=is_grace_cand,
                    is_hollow=False,
                )
            )

        # 5. 過濾單根符幹尾端的 16 分音符短勾槓與單顆裝飾音上方的斜線小旗標
        main_cands = [c for c in candidates if c.area_ratio >= 0.90]
        grace_cands = [c for c in candidates if c.area_ratio < 0.90]
        filtered_cands: List[_RawNote] = []
        for c in candidates:
            if c.area_ratio < 0.90:
                attached_main = [
                    m for m in main_cands
                    if abs(c.x - m.x) <= sp * 1.35 and abs(c.y - m.y) >= sp * 1.50
                ]
                if attached_main:
                    short_hooks.append((c.x, c.y))
                    continue
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
                                y=hcy,
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

        # 8. 計算符尾橫槓數量与附點
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
                dot_pts.append((xl + float(dcent[dlbl][0]), float(dcent[dlbl][1])))

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
    def _detect_barlines_and_repeats(
        strip_bin: np.ndarray,
        sy1: float,
        sy2: float,
        sp: float,
        xl: int,
        xr: int,
        note_pts: List[Tuple[float, float]],
        horiz_bars: Optional[List[_HorizBar]] = None,
    ) -> List[_BarlineInfo]:
        """在水平拉直的五線譜帶內偵測小節線與前/後/雙向反覆記號 (||: / :|| / :||:)"""
        iy1 = max(0, int(round(sy1)))
        iy2 = min(strip_bin.shape[0], int(round(sy2)) + 1)
        staff_h = max(1, iy2 - iy1)
        if staff_h < 12:
            return []

        xr_pad = min(strip_bin.shape[1], xr + 4)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(4, int(staff_h * 0.80))))
        v_lines = cv2.morphologyEx(strip_bin[iy1:iy2, xl:xr_pad], cv2.MORPH_OPEN, v_kernel)
        v_proj = np.sum(v_lines > 0, axis=0)
        bx = np.where(v_proj > staff_h * 0.78)[0]
        if len(bx) == 0:
            return []

        # 擷取第二間 (sy1+sp .. sy1+2*sp) 與第三間 (sy1+2*sp .. sy1+3*sp) 的無幹小圓點，用於辨識反覆記號雙點 ":"
        by0 = max(0, int(round(sy1 + 0.80 * sp)))
        by1_d = min(strip_bin.shape[0], int(round(sy1 + 3.20 * sp)))
        dots: List[Tuple[float, float, int, int, int]] = []
        if by1_d > by0 + 6:
            band = strip_bin[by0:by1_d, :].copy()
            h_line = cv2.morphologyEx(band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(sp * 1.2)), 1)))
            v_stem = cv2.morphologyEx(band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(6, int(sp * 1.1)))))
            dots_img = cv2.subtract(cv2.subtract(band, h_line), v_stem)
            nl, _, d_st, d_ct = cv2.connectedComponentsWithStats(dots_img)
            for l_i in range(1, nl):
                dw = int(d_st[l_i, cv2.CC_STAT_WIDTH])
                dh = int(d_st[l_i, cv2.CC_STAT_HEIGHT])
                da = int(d_st[l_i, cv2.CC_STAT_AREA])
                dcx = float(d_ct[l_i][0])
                dcy = float(by0) + float(d_ct[l_i][1])
                if 2 <= dw <= sp * 1.05 and 2 <= dh <= sp * 0.95 and sp * sp * 0.028 <= da <= sp * sp * 0.45:
                    dots.append((dcx, dcy, dw, dh, da))

        def is_valid_barline(c_left: float, c_right: float) -> bool:
            cand_x = 0.5 * (c_left + c_right)
            if any(abs(cand_x - nx) <= sp * 0.88 for nx, _ in note_pts):
                return False
            # 單細線若連接多符幹節奏連槓 (n_stems >= 2 且寬度 <= 11.5*sp)，必為十六/八分音符尾端符幹而非小節線
            if (c_right - c_left) < sp * 0.55 and horiz_bars:
                if any(
                    hb.n_stems >= 2
                    and hb.width <= sp * 11.5
                    and (hb.x1 - sp * 0.45 <= cand_x <= hb.x2 + sp * 0.45)
                    for hb in horiz_bars
                ):
                    return False
            if (c_right - c_left) < sp * 0.55:
                cx_i = int(np.clip(round(cand_x), 1, strip_bin.shape[1] - 2))
                up_r1 = max(0, iy1 - int(round(sp * 1.35)))
                up_r2 = max(0, iy1 - int(round(sp * 0.45)))
                dn_r1 = min(strip_bin.shape[0], iy2 + int(round(sp * 0.45)))
                dn_r2 = min(strip_bin.shape[0], iy2 + int(round(sp * 1.35)))
                if up_r2 > up_r1 + 2:
                    if float(np.mean(np.max(strip_bin[up_r1:up_r2, cx_i - 1:cx_i + 2], axis=1) > 0)) > 0.80:
                        return False
                if dn_r2 > dn_r1 + 2:
                    if float(np.mean(np.max(strip_bin[dn_r1:dn_r2, cx_i - 1:cx_i + 2], axis=1) > 0)) > 0.80:
                        return False
            return True

        def has_repeat_dots(x_min: float, x_max: float, is_double_bar: bool, span_px: float) -> bool:
            # 音樂排版規範：反覆記號必搭配雙小節線/粗細小節線（總跨度 >= 0.50*sp），單一細小節線絕非反覆記號！
            if span_px < sp * 0.50:
                return False
            up = [d for d in dots if x_min <= d[0] <= x_max and (sy1 + 1.05 * sp) <= d[1] <= (sy1 + 1.95 * sp)]
            dn = [d for d in dots if x_min <= d[0] <= x_max and (sy1 + 2.05 * sp) <= d[1] <= (sy1 + 2.95 * sp)]
            for u in up:
                for v in dn:
                    if abs(u[0] - v[0]) <= sp * 0.45:
                        return True
            if is_double_bar:
                centered = [
                    d for d in (up + dn)
                    if d[4] >= sp * sp * 0.10
                    and 0.50 <= (d[2] / max(1.0, float(d[3]))) <= 1.85
                    and (
                        ((sy1 + 1.22 * sp) <= d[1] <= (sy1 + 1.78 * sp))
                        or ((sy1 + 2.22 * sp) <= d[1] <= (sy1 + 2.78 * sp))
                    )
                ]
                if centered:
                    return True
            return False

        clusters: List[Tuple[int, int]] = []
        b_start = b_prev = int(bx[0])
        for x in bx[1:]:
            x = int(x)
            if x > b_prev + max(5, int(sp * 1.15)):
                clusters.append((b_start, b_prev))
                b_start = x
            b_prev = x
        clusters.append((b_start, b_prev))

        bar_infos: List[_BarlineInfo] = []
        for bs, bp in clusters:
            c_l = float(xl + bs)
            c_r = float(xl + bp)
            if not is_valid_barline(c_l, c_r):
                continue
            vc_slice = v_proj[bs:bp + 1]
            span_px = c_r - c_l
            is_double = (span_px >= sp * 0.55) and (int(np.min(vc_slice)) == 0 or span_px >= sp * 0.75)
            rep_end = has_repeat_dots(c_l - sp * 1.48, c_l - 1.0, is_double, span_px)
            rep_start = has_repeat_dots(c_r + 1.0, c_r + sp * 1.48, is_double, span_px)
            bar_infos.append(
                _BarlineInfo(
                    x_center=0.5 * (c_l + c_r),
                    x_left=c_l,
                    x_right=c_r,
                    repeat_end_left=rep_end,
                    repeat_start_right=rep_start,
                )
            )
        return bar_infos

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
