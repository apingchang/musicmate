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
    accidental: int = 0
    articulation: str = ""

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
    detected_time_sig: Optional[str] = None
    detected_bpm: Optional[int] = None
    staves: List[StaffInfo] = field(default_factory=list)
    events: List[NoteEvent] = field(default_factory=list)


@dataclass
class ScoreAnalysis:
    """整份樂譜的分析結果與總演奏時間軸（含反覆記號展開序列與自動辨識拍號/BPM）"""
    pages: List[PageAnalysis] = field(default_factory=list)
    events: List[NoteEvent] = field(default_factory=list)
    playback_measures: List[PlaybackMeasure] = field(default_factory=list)
    repeat_sections: List[Tuple[int, int]] = field(default_factory=list)
    total_beats: float = 0.0
    total_measures: int = 0
    beats_per_measure: float = 4.0
    enable_repeats: bool = True
    detected_time_sig: Optional[str] = None
    detected_bpm: Optional[int] = None

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
    accidental: int = 0
    has_natural: bool = False
    articulation: str = ""


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
            if analysis.detected_time_sig is None and page_res.detected_time_sig:
                analysis.detected_time_sig = page_res.detected_time_sig
            if analysis.detected_bpm is None and page_res.detected_bpm:
                analysis.detected_bpm = page_res.detected_bpm
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
            if m.rest_measures > 0 or not m.notes:
                m_dur = beats_per_measure * max(1, m.rest_measures)
            else:
                m_dur = sum(ev0.duration_beats for ev0 in m.notes)
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

        # 1. 自適應標準化分析寬度（橫向截圖約 1300px，直向整頁 A4 多行樂譜自動提升至最高 2150px 確保密集十六分音符間距充足）
        aspect_hw = orig_h / max(1.0, float(orig_w))
        target_w = int(np.clip(self.BASE_WIDTH * max(1.0, aspect_hw * 1.15), 1300, 2150))
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

            # 4b. 若為本頁第一行五線譜，掃描其正上方速度標示區（如 Allegro moderato (♩ = 116)）自動辨識 BPM
            if s_idx == 0 and page_res.detected_bpm is None:
                st1_top_page_y = float(np.mean(y_shift[xl:xr]) + sy1)
                page_res.detected_bpm = self._detect_tempo_bpm(clean_gray, st1_top_page_y, sp)

            # 5. 在 100% 拉直的五線譜帶上偵測譜頭拍號、獨立音符、臨時升降記號、附點、保持音與小節線
            raw_notes, horiz_bars, header_end_x, det_ts = self._detect_notes_and_bars(
                strip_bin, sy1, sy2, sp, xl, xr
            )
            if s_idx == 0 and det_ts and page_res.detected_time_sig is None:
                page_res.detected_time_sig = det_ts

            main_note_pts = [(n.x, n.y) for n in raw_notes if not n.is_grace]
            bar_infos = self._detect_barlines_and_repeats(
                strip_bin, sy1, sy2, sp, xl, xr, main_note_pts, horiz_bars
            )

            # 6. 過濾緊鄰小節線的反覆記號雙點 (|: / :|，必位於譜內第 2~3 間且寬度 < 0.80*sp) 與黏貼於小節線上的假裝飾音（如圓滑線交叉點），並清除誤吸反覆點的假附點
            if bar_infos:
                raw_notes = [
                    n for n in raw_notes
                    if not (
                        (
                            n.width < sp * 0.80
                            and (sy1 + sp * 0.80) <= n.y <= (sy2 - sp * 0.80)
                            and any(b.x_left - sp * 1.50 <= n.x <= b.x_right + sp * 1.50 for b in bar_infos)
                        )
                        or (
                            n.is_grace
                            and any(b.x_left - sp * 0.85 <= n.x <= b.x_right + sp * 0.85 for b in bar_infos)
                        )
                    )
                ]
                for n in raw_notes:
                    if n.is_dotted and any(
                        b.repeat_end_left and 0.0 < (b.x_left - n.x) <= sp * 2.4 for b in bar_infos
                    ):
                        n.is_dotted = False
                    # 若臨時記號偵測區不慎跨過小節線，清除緊貼小節線右側第一顆音的假升降號
                    if (n.accidental != 0 or n.has_natural) and any(
                        0.0 < (n.x - b.x_right) <= sp * 1.35 for b in bar_infos
                    ):
                        n.accidental = 0
                        n.has_natural = False

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
                # 若音符帶有延長記號 (Fermata 𝄐)，自動將演奏持續拍數延長為 1.75 倍
                for idx_d, n_d in enumerate(m_notes_all):
                    if "延長記號" in n_d.articulation and not n_d.is_grace:
                        durations[idx_d] *= 1.75
                m_beat_end = m_beat_start + sum(durations)
                measure_info.beat_end = m_beat_end
                local_beat = m_beat_start
                total_in_m = len(m_notes_all)
                # 同小節內臨時升降記號延續表 {diatonic_step: accidental (-1/0/+1)}，跨小節自動重置
                measure_acc_state = {}
                for k_idx, (note_obj, dur, n_type) in enumerate(zip(m_notes_all, durations, note_types), start=1):
                    nx_n, ny_n = to_page_norm(note_obj.x, note_obj.y)
                    step_idx = int(round((sy2 - note_obj.y) / (sp / 2.0)))
                    if not note_obj.is_grace:
                        if note_obj.accidental != 0:
                            measure_acc_state[step_idx] = note_obj.accidental
                        elif note_obj.has_natural:
                            measure_acc_state[step_idx] = 0
                    eff_acc = note_obj.accidental if note_obj.is_grace else measure_acc_state.get(step_idx, 0)
                    midi_val, n_name = self._y_to_treble_pitch(
                        note_obj.y, sy2, sp, accidental=eff_acc, show_natural=note_obj.has_natural
                    )
                    full_type = f"{n_type} • {note_obj.articulation}" if note_obj.articulation else n_type
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
                        note_type=full_type,
                        accidental=eff_acc,
                        articulation=note_obj.articulation,
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

            cand_combs: List[Tuple[float, float, float, float, float]] = []
            for idx in range(max(0, len(peaks) - 4)):
                grp_y = [peaks[idx + k][0] for k in range(5)]
                grp_s = [peaks[idx + k][1] for k in range(5)]
                diffs = np.diff(grp_y)
                med = float(np.median(diffs))
                if min_sp <= med <= max_sp and np.all(np.abs(diffs - med) <= max(2.2, med * 0.22)):
                    # 以總強度 + 2.5 倍最弱線強度評分，確保 5 條完整譜線必勝過「4 條譜線 + 1 條高音加線/反覆括號線」！
                    c_score = float(sum(grp_s) + 2.5 * min(grp_s))
                    cand_combs.append((c_score, float(xc), grp_y[0], grp_y[4], med))
            cand_combs.sort(key=lambda item: item[0], reverse=True)
            combs: List[Tuple[float, float, float, float]] = []
            for _, c_xc, cy0, cy4, cmed in cand_combs:
                c_mid = 0.5 * (cy0 + cy4)
                if all(abs(c_mid - 0.5 * (e[1] + e[2])) >= cmed * 3.5 for e in combs):
                    combs.append((c_xc, cy0, cy4, cmed))
            combs.sort(key=lambda e: e[1])
            slice_combs.append(combs)

        # 跨分片串聯同一行五線譜軌跡（限制相鄰分片跳動 < 0.75*sp，徹底杜絕跨片誤跳 1 個譜間距！）
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
                    max_jump = max(lsp * 0.72, abs(xc - lx) * 0.085)
                    if abs(ymid - ypred) <= max_jump and abs(sp - lsp) <= lsp * 0.22:
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

        # 第一階段：直接以分片追蹤點 (tx_trk, ty_trk) 建構真實書頁彎曲/傾斜基準曲線，再做微幅斜率精調（徹底解決右半書頁彎曲偏移 1 線問題！）
        sw_small = min(w, 650)
        small_h_mat = cv2.resize(band_horiz.astype(np.float32), (sw_small, bh), interpolation=cv2.INTER_AREA)
        xs_small_px = np.linspace(0.0, float(w - 1), sw_small, dtype=np.float64)
        xs_norm = np.linspace(-0.5, 0.5, sw_small, dtype=np.float32)

        tx_trk = np.array([c[0] for c in track], dtype=np.float64)
        ty_trk = np.array([0.5 * (c[1] + c[2]) for c in track], dtype=np.float64)
        x_all = np.arange(w, dtype=np.float64)

        if len(tx_trk) >= 4 and (tx_trk[-1] - tx_trk[0]) >= w * 0.32:
            # 二階多項式離群值剔除，保留平滑書頁曲線並排除單片雜訊
            if len(tx_trk) >= 5:
                poly2 = np.polyfit(tx_trk, ty_trk, deg=2)
                resid = np.abs(ty_trk - np.polyval(poly2, tx_trk))
                keep_m = resid <= max(2.5, sp_est * 0.48)
                if int(np.sum(keep_m)) >= 4:
                    tx_trk = tx_trk[keep_m]
                    ty_trk = ty_trk[keep_m]

            raw_curve = np.interp(x_all, tx_trk, ty_trk)
            k_l = min(2, len(tx_trk) - 1)
            sl_l = float(np.clip((ty_trk[k_l] - ty_trk[0]) / max(1.0, tx_trk[k_l] - tx_trk[0]), -0.12, 0.12))
            sl_r = float(np.clip((ty_trk[-1] - ty_trk[-1 - k_l]) / max(1.0, tx_trk[-1] - tx_trk[-1 - k_l]), -0.12, 0.12))
            left_m = x_all < tx_trk[0]
            right_m = x_all > tx_trk[-1]
            if np.any(left_m):
                raw_curve[left_m] = ty_trk[0] + sl_l * (x_all[left_m] - tx_trk[0])
            if np.any(right_m):
                raw_curve[right_m] = ty_trk[-1] + sl_r * (x_all[right_m] - tx_trk[-1])

            g_win = max(31, (int(w // 10) | 1))
            pad_g = g_win
            left_pad = raw_curve[0] + sl_l * np.arange(-pad_g, 0, dtype=np.float64)
            right_pad = raw_curve[-1] + sl_r * np.arange(1, pad_g + 1, dtype=np.float64)
            padded_curve = np.concatenate([left_pad, raw_curve, right_pad]).astype(np.float32)
            raw_curve = cv2.GaussianBlur(
                padded_curve.reshape(1, -1), (g_win, 1), 0
            ).ravel()[pad_g:pad_g + w].astype(np.float64)
            init_offset_full = raw_curve - float(np.mean(raw_curve))
            init_offset_small = np.interp(xs_small_px, x_all, init_offset_full).astype(np.float32)

            def eval_curve_sharpness(delta_dy: float) -> float:
                shifts = np.round(init_offset_small + delta_dy * xs_norm).astype(np.int32)
                r_idx = np.arange(bh, dtype=np.int32)[:, None] + shifts[None, :]
                valid_m = (r_idx >= 0) & (r_idx < bh)
                warped = small_h_mat[np.clip(r_idx, 0, bh - 1), np.arange(sw_small)[None, :]] * valid_m
                proj = np.sum(warped, axis=1)
                return float(np.sum(np.diff(proj) ** 2))

            fine_span = w * math.tan(math.radians(0.45))
            best_delta_dy = 0.0
            best_score = -1.0
            for d_cand in np.linspace(-fine_span, fine_span, 19):
                sc = eval_curve_sharpness(float(d_cand))
                if sc > best_score:
                    best_score = sc
                    best_delta_dy = float(d_cand)
            linear_offset = init_offset_full + best_delta_dy * ((x_all - w / 2.0) / max(1.0, float(w)))
        else:
            def eval_slope_sharpness(dy_total: float) -> float:
                shifts = np.round(dy_total * xs_norm).astype(np.int32)
                r_idx = np.arange(bh, dtype=np.int32)[:, None] + shifts[None, :]
                valid_m = (r_idx >= 0) & (r_idx < bh)
                warped = small_h_mat[np.clip(r_idx, 0, bh - 1), np.arange(sw_small)[None, :]] * valid_m
                proj = np.sum(warped, axis=1)
                return float(np.sum(np.diff(proj) ** 2))

            pair_slopes = []
            for i_p in range(len(tx_trk)):
                for j_p in range(i_p + 1, len(tx_trk)):
                    if tx_trk[j_p] - tx_trk[i_p] >= w * 0.15:
                        pair_slopes.append((ty_trk[j_p] - ty_trk[i_p]) / (tx_trk[j_p] - tx_trk[i_p]))
            init_dy = float(np.median(pair_slopes) * w) if pair_slopes else 0.0

            search_span = w * math.tan(math.radians(1.5))
            dy_lo = min(init_dy - search_span, -search_span * 0.8)
            dy_hi = max(init_dy + search_span, search_span * 0.8)
            best_dy = init_dy
            best_score = -1.0
            for dy_cand in np.linspace(dy_lo, dy_hi, 37):
                sc = eval_slope_sharpness(float(dy_cand))
                if sc > best_score:
                    best_score = sc
                    best_dy = float(dy_cand)

            for dy_cand in np.linspace(best_dy - search_span / 12.0, best_dy + search_span / 12.0, 17):
                sc = eval_slope_sharpness(float(dy_cand))
                if sc > best_score:
                    best_score = sc
                    best_dy = float(dy_cand)

            linear_offset = best_dy * ((x_all - w / 2.0) / max(1.0, float(w)))

        pad = int(round(sp_est * 3.8))
        shifts_full = np.round(linear_offset).astype(np.int32)
        r_idx_f = np.arange(bh, dtype=np.int32)[:, None] + shifts_full[None, :]
        valid_f = (r_idx_f >= 0) & (r_idx_f < bh)
        aligned_horiz = band_horiz[np.clip(r_idx_f, 0, bh - 1), np.arange(w)[None, :]] * valid_f
        full_proj = np.sum(aligned_horiz > 0, axis=1).astype(np.float32)
        ps_full = np.convolve(full_proj, np.array([0.25, 0.6, 1.0, 0.6, 0.25], dtype=np.float32), mode="same")

        # 5 條線梳狀濾波器搜尋：同時要求總能量與最弱單線能量最高，徹底杜絕上加線/反覆括號導致整框上移 1 線！
        best_comb_score = -1.0
        best_y0_rel = float(bh / 2.0 - 2.0 * sp_est)
        best_sp = sp_est
        for sp_try in np.linspace(sp_est * 0.88, sp_est * 1.12, 17):
            offsets_5 = np.round(np.arange(5) * sp_try).astype(np.int32)
            max_y0 = bh - 1 - int(offsets_5[-1])
            if max_y0 <= 1:
                continue
            y0_candidates = np.arange(1, max_y0)
            teeth_5 = ps_full[y0_candidates[:, None] + offsets_5[None, :]]
            comb_vals = np.sum(teeth_5, axis=1) + 2.5 * np.min(teeth_5, axis=1)
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

        # 以較短水平核精確定位 5 條譜線的真實左右端點（避免 w//35 長核截斷行尾最後一小節如 M47 的 B4！）
        horiz_ends = cv2.morphologyEx(
            strip_bin,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(round(sp * 0.90))), 1)),
        )
        line_hits = np.zeros(w, dtype=np.int32)
        for rl in refined_lines:
            ry0 = max(0, int(round(rl)) - 2)
            ry1 = min(strip_h, int(round(rl)) + 3)
            line_hits += np.any(horiz_ends[ry0:ry1, :] > 0, axis=0).astype(np.int32)
        xs_col = np.where(line_hits >= 3)[0]
        if len(xs_col) == 0:
            col_proj = np.sum(horiz[max(0, int(sy1 - 3)):min(strip_h, int(sy2 + 4)), :] > 0, axis=0)
            xs_col = np.where(col_proj >= 2)[0]
        xl = int(xs_col[0]) if len(xs_col) > 0 else 0
        xr = int(xs_col[-1]) if len(xs_col) > 0 else w - 1

        # 若行尾緊接終止小節線，將 xr 向右延伸涵蓋該垂直小節線
        r_search0 = max(0, xr - 4)
        r_search1 = min(w, xr + max(6, int(round(sp * 1.4))))
        if r_search1 > r_search0 and (sy2 - sy1) > 8:
            v_end = cv2.morphologyEx(
                strip_bin[:, r_search0:r_search1],
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, int(round((sy2 - sy1) * 0.80))))),
            )
            v_end_cols = np.where(
                np.sum(v_end[max(0, int(sy1)):min(strip_h, int(sy2) + 1), :] > 0, axis=0) >= int((sy2 - sy1) * 0.80)
            )[0]
            if len(v_end_cols) > 0:
                xr = min(w - 1, r_search0 + int(v_end_cols[-1]) + 2)

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
    ) -> Tuple[List[_RawNote], List[_HorizBar], float, Optional[str]]:
        """在已水平拉直的 strip_bin 上辨識譜頭拍號 (如 2/4)，並切出獨立音符、臨時升降記號、附點與保持音"""
        roi = strip_bin[:, xl:xr].copy()

        # 1. 量測此行 5 條五線譜線實際第 96 百分位線寬，全域抹除所有細橫線（五線譜線 + 加線翅膀），並以原圖遮罩回補 1px
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

        # 2. 自動偵測譜頭結束位置並辨識拍號（如 2/4、3/4、4/4、6/8）
        header_skip = sp * 4.2
        detected_ts: Optional[str] = None
        scan_w = min(roi.shape[1], int(sp * 9.5))
        num_h, _, hstats, _ = cv2.connectedComponentsWithStats(vert_keep[:, :scan_w])
        ts_boxes: List[Tuple[int, int]] = []
        for lbl in range(1, num_h):
            bx = int(hstats[lbl, cv2.CC_STAT_LEFT])
            by = int(hstats[lbl, cv2.CC_STAT_TOP])
            bw = int(hstats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(hstats[lbl, cv2.CC_STAT_HEIGHT])
            if bx < sp * 3.8 and bh > sp * 4.8:
                header_skip = max(header_skip, bx + bw + sp * 0.6)
            elif sp * 2.5 <= bx <= sp * 8.8 and sp * 0.70 <= bw <= sp * 2.5:
                if by >= sy1 - sp * 0.55 and (by + bh) <= sy2 + sp * 0.55:
                    x0 = max(0, bx - 2)
                    x1 = min(scan_w, bx + bw + 2)
                    up_ink = int(np.sum(vert_keep[max(0, int(sy1)):min(roi.shape[0], int(sy1 + 1.9 * sp)), x0:x1] > 0))
                    lo_ink = int(np.sum(vert_keep[max(0, int(sy1 + 2.1 * sp)):min(roi.shape[0], int(sy2)), x0:x1] > 0))
                    if up_ink >= sp * sp * 0.35 and lo_ink >= sp * sp * 0.35:
                        header_skip = max(header_skip, bx + bw + sp * 0.8)
                        ts_boxes.append((x0, x1))

        if ts_boxes:
            tx0 = min(b[0] for b in ts_boxes)
            tx1 = max(b[1] for b in ts_boxes)
            detected_ts = ScoreAnalyzer._recognize_time_signature_patch(vert_keep, sy1, sy2, sp, tx0, tx1)

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

        # 3b. 先在未經侵蝕的原解析度影像上擷取「白邊隔離圓點」、「保持音短橫槓 (-)」與「延長記號 Fermata (𝄐 / 𝄑)」
        raw_roi = strip_bin[:, xl:xr]
        h_long = cv2.morphologyEx(raw_roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, int(round(sp * 1.25))), 1)))
        no_staff_roi = cv2.subtract(raw_roi, h_long)
        v_tall = cv2.morphologyEx(raw_roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, int(round(sp * 1.15))))))
        dot_cand_img = cv2.subtract(no_staff_roi, v_tall)

        num_d, _, dstats, dcent = cv2.connectedComponentsWithStats(dot_cand_img)
        true_dots: List[Tuple[float, float]] = []
        fermata_dots: List[Tuple[float, float]] = []
        for dlbl in range(1, num_d):
            dx0 = int(dstats[dlbl, cv2.CC_STAT_LEFT])
            dy0 = int(dstats[dlbl, cv2.CC_STAT_TOP])
            dw = int(dstats[dlbl, cv2.CC_STAT_WIDTH])
            dh = int(dstats[dlbl, cv2.CC_STAT_HEIGHT])
            da = int(dstats[dlbl, cv2.CC_STAT_AREA])
            dcx = float(xl) + float(dcent[dlbl][0])
            dcy = float(dcent[dlbl][1])

            rx0, rx1 = max(0, dx0 - 2), min(no_staff_roi.shape[1], dx0 + dw + 2)
            ry0, ry1_b = max(0, dy0 - 2), min(no_staff_roi.shape[0], dy0 + dh + 2)
            ring_ink = int(np.sum(no_staff_roi[ry0:ry1_b, rx0:rx1] > 0)) - da
            ring_area = max(1, (rx1 - rx0) * (ry1_b - ry0) - dw * dh)
            ring_ratio = ring_ink / float(ring_area)

            if (
                3 <= dw <= max(6, int(round(sp * 0.92)))
                and 3 <= dh <= max(6, int(round(sp * 0.85)))
                and 0.48 <= (dw / max(1.0, float(dh))) <= 1.90
                and sp * sp * 0.10 <= da <= sp * sp * 0.52
                and (da / max(1.0, float(dw * dh))) >= 0.45
                and ring_ratio <= 0.16
            ):
                fermata_dots.append((dcx, dcy))
                if dw <= max(5, int(round(sp * 0.62))) and dh <= max(5, int(round(sp * 0.62))) and da <= sp * sp * 0.38:
                    true_dots.append((dcx, dcy))

        # 偵測延長記號 Fermata (𝄐 / 𝄑)：直接在原始 raw_roi 上尋找五線譜上方或下方、朝譜面內凹的孤立彎弧，且正中央開口內嵌一顆圓點！
        fermata_zones: List[Tuple[float, float, float, float, float]] = []
        used_f_dots: List[Tuple[float, float]] = []
        num_raw, _, rstats, rcent = cv2.connectedComponentsWithStats(raw_roi)
        for nlbl in range(1, num_raw):
            ax0 = int(rstats[nlbl, cv2.CC_STAT_LEFT])
            ay0 = int(rstats[nlbl, cv2.CC_STAT_TOP])
            aw = int(rstats[nlbl, cv2.CC_STAT_WIDTH])
            ah = int(rstats[nlbl, cv2.CC_STAT_HEIGHT])
            if not (sp * 1.25 <= aw <= sp * 3.65 and sp * 0.45 <= ah <= sp * 2.15):
                continue
            is_upper_f = (ay0 + ah) <= (sy1 + sp * 0.35)
            is_lower_f = ay0 >= (sy2 - sp * 0.35)
            if not (is_upper_f or is_lower_f):
                continue
            arch_p = raw_roi[ay0:ay0 + ah, ax0:ax0 + aw] > 0
            ew = max(1, aw // 5)
            ys_mid = np.where(arch_p[:, aw // 3:max(aw // 3 + 1, 2 * aw // 3)])[0]
            ys_l = np.where(arch_p[:, :ew])[0]
            ys_r = np.where(arch_p[:, aw - ew:])[0]
            if len(ys_mid) == 0 or len(ys_l) == 0 or len(ys_r) == 0:
                continue
            y_mid_m = float(np.mean(ys_mid))
            y_end_m = 0.5 * (float(np.mean(ys_l)) + float(np.mean(ys_r)))
            if is_upper_f and not (y_mid_m < y_end_m - 1.0):
                continue
            if is_lower_f and not (y_mid_m > y_end_m + 1.0):
                continue
            fcx = float(xl) + float(ax0) + aw * 0.5
            matched_dots = [
                (fdx, fdy) for fdx, fdy in fermata_dots
                if abs(fdx - fcx) <= aw * 0.32
                and (
                    (is_upper_f and (ay0 + ah * 0.20) <= fdy <= (ay0 + ah + sp * 0.65))
                    or (is_lower_f and (ay0 - sp * 0.65) <= fdy <= (ay0 + ah * 0.80))
                )
            ]
            if matched_dots:
                fermata_zones.append((float(xl + ax0 - 4), float(ay0 - 4), float(xl + ax0 + aw + 4), float(ay0 + ah + 4), fcx))
                used_f_dots.extend(matched_dots)

        if used_f_dots:
            true_dots = [
                (dx, dy) for dx, dy in true_dots
                if all(math.hypot(dx - ux, dy - uy) > 3.5 for ux, uy in used_f_dots)
            ]

        tenuto_bars: List[Tuple[float, float]] = []
        num_raw, _, rstats, rcent = cv2.connectedComponentsWithStats(raw_roi)
        for rlbl in range(1, num_raw):
            tw = int(rstats[rlbl, cv2.CC_STAT_WIDTH])
            th = int(rstats[rlbl, cv2.CC_STAT_HEIGHT])
            ta = int(rstats[rlbl, cv2.CC_STAT_AREA])
            if (
                sp * 0.65 <= tw <= sp * 1.55
                and 1 <= th <= max(5, int(round(sp * 0.42)))
                and (tw / max(1.0, float(th))) >= 2.3
                and (ta / max(1.0, float(tw * th))) >= 0.68
            ):
                tenuto_bars.append((float(xl) + float(rcent[rlbl][0]), float(rcent[rlbl][1])))

        # 4. 預先分離貫穿全譜的小節線直欄（必須同時觸及頂線與底線且不大幅突出譜外，絕不誤把 E5/F5 長符幹當成小節線！）
        iy_bar1, iy_bar2 = max(0, int(round(sy1))), min(roi.shape[0], int(round(sy2)) + 1)
        if iy_bar2 > iy_bar1 + 8:
            span_h = iy_bar2 - iy_bar1
            cand_bcs = np.where(np.sum(v_stems[iy_bar1:iy_bar2, :] > 0, axis=0) >= int(span_h * 0.92))[0]
            top_touch = np.any(v_stems[iy_bar1:min(iy_bar2, iy_bar1 + 3), :] > 0, axis=0)
            bot_touch = np.any(v_stems[max(iy_bar1, iy_bar2 - 3):iy_bar2, :] > 0, axis=0)
            over_top = np.sum(v_stems[max(0, iy_bar1 - int(sp * 0.95)):max(0, iy_bar1 - 3), :] > 0, axis=0)
            over_bot = np.sum(v_stems[min(roi.shape[0], iy_bar2 + 3):min(roi.shape[0], iy_bar2 + int(sp * 0.95)), :] > 0, axis=0)
            bar_cols_roi = np.array(
                [bc for bc in cand_bcs if top_touch[bc] and bot_touch[bc] and over_top[bc] <= 2 and over_bot[bc] <= 2],
                dtype=np.int64,
            )
        else:
            bar_cols_roi = np.array([], dtype=np.int64)
        pure_stems_no_bars = pure_stems.copy()
        for bc in bar_cols_roi:
            pure_stems_no_bars[:, max(0, int(bc) - 2):min(roi.shape[1], int(bc) + 3)] = 0

        h_acc_lines = cv2.morphologyEx(
            roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(14, int(round(sp * 1.45))), 1))
        )
        vk_acc = cv2.subtract(cv2.subtract(roi, h_acc_lines), cv2.dilate(beams_mask, np.ones((2, 2), np.uint8)))

        roi_clean = cv2.subtract(vert_keep, cv2.dilate(beams_mask, np.ones((2, 2), np.uint8)))
        roi_clean = cv2.subtract(roi_clean, pure_stems)

        nh_w = max(3, int(round(sp * 0.50)))
        nh_h = max(3, int(round(sp * 0.44)))
        nh_mask = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nh_w, nh_h)))

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(nh_mask)
        all_blob_pts = [
            (float(centroids[i][0]), float(centroids[i][1]), int(stats[i, cv2.CC_STAT_AREA]))
            for i in range(1, num_labels)
        ]
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
            abs_cx = xl + cx
            if any(fx0 <= abs_cx <= fx1 and fy0 <= cy <= fy1 for fx0, fy0, fx1, fy1, _ in fermata_zones):
                continue

            aspect = cw / max(1.0, float(ch))
            area_ratio = area / max(1.0, sp * sp)

            # 寬區塊切分 (cw > 1.92*sp 且左右雙峰之間存在真實鞍部凹陷)：相容「[升降號 + 符頭]」與「[相鄰雙符頭微觸]」，絕不誤切單顆寬橢圓符頭！
            if (
                (sp * 1.92 < cw <= sp * 3.55)
                and (sp * 0.36 <= ch <= sp * 1.85)
                and ((sy1 - sp * 2.5) <= cy <= (sy2 + sp * 2.5))
            ):
                col_hs_w = np.sum(nh_mask[ny:ny + ch, nx:nx + cw] > 0, axis=0)
                mid_l = max(2, int(cw * 0.25))
                mid_r = max(mid_l + 1, int(cw * 0.75))
                neck_idx = mid_l + int(np.argmin(col_hs_w[mid_l:mid_r]))
                neck_h = int(col_hs_w[neck_idx])
                l_peak = int(np.max(col_hs_w[:neck_idx])) if neck_idx > 0 else 0
                r_peak = int(np.max(col_hs_w[neck_idx + 1:])) if neck_idx + 1 < cw else 0
                has_saddle = (
                    l_peak >= max(4, int(sp * 0.42))
                    and r_peak >= max(4, int(sp * 0.42))
                    and neck_h <= int(min(l_peak, r_peak) * 0.82)
                )
                if cw >= sp * 2.35 or has_saddle:
                    sub_w_r = min(cw - 4, max(6, int(round(sp * 1.32))))
                    sub_x0_r = nx + cw - sub_w_r
                    sub_patch_r = nh_mask[ny:ny + ch, sub_x0_r:nx + cw]
                    sy_r, sx_r = np.where(sub_patch_r > 0)
                    if len(sx_r) > 0:
                        sub_ch_r = int(sy_r.max() - sy_r.min() + 1)
                        sub_ar_r = float(len(sx_r)) / max(1.0, sp * sp)
                        sub_cx_r = float(sub_x0_r) + float(np.mean(sx_r))
                        sub_cy_r = float(ny) + float(np.mean(sy_r))
                        if (sp * 0.36 <= sub_ch_r <= sp * 1.65) and sub_ar_r >= 0.55 and sub_cx_r >= header_skip:
                            candidates.append(
                                _RawNote(
                                    x=xl + sub_cx_r,
                                    y=sub_cy_r,
                                    width=float(sub_w_r),
                                    area_ratio=max(0.95, sub_ar_r),
                                    is_grace=False,
                                    is_hollow=False,
                                )
                            )
                    sub_w_l = min(sub_x0_r - nx + 2, max(6, int(round(sp * 1.32))))
                    if sub_w_l >= int(round(sp * 0.75)):
                        sub_patch_l = nh_mask[ny:ny + ch, nx:nx + sub_w_l]
                        sy_l, sx_l = np.where(sub_patch_l > 0)
                        if len(sx_l) > 0:
                            sub_ch_l = int(sy_l.max() - sy_l.min() + 1)
                            sub_ar_l = float(len(sx_l)) / max(1.0, sp * sp)
                            sub_cx_l = float(nx) + float(np.mean(sx_l))
                            sub_cy_l = float(ny) + float(np.mean(sy_l))
                            if (sp * 0.36 <= sub_ch_l <= sp * 1.65) and sub_ar_l >= 0.52 and sub_cx_l >= header_skip:
                                candidates.append(
                                    _RawNote(
                                        x=xl + sub_cx_l,
                                        y=sub_cy_l,
                                        width=float(sub_w_l),
                                        area_ratio=max(0.55, sub_ar_l),
                                        is_grace=True,
                                        is_hollow=False,
                                    )
                                )
                    continue

            if not (
                (sp * 0.36 <= ch <= sp * 1.58)
                and (sp * 0.42 <= cw <= sp * 2.25)
                and (0.50 <= aspect <= 2.30)
                and (area_ratio >= 0.20)
                and ((sy1 - sp * 2.5) <= cy <= (sy2 + sp * 2.5))
            ):
                continue

            sx0, sx1 = max(0, nx - 2), min(roi.shape[1], nx + cw + 2)
            # 排除全譜高垂直小節線在譜線交點處殘留的窄小方塊 (cw < 0.78*sp 且位於貫穿五線譜之直線上)
            if cw < sp * 0.78:
                iy_s1, iy_s2 = max(0, int(round(sy1))), min(roi.shape[0], int(round(sy2)) + 1)
                if iy_s2 > iy_s1 + 6 and sx1 > sx0:
                    bar_span = int(np.max(np.sum(v_stems[iy_s1:iy_s2, sx0:sx1] > 0, axis=0)))
                    if bar_span >= int((iy_s2 - iy_s1) * 0.78):
                        continue

            # 排除獨立臨時升記號 ♯（四象限同時貫穿上下）與還原記號 ♮（左上與右下延伸）被誤認為實心主音符（如 M51 #G4 誤多出 F4）
            if cw <= sp * 1.65:
                acc_x0 = max(0, nx - 3)
                acc_x1 = min(roi.shape[1], nx + cw + 3)
                acc_mid_x = (acc_x0 + acc_x1) // 2
                acc_yu0 = max(0, int(round(cy - sp * 1.45)))
                acc_yu1 = max(acc_yu0 + 1, int(round(cy - sp * 0.55)))
                acc_yd0 = min(roi.shape[0] - 1, int(round(cy + sp * 0.55)))
                acc_yd1 = min(roi.shape[0], int(round(cy + sp * 1.35)))
                if acc_mid_x > acc_x0 and acc_x1 > acc_mid_x and acc_yu1 > acc_yu0 and acc_yd1 > acc_yd0:
                    a_ul = float(np.mean(np.any(vk_acc[acc_yu0:acc_yu1, acc_x0:acc_mid_x] > 0, axis=1)))
                    a_ur = float(np.mean(np.any(vk_acc[acc_yu0:acc_yu1, acc_mid_x:acc_x1] > 0, axis=1)))
                    a_dl = float(np.mean(np.any(vk_acc[acc_yd0:acc_yd1, acc_x0:acc_mid_x] > 0, axis=1)))
                    a_dr = float(np.mean(np.any(vk_acc[acc_yd0:acc_yd1, acc_mid_x:acc_x1] > 0, axis=1)))
                    # 同步檢測小型臨時升記號（如 M45 倚音前的縮小版 #，上下延伸約 ±0.88*sp）
                    sm_yu0 = max(0, int(round(cy - sp * 0.92)))
                    sm_yu1 = max(sm_yu0 + 1, int(round(cy - sp * 0.34)))
                    sm_yd0 = min(roi.shape[0] - 1, int(round(cy + sp * 0.34)))
                    sm_yd1 = min(roi.shape[0], int(round(cy + sp * 0.92)))
                    s_ul = float(np.mean(np.any(vk_acc[sm_yu0:sm_yu1, acc_x0:acc_mid_x] > 0, axis=1))) if sm_yu1 > sm_yu0 else 0.0
                    s_ur = float(np.mean(np.any(vk_acc[sm_yu0:sm_yu1, acc_mid_x:acc_x1] > 0, axis=1))) if sm_yu1 > sm_yu0 else 0.0
                    s_dl = float(np.mean(np.any(vk_acc[sm_yd0:sm_yd1, acc_x0:acc_mid_x] > 0, axis=1))) if sm_yd1 > sm_yd0 else 0.0
                    s_dr = float(np.mean(np.any(vk_acc[sm_yd0:sm_yd1, acc_mid_x:acc_x1] > 0, axis=1))) if sm_yd1 > sm_yd0 else 0.0
                    if (
                        (a_ul >= 0.48 and a_ur >= 0.48 and a_dl >= 0.48 and a_dr >= 0.48)
                        or (cw <= sp * 1.30 and s_ul >= 0.52 and s_ur >= 0.52 and s_dl >= 0.50 and s_dr >= 0.50)
                        or (
                            a_ul >= 0.72 and a_ur >= 0.72
                            and (a_dl >= 0.15 or a_dr >= 0.15)
                            and area_ratio >= 0.62
                            and sp * 0.95 <= cw <= sp * 1.45
                        )
                        or (a_ul >= 0.62 and a_dr >= 0.60 and a_ur <= 0.42 and a_dl <= 0.42)
                    ):
                        continue

            sy0, sy1_p = max(0, ny - int(sp * 0.50)), min(roi.shape[0], ny + ch + int(sp * 0.50))
            stem_cols = np.where(np.sum(pure_stems_no_bars[sy0:sy1_p, sx0:sx1] > 0, axis=0) >= 3)[0]
            min_stem_gap = max(5, int(round(sp * 0.65)))
            n_stems = (1 + int(np.sum(np.diff(stem_cols) > min_stem_gap))) if len(stem_cols) > 0 else 0

            # 僅在「雙連槓細扁殘條 (n_stems >= 2 且 area_ratio < 0.62)」或「位於譜頂/譜底附近且同側確實串接 >=2 顆較小倚音符頭的雙倚音短連槓 (如 M4/M28，即使較高倚音符幹較短使 n_stems == 1)」時視為 horiz_bar，絕不誤刪帶彎勾八分音符 (如 M17 G5) 或緊鄰十六分音符 (如 M5 D5)！
            if n_stems >= 1:
                sub_grace_below = [
                    bx_o for bx_o, by_o, ba_o in all_blob_pts
                    if (nx - sp * 0.65) <= bx_o <= (nx + cw + sp * 0.65)
                    and sp * 0.65 <= (by_o - cy) <= sp * 3.15
                    and ba_o < area * 0.95
                ]
                sub_grace_above = [
                    bx_o for bx_o, by_o, ba_o in all_blob_pts
                    if (nx - sp * 0.65) <= bx_o <= (nx + cw + sp * 0.65)
                    and sp * 0.65 <= (cy - by_o) <= sp * 3.15
                    and ba_o < area * 0.95
                ]
                is_grace_minibeam = (
                    (cy < sy1 + sp * 0.45 and len(sub_grace_below) >= 2 and (max(sub_grace_below) - min(sub_grace_below)) >= sp * 0.42)
                    or (cy > sy2 - sp * 0.45 and len(sub_grace_above) >= 2 and (max(sub_grace_above) - min(sub_grace_above)) >= sp * 0.42)
                )
                if (n_stems >= 2 and area_ratio < 0.62) or is_grace_minibeam:
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

            restored_cy = False
            is_grace_cand = (area_ratio < 0.72) or (cw < sp * 1.06 and n_stems >= 1)
            # 若符頭帶有左側標準長下垂符幹（倚音符幹一律朝上，絕無長下垂單幹），且寬度充足或連通至下方八分/十六分主連槓（如 M21 E5、M47 行尾被譜線削薄的 B4），必為一般主音符！
            if is_grace_cand:
                dn_x0 = max(0, nx - max(3, int(round(sp * 0.58))))
                dn_x1 = min(roi.shape[1], nx + int(cw * 0.55) + 2)
                dn_check = pure_stems_no_bars[min(roi.shape[0] - 1, ny + ch - 2):min(roi.shape[0], ny + ch + int(sp * 2.8)), dn_x0:dn_x1]
                up_check = pure_stems_no_bars[max(0, ny - int(sp * 2.2)):max(1, ny + 2), max(0, nx - 2):min(roi.shape[1], nx + cw + 2)]
                dn_len = int(np.max(np.sum(dn_check > 0, axis=0))) if dn_check.size > 0 else 0
                up_len = int(np.max(np.sum(up_check > 0, axis=0))) if up_check.size > 0 else 0
                conn_lower_beam = any(
                    hb.n_stems >= 2 and (hb.x1 - sp * 0.85) <= abs_cx <= (hb.x2 + sp * 0.95) and hb.y_center > cy + sp * 0.85
                    for hb in horiz_bars
                )
                if dn_len >= int(sp * 1.45) and up_len < int(sp * 0.55) and (
                    (cw >= sp * 1.05 and area_ratio >= 0.56)
                    or (conn_lower_beam and cw >= sp * 0.55)
                    or (dn_len >= int(sp * 1.50) and cw >= sp * 0.62 and area_ratio >= 0.24)
                ):
                    is_grace_cand = False
                    area_ratio = max(area_ratio, 0.95)
                    # 若騎線符頭下半月牙被 nh_mask 削去導致 ch 偏扁，由 vert_keep 上下極值中點還原完整橢圓垂直中心（確保 M47 B4 100% 對齊第 3 線！）
                    if ch < sp * 0.65:
                        rc_y0 = max(0, ny - 2)
                        rc_y1 = min(roi.shape[0], ny + ch + max(4, int(round(sp * 0.55))))
                        vk_sub = cv2.subtract(vert_keep[rc_y0:rc_y1, max(0, nx - 2):min(roi.shape[1], nx + cw + 1)], pure_stems[rc_y0:rc_y1, max(0, nx - 2):min(roi.shape[1], nx + cw + 1)])
                        rc_ys, _ = np.where(vk_sub > 0)
                        if len(rc_ys) >= 4:
                            cy = float(rc_y0) + 0.5 * (float(rc_ys.min()) + float(rc_ys.max()))
                            ch = max(ch, int(rc_ys.max() - rc_ys.min() + 1))
                            restored_cy = True
            if is_grace_cand and n_stems == 0:
                # 若該小符頭正上方/正下方已有雙倚音短連槓（如 M4/M28 較高的 (D5) 倚音，其符幹較短 < 1.25*sp），仍須保留！
                under_grace_beam = any(
                    (hb.x1 - sp * 0.55) <= abs_cx <= (hb.x2 + sp * 0.55)
                    and sp * 0.65 <= abs(hb.y_center - cy) <= sp * 3.15
                    and hb.width <= sp * 2.8
                    for hb in horiz_bars
                )
                if not under_grace_beam:
                    continue

            # 排除位於五線譜上方的重音記號 > / ^（左側開口中空、右側收斂成尖端，如 M21 E5 上方的 > 誤判為 A5）
            if cy < sy1 - sp * 0.25 and cw >= 6 and ch >= 6:
                raw_sub = roi[ny:ny + ch, nx:nx + cw] > 0
                l_w = max(2, cw // 3)
                r_x0 = max(l_w, cw - max(2, cw // 3))
                l_part = raw_sub[:, :l_w]
                r_part = raw_sub[:, r_x0:]
                mid_r0 = max(1, ch // 2 - 1)
                mid_r1 = min(ch - 1, ch // 2 + 2)
                l_top_has = bool(np.any(l_part[:mid_r0, :]))
                l_bot_has = bool(np.any(l_part[mid_r1:, :]))
                l_mid_fill = float(np.mean(l_part[mid_r0:mid_r1, :])) if mid_r1 > mid_r0 else 1.0
                r_ys = np.where(np.any(r_part, axis=1))[0]
                r_span = int(r_ys[-1] - r_ys[0] + 1) if len(r_ys) > 0 else 0
                if l_top_has and l_bot_has and l_mid_fill < 0.25 and r_span <= int(ch * 0.68):
                    continue

            # 位於五線譜上下界之外的真實實心音符必有「直接接觸自身邊界」的長符幹（徹底杜絕重音記號 > 跨越空白間隙偷借下方 E5 的符幹！）
            if cy > sy2 + sp * 0.55:
                sx_low0, sx_low1 = max(0, nx - 1), min(roi.shape[1], nx + cw + 1)
                touch_low = pure_stems_no_bars[max(0, ny - 4):min(roi.shape[0], ny + ch + 3), sx_low0:sx_low1]
                up_low = pure_stems_no_bars[max(0, ny - int(sp * 2.5)):min(roi.shape[0], ny + 3), sx_low0:sx_low1]
                if (
                    touch_low.size == 0 or int(np.sum(touch_low > 0)) < 2
                    or up_low.size == 0 or int(np.max(np.sum(up_low > 0, axis=0))) < max(7, int(sp * 1.05))
                ):
                    continue
            elif cy < sy1 - sp * 0.55 and n_stems == 0:
                sx_hi0, sx_hi1 = max(0, nx - 1), min(roi.shape[1], nx + cw + 1)
                touch_hi = pure_stems_no_bars[max(0, ny - 3):min(roi.shape[0], ny + ch + 4), sx_hi0:sx_hi1]
                sy_ext0 = max(0, ny - int(sp * 1.8))
                sy_ext1 = min(roi.shape[0], ny + ch + int(sp * 1.8))
                ext_stem = int(np.max(np.sum(pure_stems_no_bars[sy_ext0:sy_ext1, sx_hi0:sx_hi1] > 0, axis=0))) if sx_hi1 > sx_hi0 else 0
                if touch_hi.size == 0 or int(np.sum(touch_hi > 0)) < 2 or ext_stem < max(7, int(sp * 1.05)):
                    continue

            # 以符頭核心厚實直欄（排除邊緣 1~2px 符幹與譜線十字交點殘刺）計算精準垂直中心 core_cy，避免 C5 被交點往下拉成 B4！
            if not restored_cy:
                blob_p = nh_mask[ny:ny + ch, nx:nx + cw] > 0
                col_hs = np.sum(blob_p, axis=0)
                max_ch = int(np.max(col_hs)) if col_hs.size > 0 else 0
                if max_ch >= 4:
                    core_cols = col_hs >= max(2, int(round(max_ch * 0.52)))
                    cy_ys, _ = np.where(blob_p[:, core_cols])
                    if len(cy_ys) > 0:
                        cy = float(ny) + float(np.mean(cy_ys))

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

        # 4b. 補救被主符頭開運算或厚譜線侵蝕之小型單倚音（如 M45 譜線上的帶斜線倚音）：閉合譜線細縫還原扁小符頭，結合右側向上細符幹與頂端斜線判別
        grace_rejoined = cv2.morphologyEx(
            no_staff_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, max_thin)))
        )
        grace_clean = cv2.subtract(
            cv2.subtract(grace_rejoined, cv2.dilate(beams_mask, np.ones((2, 2), np.uint8))),
            pure_stems,
        )
        grace_src = cv2.bitwise_or(roi_clean, grace_clean)
        sm_w = max(2, int(round(sp * 0.22)))
        sm_h = max(2, int(round(sp * 0.20)))
        sm_mask = cv2.morphologyEx(grace_src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (sm_w, sm_h)))
        sm_num, _, sm_stats, sm_cent = cv2.connectedComponentsWithStats(sm_mask)
        sm_order = sorted(range(1, sm_num), key=lambda idx: int(sm_stats[idx, cv2.CC_STAT_TOP]))
        for slbl in sm_order:
            snx = int(sm_stats[slbl, cv2.CC_STAT_LEFT])
            sny = int(sm_stats[slbl, cv2.CC_STAT_TOP])
            scw = int(sm_stats[slbl, cv2.CC_STAT_WIDTH])
            sch = int(sm_stats[slbl, cv2.CC_STAT_HEIGHT])
            sarea = int(sm_stats[slbl, cv2.CC_STAT_AREA])
            scx, scy = float(sm_cent[slbl][0]), float(sm_cent[slbl][1])
            if scx < header_skip:
                continue
            s_abs_cx = xl + scx
            if any(fx0 <= s_abs_cx <= fx1 and fy0 <= scy <= fy1 for fx0, fy0, fx1, fy1, _ in fermata_zones):
                continue
            s_ar = sarea / max(1.0, sp * sp)
            s_asp = scw / max(1.0, float(sch))
            if not (
                (sp * 0.19 <= sch <= sp * 0.75)
                and (sp * 0.30 <= scw <= sp * 0.98)
                and (0.70 <= s_asp <= 2.60)
                and (0.08 <= s_ar < 0.55)
                and ((sy1 - sp * 1.8) <= scy <= (sy2 + sp * 1.8))
            ):
                continue
            if any(abs(s_abs_cx - c.x) <= sp * 0.90 and abs(scy - c.y) <= sp * 2.20 for c in candidates):
                continue
            if len(bar_cols_roi) > 0 and np.any(np.abs(bar_cols_roi - scx) <= sp * 0.80):
                continue
            sx_r0 = max(0, int(round(scx)))
            sx_r1 = min(roi.shape[1], snx + scw + max(3, int(round(sp * 0.45))))
            sx_l0 = max(0, snx - 2)
            sx_l1 = max(sx_l0 + 1, int(round(scx)))
            sy_u0 = max(0, sny - int(round(sp * 2.3)))
            sy_u1 = max(sy_u0 + 1, sny - int(round(sp * 0.30)))
            sy_d0 = min(roi.shape[0] - 1, sny + sch)
            sy_d1 = min(roi.shape[0], sny + sch + int(round(sp * 1.15)))
            up_r_stem = int(np.max(np.sum(pure_stems_no_bars[sy_u0:sny + 2, sx_r0:sx_r1] > 0, axis=0))) if sx_r1 > sx_r0 else 0
            up_r_cov = float(np.mean(np.any(vk_acc[sy_u0:sy_u1, sx_r0:sx_r1] > 0, axis=1))) if (sx_r1 > sx_r0 and sy_u1 > sy_u0) else 0.0
            up_l_cov = float(np.mean(np.any(vk_acc[sy_u0:sy_u1, sx_l0:sx_l1] > 0, axis=1))) if (sx_l1 > sx_l0 and sy_u1 > sy_u0) else 0.0
            dn_all = int(np.max(np.sum(pure_stems_no_bars[sy_d0:sy_d1, sx_l0:sx_r1] > 0, axis=0))) if (sy_d1 > sy_d0 and sx_r1 > sx_l0) else 0
            dn_l_cov = float(np.mean(np.any(vk_acc[sy_d0:min(roi.shape[0], sny + sch + int(round(sp * 0.75))), sx_l0:sx_l1] > 0, axis=1))) if (sx_l1 > sx_l0 and sy_d1 > sy_d0) else 0.0
            has_grace_up_stem = (up_r_stem >= max(6, int(round(sp * 1.05)))) or (up_r_cov >= 0.55)
            if (not has_grace_up_stem) or up_l_cov >= 0.35 or dn_l_cov >= 0.45 or dn_all >= max(5, int(round(sp * 0.65))):
                continue
            flag_y0 = max(0, sny - int(round(sp * 2.4)))
            flag_y1 = max(flag_y0 + 1, sny - int(round(sp * 0.55)))
            flag_x0 = max(0, snx - int(round(sp * 0.25)))
            flag_x1 = min(roi.shape[1], snx + scw + int(round(sp * 0.95)))
            flag_patch = cv2.subtract(roi_clean[flag_y0:flag_y1, flag_x0:flag_x1], beams_mask[flag_y0:flag_y1, flag_x0:flag_x1])
            if int(np.sum(flag_patch > 0)) < max(4, int(round(sp * 0.35))):
                continue
            candidates.append(
                _RawNote(
                    x=s_abs_cx,
                    y=scy,
                    width=float(scw),
                    area_ratio=s_ar,
                    is_grace=True,
                    is_hollow=False,
                )
            )

        # 5. 過濾單根符幹尾端的 16 分音符短勾槓與單顆裝飾音上方的斜線小旗標
        main_cands = [c for c in candidates if c.area_ratio >= 0.90]
        grace_cands = [c for c in candidates if c.area_ratio < 0.90]
        filtered_cands: List[_RawNote] = []
        for c in candidates:
            if c.area_ratio < 0.90:
                if c.area_ratio < 0.48 and any(
                    abs(c.x - m.x) <= sp * 0.85 and abs(c.y - m.y) < sp * 1.55
                    for m in main_cands
                ):
                    continue
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

        # 6. 空心符頭（二分音符 / 全音符）偵測：僅移除純水平譜線與粗連槓（排除 16 分雙連槓間的白縫如 M20/M22），而不減去 pure_stems，確保二分音符側面符幹處圓環 100% 封閉不斷裂！
        intact_for_hollow = cv2.subtract(roi, cv2.subtract(h_lines_only, vert_keep))
        intact_for_hollow = cv2.subtract(intact_for_hollow, cv2.dilate(beams_mask, np.ones((2, 2), np.uint8)))
        intact_for_hollow = cv2.morphologyEx(
            intact_for_hollow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        contours, hierarchy = cv2.findContours(intact_for_hollow, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            hier = hierarchy[0]
            for c_idx, h_info in enumerate(hier):
                parent_idx = h_info[3]
                if parent_idx < 0:
                    continue
                hx, hy, hw, hh = cv2.boundingRect(contours[c_idx])
                hole_area = cv2.contourArea(contours[c_idx])
                fill_ratio = hole_area / max(1.0, float(hw * hh))
                hcx = hx + hw / 2.0
                hcy = hy + hh / 2.0
                if hcx < header_skip:
                    continue
                abs_x = xl + hcx
                if any(fx0 <= abs_x <= fx1 and fy0 <= hcy <= fy1 for fx0, fy0, fx1, fy1, _ in fermata_zones):
                    continue
                # 真實空心二分/全音符內孔必為橫向橢圓 (hw >= hh * 1.02)，可天然排除雙小節線窄縫與直立降記號 ♭ 圓肚！
                if (
                    (sp * 0.34 <= hw <= sp * 1.18)
                    and (sp * 0.24 <= hh <= sp * 0.85)
                    and (float(hw) >= float(hh) * 1.02)
                    and (fill_ratio >= 0.35)
                    and (sp * sp * 0.11 <= hole_area <= sp * sp * 0.72)
                    and ((sy1 - sp * 2.0) <= hcy <= (sy2 + sp * 2.0))
                ):
                    # 確認內孔上下左右 4 側緊鄰皆有實質符頭黑邊包覆，且左右外側非連續寬黑塊
                    t_rim = intact_for_hollow[max(0, hy - 4):max(1, hy), hx:hx + hw]
                    b_rim = intact_for_hollow[min(roi.shape[0] - 1, hy + hh):min(roi.shape[0], hy + hh + 4), hx:hx + hw]
                    l_rim = intact_for_hollow[hy:hy + hh, max(0, hx - 4):max(1, hx)]
                    r_rim = intact_for_hollow[hy:hy + hh, min(roi.shape[1] - 1, hx + hw):min(roi.shape[1], hx + hw + 4)]
                    if (
                        t_rim.size == 0 or b_rim.size == 0 or l_rim.size == 0 or r_rim.size == 0
                        or float(np.mean(t_rim > 0)) < 0.30
                        or float(np.mean(b_rim > 0)) < 0.30
                        or float(np.mean(l_rim > 0)) < 0.30
                        or float(np.mean(r_rim > 0)) < 0.30
                    ):
                        continue

                    if all(
                        not (
                            abs(abs_x - c.x) <= sp * 0.70
                            and c.area_ratio >= 0.90
                            and (abs(hcy - c.y) <= sp * 0.85 or hcy < sy1 - sp * 0.35 or hcy > sy2 + sp * 0.35)
                        )
                        for c in candidates if not c.is_grace
                    ):
                        # 位於五線譜上下界之外的空心音符（如 A5/C6/C4）必有直接接觸自身邊界的長符幹或加線，徹底排除 M21「圓滑線 + 重音 >」圍成的封閉三角孔誤判為 A5！
                        if hcy < sy1 - sp * 0.45 or hcy > sy2 + sp * 0.45:
                            ht_x0 = max(0, int(hx - sp * 0.45))
                            ht_x1 = min(roi.shape[1], int(hx + hw + sp * 0.45))
                            ht_touch = pure_stems_no_bars[max(0, hy - 4):min(roi.shape[0], hy + hh + 5), ht_x0:ht_x1]
                            ht_ext = pure_stems_no_bars[max(0, int(hcy - sp * 2.6)):min(roi.shape[0], int(hcy + sp * 2.6)), ht_x0:ht_x1]
                            has_real_h_stem = (
                                ht_touch.size > 0 and int(np.sum(ht_touch > 0)) >= 2
                                and ht_ext.size > 0 and int(np.max(np.sum(ht_ext > 0, axis=0))) >= int(sp * 1.2)
                            )
                            # 若無符幹（如譜外全音符），則左右兩側必有水平加線延伸
                            ll_y0, ll_y1 = max(0, int(round(hcy)) - 2), min(roi.shape[0], int(round(hcy)) + 3)
                            ll_l = h_lines_only[ll_y0:ll_y1, max(0, int(hx - sp * 0.55)):max(1, hx)]
                            ll_r = h_lines_only[ll_y0:ll_y1, min(roi.shape[1] - 1, hx + hw):min(roi.shape[1], int(hx + hw + sp * 0.55))]
                            has_ledger = (ll_l.size > 0 and np.any(ll_l > 0)) and (ll_r.size > 0 and np.any(ll_r > 0))
                            if not (has_real_h_stem or has_ledger):
                                continue
                        # 排除夾在兩條垂直符幹之間的白縫（如雙十六分連槓間距）誤判為空心音符：真實空心音符最多只有單側一根符幹！
                        hl_band = pure_stems[max(0, int(hcy - sp * 2.2)):min(roi.shape[0], int(hcy + sp * 2.2)), max(0, int(hx - sp * 0.45)):max(1, int(hx + 2))]
                        hr_band = pure_stems[max(0, int(hcy - sp * 2.2)):min(roi.shape[0], int(hcy + sp * 2.2)), min(roi.shape[1] - 1, int(hx + hw - 2)):min(roi.shape[1], int(hx + hw + sp * 0.45))]
                        hl_st = int(np.max(np.sum(hl_band > 0, axis=0))) if hl_band.size > 0 else 0
                        hr_st = int(np.max(np.sum(hr_band > 0, axis=0))) if hr_band.size > 0 else 0
                        if hl_st >= int(sp * 0.9) and hr_st >= int(sp * 0.9):
                            continue

                        # 排除降記號 ♭ ("b" 形) 或升降記號 ♯/♮ 內部小方孔誤判為空心二分音符！
                        if hole_area < sp * sp * 0.36:
                            l_up = pure_stems[max(0, int(hcy - sp * 2.2)):max(1, int(hcy - sp * 0.3)), max(0, int(hx - sp * 0.45)):min(roi.shape[1], int(hcx))]
                            r_up = pure_stems[max(0, int(hcy - sp * 2.2)):max(1, int(hcy - sp * 0.3)), max(0, int(hcx)):min(roi.shape[1], int(hx + hw + sp * 0.45))]
                            all_dn = pure_stems[min(roi.shape[0] - 1, int(hcy + sp * 0.5)):min(roi.shape[0], int(hcy + sp * 2.0)), max(0, int(hx - sp * 0.4)):min(roi.shape[1], int(hx + hw + sp * 0.4))]
                            lu_h = int(np.max(np.sum(l_up > 0, axis=0))) if l_up.size > 0 else 0
                            ru_h = int(np.max(np.sum(r_up > 0, axis=0))) if r_up.size > 0 else 0
                            dn_h = int(np.max(np.sum(all_dn > 0, axis=0))) if all_dn.size > 0 else 0
                            if lu_h >= int(sp * 0.9) and ru_h < int(sp * 0.5) and dn_h < int(sp * 0.5):
                                continue
                            if hole_area < sp * sp * 0.28 and lu_h >= int(sp * 0.7) and dn_h >= int(sp * 0.7):
                                continue

                        # 清除空心符頭上下邊緣在 nh_mask 中殘留的碎小假音符邊框碎片
                        candidates = [
                            c for c in candidates
                            if not (abs(c.x - abs_x) <= sp * 0.75 and abs(c.y - hcy) <= sp * 0.85 and c.area_ratio < 0.98)
                        ]

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
                                width=float(hw + 6),
                                area_ratio=1.15,
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
                # 實心音符與譜外假空心孔垂直同軸時，一律保留實心音符（絕不讓 [圓滑線+重音>] 空心孔覆寫下方 E5！）
                if prev.is_hollow and not cand.is_hollow:
                    merged[-1] = cand
                    continue
                if not prev.is_hollow and cand.is_hollow:
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

        # 7b. 四象限筆畫拓樸臨時記號辨識（♯ 升半音 +1、♭ 降半音 -1、♮ 還原記號），並自動清除被誤認為小裝飾音的記號殘片
        vk_full = np.zeros_like(strip_bin)
        vk_full[:, xl:xr] = vk_acc
        main_notes_list = [n for n in merged if not n.is_grace]
        accidental_kill_zones: List[Tuple[float, float]] = []

        for m_idx, mn in enumerate(main_notes_list):
            ax_right = int(round(mn.x - max(mn.width * 0.56, sp * 0.86)))
            ax_left = max(int(round(header_end_x)), int(round(mn.x - sp * 2.22)))
            if m_idx > 0:
                ax_left = max(ax_left, int(round(main_notes_list[m_idx - 1].x + sp * 0.72)))
            if len(bar_cols_roi) > 0:
                bars_in_span = bar_cols_roi[(xl + bar_cols_roi >= ax_left - 2) & (xl + bar_cols_roi <= ax_right)]
                if len(bars_in_span) > 0:
                    ax_left = max(ax_left, int(xl + bars_in_span[-1] + 3))
            if ax_right - ax_left < int(sp * 0.50):
                continue

            # 若該區間上方存在雙倚音短連槓（如 M4 的 (C5)-(D5) 雙裝飾音），必為倚音群而非臨時升降號！
            has_upper_grace_beam = any(
                hb.y_center < mn.y - sp * 0.75
                and hb.width <= sp * 2.8
                and not (hb.x2 < ax_left - 2 or hb.x1 > ax_right + 2)
                for hb in horiz_bars
            )
            if has_upper_grace_beam:
                continue
            graces_in_z = [g for g in merged if g.is_grace and (ax_left - 2) <= g.x <= (ax_right + 2)]
            if len(graces_in_z) >= 2:
                continue

            ym0 = max(0, int(round(mn.y - sp * 0.50)))
            ym1 = min(strip_bin.shape[0], int(round(mn.y + sp * 0.50)))
            mid_p = vk_full[ym0:ym1, ax_left:ax_right] > 0
            if int(np.sum(mid_p)) < int(sp * sp * 0.24):
                continue
            cols_has = np.where(np.sum(mid_p, axis=0) >= 2)[0]
            if len(cols_has) < int(sp * 0.40):
                continue
            bx0 = ax_left + int(cols_has[0])
            bx1 = ax_left + int(cols_has[-1]) + 1
            bw_acc = bx1 - bx0
            if bw_acc < int(sp * 0.42) or bw_acc > int(round(sp * 1.48)):
                continue

            mid_x = (bx0 + bx1) // 2
            yu0 = max(0, int(round(mn.y - sp * 1.45)))
            yu1 = max(yu0 + 1, int(round(mn.y - sp * 0.55)))
            yd0 = min(strip_bin.shape[0] - 1, int(round(mn.y + sp * 0.55)))
            yd1 = min(strip_bin.shape[0], int(round(mn.y + sp * 1.35)))
            yd_deep0 = min(strip_bin.shape[0] - 1, int(round(mn.y + sp * 0.85)))
            yd_deep1 = min(strip_bin.shape[0], int(round(mn.y + sp * 1.48)))

            ul = float(np.mean(np.any(vk_full[yu0:yu1, bx0:mid_x] > 0, axis=1))) if mid_x > bx0 else 0.0
            ur = float(np.mean(np.any(vk_full[yu0:yu1, mid_x:bx1] > 0, axis=1))) if bx1 > mid_x else 0.0
            dl = float(np.mean(np.any(vk_full[yd0:yd1, bx0:mid_x] > 0, axis=1))) if mid_x > bx0 else 0.0
            dr = float(np.mean(np.any(vk_full[yd0:yd1, mid_x:bx1] > 0, axis=1))) if bx1 > mid_x else 0.0
            dr_deep = float(np.mean(np.any(vk_full[yd_deep0:yd_deep1, mid_x:bx1] > 0, axis=1))) if (yd_deep1 > yd_deep0 and bx1 > mid_x) else 0.0

            if ul >= 0.52 and ur >= 0.52 and dl >= 0.48 and dr >= 0.48:
                # ♯ 升記號：左右兩根直幹同時貫穿上下四象限
                mn.accidental = 1
                accidental_kill_zones.append((float(bx0 - 3), float(bx1 + 3)))
            elif ul >= 0.62 and ur <= 0.42 and dl <= 0.42 and dr >= 0.60 and dr_deep >= 0.50:
                # ♮ 還原記號：左幹僅向上、右幹僅向下延伸
                mn.accidental = 0
                mn.has_natural = True
                accidental_kill_zones.append((float(bx0 - 3), float(bx1 + 3)))
            elif ul >= 0.65 and ur <= 0.38 and dl <= 0.25 and dr_deep <= 0.25:
                # ♭ 降記號：左幹向上、右下圓肚止於符頭下緣
                mn.accidental = -1
                accidental_kill_zones.append((float(bx0 - 3), float(bx1 + 3)))

        if accidental_kill_zones:
            merged = [
                n for n in merged
                if not (
                    (n.is_grace or n.is_hollow)
                    and any(z0 <= n.x <= z1 for z0, z1 in accidental_kill_zones)
                )
            ]

        # 7b-2. 為獨立單顆裝飾音偵測緊鄰左側之縮小版臨時升記號（如 M45 開頭的 #F5 倚音）
        grace_kill_zones: List[Tuple[float, float]] = []
        for g_idx, gn in enumerate(merged):
            if not gn.is_grace:
                continue
            gx_r = int(round(gn.x - max(gn.width * 0.50, sp * 0.38)))
            gx_l = max(int(round(header_end_x)), int(round(gn.x - sp * 1.65)))
            if g_idx > 0 and not merged[g_idx - 1].is_grace:
                gx_l = max(gx_l, int(round(merged[g_idx - 1].x + sp * 0.65)))
            if len(bar_cols_roi) > 0:
                gbars = bar_cols_roi[(xl + bar_cols_roi >= gx_l - 2) & (xl + bar_cols_roi <= gx_r)]
                if len(gbars) > 0:
                    gx_l = max(gx_l, int(xl + gbars[-1] + 3))
            if gx_r - gx_l < int(sp * 0.36):
                continue
            gym0 = max(0, int(round(gn.y - sp * 0.42)))
            gym1 = min(strip_bin.shape[0], int(round(gn.y + sp * 0.42)))
            gmid = vk_full[gym0:gym1, gx_l:gx_r] > 0
            if int(np.sum(gmid)) < int(sp * sp * 0.14):
                continue
            gcols = np.where(np.sum(gmid, axis=0) >= 2)[0]
            if len(gcols) < max(3, int(sp * 0.28)):
                continue
            gbx0 = gx_l + int(gcols[0])
            gbx1 = gx_l + int(gcols[-1]) + 1
            if (gbx1 - gbx0) > int(round(sp * 1.15)):
                continue
            gmid_x = (gbx0 + gbx1) // 2
            gyu0 = max(0, int(round(gn.y - sp * 0.95)))
            gyu1 = max(gyu0 + 1, int(round(gn.y - sp * 0.34)))
            gyd0 = min(strip_bin.shape[0] - 1, int(round(gn.y + sp * 0.34)))
            gyd1 = min(strip_bin.shape[0], int(round(gn.y + sp * 0.95)))
            gul = float(np.mean(np.any(vk_full[gyu0:gyu1, gbx0:gmid_x] > 0, axis=1))) if gmid_x > gbx0 else 0.0
            gur = float(np.mean(np.any(vk_full[gyu0:gyu1, gmid_x:gbx1] > 0, axis=1))) if gbx1 > gmid_x else 0.0
            gdl = float(np.mean(np.any(vk_full[gyd0:gyd1, gbx0:gmid_x] > 0, axis=1))) if gmid_x > gbx0 else 0.0
            gdr = float(np.mean(np.any(vk_full[gyd0:gyd1, gmid_x:gbx1] > 0, axis=1))) if gbx1 > gmid_x else 0.0
            if gul >= 0.48 and gur >= 0.48 and gdl >= 0.45 and gdr >= 0.45:
                gn.accidental = 1
                grace_kill_zones.append((float(gbx0 - 2), float(gbx1 + 2)))
        if grace_kill_zones:
            merged = [
                n for n in merged
                if not (n.is_grace and n.accidental == 0 and any(z0 <= n.x <= z1 for z0, z1 in grace_kill_zones))
            ]

        # 7c. 清除升降號殘片後，將連接主連槓之偏小十六分主音符（如 M5 的 D5）還原為一般主音符！
        for n in merged:
            if n.is_grace and n.area_ratio >= 0.52 and n.width >= sp * 0.80:
                rx_c = int(round(n.x - xl))
                sx0_g = max(0, rx_c - int(sp * 0.65))
                sx1_g = min(roi.shape[1], rx_c + int(sp * 0.65))
                up_s = int(np.max(np.sum(pure_stems[max(0, int(n.y - sp * 2.5)):max(1, int(n.y - sp * 0.3)), sx0_g:sx1_g] > 0, axis=0))) if sx1_g > sx0_g else 0
                dn_s = int(np.max(np.sum(pure_stems[min(roi.shape[0] - 1, int(n.y + sp * 0.3)):min(roi.shape[0], int(n.y + sp * 2.8)), sx0_g:sx1_g] > 0, axis=0))) if sx1_g > sx0_g else 0
                has_lower_main_beam = any(
                    hb.n_stems >= 2 and (hb.x1 - sp * 0.6 <= n.x <= hb.x2 + sp * 0.6) and hb.y_center > n.y + sp * 0.9
                    for hb in horiz_bars
                )
                has_upper_main_beam = any(
                    hb.n_stems >= 2 and hb.width >= sp * 2.35 and (hb.x1 - sp * 0.6 <= n.x <= hb.x2 + sp * 0.6) and hb.y_center < n.y - sp * 0.9
                    for hb in horiz_bars
                )
                if (up_s < int(sp * 0.7) and dn_s >= int(sp * 1.1) and has_lower_main_beam) or (
                    dn_s < int(sp * 0.7) and up_s >= int(sp * 1.1) and has_upper_main_beam
                ):
                    n.is_grace = False

        # 8. 為每顆音符計算連槓數、附點與演奏記號（延長記號 𝄐 / 保持音 - / 跳音 .）
        for note in merged:
            if note.is_grace:
                note.beam_count = 2
                continue
            if not note.is_hollow:
                beam_ys = []
                covering_bars = []
                for bar in horiz_bars:
                    if bar.x1 - sp * 0.7 <= note.x <= bar.x2 + sp * 0.7:
                        if abs(bar.y_center - note.y) >= sp * 0.8:
                            beam_ys.append(bar.y_center)
                            covering_bars.append(bar)
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

                # 若雙十六分連槓因間距緊密在 beams_mask 中融合為單一厚橫槓 (局部厚度 >= 1.12*sp)，依符幹左右兩側局部剖面精準還原雙連槓數！
                if covering_bars:
                    rx_n = int(round(note.x - xl))
                    for side_dx in (-int(round(sp * 0.58)), int(round(sp * 0.58))):
                        sx_s = rx_n + side_dx
                        if sx_s < 2 or sx_s >= roi.shape[1] - 2:
                            continue
                        abs_sx = float(xl + sx_s)
                        side_bars = [b for b in covering_bars if (b.x1 - 2.0) <= abs_sx <= (b.x2 + 2.0)]
                        if not side_bars:
                            continue
                        yb0 = max(0, int(min(b.y_center - b.height * 0.65 for b in side_bars)) - 2)
                        yb1 = min(roi.shape[0], int(max(b.y_center + b.height * 0.65 for b in side_bars)) + 3)
                        if yb1 <= yb0 + 3:
                            continue
                        col_b = np.mean(beams_mask[yb0:yb1, sx_s - 2:sx_s + 3] > 0, axis=1) >= 0.5
                        side_cnt = 0
                        in_r = False
                        r_start = 0
                        for idx_r, v_r in enumerate(col_b):
                            if v_r and not in_r:
                                r_start = idx_r
                                in_r = True
                            elif not v_r and in_r:
                                r_h = idx_r - r_start
                                if r_h >= 2:
                                    side_cnt += 2 if r_h >= sp * 1.12 else 1
                                in_r = False
                        if in_r:
                            r_h = len(col_b) - r_start
                            if r_h >= 2:
                                side_cnt += 2 if r_h >= sp * 1.12 else 1
                        distinct_beams = max(distinct_beams, side_cnt)

                # 若尚未計入連槓或粗短勾，檢查獨立符幹末端右側是否有細弧形八分音符符尾勾 (Flag，無論朝上或朝下符幹，符尾勾必向符幹右側延伸！)
                if distinct_beams == 0:
                    rx_f = int(round(note.x - xl))
                    sx_f0 = max(0, rx_f - int(round(sp * 0.75)))
                    sx_f1 = min(roi.shape[1], rx_f + int(round(sp * 0.75)))
                    if sx_f1 > sx_f0:
                        dn_band = pure_stems[min(roi.shape[0] - 1, int(round(note.y + sp * 0.4))):min(roi.shape[0], int(round(note.y + sp * 3.8))), sx_f0:sx_f1]
                        up_band = pure_stems[max(0, int(round(note.y - sp * 3.8))):max(1, int(round(note.y - sp * 0.4))), sx_f0:sx_f1]
                        dn_col_sums = np.sum(dn_band > 0, axis=0) if dn_band.size > 0 else np.zeros(1)
                        up_col_sums = np.sum(up_band > 0, axis=0) if up_band.size > 0 else np.zeros(1)
                        if int(np.max(dn_col_sums)) >= int(sp * 1.6):
                            st_x = sx_f0 + int(np.argmax(dn_col_sums))
                            ys_st = np.where(pure_stems[int(round(note.y + sp * 0.4)):min(roi.shape[0], int(round(note.y + sp * 3.8))), st_x] > 0)[0]
                            if len(ys_st) > 0:
                                y_tip = int(round(note.y + sp * 0.4)) + int(ys_st[-1])
                                fy0 = max(int(round(note.y + sp * 0.9)), y_tip - int(round(sp * 1.65)))
                                fy1 = min(roi.shape[0], y_tip + 3)
                                fx0 = min(roi.shape[1], st_x + 3)
                                fx1 = min(roi.shape[1], st_x + int(round(sp * 1.15)))
                                if fy1 > fy0 and fx1 > fx0:
                                    f_patch = roi_clean[fy0:fy1, fx0:fx1] > 0
                                    if int(np.sum(f_patch)) >= max(6, int(sp * sp * 0.14)) and int(np.sum(np.any(f_patch, axis=0))) >= max(3, int(sp * 0.30)):
                                        distinct_beams = 1
                        elif int(np.max(up_col_sums)) >= int(sp * 1.6):
                            st_x = sx_f0 + int(np.argmax(up_col_sums))
                            ys_st = np.where(pure_stems[max(0, int(round(note.y - sp * 3.8))):max(1, int(round(note.y - sp * 0.4))), st_x] > 0)[0]
                            if len(ys_st) > 0:
                                y_tip = max(0, int(round(note.y - sp * 3.8))) + int(ys_st[0])
                                fy0 = max(0, y_tip - 2)
                                fy1 = min(int(round(note.y - sp * 0.9)), y_tip + int(round(sp * 1.65)))
                                fx0 = min(roi.shape[1], st_x + 3)
                                fx1 = min(roi.shape[1], st_x + int(round(sp * 1.15)))
                                if fy1 > fy0 and fx1 > fx0:
                                    f_patch = roi_clean[fy0:fy1, fx0:fx1] > 0
                                    if int(np.sum(f_patch)) >= max(6, int(sp * sp * 0.14)) and int(np.sum(np.any(f_patch, axis=0))) >= max(3, int(sp * 0.30)):
                                        distinct_beams = 1

                note.beam_count = min(3, distinct_beams)

            # 附點判定：位於符頭右側（含加線音符右上間 dy ∈ [-0.78*sp, +0.38*sp]），且不屬於後方相鄰音符
            min_dot_dx = max(note.width * 0.42, sp * 0.62)
            note.is_dotted = any(
                (min_dot_dx <= (dx - note.x) <= sp * 1.85)
                and (-sp * 0.78 <= (dy - note.y) <= sp * 0.38)
                and all(abs(dx - other.x) > sp * 0.58 for other in merged if other is not note)
                for dx, dy in true_dots
            )

            # 演奏記號判定：優先檢查「延長記號 (𝄐)」，其次檢查「保持音 (-)」與「跳音 (.)」
            if any(abs(note.x - fcx) <= sp * 1.10 for _, _, _, _, fcx in fermata_zones):
                note.articulation = "延長記號 (𝄐)"
            elif any(
                abs(tx - note.x) <= sp * 0.58 and (sp * 0.62 <= abs(ty - note.y) <= sp * 2.35)
                for tx, ty in tenuto_bars
            ):
                note.articulation = "保持音 (-)"
            elif any(
                abs(dx - note.x) <= sp * 0.48 and (sp * 0.65 <= abs(dy - note.y) <= sp * 2.25)
                for dx, dy in true_dots
            ):
                note.articulation = "跳音 (.)"

        return merged, horiz_bars, header_end_x, detected_ts

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
    def _ocr_single_digit(patch_bin: np.ndarray) -> int:
        """辨識單一獨立數字區塊 (0..9)；若為左/右括號 ( ) 則回傳 -1"""
        ys, xs = np.where(patch_bin > 0)
        if len(xs) == 0:
            return -1
        p = (patch_bin[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0).astype(np.uint8)
        h, w = p.shape[:2]
        if h < 5:
            return -1
        aspect = w / float(h)

        # 排除左右圓括號 ( )：窄長且中段明顯向單側彎曲
        if aspect <= 0.48 and h >= 8:
            xt = np.where(p[:max(2, h // 5), :])[1]
            xm = np.where(p[h // 2 - 1:h // 2 + 2, :])[1]
            xb = np.where(p[-max(2, h // 5):, :])[1]
            if len(xt) > 0 and len(xm) > 0 and len(xb) > 0:
                bow = float(np.mean(xm)) - 0.5 * (float(np.mean(xt)) + float(np.mean(xb)))
                if abs(bow) >= max(1.15, w * 0.22):
                    return -1

        # 先計算內部封閉孔洞與水平列填充率
        pad_p = cv2.copyMakeBorder(p * 255, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        cnts, hier = cv2.findContours(pad_p, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = []
        if hier is not None:
            for ci, hi in enumerate(hier[0]):
                if hi[3] >= 0:
                    ha = cv2.contourArea(cnts[ci])
                    if ha >= max(2.5, h * w * 0.025):
                        _, hy, _, hh = cv2.boundingRect(cnts[ci])
                        holes.append(((hy - 2 + hh / 2.0) / float(h), ha))

        row_fills = np.mean(p, axis=1)
        top_fill = float(np.mean(p[:max(2, int(h * 0.20)), :]))
        bot_fill = float(np.mean(p[max(0, h - max(2, int(h * 0.20))):, :]))

        holes_bot = [hl for hl in holes if hl[0] >= 0.52]
        if len(holes) >= 2:
            return 8
        # "4" 的明確特徵：中下段 48%~85% 有橫跨全寬的水平橫槓、底部僅單柱窄腳，且下半部絕無封閉圓孔（區隔 "6"！）
        if np.max(row_fills[int(h * 0.48):max(int(h * 0.48) + 1, int(h * 0.85))]) >= 0.78 and bot_fill <= 0.62 and len(holes_bot) == 0:
            return 4

        # 直立單柱特徵辨識 "1"（無封閉孔洞，同時相容無襯線細 1 與寬底襯線印刷體 1）
        mid_band = p[int(h * 0.30):int(h * 0.72), :]
        mid_w_ratio = float(np.mean(np.sum(mid_band, axis=1))) / float(h) if mid_band.size > 0 else aspect
        max_col_fill = float(np.max(np.sum(p, axis=0))) / float(h)
        shaft_zone = p[int(h * 0.35):max(int(h * 0.35) + 1, int(h * 0.85)), :]
        qw = max(1, w // 4)
        pure_stem_rows = int(np.sum((np.sum(shaft_zone[:, :qw], axis=1) == 0) & (np.sum(shaft_zone[:, w - qw:], axis=1) == 0)))
        is_serif_one = (
            len(holes) == 0
            and w >= 6
            and max_col_fill >= 0.93
            and aspect <= 0.95
            and pure_stem_rows >= max(2, int(h * 0.14))
            and bot_fill >= 0.65
        )
        if aspect <= 0.44 or (len(holes) == 0 and mid_w_ratio <= 0.36 and max_col_fill >= 0.92 and aspect <= 0.92) or is_serif_one:
            return 1
        # 印刷襯線體 "2"：底部為寬平實心橫底 (bot_fill >= 0.72) 且下半部無封閉圓孔（即便左上襯線圓球微觸拱頂形成上方小孔仍為 2！）
        holes_bot = [hl for hl in holes if hl[0] >= 0.52]
        if bot_fill >= 0.72 and len(holes_bot) == 0:
            ul = float(np.mean(p[int(h * 0.18):max(int(h * 0.18) + 1, int(h * 0.45)), :max(1, w // 2)]))
            ur = float(np.mean(p[int(h * 0.18):max(int(h * 0.18) + 1, int(h * 0.45)), max(1, w // 2):]))
            return 2 if ur >= ul * 0.85 else 5
        if len(holes) == 1:
            hy_rel = holes[0][0]
            if hy_rel >= 0.53:
                return 6
            if hy_rel <= 0.45:
                return 9
            return 0
        if top_fill >= 0.74 and bot_fill <= 0.40:
            return 7

        norm_d = cv2.resize(p * 255, (24, 32), interpolation=cv2.INTER_AREA)
        best_d, best_sc = 3, -1.0
        for d in range(10):
            for font in (cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX):
                t = np.zeros((40, 32), dtype=np.uint8)
                cv2.putText(t, str(d), (3, 32), font, 1.0, 255, 2)
                ty, tx = np.where(t > 0)
                if len(tx) > 0:
                    t = cv2.resize(t[ty.min():ty.max() + 1, tx.min():tx.max() + 1], (24, 32))
                sc = float(cv2.matchTemplate(norm_d, t, cv2.TM_CCOEFF_NORMED)[0, 0])
                if sc > best_sc:
                    best_sc, best_d = sc, d
        return best_d

    @classmethod
    def _recognize_time_signature_patch(
        cls,
        vert_keep: np.ndarray,
        sy1: float,
        sy2: float,
        sp: float,
        x0: int,
        x1: int,
    ) -> Optional[str]:
        """辨識第一行譜頭的上下拍號數字（支援 2/4、3/4、4/4、6/8）"""
        mid_y = int(round(0.5 * (sy1 + sy2)))
        top_p = vert_keep[max(0, int(round(sy1 - sp * 0.2))):min(vert_keep.shape[0], mid_y), x0:x1]
        bot_p = vert_keep[max(0, mid_y):min(vert_keep.shape[0], int(round(sy2 + sp * 0.2))), x0:x1]
        if top_p.size == 0 or bot_p.size == 0:
            return None

        num = cls._ocr_single_digit(top_p)
        den = cls._ocr_single_digit(bot_p)
        cand = f"{num}/{den}"
        if cand in ("2/4", "3/4", "4/4", "6/8"):
            return cand
        # 若分母為 4 且分子落在 2..4，或分母為 8 且分子為 6
        if den == 4 and num in (2, 3, 4):
            return f"{num}/4"
        if den == 8 and num == 6:
            return "6/8"
        return None

    @classmethod
    def _detect_tempo_bpm(
        cls,
        clean_gray: np.ndarray,
        staff1_top_y: float,
        sp: float,
    ) -> Optional[int]:
        """掃描第一行五線譜上方的速度標記區（如 Allegro moderato (♩ = 116)），自動擷取 BPM 數值"""
        h, w = clean_gray.shape[:2]
        hy1 = max(0, int(round(staff1_top_y - sp * 7.8)))
        hy2 = max(hy1 + 8, int(round(staff1_top_y - sp * 0.55)))
        if hy2 <= hy1 + 8:
            return None

        hdr_crop = clean_gray[hy1:hy2, :int(w * 0.85)]
        # 以較嚴格門檻 (< 182) 尋找等號 "=" 的上下兩條平行短橫線（避免抗鋸齒灰階把上下雙橫線黏成單塊）
        eq_bin = ((hdr_crop < 182).astype(np.uint8)) * 255
        nl_eq, _, st_eq, ct_eq = cv2.connectedComponentsWithStats(eq_bin)

        # 以柔和門檻 (< 202) 讀取右側數字（完整保留襯線體 6 / 9 / 0 的細弧線封閉孔洞）
        hdr_bin = ((hdr_crop < 202).astype(np.uint8)) * 255
        nl, _, st, ct = cv2.connectedComponentsWithStats(hdr_bin)

        # 尋找速度算式中的等號 "="：兩條等寬、水平對齊、上下緊鄰的獨立短橫線
        horiz_pieces = []
        for lbl in range(1, nl_eq):
            bx, by, bw, bh, ba = [int(st_eq[lbl, k]) for k in range(5)]
            if (
                sp * 0.36 <= bw <= sp * 1.85
                and 1 <= bh <= max(5, int(round(sp * 0.45)))
                and bw >= bh * 1.9
            ):
                horiz_pieces.append((lbl, bx, by, bw, bh, float(ct_eq[lbl][0]), float(ct_eq[lbl][1])))

        eq_candidates = []
        for i in range(len(horiz_pieces)):
            for j in range(i + 1, len(horiz_pieces)):
                b1, b2 = horiz_pieces[i], horiz_pieces[j]
                if b1[6] > b2[6]:
                    b1, b2 = b2, b1
                if (
                    abs(b1[5] - b2[5]) <= sp * 0.30
                    and abs(b1[3] - b2[3]) <= sp * 0.38
                    and (sp * 0.14 <= (b2[6] - b1[6]) <= sp * 0.95)
                ):
                    eq_candidates.append((b1, b2))

        if not eq_candidates:
            return None

        # 由最靠近第一行五線譜的速度列往上依序檢查每個候選等號（排除標題裝飾雙橫線！）
        eq_candidates.sort(key=lambda pair: pair[1][6], reverse=True)

        for b1, b2 in eq_candidates:
            eq_right = max(b1[1] + b1[3], b2[1] + b2[3])
            eq_cy = 0.5 * (b1[6] + b2[6])

            raw_boxes: List[Tuple[int, int, int, int]] = []
            for lbl in range(1, nl):
                bx, by, bw, bh, ba = [int(st[lbl, k]) for k in range(5)]
                cy = float(ct[lbl][1])
                if (
                    eq_right + 2 <= bx <= int(round(eq_right + sp * 7.2))
                    and abs(cy - eq_cy) <= sp * 1.25
                    and sp * 0.65 <= bh <= sp * 2.8
                    and 2 <= bw <= sp * 4.8
                ):
                    raw_boxes.append((bx, by, bx + bw, by + bh))

            if not raw_boxes:
                continue

            raw_boxes.sort(key=lambda r: r[0])
            merged_boxes: List[List[int]] = [[*raw_boxes[0]]]
            for bx0, by0, bx1, by1 in raw_boxes[1:]:
                prev = merged_boxes[-1]
                if bx0 < prev[2] - 1:
                    prev[0] = min(prev[0], bx0)
                    prev[1] = min(prev[1], by0)
                    prev[2] = max(prev[2], bx1)
                    prev[3] = max(prev[3], by1)
                else:
                    merged_boxes.append([bx0, by0, bx1, by1])

            # 若多個襯線數字底部微黏成單一寬區塊 (bw > bh * 1.15)，依垂直投影低谷自動切開為獨立字元！
            split_boxes: List[Tuple[int, int, int, int]] = []
            for bx0, by0, bx1, by1 in merged_boxes:
                bw_m = bx1 - bx0
                bh_m = max(1, by1 - by0)
                if bw_m > bh_m * 1.15 and bw_m >= 14:
                    upper_h = max(3, int(bh_m * 0.72))
                    col_sums = np.sum(hdr_bin[by0:by0 + upper_h, bx0:bx1] > 0, axis=0)
                    full_sums = np.sum(hdr_bin[by0:by1, bx0:bx1] > 0, axis=0)
                    valley_max = max(2, int(upper_h * 0.22))
                    cuts = [0]
                    c_i = 4
                    while c_i < bw_m - 4:
                        if col_sums[c_i] <= valley_max and (c_i - cuts[-1]) >= 5:
                            v_end = c_i
                            while v_end + 1 < bw_m - 3 and col_sums[v_end + 1] <= valley_max:
                                v_end += 1
                            sub_f = full_sums[c_i:v_end + 1]
                            best_cut = v_end - int(np.argmin(sub_f[::-1]))
                            cuts.append(best_cut)
                            c_i = v_end + 4
                            continue
                        c_i += 1
                    cuts.append(bw_m)
                    for k_c in range(len(cuts) - 1):
                        if cuts[k_c + 1] - cuts[k_c] >= 4:
                            split_boxes.append((bx0 + cuts[k_c], by0, bx0 + cuts[k_c + 1], by1))
                else:
                    split_boxes.append((bx0, by0, bx1, by1))

            digits_str = ""
            last_x1 = eq_right
            for bx0, by0, bx1, by1 in split_boxes:
                if (bx0 - last_x1) > sp * 1.8 and len(digits_str) >= 2:
                    break
                d_val = cls._ocr_single_digit(hdr_bin[by0:by1, bx0:bx1])
                if d_val < 0:
                    if len(digits_str) >= 2:
                        break
                    continue
                digits_str += str(d_val)
                last_x1 = bx1
                if len(digits_str) >= 3:
                    break

            if len(digits_str) >= 2:
                try:
                    bpm_val = int(digits_str)
                    if 36 <= bpm_val <= 240:
                        return bpm_val
                except ValueError:
                    pass
        return None

    @staticmethod
    def _y_to_treble_pitch(
        ny: float,
        y_bot: float,
        sp: float,
        accidental: int = 0,
        show_natural: bool = False,
    ) -> Tuple[int, str]:
        if sp <= 0:
            return 60, "C4"
        step = int(round((y_bot - ny) / (sp / 2.0)))
        step = max(-14, min(20, step))
        diatonic_idx = 2 + step
        octave = 4 + (diatonic_idx // 7)
        deg = diatonic_idx % 7
        semitones = (0, 2, 4, 5, 7, 9, 11)
        names = ("C", "D", "E", "F", "G", "A", "B")
        midi = (octave + 1) * 12 + semitones[deg] + int(accidental)
        midi = max(36, min(96, midi))
        if accidental > 0:
            acc_str = "#"
        elif accidental < 0:
            acc_str = "b"
        elif show_natural:
            acc_str = "♮"
        else:
            acc_str = ""
        if acc_str == "♮":
            return midi, f"♮{names[deg]}{octave}"
        return midi, f"{names[deg]}{acc_str}{octave}"
