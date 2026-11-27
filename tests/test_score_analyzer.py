"""樂譜影像結構分析 (ScoreAnalyzer) 與方案 A 音符追蹤測試"""

import os
import tempfile
import unittest

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"
_app = QApplication.instance()
if not _app:
    _app = QApplication([])

from src.core.score_analyzer import ScoreAnalyzer
from src.core.note_synth import NoteSynthesizer
from src.ui.auto_scroll_page import AutoScrollPage


def _create_synthetic_score_rgb(width: int = 1000, height: int = 600) -> np.ndarray:
    """產生含 2 行五線譜、小節線與實心音符的合成樂譜影像 (RGB)"""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    sp = 12
    for staff_y in (140, 360):
        # 繪製 5 條五線譜橫線
        for i in range(5):
            y = staff_y + i * sp
            cv2.line(img, (80, y), (920, y), (20, 20, 20), 2)
        y_top = staff_y
        y_bot = staff_y + 4 * sp
        # 繪製小節線（3 個小節）
        for bx in (360, 640, 920):
            cv2.line(img, (bx, y_top), (bx, y_bot), (20, 20, 20), 2)
        # 在各小節內繪製實心音符符頭
        note_xs = [180, 250, 310, 430, 500, 570, 710, 810]
        for idx, nx in enumerate(note_xs):
            ny = staff_y + (idx % 4) * sp + sp // 2
            cv2.ellipse(img, (nx, ny), (7, 5), -15, 0, 360, (15, 15, 15), -1)
    return img


class TestScoreAnalyzerAndTracking(unittest.TestCase):
    def setUp(self):
        self.analyzer = ScoreAnalyzer()
        self.temp_files = []

    def tearDown(self):
        for f in self.temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def test_detect_staves_measures_and_notes(self):
        rgb = _create_synthetic_score_rgb()
        res = self.analyzer.analyze_pages([rgb], beats_per_measure=4.0)

        self.assertTrue(res.has_staves)
        self.assertEqual(len(res.pages), 1)
        self.assertEqual(len(res.pages[0].staves), 2)
        self.assertEqual(res.total_measures, 6)
        self.assertAlmostEqual(res.total_beats, 24.0, places=2)
        self.assertGreaterEqual(len(res.events), 14)

        # 驗證時間軸單調遞增且每個非休止事件都有有效 MIDI 音高
        for i in range(1, len(res.events)):
            self.assertGreaterEqual(res.events[i].beat_offset, res.events[i - 1].beat_offset)
        pitched = [ev for ev in res.events if not ev.is_rest]
        self.assertGreater(len(pitched), 0)
        for ev in pitched:
            self.assertGreaterEqual(ev.midi_note, 36)
            self.assertLessEqual(ev.midi_note, 96)
            self.assertTrue(len(ev.note_name) >= 2)

    def test_note_synthesizer_wave_generation(self):
        synth = NoteSynthesizer()
        self.assertAlmostEqual(synth.midi_to_freq(69), 440.0, places=2)
        self.assertTrue(synth.using_soundfont)
        expected_instruments = [
            "鋼琴", "薩克斯風", "長笛", "豎笛", "雙簧管", "低音管",
            "小提琴", "大提琴", "低音提琴", "小號", "法國號", "管風琴",
            "馬林巴", "定音鼓", "三角鐵", "小鼓", "大鼓",
        ]
        for timbre in expected_instruments:
            self.assertIn(timbre, NoteSynthesizer.TIMBRES)
            synth.timbre = timbre
            pcm = synth.synthesize_wave(60, 0.25)
            self.assertEqual(pcm.dtype, np.int16)
            self.assertEqual(pcm.ndim, 2)
            self.assertEqual(pcm.shape[1], 2)
            self.assertGreater(pcm.shape[0], 10000)
            self.assertGreater(int(np.max(np.abs(pcm))), 2000)

    def test_multi_measure_rest_detection(self):
        """驗證五線譜中央粗橫條 + 上方數字能正確辨識為休止 2 小節"""
        img = np.full((320, 900, 3), 255, dtype=np.uint8)
        sp = 12
        staff_y = 140
        for i in range(5):
            cv2.line(img, (60, staff_y + i * sp), (840, staff_y + i * sp), (20, 20, 20), 2)
        y_top, y_bot = staff_y, staff_y + 4 * sp
        # 小節線：M1 休止 2 小節 (60..320)，M3 有音符 (320..840)
        for bx in (320, 840):
            cv2.line(img, (bx, y_top), (bx, y_bot), (20, 20, 20), 2)
        # 在第一小節中央繪製粗橫條與上方數字 "2"
        cv2.rectangle(img, (160, staff_y + 2 * sp - 3), (240, staff_y + 2 * sp + 3), (15, 15, 15), -1)
        cv2.putText(img, "2", (192, staff_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2)
        # 在第二段小節繪製 2 個實心四分音符
        for nx in (440, 620):
            cv2.ellipse(img, (nx, staff_y + 2 * sp), (7, 5), -15, 0, 360, (15, 15, 15), -1)

        res = self.analyzer.analyze_pages([img], beats_per_measure=2.0)
        self.assertTrue(res.has_staves)
        first_ev = res.events[0]
        self.assertTrue(first_ev.is_rest)
        self.assertEqual(first_ev.rest_measures, 2)
        self.assertAlmostEqual(first_ev.duration_beats, 4.0, places=2)

        # 驗證放大到 2400px 高解析度時，粗體數字 "2" 跨在第一線上仍能精準辨識為休止 2 小節
        hires = cv2.resize(img, (2400, 850), interpolation=cv2.INTER_CUBIC)
        res_hi = self.analyzer.analyze_pages([hires], beats_per_measure=2.0)
        self.assertTrue(res_hi.has_staves)
        self.assertTrue(res_hi.events[0].is_rest)
        self.assertEqual(res_hi.events[0].rest_measures, 2)
        self.assertEqual(res_hi.events[0].note_type, "休止 2 小節")

    def test_shaded_paper_with_time_sig_2_4_and_2_measure_rest(self):
        """驗證泛黃漸層底色樂譜先去底色強化後，開頭譜號與 2/4 拍號不會誤判為音符，且第一小節精準辨識為休止 2 小節"""
        h, w = 360, 1000
        # 建立泛黃且左暗右亮的不均勻紙張底色 + 模擬背面淡淡透字雜訊
        x_grad = np.linspace(195, 238, w, dtype=np.float32)[None, :]
        bg_r = np.clip(x_grad + 6, 0, 255).astype(np.uint8)
        bg_g = np.clip(x_grad + 2, 0, 255).astype(np.uint8)
        bg_b = np.clip(x_grad - 18, 0, 255).astype(np.uint8)
        img = np.stack([np.broadcast_to(bg_r, (h, w)), np.broadcast_to(bg_g, (h, w)), np.broadcast_to(bg_b, (h, w))], axis=-1).copy()
        cv2.rectangle(img, (120, 80), (420, 96), (205, 200, 185), -1)  # 淡透字陰影

        sp = 12
        staff_y = 150
        for i in range(5):
            cv2.line(img, (80, staff_y + i * sp), (920, staff_y + i * sp), (35, 35, 35), 2)
        y_top, y_bot = staff_y, staff_y + 4 * sp

        # 繪製高音譜號（穿透上下界）+ 2/4 拍號（位於 dx ≈ 5.2 * sp）
        cv2.ellipse(img, (104, staff_y + 2 * sp), (9, 32), 8, 0, 360, (30, 30, 30), 2)
        cv2.line(img, (105, staff_y - 16), (105, y_bot + 18), (30, 30, 30), 2)
        cv2.putText(img, "2", (136, staff_y + 2 * sp - 1), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (28, 28, 28), 2)
        cv2.putText(img, "4", (136, y_bot - 1), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (28, 28, 28), 2)

        # 第一小節：休止 2 小節粗橫條 + 上方數字 "2" + 小節線
        cv2.rectangle(img, (215, staff_y + 2 * sp - 3), (275, staff_y + 2 * sp + 3), (28, 28, 28), -1)
        cv2.putText(img, "2", (237, staff_y - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.72, (28, 28, 28), 2)
        cv2.line(img, (340, y_top), (340, y_bot), (30, 30, 30), 3)
        cv2.line(img, (920, y_top), (920, y_bot), (30, 30, 30), 2)

        # 第二段小節（M3）：2 個獨立四分音符（含音幹）
        for nx, ny in ((450, staff_y + sp + sp // 2), (620, staff_y + 2 * sp)):
            cv2.ellipse(img, (nx, ny), (7, 5), -18, 0, 360, (25, 25, 25), -1)
            cv2.line(img, (nx - 6, ny), (nx - 6, ny + 3 * sp), (25, 25, 25), 2)

        enh = ScoreAnalyzer.enhance_score_rgb(img)
        # 驗證原本泛黃左暗右亮底色 + 淡透字區域已被洗白為純白 (>= 252)
        self.assertGreater(float(np.mean(enh[20:105, 50:450])), 252.0)

        res = self.analyzer.analyze_pages([img], beats_per_measure=2.0)
        self.assertTrue(res.has_staves)
        self.assertGreaterEqual(len(res.events), 3)
        # 事件 #1 必須是「休止 2 小節」，絕不能把 2/4 或 2 誤判成音符
        ev0 = res.events[0]
        self.assertTrue(ev0.is_rest)
        self.assertEqual(ev0.rest_measures, 2)
        self.assertEqual(ev0.note_type, "休止 2 小節")
        # 後續 2 個音符皆在 x=340 小節線之後
        self.assertFalse(res.events[1].is_rest)
        self.assertGreater(res.events[1].x_norm, 0.38)

    def test_perspective_keystone_multi_staff_independent_tilt(self):
        """驗證同一頁上下行五線譜具有相反透視傾斜角（上行 -2.0°、下行 +1.5°）時，仍能逐行拉直並精準辨識休止 2 小節與音符"""
        s_top = np.full((260, 1100, 3), 240, dtype=np.uint8)
        sp = 12
        sy = 120
        for i in range(5):
            cv2.line(s_top, (80, sy + i * sp), (1020, sy + i * sp), (25, 25, 25), 2)
        cv2.putText(s_top, "2", (135, sy + 2 * sp - 1), cv2.FONT_HERSHEY_TRIPLEX, 0.72, (25, 25, 25), 2)
        cv2.putText(s_top, "4", (135, sy + 4 * sp - 1), cv2.FONT_HERSHEY_TRIPLEX, 0.72, (25, 25, 25), 2)
        cv2.rectangle(s_top, (220, sy + 2 * sp - 3), (285, sy + 2 * sp + 3), (20, 20, 20), -1)
        cv2.putText(s_top, "2", (244, sy - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.72, (20, 20, 20), 2)
        cv2.line(s_top, (360, sy), (360, sy + 4 * sp), (25, 25, 25), 3)
        cv2.line(s_top, (1020, sy), (1020, sy + 4 * sp), (25, 25, 25), 2)
        for nx in (480, 660, 840):
            cv2.ellipse(s_top, (nx, sy + 2 * sp), (8, 6), -18, 0, 360, (20, 20, 20), -1)
            cv2.line(s_top, (nx + 7, sy - sp), (nx + 7, sy + 2 * sp), (20, 20, 20), 2)

        # 上行旋轉 +2.0°（左低右高），下行旋轉 -1.5°（左高右低）模擬手機翻拍透視變形
        s1 = cv2.warpAffine(s_top, cv2.getRotationMatrix2D((550, 130), 2.0, 1.0), (1100, 260), borderValue=(240, 240, 240))
        s2 = cv2.warpAffine(s_top, cv2.getRotationMatrix2D((550, 130), -1.5, 1.0), (1100, 260), borderValue=(240, 240, 240))
        keystone_page = np.vstack([s1, s2])

        res = self.analyzer.analyze_pages([keystone_page], beats_per_measure=2.0)
        self.assertEqual(len(res.pages[0].staves), 2)
        # 驗證第一行與第二行的譜線高度皆未被壓扁（各約 4*sp / 520 ≈ 0.092）
        for st in res.pages[0].staves:
            self.assertGreater(st.staff_height_norm, 0.075)
            self.assertEqual(st.measures[0].rest_measures, 2)
            self.assertEqual(len(st.measures[1].notes), 3)

    def test_repeat_signs_detection_and_playback_unrolling(self):
        """驗證前反覆記號 ||: 與後反覆記號 :|| 能精準辨識，並自動展開第 1 遍與第 2 遍回跳播放及支援開關切換"""
        img = np.full((300, 1100, 3), 255, dtype=np.uint8)
        sp = 14
        sy = 120
        for i in range(5):
            cv2.line(img, (60, sy + i * sp), (1040, sy + i * sp), (20, 20, 20), 2)
        y_top, y_bot = sy, sy + 4 * sp

        # M1 (x=120..320): 一般引子小節 (1 顆音符)
        cv2.ellipse(img, (230, sy + 2 * sp), (8, 6), -15, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (237, sy - sp), (237, sy + 2 * sp), (15, 15, 15), 2)

        # 前反覆記號 ||: 於 x=320 (粗線 + 細線 + 右側第二、三間雙圓點) -> M2 起點
        cv2.rectangle(img, (316, y_top), (322, y_bot), (15, 15, 15), -1)
        cv2.line(img, (327, y_top), (327, y_bot), (15, 15, 15), 2)
        cv2.circle(img, (336, int(sy + 1.5 * sp)), 3, (15, 15, 15), -1)
        cv2.circle(img, (336, int(sy + 2.5 * sp)), 3, (15, 15, 15), -1)

        # M2 (x=327..660): 1 顆音符 + 一般單細小節線 x=660
        cv2.ellipse(img, (490, sy + sp), (8, 6), -15, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (497, sy - sp), (497, sy + sp), (15, 15, 15), 2)
        cv2.line(img, (660, y_top), (660, y_bot), (20, 20, 20), 2)

        # M3 (x=660..1035): 1 顆音符 + 後反覆記號 :|| 於結尾 x=1030 (左側雙圓點 + 細線 + 粗線)
        cv2.ellipse(img, (830, sy + 2 * sp), (8, 6), -15, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (837, sy - sp), (837, sy + 2 * sp), (15, 15, 15), 2)
        cv2.circle(img, (1019, int(sy + 1.5 * sp)), 3, (15, 15, 15), -1)
        cv2.circle(img, (1019, int(sy + 2.5 * sp)), 3, (15, 15, 15), -1)
        cv2.line(img, (1028, y_top), (1028, y_bot), (15, 15, 15), 2)
        cv2.rectangle(img, (1033, y_top), (1039, y_bot), (15, 15, 15), -1)

        # 1. 驗證啟用反覆時：實體 3 小節 (M1, M2, M3)，反覆區段為 [(2, 3)]，播放順序為 M1 -> M2(1) -> M3(1) -> M2(2) -> M3(2) 共 5 小節！
        res_on = self.analyzer.analyze_pages([img], beats_per_measure=2.0, enable_repeats=True)
        self.assertEqual(res_on.total_measures, 3)
        self.assertEqual(res_on.repeat_sections, [(2, 3)])
        seq_on = [(pm.measure.measure_number, pm.repeat_pass) for pm in res_on.playback_measures]
        self.assertEqual(seq_on, [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2)])
        self.assertAlmostEqual(res_on.total_beats, 10.0, places=2)

        # 2. 驗證 UI 開關切換（關閉反覆 -> 單遍 6 拍；重新開啟 -> 雙遍 10 拍）
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.temp_files.append(path)

        page = AutoScrollPage()
        page.sound_btn.setChecked(False)
        page.time_sig_combo.setCurrentText("2/4")
        self.assertTrue(page.load_file(path))
        self.assertAlmostEqual(page._analysis.total_beats, 10.0, places=2)

        # 步進到第 4 個播放小節應跳回 M2 [第 2 遍]
        for _ in range(3):
            page._on_next_measure()
        _, ev_pass2 = page._find_event_at_beat(page._current_beat)
        self.assertEqual(ev_pass2.measure_number, 2)
        self.assertEqual(ev_pass2.repeat_pass, 2)
        self.assertIn("第2遍", page.measure_label.text())

        # 關閉「啟用反覆」後改為直通 3 小節 (6.0 拍)
        page.repeat_chk.setChecked(False)
        self.assertAlmostEqual(page._analysis.total_beats, 6.0, places=2)

    def test_measure_step_and_note_step_calibration_ui(self):
        """驗證逐小節播放、逐音符步進與詳細音符資訊列及偵測標記疊層"""
        rgb = _create_synthetic_score_rgb()
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.temp_files.append(path)

        page = AutoScrollPage()
        page.sound_btn.setChecked(False)  # 測試環境靜音
        self.assertTrue(page.load_file(path))

        # 1. 測試逐音符步進：下一音 / 上一音
        page._on_next_note()
        idx1, ev1 = page._find_event_at_beat(page._current_beat)
        self.assertEqual(idx1, 1)
        self.assertEqual(ev1.event_index, 2)
        detail_txt = page.note_detail_label.text()
        self.assertIn("事件 #2", detail_txt)
        self.assertIn("音符", detail_txt)
        self.assertIn("Hz", detail_txt)

        page._on_prev_note()
        idx0, _ = page._find_event_at_beat(page._current_beat)
        self.assertEqual(idx0, 0)

        # 2. 測試逐小節跳轉與「播本小節」自動設定停止拍點
        page._on_next_measure()
        _, ev_m2 = page._find_event_at_beat(page._current_beat)
        self.assertEqual(ev_m2.measure_number, 2)

        page._on_play_current_measure()
        self.assertTrue(page._is_playing)
        self.assertIsNotNone(page._stop_at_beat)
        page._stop_playing()

        # 3. 測試開啟「顯示偵測標記」與切換「去底色強化」渲染無誤
        self.assertTrue(page.enhance_view_chk.isChecked())
        page.enhance_view_chk.setChecked(False)
        page.enhance_view_chk.setChecked(True)
        page.debug_overlay_chk.setChecked(True)
        pm_grab = page.score_view.page_labels[0].grab()
        self.assertFalse(pm_grab.isNull())

    def test_auto_scroll_page_note_tracking_and_click_seek(self):
        rgb = _create_synthetic_score_rgb()
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.temp_files.append(path)

        page = AutoScrollPage()
        self.assertTrue(page.load_file(path))
        self.assertTrue(page._use_note_tracking())
        self.assertEqual(page._analysis.total_measures, 6)

        # 驗證切換拍號為 2/4 後總拍數自動重算
        page.time_sig_combo.setCurrentText("2/4")
        self.assertAlmostEqual(page._analysis.total_beats, 12.0, places=2)

        # 驗證點擊第二行五線譜可跳轉至後段小節
        page._on_score_clicked(0, 0.5, 0.64)
        self.assertGreater(page._current_beat, 0.0)
        self.assertIn("小節：", page.measure_label.text())

        # 驗證觸發播放與單次渲染不報錯
        page._on_play_toggle()
        self.assertTrue(page._is_playing)
        page.score_view.page_labels[0].grab()
        page._on_play_toggle()
        self.assertFalse(page._is_playing)

    def test_auto_time_sig_bpm_accidentals_dots_and_tenuto(self):
        """驗證 6 項校準功能：自動拍號 (2/4)、自動速度 (♩=116)、保持音橫槓不誤判附點、加線附點八分音符、降記號 Bb5 與升記號 G#5"""
        img = np.full((360, 1250, 3), 255, dtype=np.uint8)
        sp = 12
        sy = 160
        for i in range(5):
            cv2.line(img, (60, sy + i * sp), (1190, sy + i * sp), (25, 25, 25), 2)
        y_top, y_bot = sy, sy + 4 * sp

        # 譜頭上方速度標記：(♩ = 116)
        cv2.putText(img, "Allegro (", (110, sy - 38), cv2.FONT_HERSHEY_TRIPLEX, 0.62, (20, 20, 20), 1)
        cv2.ellipse(img, (238, sy - 42), (5, 4), -20, 0, 360, (20, 20, 20), -1)
        cv2.line(img, (242, sy - 64), (242, sy - 42), (20, 20, 20), 2)
        cv2.rectangle(img, (254, sy - 49), (266, sy - 47), (20, 20, 20), -1)
        cv2.rectangle(img, (254, sy - 43), (266, sy - 41), (20, 20, 20), -1)
        cv2.putText(img, "116)", (276, sy - 38), cv2.FONT_HERSHEY_TRIPLEX, 0.62, (20, 20, 20), 2)

        # 譜號 + 2/4 拍號
        cv2.ellipse(img, (86, sy + 2 * sp), (9, 30), 8, 0, 360, (20, 20, 20), 2)
        cv2.line(img, (87, sy - 16), (87, y_bot + 16), (20, 20, 20), 2)
        cv2.putText(img, "2", (120, sy + 2 * sp - 1), cv2.FONT_HERSHEY_TRIPLEX, 0.72, (20, 20, 20), 2)
        cv2.putText(img, "4", (120, y_bot - 1), cv2.FONT_HERSHEY_TRIPLEX, 0.72, (20, 20, 20), 2)

        # 小節線：M1(保持音四分音符 E5), M2(加線附點八分音符 A5. + 十六分 E5), M3(降記號 Bb5 + 升記號 G#5)
        for bx in (420, 780, 1190):
            cv2.line(img, (bx, y_top), (bx, y_bot), (25, 25, 25), 2)

        # M1: E5 四分音符 (第四間 y = sy + 0.5*sp) + 正上方保持音短橫槓 (-)
        e5_x, e5_y = 270, int(sy + 0.5 * sp)
        cv2.ellipse(img, (e5_x, e5_y), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (e5_x - 7, e5_y), (e5_x - 7, e5_y + 3 * sp), (15, 15, 15), 2)
        cv2.rectangle(img, (e5_x - 6, sy - 8), (e5_x + 6, sy - 6), (15, 15, 15), -1)

        # M2: 上加一線 A5 (y = sy - sp) 附點八分音符 + E5 十六分音符（共用主橫槓 + 右側短雙槓）
        a5_x, a5_y = 530, sy - sp
        cv2.line(img, (a5_x - 12, a5_y), (a5_x + 12, a5_y), (25, 25, 25), 2)  # 加線
        cv2.ellipse(img, (a5_x, a5_y), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.circle(img, (a5_x + 14, a5_y - 4), 3, (15, 15, 15), -1)  # 右上間附點
        cv2.line(img, (a5_x - 7, a5_y), (a5_x - 7, a5_y + 3 * sp + 6), (15, 15, 15), 2)

        n2_x, n2_y = 630, int(sy + 0.5 * sp)
        cv2.ellipse(img, (n2_x, n2_y), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (n2_x - 7, n2_y), (n2_x - 7, a5_y + 3 * sp + 6), (15, 15, 15), 2)
        cv2.rectangle(img, (a5_x - 7, a5_y + 3 * sp + 3), (n2_x - 6, a5_y + 3 * sp + 8), (15, 15, 15), -1)

        # M3: 降 B5 (Bb5, y = sy - 1.5*sp, MIDI 82) 與 升 G5 (G#5, y = sy - 0.5*sp, MIDI 80)
        bb_x, bb_y = 910, int(sy - 1.5 * sp)
        cv2.line(img, (bb_x - 16, sy - sp), (bb_x + 12, sy - sp), (25, 25, 25), 2)
        # 繪製降記號 ♭（左側朝上長幹 + 右下圓肚，下方無延伸）
        fx = bb_x - 19
        cv2.line(img, (fx, bb_y - 16), (fx, bb_y + 5), (15, 15, 15), 2)
        cv2.ellipse(img, (fx + 4, bb_y + 1), (4, 5), 0, -90, 90, (15, 15, 15), 2)
        cv2.ellipse(img, (bb_x, bb_y), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (bb_x - 7, bb_y), (bb_x - 7, bb_y + 3 * sp), (15, 15, 15), 2)

        gs_x, gs_y = 1070, int(sy - 0.5 * sp)
        # 繪製升記號 ♯（雙直幹貫穿上下 + 雙斜橫槓）
        sx_l, sx_r = gs_x - 21, gs_x - 15
        cv2.line(img, (sx_l, gs_y - 14), (sx_l, gs_y + 14), (15, 15, 15), 2)
        cv2.line(img, (sx_r, gs_y - 15), (sx_r, gs_y + 13), (15, 15, 15), 2)
        cv2.line(img, (sx_l - 2, gs_y - 4), (sx_r + 2, gs_y - 6), (15, 15, 15), 3)
        cv2.line(img, (sx_l - 2, gs_y + 5), (sx_r + 2, gs_y + 3), (15, 15, 15), 3)
        cv2.ellipse(img, (gs_x, gs_y), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (gs_x - 7, gs_y), (gs_x - 7, gs_y + 3 * sp), (15, 15, 15), 2)

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.temp_files.append(path)

        page = AutoScrollPage()
        page.sound_btn.setChecked(False)
        # 初始為預設 4/4 與 100 BPM，驗證 load_file 後自動切換為 2/4 與 116 BPM
        self.assertEqual(page.time_sig_combo.currentText(), "4/4")
        self.assertEqual(page.bpm_spin.value(), 100)
        self.assertTrue(page.load_file(path))

        self.assertEqual(page._analysis.detected_time_sig, "2/4")
        self.assertEqual(page.time_sig_combo.currentText(), "2/4")
        self.assertEqual(page._analysis.detected_bpm, 116)
        self.assertEqual(page.bpm_spin.value(), 116)

        st0 = page._analysis.pages[0].staves[0]
        m1_ev = st0.measures[0].notes[0]
        self.assertEqual(m1_ev.note_name, "E5")
        self.assertFalse(m1_ev.is_dotted)
        self.assertEqual(m1_ev.articulation, "保持音 (-)")
        self.assertIn("四分音符 (1/4)", m1_ev.note_type)
        self.assertIn("保持音 (-)", m1_ev.note_type)

        m2_ev0 = st0.measures[1].notes[0]
        self.assertEqual(m2_ev0.note_name, "A5")
        self.assertTrue(m2_ev0.is_dotted)
        self.assertIn("附點八分音符", m2_ev0.note_type)

        m3_notes = st0.measures[2].notes
        self.assertEqual(len(m3_notes), 2)
        self.assertEqual(m3_notes[0].note_name, "Bb5")
        self.assertEqual(m3_notes[0].midi_note, 82)
        self.assertEqual(m3_notes[0].accidental, -1)
        self.assertEqual(m3_notes[1].note_name, "G#5")
        self.assertEqual(m3_notes[1].midi_note, 80)
        self.assertEqual(m3_notes[1].accidental, 1)

    def test_fermata_hollow_half_note_and_volta_anti_shift(self):
        """驗證五線譜上方有反覆括號與延長記號 (Fermata 𝄐) 時：五線譜不偏移、延長記號不誤判為 C6，且空心二分音符 (F4 / A4) 與延音正確辨識"""
        img = np.full((340, 1100, 3), 255, dtype=np.uint8)
        sp = 14
        sy = 150
        for i in range(5):
            cv2.line(img, (60, sy + i * sp), (1040, sy + i * sp), (25, 25, 25), 2)
        y_top, y_bot = sy, sy + 4 * sp

        # 在五線譜上方 1*sp 處繪製長反覆跳房子括號 (1. ┌────────┐)，測試五線譜不會被吸上去偏移 1 線！
        cv2.line(img, (120, sy - sp), (320, sy - sp), (25, 25, 25), 2)
        cv2.line(img, (120, sy - sp), (120, sy - 3), (25, 25, 25), 2)

        # 小節線：M1 (F4 二分音符 + 延長記號 𝄐), M2 (A4 二分音符 + 延長記號 𝄐 + 終止雙小節線)
        cv2.line(img, (560, y_top), (560, y_bot), (25, 25, 25), 2)
        cv2.line(img, (1032, y_top), (1032, y_bot), (20, 20, 20), 2)
        cv2.rectangle(img, (1037, y_top), (1042, y_bot), (20, 20, 20), -1)

        def draw_hollow_half_with_fermata(nx: int, ny: int):
            # 空心橢圓符頭 + 右側朝上符幹
            cv2.ellipse(img, (nx, ny), (9, 6), -20, 0, 360, (15, 15, 15), -1)
            cv2.ellipse(img, (nx, ny), (5, 3), -20, 0, 360, (255, 255, 255), -1)
            cv2.line(img, (nx + 8, ny - int(3.3 * sp)), (nx + 8, ny), (15, 15, 15), 2)
            # 五線譜正上方繪製延長記號 Fermata (𝄐：內凹彎弧 + 中央圓點)
            fy = sy - int(1.9 * sp)
            cv2.ellipse(img, (nx + 2, fy + 10), (17, 12), 0, 190, 350, (15, 15, 15), 3)
            cv2.circle(img, (nx + 2, fy + 8), 3, (15, 15, 15), -1)

        # M1: 第一間 F4 (y = y_bot - 0.5*sp, MIDI 65) + Fermata
        draw_hollow_half_with_fermata(420, int(y_bot - 0.5 * sp))
        # M2: 第二間 A4 (y = y_bot - 1.5*sp, MIDI 69) + Fermata
        draw_hollow_half_with_fermata(820, int(y_bot - 1.5 * sp))

        res = self.analyzer.analyze_pages([img], beats_per_measure=2.0)
        self.assertTrue(res.has_staves)
        st0 = res.pages[0].staves[0]
        # 驗證五線譜上下界精準對齊 y=150..206，絕未上移 1 個間距
        self.assertAlmostEqual(st0.y_top_norm * 340.0, float(y_top), delta=2.5)
        self.assertAlmostEqual(st0.y_bot_norm * 340.0, float(y_bot), delta=2.5)

        self.assertEqual(len(st0.measures[0].notes), 1)
        ev_f4 = st0.measures[0].notes[0]
        self.assertEqual(ev_f4.note_name, "F4")
        self.assertEqual(ev_f4.midi_note, 65)
        self.assertTrue(ev_f4.is_hollow)
        self.assertEqual(ev_f4.articulation, "延長記號 (𝄐)")
        self.assertIn("二分音符 (1/2)", ev_f4.note_type)
        self.assertIn("延長記號 (𝄐)", ev_f4.note_type)
        self.assertAlmostEqual(ev_f4.duration_beats, 3.5, places=2)

        self.assertEqual(len(st0.measures[1].notes), 1)
        ev_a4 = st0.measures[1].notes[0]
        self.assertEqual(ev_a4.note_name, "A4")
        self.assertEqual(ev_a4.midi_note, 69)
        self.assertTrue(ev_a4.is_hollow)
        self.assertEqual(ev_a4.articulation, "延長記號 (𝄐)")
        self.assertIn("二分音符 (1/2)", ev_a4.note_type)
        self.assertIn("延長記號 (𝄐)", ev_a4.note_type)
        self.assertAlmostEqual(ev_a4.duration_beats, 3.5, places=2)

    def test_curved_book_staff_flagged_eighth_and_dense_16ths(self):
        """驗證書頁右半彎曲 + 漸弱記號 > 下：M5 的 C5/D5 不偏不漏，且右半 M17 帶彎勾八分音符與 M18 皆精準維持 G5（不誤判為 (C5) 或 B5）"""
        w, h = 1300, 320
        flat = np.full((h, w, 3), 255, dtype=np.uint8)
        sp = 14
        sy = 130
        for i in range(5):
            cv2.line(flat, (60, sy + i * sp), (1240, sy + i * sp), (25, 25, 25), 2)
        y_top, y_bot = sy, sy + 4 * sp

        # 小節線：M1 (密集 C5-C5-D5-C5-E5) | M2 (♮G5, G5 四分, 帶彎勾 G5 八分) | M3 (G5, G5 + 下方漸弱記號 >)
        for bx in (560, 940, 1240):
            cv2.line(flat, (bx, y_top), (bx, y_bot), (25, 25, 25), 2)

        # M1: C5(1/8) - C5(1/16) - D5(1/16) - C5(1/8) - E5(1/8)
        c5_y = int(y_bot - 2.5 * sp)
        d5_y = int(y_bot - 3.0 * sp)
        e5_y = int(y_bot - 3.5 * sp)
        m1_pts = [(220, c5_y), (268, c5_y), (294, d5_y), (322, c5_y), (375, e5_y)]
        beam_y = y_bot + sp
        for nx, ny in m1_pts:
            cv2.ellipse(flat, (nx, ny), (8, 6), -18, 0, 360, (15, 15, 15), -1)
            cv2.line(flat, (nx - 7, ny), (nx - 7, beam_y), (15, 15, 15), 2)
        cv2.rectangle(flat, (213, beam_y - 4), (295, beam_y), (15, 15, 15), -1)
        cv2.rectangle(flat, (261, beam_y - 11), (295, beam_y - 7), (15, 15, 15), -1)
        cv2.rectangle(flat, (315, beam_y - 4), (369, beam_y), (15, 15, 15), -1)

        # M2: G5 四分音符 + 帶右側弧形符尾彎勾的單顆 G5 八分音符（驗證不漏檢且彎勾不變成 (C5)！）
        g5_y = int(y_top - 0.5 * sp)
        cv2.ellipse(flat, (680, g5_y), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(flat, (673, g5_y), (673, g5_y + 3 * sp + 4), (15, 15, 15), 2)

        fx_n = 830
        cv2.ellipse(flat, (fx_n, g5_y), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(flat, (fx_n - 7, g5_y), (fx_n - 7, g5_y + int(3.4 * sp)), (15, 15, 15), 2)
        # 繪製朝上彎回的八分音符符尾勾
        pts_flag = np.array(
            [[fx_n - 7, g5_y + int(3.4 * sp)], [fx_n + 4, g5_y + int(2.6 * sp)], [fx_n + 7, g5_y + int(1.6 * sp)]],
            dtype=np.int32,
        )
        cv2.polylines(flat, [pts_flag], False, (15, 15, 15), 3)

        # M3: 兩顆 G5 四分音符 + 五線譜下方 1*sp 處長漸弱記號 >
        for nx in (1030, 1140):
            cv2.ellipse(flat, (nx, g5_y), (8, 6), -18, 0, 360, (15, 15, 15), -1)
            cv2.line(flat, (nx - 7, g5_y), (nx - 7, g5_y + 3 * sp + 4), (15, 15, 15), 2)
        cv2.line(flat, (990, y_bot + sp - 3), (1190, y_bot + sp + 3), (20, 20, 20), 2)
        cv2.line(flat, (990, y_bot + sp + 9), (1190, y_bot + sp + 3), (20, 20, 20), 2)

        # 模擬真實書頁右半明顯上彎（左半平緩、右半上翹 18px > 1.2*sp！）
        x_grid = np.arange(w, dtype=np.float32)[None, :]
        y_grid = np.arange(h, dtype=np.float32)[:, None]
        curve_shift = 18.0 * np.clip((x_grid - w * 0.40) / (w * 0.60), 0.0, 1.0) ** 1.6
        curved_img = cv2.remap(
            flat,
            np.broadcast_to(x_grid, (h, w)),
            y_grid + curve_shift,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        res = self.analyzer.analyze_pages([curved_img], beats_per_measure=2.0)
        self.assertTrue(res.has_staves)
        st0 = res.pages[0].staves[0]
        self.assertEqual([n.note_name for n in st0.measures[0].notes], ["C5", "C5", "D5", "C5", "E5"])
        self.assertEqual([n.note_name for n in st0.measures[1].notes], ["G5", "G5"])
        self.assertIn("八分音符", st0.measures[1].notes[1].note_type)
        self.assertEqual([n.note_name for n in st0.measures[2].notes], ["G5", "G5"])

    def test_grace_minibeam_interbeam_hollow_sharp_and_barline_slur(self):
        """驗證 OMR v9 五項精修：雙倚音短連槓不誤判 G5、16 分雙連槓白縫不誤判空心音、譜線上斜線倚音補救、#G4 不誤拆 F4+G4、小節線圓滑線交叉不誤判 (F5)"""
        w, h = 1520, 260
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        sp = 15
        y_top = 100
        y_bot = y_top + 4 * sp
        for i in range(5):
            y = y_top + i * sp
            cv2.line(img, (40, y), (w - 40, y), (25, 25, 25), 2)
        # 左側高音譜號區域
        cv2.ellipse(img, (75, y_top + 2 * sp), (14, 24), 0, 0, 360, (25, 25, 25), 3)
        cv2.line(img, (75, y_top - 18), (75, y_bot + 18), (25, 25, 25), 3)

        # 5 個小節分隔線
        for bx in (420, 680, 940, 1220, 1460):
            cv2.line(img, (bx, y_top), (bx, y_bot), (25, 25, 25), 2)

        # M1: 雙倚音 (C5)-(D5) 附上方雙連槓 + 主音 E5（驗證不誤判 G5 且保留雙倚音）
        c5_y = int(y_bot - 2.5 * sp)
        d5_y = int(y_bot - 3.0 * sp)
        e5_y = int(y_bot - 3.5 * sp)
        cv2.ellipse(img, (215, c5_y), (5, 4), -18, 0, 360, (15, 15, 15), -1)
        cv2.ellipse(img, (238, d5_y), (5, 4), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (219, c5_y), (219, y_top - 4), (15, 15, 15), 2)
        cv2.line(img, (242, d5_y), (242, y_top - 4), (15, 15, 15), 2)
        cv2.rectangle(img, (219, y_top - 7), (243, y_top - 3), (15, 15, 15), -1)
        cv2.rectangle(img, (219, y_top - 1), (243, y_top + 2), (15, 15, 15), -1)
        cv2.ellipse(img, (285, e5_y), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (277, e5_y), (277, y_bot + 8), (15, 15, 15), 2)

        # M2: 兩顆 16 分音符 G5 + F5（下方平行雙連槓間留白縫，驗證絕不誤生第 3 顆空心音符！）
        g5_y = int(y_top - 0.5 * sp)
        f5_y = y_top
        b1_y = y_top + int(2.7 * sp)
        b2_y = b1_y + 8
        for nx, ny in ((515, g5_y), (575, f5_y)):
            cv2.ellipse(img, (nx, ny), (8, 6), -18, 0, 360, (15, 15, 15), -1)
            cv2.line(img, (nx - 7, ny), (nx - 7, b2_y), (15, 15, 15), 2)
        cv2.rectangle(img, (508, b1_y - 3), (569, b1_y + 1), (15, 15, 15), -1)
        cv2.rectangle(img, (508, b2_y - 2), (569, b2_y + 2), (15, 15, 15), -1)

        # M3: 開頭小型斜線裝飾音 (F5 譜線上) + 主音 E5（驗證小型斜線倚音能被補救抓出）
        gx = 745
        cv2.ellipse(img, (gx, f5_y + 2), (4, 3), -15, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (gx + 4, f5_y + 2), (gx + 4, f5_y - 26), (15, 15, 15), 1)
        cv2.line(img, (gx + 4, f5_y - 26), (gx + 13, f5_y - 16), (15, 15, 15), 2)
        cv2.line(img, (gx - 1, f5_y - 12), (gx + 11, f5_y - 22), (15, 15, 15), 2)
        cv2.ellipse(img, (815, e5_y), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (807, e5_y), (807, y_bot + 8), (15, 15, 15), 2)

        # M4: 大型升記號 ♯ + G4（驗證 ♯ 不誤判為 F4，且 G4 正確升半音為 G#4）+ 跨小節線圓滑線穿過頂線 F5
        g4_y = int(y_bot - 1.0 * sp)
        sx = 1088
        cv2.line(img, (sx - 4, g4_y - 22), (sx - 4, g4_y + 21), (15, 15, 15), 2)
        cv2.line(img, (sx + 4, g4_y - 23), (sx + 4, g4_y + 20), (15, 15, 15), 2)
        cv2.line(img, (sx - 9, g4_y - 4), (sx + 9, g4_y - 7), (15, 15, 15), 4)
        cv2.line(img, (sx - 9, g4_y + 6), (sx + 9, g4_y + 3), (15, 15, 15), 4)
        cv2.ellipse(img, (1112, g4_y), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (1119, g4_y), (1119, g4_y - 3 * sp), (15, 15, 15), 2)
        # 在 M4/M5 小節線 (x=1220) 頂端 F5 處畫一道跨線圓滑線弧
        cv2.ellipse(img, (1216, y_top + 22), (34, 24), 0, 210, 330, (15, 15, 15), 2)

        # M5: 單顆 A4 四分音符（驗證小節線處絕不出現假裝飾音 (F5)）
        a4_y = int(y_bot - 1.5 * sp)
        cv2.ellipse(img, (1330, a4_y), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (1337, a4_y), (1337, a4_y - 3 * sp), (15, 15, 15), 2)

        res = self.analyzer.analyze_pages([img], beats_per_measure=2.0)
        self.assertTrue(res.has_staves)
        st0 = res.pages[0].staves[0]
        self.assertGreaterEqual(len(st0.measures), 5)

        # 1. M1: (C5), (D5), E5 — 不可有假 G5，雙倚音皆須保留
        m1_notes = st0.measures[0].notes
        self.assertEqual([n.note_name for n in m1_notes], ["(C5)", "(D5)", "E5"])
        self.assertTrue(m1_notes[0].is_grace and m1_notes[1].is_grace and not m1_notes[2].is_grace)

        # 2. M2: 僅 2 顆音 (G5, F5)，絕不可把雙連槓白縫誤判為第 3 顆空心音符
        m2_notes = st0.measures[1].notes
        self.assertEqual(len(m2_notes), 2)
        self.assertEqual([n.note_name for n in m2_notes], ["G5", "F5"])

        # 3. M3: 開頭小型斜線裝飾音必被抓出，後接主音 E5
        m3_notes = st0.measures[2].notes
        self.assertEqual(len(m3_notes), 2)
        self.assertTrue(m3_notes[0].is_grace)
        self.assertEqual(m3_notes[1].note_name, "E5")

        # 4. M4: 僅 1 顆 G#4 (MIDI 68)，升記號絕不可誤判為 F4
        m4_notes = st0.measures[3].notes
        self.assertEqual(len(m4_notes), 1)
        self.assertEqual(m4_notes[0].note_name, "G#4")
        self.assertEqual(m4_notes[0].midi_note, 68)

        # 5. M5: 僅 1 顆 A4，小節線圓滑線交叉處絕不可誤判為 (F5)
        m5_notes = st0.measures[4].notes
        self.assertEqual([n.note_name for n in m5_notes], ["A4"])


if __name__ == "__main__":
    unittest.main()
