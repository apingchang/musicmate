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
        for timbre in ("鋼琴", "薩克斯風", "長笛"):
            synth.timbre = timbre
            pcm = synth.synthesize_wave(60, 0.25)
            self.assertEqual(pcm.dtype, np.int16)
            self.assertGreater(len(pcm), 4000)
            self.assertGreater(int(np.max(np.abs(pcm))), 1000)

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


if __name__ == "__main__":
    unittest.main()
