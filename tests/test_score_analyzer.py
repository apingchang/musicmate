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

        # 3. 測試開啟「顯示偵測標記」渲染無誤
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
