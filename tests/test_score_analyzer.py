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

        # 驗證時間軸單調遞增
        for i in range(1, len(res.events)):
            self.assertGreaterEqual(res.events[i].beat_offset, res.events[i - 1].beat_offset)

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
