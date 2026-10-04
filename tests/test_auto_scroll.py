"""自動翻譜與樂譜載入測試"""

import unittest
import tempfile
import os
import fitz
from PIL import Image
from PyQt6.QtWidgets import QApplication

# 確保在無顯示器環境下建立 headless QApplication
os.environ["QT_QPA_PLATFORM"] = "offscreen"
app = QApplication.instance()
if not app:
    app = QApplication([])

from src.ui.auto_scroll_page import AutoScrollPage


class TestAutoScrollPage(unittest.TestCase):
    def setUp(self):
        self.page = AutoScrollPage()
        self.temp_files = []

    def tearDown(self):
        for f in self.temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def _create_dummy_pdf(self, pages=2):
        doc = fitz.open()
        for i in range(pages):
            p = doc.new_page(width=500, height=700)
            p.draw_rect(fitz.Rect(20, 20, 480, 680), color=(0, 0, 0))
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(path)
        doc.close()
        self.temp_files.append(path)
        return path

    def _create_dummy_image(self, ext=".png"):
        fd, path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        img = Image.new("RGB", (600, 800), color=(255, 255, 255))
        img.save(path)
        self.temp_files.append(path)
        return path

    def test_load_pdf(self):
        pdf_path = self._create_dummy_pdf(pages=3)
        res = self.page.load_file(pdf_path)
        self.assertTrue(res)
        self.assertEqual(self.page._page_count, 3)
        self.assertEqual(len(self.page._raw_pixmaps), 3)
        self.assertIn("3 頁", self.page.page_label.text())

    def test_load_image(self):
        img_path = self._create_dummy_image(ext=".jpg")
        res = self.page.load_file(img_path)
        self.assertTrue(res)
        self.assertEqual(self.page._page_count, 1)
        self.assertEqual(len(self.page._raw_pixmaps), 1)
        self.assertIn("1 頁", self.page.page_label.text())

    def test_play_and_speed(self):
        pdf_path = self._create_dummy_pdf(pages=2)
        self.page.load_file(pdf_path)
        
        # 預設非播放狀態
        self.assertFalse(self.page._is_playing)
        
        # 開始播放
        self.page._on_play_toggle()
        self.assertTrue(self.page._is_playing)
        self.assertTrue(self.page._scroll_timer.isActive())
        
        # 調整速度
        self.page._on_speed_change(20)  # 2.0x
        self.assertEqual(self.page._scroll_speed, 2.0)
        self.assertEqual(self.page.speed_value_label.text(), "2.0x")
        
        # 暫停播放
        self.page._on_play_toggle()
        self.assertFalse(self.page._is_playing)
        self.assertFalse(self.page._scroll_timer.isActive())

    def test_zoom_adjust(self):
        img_path = self._create_dummy_image(ext=".png")
        self.page.load_file(img_path)
        init_zoom = self.page._zoom_factor
        self.page._adjust_zoom(0.2)
        self.assertAlmostEqual(self.page._zoom_factor, init_zoom + 0.2, places=1)


if __name__ == "__main__":
    unittest.main()
