"""自動翻譜頁 - 支援 PDF 與拍照圖片 (JPG/PNG/WEBP) 樂譜載入與平滑自動滾動播放"""

import os
import fitz  # PyMuPDF
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent, QDragEnterEvent, QDropEvent


class ScoreViewWidget(QWidget):
    """樂譜多頁連續排列元件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(15)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.page_labels = []

    def clear_pages(self):
        """清除所有頁面"""
        for lbl in self.page_labels:
            self.layout.removeWidget(lbl)
            lbl.deleteLater()
        self.page_labels.clear()

    def set_pages(self, pixmaps: list[QPixmap]):
        """加入所有頁面圖片"""
        self.clear_pages()
        for i, pm in enumerate(pixmaps):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setPixmap(pm)
            lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            lbl.setStyleSheet(
                "QLabel { background-color: white; border: 1px solid #dcdcdc; "
                "border-radius: 4px; }"
            )
            self.layout.addWidget(lbl)
            self.page_labels.append(lbl)


class AutoScrollPage(QWidget):
    """自動翻譜頁面"""

    def __init__(self):
        super().__init__()
        self._doc = None
        self._page_count = 0
        self._current_page = 0
        self._is_playing = False
        self._scroll_speed = 1.0  # 0.5x ~ 3.0x
        self._zoom_factor = 1.0   # 縮放比例 0.5 ~ 2.0
        self._raw_pixmaps = []    # 原始高品質 QPixmap 暫存
        self._scroll_subpixel = 0.0  # 亞像素浮點滾動累積

        # 滾動定時器（約 30 FPS）
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setInterval(33)  # 33 ms ≈ 30 FPS
        self._scroll_timer.timeout.connect(self._on_scroll_tick)

        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 頂部控制工具列
        toolbar = QFrame()
        toolbar.setStyleSheet(
            "QFrame { background-color: #f8f9fa; border-bottom: 1px solid #e2e8f0; padding: 6px; }"
        )
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(12, 6, 12, 6)
        tb_layout.setSpacing(10)

        self.open_btn = QPushButton("📂 開啟樂譜")
        self.open_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 14px; background-color: #2b579a; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1e3f73; }"
        )
        self.open_btn.clicked.connect(self._on_open)
        tb_layout.addWidget(self.open_btn)

        self.play_btn = QPushButton("▶ 播放 (Space)")
        self.play_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 16px; background-color: #107c41; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0c5e31; }"
        )
        self.play_btn.clicked.connect(self._on_play_toggle)
        tb_layout.addWidget(self.play_btn)

        tb_layout.addSpacing(15)

        # 滾動速度設定
        tb_layout.addWidget(QLabel("滾動速度："))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(5)   # 0.5x
        self.speed_slider.setMaximum(30)  # 3.0x
        self.speed_slider.setValue(10)    # 1.0x
        self.speed_slider.setFixedWidth(120)
        self.speed_slider.valueChanged.connect(self._on_speed_change)
        tb_layout.addWidget(self.speed_slider)

        self.speed_value_label = QLabel("1.0x")
        self.speed_value_label.setFixedWidth(35)
        tb_layout.addWidget(self.speed_value_label)

        tb_layout.addSpacing(15)

        # 縮放設定
        zoom_out_btn = QPushButton("🔍-")
        zoom_out_btn.setFixedSize(32, 28)
        zoom_out_btn.clicked.connect(lambda: self._adjust_zoom(-0.1))
        tb_layout.addWidget(zoom_out_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(45)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tb_layout.addWidget(self.zoom_label)

        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setFixedSize(32, 28)
        zoom_in_btn.clicked.connect(lambda: self._adjust_zoom(0.1))
        tb_layout.addWidget(zoom_in_btn)

        fit_width_btn = QPushButton("適應寬度")
        fit_width_btn.clicked.connect(self._fit_to_width)
        tb_layout.addWidget(fit_width_btn)

        tb_layout.addStretch()

        # 頁數與快速跳頁
        self.prev_btn = QPushButton("◀ 上一頁")
        self.prev_btn.clicked.connect(self._on_prev_page)
        tb_layout.addWidget(self.prev_btn)

        self.page_label = QLabel("第 0 / 0 頁")
        self.page_label.setStyleSheet("font-weight: bold; color: #333;")
        tb_layout.addWidget(self.page_label)

        self.next_btn = QPushButton("下一頁 ▶")
        self.next_btn.clicked.connect(self._on_next_page)
        tb_layout.addWidget(self.next_btn)

        # 重置頂端按鈕
        self.top_btn = QPushButton("⏫ 回頂端")
        self.top_btn.clicked.connect(self._scroll_to_top)
        tb_layout.addWidget(self.top_btn)

        main_layout.addWidget(toolbar)

        # 樂譜捲動檢視區域 (QScrollArea)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setStyleSheet(
            "QScrollArea { background-color: #525659; border: none; }"
        )
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)

        # 內部容器
        self.score_view = ScoreViewWidget()
        self.scroll_area.setWidget(self.score_view)

        # 空白初始提示
        self.placeholder_label = QLabel(
            "<h3>🎼 尚未載入樂譜</h3>"
            "<p>支援 <b>PDF 樂譜</b> 與 <b>手機拍照圖片 (JPG / PNG / WEBP)</b></p>"
            "<p>點擊上方「<b>📂 開啟樂譜</b>」或直接將檔案<b>拖曳至此處</b></p>"
        )
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet(
            "QLabel { color: #d0d0d0; font-size: 15px; padding: 60px; line-height: 160%; }"
        )
        self.score_view.layout.addWidget(self.placeholder_label)

        main_layout.addWidget(self.scroll_area)

        # 底部狀態與快捷鍵指引列
        self.status_bar = QFrame()
        self.status_bar.setStyleSheet(
            "QFrame { background-color: #f1f3f5; border-top: 1px solid #ced4da; padding: 4px 12px; }"
        )
        sb_layout = QHBoxLayout(self.status_bar)
        sb_layout.setContentsMargins(0, 0, 0, 0)

        self.status_text = QLabel("準備就緒。支援拖曳 PDF / 圖檔。")
        self.status_text.setStyleSheet("color: #495057; font-size: 12px;")
        sb_layout.addWidget(self.status_text)

        sb_layout.addStretch()

        shortcut_hint = QLabel("快捷鍵：[Space] 播放/暫停 | [PgUp/PgDn] 翻頁 | [Home] 回頂端 | [F11] 全螢幕")
        shortcut_hint.setStyleSheet("color: #6c757d; font-size: 12px;")
        sb_layout.addWidget(shortcut_hint)

        main_layout.addWidget(self.status_bar)

    def load_file(self, file_path: str):
        """載入 PDF 或單一/多張圖片檔案"""
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "錯誤", f"找不到檔案：{file_path}")
            return False

        ext = os.path.splitext(file_path)[1].lower()
        try:
            self._stop_playing()
            self._raw_pixmaps.clear()
            self._doc = None

            if ext == ".pdf":
                doc = fitz.open(file_path)
            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
                # 使用 PyMuPDF 將圖片載入為單頁 PDF
                img_doc = fitz.open(file_path)
                pdf_bytes = img_doc.convert_to_pdf()
                doc = fitz.open("pdf", pdf_bytes)
                img_doc.close()
            else:
                QMessageBox.warning(self, "不支援的格式", f"未支援的樂譜檔案格式：{ext}\n請選擇 PDF 或圖片檔。")
                return False

            self._doc = doc
            self._page_count = doc.page_count
            self._current_page = 0

            # 預設自適應寬度
            self._render_all_pages()
            self._fit_to_width()

            file_name = os.path.basename(file_path)
            self.status_text.setText(f"已載入樂譜：{file_name}（共 {self._page_count} 頁）")
            self.page_label.setText(f"第 1 / {self._page_count} 頁")
            return True

        except Exception as e:
            QMessageBox.critical(self, "開啟失敗", f"無法解析樂譜檔案：\n{str(e)}")
            return False

    def _render_all_pages(self):
        """將 PDF 每頁渲染為 QPixmap 備用"""
        if not self._doc:
            return

        self._raw_pixmaps.clear()
        # 採用 150 DPI 高解析度渲染樂譜
        zoom_matrix = fitz.Matrix(150 / 72, 150 / 72)
        for page_idx in range(self._page_count):
            page = self._doc[page_idx]
            pix = page.get_pixmap(matrix=zoom_matrix)
            qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qimg.copy())  # copy 確保記憶體安全
            self._raw_pixmaps.append(pm)

    def _update_view_scale(self):
        """依照目前 zoom_factor 縮放並呈現在 QScrollArea 容器"""
        if not self._raw_pixmaps:
            return

        # 隱藏 placeholder
        if self.placeholder_label:
            self.placeholder_label.hide()

        scaled_pixmaps = []
        for pm in self._raw_pixmaps:
            target_w = max(200, int(pm.width() * self._zoom_factor))
            scaled_pm = pm.scaledToWidth(target_w, Qt.TransformationMode.SmoothTransformation)
            scaled_pixmaps.append(scaled_pm)

        self.score_view.set_pages(scaled_pixmaps)
        self.zoom_label.setText(f"{int(self._zoom_factor * 100)}%")

    def _fit_to_width(self):
        """自動調整縮放以適應螢幕可視寬度"""
        if not self._raw_pixmaps:
            return

        viewport_w = self.scroll_area.viewport().width() - 40  # 扣除 margins
        if viewport_w <= 100:
            viewport_w = 800

        base_w = self._raw_pixmaps[0].width()
        if base_w > 0:
            factor = viewport_w / base_w
            self._zoom_factor = max(0.2, min(3.0, factor))
            self._update_view_scale()

    def _adjust_zoom(self, delta: float):
        """微調縮放比例"""
        if not self._raw_pixmaps:
            return
        new_factor = round(self._zoom_factor + delta, 1)
        self._zoom_factor = max(0.3, min(3.0, new_factor))
        self._update_view_scale()

    def _on_open(self):
        """開啟檔案對話框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟樂譜檔案", "",
            "樂譜與圖片 (*.pdf *.jpg *.jpeg *.png *.webp *.bmp);;PDF 樂譜 (*.pdf);;圖片 (*.jpg *.jpeg *.png *.webp);;所有檔案 (*)"
        )
        if file_path:
            self.load_file(file_path)

    def _on_play_toggle(self):
        """切換播放/暫停滾動"""
        if not self._raw_pixmaps:
            return
        if self._is_playing:
            self._stop_playing()
        else:
            self._start_playing()

    def _start_playing(self):
        self._is_playing = True
        self.play_btn.setText("⏸ 暫停 (Space)")
        self.play_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 16px; background-color: #d83b01; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #a82e00; }"
        )
        self._scroll_subpixel = 0.0
        self._scroll_timer.start()

    def _stop_playing(self):
        self._is_playing = False
        self._scroll_timer.stop()
        self.play_btn.setText("▶ 播放 (Space)")
        self.play_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 16px; background-color: #107c41; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0c5e31; }"
        )

    def _on_speed_change(self, value):
        self._scroll_speed = value / 10.0
        self.speed_value_label.setText(f"{self._scroll_speed:.1f}x")

    def _on_scroll_tick(self):
        """定時器每 33ms 觸發滾動"""
        if not self._is_playing:
            return

        vbar = self.scroll_area.verticalScrollBar()
        if vbar.value() >= vbar.maximum():
            # 到底部自動停止
            self._stop_playing()
            self.status_text.setText("樂譜已滾動至結尾。")
            return

        # 基礎速度：1.0x 約 1.0 px 每 tick (≈ 30 px/秒)
        step = self._scroll_speed * 1.0
        self._scroll_subpixel += step
        int_step = int(self._scroll_subpixel)
        if int_step > 0:
            self._scroll_subpixel -= int_step
            vbar.setValue(vbar.value() + int_step)

    def _on_scroll_changed(self, value):
        """根據當前滾動位置動態估算當前顯示頁面"""
        if not self.score_view.page_labels or self._page_count == 0:
            return

        # 檢測視窗中心點對應哪一頁
        viewport_mid = value + (self.scroll_area.viewport().height() // 2)

        cur_idx = 0
        for i, lbl in enumerate(self.score_view.page_labels):
            lbl_top = lbl.y()
            lbl_bottom = lbl_top + lbl.height()
            if lbl_top <= viewport_mid <= lbl_bottom:
                cur_idx = i
                break
            elif lbl_top > viewport_mid:
                cur_idx = max(0, i - 1)
                break
        else:
            cur_idx = self._page_count - 1

        self._current_page = cur_idx
        self.page_label.setText(f"第 {cur_idx + 1} / {self._page_count} 頁")

    def _on_prev_page(self):
        """手動跳轉至上一頁"""
        if not self.score_view.page_labels:
            return
        target_page = max(0, self._current_page - 1)
        self._scroll_to_page(target_page)

    def _on_next_page(self):
        """手動跳轉至下一頁"""
        if not self.score_view.page_labels:
            return
        target_page = min(self._page_count - 1, self._current_page + 1)
        self._scroll_to_page(target_page)

    def _scroll_to_page(self, page_index: int):
        """將捲軸滾動到指定頁面開頭"""
        if 0 <= page_index < len(self.score_view.page_labels):
            target_y = self.score_view.page_labels[page_index].y()
            self.scroll_area.verticalScrollBar().setValue(target_y)
            self._current_page = page_index
            self.page_label.setText(f"第 {page_index + 1} / {self._page_count} 頁")

    def _scroll_to_top(self):
        """回到最頂端"""
        self.scroll_area.verticalScrollBar().setValue(0)
        self._current_page = 0
        if self._page_count > 0:
            self.page_label.setText(f"第 1 / {self._page_count} 頁")

    # --- 拖放事件支援 ---
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path:
                self.load_file(file_path)
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    # --- 鍵盤快捷鍵 ---
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self._on_play_toggle()
            event.accept()
        elif key in (Qt.Key.Key_PageUp, Qt.Key.Key_Left):
            self._on_prev_page()
            event.accept()
        elif key in (Qt.Key.Key_PageDown, Qt.Key.Key_Right):
            self._on_next_page()
            event.accept()
        elif key == Qt.Key.Key_Home:
            self._scroll_to_top()
            event.accept()
        elif key == Qt.Key.Key_F11 or key == Qt.Key.Key_F:
            # 全螢幕切換
            win = self.window()
            if win:
                if win.isFullScreen():
                    win.showNormal()
                else:
                    win.showFullScreen()
            event.accept()
        else:
            super().keyPressEvent(event)