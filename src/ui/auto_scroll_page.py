"""自動翻譜頁 - 顯示樂譜 PDF/圖片並自動滾動"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog,
    QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
import fitz  # PyMuPDF


class AutoScrollPage(QWidget):
    """自動翻譜頁面"""

    def __init__(self):
        super().__init__()
        self._pdf_document = None
        self._current_page = 0
        self._is_playing = False
        self._scroll_speed = 1.0  # 0.5x ~ 3.0x
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 控制工具列
        toolbar = QHBoxLayout()
        self.open_btn = QPushButton("📂 開啟")
        self.open_btn.clicked.connect(self._on_open)
        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.clicked.connect(self._on_play_toggle)

        speed_label = QLabel("速度：")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(5)
        self.speed_slider.setMaximum(30)
        self.speed_slider.setValue(10)  # 1.0x
        self.speed_slider.setMaximumWidth(200)
        self.speed_slider.valueChanged.connect(self._on_speed_change)
        self.speed_value_label = QLabel("1.0x")

        prev_btn = QPushButton("◀ 上一頁")
        prev_btn.clicked.connect(self._on_prev_page)
        self.page_label = QLabel("第 0 / 0 頁")
        next_btn = QPushButton("下一頁 ▶")
        next_btn.clicked.connect(self._on_next_page)

        toolbar.addWidget(self.open_btn)
        toolbar.addWidget(self.play_btn)
        toolbar.addSpacing(20)
        toolbar.addWidget(speed_label)
        toolbar.addWidget(self.speed_slider)
        toolbar.addWidget(self.speed_value_label)
        toolbar.addStretch()
        toolbar.addWidget(prev_btn)
        toolbar.addWidget(self.page_label)
        toolbar.addWidget(next_btn)

        layout.addLayout(toolbar)

        # 樂譜顯示區（Placeholder）
        self.score_label = QLabel(
            "<br><br><br>"
            "尚未開啟樂譜<br>"
            "點擊「開啟」選擇 PDF 或圖片檔案<br>"
            "或將檔案拖曳至此"
        )
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; "
            "border: 1px solid #ccc; font-size: 16px; color: #666; }"
        )
        layout.addWidget(self.score_label)

        # 狀態列
        self.status_label = QLabel("快捷鍵：Space=播放  ◀▶=換頁  F=全螢幕")
        self.status_label.setStyleSheet("color: #888; font-size: 12px; padding: 4px;")
        layout.addWidget(self.status_label)

    def load_file(self, file_path: str):
        """載入樂譜檔案"""
        try:
            self._pdf_document = fitz.open(file_path)
            self._current_page = 0
            self._show_page(0)
            self.page_label.setText(
                f"第 {self._current_page + 1} / {self._pdf_document.page_count} 頁"
            )
            self.score_label.setText(
                f"已載入：{file_path.split('/')[-1]}\n"
                f"共 {self._pdf_document.page_count} 頁"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "錯誤", f"無法開啟檔案：\n{str(e)}"
            )

    def _on_open(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟樂譜", "",
            "樂譜檔案 (*.pdf);;圖片 (*.jpg *.png);;所有檔案 (*)"
        )
        if file_path:
            self.load_file(file_path)

    def _on_play_toggle(self):
        self._is_playing = not self._is_playing
        self.play_btn.setText("⏸ 暫停" if self._is_playing else "▶ 播放")

    def _on_speed_change(self, value):
        self._scroll_speed = value / 10.0
        self.speed_value_label.setText(f"{self._scroll_speed:.1f}x")

    def _on_prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._show_page(self._current_page)

    def _on_next_page(self):
        if self._pdf_document and self._current_page < self._pdf_document.page_count - 1:
            self._current_page += 1
            self._show_page(self._current_page)

    def _show_page(self, page_index: int):
        self.page_label.setText(
            f"第 {page_index + 1} / {self._pdf_document.page_count} 頁"
        )