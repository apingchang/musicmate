"""自動翻譜頁 - 支援 PDF 與拍照圖片 (JPG/PNG/WEBP) 樂譜載入、音符偵測與方案 A 演奏進度追蹤"""

import os
import time
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QSizePolicy,
    QSpinBox, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import (
    QImage, QPixmap, QKeyEvent, QDragEnterEvent, QDropEvent,
    QPainter, QColor, QPen, QBrush, QPolygonF, QMouseEvent
)

from src.core.score_analyzer import ScoreAnalyzer, ScoreAnalysis, StaffInfo, NoteEvent
from src.core.note_synth import NoteSynthesizer


class ScorePageWidget(QWidget):
    """單頁樂譜顯示元件，內建方案 A（整行淡色高亮 + 半透明垂直光條 + 頂部紅標 ▼ + 音符光圈）"""

    clicked = pyqtSignal(int, float, float)  # (page_index, x_norm, y_norm)

    def __init__(self, page_index: int, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.page_index = page_index
        self._pixmap = pixmap
        self.setFixedSize(pixmap.size())
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # 演奏游標狀態（歸一化座標 0.0~1.0）
        self._active_staff: Optional[StaffInfo] = None
        self._cursor_x_norm: Optional[float] = None
        self._note_x_norm: Optional[float] = None
        self._note_y_norm: Optional[float] = None
        self._is_rest: bool = False

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.setFixedSize(pixmap.size())
        self.update()

    def set_playhead(
        self,
        staff: Optional[StaffInfo],
        cursor_x_norm: Optional[float],
        note_x_norm: Optional[float] = None,
        note_y_norm: Optional[float] = None,
        is_rest: bool = False,
    ):
        """更新本頁的演奏游標位置；若 staff 為 None 則清除本頁游標"""
        self._active_staff = staff
        self._cursor_x_norm = cursor_x_norm
        self._note_x_norm = note_x_norm
        self._note_y_norm = note_y_norm
        self._is_rest = is_rest
        self.update()

    def clear_playhead(self):
        if self._active_staff is not None or self._cursor_x_norm is not None:
            self._active_staff = None
            self._cursor_x_norm = None
            self._note_x_norm = None
            self._note_y_norm = None
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.width() > 0 and self.height() > 0:
            x_norm = event.position().x() / float(self.width())
            y_norm = event.position().y() / float(self.height())
            self.clicked.emit(self.page_index, x_norm, y_norm)
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. 繪製樂譜底圖與細框
        painter.drawPixmap(0, 0, self._pixmap)
        painter.setPen(QPen(QColor(210, 214, 220), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        # 2. 方案 A：若本頁有作用中的五線譜與游標，繪製高亮與垂直光條
        if self._active_staff is None or self._cursor_x_norm is None:
            return

        w = float(self.width())
        h = float(self.height())
        st = self._active_staff

        sp_px = max(5.0, st.spacing_norm * h)
        pad_y = sp_px * 1.6
        band_top = max(4.0, st.y_top_norm * h - pad_y)
        band_bot = min(h - 4.0, st.y_bot_norm * h + pad_y)
        band_h = max(16.0, band_bot - band_top)

        staff_left = max(4.0, st.x_left_norm * w - sp_px * 0.8)
        staff_right = min(w - 4.0, st.x_right_norm * w + sp_px * 0.8)
        staff_w = max(20.0, staff_right - staff_left)

        # 2a. 整行五線譜暖黃淡色底光（不遮擋音符與符號）
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 224, 102, 58)))
        painter.drawRoundedRect(QRectF(staff_left, band_top, staff_w, band_h), 6.0, 6.0)

        # 左側加上一條極細暖色導引邊線，換行一眼辨識
        painter.setBrush(QBrush(QColor(235, 160, 20, 180)))
        painter.drawRoundedRect(QRectF(staff_left, band_top, 4.0, band_h), 2.0, 2.0)

        # 2b. 半透明垂直光條 + 紅色中心細線
        cx = self._cursor_x_norm * w
        bar_w = max(10.0, sp_px * 1.35)
        painter.setBrush(QBrush(QColor(231, 76, 60, 46)))
        painter.drawRoundedRect(QRectF(cx - bar_w / 2.0, band_top, bar_w, band_h), 4.0, 4.0)

        painter.setPen(QPen(QColor(220, 38, 38, 210), 2.0))
        painter.drawLine(QPointF(cx, band_top), QPointF(cx, band_bot))

        # 2c. 頂部紅色倒三角指標 ▼
        tri_half_w = max(6.0, sp_px * 0.72)
        tri_h = max(8.0, sp_px * 0.95)
        tri_tip_y = band_top + 2.0
        tri_top_y = max(2.0, tri_tip_y - tri_h)
        triangle = QPolygonF([
            QPointF(cx - tri_half_w, tri_top_y),
            QPointF(cx + tri_half_w, tri_top_y),
            QPointF(cx, tri_tip_y),
        ])
        painter.setPen(QPen(QColor(180, 20, 20, 230), 1.0))
        painter.setBrush(QBrush(QColor(231, 45, 45, 235)))
        painter.drawPolygon(triangle)

        # 2d. 目前播放音符柔和光圈（非休止符時標示於符頭位置）
        if not self._is_rest and self._note_x_norm is not None and self._note_y_norm is not None:
            nx = self._note_x_norm * w
            ny = self._note_y_norm * h
            r = max(6.0, sp_px * 0.85)
            painter.setPen(QPen(QColor(220, 38, 38, 215), 2.0))
            painter.setBrush(QBrush(QColor(239, 68, 68, 65)))
            painter.drawEllipse(QPointF(nx, ny), r, r)


class ScoreViewWidget(QWidget):
    """樂譜多頁連續排列元件"""

    page_clicked = pyqtSignal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(15)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.page_labels: List[ScorePageWidget] = []

    def clear_pages(self):
        """清除所有頁面"""
        for widget in self.page_labels:
            self.layout.removeWidget(widget)
            widget.deleteLater()
        self.page_labels.clear()

    def set_pages(self, pixmaps: List[QPixmap]):
        """加入所有頁面元件"""
        self.clear_pages()
        for i, pm in enumerate(pixmaps):
            page_w = ScorePageWidget(i, pm)
            page_w.clicked.connect(self.page_clicked.emit)
            self.layout.addWidget(page_w)
            self.page_labels.append(page_w)

    def update_playhead(
        self,
        page_index: int,
        staff: Optional[StaffInfo],
        cursor_x_norm: Optional[float],
        note_x_norm: Optional[float] = None,
        note_y_norm: Optional[float] = None,
        is_rest: bool = False,
    ):
        for i, page_w in enumerate(self.page_labels):
            if i == page_index and staff is not None and cursor_x_norm is not None:
                page_w.set_playhead(staff, cursor_x_norm, note_x_norm, note_y_norm, is_rest)
            else:
                page_w.clear_playhead()

    def clear_playhead(self):
        for page_w in self.page_labels:
            page_w.clear_playhead()


class AutoScrollPage(QWidget):
    """自動翻譜頁面（支援音符即時追蹤 + 方案 A 視覺游標 + 純捲動備援）"""

    TIME_SIGNATURES = {
        "2/4": 2.0,
        "3/4": 3.0,
        "4/4": 4.0,
        "6/8": 3.0,
    }

    def __init__(self):
        super().__init__()
        self._doc = None
        self._page_count = 0
        self._current_page = 0
        self._is_playing = False
        self._scroll_speed = 1.0      # 0.5x ~ 3.0x
        self._bpm = 100               # 預設演奏速度 (BPM)
        self._beats_per_measure = 4.0 # 預設 4/4 拍
        self._zoom_factor = 1.0       # 縮放比例 0.3 ~ 3.0
        self._raw_pixmaps: List[QPixmap] = []
        self._raw_rgb_pages: List[np.ndarray] = []
        self._scroll_subpixel = 0.0

        # 音符分析與演奏時間軸狀態
        self._analyzer = ScoreAnalyzer()
        self._analysis: Optional[ScoreAnalysis] = None
        self._synth = NoteSynthesizer()
        self._track_mode = "note"  # "note" (方案A音符追蹤) | "scroll" (純垂直捲動)
        self._current_beat = 0.0
        self._last_tick_time = 0.0
        self._last_scrolled_staff: Tuple[int, int] = (-1, -1)
        self._last_played_event_idx: int = -1

        # 滾動 / 游標定時器（約 30 FPS）
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

        # 頂部雙列工具列（徹底避免視窗較窄時按鈕文字被截斷）
        toolbar = QFrame()
        toolbar.setStyleSheet(
            "QFrame { background-color: #f8f9fa; border-bottom: 1px solid #e2e8f0; }"
        )
        tb_vbox = QVBoxLayout(toolbar)
        tb_vbox.setContentsMargins(10, 6, 10, 6)
        tb_vbox.setSpacing(6)

        # --- 第一列：開檔、播放控制、BPM、拍號、速度倍率 ---
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        self.open_btn = QPushButton("📂 開啟樂譜")
        self.open_btn.setMinimumHeight(30)
        self.open_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 12px; background-color: #2b579a; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1e3f73; }"
        )
        self.open_btn.clicked.connect(self._on_open)
        row1.addWidget(self.open_btn)

        self.play_btn = QPushButton("▶ 播放 (Space)")
        self.play_btn.setMinimumHeight(30)
        self.play_btn.setMinimumWidth(132)
        self.play_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 14px; background-color: #107c41; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0c5e31; }"
        )
        self.play_btn.clicked.connect(self._on_play_toggle)
        row1.addWidget(self.play_btn)

        self.restart_btn = QPushButton("⏮ 重頭")
        self.restart_btn.setMinimumHeight(30)
        self.restart_btn.clicked.connect(self._on_restart)
        row1.addWidget(self.restart_btn)

        row1.addSpacing(8)

        # BPM 設定
        row1.addWidget(QLabel("BPM："))
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(30, 260)
        self.bpm_spin.setValue(self._bpm)
        self.bpm_spin.setFixedWidth(68)
        self.bpm_spin.setMinimumHeight(28)
        self.bpm_spin.valueChanged.connect(self._on_bpm_change)
        row1.addWidget(self.bpm_spin)

        # 拍號設定
        row1.addWidget(QLabel("拍號："))
        self.time_sig_combo = QComboBox()
        self.time_sig_combo.addItems(list(self.TIME_SIGNATURES.keys()))
        self.time_sig_combo.setCurrentText("4/4")
        self.time_sig_combo.setFixedWidth(64)
        self.time_sig_combo.setMinimumHeight(28)
        self.time_sig_combo.currentTextChanged.connect(self._on_time_sig_change)
        row1.addWidget(self.time_sig_combo)

        row1.addSpacing(8)

        # 速度倍率滑桿
        row1.addWidget(QLabel("速度倍率："))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(5)   # 0.5x
        self.speed_slider.setMaximum(30)  # 3.0x
        self.speed_slider.setValue(10)    # 1.0x
        self.speed_slider.setFixedWidth(110)
        self.speed_slider.valueChanged.connect(self._on_speed_change)
        row1.addWidget(self.speed_slider)

        self.speed_value_label = QLabel("1.0x")
        self.speed_value_label.setMinimumWidth(42)
        self.speed_value_label.setStyleSheet("font-weight: bold; color: #333;")
        row1.addWidget(self.speed_value_label)

        row1.addStretch()

        # 聲音開關與音色選擇
        self.sound_btn = QPushButton("🔊 聲音：開")
        self.sound_btn.setCheckable(True)
        self.sound_btn.setChecked(True)
        self.sound_btn.setMinimumHeight(28)
        self.sound_btn.setStyleSheet(
            "QPushButton { padding: 2px 10px; border-radius: 4px; font-weight: bold; "
            "background-color: #e6f4ea; color: #137333; border: 1px solid #a8dab5; }"
            "QPushButton:!checked { background-color: #f1f3f4; color: #5f6368; border: 1px solid #dadce0; }"
        )
        self.sound_btn.toggled.connect(self._on_sound_toggle)
        row1.addWidget(self.sound_btn)

        self.timbre_combo = QComboBox()
        self.timbre_combo.addItems(["鋼琴", "薩克斯風", "長笛"])
        self.timbre_combo.setMinimumWidth(84)
        self.timbre_combo.setMinimumHeight(28)
        self.timbre_combo.currentTextChanged.connect(self._on_timbre_change)
        row1.addWidget(self.timbre_combo)

        # 追蹤模式切換
        row1.addWidget(QLabel("模式："))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["🎯 音符追蹤", "📜 純垂直捲動"])
        self.mode_combo.setMinimumWidth(118)
        self.mode_combo.setMinimumHeight(28)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
        row1.addWidget(self.mode_combo)

        tb_vbox.addLayout(row1)

        # --- 第二列：縮放、目前小節顯示、頁數切換、回頂端 ---
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        zoom_out_btn = QPushButton("🔍-")
        zoom_out_btn.setFixedSize(36, 26)
        zoom_out_btn.clicked.connect(lambda: self._adjust_zoom(-0.1))
        row2.addWidget(zoom_out_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(48)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row2.addWidget(self.zoom_label)

        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setFixedSize(36, 26)
        zoom_in_btn.clicked.connect(lambda: self._adjust_zoom(0.1))
        row2.addWidget(zoom_in_btn)

        fit_width_btn = QPushButton("適應寬度")
        fit_width_btn.setMinimumHeight(26)
        fit_width_btn.clicked.connect(self._fit_to_width)
        row2.addWidget(fit_width_btn)

        row2.addSpacing(12)

        self.measure_label = QLabel("小節：-- / --")
        self.measure_label.setStyleSheet(
            "font-weight: bold; color: #b91c1c; background-color: #fef2f2; "
            "border: 1px solid #fecaca; border-radius: 4px; padding: 2px 10px;"
        )
        row2.addWidget(self.measure_label)

        row2.addStretch()

        self.prev_btn = QPushButton("◀ 上一頁")
        self.prev_btn.setMinimumHeight(26)
        self.prev_btn.clicked.connect(self._on_prev_page)
        row2.addWidget(self.prev_btn)

        self.page_label = QLabel("第 0 / 0 頁")
        self.page_label.setMinimumWidth(78)
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_label.setStyleSheet("font-weight: bold; color: #333;")
        row2.addWidget(self.page_label)

        self.next_btn = QPushButton("下一頁 ▶")
        self.next_btn.setMinimumHeight(26)
        self.next_btn.clicked.connect(self._on_next_page)
        row2.addWidget(self.next_btn)

        self.top_btn = QPushButton("⏫ 回頂端")
        self.top_btn.setMinimumHeight(26)
        self.top_btn.clicked.connect(self._scroll_to_top)
        row2.addWidget(self.top_btn)

        tb_vbox.addLayout(row2)
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
        self.score_view.page_clicked.connect(self._on_score_clicked)
        self.scroll_area.setWidget(self.score_view)

        # 空白初始提示
        self.placeholder_label = QLabel(
            "<h3>🎼 尚未載入樂譜</h3>"
            "<p>支援 <b>PDF 樂譜</b> 與 <b>手機拍照圖片 (JPG / PNG / WEBP)</b></p>"
            "<p>載入後自動偵測五線譜、小節與音符位置，播放時以 <b>整行高亮 + 垂直光條</b> 即時追蹤</p>"
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

        self.status_text = QLabel("準備就緒。支援拖曳 PDF / 圖檔，點擊樂譜任意音符可跳轉位置。")
        self.status_text.setStyleSheet("color: #495057; font-size: 12px;")
        sb_layout.addWidget(self.status_text)

        sb_layout.addStretch()

        shortcut_hint = QLabel("快捷鍵：[Space] 播放/暫停 | [點擊樂譜] 跳轉 | [PgUp/PgDn] 換頁 | [Home] 回頂端 | [F11] 全螢幕")
        shortcut_hint.setStyleSheet("color: #6c757d; font-size: 12px;")
        sb_layout.addWidget(shortcut_hint)

        main_layout.addWidget(self.status_bar)

    def load_file(self, file_path: str):
        """載入 PDF 或圖片檔案並自動執行五線譜與音符分析"""
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "錯誤", f"找不到檔案：{file_path}")
            return False

        ext = os.path.splitext(file_path)[1].lower()
        try:
            self._stop_playing()
            self._raw_pixmaps.clear()
            self._raw_rgb_pages.clear()
            self._analysis = None
            self._doc = None

            if ext == ".pdf":
                doc = fitz.open(file_path)
            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
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
            self._current_beat = 0.0
            self._last_scrolled_staff = (-1, -1)

            # 1. 高解析渲染所有頁面（同時保留 RGB ndarray 供 OpenCV 分析）
            self._render_all_pages()

            # 2. 執行 OpenCV 五線譜 / 小節 / 音符結構分析
            self._run_score_analysis()

            # 3. 預設自適應寬度呈現並顯示初始游標
            self._fit_to_width()

            file_name = os.path.basename(file_path)
            if self._analysis and self._analysis.has_staves:
                n_staves = sum(len(p.staves) for p in self._analysis.pages)
                n_events = len(self._analysis.events)
                self.status_text.setText(
                    f"已載入：{file_name}（{self._page_count} 頁 | 偵測到 {n_staves} 行五線譜、"
                    f"{self._analysis.total_measures} 小節、{n_events} 個音符位置，點擊樂譜可跳轉）"
                )
                self._sync_playhead_visual(scroll_to_staff=False)
            else:
                self.status_text.setText(
                    f"已載入：{file_name}（共 {self._page_count} 頁，未偵測到五線譜，已切換為純垂直捲動模式）"
                )
                self.measure_label.setText("小節：-- / --")

            self.page_label.setText(f"第 1 / {self._page_count} 頁")
            return True

        except Exception as e:
            QMessageBox.critical(self, "開啟失敗", f"無法解析樂譜檔案：\n{str(e)}")
            return False

    def _render_all_pages(self):
        """將每頁渲染為 QPixmap 與 RGB numpy array"""
        if not self._doc:
            return

        self._raw_pixmaps.clear()
        self._raw_rgb_pages.clear()
        zoom_matrix = fitz.Matrix(150 / 72, 150 / 72)
        for page_idx in range(self._page_count):
            page = self._doc[page_idx]
            pix = page.get_pixmap(matrix=zoom_matrix, alpha=False)
            qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qimg.copy())
            self._raw_pixmaps.append(pm)

            # 取得 RGB numpy array 供 ScoreAnalyzer 使用
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
            self._raw_rgb_pages.append(rgb)

    def _run_score_analysis(self):
        """呼叫 ScoreAnalyzer 分析所有頁面的五線譜與音符時間軸"""
        if not self._raw_rgb_pages:
            self._analysis = None
            return
        try:
            self._analysis = self._analyzer.analyze_pages(
                self._raw_rgb_pages,
                beats_per_measure=self._beats_per_measure,
            )
        except Exception:
            self._analysis = None

    def _update_view_scale(self):
        """依照目前 zoom_factor 縮放並呈現在 QScrollArea 容器"""
        if not self._raw_pixmaps:
            return

        if self.placeholder_label:
            self.placeholder_label.hide()

        scaled_pixmaps = []
        for pm in self._raw_pixmaps:
            target_w = max(200, int(pm.width() * self._zoom_factor))
            scaled_pm = pm.scaledToWidth(target_w, Qt.TransformationMode.SmoothTransformation)
            scaled_pixmaps.append(scaled_pm)

        self.score_view.set_pages(scaled_pixmaps)
        self.zoom_label.setText(f"{int(self._zoom_factor * 100)}%")

        # 縮放後重新繪製當前游標位置
        if self._use_note_tracking():
            self._sync_playhead_visual(scroll_to_staff=False)

    def _use_note_tracking(self) -> bool:
        """是否啟用方案 A 音符追蹤模式"""
        return (
            self._track_mode == "note"
            and self._analysis is not None
            and self._analysis.has_staves
            and len(self._analysis.events) > 0
        )

    def _fit_to_width(self):
        """自動調整縮放以適應螢幕可視寬度"""
        if not self._raw_pixmaps:
            return

        viewport_w = self.scroll_area.viewport().width() - 40
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
        """切換播放/暫停"""
        if not self._raw_pixmaps:
            return
        if self._is_playing:
            self._stop_playing()
        else:
            self._start_playing()

    def _start_playing(self):
        if self._use_note_tracking() and self._analysis:
            # 若已播完結尾，自動從頭開始
            if self._current_beat >= self._analysis.total_beats:
                self._current_beat = 0.0
                self._last_scrolled_staff = (-1, -1)

        self._is_playing = True
        self.play_btn.setText("⏸ 暫停 (Space)")
        self.play_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 14px; background-color: #d83b01; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #a82e00; }"
        )
        self._scroll_subpixel = 0.0
        self._last_played_event_idx = -1
        self._last_tick_time = time.perf_counter()
        self._scroll_timer.start()

    def _stop_playing(self):
        self._is_playing = False
        self._scroll_timer.stop()
        self._synth.stop_all()
        self._last_played_event_idx = -1
        self.play_btn.setText("▶ 播放 (Space)")
        self.play_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 14px; background-color: #107c41; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0c5e31; }"
        )

    def _on_sound_toggle(self, checked: bool):
        self._synth.enabled = checked
        self.sound_btn.setText("🔊 聲音：開" if checked else "🔇 聲音：關")

    def _on_timbre_change(self, text: str):
        self._synth.timbre = text

    def _on_restart(self):
        """回到第 1 小節並捲回頂端"""
        self._current_beat = 0.0
        self._last_scrolled_staff = (-1, -1)
        self._last_played_event_idx = -1
        self._synth.stop_all()
        self._scroll_to_top()
        if self._use_note_tracking():
            self._sync_playhead_visual(scroll_to_staff=True)

    def _on_bpm_change(self, value: int):
        self._bpm = value

    def _on_time_sig_change(self, text: str):
        beats = self.TIME_SIGNATURES.get(text, 4.0)
        if beats != self._beats_per_measure:
            self._beats_per_measure = beats
            if self._raw_rgb_pages:
                was_playing = self._is_playing
                self._stop_playing()
                self._run_score_analysis()
                self._current_beat = 0.0
                self._last_scrolled_staff = (-1, -1)
                if self._use_note_tracking():
                    self._sync_playhead_visual(scroll_to_staff=False)
                if was_playing:
                    self._start_playing()

    def _on_mode_change(self, index: int):
        self._track_mode = "note" if index == 0 else "scroll"
        if self._use_note_tracking():
            self._sync_playhead_visual(scroll_to_staff=False)
        else:
            self.score_view.clear_playhead()

    def _on_speed_change(self, value):
        self._scroll_speed = value / 10.0
        self.speed_value_label.setText(f"{self._scroll_speed:.1f}x")

    def _on_scroll_tick(self):
        """定時器每 33ms 觸發：依模式推進音符游標或垂直捲軸"""
        if not self._is_playing:
            return

        if self._use_note_tracking() and self._analysis:
            now = time.perf_counter()
            dt = max(0.0, min(0.25, now - self._last_tick_time))
            self._last_tick_time = now

            beats_per_sec = (self._bpm / 60.0) * self._scroll_speed
            self._current_beat += dt * beats_per_sec

            if self._current_beat >= self._analysis.total_beats:
                self._current_beat = self._analysis.total_beats
                self._sync_playhead_visual(scroll_to_staff=True)
                self._stop_playing()
                self.status_text.setText("樂譜已播放至最後一小節結尾。")
                return

            self._sync_playhead_visual(scroll_to_staff=True)
            return

        # 純垂直捲動模式（無五線譜或使用者手動切換）
        vbar = self.scroll_area.verticalScrollBar()
        if vbar.value() >= vbar.maximum():
            self._stop_playing()
            self.status_text.setText("樂譜已滾動至結尾。")
            return

        step = self._scroll_speed * 1.0
        self._scroll_subpixel += step
        int_step = int(self._scroll_subpixel)
        if int_step > 0:
            self._scroll_subpixel -= int_step
            vbar.setValue(vbar.value() + int_step)

    def _find_event_at_beat(self, beat: float) -> Tuple[int, NoteEvent]:
        """依據目前拍數二分搜尋對應的 NoteEvent 與其索引"""
        events = self._analysis.events
        lo, hi = 0, len(events) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            ev = events[mid]
            if beat < ev.beat_offset:
                hi = mid - 1
            elif beat >= ev.beat_offset + ev.duration_beats:
                lo = mid + 1
            else:
                return mid, ev
        idx = max(0, min(len(events) - 1, lo))
        return idx, events[idx]

    def _sync_playhead_visual(self, scroll_to_staff: bool = True):
        """根據 _current_beat 計算平滑游標位置並更新畫面"""
        if not self._analysis or not self._analysis.events:
            return

        idx, ev = self._find_event_at_beat(self._current_beat)
        staff = self._analysis.get_staff(ev.page_index, ev.staff_index)
        if staff is None:
            return

        # 計算游標在目前音符區間內的平滑水平內插位置
        frac = 0.0
        if ev.duration_beats > 1e-6:
            frac = max(0.0, min(1.0, (self._current_beat - ev.beat_offset) / ev.duration_beats))

        # 尋找同一行的下一個事件以決定滑動終點
        next_x = staff.x_right_norm
        if idx + 1 < len(self._analysis.events):
            nxt = self._analysis.events[idx + 1]
            if nxt.page_index == ev.page_index and nxt.staff_index == ev.staff_index:
                next_x = nxt.x_norm
            else:
                # 本行最後一個音符：滑向本小節尾端
                for m in staff.measures:
                    if m.measure_number == ev.measure_number:
                        next_x = m.x_end_norm
                        break

        # 休止小節從小節左緣滑到右緣；一般音符從符頭平滑移向下一個音符
        if ev.is_rest:
            start_x = ev.x_norm
            end_x = next_x
            for m in staff.measures:
                if m.measure_number == ev.measure_number:
                    start_x = m.x_start_norm
                    end_x = m.x_end_norm
                    break
            cursor_x = start_x + (end_x - start_x) * frac
        else:
            cursor_x = ev.x_norm + (next_x - ev.x_norm) * frac

        self.score_view.update_playhead(
            page_index=ev.page_index,
            staff=staff,
            cursor_x_norm=cursor_x,
            note_x_norm=ev.x_norm,
            note_y_norm=ev.y_norm,
            is_rest=ev.is_rest,
        )
        note_tag = f"  ♪ {ev.note_name}" if (not ev.is_rest and ev.note_name) else "  (休止)"
        self.measure_label.setText(f"小節：{ev.measure_number} / {self._analysis.total_measures}{note_tag}")

        # 播放中進入新音符時，即時合成並發聲
        if self._is_playing and idx != self._last_played_event_idx:
            self._last_played_event_idx = idx
            if not ev.is_rest and ev.midi_note > 0:
                beats_per_sec = max(0.2, (self._bpm / 60.0) * self._scroll_speed)
                dur_sec = max(0.10, (ev.duration_beats / beats_per_sec) * 0.95)
                self._synth.play_note(ev.midi_note, duration_sec=dur_sec)
            else:
                self._synth.stop_all()

        # 換行時自動將作用中的五線譜捲動到視窗舒適閱讀位置（上方 35% 處）
        staff_key = (ev.page_index, ev.staff_index)
        if scroll_to_staff and staff_key != self._last_scrolled_staff:
            self._last_scrolled_staff = staff_key
            self._scroll_to_staff(ev.page_index, staff)

    def _scroll_to_staff(self, page_index: int, staff: StaffInfo):
        """將指定頁面的五線譜平滑捲動至視窗上方約 1/3 處，保留下方視讀空間"""
        if not (0 <= page_index < len(self.score_view.page_labels)):
            return
        page_w = self.score_view.page_labels[page_index]
        staff_mid_y = page_w.y() + int(((staff.y_top_norm + staff.y_bot_norm) / 2.0) * page_w.height())
        viewport_h = self.scroll_area.viewport().height()
        target_scroll = max(0, staff_mid_y - int(viewport_h * 0.35))
        self.scroll_area.verticalScrollBar().setValue(target_scroll)

    def _on_score_clicked(self, page_index: int, x_norm: float, y_norm: float):
        """滑鼠點擊樂譜任意位置時，跳轉演奏游標到最接近的音符"""
        if not self._use_note_tracking() or not self._analysis:
            return
        if not (0 <= page_index < len(self._analysis.pages)):
            return

        page_res = self._analysis.pages[page_index]
        if not page_res.staves:
            return

        # 先找垂直距離最近的五線譜行
        best_staff = min(
            page_res.staves,
            key=lambda s: abs(((s.y_top_norm + s.y_bot_norm) / 2.0) - y_norm),
        )
        staff_events = [
            ev for ev in page_res.events
            if ev.staff_index == best_staff.staff_index
        ]
        if not staff_events:
            return

        # 再找水平距離最近的音符事件
        nearest_ev = min(staff_events, key=lambda ev: abs(ev.x_norm - x_norm))
        self._current_beat = nearest_ev.beat_offset
        self._last_scrolled_staff = (page_index, best_staff.staff_index)
        self._last_played_event_idx = -1
        self._sync_playhead_visual(scroll_to_staff=False)
        if not self._is_playing and not nearest_ev.is_rest and nearest_ev.midi_note > 0:
            self._synth.play_note(nearest_ev.midi_note, duration_sec=0.25)
        self.status_text.setText(f"已跳轉至第 {nearest_ev.measure_number} 小節。")

    def _on_scroll_changed(self, value):
        """根據當前滾動位置動態估算當前顯示頁面"""
        if not self.score_view.page_labels or self._page_count == 0:
            return

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
            self._on_restart()
            event.accept()
        elif key == Qt.Key.Key_F11 or key == Qt.Key.Key_F:
            win = self.window()
            if win:
                if win.isFullScreen():
                    win.showNormal()
                else:
                    win.showFullScreen()
            event.accept()
        else:
            super().keyPressEvent(event)
