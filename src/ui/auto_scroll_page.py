"""自動翻譜頁 - 支援 PDF/拍照樂譜載入、OpenCV OMR 音符分析、方案 A 游標、逐小節/逐音符步進校準與即時發聲"""

import os
import time
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QSizePolicy,
    QSpinBox, QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import (
    QImage, QPixmap, QKeyEvent, QDragEnterEvent, QDropEvent,
    QPainter, QColor, QPen, QBrush, QPolygonF, QMouseEvent, QFont
)

from src.core.score_analyzer import (
    ScoreAnalyzer, ScoreAnalysis, PageAnalysis, StaffInfo, NoteEvent
)
from src.core.note_synth import NoteSynthesizer


class ScorePageWidget(QWidget):
    """單頁樂譜顯示元件，內建方案 A 演奏游標與 OMR 偵測校準標記疊層"""

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
        self._is_grace: bool = False

        # OMR 視覺校準標記狀態
        self._page_analysis: Optional[PageAnalysis] = None
        self._show_debug_overlay: bool = False

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.setFixedSize(pixmap.size())
        self.update()

    def set_page_analysis(self, page_analysis: Optional[PageAnalysis], show_debug: bool):
        self._page_analysis = page_analysis
        self._show_debug_overlay = show_debug
        self.update()

    def set_show_debug(self, show_debug: bool):
        if self._show_debug_overlay != show_debug:
            self._show_debug_overlay = show_debug
            self.update()

    def set_playhead(
        self,
        staff: Optional[StaffInfo],
        cursor_x_norm: Optional[float],
        note_x_norm: Optional[float] = None,
        note_y_norm: Optional[float] = None,
        is_rest: bool = False,
        is_grace: bool = False,
    ):
        """更新本頁的演奏游標位置；若 staff 為 None 則清除本頁游標"""
        self._active_staff = staff
        self._cursor_x_norm = cursor_x_norm
        self._note_x_norm = note_x_norm
        self._note_y_norm = note_y_norm
        self._is_rest = is_rest
        self._is_grace = is_grace
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

        w = float(self.width())
        h = float(self.height())

        # 2. 若開啟「🔬 顯示 OMR 校準標記」，繪製小節線、小節編號與所有偵測音符點
        if self._show_debug_overlay and self._page_analysis is not None:
            self._paint_debug_overlay(painter, w, h)

        # 3. 方案 A：繪製作用中五線譜整行高亮 + 半透明垂直光條 + 頂部紅標 ▼ + 當前音符光圈
        if self._active_staff is None or self._cursor_x_norm is None:
            return

        st = self._active_staff
        sp_px = max(5.0, st.spacing_norm * h)
        pad_y = sp_px * 1.6
        band_top = max(4.0, st.y_top_norm * h - pad_y)
        band_bot = min(h - 4.0, st.y_bot_norm * h + pad_y)
        band_h = max(16.0, band_bot - band_top)

        staff_left = max(4.0, st.x_left_norm * w - sp_px * 0.8)
        staff_right = min(w - 4.0, st.x_right_norm * w + sp_px * 0.8)
        staff_w = max(20.0, staff_right - staff_left)

        # 3a. 整行五線譜暖黃淡色底光
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 224, 102, 55)))
        painter.drawRoundedRect(QRectF(staff_left, band_top, staff_w, band_h), 6.0, 6.0)

        painter.setBrush(QBrush(QColor(235, 160, 20, 185)))
        painter.drawRoundedRect(QRectF(staff_left, band_top, 4.0, band_h), 2.0, 2.0)

        # 3b. 半透明垂直光條 + 紅色中心細線
        cx = self._cursor_x_norm * w
        bar_w = max(10.0, sp_px * 1.35)
        painter.setBrush(QBrush(QColor(231, 76, 60, 46)))
        painter.drawRoundedRect(QRectF(cx - bar_w / 2.0, band_top, bar_w, band_h), 4.0, 4.0)

        painter.setPen(QPen(QColor(220, 38, 38, 215), 2.0))
        painter.drawLine(QPointF(cx, band_top), QPointF(cx, band_bot))

        # 3c. 頂部紅色倒三角指標 ▼
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

        # 3d. 目前播放音符柔和光圈（區分一般主音符與紫色小裝飾音）
        if not self._is_rest and self._note_x_norm is not None and self._note_y_norm is not None:
            nx = self._note_x_norm * w
            ny = self._note_y_norm * h
            if self._is_grace:
                r = max(5.0, sp_px * 0.68)
                painter.setPen(QPen(QColor(147, 51, 234, 230), 2.0))
                painter.setBrush(QBrush(QColor(168, 85, 247, 75)))
            else:
                r = max(6.5, sp_px * 0.88)
                painter.setPen(QPen(QColor(220, 38, 38, 225), 2.2))
                painter.setBrush(QBrush(QColor(239, 68, 68, 70)))
            painter.drawEllipse(QPointF(nx, ny), r, r)

    def _paint_debug_overlay(self, painter: QPainter, w: float, h: float):
        """繪製 OMR 校準輔助層：小節分界線、小節編號、休止框與每個音符中心標記"""
        font_small = QFont("Sans Serif", 8)
        font_small.setBold(True)
        painter.setFont(font_small)

        for st in self._page_analysis.staves:
            sp_px = max(5.0, st.spacing_norm * h)
            yt = st.y_top_norm * h
            yb = st.y_bot_norm * h

            # 五線譜外框（淡綠細虛線）
            pen_staff = QPen(QColor(16, 185, 129, 120), 1.0, Qt.PenStyle.DashLine)
            painter.setPen(pen_staff)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(st.x_left_norm * w, yt, (st.x_right_norm - st.x_left_norm) * w, yb - yt))

            for m in st.measures:
                mx1 = m.x_start_norm * w
                mx2 = m.x_end_norm * w
                # 小節起點藍色虛線與小節號 M#
                painter.setPen(QPen(QColor(37, 99, 235, 165), 1.2, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(mx1, yt - sp_px * 1.2), QPointF(mx1, yb + sp_px * 0.8))

                m_title = f"M{m.measure_number}" if m.rest_measures <= 1 else f"M{m.measure_number}..{m.measure_number + m.rest_measures - 1}"
                painter.setPen(QColor(29, 78, 216, 220))
                painter.drawText(QPointF(mx1 + 3.0, yt - sp_px * 1.3), m_title)

                # 若為休止小節，畫淡橘色區塊標示
                if m.rest_measures > 0:
                    painter.setPen(QPen(QColor(234, 88, 12, 160), 1.2, Qt.PenStyle.DotLine))
                    painter.setBrush(QBrush(QColor(251, 146, 60, 45)))
                    painter.drawRoundedRect(QRectF(mx1 + 2, yt, max(8.0, mx2 - mx1 - 4), yb - yt), 4.0, 4.0)

                # 標示每個偵測到的音符中心點與音名
                for ev in m.notes:
                    if ev.is_rest:
                        continue
                    ex = ev.x_norm * w
                    ey = ev.y_norm * h
                    if ev.is_grace:
                        color = QColor(147, 51, 234, 215)  # 紫色：裝飾音
                        r = 3.2
                    elif ev.is_hollow:
                        color = QColor(5, 150, 105, 215)   # 綠色：空心音符(二分/全音)
                        r = 4.2
                    else:
                        color = QColor(2, 132, 199, 215)   # 藍色：實心主音符
                        r = 3.8
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(QPointF(ex, ey), r, r)
                    painter.setPen(color)
                    painter.drawText(QPointF(ex - 7.0, yb + sp_px * 2.1), ev.note_name)


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
        for widget in self.page_labels:
            self.layout.removeWidget(widget)
            widget.deleteLater()
        self.page_labels.clear()

    def set_pages(
        self,
        pixmaps: List[QPixmap],
        analysis: Optional[ScoreAnalysis] = None,
        show_debug: bool = False,
    ):
        self.clear_pages()
        for i, pm in enumerate(pixmaps):
            page_w = ScorePageWidget(i, pm)
            if analysis and i < len(analysis.pages):
                page_w.set_page_analysis(analysis.pages[i], show_debug)
            page_w.clicked.connect(self.page_clicked.emit)
            self.layout.addWidget(page_w)
            self.page_labels.append(page_w)

    def set_show_debug(self, show_debug: bool):
        for page_w in self.page_labels:
            page_w.set_show_debug(show_debug)

    def update_playhead(
        self,
        page_index: int,
        staff: Optional[StaffInfo],
        cursor_x_norm: Optional[float],
        note_x_norm: Optional[float] = None,
        note_y_norm: Optional[float] = None,
        is_rest: bool = False,
        is_grace: bool = False,
    ):
        for i, page_w in enumerate(self.page_labels):
            if i == page_index and staff is not None and cursor_x_norm is not None:
                page_w.set_playhead(staff, cursor_x_norm, note_x_norm, note_y_norm, is_rest, is_grace)
            else:
                page_w.clear_playhead()

    def clear_playhead(self):
        for page_w in self.page_labels:
            page_w.clear_playhead()


class AutoScrollPage(QWidget):
    """自動翻譜與樂譜音符校準播放頁面"""

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
        self._scroll_speed = 1.0
        self._bpm = 100
        self._beats_per_measure = 4.0
        self._zoom_factor = 1.0
        self._raw_pixmaps: List[QPixmap] = []
        self._enhanced_pixmaps: List[QPixmap] = []
        self._raw_rgb_pages: List[np.ndarray] = []
        self._enhance_view: bool = True
        self._scroll_subpixel = 0.0

        # 音符分析、合成器與步進校準狀態
        self._analyzer = ScoreAnalyzer()
        self._analysis: Optional[ScoreAnalysis] = None
        self._synth = NoteSynthesizer()
        self._track_mode = "note"  # "note" | "scroll"
        self._current_beat = 0.0
        self._stop_at_beat: Optional[float] = None  # 用於「單小節播放」結束自動停止
        self._last_tick_time = 0.0
        self._last_scrolled_staff: Tuple[int, int] = (-1, -1)
        self._last_played_event_idx: int = -1
        self._show_debug_overlay: bool = False

        # 滾動 / 游標定時器（約 30 FPS）
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setInterval(33)
        self._scroll_timer.timeout.connect(self._on_scroll_tick)

        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 頂部控制工具列面板
        toolbar = QFrame()
        toolbar.setStyleSheet(
            "QFrame { background-color: #f8f9fa; border-bottom: 1px solid #cbd5e1; }"
        )
        tb_vbox = QVBoxLayout(toolbar)
        tb_vbox.setContentsMargins(10, 6, 10, 6)
        tb_vbox.setSpacing(5)

        # === 第一列：開啟、全曲播放、BPM、拍號、速度、聲音與音色、模式 ===
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        self.open_btn = QPushButton("📂 開啟樂譜")
        self.open_btn.setMinimumHeight(28)
        self.open_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 12px; background-color: #2b579a; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1e3f73; }"
        )
        self.open_btn.clicked.connect(self._on_open)
        row1.addWidget(self.open_btn)

        self.play_btn = QPushButton("▶ 播放 (Space)")
        self.play_btn.setMinimumHeight(28)
        self.play_btn.setMinimumWidth(128)
        self.play_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 14px; background-color: #107c41; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0c5e31; }"
        )
        self.play_btn.clicked.connect(self._on_play_toggle)
        row1.addWidget(self.play_btn)

        self.restart_btn = QPushButton("⏮ 重頭")
        self.restart_btn.setMinimumHeight(28)
        self.restart_btn.clicked.connect(self._on_restart)
        row1.addWidget(self.restart_btn)

        row1.addSpacing(6)
        row1.addWidget(QLabel("BPM："))
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(30, 260)
        self.bpm_spin.setValue(self._bpm)
        self.bpm_spin.setFixedWidth(66)
        self.bpm_spin.setMinimumHeight(26)
        self.bpm_spin.valueChanged.connect(self._on_bpm_change)
        row1.addWidget(self.bpm_spin)

        row1.addWidget(QLabel("拍號："))
        self.time_sig_combo = QComboBox()
        self.time_sig_combo.addItems(list(self.TIME_SIGNATURES.keys()))
        self.time_sig_combo.setCurrentText("4/4")
        self.time_sig_combo.setFixedWidth(62)
        self.time_sig_combo.setMinimumHeight(26)
        self.time_sig_combo.currentTextChanged.connect(self._on_time_sig_change)
        row1.addWidget(self.time_sig_combo)

        row1.addWidget(QLabel("倍率："))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(5)
        self.speed_slider.setMaximum(30)
        self.speed_slider.setValue(10)
        self.speed_slider.setFixedWidth(95)
        self.speed_slider.valueChanged.connect(self._on_speed_change)
        row1.addWidget(self.speed_slider)

        self.speed_value_label = QLabel("1.0x")
        self.speed_value_label.setMinimumWidth(38)
        self.speed_value_label.setStyleSheet("font-weight: bold; color: #333;")
        row1.addWidget(self.speed_value_label)

        row1.addStretch()

        self.sound_btn = QPushButton("🔊 聲音：開")
        self.sound_btn.setCheckable(True)
        self.sound_btn.setChecked(True)
        self.sound_btn.setMinimumHeight(26)
        self.sound_btn.setStyleSheet(
            "QPushButton { padding: 2px 10px; border-radius: 4px; font-weight: bold; "
            "background-color: #e6f4ea; color: #137333; border: 1px solid #a8dab5; }"
            "QPushButton:!checked { background-color: #f1f3f4; color: #5f6368; border: 1px solid #dadce0; }"
        )
        self.sound_btn.toggled.connect(self._on_sound_toggle)
        row1.addWidget(self.sound_btn)

        self.timbre_combo = QComboBox()
        self.timbre_combo.addItems(["鋼琴", "薩克斯風", "長笛"])
        self.timbre_combo.setMinimumWidth(82)
        self.timbre_combo.setMinimumHeight(26)
        self.timbre_combo.currentTextChanged.connect(self._on_timbre_change)
        row1.addWidget(self.timbre_combo)

        row1.addWidget(QLabel("模式："))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["🎯 音符追蹤", "📜 純垂直捲動"])
        self.mode_combo.setMinimumWidth(114)
        self.mode_combo.setMinimumHeight(26)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
        row1.addWidget(self.mode_combo)

        tb_vbox.addLayout(row1)

        # === 第二列：逐小節播放、逐音符步進、OMR 標記開關、縮放與換頁 ===
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        # 逐小節控制群組
        lbl_m = QLabel("逐小節：")
        lbl_m.setStyleSheet("font-weight: bold; color: #1e40af;")
        row2.addWidget(lbl_m)

        self.prev_measure_btn = QPushButton("◀ 上一小節")
        self.prev_measure_btn.setMinimumHeight(26)
        self.prev_measure_btn.clicked.connect(self._on_prev_measure)
        row2.addWidget(self.prev_measure_btn)

        self.play_measure_btn = QPushButton("▶ 播本小節")
        self.play_measure_btn.setMinimumHeight(26)
        self.play_measure_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #dbeafe; color: #1d4ed8; "
            "border: 1px solid #93c5fd; border-radius: 4px; padding: 2px 10px; }"
            "QPushButton:hover { background-color: #bfdbfe; }"
        )
        self.play_measure_btn.clicked.connect(self._on_play_current_measure)
        row2.addWidget(self.play_measure_btn)

        self.next_measure_btn = QPushButton("下一小節 ▶")
        self.next_measure_btn.setMinimumHeight(26)
        self.next_measure_btn.clicked.connect(self._on_next_measure)
        row2.addWidget(self.next_measure_btn)

        row2.addSpacing(8)

        # 逐音符控制群組
        lbl_n = QLabel("逐音符：")
        lbl_n.setStyleSheet("font-weight: bold; color: #9a3412;")
        row2.addWidget(lbl_n)

        self.prev_note_btn = QPushButton("◀ 上一音 ([)")
        self.prev_note_btn.setMinimumHeight(26)
        self.prev_note_btn.clicked.connect(self._on_prev_note)
        row2.addWidget(self.prev_note_btn)

        self.replay_note_btn = QPushButton("🔊 試聽此音")
        self.replay_note_btn.setMinimumHeight(26)
        self.replay_note_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #ffedd5; color: #c2410c; "
            "border: 1px solid #fdba74; border-radius: 4px; padding: 2px 10px; }"
            "QPushButton:hover { background-color: #fed7aa; }"
        )
        self.replay_note_btn.clicked.connect(self._on_replay_current_note)
        row2.addWidget(self.replay_note_btn)

        self.next_note_btn = QPushButton("下一音 (]) ▶")
        self.next_note_btn.setMinimumHeight(26)
        self.next_note_btn.clicked.connect(self._on_next_note)
        row2.addWidget(self.next_note_btn)

        row2.addSpacing(8)

        # 去底色強化與 OMR 視覺校準疊層開關
        self.enhance_view_chk = QCheckBox("✨ 去底色強化")
        self.enhance_view_chk.setChecked(True)
        self.enhance_view_chk.setToolTip("自動濾除泛黃底色、陰影與背面透字，強化五線譜與音符黑度")
        self.enhance_view_chk.setStyleSheet("font-weight: bold; color: #7c3aed;")
        self.enhance_view_chk.toggled.connect(self._on_enhance_view_toggle)
        row2.addWidget(self.enhance_view_chk)

        self.debug_overlay_chk = QCheckBox("🔬 顯示偵測標記")
        self.debug_overlay_chk.setStyleSheet("font-weight: bold; color: #0f766e;")
        self.debug_overlay_chk.toggled.connect(self._on_debug_overlay_toggle)
        row2.addWidget(self.debug_overlay_chk)

        row2.addStretch()

        # 縮放與頁數
        zoom_out_btn = QPushButton("🔍-")
        zoom_out_btn.setFixedSize(34, 25)
        zoom_out_btn.clicked.connect(lambda: self._adjust_zoom(-0.1))
        row2.addWidget(zoom_out_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(44)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row2.addWidget(self.zoom_label)

        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setFixedSize(34, 25)
        zoom_in_btn.clicked.connect(lambda: self._adjust_zoom(0.1))
        row2.addWidget(zoom_in_btn)

        fit_width_btn = QPushButton("適應寬度")
        fit_width_btn.setMinimumHeight(25)
        fit_width_btn.clicked.connect(self._fit_to_width)
        row2.addWidget(fit_width_btn)

        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedSize(28, 25)
        self.prev_btn.clicked.connect(self._on_prev_page)
        row2.addWidget(self.prev_btn)

        self.page_label = QLabel("第 0 / 0 頁")
        self.page_label.setMinimumWidth(72)
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_label.setStyleSheet("font-weight: bold; color: #333;")
        row2.addWidget(self.page_label)

        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedSize(28, 25)
        self.next_btn.clicked.connect(self._on_next_page)
        row2.addWidget(self.next_btn)

        self.top_btn = QPushButton("⏫ 回頂端")
        self.top_btn.setMinimumHeight(25)
        self.top_btn.clicked.connect(self._scroll_to_top)
        row2.addWidget(self.top_btn)

        tb_vbox.addLayout(row2)
        main_layout.addWidget(toolbar)

        # === 音符即時資訊檢視與校準列 (Note Inspector & Calibration Panel) ===
        self.inspector_bar = QFrame()
        self.inspector_bar.setStyleSheet(
            "QFrame { background-color: #fffbeb; border-bottom: 1px solid #fde68a; padding: 4px 12px; }"
        )
        insp_layout = QHBoxLayout(self.inspector_bar)
        insp_layout.setContentsMargins(6, 2, 6, 2)
        insp_layout.setSpacing(14)

        self.measure_label = QLabel("小節：-- / --")
        self.measure_label.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #b91c1c; background-color: #fef2f2; "
            "border: 1px solid #fecaca; border-radius: 4px; padding: 2px 10px;"
        )
        insp_layout.addWidget(self.measure_label)

        self.note_detail_label = QLabel(
            "🔎 音符校準資訊：尚未載入樂譜（載入後可使用「逐小節」或「逐音符 [ / ]」逐步檢視每個音符的音高、時值與位置）"
        )
        self.note_detail_label.setStyleSheet("font-size: 13px; color: #1e293b; font-weight: 500;")
        insp_layout.addWidget(self.note_detail_label, 1)

        main_layout.addWidget(self.inspector_bar)

        # === 樂譜捲動檢視區域 (QScrollArea) ===
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setStyleSheet(
            "QScrollArea { background-color: #525659; border: none; }"
        )
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)

        self.score_view = ScoreViewWidget()
        self.score_view.page_clicked.connect(self._on_score_clicked)
        self.scroll_area.setWidget(self.score_view)

        self.placeholder_label = QLabel(
            "<h3>🎼 尚未載入樂譜</h3>"
            "<p>支援 <b>PDF 樂譜</b> 與 <b>手機拍照圖片 (JPG / PNG / WEBP)</b></p>"
            "<p>載入後自動偵測五線譜、休止符、小節與音符時值，支援 <b>全曲播放 / 逐小節播放 / 逐音符步進校準</b></p>"
            "<p>點擊上方「<b>📂 開啟樂譜</b>」或直接將檔案<b>拖曳至此處</b></p>"
        )
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet(
            "QLabel { color: #d0d0d0; font-size: 15px; padding: 60px; line-height: 160%; }"
        )
        self.score_view.layout.addWidget(self.placeholder_label)

        main_layout.addWidget(self.scroll_area)

        # === 底部狀態與快捷鍵指引列 ===
        self.status_bar = QFrame()
        self.status_bar.setStyleSheet(
            "QFrame { background-color: #f1f3f5; border-top: 1px solid #ced4da; padding: 4px 12px; }"
        )
        sb_layout = QHBoxLayout(self.status_bar)
        sb_layout.setContentsMargins(0, 0, 0, 0)

        self.status_text = QLabel("準備就緒。支援拖曳 PDF / 圖檔，點擊樂譜任意音符可跳轉並檢視資訊。")
        self.status_text.setStyleSheet("color: #495057; font-size: 12px;")
        sb_layout.addWidget(self.status_text)

        sb_layout.addStretch()

        shortcut_hint = QLabel(
            "快捷鍵：[Space] 播放/暫停 | [ [ / ] ] 上/下一音符 | [M] 播本小節 | [點擊樂譜] 跳轉試聽 | [F11] 全螢幕"
        )
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
            self._enhanced_pixmaps.clear()
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
            self._stop_at_beat = None
            self._last_scrolled_staff = (-1, -1)
            self._last_played_event_idx = -1

            # 1. 高解析渲染所有頁面
            self._render_all_pages()

            # 2. 執行尺度不變 OpenCV OMR 結構分析
            self._run_score_analysis()

            # 3. 預設自適應寬度呈現並顯示初始游標與第一音符校準資訊
            self._fit_to_width()

            file_name = os.path.basename(file_path)
            if self._analysis and self._analysis.has_staves:
                n_staves = sum(len(p.staves) for p in self._analysis.pages)
                n_events = len(self._analysis.events)
                self.status_text.setText(
                    f"已載入：{file_name}（{self._page_count} 頁 | 偵測到 {n_staves} 行五線譜、"
                    f"共 {self._analysis.total_measures} 小節、{n_events} 個演奏事件）"
                )
                self._sync_playhead_visual(scroll_to_staff=False)
            else:
                self.status_text.setText(
                    f"已載入：{file_name}（共 {self._page_count} 頁，未偵測到五線譜，已切換為純垂直捲動模式）"
                )
                self.measure_label.setText("小節：-- / --")
                self.note_detail_label.setText("🔎 此檔案未偵測到五線譜結構，目前使用純垂直捲動模式。")

            self.page_label.setText(f"第 1 / {self._page_count} 頁")
            return True

        except Exception as e:
            QMessageBox.critical(self, "開啟失敗", f"無法解析樂譜檔案：\n{str(e)}")
            return False

    def _render_all_pages(self):
        if not self._doc:
            return

        self._raw_pixmaps.clear()
        self._enhanced_pixmaps.clear()
        self._raw_rgb_pages.clear()
        zoom_matrix = fitz.Matrix(150 / 72, 150 / 72)
        for page_idx in range(self._page_count):
            page = self._doc[page_idx]
            pix = page.get_pixmap(matrix=zoom_matrix, alpha=False)
            qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qimg.copy())
            self._raw_pixmaps.append(pm)

            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
            self._raw_rgb_pages.append(rgb)

            enh_rgb = np.ascontiguousarray(ScoreAnalyzer.enhance_score_rgb(rgb))
            eh, ew, _ = enh_rgb.shape
            enh_qimg = QImage(enh_rgb.data, ew, eh, ew * 3, QImage.Format.Format_RGB888)
            self._enhanced_pixmaps.append(QPixmap.fromImage(enh_qimg.copy()))

    def _run_score_analysis(self):
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
        if not self._raw_pixmaps:
            return

        if self.placeholder_label:
            self.placeholder_label.hide()

        source_pixmaps = (
            self._enhanced_pixmaps
            if (self._enhance_view and len(self._enhanced_pixmaps) == len(self._raw_pixmaps))
            else self._raw_pixmaps
        )

        scaled_pixmaps = []
        for pm in source_pixmaps:
            target_w = max(200, int(pm.width() * self._zoom_factor))
            scaled_pm = pm.scaledToWidth(target_w, Qt.TransformationMode.SmoothTransformation)
            scaled_pixmaps.append(scaled_pm)

        self.score_view.set_pages(scaled_pixmaps, self._analysis, self._show_debug_overlay)
        self.zoom_label.setText(f"{int(self._zoom_factor * 100)}%")

        if self._use_note_tracking():
            self._sync_playhead_visual(scroll_to_staff=False)

    def _use_note_tracking(self) -> bool:
        return (
            self._track_mode == "note"
            and self._analysis is not None
            and self._analysis.has_staves
            and len(self._analysis.events) > 0
        )

    def _fit_to_width(self):
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
        if not self._raw_pixmaps:
            return
        new_factor = round(self._zoom_factor + delta, 1)
        self._zoom_factor = max(0.3, min(3.0, new_factor))
        self._update_view_scale()

    def _on_open(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟樂譜檔案", "",
            "樂譜與圖片 (*.pdf *.jpg *.jpeg *.png *.webp *.bmp);;PDF 樂譜 (*.pdf);;圖片 (*.jpg *.jpeg *.png *.webp);;所有檔案 (*)"
        )
        if file_path:
            self.load_file(file_path)

    def _on_play_toggle(self):
        if not self._raw_pixmaps:
            return
        if self._is_playing:
            self._stop_playing()
        else:
            self._stop_at_beat = None  # 全曲連續播放模式
            self._start_playing()

    def _start_playing(self):
        if self._use_note_tracking() and self._analysis:
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
        # 立即觸發當前音符發聲與畫面更新
        if self._use_note_tracking():
            self._sync_playhead_visual(scroll_to_staff=True)
        self._scroll_timer.start()

    def _stop_playing(self):
        self._is_playing = False
        self._stop_at_beat = None
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

    def _on_enhance_view_toggle(self, checked: bool):
        self._enhance_view = checked
        self._update_view_scale()

    def _on_debug_overlay_toggle(self, checked: bool):
        self._show_debug_overlay = checked
        self.score_view.set_show_debug(checked)

    def _on_restart(self):
        self._stop_playing()
        self._current_beat = 0.0
        self._last_scrolled_staff = (-1, -1)
        self._scroll_to_top()
        if self._use_note_tracking():
            self._sync_playhead_visual(scroll_to_staff=True)

    # === 逐小節播放與跳轉控制 ===
    def _get_all_measures(self):
        if not self._analysis:
            return []
        measures = []
        for p in self._analysis.pages:
            for s in p.staves:
                measures.extend(s.measures)
        return measures

    def _find_current_measure_idx(self, measures) -> int:
        if not measures:
            return -1
        _, ev = self._find_event_at_beat(self._current_beat)
        for i, m in enumerate(measures):
            if m.measure_number == ev.measure_number:
                return i
        return 0

    def _on_prev_measure(self):
        """跳至上一小節開頭並試聽第一音"""
        if not self._use_note_tracking():
            return
        self._stop_playing()
        measures = self._get_all_measures()
        if not measures:
            return
        cur_m_idx = self._find_current_measure_idx(measures)
        target_idx = max(0, cur_m_idx - 1)
        self._jump_to_measure_obj(measures[target_idx], preview_sound=True)

    def _on_next_measure(self):
        """跳至下一小節開頭並試聽第一音"""
        if not self._use_note_tracking():
            return
        self._stop_playing()
        measures = self._get_all_measures()
        if not measures:
            return
        cur_m_idx = self._find_current_measure_idx(measures)
        target_idx = min(len(measures) - 1, cur_m_idx + 1)
        self._jump_to_measure_obj(measures[target_idx], preview_sound=True)

    def _on_play_current_measure(self):
        """從當前小節第一拍完整播放完本小節後自動暫停"""
        if not self._use_note_tracking():
            return
        measures = self._get_all_measures()
        if not measures:
            return
        cur_m_idx = self._find_current_measure_idx(measures)
        m = measures[cur_m_idx]
        self._stop_playing()
        self._current_beat = m.beat_start
        self._stop_at_beat = m.beat_end
        self.status_text.setText(f"正在單獨播放第 {m.measure_number} 小節（播畢自動暫停）...")
        self._start_playing()
        # _start_playing 不清除剛設定的 _stop_at_beat
        self._stop_at_beat = m.beat_end

    def _jump_to_measure_obj(self, m, preview_sound: bool = True):
        self._current_beat = m.beat_start
        self._sync_playhead_visual(scroll_to_staff=True)
        if preview_sound and m.notes:
            first_ev = m.notes[0]
            if not first_ev.is_rest and first_ev.midi_note > 0:
                self._synth.play_note(first_ev.midi_note, duration_sec=0.30)
        self.status_text.setText(f"已跳至第 {m.measure_number} 小節（共 {len(m.notes)} 個事件）。")

    # === 逐音符步進與單音校準試聽 ===
    def _on_prev_note(self):
        """退回上一個音符，顯示詳細資訊並發聲"""
        if not self._use_note_tracking() or not self._analysis:
            return
        self._stop_playing()
        idx, _ = self._find_event_at_beat(self._current_beat)
        new_idx = max(0, idx - 1)
        self._step_to_event_index(new_idx)

    def _on_next_note(self):
        """前進下一個音符，顯示詳細資訊並發聲"""
        if not self._use_note_tracking() or not self._analysis:
            return
        self._stop_playing()
        idx, _ = self._find_event_at_beat(self._current_beat)
        new_idx = min(len(self._analysis.events) - 1, idx + 1)
        self._step_to_event_index(new_idx)

    def _on_replay_current_note(self):
        """重新試聽目前停留的音符"""
        if not self._use_note_tracking() or not self._analysis:
            return
        self._stop_playing()
        idx, _ = self._find_event_at_beat(self._current_beat)
        self._step_to_event_index(idx)

    def _step_to_event_index(self, idx: int):
        if not self._analysis or not (0 <= idx < len(self._analysis.events)):
            return
        ev = self._analysis.events[idx]
        # 微加 1e-4 確保二分搜尋精準命中此事件
        self._current_beat = ev.beat_offset + min(1e-4, ev.duration_beats * 0.05)
        self._sync_playhead_visual(scroll_to_staff=True)
        if not ev.is_rest and ev.midi_note > 0:
            beats_per_sec = max(0.2, (self._bpm / 60.0) * self._scroll_speed)
            dur_sec = max(0.18, min(0.8, ev.duration_beats / beats_per_sec))
            self._synth.play_note(ev.midi_note, duration_sec=dur_sec)
        else:
            self._synth.stop_all()
        self.status_text.setText(
            f"單音步進 [#{ev.event_index}/{len(self._analysis.events)}]："
            f"第 {ev.measure_number} 小節第 {ev.note_index_in_measure} 個 → {ev.note_name} ({ev.note_type})"
        )

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
                self._update_view_scale()
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
        if not self._is_playing:
            return

        if self._use_note_tracking() and self._analysis:
            now = time.perf_counter()
            dt = max(0.0, min(0.25, now - self._last_tick_time))
            self._last_tick_time = now

            beats_per_sec = (self._bpm / 60.0) * self._scroll_speed
            self._current_beat += dt * beats_per_sec

            # 若為「單小節播放」，到達本小節結尾時自動停止並停在下一小節起點（或本小節尾）
            if self._stop_at_beat is not None and self._current_beat >= self._stop_at_beat:
                target_end = min(self._analysis.total_beats, self._stop_at_beat)
                self._stop_playing()
                # 將游標停在下一小節開頭方便接著按「播本小節」繼續聽下一小節！
                self._current_beat = target_end
                if self._current_beat < self._analysis.total_beats:
                    self._sync_playhead_visual(scroll_to_staff=True)
                    _, next_ev = self._find_event_at_beat(self._current_beat)
                    self.status_text.setText(
                        f"單小節播放完畢，已自動停在第 {next_ev.measure_number} 小節開頭（可再按「播本小節」或 [M] 繼續）。"
                    )
                else:
                    self._current_beat = max(0.0, self._analysis.total_beats - 1e-4)
                    self._sync_playhead_visual(scroll_to_staff=True)
                    self.status_text.setText("最後一小節播放完畢。")
                return

            if self._current_beat >= self._analysis.total_beats:
                self._current_beat = max(0.0, self._analysis.total_beats - 1e-4)
                self._sync_playhead_visual(scroll_to_staff=True)
                self._stop_playing()
                self.status_text.setText("樂譜已播放至最後一小節結尾。")
                return

            self._sync_playhead_visual(scroll_to_staff=True)
            return

        # 純垂直捲動模式
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
        """根據 _current_beat 計算游標位置、更新音符詳細校準面板，並於播放時發聲"""
        if not self._analysis or not self._analysis.events:
            return

        idx, ev = self._find_event_at_beat(self._current_beat)
        staff = self._analysis.get_staff(ev.page_index, ev.staff_index)
        if staff is None:
            return

        frac = 0.0
        if ev.duration_beats > 1e-6:
            frac = max(0.0, min(1.0, (self._current_beat - ev.beat_offset) / ev.duration_beats))

        # 單音步進暫停狀態下，讓垂直光條精準對齊在該音符符頭正中央（便於校準比對！）
        if not self._is_playing:
            cursor_x = ev.x_norm
        elif ev.is_rest:
            start_x = ev.x_norm
            end_x = staff.x_right_norm
            for m in staff.measures:
                if m.measure_number == ev.measure_number:
                    start_x = m.x_start_norm
                    end_x = m.x_end_norm
                    break
            cursor_x = start_x + (end_x - start_x) * frac
        else:
            next_x = staff.x_right_norm
            if idx + 1 < len(self._analysis.events):
                nxt = self._analysis.events[idx + 1]
                if nxt.page_index == ev.page_index and nxt.staff_index == ev.staff_index:
                    next_x = nxt.x_norm
                else:
                    for m in staff.measures:
                        if m.measure_number == ev.measure_number:
                            next_x = m.x_end_norm
                            break
            cursor_x = ev.x_norm + (next_x - ev.x_norm) * frac

        self.score_view.update_playhead(
            page_index=ev.page_index,
            staff=staff,
            cursor_x_norm=cursor_x,
            note_x_norm=ev.x_norm,
            note_y_norm=ev.y_norm,
            is_rest=ev.is_rest,
            is_grace=ev.is_grace,
        )

        # 更新左上徽章與音符校準詳細資訊列
        if ev.is_rest:
            m_range = (
                f"{ev.measure_number}~{ev.measure_number + ev.rest_measures - 1}"
                if ev.rest_measures > 1 else f"{ev.measure_number}"
            )
            self.measure_label.setText(f"小節：{m_range} / {self._analysis.total_measures}  ({ev.note_type})")
            self.note_detail_label.setText(
                f"🔎 事件 #{ev.event_index}/{len(self._analysis.events)}  |  "
                f"第 {ev.staff_index + 1} 行 • 第 {m_range} 小節 (第 1/1 個事件)  |  "
                f"類型：【{ev.note_type}】  |  音高：靜音 (Rest)  |  "
                f"時值：{ev.duration_beats:.2f} 拍 (起始 {ev.beat_offset:.2f} 拍)"
            )
        else:
            self.measure_label.setText(
                f"小節：{ev.measure_number} / {self._analysis.total_measures}  ♪ {ev.note_name}"
            )
            self.note_detail_label.setText(
                f"🔎 事件 #{ev.event_index}/{len(self._analysis.events)}  |  "
                f"第 {ev.staff_index + 1} 行 • 第 {ev.measure_number} 小節 "
                f"第 {ev.note_index_in_measure}/{ev.notes_in_measure} 個音符  |  "
                f"類型：【{ev.note_type}】  |  "
                f"音高：{ev.note_name} (MIDI {ev.midi_note} • {ev.freq_hz:.1f} Hz)  |  "
                f"時長：{ev.duration_beats:.2f} 拍 (橫槓={ev.beam_count})"
            )

        # 連續播放中進入新音符時發聲
        if self._is_playing and idx != self._last_played_event_idx:
            self._last_played_event_idx = idx
            if not ev.is_rest and ev.midi_note > 0:
                beats_per_sec = max(0.2, (self._bpm / 60.0) * self._scroll_speed)
                dur_sec = max(0.09, (ev.duration_beats / beats_per_sec) * 0.95)
                self._synth.play_note(ev.midi_note, duration_sec=dur_sec)
            else:
                self._synth.stop_all()

        staff_key = (ev.page_index, ev.staff_index)
        if scroll_to_staff and staff_key != self._last_scrolled_staff:
            self._last_scrolled_staff = staff_key
            self._scroll_to_staff(ev.page_index, staff)

    def _scroll_to_staff(self, page_index: int, staff: StaffInfo):
        if not (0 <= page_index < len(self.score_view.page_labels)):
            return
        page_w = self.score_view.page_labels[page_index]
        staff_mid_y = page_w.y() + int(((staff.y_top_norm + staff.y_bot_norm) / 2.0) * page_w.height())
        viewport_h = self.scroll_area.viewport().height()
        target_scroll = max(0, staff_mid_y - int(viewport_h * 0.35))
        self.scroll_area.verticalScrollBar().setValue(target_scroll)

    def _on_score_clicked(self, page_index: int, x_norm: float, y_norm: float):
        """點擊樂譜任意音符：直接跳轉至該音符、更新校準面板並試聽音高"""
        if not self._use_note_tracking() or not self._analysis:
            return
        if not (0 <= page_index < len(self._analysis.pages)):
            return

        page_res = self._analysis.pages[page_index]
        if not page_res.staves:
            return

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

        nearest_ev = min(staff_events, key=lambda ev: abs(ev.x_norm - x_norm))
        self._current_beat = nearest_ev.beat_offset + min(1e-4, nearest_ev.duration_beats * 0.05)
        self._last_scrolled_staff = (page_index, best_staff.staff_index)
        self._last_played_event_idx = -1
        self._sync_playhead_visual(scroll_to_staff=False)
        if not self._is_playing and not nearest_ev.is_rest and nearest_ev.midi_note > 0:
            self._synth.play_note(nearest_ev.midi_note, duration_sec=0.28)
        self.status_text.setText(
            f"已點選：第 {nearest_ev.measure_number} 小節第 {nearest_ev.note_index_in_measure} 個 → "
            f"{nearest_ev.note_name} ({nearest_ev.note_type})"
        )

    def _on_scroll_changed(self, value):
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
        if not self.score_view.page_labels:
            return
        self._scroll_to_page(max(0, self._current_page - 1))

    def _on_next_page(self):
        if not self.score_view.page_labels:
            return
        self._scroll_to_page(min(self._page_count - 1, self._current_page + 1))

    def _scroll_to_page(self, page_index: int):
        if 0 <= page_index < len(self.score_view.page_labels):
            target_y = self.score_view.page_labels[page_index].y()
            self.scroll_area.verticalScrollBar().setValue(target_y)
            self._current_page = page_index
            self.page_label.setText(f"第 {page_index + 1} / {self._page_count} 頁")

    def _scroll_to_top(self):
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
        elif key == Qt.Key.Key_BracketLeft:
            self._on_prev_note()
            event.accept()
        elif key == Qt.Key.Key_BracketRight:
            self._on_next_note()
            event.accept()
        elif key == Qt.Key.Key_M:
            self._on_play_current_measure()
            event.accept()
        elif key in (Qt.Key.Key_PageUp, Qt.Key.Key_Left):
            self._on_prev_measure()
            event.accept()
        elif key in (Qt.Key.Key_PageDown, Qt.Key.Key_Right):
            self._on_next_measure()
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
