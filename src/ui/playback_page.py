"""樂譜播放頁"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox
)
from PyQt6.QtCore import Qt


class PlaybackPage(QWidget):
    """樂譜播放頁面"""

    def __init__(self):
        super().__init__()
        self._is_playing = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 目前樂譜資訊
        self.info_label = QLabel("尚未載入樂譜")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet(
            "background-color: #f5f5f5; padding: 10px; "
            "border-radius: 6px; font-size: 14px; color: #666;"
        )
        layout.addWidget(self.info_label)

        # 播放控制按鈕列
        controls = QHBoxLayout()
        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.setFixedHeight(50)
        self.play_btn.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; "
            "font-size: 20px; border-radius: 8px; }"
        )
        self.play_btn.clicked.connect(self._on_play_toggle)
        stop_btn = QPushButton("⏹ 停止")
        stop_btn.clicked.connect(self._on_stop)
        controls.addWidget(self.play_btn)
        controls.addWidget(stop_btn)
        layout.addLayout(controls)

        # 進度條
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setMaximumWidth(600)
        self.progress_slider.setEnabled(False)
        layout.addWidget(self.progress_slider)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("color: #666;")
        layout.addWidget(self.time_label)

        # 速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("速度："))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(5)
        self.speed_slider.setMaximum(20)
        self.speed_slider.setValue(10)  # 1.0x
        self.speed_slider.setMaximumWidth(300)
        self.speed_slider.valueChanged.connect(self._on_speed_change)
        self.speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        speed_layout.addStretch()
        layout.addLayout(speed_layout)

        # 音色選擇
        timbre_layout = QHBoxLayout()
        timbre_layout.addWidget(QLabel("音色："))
        self.timbre_combo = QComboBox()
        self.timbre_combo.addItems(
            ["鋼琴", "吉他", "小提琴", "長笛", "豎笛", "小喇叭"]
        )
        timbre_layout.addWidget(self.timbre_combo)
        timbre_layout.addStretch()
        layout.addLayout(timbre_layout)

        # 音量
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("🔊"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(80)
        self.volume_slider.setMaximumWidth(300)
        vol_layout.addWidget(self.volume_slider)
        vol_layout.addStretch()
        layout.addLayout(vol_layout)

        # 循環播放
        loop_layout = QHBoxLayout()
        self.loop_btn = QPushButton("🔄 循環播放：關")
        self.loop_btn.setCheckable(True)
        self.loop_btn.clicked.connect(self._on_loop_toggle)
        loop_layout.addWidget(self.loop_btn)

        loop_layout.addWidget(QLabel("小節"))
        self.loop_start = QSlider(Qt.Orientation.Horizontal)
        self.loop_start.setMinimum(1)
        self.loop_start.setMaximum(100)
        self.loop_start.setValue(1)
        self.loop_start.setMaximumWidth(100)
        loop_layout.addWidget(self.loop_start)

        loop_layout.addWidget(QLabel("到"))
        self.loop_end = QSlider(Qt.Orientation.Horizontal)
        self.loop_end.setMinimum(1)
        self.loop_end.setMaximum(100)
        self.loop_end.setValue(8)
        self.loop_end.setMaximumWidth(100)
        loop_layout.addWidget(self.loop_end)
        loop_layout.addStretch()
        layout.addLayout(loop_layout)

        # 移調
        trans_layout = QHBoxLayout()
        trans_layout.addWidget(QLabel("移調："))
        self.trans_btn = QPushButton("-")
        self.trans_spin = QLabel("0")
        self.trans_btn_plus = QPushButton("+")
        self.trans_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trans_spin.setMinimumWidth(40)
        self.trans_btn.clicked.connect(lambda: self._adjust_trans(-1))
        self.trans_btn_plus.clicked.connect(lambda: self._adjust_trans(1))
        trans_layout.addWidget(self.trans_btn)
        trans_layout.addWidget(self.trans_spin)
        trans_layout.addWidget(self.trans_btn_plus)
        trans_layout.addWidget(QLabel("半音"))
        trans_layout.addStretch()
        layout.addLayout(trans_layout)

        # 連動狀態
        link_label = QLabel("🔗 已連動自動翻譜")
        link_label.setStyleSheet("color: #0078D4; font-size: 13px;")
        layout.addWidget(link_label)

        layout.addStretch()

    def _on_play_toggle(self):
        self._is_playing = not self._is_playing
        self.play_btn.setText("⏸ 暫停" if self._is_playing else "▶ 播放")

    def _on_stop(self):
        self._is_playing = False
        self.play_btn.setText("▶ 播放")
        self.progress_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")

    def _on_speed_change(self, value):
        self.speed_label.setText(f"{value / 10:.1f}x")

    def _on_loop_toggle(self, checked):
        self.loop_btn.setText(f"🔄 循環播放：{'開' if checked else '關'}")

    def _adjust_trans(self, delta):
        current = int(self.trans_spin.text())
        new_val = max(-12, min(12, current + delta))
        self.trans_spin.setText(str(new_val))