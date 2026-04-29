"""節拍器頁"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QButtonGroup
)
from PyQt6.QtCore import Qt, QTimer


class MetronomePage(QWidget):
    """節拍器頁面"""

    def __init__(self):
        super().__init__()
        self._bpm = 120
        self._is_playing = False
        self._current_beat = 0
        self._beats_per_measure = 4
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # BPM 大字體顯示
        self.bpm_label = QLabel("120")
        self.bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bpm_label.setStyleSheet(
            "font-size: 96px; font-weight: bold; color: #0078D4;"
        )
        bpm_desc = QLabel("BPM")
        bpm_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bpm_desc.setStyleSheet("font-size: 20px; color: #666;")

        # BPM 微調按鈕
        bpm_control = QHBoxLayout()
        for delta, label in [(-10, "-10"), (-1, "-1"), (1, "+1"), (10, "+10")]:
            btn = QPushButton(label)
            btn.setFixedWidth(60)
            btn.clicked.connect(lambda checked, d=delta: self._adjust_bpm(d))
            bpm_control.addWidget(btn)

        # 快捷 BPM 按鈕
        quick_bpm = QHBoxLayout()
        for bpm in [60, 80, 100, 120]:
            btn = QPushButton(str(bpm))
            btn.setFixedWidth(60)
            btn.clicked.connect(lambda checked, b=bpm: self._set_bpm(b))
            quick_bpm.addWidget(btn)

        # 拍號選擇
        ts_label = QLabel("拍號")
        ts_label.setStyleSheet("font-weight: bold;")
        time_sig_group = QButtonGroup()
        time_sigs = [("2/4", 2), ("3/4", 3), ("4/4", 4),
                     ("5/4", 5), ("6/8", 6), ("7/8", 7), ("12/8", 12)]
        ts_layout = QHBoxLayout()
        self._ts_buttons = {}
        for label, beats in time_sigs:
            btn = QPushButton(label)
            btn.setFixedWidth(50)
            btn.setCheckable(True)
            if beats == 4:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, b=beats: self._set_time_sig(b))
            time_sig_group.addButton(btn)
            ts_layout.addWidget(btn)
            self._ts_buttons[beats] = btn

        # 節拍指示燈
        self.beat_indicators = QHBoxLayout()
        self._beat_lights = []
        for i in range(12):
            light = QLabel("●")
            light.setAlignment(Qt.AlignmentFlag.AlignCenter)
            light.setStyleSheet("font-size: 28px; color: #ccc;")
            self._beat_lights.append(light)
            self.beat_indicators.addWidget(light)
        self._update_indicator_state()

        # 音效選擇
        sound_label = QLabel("音效")
        sound_label.setStyleSheet("font-weight: bold;")
        sound_layout = QHBoxLayout()
        sounds = ["Click", "木魚", "Digital", "小鼓", "叮叮聲", "狗吠"]
        self._sound_buttons = {}
        sound_group = QButtonGroup()
        for i, sound in enumerate(sounds):
            btn = QPushButton(sound)
            btn.setCheckable(True)
            if i == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, s=sound: self._set_sound(s))
            sound_group.addButton(btn)
            sound_layout.addWidget(btn)
            self._sound_buttons[sound] = btn

        # 音量
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("🔊"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(70)
        self.volume_slider.setMaximumWidth(200)
        vol_layout.addWidget(self.volume_slider)
        vol_layout.addStretch()

        # 計時器
        timer_label = QLabel("計時器")
        timer_label.setStyleSheet("font-weight: bold;")
        timer_layout = QHBoxLayout()
        for minutes in [5, 10, 15, 30, 45, 60]:
            btn = QPushButton(f"{minutes}min")
            btn.setFixedWidth(60)
            timer_layout.addWidget(btn)

        # 啟動/停止
        self.start_btn = QPushButton("▶ START")
        self.start_btn.setFixedHeight(50)
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; "
            "font-size: 18px; border-radius: 8px; }"
        )
        self.start_btn.clicked.connect(self._on_toggle)

        layout.addWidget(self.bpm_label)
        layout.addWidget(bpm_desc)
        layout.addLayout(bpm_control)
        layout.addLayout(quick_bpm)
        layout.addSpacing(10)
        layout.addWidget(ts_label)
        layout.addLayout(ts_layout)
        layout.addSpacing(10)
        layout.addLayout(self.beat_indicators)
        layout.addSpacing(10)
        layout.addWidget(sound_label)
        layout.addLayout(sound_layout)
        layout.addLayout(vol_layout)
        layout.addSpacing(10)
        layout.addWidget(timer_label)
        layout.addLayout(timer_layout)
        layout.addWidget(self.start_btn)
        layout.addStretch()

    def _adjust_bpm(self, delta):
        self._bpm = max(40, min(240, self._bpm + delta))
        self.bpm_label.setText(str(self._bpm))

    def _set_bpm(self, bpm):
        self._bpm = bpm
        self.bpm_label.setText(str(self._bpm))

    def _set_time_sig(self, beats):
        self._beats_per_measure = beats
        self._current_beat = 0
        self._update_indicator_state()

    def _set_sound(self, sound):
        # TODO: 實作音效切換
        pass

    def _update_indicator_state(self):
        for i, light in enumerate(self._beat_lights):
            if i < self._beats_per_measure:
                if i == self._current_beat:
                    color = "#0078D4" if i == 0 else "#0099DD"
                else:
                    color = "#ccc"
                light.setStyleSheet(f"font-size: 28px; color: {color};")
            else:
                light.setStyleSheet("font-size: 28px; color: #eee;")

    def _on_toggle(self):
        self._is_playing = not self._is_playing
        if self._is_playing:
            self.start_btn.setText("⏹ STOP")
            self.start_btn.setStyleSheet(
                "QPushButton { background-color: #d32f2f; color: white; "
                "font-size: 18px; border-radius: 8px; }"
            )
        else:
            self.start_btn.setText("▶ START")
            self.start_btn.setStyleSheet(
                "QPushButton { background-color: #0078D4; color: white; "
                "font-size: 18px; border-radius: 8px; }"
            )
            self._current_beat = 0
            self._update_indicator_state()