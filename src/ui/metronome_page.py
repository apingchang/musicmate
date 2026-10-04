"""節拍器頁

整合：
- 跨平台精準打拍發聲引擎 (Metronome Core)
- 動態節拍指示燈動畫 (pyqtSignal 跨執行緒通知)
- 🥁 Tap Tempo 點擊打拍測速功能
- ⏳ 練琴倒數計時器（5~60分鐘設定、即時倒數、完成提示鈴聲）
"""

import time
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QButtonGroup, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from src.core.metronome import Metronome


class MetronomePage(QWidget):
    """節拍器頁面"""

    # 定義跨執行緒安全的 Qt 信號：(beat_index, is_accent)
    tick_signal = pyqtSignal(int, bool)

    def __init__(self):
        super().__init__()
        self._bpm = 120
        self._is_playing = False
        self._current_beat = 0
        self._beats_per_measure = 4

        # Tap Tempo 點擊時間記錄
        self._tap_times = []

        # 練琴計時器狀態
        self._timer_duration_secs = 15 * 60  # 預設 15 分鐘
        self._timer_remaining_secs = 15 * 60
        self._timer_running = False
        self._practice_timer = QTimer(self)
        self._practice_timer.setInterval(1000)
        self._practice_timer.timeout.connect(self._on_timer_tick)

        # 初始化後端節拍器引擎
        self.metronome = Metronome()
        self.metronome.bpm = self._bpm
        self.metronome.beats_per_measure = self._beats_per_measure
        self.metronome.volume = 0.7
        self.metronome.set_tick_callback(self._on_metronome_tick)

        # 連接每拍信號到 UI 更新槽函式
        self.tick_signal.connect(self._handle_ui_tick)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 16, 24, 16)

        # 1. BPM 大字體顯示
        self.bpm_label = QLabel(str(self._bpm))
        self.bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bpm_label.setStyleSheet(
            "font-size: 80px; font-weight: bold; color: #0078D4; margin: 0px;"
        )
        bpm_desc = QLabel("BPM (拍/分鐘)")
        bpm_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bpm_desc.setStyleSheet("font-size: 16px; color: #666; margin-bottom: 4px;")

        # 2. BPM 微調與 Tap Tempo 控制區
        bpm_control = QHBoxLayout()
        bpm_control.addStretch()
        for delta, label in [(-10, "-10"), (-1, "-1"), (1, "+1"), (10, "+10")]:
            btn = QPushButton(label)
            btn.setFixedWidth(54)
            btn.setFixedHeight(34)
            btn.clicked.connect(lambda checked, d=delta: self._adjust_bpm(d))
            bpm_control.addWidget(btn)

        # 🥁 Tap Tempo 按鈕
        self.tap_btn = QPushButton("🥁 Tap Tempo (打拍測速)")
        self.tap_btn.setFixedHeight(34)
        self.tap_btn.setStyleSheet(
            "QPushButton { background-color: #f3f3f3; color: #0078D4; font-weight: bold; border: 1px solid #ccc; border-radius: 4px; padding: 0 12px; }"
            "QPushButton:hover { background-color: #e5f1fb; border-color: #0078D4; }"
        )
        self.tap_btn.clicked.connect(self._on_tap_tempo)
        bpm_control.addWidget(self.tap_btn)
        bpm_control.addStretch()

        # 快捷 BPM 按鈕
        quick_bpm = QHBoxLayout()
        quick_bpm.addStretch()
        quick_bpm.addWidget(QLabel("快捷："))
        for bpm in [60, 80, 100, 120, 140, 160]:
            btn = QPushButton(str(bpm))
            btn.setFixedWidth(52)
            btn.clicked.connect(lambda checked, b=bpm: self._set_bpm(b))
            quick_bpm.addWidget(btn)
        quick_bpm.addStretch()

        # 3. 拍號選擇
        ts_label = QLabel("拍號選擇")
        ts_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        time_sig_group = QButtonGroup(self)
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
        ts_layout.addStretch()

        # 4. 節拍指示燈
        self.beat_indicators = QHBoxLayout()
        self._beat_lights = []
        for i in range(12):
            light = QLabel("●")
            light.setAlignment(Qt.AlignmentFlag.AlignCenter)
            light.setStyleSheet("font-size: 26px; color: #ccc;")
            self._beat_lights.append(light)
            self.beat_indicators.addWidget(light)
        self._update_indicator_state()

        # 5. 音效與音量
        sound_header = QLabel("音色與音量")
        sound_header.setStyleSheet("font-weight: bold; font-size: 13px;")
        sound_layout = QHBoxLayout()
        self.sounds = ["Click", "木魚", "Digital", "小鼓", "叮叮聲", "狗吠"]
        self._sound_buttons = {}
        sound_group = QButtonGroup(self)
        for i, sound in enumerate(self.sounds):
            btn = QPushButton(sound)
            btn.setCheckable(True)
            if i == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, idx=i: self._set_sound(idx))
            sound_group.addButton(btn)
            sound_layout.addWidget(btn)
            self._sound_buttons[sound] = btn

        sound_layout.addSpacing(16)
        sound_layout.addWidget(QLabel("🔊"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(70)
        self.volume_slider.setMaximumWidth(160)
        self.volume_slider.valueChanged.connect(self._on_volume_change)
        sound_layout.addWidget(self.volume_slider)
        sound_layout.addStretch()

        # 6. 練琴倒數計時器區塊
        timer_header = QLabel("練琴倒數計時器")
        timer_header.setStyleSheet("font-weight: bold; font-size: 13px;")
        timer_layout = QHBoxLayout()
        self._timer_buttons = {}
        timer_btn_group = QButtonGroup(self)
        for minutes in [5, 10, 15, 30, 45, 60]:
            btn = QPushButton(f"{minutes}min")
            btn.setFixedWidth(56)
            btn.setCheckable(True)
            if minutes == 15:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, m=minutes: self._select_timer_duration(m))
            timer_btn_group.addButton(btn)
            timer_layout.addWidget(btn)
            self._timer_buttons[minutes] = btn

        timer_layout.addSpacing(16)
        # 倒數時間顯示
        self.timer_display = QLabel("15:00")
        self.timer_display.setStyleSheet("font-size: 24px; font-weight: bold; color: #107C41; padding: 0 8px;")
        timer_layout.addWidget(self.timer_display)

        # 計時器控制按鈕
        self.timer_toggle_btn = QPushButton("▶ 開始計時")
        self.timer_toggle_btn.setFixedWidth(90)
        self.timer_toggle_btn.clicked.connect(self._on_timer_toggle)
        timer_layout.addWidget(self.timer_toggle_btn)

        self.timer_reset_btn = QPushButton("🔄 重設")
        self.timer_reset_btn.setFixedWidth(64)
        self.timer_reset_btn.clicked.connect(self._on_timer_reset)
        timer_layout.addWidget(self.timer_reset_btn)
        timer_layout.addStretch()

        # 7. 主啟動/停止按鈕
        self.start_btn = QPushButton("▶ START")
        self.start_btn.setFixedHeight(50)
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; "
            "font-size: 20px; font-weight: bold; border-radius: 8px; }"
            "QPushButton:hover { background-color: #106EBE; }"
        )
        self.start_btn.clicked.connect(self._on_toggle)

        # 加入佈局
        layout.addWidget(self.bpm_label)
        layout.addWidget(bpm_desc)
        layout.addLayout(bpm_control)
        layout.addLayout(quick_bpm)
        layout.addSpacing(4)
        layout.addWidget(ts_label)
        layout.addLayout(ts_layout)
        layout.addSpacing(4)
        layout.addLayout(self.beat_indicators)
        layout.addSpacing(4)
        layout.addWidget(sound_header)
        layout.addLayout(sound_layout)
        layout.addSpacing(4)
        layout.addWidget(timer_header)
        layout.addLayout(timer_layout)
        layout.addSpacing(8)
        layout.addWidget(self.start_btn)
        layout.addStretch()

    def _adjust_bpm(self, delta):
        self._bpm = max(40, min(240, self._bpm + delta))
        self.bpm_label.setText(str(self._bpm))
        self.metronome.bpm = self._bpm

    def _set_bpm(self, bpm):
        self._bpm = max(40, min(240, int(bpm)))
        self.bpm_label.setText(str(self._bpm))
        self.metronome.bpm = self._bpm

    def _on_tap_tempo(self):
        """Tap Tempo 點擊測速"""
        now = time.perf_counter()
        # 若距離上次點擊超過 2.0 秒，清空重計
        if self._tap_times and (now - self._tap_times[-1]) > 2.0:
            self._tap_times.clear()

        self._tap_times.append(now)
        # 最多保留最近 6 次點擊
        if len(self._tap_times) > 6:
            self._tap_times.pop(0)

        if len(self._tap_times) >= 2:
            diffs = [self._tap_times[i] - self._tap_times[i - 1] for i in range(1, len(self._tap_times))]
            avg_diff = sum(diffs) / len(diffs)
            if avg_diff > 0:
                bpm = int(round(60.0 / avg_diff))
                bpm = max(40, min(240, bpm))
                self._set_bpm(bpm)
                self.tap_btn.setText(f"🥁 Tap: {bpm} BPM")
        else:
            self.tap_btn.setText("🥁 再點一下測速...")

    def _set_time_sig(self, beats):
        self._beats_per_measure = beats
        self._current_beat = 0
        self.metronome.beats_per_measure = beats
        self._update_indicator_state()

    def _set_sound(self, sound_index: int):
        self.metronome.set_sound(sound_index)

    def _on_volume_change(self, value: int):
        self.metronome.volume = value / 100.0

    # ----- 練琴計時器邏輯 -----
    def _select_timer_duration(self, minutes: int):
        """選擇計時器時長"""
        self._timer_duration_secs = minutes * 60
        self._on_timer_reset()

    def _update_timer_display(self):
        """更新倒數時間顯示"""
        mins = self._timer_remaining_secs // 60
        secs = self._timer_remaining_secs % 60
        self.timer_display.setText(f"{mins:02d}:{secs:02d}")

    def _on_timer_toggle(self):
        """開始/暫停計時"""
        if not self._timer_running:
            if self._timer_remaining_secs <= 0:
                self._timer_remaining_secs = self._timer_duration_secs
            self._practice_timer.start()
            self._timer_running = True
            self.timer_toggle_btn.setText("⏸ 暫停")
        else:
            self._practice_timer.stop()
            self._timer_running = False
            self.timer_toggle_btn.setText("▶ 繼續計時")

    def _on_timer_reset(self):
        """重設計時器"""
        self._practice_timer.stop()
        self._timer_running = False
        self._timer_remaining_secs = self._timer_duration_secs
        self._update_timer_display()
        self.timer_toggle_btn.setText("▶ 開始計時")

    def _on_timer_tick(self):
        """每秒倒數"""
        if self._timer_remaining_secs > 0:
            self._timer_remaining_secs -= 1
            self._update_timer_display()

        if self._timer_remaining_secs <= 0:
            self._practice_timer.stop()
            self._timer_running = False
            self.timer_toggle_btn.setText("▶ 開始計時")
            self._on_timer_finished()

    def _on_timer_finished(self):
        """計時結束提醒"""
        # 播放 3 聲叮叮提示音
        try:
            self.metronome.set_sound(4)  # 4: 叮叮聲
            for _ in range(3):
                self.metronome._play_sound(is_accent=True)
                time.sleep(0.18)
        except Exception:
            pass

        # 若節拍器在運行，自動停止
        if self._is_playing:
            self._on_toggle()

        mins = self._timer_duration_secs // 60
        QMessageBox.information(
            self,
            "練琴計時完成 🎉",
            f"⏰ 恭喜！您已完成本次 {mins} 分鐘練琴目標！\n\n適度休息與放鬆手指能讓練習效果更好哦！"
        )

    # ----- 節拍器核心連動 -----
    def _on_metronome_tick(self, beat_index: int, is_accent: bool):
        self.tick_signal.emit(beat_index, is_accent)

    def _handle_ui_tick(self, beat_index: int, is_accent: bool):
        self._current_beat = beat_index
        self._update_indicator_state()

    def _update_indicator_state(self):
        for i, light in enumerate(self._beat_lights):
            if i < self._beats_per_measure:
                if self._is_playing and i == self._current_beat:
                    color = "#E81123" if i == 0 else "#0078D4"
                else:
                    color = "#d0d0d0"
                light.setStyleSheet(f"font-size: 26px; color: {color};")
            else:
                light.setStyleSheet("font-size: 26px; color: #f2f2f2;")

    def _on_toggle(self):
        self._is_playing = not self._is_playing
        if self._is_playing:
            self.start_btn.setText("⏹ STOP")
            self.start_btn.setStyleSheet(
                "QPushButton { background-color: #d32f2f; color: white; "
                "font-size: 20px; font-weight: bold; border-radius: 8px; }"
                "QPushButton:hover { background-color: #b71c1c; }"
            )
            self._current_beat = 0
            self.metronome.start()
        else:
            self.start_btn.setText("▶ START")
            self.start_btn.setStyleSheet(
                "QPushButton { background-color: #0078D4; color: white; "
                "font-size: 20px; font-weight: bold; border-radius: 8px; }"
                "QPushButton:hover { background-color: #106EBE; }"
            )
            self.metronome.stop()
            self._current_beat = 0
            self._update_indicator_state()

    def closeEvent(self, event):
        if self.metronome:
            self.metronome.stop()
        if self._practice_timer:
            self._practice_timer.stop()
        super().closeEvent(event)