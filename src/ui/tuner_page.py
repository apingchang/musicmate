"""調音器頁面

整合跨平台麥克風收音、YIN 音高偵測演算法、半圓弧動態儀表與樂器預設弦對應
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QButtonGroup, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from src.core.tuner import Tuner
from src.ui.tuner_meter import TunerMeterWidget


class TunerPage(QWidget):
    """調音器頁面"""

    # 跨執行緒安全 Qt 信號：(note_name, octave, cents, frequency, is_in_tune)
    pitch_signal = pyqtSignal(str, int, float, float, bool)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._ref_a4 = 440.0
        self._current_instrument = "吉他"

        # 核心引擎實例化
        self.tuner = Tuner()
        self.tuner.ref_a4 = self._ref_a4
        self.tuner.instrument = self._current_instrument
        self.tuner.set_pitch_callback(self._on_core_pitch)
        self.tuner.set_error_callback(self._on_core_error)

        self.pitch_signal.connect(self._handle_ui_pitch)
        self.error_signal.connect(self._handle_ui_error)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 16, 24, 16)

        # 1. 半圓弧動態指針儀表
        self.meter_widget = TunerMeterWidget(self)
        layout.addWidget(self.meter_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        # 2. 音高超大字體顯示
        self.note_label = QLabel("--")
        self.note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note_label.setStyleSheet("font-size: 72px; font-weight: bold; color: #333333;")

        # Cents 偏差值與 Hz 頻率顯示
        self.cents_label = QLabel("請啟動麥克風並彈奏樂器單音")
        self.cents_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cents_label.setStyleSheet("font-size: 18px; color: #666666; font-weight: 500;")

        # 目標弦提示
        self.target_label = QLabel("標準吉他定音：E2 - A2 - D3 - G3 - B3 - E4")
        self.target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_label.setStyleSheet("font-size: 14px; color: #0078D4; margin-bottom: 4px;")

        layout.addWidget(self.note_label)
        layout.addWidget(self.cents_label)
        layout.addWidget(self.target_label)

        # 3. 麥克風控制按鈕
        mic_layout = QHBoxLayout()
        self.mic_btn = QPushButton("🎤 啟動麥克風")
        self.mic_btn.setFixedHeight(50)
        self.mic_btn.setMinimumWidth(260)
        self.mic_btn.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; "
            "font-size: 18px; font-weight: bold; border-radius: 8px; }"
            "QPushButton:hover { background-color: #106EBE; }"
        )
        self.mic_btn.clicked.connect(self._on_mic_toggle)
        mic_layout.addStretch()
        mic_layout.addWidget(self.mic_btn)
        mic_layout.addStretch()
        layout.addLayout(mic_layout)

        # 狀態提示小字
        self.status_tip = QLabel("狀態：麥克風已就緒（點擊啟動開始調音）")
        self.status_tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_tip.setStyleSheet("font-size: 12px; color: #888;")
        layout.addWidget(self.status_tip)

        layout.addSpacing(6)

        # 4. 參考音 A4 選擇
        ref_header = QLabel("參考音 A4 基準頻率")
        ref_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        ref_layout = QHBoxLayout()
        ref_group = QButtonGroup(self)
        for freq, label in [(432, "432 Hz"), (440, "440 Hz (標準)"), (442, "442 Hz (管弦)"), (443, "443 Hz")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            if freq == 440:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, f=freq: self._set_ref_a4(f))
            ref_group.addButton(btn)
            ref_layout.addWidget(btn)
        ref_layout.addStretch()

        # 5. 樂器預設切換
        inst_header = QLabel("樂器預設")
        inst_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        inst_layout = QHBoxLayout()
        instruments = ["吉他", "烏克麗麗", "小提琴", "大提琴", "長笛", "鋼琴"]
        self._inst_buttons = {}
        inst_group = QButtonGroup(self)
        for i, inst in enumerate(instruments):
            btn = QPushButton(inst)
            btn.setCheckable(True)
            if i == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, name=inst: self._set_instrument(name))
            inst_group.addButton(btn)
            inst_layout.addWidget(btn)
            self._inst_buttons[inst] = btn
        inst_layout.addStretch()

        layout.addWidget(ref_header)
        layout.addLayout(ref_layout)
        layout.addSpacing(6)
        layout.addWidget(inst_header)
        layout.addLayout(inst_layout)
        layout.addStretch()

    def _on_mic_toggle(self):
        """切換麥克風收音"""
        if not self.tuner.is_listening:
            success = self.tuner.start()
            if success:
                self.mic_btn.setText("⏹ 停止麥克風")
                self.mic_btn.setStyleSheet(
                    "QPushButton { background-color: #d32f2f; color: white; "
                    "font-size: 18px; font-weight: bold; border-radius: 8px; }"
                    "QPushButton:hover { background-color: #b71c1c; }"
                )
                self.status_tip.setText("狀態：● 正在聆聽麥克風音訊輸入...")
                self.status_tip.setStyleSheet("font-size: 12px; color: #107C41; font-weight: bold;")
                self.cents_label.setText("請彈奏樂器單音...")
        else:
            self.tuner.stop()
            self.mic_btn.setText("🎤 啟動麥克風")
            self.mic_btn.setStyleSheet(
                "QPushButton { background-color: #0078D4; color: white; "
                "font-size: 18px; font-weight: bold; border-radius: 8px; }"
                "QPushButton:hover { background-color: #106EBE; }"
            )
            self.status_tip.setText("狀態：麥克風已停止")
            self.status_tip.setStyleSheet("font-size: 12px; color: #888;")
            self.note_label.setText("--")
            self.note_label.setStyleSheet("font-size: 72px; font-weight: bold; color: #333333;")
            self.cents_label.setText("麥克風已關閉")
            self.meter_widget.set_cents(0, False, False)

    def _on_core_pitch(self, note: str, octave: int, cents: float, freq: float, in_tune: bool):
        self.pitch_signal.emit(note, octave, cents, freq, in_tune)

    def _on_core_error(self, message: str):
        self.error_signal.emit(message)

    def _handle_ui_pitch(self, note: str, octave: int, cents: float, freq: float, in_tune: bool):
        """在 Qt 主執行緒更新調音器畫面"""
        if not self.tuner.is_listening:
            return

        if note == "--":
            # 靜音中
            self.note_label.setText("--")
            self.note_label.setStyleSheet("font-size: 72px; font-weight: bold; color: #888888;")
            self.cents_label.setText("聆聽中... (請彈奏單音)")
            self.meter_widget.set_cents(0.0, False, False)
            return

        # 顯示音名與八度
        note_str = f"{note}{octave}"
        cents_sign = f"+{cents:.1f}" if cents > 0 else f"{cents:.1f}"

        if in_tune:
            # 準確 (±5 cents 以內)：鮮明綠色提示
            self.note_label.setText(f"{note_str}  ✓")
            self.note_label.setStyleSheet("font-size: 72px; font-weight: bold; color: #107C41;")
            self.cents_label.setText(f"標準準音！ ({cents_sign} cents | {freq:.1f} Hz)")
            self.cents_label.setStyleSheet("font-size: 18px; color: #107C41; font-weight: bold;")
        else:
            self.note_label.setText(note_str)
            # 根據偏低或偏高提示
            direction = "偏低（請鎖緊弦）" if cents < 0 else "偏高（請放鬆弦）"
            self.note_label.setStyleSheet("font-size: 72px; font-weight: bold; color: #0078D4;")
            self.cents_label.setText(f"{cents_sign} cents  •  {direction}  ({freq:.1f} Hz)")
            self.cents_label.setStyleSheet("font-size: 18px; color: #333333; font-weight: 500;")

        # 更新半圓弧動態儀表指針
        self.meter_widget.set_cents(cents, in_tune, True)

        # 比對樂器預設弦
        target = self.tuner.get_closest_preset_target(freq)
        if target and target[0] != "--":
            t_name, t_freq, t_diff = target
            diff_sign = f"+{t_diff:.1f}" if t_diff > 0 else f"{t_diff:.1f}"
            self.target_label.setText(f"目前對應：{t_name}（標準 {t_freq:.1f} Hz，相差 {diff_sign} Hz）")

    def _handle_ui_error(self, message: str):
        """處理收音錯誤"""
        self._on_mic_toggle()  # 復原按鈕狀態
        self.status_tip.setText(f"錯誤：{message}")
        self.status_tip.setStyleSheet("font-size: 12px; color: #d32f2f; font-weight: bold;")
        QMessageBox.warning(self, "麥克風存取提示", f"{message}\n\n請確認麥克風已接上，且系統隱私設定已允許存取麥克風。")

    def _set_ref_a4(self, freq: float):
        self._ref_a4 = freq
        self.tuner.ref_a4 = freq

    def _set_instrument(self, name: str):
        self._current_instrument = name
        self.tuner.instrument = name
        presets = self.tuner.INSTRUMENTS_PRESETS.get(name, [])
        if presets and presets[0][1] > 0:
            notes_str = " - ".join([f"{item[0]}" for item in presets])
            self.target_label.setText(f"{name}定音：{notes_str}")
        else:
            self.target_label.setText(f"{name}：全音域自動偵測")

    def closeEvent(self, event):
        """關閉時安全停止麥克風"""
        if self.tuner:
            self.tuner.stop()
        super().closeEvent(event)