"""調音器頁"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QButtonGroup
)
from PyQt6.QtCore import Qt


class TunerPage(QWidget):
    """調音器頁面"""

    def __init__(self):
        super().__init__()
        self._mic_active = False
        self._ref_a4 = 440
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 半圓弧形指針儀表（Placeholder）
        meter_label = QLabel(
            "🎯\n\n"
            "●━━━━━━━━━●\n"
            "  -5    0    +5   cents"
        )
        meter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meter_label.setStyleSheet(
            "font-size: 48px; color: #0078D4; padding: 20px;"
        )
        layout.addWidget(meter_label)

        # 音高顯示
        self.note_label = QLabel("--")
        self.note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note_label.setStyleSheet("font-size: 64px; font-weight: bold; color: #333;")
        self.cents_label = QLabel("0 cents | -- Hz")
        self.cents_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cents_label.setStyleSheet("font-size: 18px; color: #666;")

        # 麥克風按鈕
        self.mic_btn = QPushButton("🎤 啟動")
        self.mic_btn.setFixedHeight(50)
        self.mic_btn.setStyleSheet(
            "QPushButton { background-color: #666; color: white; "
            "font-size: 18px; border-radius: 8px; }"
        )
        self.mic_btn.clicked.connect(self._on_mic_toggle)
        layout.addWidget(self.mic_btn)

        # 參考音
        ref_label = QLabel("參考音 A4")
        ref_label.setStyleSheet("font-weight: bold;")
        ref_layout = QHBoxLayout()
        ref_group = QButtonGroup()
        for freq, label in [(432, "432 Hz"), (440, "440 Hz"), (442, "442 Hz"), (443, "443 Hz")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            if freq == 440:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, f=freq: self._set_ref_a4(f))
            ref_group.addButton(btn)
            ref_layout.addWidget(btn)

        # 樂器預設
        inst_label = QLabel("樂器預設")
        inst_label.setStyleSheet("font-weight: bold;")
        inst_layout = QHBoxLayout()
        instruments = ["吉他", "烏克麗麗", "小提琴", "大提琴", "長笛", "鋼琴"]
        self._inst_buttons = {}
        inst_group = QButtonGroup()
        for inst in instruments:
            btn = QPushButton(inst)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, i=inst: self._set_instrument(i))
            inst_group.addButton(btn)
            inst_layout.addWidget(btn)
            self._inst_buttons[inst] = btn

        layout.addWidget(ref_label)
        layout.addLayout(ref_layout)
        layout.addWidget(inst_label)
        layout.addLayout(inst_layout)
        layout.addStretch()

    def _on_mic_toggle(self):
        self._mic_active = not self._mic_active
        if self._mic_active:
            self.mic_btn.setText("🎤 停用")
            self.mic_btn.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; "
                "font-size: 18px; border-radius: 8px; }"
            )
        else:
            self.mic_btn.setText("🎤 啟動")
            self.mic_btn.setStyleSheet(
                "QPushButton { background-color: #666; color: white; "
                "font-size: 18px; border-radius: 8px; }"
            )
            self.note_label.setText("--")
            self.cents_label.setText("0 cents | -- Hz")

    def _set_ref_a4(self, freq):
        self._ref_a4 = freq

    def _set_instrument(self, instrument):
        # TODO: 根據樂器調整調音範圍
        pass