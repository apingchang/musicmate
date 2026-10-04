"""調音器半圓弧動態指針儀表組件"""

import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF
from PyQt6.QtCore import Qt, QPointF, QTimer


class TunerMeterWidget(QWidget):
    """半圓弧調音動態指針儀表 (QPainter 自繪組件)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 200)

        self._target_cents = 0.0
        self._current_cents = 0.0
        self._is_in_tune = False
        self._is_active = False

        # 平滑動畫定時器 (約 30 FPS)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._update_animation)
        self._anim_timer.start()

    def set_cents(self, cents: float, is_in_tune: bool, is_active: bool):
        """更新目標 cents 與準音狀態"""
        self._target_cents = max(-50.0, min(50.0, float(cents)))
        self._is_in_tune = is_in_tune
        self._is_active = is_active

    def _update_animation(self):
        """指針阻尼平滑逼近"""
        if not self._is_active:
            # 靜音時緩慢回到 0
            self._target_cents = 0.0

        diff = self._target_cents - self._current_cents
        if abs(diff) > 0.1:
            self._current_cents += diff * 0.35
            self.update()
        elif self._current_cents != self._target_cents:
            self._current_cents = self._target_cents
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # 中心旋轉基點
        center_x = width / 2.0
        center_y = height * 0.88
        radius = min(width * 0.42, height * 0.72)

        # 1. 繪製弧線底軌
        pen_bg = QPen(QColor(220, 224, 230), 6)
        pen_bg.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_bg)
        painter.drawArc(
            int(center_x - radius),
            int(center_y - radius),
            int(radius * 2),
            int(radius * 2),
            30 * 16,
            120 * 16
        )

        # 2. 繪製中央綠色「準音安全區間」（-5 到 +5 cents）
        pen_green = QPen(QColor(16, 124, 65), 7)
        pen_green.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen_green)
        # -5 到 +5 cents 在 120 度跨度中約佔 12 度（84° 到 96°）
        painter.drawArc(
            int(center_x - radius),
            int(center_y - radius),
            int(radius * 2),
            int(radius * 2),
            (90 - 6) * 16,
            12 * 16
        )

        # 3. 繪製刻度線與文字標記
        # cents 範圍 -50 到 +50，對應角度從 150° (左) 到 30° (右)
        ticks = [
            (-50, "-50", True),
            (-25, "-25", True),
            (-10, "-10", True),
            (-5, "-5", False),
            (0, "0", True),
            (5, "+5", False),
            (10, "+10", True),
            (25, "+25", True),
            (50, "+50", True),
        ]

        font = QFont("Segoe UI", 9)
        painter.setFont(font)

        for cents_val, label, is_major in ticks:
            # 計算該刻度的角度 (弧度)：0 cents 為垂直向上 (pi/2)
            angle_deg = 90.0 - (cents_val / 50.0) * 60.0
            angle_rad = math.radians(angle_deg)

            # 刻度線起點與終點
            inner_r = radius - (12 if is_major else 6)
            outer_r = radius + 2

            x_in = center_x + inner_r * math.cos(angle_rad)
            y_in = center_y - inner_r * math.sin(angle_rad)
            x_out = center_x + outer_r * math.cos(angle_rad)
            y_out = center_y - outer_r * math.sin(angle_rad)

            if cents_val == 0:
                pen_tick = QPen(QColor(16, 124, 65), 3)
            elif abs(cents_val) <= 5:
                pen_tick = QPen(QColor(16, 124, 65), 2)
            else:
                pen_tick = QPen(QColor(140, 140, 140), 2 if is_major else 1)

            painter.setPen(pen_tick)
            painter.drawLine(QPointF(x_in, y_in), QPointF(x_out, y_out))

            # 標籤文字
            if is_major:
                text_r = radius - 24
                tx = center_x + text_r * math.cos(angle_rad)
                ty = center_y - text_r * math.sin(angle_rad)
                painter.setPen(QColor(100, 100, 100))
                painter.drawText(int(tx - 20), int(ty - 10), 40, 20, Qt.AlignmentFlag.AlignCenter, label)

        # 4. 繪製指針 (Needle)
        needle_angle_deg = 90.0 - (self._current_cents / 50.0) * 60.0
        needle_rad = math.radians(needle_angle_deg)

        needle_length = radius - 10
        tip_x = center_x + needle_length * math.cos(needle_rad)
        tip_y = center_y - needle_length * math.sin(needle_rad)

        # 指針顏色依據是否準音與活躍狀態
        if not self._is_active:
            needle_color = QColor(180, 180, 180)
        elif self._is_in_tune:
            needle_color = QColor(16, 124, 65)  # 準音鮮明綠色
        elif abs(self._current_cents) < 15:
            needle_color = QColor(0, 120, 212)  # 接近時 Fluent 藍
        else:
            needle_color = QColor(220, 60, 60)  # 偏離較多為紅色

        # 繪製三角指針本體
        pen_needle = QPen(needle_color, 3)
        pen_needle.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_needle)
        painter.drawLine(QPointF(center_x, center_y), QPointF(tip_x, tip_y))

        # 5. 繪製底部中心轉軸圓點
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(needle_color)
        painter.drawEllipse(QPointF(center_x, center_y), 9, 9)

        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(QPointF(center_x, center_y), 4, 4)
