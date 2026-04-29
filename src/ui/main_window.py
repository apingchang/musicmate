"""主視窗 - 包含四個 Tab 頁面"""

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QLabel, QTabBar, QMessageBox
)
from PyQt6.QtCore import Qt

from src.ui.auto_scroll_page import AutoScrollPage
from src.ui.metronome_page import MetronomePage
from src.ui.tuner_page import TunerPage
from src.ui.playback_page import PlaybackPage


class MainWindow(QMainWindow):
    """練琴寶主視窗"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("練琴寶 MusicMate")
        self.setMinimumSize(1200, 800)

        self._setup_ui()
        self._create_menu()

    def _setup_ui(self):
        """建立 UI 介面"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.addTab(AutoScrollPage(), "自動翻譜")
        self.tabs.addTab(MetronomePage(), "節拍器")
        self.tabs.addTab(TunerPage(), "調音器")
        self.tabs.addTab(PlaybackPage(), "樂譜播放")

        layout.addWidget(self.tabs)

    def _create_menu(self):
        """建立選單"""
        menubar = self.menuBar()

        # 檔案選單
        file_menu = menubar.addMenu("檔案")

        open_action = file_menu.addAction("開啟樂譜...")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)

        recent_menu = file_menu.addMenu("最近開啟的檔案")
        recent_menu.setEnabled(False)  # TODO: 實作最近檔案功能

        file_menu.addSeparator()
        exit_action = file_menu.addAction("離開")
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)

        # 設定選單
        settings_menu = menubar.addMenu("設定")
        prefs_action = settings_menu.addAction("偏好設定...")
        prefs_action.setShortcut("Ctrl+,")

        # 說明選單
        help_menu = menubar.addMenu("說明")
        about_action = help_menu.addAction("關於練琴寶")
        about_action.triggered.connect(self._show_about)

    def _open_file(self):
        """開啟樂譜檔案"""
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "開啟樂譜",
            "",
            "樂譜檔案 (*.pdf *.pdf);;圖片 (*.jpg *.png);;所有檔案 (*)"
        )
        if file_path:
            # 切換到自動翻譜頁並載入
            self.tabs.setCurrentIndex(0)
            page = self.tabs.widget(0)
            page.load_file(file_path)

    def _show_about(self):
        """關於對話框"""
        QMessageBox.about(
            self,
            "關於練琴寶",
            "<b>練琴寶 MusicMate</b><br>"
            f"版本：1.1.0<br>"
            "© 2026 William Chang<br><br>"
            "練琴時智能助手：自動翻譜 + 節拍器<br>"
            "+ 調音器 + 樂譜播放<br><br>"
            "授權條款：MIT License"
        )

    def closeEvent(self, event):
        """關閉程式時的處理"""
        reply = QMessageBox.question(
            self,
            "確認離開",
            "確定要關閉練琴寶嗎？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()