# MusicMate 練琴寶

練琴時智能助手 — 自動翻譜 + 節拍器 + 調音器 + 樂譜播放，專為 Windows 11 設計。

## 功能特色

- **自動翻譜** — 開啟 PDF 樂譜，自動依樂曲進度翻頁
- **節拍器** — 40~240 BPM，6 種音效（Click / 木魚 / 小鼓 / 叮叮聲 / 狗吠等）
- **調音器** — 即時音高偵測，支援 FFT + 自相關混合演算法
- **樂譜播放** — MIDI 合成器播放，0.5x~2.0x 速度、移調、循環練習

## 技術棧

- Python 3.12+ / PyQt6
- OMR：Audiveris + OpenCV
- MIDI：mido + pygame
- 音訊：sounddevice、numpy

## 安裝

```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安裝 dependencies
pip install -r requirements.txt
```

## 執行

```bash
python -m src.main
```

## 開發

```bash
# 執行測試
pytest

# 程式碼風格檢查
pytest --lint
```

## 打包與發布 (Packaging & Release)

### 1. 本地打包 (Local Build)
在本地環境（Windows 或 Ubuntu）一鍵打包為單一可執行檔：
```bash
python scripts/build.py
```
打包完成後可在 `dist/` 目錄找到執行檔：
- Windows: `dist/MusicMate.exe`
- Ubuntu: `dist/MusicMate`

### 2. 自動發布 (CI/CD Release)
專案配置了 GitHub Actions 跨平台矩陣打包工作流：
- **發布新版本**：推送版本標籤即可自動於 GitHub Releases 產生 Windows `.exe` 與 Ubuntu 執行檔壓縮包：
  ```bash
  git tag v1.1.0
  git push origin v1.1.0
  ```
- **手動觸發**：在 GitHub 頁面進入 **Actions** → 選擇 **Build and Release MusicMate** → 點擊 **Run workflow**。


## 授權

MIT License © 2026 William Chang