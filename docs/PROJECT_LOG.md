# 練琴寶 / MusicMate 專案開發日誌

## 📋 基本資訊

- **專案名稱**：練琴寶
- **英文名稱**：MusicMate
- **建立日期**：2026-04-19
- **負責人**：William（客人）/ 夥計（主持協調）

---

## 🎯 產品定位

### 核心功能
1. **自動翻譜** — 聽樂器演奏，自動辨識樂譜並翻頁
2. **節拍器** — 練琴時幫忙打拍子
3. **調音器** — 協助樂器調音

### 目標用戶
- 樂器學習者（鋼琴、小提琴、吉他等）
- 需要一邊練琴一邊翻譜的人
- 想要提升練琴效率的音樂愛好者

---

## 📝 開發記錄

### 2026-04-19

#### Step 1：專案命名
- William 提出需求：想做一個幫忙翻譜+節拍器+調音器的練琴輔助軟體
- 命名候選：練琴寶 / 琴伴 / 練譜通 / 樂聲幫手 / MusicMate
- **決定**：練琴寶（中文）/ MusicMate（英文）

#### Step 2：命名確定
- William 確認使用「練琴寶」和「MusicMate」
- 請夥計開始記錄所有討論

#### Step 2：命名確定
- William 確認使用「練琴寶」和「MusicMate」
- 請夥計開始記錄所有討論

#### Step 3：開發平台決定
- **首選開發平台**：Windows 11
- **未來支援**：Android、iOS（iPhone/iPad）、平板電腦
- **程式語言**：Python
- **優先順序**：先 Windows，未來再 porting 到行動裝置

#### Step 4：樂譜輸入方式決定
- **光學樂譜辨識（OMR）**：不論 PDF 或圖檔都先經過 OMR 識別
- **辨識目標**：音符、節拍、速度（tempo）、拍號等完整樂理資訊
- **第一版就實作**：讓用戶掃描/拍照樂譜即可使用
- **技術重點**：需要整合 OMR 引擎，使用 **OpenCV** 進行影像處理
- **OMR 方案**：使用 OpenCV 做影像前處理（如降噪、傾斜校正、二值化），搭配深度學習模型辨識音符

### Step 5：新增功能 — 樂譜播放（Audio Playback）
- **功能說明**：使用電腦音樂合成器（Soundfont/MIDI Synthesizer）播放讀進來的樂譜
- **技術需求**：
  1. OMR 辨識後的音符結構 → 轉換為 MIDI 訊息
  2. 使用電腦內建的 General MIDI 音源播放
  3.支援鋼琴音色或其他樂器音色
  4. 可調整播放速度（0.5x ~ 2.0x）
- **技術方案**：Python 的 `mido`（MIDI處理）+ `pygame` 或 `simpleaudio`（音訊播放）
- **與自動翻譜連動**：邊播放邊自動翻譜

---

## 📁 規格文件列表

| 檔案 | 說明 |
|------|------|
| `docs/MusicMate_Product_Spec_v0.2.docx` | 完整產品規格書（Word 格式） |
| `docs/MusicMate_UI_Spec_v0.1.docx` | Windows 版 UI 規格（Word 格式） |
| `projects/musicmate/docs/Chat_Log_MusicMate.docx` | 練琴寶專案聊天記錄（Word 格式） |
| `Chat_Log_Full.docx` | William 全部對話記錄（移至 workspace 根目錄） |
| `docs/UI_SPEC.md` | Windows 版 UI 規格（Markdown 備份） |
| `docs/PROJECT_LOG.md` | 本開發日誌 |
| `specs/` | 預留給詳細技術規格 |
| `scripts/gen_spec_docx.py` | 完整規格 Word 生成腳本 |

## 🎨 UI 規格摘要（v0.1 Draft）

### 整體架構
- **視窗模型**：單視窗 + 底部 Tab 導航
- **最小尺寸**：800 × 600 px
- **預設尺寸**：1200 × 800 px
- **主題**：跟隨 Windows 系統深色/淺色設定

### 三大功能頁面

#### 1. 自動翻譜頁（Sheet Music Auto-Scroll）
| 區塊 | 功能 |
|------|------|
| 樂譜顯示區 | 顯示 PDF/圖片，支援滑鼠滾輪、雙擊全螢幕 |
| 控制工具列 | 開啟樂譜、播放/暫停、速度滑桿（0.5x~3.0x）|
| 頁面導航 | 上一頁/下一頁、當前頁/總頁數顯示 |
| 快捷鍵 | Space=播放/暫停、←→=換頁、F=全螢幕 |

#### 2. 節拍器頁（Metronome）
| 區塊 | 功能 |
|------|------|
| 節拍顯示 | BPM 大字體、拍號、閃爍動畫 |
| BPM 控制 | 40~240、-10/-1/+1/+10、快捷60/80/100/120 |
| 拍號 | 2/4、3/4、4/4、5/4、6/8、7/8、12/8 |
| 音效 | Click/Woodblock/Digital、音量控制 |
| 計時器 | 5/10/15/30/45/60分鐘，倒數提醒 |

#### 3. 調音器頁（Tuner）
| 區塊 | 功能 |
|------|------|
| 指針顯示 | 半圓弧形，綠色=準確（±5 cents），紅色=需調整 |
| 音高顯示 | 音名、Cents 偏差值、Hz 頻率 |
| 麥克風 | 啟動/停用、收音指示燈 |
| 參考音 | A4 設定（430~450 Hz）、快捷432/440/442 Hz |
| 樂器預設 | 吉他/烏克麗麗/小提琴/大提琴/長笛等 |

### 設計語言
| 項目 | 說明 |
|------|------|
| 主色調 | 深藍 `#0078D4`（Fluent 藍） |
| 字體 | Segoe UI Variable |
| 間距 | 8px 基礎單位 |
| 圓角 | 4px（按鈕）、8px（卡片）、12px（面板） |

### 預留未來功能
- 音軌錄音
- 練習日誌
- 樂譜庫
- 多視窗支援

---
## 🔜 接下來

- Step 4：William 確認 UI 規格 → 提出修改意見
- Step 5：軟體工程師上線 → 設計系統架構與流程
- Step 6：正式開始寫程式

---

#### Step 5（更新）：音色選擇新增
- **額外音色**：豎笛（Clarinet）、小喇叭（Trumpet）
- 來自 William 的要求

### 2026-05-08

#### GitHub 版控確認啟用
- **Repo URL**：`https://github.com/apingchang/musicmate`
- **Repo 已存在**，初始 commit 已有基本專案結構（src/、docs/、scripts/、tests/）
- 確認 `docs/` 和 `scripts/` 有未追蹤檔案尚未 commit

#### 開發環境確認
- **正式開發環境**：Windows 11 Host OS + PyCharm Community
- **驗證環境**：Ubuntu（VirtualBox）
- **兩地共同維護同一個 GitHub repo**，透過 push/pull 同步
- **Python 版本**：3.12+

#### PyCharm 設定完成
- William 在 Windows 上成功 Clone 並開啟專案
- 設定 `src/` 為 Sources Root
- 安裝依賴：`pip install -r requirements.txt`
- **基本視窗成功顯示**，PyQt6 正常運作 ✅

#### 跨平台開發原則（確立的共識）
- Windows 和 Linux 共用同一個 repo、同一份 source code
- 平台專屬程式碼寫在 `if sys.platform == "win32"` / `"linux"` 判斷裡
- requirements.txt 可拆分成 `requirements-win.txt` / `requirements-linux.txt`
- `.idea/` 設定檔存在各自本機，透過 `.gitignore` 排除

---

*本文件由夥計維護，记录所有專案相關討論。*
