"""OMR 光學樂譜辨識處理

使用 Audiveris 進行樂譜辨識
流程：PDF → PyMuPDF 轉圖片 → OpenCV 前處理 → Audiveris CLI → MusicXML → music21 → JSON
"""

import subprocess
import json
import tempfile
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import cv2
import numpy as np


# TODO: 實作 Audiveris CLI 呼叫與錯誤處理
# TODO: 實作 music21 MusicXML 解析


class OMRProcessor:
    """OMR 處理引擎"""

    def __init__(self):
        self._audiveris_path: Optional[Path] = None
        self._timeout = 60  # 秒

    def set_audiveris_path(self, path: Path):
        """設定 Audiveris 安裝路徑"""
        self._audiveris_path = path

    def process_pdf(self, pdf_path: str) -> dict:
        """處理 PDF 樂譜，回傳音符結構（dict）

        這是主要公開 API，供 UI 層調用
        """
        # Step 1: PDF 轉圖片
        images = self._pdf_to_images(pdf_path)

        # Step 2: 圖片前處理（OpenCV）
        processed = [self._preprocess_image(img) for img in images]

        # Step 3: 呼叫 Audiveris 辨識
        musicxml_path = self._run_audiveris(processed)

        # Step 4: 解析 MusicXML 為 JSON
        return self._parse_musicxml(musicxml_path)

    def _pdf_to_images(self, pdf_path: str) -> list:
        """將 PDF 每頁轉為圖片"""
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            # 300 DPI
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img_data.reshape(pix.height, pix.width, 3)
            images.append(img)
        doc.close()
        return images

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """OpenCV 影像前處理"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        # 傾斜校正（需要偵測水平線再旋轉）
        deskewed = denoised  # TODO: 實作
        binary = cv2.adaptiveThreshold(
            deskewed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return binary

    def _run_audiveris(self, processed_images: list) -> Path:
        """呼叫 Audiveris CLI 進行辨識"""
        # TODO: 實作 subprocess 呼叫
        #   java -jar audiveris-cli.jar -export -output out/ score.pdf
        #   讀取输出的 .omr 檔案中的 MusicXML
        raise NotImplementedError("Audiveris 整合待實作")

    def _parse_musicxml(self, musicxml_path: Path) -> dict:
        """使用 music21 解析 MusicXML"""
        # TODO: 實作 music21 解析
        #   from music21 import converter
        #   score = converter.parse(musicxml_path)
        #   轉換為內部 JSON 結構
        raise NotImplementedError("MusicXML 解析待實作")