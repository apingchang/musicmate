# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

block_cipher = None

# 取得專案根目錄路徑
project_root = Path('.').resolve()

datas = []

# 打包內建真實樂器 SoundFont 音色庫等資源目錄
if (project_root / "assets").exists():
    datas.append(('assets', 'assets'))

hidden_imports = [
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'fitz',
    'cv2',
    'numpy',
    'sounddevice',
    'cffi',
    'mido',
    'pygame',
    'music21',
    'tinysoundfont',
    '_tinysoundfont',
]

a = Analysis(
    ['src/main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MusicMate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
