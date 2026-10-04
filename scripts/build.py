#!/usr/bin/env python3
"""MusicMate 本地端打包腳本

使用 PyInstaller 將 MusicMate 打包為單一二進位執行檔：
- Windows: dist/MusicMate.exe
- Linux/Ubuntu: dist/MusicMate
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    print(f"=== MusicMate 打包流程 ===")
    print(f"作業系統：{platform.system()} ({platform.machine()})")
    print(f"專案目錄：{repo_root}")

    # 檢查是否安裝 pyinstaller
    try:
        import PyInstaller
        print(f"PyInstaller 版本：{PyInstaller.__version__}")
    except ImportError:
        print("未安裝 PyInstaller，正在嘗試安裝...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller>=6.0.0"])

    spec_file = repo_root / "musicmate.spec"
    if not spec_file.exists():
        print(f"錯誤：找不到 spec 檔案：{spec_file}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "-y",
        str(spec_file)
    ]

    print(f"\n執行指令：{' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        exe_ext = ".exe" if platform.system() == "Windows" else ""
        output_file = repo_root / "dist" / f"MusicMate{exe_ext}"
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print("\n🎉 打包成功！")
            print(f"產出檔案：{output_file} ({size_mb:.2f} MB)")
        else:
            print(f"\n編譯完成，請檢查 dist/ 目錄。")
    else:
        print("\n❌ 打包失敗，請檢視上方錯誤訊息。")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
