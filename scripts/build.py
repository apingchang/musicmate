#!/usr/bin/env python3
"""MusicMate 本地端打包腳本

在本地快速打包為二進位執行檔，並自動放置到 release/ 資料夾：
- Linux/Ubuntu: release/MusicMate
- Windows: release/MusicMate.exe
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    print(f"=== MusicMate 本地打包流程 ===")
    print(f"作業系統：{platform.system()} ({platform.machine()})")
    print(f"專案目錄：{repo_root}")

    # 檢查是否有 .venv
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    
    python_bin = str(venv_python) if venv_python.exists() else sys.executable
    print(f"使用 Python 環境：{python_bin}")

    # 檢查 pyinstaller
    check_pi = subprocess.run([python_bin, "-m", "PyInstaller", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check_pi.returncode != 0:
        print("未安裝 PyInstaller，正在安裝依賴...")
        subprocess.check_call([python_bin, "-m", "pip", "install", "-r", "requirements.txt"])

    spec_file = repo_root / "musicmate.spec"
    if not spec_file.exists():
        print(f"錯誤：找不到 spec 檔案：{spec_file}")
        sys.exit(1)

    cmd = [
        python_bin,
        "-m",
        "PyInstaller",
        "--clean",
        "-y",
        str(spec_file)
    ]

    print(f"\n執行打包指令：{' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        exe_ext = ".exe" if platform.system() == "Windows" else ""
        built_file = repo_root / "dist" / f"MusicMate{exe_ext}"
        release_dir = repo_root / "release"
        release_dir.mkdir(exist_ok=True)
        target_file = release_dir / f"MusicMate{exe_ext}"

        if built_file.exists():
            shutil.copy2(built_file, target_file)
            if platform.system() != "Windows":
                os.chmod(target_file, 0o755)
            size_mb = target_file.stat().st_size / (1024 * 1024)
            print("\n" + "=" * 50)
            print("🎉 本地打包成功！")
            print(f"輸出目標：{target_file} ({size_mb:.2f} MB)")
            print("=" * 50 + "\n")

            # 同步檔案至 Windows PyCharm 測試目錄
            sync_target = Path("/mnt/my_book/NTHU_GDrive/MyProjects/PycharmProjects/musicmate")
            if sync_target.parent.exists():
                print(f"正在同步專案檔案至 Windows PyCharm 目錄：{sync_target} ...")
                rsync_cmd = [
                    "rsync", "-rtv", "--modify-window=2",
                    "--exclude=.git",
                    "--exclude=.venv",
                    "--exclude=venv",
                    "--exclude=__pycache__",
                    "--exclude=*.pyc",
                    "--exclude=build",
                    "--exclude=dist",
                    "--exclude=release",
                    f"{repo_root}/",
                    f"{sync_target}/"
                ]
                sync_res = subprocess.run(rsync_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if sync_res.returncode == 0:
                    print("✓ 成功同步最新專案檔案至 Windows PyCharm 目錄！")
                else:
                    print(f"⚠️ 同步時發生警示：{sync_res.stderr.strip()}")
            else:
                print(f"提示：未掛載或找不到目錄 {sync_target.parent}，跳過同步。")
        else:
            print(f"\n編譯完成，但未找到產出檔案：{built_file}")
    else:
        print("\n❌ 打包失敗，請檢視上方錯誤訊息。")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()

