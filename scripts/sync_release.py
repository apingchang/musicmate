#!/usr/bin/env python3
"""從 GitHub Releases 同步最新 Windows 與 Ubuntu 執行檔到本地 release/ 資料夾"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    release_dir = repo_root / "release"
    release_dir.mkdir(exist_ok=True)

    print("正在查詢 GitHub 最新發布版本...")
    api_url = "https://api.github.com/repos/apingchang/musicmate/releases/latest"
    req = urllib.request.Request(api_url, headers={"User-Agent": "MusicMate-Sync"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"無法取得 GitHub Releases：{e}")
        sys.exit(1)

    tag_name = data.get("tag_name")
    print(f"最新版本：{tag_name}")

    assets = data.get("assets", [])
    win_asset = next((a for a in assets if "windows" in a["name"].lower()), None)
    ubuntu_asset = next((a for a in assets if "ubuntu" in a["name"].lower()), None)

    if win_asset:
        win_zip = release_dir / win_asset["name"]
        print(f"正在下載 Windows 執行檔 ({win_asset['size'] // (1024*1024)} MB)...")
        subprocess.run(["curl", "-C", "-", "-L", "-o", str(win_zip), win_asset["browser_download_url"]], check=True)
        print("正在解壓縮 Windows 執行檔...")
        import zipfile
        with zipfile.ZipFile(win_zip, 'r') as z:
            z.extractall(release_dir)
        win_zip.unlink(missing_ok=True)
        print("✓ Windows 執行檔就緒：release/MusicMate.exe")

    if ubuntu_asset:
        ubuntu_tar = release_dir / ubuntu_asset["name"]
        print(f"正在下載 Ubuntu 執行檔 ({ubuntu_asset['size'] // (1024*1024)} MB)...")
        subprocess.run(["curl", "-C", "-", "-L", "-o", str(ubuntu_tar), ubuntu_asset["browser_download_url"]], check=True)
        print("正在解壓縮 Ubuntu 執行檔...")
        import tarfile
        with tarfile.open(ubuntu_tar, 'r:gz') as t:
            t.extractall(release_dir)
        ubuntu_tar.unlink(missing_ok=True)
        ubuntu_bin = release_dir / "MusicMate"
        if ubuntu_bin.exists():
            os.chmod(ubuntu_bin, 0o755)
        print("✓ Ubuntu 執行檔就緒：release/MusicMate")

    print(f"\n🎉 雙平台最新執行檔已成功同步至：{release_dir}")


if __name__ == "__main__":
    main()
