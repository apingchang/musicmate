#!/usr/bin/env python3
"""自動化發布腳本 (Automated Release Script)

自動判斷或指定版號、提交未暫存改動、打 Tag 並推送至 GitHub 觸發雙平台打包。
使用方式：
    python scripts/release.py            # 自動將 patch 版號 +1 (例如 v1.1.0 -> v1.1.1)
    python scripts/release.py 1.2.0      # 指定版本號為 v1.2.0
    python scripts/release.py --minor    # minor 版號 +1 (例如 v1.1.0 -> v1.2.0)
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, check=True):
    print(f"執行：{' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=isinstance(cmd, str))
    if check and res.returncode != 0:
        print(f"錯誤：{res.stderr.strip()}", file=sys.stderr)
        sys.exit(res.returncode)
    return res.stdout.strip()


def get_latest_tag():
    tags = run_cmd(["git", "tag", "-l", "v*"], check=False).split()
    valid_tags = []
    for t in tags:
        m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)$", t)
        if m:
            valid_tags.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), t))
    if not valid_tags:
        return "v1.1.0"
    valid_tags.sort()
    return valid_tags[-1][3]


def bump_version(current_tag, bump_type="patch"):
    m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)$", current_tag)
    if not m:
        return "v1.1.1"
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if bump_type == "major":
        return f"v{major + 1}.0.0"
    elif bump_type == "minor":
        return f"v{major}.{minor + 1}.0"
    else:
        return f"v{major}.{minor}.{patch + 1}"


def main():
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    # 1. 決定新版號
    target_tag = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--minor":
            latest = get_latest_tag()
            target_tag = bump_version(latest, "minor")
        elif arg == "--major":
            latest = get_latest_tag()
            target_tag = bump_version(latest, "major")
        else:
            clean = arg.lstrip("v")
            target_tag = f"v{clean}"
    else:
        latest = get_latest_tag()
        target_tag = bump_version(latest, "patch")

    print(f"\n🚀 即將發布版本：{target_tag}\n")

    # 2. 檢查工作區變更並自動 commit
    status = run_cmd(["git", "status", "--porcelain"])
    if status:
        print("偵測到尚未提交的改動，正在自動加入並提交...")
        run_cmd(["git", "add", "."])
        run_cmd(["git", "commit", "-m", f"chore(release): 發布 {target_tag}"])
    else:
        print("工作區乾淨，無需額外 commit。")

    # 3. 推送 main 分支
    print("\n推送到 origin main...")
    run_cmd(["git", "push", "origin", "main"])

    # 4. 建立並推送 tag
    print(f"\n建立標籤 {target_tag} 並推送...")
    run_cmd(["git", "tag", "-a", target_tag, "-m", f"Release {target_tag}"])
    run_cmd(["git", "push", "origin", target_tag])

    print("\n" + "=" * 50)
    print(f"🎉 版本 {target_tag} 已成功推送到 GitHub！")
    print(f"雲端建置已自動觸發：")
    print(f"👉 Actions 進度：https://github.com/apingchang/musicmate/actions")
    print(f"👉 Releases 下載：https://github.com/apingchang/musicmate/releases")
    print("=" * 50)

    # 5. 若有 --sync 參數或預設直接同步
    if "--no-sync" not in sys.argv:
        print("\n⏳ 正在等待 GitHub Actions 雲端雙平台建置完成...")
        import urllib.request, json, time
        api_runs = "https://api.github.com/repos/apingchang/musicmate/actions/runs"
        success = False
        for _ in range(60):  # 最多等待 10 分鐘
            time.sleep(10)
            try:
                req = urllib.request.Request(api_runs, headers={"User-Agent": "MusicMate-Release"})
                with urllib.request.urlopen(req) as resp:
                    runs = json.loads(resp.read().decode()).get("workflow_runs", [])
                    matched = [r for r in runs if r.get("head_branch") == target_tag]
                    if matched and matched[0].get("status") == "completed":
                        if matched[0].get("conclusion") == "success":
                            print(f"\n✓ 雲端建置成功！即將自動同步至本地 release/ 資料夾...")
                            success = True
                        else:
                            print(f"\n⚠️ 雲端建置狀態：{matched[0].get('conclusion')}，請檢查 GitHub Actions。")
                        break
            except Exception:
                pass
            print(".", end="", flush=True)

        if success:
            sync_script = repo_root / "scripts" / "sync_release.py"
            subprocess.run([sys.executable, str(sync_script)])

    # 同步專案原始碼至 Windows PyCharm 測試目錄
    sync_target = Path("/mnt/my_book/NTHU_GDrive/MyProjects/PycharmProjects/musicmate")
    if sync_target.parent.exists():
        print(f"\n正在同步專案檔案至 Windows PyCharm 目錄：{sync_target} ...")
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


if __name__ == "__main__":
    main()


