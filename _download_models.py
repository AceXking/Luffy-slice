"""一次性脚本：下载运行时模型到 video_processor/models/。

- faster-whisper 转写模型（HuggingFace Systran/faster-whisper-<size>，走 hf-mirror 镜像）
- selfie_segmenter.tflite（Google Mediapipe 官方存储）
用完即可删除。支持断点续传（.part 临时文件）。

需要下载 large-v3 时，把下面 WHISPER_SIZES 改成 ("base", "small", "medium", "large-v3")。
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "video_processor" / "models"
WHISPER_SIZES = ("base", "small", "medium")  # 本地目录名 faster-whisper-<size>

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
SEG_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)

UA = {"User-Agent": "Mozilla/5.0"}


def download(url: str, dst: Path, retries: int = 3) -> None:
    """下载 url 到 dst，支持 .part 断点续传。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            done = part.stat().st_size if part.exists() else 0
            headers = dict(UA)
            if done:
                headers["Range"] = f"bytes={done}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if done and r.status == 206 else "wb"
                if mode == "wb":
                    done = 0
                total = int(r.headers.get("Content-Length", 0)) + done
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done * 100 / total
                            print(f"\r  {dst.name}: {done/1048576:.1f}/{total/1048576:.1f} MB ({pct:.0f}%)",
                                  end="", flush=True)
            print()
            part.replace(dst)
            return
        except Exception as e:  # noqa: BLE001
            print(f"\n  [重试 {attempt}/{retries}] {dst.name}: {type(e).__name__}: {e}")
    raise SystemExit(f"下载失败: {url}")


def list_repo_files(repo: str) -> list[str]:
    url = f"{HF_ENDPOINT}/api/models/{repo}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        data = json.load(r)
    return [s["rfilename"] for s in data.get("siblings", [])]


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)

    for i, size in enumerate(WHISPER_SIZES, 1):
        repo = f"Systran/faster-whisper-{size}"
        whisper_dir = MODELS / f"faster-whisper-{size}"
        print(f"[{i}/2] {repo}")
        files = list_repo_files(repo)
        keep = [f for f in files if not f.startswith(".") and "/" not in f]
        print(f"  仓库文件: {keep}")
        for name in keep:
            dst = whisper_dir / name
            if dst.exists() and dst.stat().st_size > 0:
                print(f"  跳过（已存在）: {name}")
                continue
            download(f"{HF_ENDPOINT}/{repo}/resolve/main/{name}", dst)

    print("[2/2] selfie_segmenter.tflite")
    dst = MODELS / "selfie_segmenter.tflite"
    if dst.exists() and dst.stat().st_size > 0:
        print("  跳过（已存在）")
    else:
        download(SEG_URL, dst)

    print("\n完成，模型清单:")
    for p in sorted(MODELS.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(MODELS)}  {p.stat().st_size/1048576:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
