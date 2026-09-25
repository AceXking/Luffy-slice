"""ffmpeg 封装层。

直接调用仓库根目录里的 ffmpeg 二进制，无需额外安装。
ffmpeg 目录: <仓库根>/ffmpeg-master-latest-win64-gpl-shared/bin
"""

import os
import subprocess
from pathlib import Path

# 自动定位工作区里的 ffmpeg 二进制
_FFMPEG_DIR = (
    Path(__file__).resolve().parents[2]
    / "ffmpeg-master-latest-win64-gpl-shared"
    / "bin"
)
FFMPEG_BIN = str(_FFMPEG_DIR / "ffmpeg.exe")
FFPROBE_BIN = str(_FFMPEG_DIR / "ffprobe.exe")


def _run(bin_path: str, args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """运行 ffmpeg / ffprobe 命令。"""
    cmd = [bin_path, *args]
    print("> " + " ".join(cmd))
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


def probe(path: str) -> dict:
    """用 ffprobe 读取视频元信息（JSON）。"""
    args = [
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = _run(FFPROBE_BIN, args)
    import json
    return json.loads(result.stdout)


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """直接透传任意 ffmpeg 参数，方便临时扩展。"""
    return _run(FFMPEG_BIN, args, **kwargs)


if __name__ == "__main__":
    print("ffmpeg:", FFMPEG_BIN, os.path.exists(FFMPEG_BIN))
    print("ffprobe:", FFPROBE_BIN, os.path.exists(FFPROBE_BIN))
