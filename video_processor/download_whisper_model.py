"""下载 faster-whisper 的 ctranslate2 本地模型到 models/faster-whisper-<size>。

用法:
    python download_whisper_model.py small
    python download_whisper_model.py medium
    python download_whisper_model.py large-v3
"""
import os
import sys
from pathlib import Path

# 用 HF 镜像（国内可达），并清掉沙箱出口代理走直连（代理对镜像 API 返回 502）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for _p in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_p, None)

from huggingface_hub import snapshot_download

REPO_PREFIX = "Systran/faster-whisper-"
# 只拉取 ctranslate2 推理必需的几个文件，避免下载多余内容
ALLOW = [
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.txt",
    "preprocessor_config.json",
    "README.md",
    "tokenizer_config.json",
]

SIZE = sys.argv[1] if len(sys.argv) > 1 else "small"

target = Path(__file__).resolve().parent / "models" / f"faster-whisper-{SIZE}"
target.mkdir(parents=True, exist_ok=True)

print(f"[download] 目标目录: {target}")
print(f"[download] 从 HF 拉取 {REPO_PREFIX}{SIZE} ...")

# 优先默认端点；若被墙可设置 HF_ENDPOINT=https://hf-mirror.com
path = snapshot_download(
    repo_id=f"{REPO_PREFIX}{SIZE}",
    local_dir=str(target),
    allow_patterns=ALLOW,
)
print(f"[download] 完成: {path}")
