"""背景视频来源模块。

支持四种来源（任选其一）：
  - local   : 本地素材库（models/backgrounds 目录下的视频），随机取一个
  - pexels  : Pexels 官方 API 实时搜索下载（需 PEXELS_API_KEY 环境变量）
  - pixabay : Pixabay 官方 API 实时搜索下载（需 PIXABAY_API_KEY 环境变量）
  - url     : 用户粘贴的直链，直接下载

统一：下载到缓存目录、挑选 720p/1080p 清晰度的 mp4、返回本地路径。
注意：背景视频的音轨不会被使用（合成阶段只取画面），天然满足"不要背景音乐"。
若背景视频比成片短则循环、比成片长则截取，由合成阶段的 _BgVideoReader 处理。
"""

import os
import random
from pathlib import Path

from .config import PEXELS_API_KEY, PIXABAY_API_KEY

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = BASE_DIR / "models" / "backgrounds"

VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv")

PEXELS_API = "https://api.pexels.com/videos/search"
PIXABAY_API = "https://pixabay.com/api/videos/"


def _download(url: str, dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=90, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    return dest


def _pick_resolution(files: list[dict]) -> dict | None:
    """从 Pexels/Pixabay 的 video_files 里挑一个 >=1280 宽里最小、否则最大。"""
    if not files:
        return None
    ok = [f for f in files if f.get("width", 0) >= 1280]
    pool = ok or files
    return max(pool, key=lambda f: f.get("width", 0))


def list_local(cache_dir: Path | None = None) -> list[Path]:
    cache_dir = cache_dir or DEFAULT_CACHE
    if not cache_dir.exists():
        return []
    return sorted(p for p in cache_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def get_local(query: str = "", index: int | None = None, cache_dir: Path | None = None) -> str:
    clips = list_local(cache_dir)
    if not clips:
        raise FileNotFoundError(
            f"本地素材库为空：{cache_dir}。请把风景/自然类视频放进去（或通过界面上传）。"
        )
    if index is not None and 0 <= index < len(clips):
        return str(clips[index])
    return str(random.choice(clips))


def get_pexels(query: str = "nature landscape", api_key: str | None = None,
               cache_dir: Path | None = None) -> str:
    api_key = api_key or os.environ.get("PEXELS_API_KEY") or PEXELS_API_KEY
    if not api_key:
        raise RuntimeError(
            "未设置 PEXELS_API_KEY。请到 https://www.pexels.com/api/ 免费申请，"
            "并设置环境变量 PEXELS_API_KEY 后重试。"
        )
    import requests

    cache_dir = cache_dir or DEFAULT_CACHE
    r = requests.get(PEXELS_API, params={"query": query, "per_page": 30},
                     headers={"Authorization": api_key}, timeout=30)
    r.raise_for_status()
    videos = r.json().get("videos", [])
    if not videos:
        raise RuntimeError(f"Pexels 未搜到与「{query}」相关的视频")
    vid = random.choice(videos)
    files = vid.get("video_files", [])
    chosen = _pick_resolution(files)
    if not chosen:
        raise RuntimeError("Pexels 返回的视频无可用文件")
    name = f"pexels_{vid.get('id', random.randint(0, 999999))}.mp4"
    return str(_download(chosen["link"], cache_dir / name))


def get_pixabay(query: str = "nature landscape", api_key: str | None = None,
                cache_dir: Path | None = None) -> str:
    api_key = api_key or os.environ.get("PIXABAY_API_KEY") or PIXABAY_API_KEY
    if not api_key:
        raise RuntimeError(
            "未设置 PIXABAY_API_KEY。请到 https://pixabay.com/api/docs/ 免费申请，"
            "并设置环境变量 PIXABAY_API_KEY 后重试。"
        )
    import requests

    cache_dir = cache_dir or DEFAULT_CACHE
    r = requests.get(PIXABAY_API, params={"key": api_key, "q": query, "per_page": 30},
                     timeout=30)
    r.raise_for_status()
    hits = r.json().get("hits", [])
    if not hits:
        raise RuntimeError(f"Pixabay 未搜到与「{query}」相关的视频")
    hit = random.choice(hits)
    files = hit.get("videos", {}).get("files", []) or []
    # Pixabay 字段结构略有不同：video_files 列表
    if not files and "video_files" in hit:
        files = hit["video_files"]
    chosen = _pick_resolution(files)
    if not chosen:
        raise RuntimeError("Pixabay 返回的视频无可用文件")
    name = f"pixabay_{hit.get('id', random.randint(0, 999999))}.mp4"
    return str(_download(chosen["link"], cache_dir / name))


def get_url(url: str, cache_dir: Path | None = None) -> str:
    cache_dir = cache_dir or DEFAULT_CACHE
    ext = ".mp4"
    for e in VIDEO_EXTS:
        if url.lower().endswith(e):
            ext = e
            break
    name = f"url_{abs(hash(url)) % 10**9}{ext}"
    return str(_download(url, cache_dir / name))


def resolve(source_type: str, query: str = "nature landscape",
            url: str | None = None, index: int | None = None,
            cache_dir: Path | None = None) -> str:
    """统一入口，返回本地视频路径。"""
    if source_type == "local":
        return get_local(query=query, index=index, cache_dir=cache_dir)
    if source_type == "pexels":
        return get_pexels(query=query, cache_dir=cache_dir)
    if source_type == "pixabay":
        return get_pixabay(query=query, cache_dir=cache_dir)
    if source_type == "url":
        if not url:
            raise ValueError("未提供直链 URL")
        return get_url(url, cache_dir=cache_dir)
    raise ValueError(f"未知背景来源: {source_type}")
