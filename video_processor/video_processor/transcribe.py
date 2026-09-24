"""语音转写字幕模块。

用 faster-whisper 加载本地 ctranslate2 模型（models/faster-whisper-<size>），
对视频音轨转写，输出带时间戳的句子与词级数据。

支持多种模型档位（base/small/medium/large-v3）：优先使用本地目录，
若本地目录不存在则首次自动从 HuggingFace 下载到 models/ 下。
CPU 上统一用 int8 量化以兼顾速度与内存。
"""

from pathlib import Path

# 模型根目录：video_processor/models/
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"

# 可选档位（从小到大）。base 最小最快，large-v3 最准但最慢（CPU 上尤甚）。
AVAILABLE_SIZES = ["base", "small", "medium", "large-v3"]
DEFAULT_SIZE = "small"

# 进程内按档位缓存，避免重复加载
_models: dict[str, object] = {}


def _local_dir(size: str) -> Path:
    return MODELS_ROOT / f"faster-whisper-{size}"


def get_model(size: str = DEFAULT_SIZE):
    """懒加载指定档位的 whisper 模型（CPU + int8）。

    优先用本地目录 models/faster-whisper-<size>，缺失时自动下载。
    """
    global _models
    if size not in _models:
        from faster_whisper import WhisperModel

        local = _local_dir(size)
        if local.exists():
            print(f"[transcribe] 加载本地模型: {local}")
            model = WhisperModel(str(local), device="cpu", compute_type="int8")
        else:
            # 首次使用会从 HuggingFace 下载，较慢，下载后落到本地目录
            print(f"[transcribe] 本地无 {size} 模型，从 HuggingFace 下载到 {local} ...")
            model = WhisperModel(
                size,
                device="cpu",
                compute_type="int8",
                download_root=str(local),
            )
        _models[size] = model
    return _models[size]


def transcribe(video_path: str, language: str = "zh", model_size: str = DEFAULT_SIZE) -> list[dict]:
    """转写视频音轨。

    参数:
        video_path: 视频文件路径
        language: 语种，默认中文 "zh"
        model_size: 模型档位，见 AVAILABLE_SIZES

    返回:
        [
            {
                "start": float, "end": float, "text": str,
                "words": [{"start": float, "end": float, "word": str}, ...]
            },
            ...
        ]
    """
    model = get_model(model_size)
    segments, info = model.transcribe(
        str(video_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,  # 过滤静音段，字幕更干净
    )
    result = []
    for seg in segments:
        words = [
            {"start": w.start, "end": w.end, "word": w.word.strip()}
            for w in (seg.words or [])
            if w.word and w.word.strip()
        ]
        result.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": words,
            }
        )
    print(
        f"[transcribe] 模型={model_size} 检测语言={info.language} "
        f"概率={info.language_probability:.2f} 共{len(result)}句"
    )
    return result
