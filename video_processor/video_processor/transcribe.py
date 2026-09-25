"""语音转写字幕模块。

用 faster-whisper 加载本地模型（models/faster-whisper-<size>），
对视频音轨转写，输出带时间戳的句子与词级数据。

模型越大越准、CPU 上越慢，默认 medium（中文口播准确率明显优于 base）。
缺失的模型可用仓库根目录的 _download_models.py 下载。
"""

import os
from pathlib import Path

MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"

# 可选的本地 whisper 模型（本地目录名 = faster-whisper-<size>）
WHISPER_SIZES = ("base", "small", "medium", "large-v3")
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "medium")

_model_cache: dict[str, object] = {}  # 进程内缓存，避免重复加载


def model_dir(name: str) -> Path:
    """返回指定尺寸模型对应的本地目录。"""
    if name not in WHISPER_SIZES:
        raise ValueError(f"未知模型尺寸: {name}（可选: {', '.join(WHISPER_SIZES)}）")
    return MODELS_ROOT / f"faster-whisper-{name}"


def get_model(name: str = DEFAULT_MODEL):
    """懒加载 whisper 模型（CPU + int8，够用且省内存）。"""
    global _model_cache
    if name not in _model_cache:
        from faster_whisper import WhisperModel

        d = model_dir(name)
        if not d.exists():
            raise FileNotFoundError(
                f"找不到本地模型目录: {d}\n"
                f"请运行仓库根目录的 _download_models.py 下载（或选小一号的模型）。"
            )
        _model_cache[name] = WhisperModel(str(d), device="cpu", compute_type="int8")
    return _model_cache[name]


def transcribe(video_path: str, language: str = "zh", model_name: str = DEFAULT_MODEL) -> list[dict]:
    """转写视频音轨。

    返回:
        [
            {
                "start": float, "end": float, "text": str,
                "words": [{"start": float, "end": float, "word": str}, ...]
            },
            ...
        ]
    """
    model = get_model(model_name)
    segments, info = model.transcribe(
        str(video_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,  # 过滤静音段，字幕更干净
        beam_size=5,  # 束搜索，比默认贪心准确率明显更高
        condition_on_previous_text=True,  # 利用上文保持用词/标点一致
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
        f"[transcribe] 模型={model_name} 检测语言={info.language} "
        f"概率={info.language_probability:.2f} 共{len(result)}句"
    )
    return result
