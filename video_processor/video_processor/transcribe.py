"""语音转写字幕模块。

用 faster-whisper 加载本地模型（models/faster-whisper-base），
对视频音轨转写，输出带时间戳的句子与词级数据。
"""

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "faster-whisper-base"

_model = None  # 进程内缓存，避免重复加载


def get_model():
    """懒加载 whisper 模型（CPU + int8，够用且省内存）。"""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"找不到本地模型目录: {MODEL_DIR}")
        _model = WhisperModel(str(MODEL_DIR), device="cpu", compute_type="int8")
    return _model


def transcribe(video_path: str, language: str = "zh") -> list[dict]:
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
    model = get_model()
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
    print(f"[transcribe] 检测语言={info.language} 概率={info.language_probability:.2f} 共{len(result)}句")
    return result
