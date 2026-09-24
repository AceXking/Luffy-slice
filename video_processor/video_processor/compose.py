"""合成管线 v3：背景 + 人物裁剪圈框(左上) + 居中稳定字幕 + 可选 BGM -> 9:16。

相对 v2 的改动：
  - 人物：不再 alpha 抠像，改为 mediapipe 仅定位包围盒 -> 从原帧裁剪该区域 ->
    正圆遮罩(蒙版) + 圆形描边环贴左上角，带投影/入场淡入。
    圆形直径与位置固定（不随逐帧包围盒变化、取消呼吸缩放），彻底杜绝人物框抖动。
  - 字幕：见 subtitles.py（稳定居中、避开人物框、不闪不漏字）。
  - 背景：纯色 / 本地素材库 / Pexels / Pixabay / 直链；视频背景循环或截取，叠加蒙版突出字幕。
  - BGM：可选背景音乐，音量小，按成片时长循环或截取，与人物原声混音。

性能（v3.1）：
  - 人物分割在降采样帧上做（segmentation.py），4K 源单帧 168ms -> ~10ms。
  - 背景影调预计算成 1 次通道混合 + 1 次逐像素乘加（_make_bg_grade），
    全画布 float 运算 213ms -> 数十 ms。
  - 人物投影遮罩只算一次（圆形恒定），不再逐帧高斯模糊。
  - 字幕改为只叠加文字小贴图（subtitles.render_tiles），去掉全画布 RGBA
    转换与合成，326ms -> 数 ms。
  - 编码 preset veryfast。
"""

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .ffmpeg_utils import FFMPEG_BIN, FFPROBE_BIN
from .segmentation import PersonSegmenter
from .subtitles import SubtitleRenderer
from .transcribe import transcribe

OUT_W, OUT_H = 1080, 1920  # 9:16

# 视觉风格预设
STYLES = {
    "luxe":      dict(accent=(255, 209, 102), vignette=0.42, scrim=0.30, desat=0.15,
                      scrim_spread=0.20, sub_dark=0.45),
    "fresh":     dict(accent=(255, 138, 128), vignette=0.28, scrim=0.22, desat=0.05,
                      scrim_spread=0.22, sub_dark=0.35),
    "cinematic": dict(accent=(255, 180, 90),  vignette=0.58, scrim=0.40, desat=0.32,
                      scrim_spread=0.16, sub_dark=0.55),
}


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _crop_to_cover(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    scale = max(w / fw, h / fh)
    nw, nh = round(fw * scale), round(fh * scale)
    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return frame[y0 : y0 + h, x0 : x0 + w]


class _BgVideoReader:
    """背景视频读取器：读完自动循环（短则循环，长则自然截取）。"""

    def __init__(self, path: str, w: int, h: int):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"背景视频打不开: {path}")
        self.w, self.h = w, h

    def next(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                frame = np.zeros((self.h, self.w, 3), np.uint8)
        return _crop_to_cover(frame, self.w, self.h)

    def release(self):
        self.cap.release()


def _has_audio(path: str) -> bool:
    r = subprocess.run(
        [FFPROBE_BIN, "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    try:
        streams = __import__("json").loads(r.stdout).get("streams", [])
    except Exception:
        return False
    return any(s.get("codec_type") == "audio" for s in streams)


def _probe_duration(path: str) -> float:
    r = subprocess.run(
        [FFPROBE_BIN, "-v", "quiet", "-print_format", "json",
         "-show_entries", "format=duration", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(__import__("json").loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _make_grade_maps(h: int, w: int, vignette: float, scrim: float,
                     scrim_center: float, scrim_spread: float, sub_dark: float):
    """预计算背景影调图: vig(暗角) + add(字幕区加法遮罩) + sub(字幕区乘法压暗)。"""
    yy = np.linspace(0, 1, h)[:, None]
    xx = np.linspace(0, 1, w)[None, :]
    r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.7071
    vig = 1.0 - vignette * np.clip(r, 0, 1) ** 2
    band = np.exp(-((yy - scrim_center) ** 2) / (2 * scrim_spread ** 2))
    add = -scrim * band
    # 字幕中心区域（画面中下部）额外压暗，突出字幕
    sub_blob = np.exp(-((yy - 0.62) ** 2 / (2 * 0.12 ** 2)
                         + (xx - 0.5) ** 2 / (2 * 0.26 ** 2)))
    sub = 1.0 - sub_dark * sub_blob
    return (vig.astype(np.float32), add.astype(np.float32), sub.astype(np.float32))


def _make_bg_grade(desat: float, bg_dim: float, vig: np.ndarray,
                   add: np.ndarray, sub: np.ndarray):
    """预计算视频背景影调的可复用算子，把逐帧 6 次全画布 float 运算压成
    "1 次通道混合 + 1 次乘加"。

    原式: c1=a*c+b*luma -> *=(1-dim) -> *=vig; +=add*255 -> *=sub
    合并: out = clip( M3 * (K @ c) + A3 )
      K  : 去饱和(灰度混合)的 3x3 通道混合矩阵（同一矩阵对所有像素）
      M3 : 逐像素乘子 (1-bg_dim)*vig*sub，展开到 3 通道
      A3 : 逐像素加项 add*255*sub，展开到 3 通道
    """
    a, b = 1.0 - desat, desat
    wb, wg, wr = 0.114, 0.587, 0.299  # BGR 顺序下的 luma 权重
    K = np.array(
        [
            [a + b * wb, b * wg, b * wr],
            [b * wb, a + b * wg, b * wr],
            [b * wb, b * wg, a + b * wr],
        ],
        np.float32,
    )
    m = ((1.0 - bg_dim) * vig * sub).astype(np.float32)
    t = (add * 255.0 * sub).astype(np.float32)
    M3 = np.repeat(m[:, :, None], 3, axis=2)
    A3 = np.repeat(t[:, :, None], 3, axis=2)
    return K, M3, A3


def _composite(canvas: np.ndarray, img_rgba: np.ndarray, px: int, py: int, alpha_mul: float = 1.0):
    ph, pw = img_rgba.shape[:2]
    px = min(max(px, 0), canvas.shape[1] - pw)
    py = min(max(py, 0), canvas.shape[0] - ph)
    a = img_rgba[:, :, 3:4].astype(np.float32) / 255.0 * alpha_mul
    roi = canvas[py : py + ph, px : px + pw].astype(np.float32)
    canvas[py : py + ph, px : px + pw] = (
        img_rgba[:, :, :3].astype(np.float32) * a + roi * (1 - a)
    ).astype(np.uint8)


def _shadow_region(canvas: np.ndarray, alpha: np.ndarray, px: int, py: int):
    """把纯黑投影叠到画布：out = dst * (1 - a)，只需一次乘暗（比 RGBA 合成省一半）。"""
    h, w = alpha.shape[:2]
    x0, y0 = max(px, 0), max(py, 0)
    x1, y1 = min(px + w, canvas.shape[1]), min(py + h, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    a = alpha[y0 - py : y1 - py, x0 - px : x1 - px].astype(np.float32) * (1.0 / 255.0)
    roi = canvas[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1.0 - a[:, :, None])).astype(np.uint8)


def _circle_mask(h: int, w: int, radius: int) -> np.ndarray:
    """正圆遮罩：圆心取像素精确中心 ((h-1)/2,(w-1)/2)，圆外为 0。

    用浮点圆心 + 0.5px 余量，保证左右/上下完全对称，
    避免整数圆心在圆边缘出现 1px 平台（白环局部突出）。
    """
    r = max(1.0, min(float(radius), min(h, w) / 2 - 0.5))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    Y, X = np.ogrid[:h, :w]
    mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2
    return mask.astype(np.uint8) * 255


def _person_crop_rgba(frame: np.ndarray, box, target_d: int,
                      border_bgr, border_w: int) -> np.ndarray | None:
    """从原帧裁剪人物区域 -> 正圆遮罩(蒙版) + 圆形描边环 -> RGBA。

    统一输出 target_d×target_d 固定方形画布：圆形直径、圆心、贴图位置全部恒定，
    与逐帧包围盒无关，彻底消除人物框抖动。bbox 只决定"取人物哪块"。

    取景用「正方形铺满」而非「等比塞入」：先在原帧 bbox 内取一个边长 = 短边的
    正方形区域，再整体缩放到 target_d。这样画布被真实画面完全填满，圆内不再
    出现等比塞入时上下/左右的纯黑边（横屏源尤其明显）。
    正方形在长边方向的对齐：横构图水平居中；竖构图纵向偏上，避免裁掉头部。
    """
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    side = max(1, min(h, w))
    ox = (w - side) // 2                                     # 横构图：水平居中
    oy = int(round(0.15 * (h - side))) if h > side else 0    # 竖构图：偏上，保住头部
    square = crop[oy:oy + side, ox:ox + side]
    interp = cv2.INTER_AREA if side > target_d else cv2.INTER_LINEAR
    rgba = cv2.cvtColor(
        cv2.resize(square, (target_d, target_d), interpolation=interp),
        cv2.COLOR_BGR2BGRA,
    )
    mask = _circle_mask(target_d, target_d, target_d // 2)
    # 圆形描边环（沿圆边）
    if border_w > 0:
        ek = max(border_w * 2 + 1, 3)
        eroded = cv2.erode(mask, np.ones((ek, ek), np.uint8))
        ring = mask - eroded
        rgba[ring > 0] = (*border_bgr, 255)
    rgba[:, :, 3] = mask
    return rgba


def _finalize_audio(silent_path: str, voice_src: str, bgm_path: str | None,
                    bgm_vol: float, voice_vol: float, out_path: str) -> str:
    """音轨合成：可选 BGM（小声、按成片时长循环/截取）+ 人物原声，混音到成片。"""
    has_voice = _has_audio(voice_src)
    silent = Path(silent_path)
    out = Path(out_path)

    if not bgm_path and not has_voice:
        silent.replace(out)
        return str(out)

    dur = _probe_duration(silent_path) or 0.0
    cmd = [FFMPEG_BIN, "-y"]

    if bgm_path and Path(bgm_path).exists():
        if has_voice:
            cmd += ["-i", silent_path, "-i", voice_src, "-i", bgm_path]
            fc = [
                f"[1:a]volume={voice_vol:.3f}[vc]",
                f"[2:a]volume={bgm_vol:.3f},aloop=loop=-1:size=2e9,atrim=0:{dur:.3f}[bg]",
                "[vc][bg]amix=inputs=2:duration=first:normalize=0[mix]",
            ]
            amap = "[mix]"
        else:
            cmd += ["-i", silent_path, "-i", bgm_path]
            fc = [f"[1:a]volume={bgm_vol:.3f},aloop=loop=-1:size=2e9,atrim=0:{dur:.3f}[bg]"]
            amap = "[bg]"
        cmd += ["-filter_complex", ";".join(fc), "-map", "0:v", "-map", amap,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)]
    else:  # 仅人物原声
        cmd += ["-i", silent_path, "-i", voice_src, "-map", "0:v", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)]

    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        # 失败则退回静音视频
        silent.replace(out)
        print("[compose] 音轨合成失败，退回静音成片")
    else:
        silent.unlink(missing_ok=True)
    return str(out)


def generate(
    input_video: str,
    output_path: str,
    background: dict,
    language: str = "zh",
    person_height_ratio: float = 0.30,
    person_pos: tuple[float, float] = (0.05, 0.05),
    border: bool = True,
    border_hex: str = "#FFFFFF",
    style: str = "luxe",
    accent_hex: str | None = None,
    bg_dim: float = 0.15,
    sub_dark: float | None = None,
    person_shadow: bool = True,
    subtitle_overrides: dict | None = None,
    bgm_path: str | None = None,
    bgm_volume: float = 0.2,
    voice_volume: float = 1.0,
    model_size: str = "small",
    progress_cb=None,
) -> str:
    sp = {**STYLES.get(style, STYLES["luxe"])}
    accent_rgb = _hex_to_rgb(accent_hex) if accent_hex else sp["accent"]
    border_bgr = _hex_to_bgr(border_hex)
    if sub_dark is None:
        sub_dark = sp["sub_dark"]

    src = cv2.VideoCapture(input_video)
    if not src.isOpened():
        raise FileNotFoundError(f"输入视频打不开: {input_video}")
    fps = src.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(src.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if background["type"] == "color":
        bg_color = np.array(_hex_to_bgr(background["color"]), np.uint8)
        bg_reader = None
    else:
        bg_reader = _BgVideoReader(background["path"], OUT_W, OUT_H)

    print("[compose] 转写中...")
    segments = transcribe(input_video, language=language, model_size=model_size)
    sub_over = dict(subtitle_overrides or {})
    sub_over.setdefault("accent", (*accent_rgb, 255))
    renderer = SubtitleRenderer(segments, OUT_W, OUT_H, **sub_over)

    vig, add, sub = _make_grade_maps(
        OUT_H, OUT_W, sp["vignette"], sp["scrim"], 0.56, sp["scrim_spread"], sub_dark
    )
    graded_color = None
    if bg_reader is None:
        graded_color = np.clip(
            np.full((OUT_H, OUT_W, 3), bg_color, np.float32) * vig[:, :, None]
            + add[:, :, None] * 255.0,
            0, 255,
        ).astype(np.uint8)

    # 视频背景影调算子（预计算：逐帧只需 1 次通道混合 + 1 次乘加）
    grade_K = grade_M3 = grade_A3 = None
    if bg_reader is not None:
        grade_K, grade_M3, grade_A3 = _make_bg_grade(sp["desat"], bg_dim, vig, add, sub)

    # 人物固定直径 + 投影遮罩（圆形恒定，模糊一次复用，避免逐帧高斯模糊）
    target_d = int(person_height_ratio * OUT_H)
    shadow_mask = None
    if person_shadow:
        d = target_d
        circ = _circle_mask(d, d, d // 2)
        ds = max(1, d // 4)
        sm = cv2.resize(circ, (ds, ds), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), sigmaX=1.0)
        sm = cv2.resize(sm, (d, d), interpolation=cv2.INTER_LINEAR)
        shadow_mask = sm.astype(np.float32) * 0.5

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    silent_path = out.parent / (out.stem + "_silent.mp4")
    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{OUT_W}x{OUT_H}", "-r", f"{fps:.6f}",
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-crf", "20", str(silent_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    margin_x = int(person_pos[0] * OUT_W)
    margin_y = int(person_pos[1] * OUT_H)
    placed_box = None
    t_render0 = time.time()

    with PersonSegmenter() as segmenter:
        i = 0
        while True:
            ok, frame = src.read()
            if not ok:
                break
            t = i / fps

            # 1) 背景（预计算算子：通道混合 + 逐像素乘加）
            if bg_reader is not None:
                cf = bg_reader.next().astype(np.float32)
                if grade_K is not None:
                    cf = cv2.transform(cf, grade_K)
                cf = cv2.multiply(cf, grade_M3)
                cv2.add(cf, grade_A3, dst=cf)
                np.clip(cf, 0, 255, out=cf)
                canvas = cf.astype(np.uint8)
            else:
                canvas = graded_color.copy()

            # 2) 人物裁剪 -> 圆形遮罩，固定直径与位置（消除逐帧抖动）
            box = segmenter.person_bbox(frame, i, fps)
            if box is not None:
                rgba = _person_crop_rgba(
                    frame, box, target_d,
                    border_bgr if border else (0, 0, 0), border_w=8 if border else 0,
                )
                if rgba is not None:
                    pw, ph = rgba.shape[1], rgba.shape[0]
                    px = min(margin_x, OUT_W - pw)
                    py = min(margin_y, OUT_H - ph)

                    ea = _smoothstep(min(t / 0.8, 1.0))  # 入场上浮淡入
                    if shadow_mask is not None and ea > 0.01:
                        sh_alpha = np.clip(shadow_mask * ea, 0, 255).astype(np.uint8)
                        _shadow_region(canvas, sh_alpha, px + 4, py + 10)
                    _composite(canvas, rgba, px, py, ea)
                    placed_box = (px, py, px + pw, py + ph)

            # 3) 字幕（避开人物框）：只叠加小贴图，不再做全画布 RGBA 转换与合成
            for tile, tx, ty in renderer.render_tiles(t, placed_box):
                _composite(canvas, tile, tx, ty)

            proc.stdin.write(canvas.astype(np.uint8).tobytes())
            i += 1
            if progress_cb and total:
                progress_cb(min(i / total, 1.0))

    src.release()
    if bg_reader is not None:
        bg_reader.release()
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 编码失败, exit={proc.returncode}")

    # 4) 音轨合成（BGM / 原声）
    final = _finalize_audio(str(silent_path), input_video, bgm_path,
                            bgm_volume, voice_volume, str(out))

    dt = time.time() - t_render0
    fps_out = i / dt if dt > 0 else 0.0
    print(f"[compose] 完成: {final} 共 {i} 帧，合成耗时 {dt:.1f}s（{fps_out:.1f} fps）")
    return final


def _smoothstep(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3.0 - 2.0 * p)
