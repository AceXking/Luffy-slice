"""人物区域定位模块。

只负责"找到人物在哪"（返回包围盒），不做 alpha 抠像合成——
合成阶段直接把该区域裁剪成圆形"圈"起来。

用 mediapipe Tasks API（selfie_segmenter）逐帧生成人物遮罩，取包围盒。
模型: models/selfie_segmenter.tflite（float16, 官方下载）

性能：selfie_segmenter 内部本来就把输入缩到 256x256 推理，喂全分辨率帧只会
白白多花拷贝/缩放成本（4K 帧 ~168ms，缩到 640 长边后 ~10ms）。因此这里先把
帧缩到 SEG_LONG_SIDE 以内再做分割，再把包围盒坐标换算回原分辨率。
"""

from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "selfie_segmenter.tflite"

# 送进分割模型前的长边上限：模型内部只按 256x256 推理，640 足够定位且很快
SEG_LONG_SIDE = 640


class PersonSegmenter:
    """逐帧人物分割器（with 上下文使用）。"""

    def __init__(self, seg_long_side: int = SEG_LONG_SIDE):
        self.seg_long_side = seg_long_side

    def __enter__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"找不到分割模型: {MODEL_PATH}")
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=False,
            output_confidence_masks=True,
        )
        self._seg = vision.ImageSegmenter.create_from_options(options)
        self._mp = mp
        self._vision = vision
        return self

    def __exit__(self, *exc):
        if getattr(self, "_seg", None) is not None:
            self._seg.close()
            self._seg = None
        return False

    # ---------- 内部 ----------

    def _downscale(self, frame_bgr):
        """按长边上限缩放，返回 (small, sx, sy)，sx/sy 为 small->原图 的放大比例。"""
        h, w = frame_bgr.shape[:2]
        long_side = max(h, w)
        if self.seg_long_side and long_side > self.seg_long_side:
            s = self.seg_long_side / long_side
            nw, nh = max(1, round(w * s)), max(1, round(h * s))
            return cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA), w / nw, h / nh
        return frame_bgr, 1.0, 1.0

    def _mask_small(self, frame_bgr, frame_index: int, fps: float):
        """在降采样帧上跑分割，返回 (small_mask | None, sx, sy)。"""
        small, sx, sy = self._downscale(frame_bgr)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = int(frame_index * 1000 / fps)
        result = self._seg.segment_for_video(mp_img, ts)
        masks = result.confidence_masks
        if not masks:
            return None, sx, sy
        m = masks[1].numpy_view() if len(masks) > 1 else masks[0].numpy_view()
        m = np.squeeze(np.asarray(m, dtype=np.float32))
        # smoothstep 软边
        m = np.clip((m - 0.30) / (0.70 - 0.30), 0.0, 1.0)
        m = m * m * (3.0 - 2.0 * m)
        return m, sx, sy

    # ---------- 对外 ----------

    def mask(self, frame_bgr, frame_index: int, fps: float):
        """输入 BGR 帧，返回原分辨率 HxW float32 遮罩（0=背景 1=人物），失败返回 None。"""
        m, sx, sy = self._mask_small(frame_bgr, frame_index, fps)
        if m is None:
            return None
        h, w = frame_bgr.shape[:2]
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    def person_bbox(self, frame_bgr, frame_index: int, fps: float, pad_ratio: float = 0.12):
        """返回人物包围盒 (x0,y0,x1,y1) 原图像素坐标（含外扩 padding），找不到返回 None。"""
        m, sx, sy = self._mask_small(frame_bgr, frame_index, fps)
        if m is None:
            return None
        ys, xs = np.where(m > 0.35)
        if len(xs) < 15:  # 太碎，视为无人
            return None
        H, W = frame_bgr.shape[:2]
        # small -> 原图坐标
        x0 = int(xs.min() * sx)
        x1 = int(xs.max() * sx) + 2
        y0 = int(ys.min() * sy)
        y1 = int(ys.max() * sy) + 2
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, W), min(y1, H)
        pad = int(pad_ratio * max(x1 - x0, y1 - y0))
        x0 = max(x0 - pad, 0)
        y0 = max(y0 - pad, 0)
        x1 = min(x1 + pad, W)
        y1 = min(y1 + pad, H)
        return (x0, y0, x1, y1)

    def cutout(self, frame_bgr, frame_index: int, fps: float):
        """（保留兼容）带 alpha 的人物图。新流程已不使用。"""
        m = self.mask(frame_bgr, frame_index, fps)
        if m is None:
            return None
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = (m * 255).astype("uint8")
        return rgba
