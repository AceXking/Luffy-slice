"""人物区域定位模块。

只负责"找到人物在哪"（返回包围盒），不再做 alpha 抠像合成——
合成阶段直接把该区域裁剪成圆角矩形"圈"起来即可，更快也更干净。

用 mediapipe Tasks API（selfie_segmenter）逐帧生成人物遮罩，取包围盒。
模型: models/selfie_segmenter.tflite（float16, 官方下载）
"""

from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "selfie_segmenter.tflite"


class PersonSegmenter:
    """逐帧人物分割器（with 上下文使用）。"""

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

    def mask(self, frame_bgr, frame_index: int, fps: float):
        """输入 BGR 帧，返回 HxW float32 遮罩（0=背景 1=人物），失败返回 None。"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = int(frame_index * 1000 / fps)
        result = self._seg.segment_for_video(mp_img, ts)
        masks = result.confidence_masks
        if not masks:
            return None
        m = masks[1].numpy_view() if len(masks) > 1 else masks[0].numpy_view()
        m = np.squeeze(np.asarray(m, dtype=np.float32))  # 可能是 (H,W,1)，压成 2D
        # smoothstep 软边
        m = np.clip((m - 0.30) / (0.70 - 0.30), 0.0, 1.0)
        m = m * m * (3.0 - 2.0 * m)
        return m

    def person_bbox(self, frame_bgr, frame_index: int, fps: float, pad_ratio: float = 0.12):
        """返回人物包围盒 (x0,y0,x1,y1) 像素坐标（含外扩 padding），找不到返回 None。"""
        m = self.mask(frame_bgr, frame_index, fps)
        if m is None:
            return None
        ys, xs = np.where(m > 0.35)
        if len(xs) < 20:  # 太碎，视为无人
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        pad = int(pad_ratio * max(x1 - x0, y1 - y0))
        x0 = max(x0 - pad, 0)
        y0 = max(y0 - pad, 0)
        x1 = min(x1 + pad, frame_bgr.shape[1])
        y1 = min(y1 + pad, frame_bgr.shape[0])
        return (x0, y0, x1, y1)

    def cutout(self, frame_bgr, frame_index: int, fps: float):
        """（保留兼容）带 alpha 的人物图。新流程已不使用。"""
        m = self.mask(frame_bgr, frame_index, fps)
        if m is None:
            return None
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = (m * 255).astype("uint8")
        return rgba
