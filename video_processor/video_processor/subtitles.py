"""现代精致「居中稳定」字幕渲染模块（v3）。

改动要点（相对 v2）：
- 不再持续向上滚动，改为"当前句居中、上一句在上淡出、下一句在下预览"，
  画面稳定，不再闪。
- 入场极快淡入（0.16s），彻底解决"读到了但字幕没出来"。
- 卡拉OK逐字点亮：已读=纯白 / 正在读=强调色 / 未读=柔和灰。
- 字幕占用"除人物框外"的全部空间：render_tiles(t, avoid_box) 动态计算锚点
  （人物在左上，则字幕居中偏下/偏右，避开左上）。
- 柔和投影保证在任意背景清晰。

性能（v3.1）：
- 不再每帧新建整张画布 RGBA 图层，改为只产出「文字小贴图 + 位置」，
  由合成层在局部区域叠加，省掉全画布 RGBA->BGRA 转换与全画布 float 合成
  （1080x1920 下单帧从 ~326ms 降到几 ms）。
- 文字贴图 / 投影贴图按内容缓存，仅在卡拉OK状态变化时重绘。

用法:
    r = SubtitleRenderer(segments, canvas_w, canvas_h, **overrides)
    tiles = r.render_tiles(t, avoid_box)   # [(bgra_uint8, x, y), ...]；无可见字幕则 []
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Windows 常见中文字体候选
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

DEFAULTS = {
    "font_size": 96,
    "max_chars_per_line": 8,
    "line_gap": 26,
    "fill": (255, 255, 255, 255),          # 已读出：纯白
    "upcoming": (168, 178, 196, 255),      # 未读出：柔和冷灰（仍清晰可读）
    "accent": (255, 209, 102, 255),        # 正在读：强调色
    "shadow": True,
    "shadow_blur": 10,
    "shadow_offset": 6,
    "shadow_alpha": 150,
    "enter_dur": 0.16,    # 入场淡入时长（极短，避免"读到了字幕没出来"）
    "linger": 1.2,        # 句说完后停留
    "fade_out": 0.6,      # 句末淡出
    "prev_alpha": 0.42,   # 上一句淡出后的残留透明度
    # 上一行淡出后的下限：多行显示时保证"更上一行"不会淡到消失，稳定铺满 5 行
    "prev_min_alpha": 0.20,
    "next_alpha": 0.38,   # 下一句预览透明度
    # 多行显示：当前句外，上下各显示 context_lines 行，合计 1 + 2*context_lines 行
    "context_lines": 2,
    "prev_alpha_fade": 0.55,  # 往上第 2 行及更远，在 prev_alpha 基础上继续衰减
    "next_alpha_fade": 0.62,  # 往下第 2 行及更远，在 next_alpha 基础上继续衰减
    "anchor_up": 0.05,    # 字幕整体往上偏移（占画面高度比例；0.05 约 96px，刚好避开人物框）
}

_TILE_CACHE_MAX = 1024


def _load_font(size: int):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _clamp(v, lo=0.0, hi=1.0):
    return min(max(v, lo), hi)


def _smoothstep(p):
    p = _clamp(p)
    return p * p * (3.0 - 2.0 * p)


class SubtitleRenderer:
    def __init__(self, segments: list[dict], canvas_w: int, canvas_h: int, **overrides):
        self.cfg = {**DEFAULTS, **overrides}
        self.W = canvas_w
        self.H = canvas_h
        self.font = _load_font(self.cfg["font_size"])
        self.lines = self._build_lines(segments)
        # 缓存：文字贴图按 (文本, 每字颜色) 缓存；投影按文本缓存
        self._tile_cache: dict = {}
        self._shadow_cache: dict = {}

    # ---------- 数据准备 ----------

    def _build_lines(self, segments: list[dict]) -> list[dict]:
        max_len = self.cfg["max_chars_per_line"]
        lines = []
        for seg in segments:
            words = seg.get("words") or []
            chars: list[dict] = []
            for w in words:
                text = w["word"].replace(" ", "")
                if not text:
                    continue
                dur = max(w["end"] - w["start"], 0.01) / len(text)
                for i, ch in enumerate(text):
                    chars.append(
                        {"ch": ch, "start": w["start"] + i * dur, "end": w["start"] + (i + 1) * dur}
                    )
            if not chars:
                text = seg["text"].replace(" ", "")
                if not text:
                    continue
                dur = max(seg["end"] - seg["start"], 0.2) / len(text)
                chars = [
                    {"ch": c, "start": seg["start"] + i * dur, "end": seg["start"] + (i + 1) * dur}
                    for i, c in enumerate(text)
                ]
            for i in range(0, len(chars), max_len):
                chunk = chars[i : i + max_len]
                lines.append(
                    {
                        "chars": chunk,
                        "text": "".join(c["ch"] for c in chunk),
                        "start": chunk[0]["start"],
                        "end": chunk[-1]["end"],
                        "seg_end": seg["end"],
                    }
                )
        return lines

    # ---------- 渲染单行 ----------

    def _shadow_tile(self, text: str, w: int, h: int, x0: int, y0: int) -> Image.Image:
        """黑色投影贴图（只与文本/几何有关，按文本缓存）。"""
        key = (text, w, h)
        sh = self._shadow_cache.get(key)
        if sh is None:
            sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            sd = ImageDraw.Draw(sh)
            x = x0
            for ch in text:
                sd.text((x, y0), ch, font=self.font, fill=(0, 0, 0, self.cfg["shadow_alpha"]))
                x += self.font.getlength(ch)
            sh = sh.filter(ImageFilter.GaussianBlur(self.cfg["shadow_blur"]))
            if len(self._shadow_cache) > _TILE_CACHE_MAX:
                self._shadow_cache.clear()
            self._shadow_cache[key] = sh
        return sh

    def _render_line(self, chars_colors: list[tuple[str, tuple]]) -> Image.Image:
        cfg = self.cfg
        text = "".join(c for c, _ in chars_colors)
        pad = 16
        bbox = self.font.getbbox(text)
        w = bbox[2] - bbox[0] + pad * 2
        h = bbox[3] - bbox[1] + pad * 2 + (cfg["shadow_offset"] if cfg["shadow"] else 0)

        txt = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(txt)
        x0 = pad - bbox[0]
        y0 = pad - bbox[1]
        x = x0
        for ch, col in chars_colors:
            d.text((x, y0), ch, font=self.font, fill=col)
            x += self.font.getlength(ch)

        if not cfg["shadow"]:
            return txt

        sh = self._shadow_tile(text, w, h, x0, y0)
        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        out.alpha_composite(sh, (0, cfg["shadow_offset"]))
        out.alpha_composite(txt, (0, 0))
        return out

    def _line_tile(self, chars_colors: list[tuple[str, tuple]]) -> Image.Image:
        """带缓存的文字贴图：同一文本+同一逐字颜色状态只渲染一次。"""
        key = (
            "".join(c for c, _ in chars_colors),
            tuple(col for _, col in chars_colors),
        )
        tile = self._tile_cache.get(key)
        if tile is None:
            tile = self._render_line(chars_colors)
            if len(self._tile_cache) > _TILE_CACHE_MAX:
                self._tile_cache.clear()
            self._tile_cache[key] = tile
        return tile

    # ---------- 锚点（避开人物框） ----------

    def _anchor(self, avoid_box):
        up = self.cfg["anchor_up"] * self.H  # 整体往上偏移的像素量
        if avoid_box:
            x0, y0, x1, y1 = avoid_box
            # 人物在左上：字幕水平居中；若框较宽占过中线，则整体右移
            if x1 > self.W * 0.45:
                ax = int((x1 + self.W) / 2)
            else:
                ax = self.W // 2
            # 若框较高，字幕下移到框下方区域
            if y1 > self.H * 0.40:
                ay = min(self.H * 0.74, y1 + (self.H - y1) * 0.5)
            else:
                ay = self.H * 0.60
            return ax, max(int(ay - up), 0)
        ay = self.H * 0.60 - up
        return self.W // 2, max(int(ay), 0)

    # ---------- 颜色与透明度 ----------

    def _colors_for(self, ln: dict, t: float, kind: str) -> list[tuple[str, tuple]]:
        cfg = self.cfg
        if kind == "prev":
            return [(c["ch"], cfg["fill"]) for c in ln["chars"]]
        if kind == "next":
            return [(c["ch"], cfg["upcoming"]) for c in ln["chars"]]
        # active：卡拉OK
        out = []
        for c in ln["chars"]:
            if t >= c["end"]:
                out.append((c["ch"], cfg["fill"]))
            elif t >= c["start"]:
                out.append((c["ch"], cfg["accent"]))
            else:
                out.append((c["ch"], cfg["upcoming"]))
        return out

    def _line_alpha(self, ln: dict, t: float, kind: str, depth: int = 1) -> float:
        cfg = self.cfg
        if kind == "active":
            p = _clamp((t - ln["start"]) / cfg["enter_dur"])
            a = _smoothstep(p)  # 极快淡入
            fade_start = ln["end"] + cfg["linger"]
            if t > fade_start:
                a *= _clamp(1.0 - (t - fade_start) / cfg["fade_out"])
            return a
        if kind == "prev":
            # 句末停留结束后才开始淡出
            base = fade_start = ln["end"] + cfg["linger"]
            if t <= base:
                a = cfg["prev_alpha"]
            else:
                a = cfg["prev_alpha"] * _clamp(1.0 - (t - base) / 0.8)
            # 越往上越淡，但不低于下限
            return max(a * (cfg["prev_alpha_fade"] ** (depth - 1)), cfg["prev_min_alpha"])
        # next：预览常驻淡，越往下越淡
        return cfg["next_alpha"] * (cfg["next_alpha_fade"] ** (depth - 1))

    # ---------- 可见性 ----------

    def _active_and_visible(self, t: float):
        """返回 (active_idx, 是否可见)。"""
        if not self.lines:
            return None, False
        active = 0
        for idx, ln in enumerate(self.lines):
            if ln["start"] <= t + 0.05:
                active = idx
        last = self.lines[-1]
        if active == len(self.lines) - 1 and t > last["end"] + self.cfg["linger"] + 0.4:
            return active, False
        if t < self.lines[0]["start"] - 0.3:
            return active, False
        return active, True

    # ---------- 主渲染 ----------

    def render_tiles(self, t: float, avoid_box=None) -> list[tuple[np.ndarray, int, int]]:
        """返回本帧需要叠加的文字贴图 [(rgba_uint8, x, y), ...]；无可见字幕返回 []。"""
        active, visible = self._active_and_visible(t)
        if not visible:
            return []

        ax, ay = self._anchor(avoid_box)

        # 当前句 + 上下各 context_lines 行（默认共 5 行）
        cl = max(0, int(self.cfg["context_lines"]))
        kinds: dict[int, tuple[str, int]] = {active: ("active", 0)}
        for step in range(1, cl + 1):
            if active - step >= 0:
                kinds[active - step] = ("prev", step)
            if active + step < len(self.lines):
                kinds[active + step] = ("next", step)

        imgs = {
            idx: self._line_tile(self._colors_for(self.lines[idx], t, kind))
            for idx, (kind, _depth) in kinds.items()
        }

        positions = {active: (ax, ay)}
        yy = ay
        for k in range(active - 1, -1, -1):
            if k not in imgs:
                break
            yy -= (imgs[k + 1].height // 2 + self.cfg["line_gap"] + imgs[k].height // 2)
            positions[k] = (ax, yy)
        yy = ay
        for k in range(active + 1, len(self.lines)):
            if k not in imgs:
                break
            yy += (imgs[k - 1].height // 2 + self.cfg["line_gap"] + imgs[k].height // 2)
            positions[k] = (ax, yy)

        tiles: list[tuple[np.ndarray, int, int]] = []
        for idx, (x, y) in positions.items():
            kind, depth = kinds.get(idx, ("active", 0))
            alpha = self._line_alpha(self.lines[idx], t, kind, depth)
            if alpha <= 0.01:
                continue
            frame = imgs[idx]
            cx = (self.W - frame.width) // 2
            px = max(cx, 0)
            py = int(y - frame.height // 2)
            # PIL 为 RGBA，合成层按 BGRA 解析 -> 转通道序（贴图很小，开销可忽略）
            bgra = np.ascontiguousarray(np.asarray(frame)[:, :, [2, 1, 0, 3]])
            if alpha < 0.999:
                bgra = bgra.copy()
                bgra[:, :, 3] = (bgra[:, :, 3].astype(np.float32) * alpha).astype(np.uint8)
            # 裁剪到画布范围内（越界部分直接裁掉，避免合成时被 clamp 平移错位）
            x0, y0 = max(px, 0), max(py, 0)
            x1 = min(px + bgra.shape[1], self.W)
            y1 = min(py + bgra.shape[0], self.H)
            if x1 <= x0 or y1 <= y0:
                continue
            if (x0, y0, x1, y1) != (px, py, px + bgra.shape[1], py + bgra.shape[0]):
                bgra = np.ascontiguousarray(bgra[y0 - py : y1 - py, x0 - px : x1 - px])
                px, py = x0, y0
            tiles.append((bgra, px, py))
        return tiles

    # ---------- 兼容：整层渲染（内部已不再使用） ----------

    def render(self, t: float, avoid_box=None) -> Image.Image | None:
        """整张画布 RGBA 图层（兼容旧接口；合成层应改用 render_tiles 以获得速度）。"""
        if not self.lines:
            return None
        _, visible = self._active_and_visible(t)
        if not visible:
            return None
        layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        for bgra, x, y in self.render_tiles(t, avoid_box):
            rgba = np.ascontiguousarray(bgra[:, :, [2, 1, 0, 3]])
            layer.alpha_composite(Image.fromarray(rgba, "RGBA"), (x, y))
        return layer
