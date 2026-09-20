"""现代精致「居中稳定」字幕渲染模块（v3）。

改动要点（相对 v2）：
- 不再持续向上滚动，改为"当前句居中、上一句在上淡出、下一句在下预览"，
  画面稳定，不再闪。
- 入场极快淡入（0.16s），彻底解决"读到了但字幕没出来"。
- 卡拉OK逐字点亮：已读=纯白 / 正在读=强调色 / 未读=柔和灰。
- 字幕占用"除人物框外"的全部空间：render(t, avoid_box) 动态计算锚点
  （人物在左上，则字幕居中偏下/偏右，避开左上）。
- 柔和投影保证在任意背景清晰。

用法:
    r = SubtitleRenderer(segments, canvas_w, canvas_h, **overrides)
    layer = r.render(t, avoid_box)   # PIL RGBA 图；无可见字幕时 None
"""

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
    "next_alpha": 0.38,   # 下一句预览透明度
}


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

        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        x = x0
        for ch, _ in chars_colors:
            sd.text((x, y0), ch, font=self.font, fill=(0, 0, 0, cfg["shadow_alpha"]))
            x += self.font.getlength(ch)
        sh = sh.filter(ImageFilter.GaussianBlur(cfg["shadow_blur"]))

        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        out.alpha_composite(sh, (0, cfg["shadow_offset"]))
        out.alpha_composite(txt, (0, 0))
        return out

    # ---------- 锚点（避开人物框） ----------

    def _anchor(self, avoid_box):
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
            return ax, int(ay)
        return self.W // 2, int(self.H * 0.60)

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

    def _line_alpha(self, ln: dict, t: float, kind: str) -> float:
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
                return cfg["prev_alpha"]
            return cfg["prev_alpha"] * _clamp(1.0 - (t - base) / 0.8)
        # next：预览常驻淡
        return cfg["next_alpha"]

    # ---------- 主渲染 ----------

    def render(self, t: float, avoid_box=None) -> Image.Image | None:
        if not self.lines:
            return None

        # 当前句 = 最新已开始（start <= t）的句子
        active = 0
        for idx, ln in enumerate(self.lines):
            if ln["start"] <= t + 0.05:
                active = idx

        # 可见性闸门：首尾空白期不画
        last = self.lines[-1]
        if active == len(self.lines) - 1 and t > last["end"] + self.cfg["linger"] + 0.4:
            return None
        if t < self.lines[0]["start"] - 0.3:
            return None

        layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        ax, ay = self._anchor(avoid_box)

        # 当前句居中，上一句在上、下一句在下
        to_draw = [(active, "active")]
        if active - 1 >= 0:
            to_draw.append((active - 1, "prev"))
        if active + 1 < len(self.lines):
            to_draw.append((active + 1, "next"))

        imgs = {idx: self._render_line(self._colors_for(self.lines[idx], t, kind))
                for idx, kind in to_draw}

        # 垂直布局：以当前句中心为 ay，依次向上/向下排
        active_img = imgs[active]
        positions = {active: (ax, ay)}
        # 向上
        yy = ay
        for k in range(active - 1, -1, -1):
            if k not in imgs:
                break
            yy -= (imgs[k + 1].height // 2 + self.cfg["line_gap"] + imgs[k].height // 2)
            positions[k] = (ax, yy)
        # 向下
        yy = ay
        for k in range(active + 1, len(self.lines)):
            if k not in imgs:
                break
            yy += (imgs[k - 1].height // 2 + self.cfg["line_gap"] + imgs[k].height // 2)
            positions[k] = (ax, yy)

        for idx, (x, y) in positions.items():
            kind = "active" if idx == active else ("prev" if idx < active else "next")
            alpha = self._line_alpha(self.lines[idx], t, kind)
            if alpha <= 0.01:
                continue
            frame = imgs[idx]
            if alpha < 0.999:
                a = frame.getchannel("A").point(lambda v: int(v * alpha))
                frame = frame.copy()
                frame.putalpha(a)
            cx = (self.W - frame.width) // 2
            layer.alpha_composite(frame, (max(cx, 0), int(y - frame.height // 2)))

        return layer
