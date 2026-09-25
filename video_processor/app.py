"""Streamlit 网页界面。

启动: streamlit run app.py
页面: 首页(元信息) / 生成(上传视频 -> 选背景/音乐/字幕 -> 生成 9:16)
"""

import os
import shutil
import uuid
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
BG_LIBRARY = BASE_DIR / "models" / "backgrounds"

# 把 config.py 里的密钥注入进程环境，优先级低于真实环境变量
from video_processor.config import PEXELS_API_KEY, PIXABAY_API_KEY
os.environ.setdefault("PEXELS_API_KEY", PEXELS_API_KEY)
os.environ.setdefault("PIXABAY_API_KEY", PIXABAY_API_KEY)

st.set_page_config(page_title="视频处理工具", page_icon="🎬", layout="wide")

page = st.sidebar.radio("页面", ["首页", "生成"], index=0)


# ---------------- 首页: 元信息 ----------------
if page == "首页":
    st.title("🎬 视频处理工具")
    st.markdown("左侧切换到「生成」制作 9:16 成品。")

    path = st.text_input("视频文件路径", placeholder=r"例如 D:\videos\demo.mp4")
    if st.button("读取信息", type="primary"):
        if not path:
            st.warning("请先填写视频文件路径")
        else:
            try:
                from video_processor import ffmpeg_utils

                info = ffmpeg_utils.probe(path)
                st.success("解析完成")
                rows = []
                for s in info.get("streams", []):
                    rows.append(
                        {
                            "类型": s.get("codec_type"),
                            "编码": s.get("codec_name"),
                            "分辨率": f"{s.get('width', '?')}x{s.get('height', '?')}"
                            if s.get("codec_type") == "video" else "-",
                        }
                    )
                st.table(rows)
            except Exception as e:  # noqa: BLE001
                st.error(f"解析失败: {e}")


# ---------------- 生成页 ----------------
elif page == "生成":
    st.title("🚀 生成 9:16 成品")
    st.markdown("人物自动裁剪圈框贴左上角 + 背景 + 居中字幕（语音自动转写）+ 可选背景音乐")

    if st.session_state.get("last_out") and Path(st.session_state["last_out"]).exists():
        st.video(st.session_state["last_out"])
        st.caption("↑ 上次生成的成品")

    with st.expander("1️⃣ 输入视频", expanded=True):
        up = st.file_uploader("上传视频", type=["mp4", "mov", "avi", "mkv", "webm"])
        input_path = None
        if up is not None:
            UPLOAD_DIR.mkdir(exist_ok=True)
            input_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{up.name}"
            if not input_path.exists():
                with open(input_path, "wb") as f:
                    shutil.copyfileobj(up, f)
            st.success(f"已保存: {input_path.name}")
        manual = st.text_input("或填写本地路径", placeholder=r"D:\videos\xxx.mp4", key="manual")
        if manual:
            input_path = manual

    with st.expander("2️⃣ 背景", expanded=True):
        bg_mode = st.radio("背景来源", ["纯色", "本地素材库", "Pexels", "Pixabay", "在线直链"],
                           horizontal=True)
        background = {"type": "color", "color": "#13243B"}
        bg_query = "nature landscape"

        if bg_mode == "纯色":
            c = st.color_picker("背景颜色", "#13243B")
            background = {"type": "color", "color": c}

        elif bg_mode == "本地素材库":
            BG_LIBRARY.mkdir(parents=True, exist_ok=True)
            bgu = st.file_uploader("上传风景视频到素材库", type=["mp4", "mov", "mkv", "webm"],
                                   key="bglib")
            if bgu is not None:
                dst = BG_LIBRARY / f"{uuid.uuid4().hex[:8]}_{bgu.name}"
                with open(dst, "wb") as f:
                    shutil.copyfileobj(bgu, f)
                st.success(f"已加入素材库: {bgu.name}")
            clips = sorted(p for p in BG_LIBRARY.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv"))
            if clips:
                st.caption(f"素材库共 {len(clips)} 个，将随机取一个（背景短则循环、长则截取）")
                for p in clips[-8:]:
                    st.write(f"• {p.name}")
            else:
                st.info("素材库为空，先上传几个风景视频，或改用其他背景来源")

        elif bg_mode in ("Pexels", "Pixabay"):
            bg_query = st.text_input("搜索关键词（英文，如 nature / city / ocean）", "nature landscape")
            key_env = "PEXELS_API_KEY" if bg_mode == "Pexels" else "PIXABAY_API_KEY"
            if not os.environ.get(key_env):
                st.warning(
                    f"未检测到 {key_env} 环境变量。请到 "
                    f"{'https://www.pexels.com/api/' if bg_mode=='Pexels' else 'https://pixabay.com/api/docs/'} "
                    f"免费申请 key，并在本机设置环境变量后重启本程序。否则请改用「本地素材库」。"
                )
            else:
                st.caption(f"将实时从 {bg_mode} 随机下载一个「{bg_query}」视频作为背景（短则循环、长则截取）")

        else:  # 在线直链
            url = st.text_input("风景视频直链 URL", placeholder="https://.../xxx.mp4")
            background = {"type": "url", "url": url}

    with st.expander("3️⃣ 背景音乐（可选）"):
        bgm_up = st.file_uploader("上传背景音乐", type=["mp3", "wav", "m4a", "aac"], key="bgm")
        bgm_path = None
        if bgm_up is not None:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            bgm_path = UPLOAD_DIR / f"bgm_{uuid.uuid4().hex[:8]}_{bgm_up.name}"
            with open(bgm_path, "wb") as f:
                shutil.copyfileobj(bgm_up, f)
            st.success(f"已保存: {bgm_up.name}")
        bgm_volume = st.slider("背景音乐音量（相对原声）", 0.0, 1.0, 0.2, 0.05)
        st.caption("背景音乐自动小声循环/截取，与人物原声混音")

    with st.expander("4️⃣ 视觉风格"):
        STYLE_OPTS = {"极简高级": "luxe", "清新暖调": "fresh", "电影氛围": "cinematic"}
        style_name = st.selectbox("风格预设", list(STYLE_OPTS.keys()), index=0)
        style_key = STYLE_OPTS[style_name]
        colp0, colp0b = st.columns(2)
        accent = colp0.color_picker("强调色(卡拉OK)", "#FFD166")
        bg_dim = colp0b.slider("背景压暗", 0.0, 0.6, 0.10, 0.01)

    with st.expander("5️⃣ 人物 / 字幕参数"):
        colp1, colp2 = st.columns(2)
        person_h = colp1.slider("人物大小(占画面高)", 0.12, 0.6, 0.20, 0.01)
        pos_x = colp2.slider("人物水平位置", 0.0, 0.7, 0.10, 0.01)
        pos_y = colp1.slider("人物垂直位置", 0.0, 0.7, 0.05, 0.01)
        colp3, colp4 = st.columns(2)
        border_on = colp3.checkbox("人物描边", True)
        border_hex = colp4.color_picker("描边颜色", "#FFFFFF")
        colp5, colp6 = st.columns(2)
        font_size = colp5.slider("字幕字号", 48, 160, 92, 4)
        max_chars = colp6.slider("每行字数", 4, 14, 10, 1)
        person_shadow = st.checkbox("人物投影(景深)", True)
        lang = st.selectbox("语音语言", ["zh", "en", "ja", "ko"], index=0)
        _MODEL_LABELS = {
            "base": "base — 最快，准确率低",
            "small": "small — 快，准确率一般",
            "medium": "medium — 较准（推荐，默认）",
            "large-v3": "large-v3 — 最准，CPU 上明显更慢",
        }
        whisper_model = st.selectbox(
            "转写模型（越大越准、越慢）",
            list(_MODEL_LABELS.keys()),
            index=2,
            format_func=lambda k: _MODEL_LABELS[k],
        )

    if st.button("🪄 生成", type="primary", use_container_width=True):
        if not input_path:
            st.warning("请先提供输入视频")
            st.stop()

        # 解析背景
        try:
            if bg_mode == "纯色":
                background_final = background
            elif bg_mode == "在线直链":
                if not background.get("url"):
                    st.warning("请填写在线视频 URL")
                    st.stop()
                from video_processor import bg_provider
                with st.spinner("下载在线背景..."):
                    path = bg_provider.resolve("url", url=background["url"])
                background_final = {"type": "video", "path": path}
            else:
                from video_processor import bg_provider
                src = {"本地素材库": "local", "Pexels": "pexels", "Pixabay": "pixabay"}[bg_mode]
                with st.spinner(f"获取{src}背景..."):
                    path = bg_provider.resolve(src, query=bg_query)
                background_final = {"type": "video", "path": path}
                st.caption(f"背景: {Path(path).name}")
        except Exception as e:  # noqa: BLE001
            st.error(f"背景获取失败: {e}")
            st.stop()

        import threading
        import queue as _queue
        from video_processor.compose import generate

        out_path = BASE_DIR / "output" / f"{uuid.uuid4().hex[:8]}_out.mp4"
        bar = st.progress(0.0, text="准备中...")
        status = st.empty()
        status.write("⏳ 转写语音中…（首次约需 10-30 秒，之后开始逐帧合成）")

        # 子线程跑生成，主线程用队列轮询实时刷新进度条（避免长阻塞时进度条不刷新）
        q: "_queue.Queue" = _queue.Queue()

        def _run():
            try:
                generate(
                    input_video=str(input_path),
                    output_path=str(out_path),
                    background=background_final,
                    language=lang,
                    whisper_model=whisper_model,
                    person_height_ratio=person_h,
                    person_pos=(pos_x, pos_y),
                    border=border_on,
                    border_hex=border_hex,
                    style=style_key,
                    accent_hex=accent,
                    bg_dim=bg_dim,
                    person_shadow=person_shadow,
                    subtitle_overrides={"font_size": font_size, "max_chars_per_line": max_chars},
                    bgm_path=str(bgm_path) if bgm_path else None,
                    bgm_volume=bgm_volume,
                    model_size=model_size,
                    progress_cb=lambda p: q.put(("prog", p)),
                )
                q.put(("done", str(out_path)))
            except Exception as e:  # noqa: BLE001
                import traceback as _tb
                q.put(("err", f"{e}\n\n{_tb.format_exc()}", None))

        threading.Thread(target=_run, daemon=True).start()
        while True:
            item = q.get()
            if item[0] == "prog":
                bar.progress(item[1], text=f"合成进度 {int(item[1] * 100)}%")
            elif item[0] == "done":
                final = item[1]
                break
            else:  # err
                st.error(f"生成失败: {item[1]}")
                st.stop()

        bar.progress(1.0, text="完成!")
        st.success("生成成功")
        st.session_state["last_out"] = final
        st.video(final)
        with open(final, "rb") as f:
            st.download_button("⬇️ 下载成品", f, file_name=Path(final).name, mime="video/mp4")
