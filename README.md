# Luffy-slice

把普通口播视频自动剪辑成 9:16 竖屏短视频的小工具：上传视频 → 自动抠出人物放左上角 + 自选背景（纯色 / 风景视频 / 在线）→ 语音转写生成「滚动反转」大字字幕 → 输出成品。

> 项目源码位于 `video_processor/` 目录。

## 目录结构

```
.
├── ffmpeg-master-latest-win64-gpl-shared/   # ffmpeg 二进制（见「环境准备」，不提交）
└── video_processor/
    ├── app.py                  # Streamlit 网页界面（首页 / 生成）
    ├── main.py                 # 命令行入口
    ├── requirements.txt
    ├── config.example.py       # 配置模板（复制为 config.py）
    ├── models/                 # 运行时模型（见「环境准备」，不提交）
    ├── uploads/                # 网页上传的视频（不提交）
    ├── output/                 # 生成的成品（不提交）
    └── video_processor/        # 核心包
        ├── ffmpeg_utils.py     # ffmpeg/ffprobe 封装
        ├── transcribe.py       # faster-whisper 转写（词级时间戳）
        ├── segmentation.py     # mediapipe 人物分割
        ├── subtitles.py        # 滚动反转字幕渲染（卡拉OK高亮 + 翻转入场 + 上滚）
        └── compose.py          # 合成管线（逐帧合成 → ffmpeg H.264）
```

## 环境准备

1. **Python 3.11+** 与虚拟环境，安装依赖：
   ```bash
   cd video_processor
   pip install -r requirements.txt
   ```
2. **ffmpeg 二进制**：代码按相对路径 `../ffmpeg-master-latest-win64-gpl-shared/bin` 定位 ffmpeg / ffprobe。
   下载 [ffmpeg-master-latest-win64-gpl-shared](https://www.gyan.dev/ffmpeg/builds/)（或官方 gpl shared 版），
   解压到**仓库根目录**，使 `ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe` 与 `ffprobe.exe` 存在。
3. **运行时模型**（不提交，需自行放置到 `video_processor/models/`）：
   - `faster-whisper-base/`：从 [HuggingFace Systran/faster-whisper-base](https://huggingface.co/Systran/faster-whisper-base) 下载全部文件，放入 `video_processor/models/faster-whisper-base/`。
   - `selfie_segmenter.tflite`：从 [Google Mediapipe](https://developers.google.com/mediapipe) 下载 selfie segmenter 模型，放入 `video_processor/models/`。
4. **配置（可选）**：复制 `video_processor/config.example.py` 为 `video_processor/config.py`，
   填入 Pexels / Pixabay API key（用于在线背景视频）；也可直接设置环境变量 `PEXELS_API_KEY` / `PIXABAY_API_KEY`。
   缺省也能用「纯色 / 本地素材库 / 在线直链」背景。

## 运行

```bash
cd video_processor
streamlit run app.py
```

打开 http://localhost:8501 ，左侧切到「生成」页：
1. 上传视频（或填本地路径）
2. 选背景：纯色取色器 / 本地背景视频 / 在线视频直链
3. 调人物大小、位置、字号
4. 点「生成」，完成后可预览和下载

## 说明

- 输出 9:16 竖屏；人物抠像逐帧跑 CPU，视频越长耗时越久。
- 字幕效果（字号 / 行宽 / 滚动速度 / 翻转时长 / 颜色）在 `video_processor/subtitles.py` 的 `DEFAULTS` 里可调。
