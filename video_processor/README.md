# video_processor

处理视频的小工具：上传普通视频 → 自动抠出人物放左上角 + 自选背景（纯色/风景视频）+ 语音转写生成「滚动反转」大字字幕 → 输出 9:16 竖屏成品。

## 结构

```
video_processor/
├── app.py                      # Streamlit 网页界面（首页 / 生成）
├── main.py                     # 命令行入口
├── requirements.txt
├── models/
│   ├── faster-whisper-base/    # 本地语音识别模型（字幕转写）
│   └── selfie_segmenter.tflite # mediapipe 人物分割模型
├── uploads/                    # 网页上传的视频
├── output/                     # 生成的成品
└── video_processor/
    ├── ffmpeg_utils.py         # ffmpeg/ffprobe 封装（用工作区二进制）
    ├── transcribe.py           # faster-whisper 转写（词级时间戳）
    ├── segmentation.py         # mediapipe 人物抠像
    ├── subtitles.py            # 滚动反转字幕渲染（卡拉OK高亮+翻转入场+上滚）
    └── compose.py              # 合成管线（逐帧合成 -> ffmpeg H.264）
```

## 运行

```bash
cd video_processor
pip install -r requirements.txt
streamlit run app.py
```

打开 http://localhost:8501 ，左侧切到「生成」页：
1. 上传视频（或填本地路径）
2. 选背景：纯色取色器 / 本地背景视频 / 在线视频直链
3. 调人物大小、位置、字号
4. 点「生成」，完成后可预览和下载

## 说明

- ffmpeg 二进制位于仓库根目录的 `ffmpeg-master-latest-win64-gpl-shared\bin`，`ffmpeg_utils.py` 自动按相对路径定位，无需安装（详见仓库根 README 的「环境准备」）。
- 依赖用虚拟环境安装 `requirements.txt` 中的包即可。
- 9 秒视频生成约 60 秒（纯色背景），人物抠像逐帧跑 CPU，视频越长越久。
- 字幕效果（字号/行宽/滚动速度/翻转时长/颜色）在 `subtitles.py` 的 `DEFAULTS` 里可调。
