@echo off
chcp 65001 >nul 2>&1
setlocal

set "PY=C:\Users\郝晨汝\.workbuddy\binaries\python\envs\video_processor\Scripts\python.exe"
set "ROOT=D:\github\Luffy-slice\video_processor"
set "LOG=%ROOT%\run.log"

cd /d "%ROOT%" || (echo [错误] 无法进入目录：%ROOT% & pause & exit /b 1)

if not exist "%PY%" (
  echo [错误] 找不到 Python 解释器：
  echo   %PY%
  echo 请先创建虚拟环境并安装依赖：
  echo   python -m venv "C:\Users\郝晨汝\.workbuddy\binaries\python\envs\video_processor"
  echo   "%PY%" -m pip install -r requirements.txt
  pause
  exit /b 1
)

REM 依赖自检
if not exist "..\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe" (
  echo [警告] 未找到 ffmpeg：..\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe
  echo        读取视频信息 / 生成成品会失败，请先按 README 放置 ffmpeg 二进制。
)
if not exist "models\faster-whisper-base\model.bin" (
  echo [警告] 未找到语音模型：models\faster-whisper-base\
  echo        字幕转写会失败，请先按 README 放置模型。
)
if not exist "models\selfie_segmenter.tflite" (
  echo [警告] 未找到分割模型：models\selfie_segmenter.tflite
)

REM 若 8501 已被占用，自动释放（避免端口冲突导致启动失败 / 闪退）
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8501" ^| findstr "LISTENING"') do (
  taskkill /PID %%a /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo ============================================================
echo   Luffy 视频生成器 - 正在启动
echo   日志文件: %LOG%
echo   启动成功后浏览器会自动打开 http://localhost:8501
echo   关闭本窗口即可停止服务
echo ============================================================

REM 6 秒后自动打开浏览器（等待服务就绪）
start "" /min cmd /c "timeout /t 6 /nobreak >nul & start "" http://localhost:8501"

"%PY%" -m streamlit run app.py --server.port 8501 --server.headless true > "%LOG%" 2>&1

echo.
echo [已退出] 服务已停止。如果启动失败，请打开上面的日志文件查看原因。
pause
