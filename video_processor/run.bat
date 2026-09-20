@echo off
chcp 65001 >nul 2>&1
setlocal

set "PY=C:\Users\jack\.workbuddy\binaries\python\envs\video_processor\Scripts\python.exe"
set "ROOT=D:\pro\QP\video_processor"
set "LOG=%ROOT%\run.log"

cd /d "%ROOT%" || (echo [错误] 无法进入目录：%ROOT% & pause & exit /b 1)

if not exist "%PY%" (
  echo [错误] 找不到 Python 解释器：
  echo   %PY%
  echo 请确认虚拟环境路径是否正确（可能需重新创建 venv）。
  pause
  exit /b 1
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
