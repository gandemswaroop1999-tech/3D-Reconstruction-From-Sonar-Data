@echo off
cd /d "%~dp0"
"R:\sonar\envs\suop-recon\python.exe" denoise_pipeline.py > denoise.log 2>&1
echo BAT_EXIT=%ERRORLEVEL% >> denoise.log
echo done > denoise_done.marker
