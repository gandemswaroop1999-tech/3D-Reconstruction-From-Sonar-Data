@echo off
cd /d "%~dp0"
del /f /q segment_v3.log segment_v3_done.marker 2>nul
"R:\sonar\envs\suop-recon\python.exe" segment_pipeline_v3.py > segment_v3.log 2>&1
echo BAT_EXIT=%ERRORLEVEL% >> segment_v3.log
echo done > segment_v3_done.marker
