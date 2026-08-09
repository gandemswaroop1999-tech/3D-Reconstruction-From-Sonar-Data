@echo off
cd /d "%~dp0"
"R:\sonar\envs\suop-recon\python.exe" segment_pipeline.py > segment.log 2>&1
echo BAT_EXIT=%ERRORLEVEL% >> segment.log
echo done > segment_done.marker
