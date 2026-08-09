@echo off
cd /d "%~dp0"
"R:\sonar\envs\suop-recon\python.exe" build_cache.py > build_cache.log 2>&1
echo BAT_EXIT=%ERRORLEVEL% >> build_cache.log
echo done > build_cache_done.marker
