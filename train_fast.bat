@echo off
cd /d "%~dp0"
echo [EVA] FAST MODE at %date% %time%
echo [EVA] Log: real_data\train_log.txt
echo.
python train_full.py --fast
if %ERRORLEVEL% NEQ 0 (
    echo [EVA] Training exited with code %ERRORLEVEL%
    pause
)
