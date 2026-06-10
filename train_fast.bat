@echo off
cd /d "C:\Users\black\OneDrive\Desktop\FCF"
echo [EVA] FAST MODE at %date% %time%
echo [EVA] Log: real_data\train_log.txt
echo.
python train_full.py --fast
if %ERRORLEVEL% NEQ 0 (
    echo [EVA] Training exited with code %ERRORLEVEL%
    pause
)
