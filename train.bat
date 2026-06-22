@echo off
cd /d "%~dp0"

if exist real_data\checkpoint_state.json (
    set EXTRA=--resume
) else (
    set EXTRA=--fresh
)

echo [EVA] Training launch at %date% %time%
echo [EVA] Log: real_data\train_log.txt
echo.
python train_full.py %EXTRA% --learned-fields --field-bits 512
if %ERRORLEVEL% NEQ 0 (
    echo [EVA] Training exited with code %ERRORLEVEL%
    pause
)
