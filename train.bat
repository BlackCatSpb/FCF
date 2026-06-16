@echo off
cd /d "%~dp0"

if exist real_data\checkpoint_state.json (
    set RESUME_ARG=--resume
) else (
    set RESUME_ARG=
)

echo [EVA] Training launch at %date% %time%
echo [EVA] Log: real_data\train_log.txt
echo.
python train_full.py %RESUME_ARG%
if %ERRORLEVEL% NEQ 0 (
    echo [EVA] Training exited with code %ERRORLEVEL%
    pause
)
