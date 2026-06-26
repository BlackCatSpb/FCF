@echo off
cd /d "%~dp0"

if exist real_data\checkpoint_state.json (
    set EXTRA=--resume
) else (
    set EXTRA=--fresh
)

set COMMON=--learned-fields --field-bits 512 --seed-e5 --e5-morph-bundle

echo [EVA] Training launch at %date% %time%
echo [EVA] Log: real_data\train_log.txt
echo [EVA] Command: python train_full.py %EXTRA% %COMMON%
echo.

python train_full.py %EXTRA% %COMMON%
if %ERRORLEVEL% NEQ 0 (
    echo [EVA] Training exited with code %ERRORLEVEL%
    pause
)
