@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if exist real_data\checkpoint_state.json (
    python train_full.py --epochs 3 --resume
) else (
    python train_full.py --epochs 3 --fresh
)
pause
