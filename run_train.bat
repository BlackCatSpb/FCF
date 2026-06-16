@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python train_full.py --epochs 3 --fresh
pause
