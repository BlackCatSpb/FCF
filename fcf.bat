@echo off
chcp 65001 >nul
cd /d "%~dp0"
python run.py --lazy-learn --checkpoint checkpoints\language\step_023000
pause
