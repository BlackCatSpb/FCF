@echo off
chcp 65001 >nul
cd /d "%~dp0"
python train_to_convergence.py
pause
