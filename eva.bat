@echo off
chcp 65001 >nul
cd /d "%~dp0"
python train_heads.py --resume checkpoints/v3/train_v3_step_13000.pt --steps 37000
pause
