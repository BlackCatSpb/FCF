@echo off
chcp 65001 >nul
cd /d "C:\Users\black\OneDrive\Desktop\FCF"
set OMP_NUM_THREADS=1
python gen_quick.py
pause
