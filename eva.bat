@echo off
chcp 65001 >nul
cd /d "%~dp0"
python train_full_corpus.py
pause
