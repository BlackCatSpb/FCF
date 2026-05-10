@echo off
chcp 65001 >nul
cd /d "%~dp0"

if "%~1"=="" (
    python run.py --interactive --checkpoint checkpoints\instruction\final
) else (
    python run.py %*
)

pause
