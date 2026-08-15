@echo off
cd /d "%~dp0"

if exist checkpoints\meta.json (
    set EXTRA=--resume
) else (
    set EXTRA=--fresh
)

set QWEN_SEED=checkpoints\qwen_concept_vectors.npy
if exist %QWEN_SEED% (
    set QWEN_ARG=--qwen-seed %QWEN_SEED%
) else (
    set QWEN_ARG=
)

set COMMON=--learned-fields --field-bits 512 --vocab-size 256000 --neg-samples 3 --context-window 4 --pmi-gate 0.0

echo [EVA] Training launch at %date% %time%
echo [EVA] Log: train.log
echo [EVA] Command: python train_full.py %EXTRA% %QWEN_ARG% %COMMON%
echo.

python train_full.py %EXTRA% %QWEN_ARG% %COMMON%
if %ERRORLEVEL% NEQ 0 (
    echo [EVA] Exited with code %ERRORLEVEL%
    pause
)
