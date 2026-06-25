@echo off
cd /d "%~dp0"

if exist real_data\checkpoint_state.json (
    set EXTRA=--resume
) else (
    set EXTRA=--fresh
)

set COMMON=--learned-fields --field-bits 512
set E5_ARGS=--seed-e5 --e5-morph-bundle
set MORPH_BPE=real_data\bpe_morph.model

if exist "%MORPH_BPE%" (
    echo [EVA] Morph-aware BPE model detected: %MORPH_BPE%
    set BPE_ARG=--morph-bpe %MORPH_BPE%
) else (
    echo [EVA] No morph BPE model, using default SentencePiece
    set BPE_ARG=
)

echo [EVA] Training launch at %date% %time%
echo [EVA] Log: real_data\train_log.txt
echo [EVA] Command: python train_full.py %EXTRA% %COMMON% %E5_ARGS% %BPE_ARG%
echo.

python train_full.py %EXTRA% %COMMON% %E5_ARGS% %BPE_ARG%
if %ERRORLEVEL% NEQ 0 (
    echo [EVA] Training exited with code %ERRORLEVEL%
    pause
)
