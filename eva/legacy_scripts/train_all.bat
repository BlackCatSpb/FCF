@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo ============================================================
echo FCF Training Pipeline
echo ============================================================
echo.
echo [1/4] Training tokenizer on training_corpus.txt ...
python -c "import sys; sys.path.insert(0,'.'); from fcf.tokenizer_utils import train_bpe_tokenizer; it = open('training_corpus.txt',encoding='utf-8'); t = train_bpe_tokenizer('tokenizer.json', (l.strip() for l in it if l.strip()), vocab_size=10000, min_frequency=1); print('Tokenizer ready:', t.get_vocab_size(), 'tokens')"
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [2/4] Language training (Point 2) - 2000 steps ...
python run.py --train-language --text-file training_corpus.txt --max-steps 2000 --device cpu
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [3/4] Instruction training (Point 3) - 500 steps ...
python run.py --train-instruction --checkpoint checkpoints\language\final --instructions-file tests\test_instructions.json --max-steps 500 --device cpu
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [4/4] Full system test ...
python run.py --full-test --checkpoint checkpoints\instruction\final
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo ============================================================
echo Training complete! Starting interactive mode...
echo ============================================================
python run.py --interactive --checkpoint checkpoints\instruction\final
goto :end

:error
echo.
echo ERROR during training!
pause
:end
