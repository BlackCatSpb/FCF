# Run training in a new terminal window with live metrics
$scriptPath = Join-Path $PSScriptRoot "train_live.py"
$corpusPath = Join-Path $PSScriptRoot "real_data\full_corpus_ru.txt"
$ckptDir = Join-Path $PSScriptRoot "checkpoints"

Start-Process -WindowStyle Normal -FilePath "powershell" -ArgumentList @"
-NoExit -Command `
  cd '$PSScriptRoot'; `
  python train_live.py --corpus '$corpusPath' --checkpoint-dir '$ckptDir' --test-every 500 --save-every 5000
"@
