param(
    [switch]$Resume,
    [switch]$QuickTest,
    [int]$MaxLines = 0
)

$FCF = "C:\Users\black\OneDrive\Desktop\FCF"
Set-Location $FCF

Write-Host "[EVA] Training launcher" -ForegroundColor Cyan
Write-Host "  Resume=$Resume  QuickTest=$QuickTest  MaxLines=$MaxLines"
Write-Host "  Log: real_data\train_log.txt"
Write-Host

if ($QuickTest) {
    $env:EVA_MAX_LINES = "100"
    Write-Host "[EVA] Quick test mode: 100 lines" -ForegroundColor Yellow
} elseif ($MaxLines -gt 0) {
    $env:EVA_MAX_LINES = "$MaxLines"
    Write-Host "[EVA] Max lines: $MaxLines" -ForegroundColor Yellow
}

python train_full.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[EVA] FAILED with code $LASTEXITCODE" -ForegroundColor Red
    pause
}
