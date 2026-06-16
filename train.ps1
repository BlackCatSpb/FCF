param(
    [switch]$Resume,
    [switch]$QuickTest,
    [int]$MaxLines = 0
)

$FCF = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $FCF

Write-Host "[EVA] Training launcher" -ForegroundColor Cyan
Write-Host "  Resume=$Resume  QuickTest=$QuickTest  MaxLines=$MaxLines"
Write-Host "  Log: real_data\train_log.txt"
Write-Host

$args = @()
if ($Resume) {
    $args += "--resume"
}
if ($QuickTest) {
    $args += "--max-lines", "100"
    Write-Host "[EVA] Quick test mode: 100 lines" -ForegroundColor Yellow
} elseif ($MaxLines -gt 0) {
    $args += "--max-lines", "$MaxLines"
    Write-Host "[EVA] Max lines: $MaxLines" -ForegroundColor Yellow
}

python train_full.py $args

if ($LASTEXITCODE -ne 0) {
    Write-Host "[EVA] FAILED with code $LASTEXITCODE" -ForegroundColor Red
    pause
}
