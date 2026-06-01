<#
.SYNOPSIS
  EVA - autonomous thinking loop launcher.
  Launches think_loop + web dashboard at http://localhost:8383
.DESCRIPTION
  Run: powershell -ExecutionPolicy Bypass -File run_eva.ps1
#>

$FCF = Split-Path -Parent $MyInvocation.MyCommand.Path
$PORT = 8383

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  EVA - autonomous thinking loop" -ForegroundColor Cyan
Write-Host "  Dashboard: http://localhost:$PORT" -ForegroundColor Cyan
Write-Host "  Ctrl+C to stop" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check Python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    Read-Host -Prompt "Press Enter"
    exit 1
}

# Start think_loop
Write-Host "Starting EVA..."
try {
    python -X utf8 (Join-Path $FCF "eva\core\think_loop.py") --port $PORT
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Read-Host -Prompt "Press Enter"
}
