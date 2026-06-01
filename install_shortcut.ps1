<#
.SYNOPSIS
  Creates EVA desktop shortcut.
#>
$wshell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')
$fcf = Split-Path -Parent $MyInvocation.MyCommand.Path
$shortcut = $wshell.CreateShortcut("$desktop\EVA.lnk")
$shortcut.TargetPath = "powershell.exe"
$shortcut.Arguments = "-NoExit -ExecutionPolicy Bypass -File `"$fcf\run_eva.ps1`""
$shortcut.WorkingDirectory = $fcf
$shortcut.Description = "EVA - autonomous thinking loop"
$shortcut.WindowStyle = 1
$py = (Get-Command python).Source
if (Test-Path $py) {
    $shortcut.IconLocation = "$py,0"
}
$shortcut.Save()
Write-Host "Shortcut created: $desktop\EVA.lnk" -ForegroundColor Green
Write-Host "Double-click EVA on your desktop to start." -ForegroundColor Cyan
