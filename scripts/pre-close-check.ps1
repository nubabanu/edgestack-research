# Pre-close heads-up. Schedule weekdays ~21:35 local (≈15:35 ET in summer;
# adjust for winter with install-autostart.ps1 -PreCloseTime):
#   see scripts/install-autostart.ps1
#
# Reads today's published tailwind calendars, applies the nightly entry-signal
# rules, and pushes one Telegram message before the 15:45 ET decision freeze.
# Silent on non-sessions and without Telegram credentials; never fails loudly.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$log = "artifacts/advisor/pre-close-check.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
"=== pre-close run $(Get-Date -Format o) ===" | Out-File $log -Append
python -m edgestack.cli pre-close-check 2>&1 | Out-File $log -Append
"=== exit $LASTEXITCODE $(Get-Date -Format o) ===" | Out-File $log -Append
exit 0
