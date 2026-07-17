# Nightly post-close job. Schedule for ~16:35 ET (22:35 CET in summer):
#   see scripts/install-autostart.ps1
#
# Regenerates the five-name basket signal from the completed close, appends
# fills/marks/exits to the append-only forward ledger, refreshes the tailwind
# calendars, and logs a one-line summary. Idempotent; safe to re-run.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$log = "artifacts/advisor/post-close-job.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
"=== post-close run $(Get-Date -Format o) ===" | Out-File $log -Append
python -m edgestack.cli post-close 2>&1 | Out-File $log -Append
"=== exit $LASTEXITCODE $(Get-Date -Format o) ===" | Out-File $log -Append
exit $LASTEXITCODE
