# One-time registration of the three scheduled tasks that make the system
# hands-free (run from any PowerShell; no admin needed for per-user tasks):
#   1. EdgeStack Mobile API  - starts the phone server at every logon
#   2. EdgeStack Post-Close  - nightly scan/ledger/calendar refresh at 22:40
#      local time (~16:40 ET during summer; adjust -PostCloseTime in winter)
#   3. EdgeStack Pre-Close   - weekday heads-up at 21:35 local (~15:35 ET in
#      summer, 10 min before the decision freeze; adjust -PreCloseTime)
#
# The one-time inbound firewall rule still needs an ADMIN terminal:
#   netsh advfirewall firewall add rule name="EdgeStack Mobile API" dir=in action=allow protocol=TCP localport=8765

param(
    [string]$PostCloseTime = "22:40",
    [string]$PreCloseTime = "21:35"
)

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pwshExe = (Get-Command pwsh).Source

schtasks /Create /F /TN "EdgeStack Mobile API" /SC ONLOGON /RL LIMITED `
    /TR "`"$pwshExe`" -NoProfile -WindowStyle Hidden -File `"$repo\scripts\serve-mobile.ps1`""
schtasks /Create /F /TN "EdgeStack Post-Close" /SC DAILY /ST $PostCloseTime /RL LIMITED `
    /TR "`"$pwshExe`" -NoProfile -WindowStyle Hidden -File `"$repo\scripts\post-close-job.ps1`""
schtasks /Create /F /TN "EdgeStack Pre-Close" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST $PreCloseTime /RL LIMITED `
    /TR "`"$pwshExe`" -NoProfile -WindowStyle Hidden -File `"$repo\scripts\pre-close-check.ps1`""

Write-Host ""
Write-Host "Registered. Verify with: schtasks /Query /TN `"EdgeStack Post-Close`""
Write-Host "The post-close job skips holidays automatically (no completed session = no new scan)."
