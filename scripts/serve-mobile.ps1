# Start the EdgeStack mobile companion server with one command.
#
# What it does, in order:
#   1. Ensures a bearer token exists at artifacts/advisor/mobile-token.txt
#      (creates a random 40-character token on first run) and prints it so it
#      can be typed into the phone's Setup screen.
#   2. Refreshes the advisor tailwind calendar the app's Timing tab reads.
#   3. Starts the read-only mobile API bound to all interfaces.
#
# Phone setup afterwards: same Wi-Fi, API base URL http://<this-PC-ip>:8765
# (ipconfig shows the address; Tailscale 100.x also works), the printed
# token, demo mode off, then "Test connection".
#
# One-time firewall rule (admin terminal):
#   netsh advfirewall firewall add rule name="EdgeStack Mobile API" dir=in action=allow protocol=TCP localport=8765

param(
    [string]$Campaign = "reversal-edge-v1-20260715-001",
    [string]$Symbol = "SPY",
    [int]$Port = 8765,
    [int]$CalendarSessions = 42
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$tokenPath = "artifacts/advisor/mobile-token.txt"
if (-not (Test-Path $tokenPath)) {
    New-Item -ItemType Directory -Force (Split-Path $tokenPath) | Out-Null
    $alphabet = [char[]]"abcdefghijklmnopqrstuvwxyz0123456789"
    $token = -join (1..40 | ForEach-Object { $alphabet | Get-Random })
    Set-Content $tokenPath -Value $token -NoNewline
    Write-Host "Created new bearer token."
}
$env:EDGESTACK_MOBILE_TOKEN = (Get-Content $tokenPath -Raw).Trim()
Write-Host "Bearer token (enter this on the phone): $env:EDGESTACK_MOBILE_TOKEN"

Write-Host "Refreshing tailwind calendar for $Symbol..."
python -m edgestack.cli tailwind-calendar --symbol $Symbol --sessions $CalendarSessions `
    --output artifacts/advisor/tailwind-calendar.json | Out-Null

Write-Host "Starting mobile API on port $Port (campaign $Campaign). Ctrl+C stops it."
python -m edgestack.cli mobile-api --host 0.0.0.0 --port $Port --campaign $Campaign
