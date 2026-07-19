# EdgeStack Android companion

## Boundary

The Android application is a paper-research viewer, not a port of the numerical
engine and not a brokerage client. Python remains authoritative for ingestion,
causal feature computation, statistical testing, Zipline confirmation, freezing,
single-use holdout evaluation, scoring, and signal creation.

```text
canonical data + frozen Python model
              |
              v
sealed holdout / paper-signal artifacts
              |
              v
read-only bearer-authenticated API
              |
              v
Android snapshot cache and Compose UI
```

This split avoids trying to run CPython 3.12 scientific wheels, DuckDB scans,
Zipline, or multi-gigabyte Parquet snapshots inside a mobile application. It also
keeps final-holdout governance outside a user-controlled phone process.

## Mobile API

Run a static demonstration:

```powershell
edgestack mobile-api --demo --host 127.0.0.1 --port 8765
```

Run against an existing promoted campaign:

```powershell
$env:EDGESTACK_MOBILE_TOKEN = '<random value with at least 24 characters>'
edgestack mobile-api `
  --campaign reversal-edge-v1-20260715-001 `
  --host 0.0.0.0 `
  --port 8765
```

The API exposes only:

- `GET /api/v1/health`, which contains no research evidence;
- `GET /api/v1/mobile/snapshot`, which requires the bearer token outside an
  explicitly selected demo process.

There are no POST, PUT, PATCH, DELETE, broker, order, or holdout-evaluation
routes. A promoted snapshot is constructed only when the result is `PASS`, is
marked `FORBIDDEN_REPLAY_ONLY`, has a corresponding paper signal, and retains an
explicit bias tier. The response includes an ETag and `private, no-cache` policy.

Use a reverse proxy with TLS and authentication controls for access outside a
trusted development network. The built-in server defaults to loopback. Do not
place tokens in YAML, Gradle configuration, screenshots, or Git.

## Android build

The checked-in wrapper uses Gradle 9.4.1 and Android Gradle Plugin 9.2.0. The app
uses Kotlin and Compose compiler 2.3.21, the stable Compose BOM 2026.06.00,
`compileSdk = 36`, `targetSdk = 36`, and `minSdk = 26`. API 37 remains a
preview SDK, so the production build does not require it.

```powershell
cd android
./gradlew.bat testDebugUnitTest assembleDebug
```

The APK is written to `android/app/build/outputs/apk/debug/app-debug.apk`.
Install it with Android Studio or `adb install -r` when a device is connected.

The emulator resolves the development machine as `10.0.2.2`. Select Setup,
disable demo mode, enter `http://10.0.2.2:8765`, and enter the bearer token.
Physical devices should use an HTTPS URL. The network security policy allows
cleartext only for `10.0.2.2` and localhost.

When the server itself runs with `--demo`, the app accepts its response only as
visibly labeled demonstration data. It does not cache that response as sealed
evidence or display it as a network-validated promoted snapshot.

## Quick start (one command on the PC)

```powershell
pwsh scripts/serve-mobile.ps1
```

The script creates (or reuses) the bearer token at
`artifacts/advisor/mobile-token.txt` and prints it, refreshes the tailwind
calendar the Timing tab reads, and starts the read-only API on port 8765.
On the phone: same Wi-Fi (or Tailscale), API base URL `http://<pc-ip>:8765`,
the printed token, demo mode off, then **Test connection** → **Save and
refresh**. The one-time inbound firewall rule for the port must be added
from an administrator terminal (the command is in the script header).

For a fully automatic setup: enable **Remember token on this device** in the
app before saving (the app then reconnects by itself every launch), and
register the server to start at logon:

```powershell
schtasks /Create /TN "EdgeStack Mobile API" /SC ONLOGON /RL LIMITED `
  /TR "pwsh -NoProfile -WindowStyle Hidden -File k:\earnmoney\scripts\serve-mobile.ps1"
```

## Screens and behavior

- **Plan** shows the next eligible closing-auction entry, submission deadline,
  time exit, cancel conditions, paper capital, and risk constraints.
- **Basket** shows every name in the tested basket. It warns against selecting
  only rank one or substituting a missing name.
- **Sniper** defaults to `NO TRADE` and requires year, month, week, and entry-day
  layers to pass together. Its conservative paper overlay caps each name at 5%,
  gross exposure at 25%, planned loss at $100 per name, and aggregate planned
  basket loss at $500 on the $100,000 reference account. These values are risk
  constraints, not validated alpha and not guaranteed realized-loss limits.
- The Sniper screen also shows separate week, month, and year decisions. The promoted
  five-session basket appears under Week with entry/review/exit and cancellation
  rules. Month and Year fail closed as `DATA_UNAVAILABLE` and cannot emit a
  ticker until their own model, freeze, and future holdout exist.
- **Timing** shows the diagnostic tailwind calendar published by the server
  (`edgestack tailwind-calendar --symbol SPY --output
  artifacts/advisor/tailwind-calendar.json`): per-session win scores
  (reliability-weighted historical hit rates, never success probabilities),
  expected basis points, active calendar conditions, and the two measurable
  execution anchors (opening/closing auction) with their overnight/intraday
  legs. Hourly and 15-minute granularity are labeled `DATA_UNAVAILABLE`
  because daily bars hold no intraday prices. The whole tab carries a
  `DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER` watermark and is
  subordinate to the validated basket; when the server has no advisor
  artifact the tab fails closed to `DATA_UNAVAILABLE`.
- **Evidence** replays holdout coverage, mean returns, terminal wealth, hashes,
  and audit events. It cannot trigger a recomputation.
- **Setup** selects demo/network mode and an endpoint. By default the bearer
  token is held in memory and must be re-entered after process death; the
  opt-in "Remember token on this device" switch seals it with a
  hardware-backed Android Keystore AES-GCM key so the app reconnects by
  itself on launch. The sealed blob is useless off-device, and turning the
  switch off (or failing to seal on devices without a Keystore) wipes it. A **Test connection**
  button probes the server before saving: it reports reachability (with a
  Wi-Fi/server/firewall checklist on failure), whether the server is SEALED
  or demo, and whether the bearer token is accepted — without loading a full
  snapshot. Cleartext `http://` is accepted only for private-LAN (RFC 1918),
  Tailscale CGNAT (`100.64.0.0/10`), and loopback addresses; everything else
  requires HTTPS. This makes home-Wi-Fi (`http://192.168.x.x:8765`) and
  Tailscale (`http://100.x.y.z:8765`) work out of the box while public
  endpoints stay TLS-only.

The decoder rejects unknown fields, unsupported schema versions, non-contiguous
ranks, duplicate recommendation IDs, a promoted model without a passed holdout,
or short candidates when shorts are disabled. Network failures fall back to the
last validated sealed snapshot; absent that, the packaged demo is shown with an
explicit warning. Demo and stale data are never styled as fresh network data.

## Not to be confused with other apps

This companion (`com.edgestack.mobile`, "EdgeStack Paper") displays ONLY
evidence that passed this repository's gauntlet: one promoted basket, sealed
holdout replays, and clearly-watermarked diagnostics. Any other app —
including other EdgeStack-branded experiments such as `com.edgestack.app` —
draws on different pipelines with different (often far looser) validation
standards; hundreds of "validated" edges in another app do not carry this
repository's evidence discipline. When claims conflict, the sealed campaign
catalog in this repository is the authority, and mixing the two apps'
conclusions defeats the entire fail-closed design.

Transport note: cleartext HTTP is restricted to private ranges (LAN
192.168.x, Tailscale 100.x, emulator 10.0.2.2). For remote use prefer
Tailscale, which encrypts end-to-end at the tunnel layer without exposing
port 8765 to the internet; native TLS on the server would additionally
require distributing a certificate the app trusts.

## Security and limitations

- This is not a secure enclave or an order-management system.
- The API token is never logged or backed up; it is persisted only when the
  user opts into "Remember token", and then only as a Keystore-sealed blob.
- Cached sealed snapshots contain research evidence but no credentials.
- Free source quotes can be delayed or revised.
- Current-constituent results remain visibly `SURVIVORSHIP_BIASED`.
- Confidence is ordinal, not a probability of profit.
- The 2×ATR level is a reference risk control, not validated alpha.
- Stop triggers are not guaranteed execution prices; gaps and slippage can make
  realized loss exceed the displayed planned-loss budget.
- The app cannot make a stale recommendation current; refresh and causal
  server-side revalidation are required.
