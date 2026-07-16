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

## Screens and behavior

- **Plan** shows the next eligible closing-auction entry, submission deadline,
  time exit, cancel conditions, paper capital, and risk constraints.
- **Basket** shows every name in the tested basket. It warns against selecting
  only rank one or substituting a missing name.
- **Evidence** replays holdout coverage, mean returns, terminal wealth, hashes,
  and audit events. It cannot trigger a recomputation.
- **Setup** selects demo/network mode and an endpoint. Bearer tokens are held in
  memory and must be re-entered after process death.

The decoder rejects unknown fields, unsupported schema versions, non-contiguous
ranks, duplicate recommendation IDs, a promoted model without a passed holdout,
or short candidates when shorts are disabled. Network failures fall back to the
last validated sealed snapshot; absent that, the packaged demo is shown with an
explicit warning. Demo and stale data are never styled as fresh network data.

## Security and limitations

- This is not a secure enclave or an order-management system.
- No API token is persisted, backed up, or added to logs.
- Cached sealed snapshots contain research evidence but no credentials.
- Free source quotes can be delayed or revised.
- Current-constituent results remain visibly `SURVIVORSHIP_BIASED`.
- Confidence is ordinal, not a probability of profit.
- The 2×ATR level is a reference risk control, not validated alpha.
- The app cannot make a stale recommendation current; refresh and causal
  server-side revalidation are required.
