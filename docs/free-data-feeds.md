# Free data feeds — earnings, point-in-time universe, intraday

Three zero-cost feeds added 2026-07-19, each honest about exactly what it
can and cannot claim. None require API keys except the optional intraday
collector (free Alpaca account).

## 1. SEC EDGAR earnings (`edgestack/data/edgar_earnings.py`)

`python -m edgestack.data.edgar_earnings` crawls, for every equity in the
sealed campaign panel:

- **Announcement moments** — every Form 8-K whose item list contains 2.02
  (Results of Operations), with the EDGAR acceptance datetime. Official,
  second-precision, point-in-time by construction.
- **Reported quarterly EPS** — the XBRL `companyconcept` series
  (`EarningsPerShareDiluted`, basic fallback), keeping only frame-tagged
  (`CYyyyyQq`) entries, EDGAR's deduplicated canonical quarters.

Exact response bytes go through the content-addressed raw store
(`artifacts/earnings/raw/`); processed tables land in
`artifacts/earnings/{announcements,eps}.parquet` with `manifest.json`
coverage statistics. Watermarks carried by every downstream study:
`PRESS_RELEASE_EQUALS_XBRL_APPROXIMATION` (the drift study assumes the
press-release EPS equals the later XBRL figure),
`XBRL_COVERAGE_EFFECTIVELY_2009_PLUS`, and
`ANNOUNCEMENT_TIMESTAMP_IS_EDGAR_ACCEPTANCE`. SEC fair-use is respected
via an identified User-Agent (`EDGESTACK_EDGAR_USER_AGENT` to override)
and a global 8 req/s ceiling.

This feed lifted the PEAD block (`configs/pead-study-v1.yaml`) and feeds
the preregistered `configs/pead-study-v2.yaml` family.

## 2. Point-in-time membership snapshots (nightly job)

`run_post_close` now writes one immutable
`artifacts/universe/membership-<session>.json` per session (current S&P 500
membership, watermarked `POINT_IN_TIME_FROM_CAPTURE_DATE_ONLY`). This does
not repair the historical panel's survivorship bias — nothing free can —
but from the first snapshot onward the forward universe is bias-free by
construction, which is exactly what the forward holdout windows need.

## 3. Forward intraday collector (`edgestack/data/intraday_collector.py`)

Free historical intraday data effectively does not exist (IEX Cloud is
dead, Polygon's free tier is gone, Alpha Vantage's is 25 calls/day), so
the collector builds the archive forward. With free Alpaca keys
(`setx ALPACA_KEY_ID ...` / `setx ALPACA_SECRET_KEY ...`) the nightly job
captures one-minute IEX-feed bars for the calendar symbols into
`artifacts/intraday/<SYMBOL>/<session>.parquet`; without keys it reports
`DATA_UNAVAILABLE` and never fails the job. Every file is stamped
`IEX_FEED_PARTIAL_VOLUME` — the IEX feed is a small slice of consolidated
volume and must never be mistaken for primary-exchange auction data. After
a year of capture, execution-window questions ("is 15:45 really better
than 10:00?") become testable on self-provenanced data.
