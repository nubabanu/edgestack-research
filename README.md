# EdgeStack

> **Research and education only. EdgeStack does not provide financial advice or
> guaranteed returns.** See the complete mandatory disclosure below.

EdgeStack is a reproducible statistical-edge discovery, validation, verdict,
stacking, timing, and paper-alert system for US equities and ETFs. It is designed
to report failed hypotheses as prominently as successful ones and to halt when a
frozen acceptance gate fails.

## Repository map

```text
edgestack/
  data/          providers, immutable cache, QA, calendars, universe
  features/      causal session, calendar, and cross-sectional features
  hypotheses/    versioned grammar and matched placebo controls
  stats/         HAC, bootstrap, FDR, DSR, SPA/Reality Check
  validation/    replication, walk-forward, CPCV/PBO, decay
  backtest/      vectorized sweep, costs, metrics, independent confirmation
  evaluation/    verdict policy and exhaustive report
  scoring/       empirical-Bayes shrinkage and correlation-aware stack
  entrytiming/   governed indicators, timers, regimes, stops, interactions
  live/          paper scanner, monitor, SQLite outbox/ledger, scheduling
  oil/           long-only oil research, paper decisions, risk lanes, ledger
  report/        daily rankings and output formats
  reversal/      top-K rule grid, residual features, rankers, purged studies
  storage/       immutable artifacts and SQLite campaign catalog
  pipeline/      hard gates and single-use holdout ceremony
  v2/            loss-aware monthly/yearly research and entitled-data gates
  config.py      strict typed YAML configuration
  models.py      immutable public domain contracts
  cli.py         command-line entry point
configs/         smoke and full frozen profiles
tests/           deterministic unit, integration, causal, and external tests
android/         native Kotlin/Jetpack Compose paper companion
```

## Quick start

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,live,confirm,ml]"
.venv\Scripts\edgestack --help
```

For the exact lock instead of an editable pip solve:

```powershell
uv sync --all-extras --frozen
uv run edgestack --help
```

The no-key profile tries Stooq first and uses Yahoo only as a whole-series
fallback. Set `TIINGO_API_KEY`, `FINNHUB_API_KEY`, or other supported secrets in
the environment; never put tokens in YAML.

When Stooq's interactive CSV endpoint requires browser verification, an official
bulk ASCII archive can be used without weakening the reconciliation gate. Put the
archive outside Git, configure both `data.providers.stooq_bulk_archive` and its
SHA-256, and use that same config for every phase. The adapter reads only the
requested ZIP member, persists those exact member bytes in the raw content store,
and rejects a changed archive. `configs/full-stooq-bulk.yaml` pins the handoff
archive `d_us_txt.zip`; its operator-attested origin remains visibly warned because
Stooq does not publish a cryptographic signature.

```powershell
edgestack ingest --config configs/full-stooq-bulk.yaml --campaign-id <id> --as-of YYYY-MM-DD
```

Stooq bulk OHLCV has no dividend/split ledger. If its historical adjustment basis
conflicts with Yahoo adjusted levels, the separately preregistered
`configs/full-stooq-action-stratified.yaml` protocol uses all available fields
without blending provider prices: it compares Stooq and Yahoo raw close returns on
non-action sessions over the fixed 20-year window, then checks Yahoo's explicit
dividend/split ledger against Yahoo adjusted returns on action sessions. Both
strata retain the same 0.5% tolerance and 99% requirement. Every artifact is
marked `SINGLE_SOURCE_ACTIONS`; this is not independent confirmation of dividends.

The gated campaign flow is:

```text
edgestack ingest --config configs/full.yaml --campaign-id <id> --as-of YYYY-MM-DD
edgestack replicate --config configs/full.yaml --campaign <id>
edgestack discover --config configs/full.yaml --campaign <id>
edgestack validate --config configs/full.yaml --campaign <id>
edgestack report --config configs/full.yaml --campaign <id> --provisional
edgestack score --config configs/full.yaml --campaign <id> --freeze
edgestack report --config configs/full.yaml --campaign <id> --finalize-holdout
edgestack live --config configs/full.yaml --campaign <id>
```

Commands refuse to run when a predecessor gate is absent or failed. Network
campaign tests are marked `external`/`campaign`; the ordinary test suite is
deterministic and does not depend on provider availability. A failed or blocked
gate exits nonzero after persisting its evidence; it never changes a frozen
threshold to make the campaign pass. The smoke profile is an engineering fixture
and is permanently marked non-promotable.

The original `FROZEN_V1` campaigns retain the mandatory all-six empirical
replication gate. `configs/full-stooq-literature-v2.yaml` is a separately
hashed, explicitly post-replication/pre-discovery protocol revision. It keeps
every replication miss visible but distinguishes an executed empirical miss
from an engine failure; discovery then requires the Chordia–Goyal–Saretto
3.8/3.4 hurdles, dependence-preserving Romano–Wolf stepdown evidence,
studentized stationary-bootstrap Sharpe evidence, and capital-scaled cost
curves. It does not rewrite or promote the earlier 5-of-6 campaign.

The exact resolved configuration is part of a campaign identity, so pass the same
`--config` file to every command. The final holdout command is intentionally
one-use: inspect the provisional report before invoking it.

### Oil paper-decision system

The isolated oil v1 subsystem is long-only and paper-only. It uses USO as the
research outcome; USO is not equivalent to eToro's rolling WTI CFD. Its fixed
counterfactual lanes risk 0.5%, 1%, 2%, 5%, and at most 10% of each lane's
current paper equity. The 10% lane is permanently marked
`HIGH_RISK_NON_PROMOTABLE`; leverage changes required paper margin, never the
lane's planned-loss budget.

```powershell
edgestack oil-research --config configs/oil-paper-v1.yaml
edgestack oil-context --spread-bps 8 --overnight-fee-usd-per-unit 0.01 --event-risk NORMAL --expires 2026-07-20T17:00:00-04:00
edgestack oil-decision --paper-equity 100000 --as-of 2026-07-20
edgestack oil-scorecard --campaign oil-paper-v1-20260718-001
```

Missing or expired operator context fails closed. Decisions are only
`NO_TRADE`, `WATCH`, or `PAPER_LONG`; every decision and proxy lifecycle event
is appended to a local SQLite ledger. `oil-schedule` installs the fixed ET
pre-open, 09:50, post-EIA Wednesday, and 16:30 refreshes. No broker credential,
order model, or order endpoint exists in this subsystem or its Android screen.

### Versioned rigor upgrades (opt-in, never retroactive)

`configs/full-literature-v2.yaml` is the recommended template for NEW
campaigns. It binds four opt-in upgrades; existing frozen campaigns and sealed
holdouts are untouched because every upgrade is a new version literal that old
configs never select:

- `holdout_gate.evaluator_version: CI_V2` — the final holdout gate requires
  the stationary-bootstrap CI **lower bound** of every edge and the composite
  to be strictly positive (the original `SIGN_V1` gate only checked the sign
  of the mean). The version is bound into the freeze manifest at score time,
  and `CI_V2` results carry report-only trend/volatility regime
  stratification of the holdout streams.
- `stats.survivor_causality_checks` (default on) — every discovery survivor
  must reproduce its signal exactly when all future sessions are truncated
  away (leak detection) and may not IMPROVE its HAC t under an extra
  execution lag (timing-artifact detection).
- `grid.extended_families` — declares quarter-end/month-end flow windows,
  Amihud illiquidity, MAX-lottery, the overnight/intraday clientele gap, and
  ETF-vs-market relative reversal through the same placebo/DSR/SPA/
  Romano–Wolf gauntlet, honestly expanding the declared trial count.
- `costs.spread_source: MEASURED_HL_FLOOR_V2` (and the same knob on the
  reversal study) — per-name monthly Corwin–Schultz/Abdi–Ranaldo spread
  estimates, floored at the assumed baseline so measurement can only make
  results harder to pass.

Report-only diagnostics that never touch sealed evidence:

```powershell
edgestack holdout-diagnostic --campaign <id>        # would the sealed result pass CI_V2?
edgestack universe-pit-audit --config configs/full.yaml   # delisted-name price coverage
edgestack universe-bias-delta --campaign <id>       # measured survivorship inflation
python -m edgestack.edges.reversal_edge cpcv-diagnostic   # selected-rule PBO in its 30-rule grid
```

With `data.universe_pit: true`, ingest reconstructs `PIT_APPROXIMATION`
membership from the Wikipedia change log and pulls delisted-name history from
the hash-pinned Stooq bulk archive; names without recoverable history become
reported coverage gaps instead of campaign failures, and the
`SURVIVORSHIP_BIASED` watermark is retained (the approximation is measured by
`universe-bias-delta`, not waved away). Ingest QA additionally cross-checks
Yahoo's corporate-action ledger against the second provider's closes on action
sessions: consistent evidence upgrades the watermark to
`ACTIONS_CROSS_CHECKED`, contradictions quarantine the symbol, and
indistinguishable events keep `SINGLE_SOURCE_ACTIONS`.

### Instrument advisor (diagnostic)

`edgestack advise --symbol GLD` builds a per-instrument timing report from
daily bars (free-chain fetch, or `--bars` for an offline parquet):

- **Tailwinds and headwinds** across week (weekday), month (turn-of-month,
  month-end, quarter-end), year (month-of-year), event (holiday/opex), and
  instrument state (trend vs MA200, 12-1 momentum, prior-week reversal, vol
  tercile) — every condition reporting its dark side: behavior in the
  opposite trend regime, worst session, and in-condition drawdown, so a
  positive edge's losses and a negative edge's wins always reach the rating.
- **Combinations with multiplicity control**: cross-kind pairs are evaluated
  as gating candidates, every pair counts toward one Bonferroni family, and a
  pair is only an `INCREMENTAL_CANDIDATE` when the joint mean beats BOTH
  components (pairwise ablation) with enough joint observations; mask overlap
  exposes correlated-voter redundancy.
- **Best/worst buy-sell windows** for the week, month, and year, plus an
  **alignment scan** of upcoming sessions ("all stars aligned" and the worst
  sessions). `--buy-date` rates one intended session with active tailwinds/
  headwinds, an overall rating, and an if-you-trade-anyway caution list.
- **Hard honesty limits**: daily bars support only the opening/closing
  auctions as execution anchors (no intra-hour claims); news is
  `DATA_UNAVAILABLE` (no licensed feed); the whole report is stamped
  `DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER` — the campaign gauntlet
  remains the only promotion path.

The statistically proper combination path also exists inside the gauntlet:
with `grid.extended_families: true`, campaigns declare calendar-gated
momentum/reversal candidates (e.g. "momentum only when the first earned
session is a Friday/turn-of-month/month-end/quarter-end session") as sixteen
additional preregistered trials with their own placebo controls.

### Loss-aware Research V2

`loss-aware-v2` is a new paper-only namespace. It never reads, modifies, or
reopens a V1 holdout. Its historical monthly (21-session) and yearly
(252-session) studies are diagnostics; promotion requires observations generated
after a model definition is frozen. Create the free-only declaration with:

```powershell
edgestack loss-aware-v2 --campaign-id forward-001
```

The declared grid contains all 900 combinations of two horizons, five compact
signal families, long-only/short-only/market-neutral baskets, ten event-veto
settings, and 1.0×/1.5×/2.0× gross leverage. Leverage is paper-only, financing is
daily SOFR + 3%, and a path reaching non-positive equity is rejected. Default
Sniper ranking is lexicographic loss-first among statistical/OOS passes:
95% expected shortfall, loss probability, adverse excursion, loss-streak risk,
then net return. Planned stop risk is shown separately from realized gap/slippage;
a stop is not a guaranteed fill price.

Free data deliberately leave `PIT_MEMBERSHIP`, `ESTIMATE_VINTAGES`, and
`AUCTION_EXECUTION` as `DATA_UNAVAILABLE`. Wikipedia history is labeled
`PIT_APPROXIMATION`; SEC acceptance timestamps provide event diagnostics but do
not substitute for historical consensus vintages. Licensed CSV/Parquet imports
must match a declared SHA-256 and retain permanent IDs, ticker validity,
`event_time`, `available_at`, revision, source, fetch time, and per-record content
hashes. Auction validation requires NBBO, trades, imbalance messages, and official
prints for every finalist.

The WAL ledger atomically records every candidate and skip. Apply a recorded mark
or replay the scorecard without fetching/backfilling history:

```powershell
edgestack live --campaign loss-aware-v2 --once `
  --v2-database artifacts/loss-aware-v2/forward.sqlite --marks marks.json
edgestack paper-scorecard --database artifacts/loss-aware-v2/forward.sqlite
```

### Selection-aware reversal study

The opt-in reversal protocol evaluates the portfolio breadth actually intended
for trading instead of treating a 50-name backtest as evidence for five names. It
predeclares `K = 3, 5, 10, 20, 50`, raw/sector-neutral/market-sector-residual
signals, and long/short sides as 30 distinct trials. It uses one-sided directed
FDR, side-specific DSR and CPCV/PBO, expanding walk-forward tests, five-year decay
analysis, baseline costs, and next-eligible-close execution.

```powershell
edgestack reversal-study `
  --config configs/reversal-study.yaml `
  --campaign <id>

# Optional purged ridge, elastic-net, LambdaMART, and meta-label diagnostics.
edgestack reversal-study `
  --config configs/reversal-study.yaml `
  --campaign <new-id> --run-ml --gpu
```

GPU trials are assigned deterministically to independent devices; two 48-GB cards
are not represented as one 96-GB pool. If CUDA is unavailable, omit `--gpu`.
Historical 15:45 quotes, point-in-time constituents, timestamped earnings, and
borrow records are never inferred from daily closing bars. Without them, the
study remains a visibly survivorship-biased, non-promotable diagnostic even when
its numerical rule tests pass. See
[selection-aware reversal research](docs/reversal-research.md).

## Data keys and paper notifications

No key is required for the selected Stooq-to-Yahoo whole-series fallback. For the
credentialed primary, set `TIINGO_API_KEY` in the process environment. Fresh-quote
adapters are available for Tiingo and Yahoo; secrets are read from environment
variables and must not be committed to YAML or artifacts.

Console delivery is the safe default. The transport classes in
`edgestack.live.notify` also provide generic webhook, Telegram, and SMTP channels.
Construct them from environment-backed secrets, map them by the names in
`live.channels`, and pass that map to the transactional outbox dispatcher. Every
delivery carries a stable event/idempotency ID. SMTP, Telegram, and webhook
delivery is at-least-once and may duplicate after an ambiguous network failure.

Run the restart/deduplication acceptance fixture before enabling any external
channel:

```powershell
edgestack live-demo --database artifacts/live_demo.sqlite
```

The campaign `live` command remains unavailable until a full empirical campaign,
real independent confirmation, freeze, and atomic holdout evaluation have all
promoted a non-empty model. There is no broker adapter or real-order path.

Finalist confirmation runs a real Zipline 3.1.1 `TradingAlgorithm` against an
in-memory asset database and the canonical adjusted OHLCV matrices. Daily data
can confirm close fills for close-to-close and overnight conventions. A
next-open intraday finalist is deliberately unable to pass timestamp agreement
without independent minute/auction data; importing Zipline alone never counts as
confirmation.

## Android paper companion

The `android/` project is a native Jetpack Compose application for reviewing a
promoted paper basket, causal entry/exit instructions, sealed holdout evidence,
evidence-aware week/month/year availability, a loss-first Sniper no-trade gate,
and the immutable audit trail. The
research engine remains on Python: NumPy,
PyArrow, DuckDB, Zipline, campaign data, and holdout access are deliberately not
embedded in an Android process. The mobile API is read-only and defines no broker
or order endpoint.

Start a local demonstration server:

```powershell
edgestack mobile-api --demo
```

For sealed campaign evidence, create a bearer token in the environment and bind
to a trusted interface. Use TLS before exposing the service beyond emulator or
localhost development.

```powershell
$env:EDGESTACK_MOBILE_TOKEN = '<at-least-24-random-characters>'
edgestack mobile-api --host 0.0.0.0 --campaign <promoted-campaign-id>
```

Open `android/` in Android Studio, or build with `android/gradlew.bat
assembleDebug`. The emulator reaches the host API at `http://10.0.2.2:8765`.
Cleartext networking is rejected for every non-local host, tokens are held only
in process memory, and a failed refresh can show only a visibly identified
sealed cache or packaged demo snapshot. See [Android companion](docs/android.md).

## Interpreting results

- `WORKS` means every frozen statistical, OOS, cost, decay, confirmation, and
  holdout requirement passed. It does not predict future profit.
- `WEAK` covers low power, cost damage, instability, or inconclusive holdout.
- `DEAD` was historically detectable but failed the recent-window decay rule.
- `FALSE_POSITIVE` failed multiple-testing/deflated-Sharpe controls or is a placebo.
- `DATA_UNAVAILABLE` and `INVALID` are execution statuses, not fabricated verdicts.

Free current-constituent data are survivorship-biased. EdgeStack visibly stamps
that limitation on every affected report and paper alert. PEAD, true historical
pre-FOMC intraday replication, session VWAP, and limit-fill efficacy remain
disabled unless suitable timestamped datasets are configured.

Literature suggests many calendar effects will be tiny or dead after costs;
short-term reversal is especially cost-sensitive, pre-FOMC drift has decayed,
and most standalone technical overlays should remain disabled. These are
expectations, never substituted for the campaign's actual frozen evidence.

See [architecture](docs/architecture.md), [methodology](docs/methodology.md),
[selection-aware reversal research](docs/reversal-research.md),
[report interpretation](docs/report-interpretation.md), and
[references](REFERENCES.md) for the design, formulas, frozen gates, limitations,
and primary literature.

## Mandatory disclosure

EdgeStack is for research and educational purposes only. It is NOT financial
advice. No trading edge is guaranteed; documented anomalies decay after
publication (McLean & Pontiff 2016 found returns 58% lower post-publication) and
often disappear after realistic transaction costs. Most classic technical
analysis has no standalone edge on liquid US large caps after costs; timing
overlays here are for execution, timing, and risk management, not alpha
generation. Backtests use free, survivorship-biased data unless stated otherwise
and overstate real performance. Live alerts are computed from free, potentially
15-minute-delayed data and may be stale or wrong; a "recommendation holds"
confirmation is a statistical statement, not a promise. Past performance does
not predict future results. Do not trade real capital based on this output without
independent professional advice and your own due diligence.
