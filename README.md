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
  report/        daily rankings and output formats
  storage/       immutable artifacts and SQLite campaign catalog
  pipeline/      hard gates and single-use holdout ceremony
  config.py      strict typed YAML configuration
  models.py      immutable public domain contracts
  cli.py         command-line entry point
configs/         smoke and full frozen profiles
tests/           deterministic unit, integration, causal, and external tests
```

## Quick start

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,live,confirm]"
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

The exact resolved configuration is part of a campaign identity, so pass the same
`--config` file to every command. The final holdout command is intentionally
one-use: inspect the provisional report before invoking it.

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
