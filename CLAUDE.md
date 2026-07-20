# CLAUDE.md — agent onboarding for EdgeStack

EdgeStack is a research-grade statistical-edge discovery engine for US equities/ETFs,
plus a paper-only live alert system and a native Android companion app. It finds,
validates, and monitors calendar/seasonal trading edges under strict multiple-testing
control. **There is no broker or order path anywhere in this repo — everything is
research and paper decisions.** Human-facing docs live in `README.md` and `docs/`;
this file is the fast path for an AI agent.

## The honesty contract (non-negotiable)

These rules are enforced in code and must be preserved by every change:

- **Never fabricate, estimate, or interpolate missing data.** Anything unknowable is
  reported literally as `DATA_UNAVAILABLE` (news, intraday hours, unsigned archives).
- **Failed hypotheses are reported as loudly as successes.** Trend/TSMOM, VIX
  risk-premium, and overnight studies all *failed* their gauntlets and stay in the
  repo as closed negatives. Never quietly delete or soften a negative result.
- **Preregistration before evaluation.** Study families are declared (trial count,
  costs, thresholds) in `configs/*.yaml` before any returns are computed. Do not
  add trials to a family after seeing results.
- **The holdout is single-use.** `report --finalize-holdout` may run once per
  campaign. Never re-run, peek at, or "just check" holdout data. Gate state lives in
  the SQLite catalog and `Catalog.require_passed` refuses to proceed on failure —
  never bypass or weaken a gate to make something run.
- **Versioned rigor upgrades are never retroactive** (see README "Versioned rigor
  upgrades"): a new protocol version cannot rewrite or promote an old campaign.
- **Diagnostics are labeled as diagnostics.** Advisor/timing output is stamped
  `DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER`; survivorship-biased data is
  watermarked (`SURVIVORSHIP_BIASED`, `PIT_APPROXIMATION`). Keep every such stamp.

Current validated-edge status: **SPY turn-of-month is the only gauntlet-passed,
activated edge** (`configs/spy-tom-edge-v1.yaml`). Everything else is diagnostic
or closed.

## Repo map

- `edgestack/` — Python package (CLI entry: `edgestack.cli:app`)
  - `data/` — providers (Stooq→Yahoo free chain, Tiingo/Finnhub keyed), immutable
    cache, QA, calendars, `universe.py` (Wikipedia S&P 500 + 9 liquid ETFs)
  - `edges/` — edge studies: `turn_of_month.py` (validated), `reversal_edge.py`
    (validated basket), `_study_common.py` (shared gauntlet engine), and the
    closed failures: `trend_study.py`, `vix_study.py`, `overnight_study.py`,
    `seasonal_study.py`, `lowvol_study.py`, `pairs_study.py`, `pead_study.py`,
    `momentum_xs_study.py`, `high52_study.py`, `volshock_study.py`,
    `preholiday_study.py`, `seasonal_intl_study.py`, `tom_intl_study.py`
    (see `docs/study-round-2026-07-19.md`, `docs/study-round-2026-07-19b-menu.md`,
    and `docs/study-round-2026-07-19c-intl.md`)
  - `stats/`, `validation/`, `backtest/`, `evaluation/`, `scoring/`, `pipeline/` —
    the gauntlet: HAC t-tests, stationary bootstrap, FDR, deflated Sharpe, SPA,
    CPCV/PBO, walk-forward, decay, holdout ceremony
  - `entrytiming/`, `reversal/`, `v2/`, `oil/` — specialized research namespaces
    (`oil/` = isolated long-only paper-decision system, `NO_TRADE/WATCH/PAPER_LONG`)
  - `advisor.py` — per-instrument diagnostic timing engine (any free-form symbol)
  - `live/` — nightly `daily_job.py:run_post_close`, forward ledger, scheduler,
    Telegram/webhook/SMTP notify
  - `mobile/` — read-only FastAPI service + snapshot codec for the Android app
- `android/` — Jetpack Compose companion app (`com.edgestack.mobile`)
- `configs/` — preregistered campaign/study YAMLs (config hash = campaign identity)
- `scripts/` — Windows PowerShell ops: `post-close-job.ps1` (nightly ~16:35 ET),
  `serve-mobile.ps1`, `install-autostart.ps1`
- `tests/` — pytest suite (deterministic by default; see markers below)
- `docs/` — architecture, methodology, report interpretation, android, research notes
- `artifacts/`, `data/` — runtime outputs and cache; **gitignored**, as are `*.sqlite`

## Dev environment

- **Python 3.12 only** (`requires-python = ">=3.12,<3.13"`). Windows host.
- Setup: `py -3.12 -m venv .venv` then `pip install -e ".[dev,live,confirm,ml]"`,
  or exact lock: `uv sync --all-extras --frozen`.
- Android: Gradle + Java 21 (`cd android && gradle testDebugUnitTest`).
- Data needs no API keys (Stooq first, Yahoo whole-series fallback; SEC EDGAR
  earnings feed via `python -m edgestack.data.edgar_earnings` — see
  `docs/free-data-feeds.md`). Optional env: `TIINGO_API_KEY`,
  `FINNHUB_API_KEY`, `EDGESTACK_TELEGRAM_TOKEN`/`_CHAT`, and
  `ALPACA_KEY_ID`/`ALPACA_SECRET_KEY` (enables forward intraday capture).
  Never put tokens in YAML. `d_us_txt.zip` (537 MB, repo root, hash-pinned) is the
  Stooq bulk archive for full-universe campaigns.

## Common commands

```powershell
python -m pytest tests/ -q            # deterministic suite (what CI runs)
python -m pytest -m "not external"    # explicit offline selection
ruff check edgestack tests            # lint (E,F,I,B,UP,SIM,RUF; line 88)
black edgestack tests                 # format
mypy edgestack                        # strict mode + pydantic plugin
```

Pytest markers: `external` (live providers), `campaign` (long research runs),
`integration`. The default suite is deterministic and provider-independent.

### Agent toolbelt — start here

`edgestack/agenttools.py` is the fast path built for AI agents: every command
prints compact JSON (roughly 10x smaller than the raw reports), never raises
(sections degrade to `NOT_AVAILABLE` with the reason), and preserves every
honesty stamp. Prefer `python -m edgestack.agenttools` over `edgestack agent`
for pure-JSON stdout (the CLI callback prints the disclaimer banner first).

```powershell
python -m edgestack.agenttools describe            # machine-readable command list
python -m edgestack.agenttools overview            # offline system status in one call
python -m edgestack.agenttools advise ACN --buy-date 2026-07-28
python -m edgestack.agenttools compare ACN,CTSH,SPY
python -m edgestack.agenttools calendar CTSH --publish   # feeds the Android app
python -m edgestack.agenttools leverage-check MU --leverage 5   # liquidation math
```

Other key CLI commands (`edgestack ...` or `python -m edgestack.cli ...`):

- `advise --symbol X [--buy-date YYYY-MM-DD]` — diagnostic timing report for any
  ticker (tailwinds/headwinds, alignment scan, buy-date rating)
- `tailwind-calendar --symbol X --output artifacts/advisor/tailwind-calendar-X.json`
  — forward calendar in the exact payload shape the app consumes
- `post-close` — full nightly loop (basket signal, forward ledger, calendars,
  Telegram). Refuses to run unless campaign gates PASS. Idempotent per session.
- `mobile-api` — read-only API for the Android app
- `oil-research` / `oil-context` / `oil-decision` / `oil-scorecard` — oil subsystem
- Campaign flow: `ingest → replicate → discover → validate → report --provisional →
  score --freeze → report --finalize-holdout → live` (each gate refuses to run if a
  predecessor failed; pass the *same* `--config` to every command)

## Live system

`scripts/post-close-job.ps1` (scheduled nightly) runs `post-close`, which writes
`artifacts/advisor/tailwind-calendar-<SYMBOL>.json` for each calendar symbol
(default `SPY,QQQ,GLD,ACN,CTSH` — defined in `edgestack/cli.py` and
`edgestack/live/daily_job.py:run_post_close`). `edgestack/mobile/service.py`
auto-discovers every `tailwind-calendar-*.json` (SPY leads) and serves them to the
Android Timing tab — adding a symbol to the nightly job is all it takes to surface
it in the app. The forward ledger (`artifacts/campaigns/<id>/forward/ledger.sqlite`)
is append-only.

## CI

`.github/workflows/ci.yml`, on pushes to `main` and `agent/**` plus PRs:
Python job (`pip install -e ".[confirm,live]"` then `pytest tests/ -q`) and
Android unit tests (`gradle testDebugUnitTest`, Java 21).

## Read next

| Doc | What it covers |
| --- | --- |
| `README.md` | Full usage: quick start, campaign flow, oil system, V2, Android |
| `docs/architecture.md` | System design and data flow |
| `docs/methodology.md` | The statistical gauntlet in detail |
| `docs/report-interpretation.md` | How to read report/advisor output |
| `docs/android.md` | Companion app build and API contract |
| `docs/reversal-research.md`, `docs/overnight-research.md` | Study write-ups |
| `REFERENCES.md` | Academic sources for each edge family |

## Gotchas

- `artifacts/`, `data/cache|raw|canonical/`, `*.sqlite` are gitignored — never
  commit runtime outputs; regenerate them via CLI commands instead.
- Config file bytes are part of a campaign's identity; editing a YAML mid-campaign
  invalidates it. Create a new versioned config instead.
- Only daily bars exist: no intraday claims are honest. Timing anchors are the two
  auctions (MOC preferred, opening auction second).
- The smoke profile is an engineering fixture and is permanently non-promotable.
- Mandatory disclaimer (README "Mandatory disclosure") applies to anything
  user-facing this repo produces: research/education only, not investment advice.
