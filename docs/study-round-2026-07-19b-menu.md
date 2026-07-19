# Study round 2026-07-19b — the generated hypothesis menu

This round is the "industrialized honest loop": candidate families generated
from the anomaly literature and economic reasoning, filtered for what the
current data tier can test honestly and what the cost model won't kill on
arrival, then preregistered and run once each. The menu below is the full
record — what went in, what stayed out, and why. Verdicts are appended
after the single evaluation of each family.

## Included (preregistered this round)

| Family | Campaign | Core claim | Why it made the cut |
| --- | --- | --- | --- |
| Cross-sectional momentum | `momentum-xs-study-v1` | 12-1 winners keep winning (Jegadeesh-Titman 1993) | The most documented anomaly in finance; never tested cross-sectionally here (the failed trend family was time-series on ETFs); monthly turnover suits the cost model |
| 52-week-high nearness | `high52-study-v1` | Stocks near their 52-week high outperform (George-Hwang 2004) | Anchoring-based, distinct from momentum in the literature; pure price, monthly |
| High-volume return premium | `volshock-study-v1` | Abnormal-volume stocks earn a visibility premium (Gervais-Kaniel-Mingelgrin 2001) | Uses the volume field nothing else exploits; monthly |
| Pre-holiday effect | `preholiday-study-v1` | The session before an exchange holiday is abnormally strong (Ariel 1990; Lakonishok-Smidt 1988) | Calendar-causal, known years ahead; ~9 events/yr keeps turnover survivable on ETFs |

Ten real trials across four families, each family accounted separately with
two placebo controls per trial, LITERATURE_V2 thresholds, forward-only
holdouts 2026-07-19 → 2028-07-19.

## Excluded or deferred (with reasons)

- **Gap-day reversal** — excluded: two auction fills per single-day event on
  single names is the overnight family's cost structure, which already
  failed; the declared cost model kills it before evaluation.
- **Pre-FOMC announcement drift (Lucca-Moench)** — deferred:
  REQUIRES_EVENT_FEED. Needs the historical FOMC announcement calendar;
  no adapter exists yet and hand-typed dates would be fabricated data.
- **Index addition/deletion effects** — deferred: needs point-in-time
  membership-change events. The nightly membership snapshots started
  2026-07-19 are accumulating exactly this; testable in a year or two.
- **January/tax-loss rebound** — excluded: overlaps the validated reversal
  basket and the month-of-year evidence already shrunk to zero on this
  panel; a new campaign would be re-testing a rejected claim.
- **Liquidity/size premium** — excluded: a current-S&P-500 panel is
  range-restricted in both size and liquidity; the design cannot separate
  the premium from the survivorship stamp.
- **Execution-window (intraday) effects** — deferred: the forward intraday
  archive only started capturing 2026-07-19.

## Verdicts

(appended after the single preholdout evaluation of each family)
