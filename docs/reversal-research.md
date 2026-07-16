# Selection-aware reversal research

This protocol turns the original five-day reversal rank into a reproducible
hypothesis family. It does not turn an in-sample rank into a recommendation.
Every output remains subject to the campaign disclaimer, point-in-time data,
independent confirmation, final holdout, and paper-forward requirements.

## Declared rule family

The study declares every combination before looking at performance:

```text
portfolio breadth K:  3, 5, 10, 20, 50
signal:               raw, sector neutral, market/sector residual
side:                 LONG, SHORT
holding period:       five sessions
execution:            next eligible close; first earned return is later still
total rule trials:    5 × 3 × 2 = 30
```

Long and short portfolios are evaluated separately. This matters economically
and statistically: short borrow, squeeze risk, and gap risk are not symmetric
with long exposure. Global one-sided FDR still sees all 30 declarations, while
DSR and CPCV/PBO use the 15 economically comparable alternatives on each side.

The raw signal is the negative trailing five-session return. The sector-neutral
signal subtracts the contemporaneous equal-weight sector move, excluding the
stock itself. The residual signal is

```text
residual_5d = stock_5d - beta_market × market_5d
                         - beta_sector × leave-one-out-sector_5d
score       = -residual_5d / (sqrt(5) × residual_volatility_20d).
```

The two betas are rolling two-factor estimates from observations ending at
`t-1`. Residual volatility is also lagged. Mutating any observation after a
decision date therefore cannot change that date's signal.

## Validation and interpretation

Each configuration receives baseline transaction and short-borrow costs before
inference. Date-level portfolio returns, not stock-date rows, are the observations.
The v3 diagnostic applies:

- minimum sample and directed positive mean;
- one-sided HAC hurdle and global BH FDR;
- side-specific Deflated Sharpe Ratio;
- 0.5x/1x/2x/4x cost sensitivity and a break-even cost multiplier;
- six-group/two-test-group CPCV with purge and embargo, selecting among the
  alternatives on that side;
- a fixed-rule expanding walk-forward report after the first five years;
- rolling and fixed five-year stability/decay classification.

`passes_rule_validation` is deliberately not `WORKS`. The walk-forward check
evaluates each fixed declaration; CPCV/PBO is the selection-aware check across
the declared family. Promotion still needs point-in-time constituents,
timestamped execution/event/borrow inputs, independent confirmation, and the
untouched holdout.

## Causal auction timing

Same-close candidate deletion is prohibited. A valid same-day closing-auction
workflow freezes a `DecisionSnapshot` before the cutoff, for example at 15:45 ET:

1. Signal, quote, event metadata, decision, cutoff, and auction each carry a
   timezone-aware availability timestamp.
2. The take/skip decision and a `0.25 × ATR` LOC limit are frozen before cutoff.
3. The later auction price may fill or miss that frozen order; it cannot change
   candidate membership.
4. If historical 15:45 quotes and event timestamps are unavailable, this path is
   `DATA_UNAVAILABLE`; a final daily close is not substituted.

The implementation also provides explicit gap fraction, direction-aware
pre-entry reversal in ATR units, event proximity, and correlation/sector/factor
trade-similarity variables. A 2-ATR stop still requires split-consistent raw
OHLC and an explicit gap-through/intraday ordering policy before it can be
claimed as tested.

## Rule → ranker → meta-model

The optional ML dataset keeps the economic reversal rule as its anchor. It adds
causal 1/2/3/5/10/20-session returns, raw/sector/residual reversal, lagged betas
and volatility, ATR, abnormal volume, overnight/intraday decomposition,
sector-relative return, liquidity, volatility expansion, and market regime.
Features not supported by the snapshot—historical earnings proximity and 15:45
partial reversal—are reported unavailable rather than reconstructed.

The label is the five-session return beginning after the next-close fill, less
lagged market/sector exposures, baseline costs, and short borrow where applicable.
Separate long and short models include:

- ridge and elastic-net ranking baselines;
- XGBoost LambdaMART with each date as one query group;
- an XGBoost take/skip meta-labeler on extreme raw-reversal candidates.

Purged chronological folds ensure training labels end before a test interval.
Every model declaration has a stable hash and is atomically claimed in the
shared SQLite experiment ledger, so two workers cannot repeat the same trial.
CUDA trial IDs map deterministically across configured devices `(0, 1)`; the
cards remain independent memory spaces.

## Current pre-holdout diagnostic

Campaign `full-stooq-literature-v2-20260715-001`, study `v3`, used observations
strictly before 2023-07-14 and did not evaluate the final holdout.

| Result | Evidence |
|---|---:|
| Declared rules | 30 |
| Discovery survivors | 15, all LONG |
| Rule-validation survivors | 11, all LONG |
| LONG family CPCV/PBO | 6.7% |
| SHORT family CPCV/PBO | 53.3% |
| Best stable full-sample Sharpe | sector-neutral LONG, K=20: 1.506 |
| Its mean net portfolio return | 13.51 bp per active day |
| Its HAC t / forward-year positive fraction | 10.99 / 96.5% |
| Its recent/prior effect ratio | 65.2% (`STABLE`) |
| Its maximum drawdown | -55.5% |

The highest full-history Sharpe was sector-neutral `K=10` at 1.558, but its
recent effect was only 49.6% of its prior median, so it is classified `DECAYING`
and fails rule validation. All 15 short configurations had negative net means;
the least negative was sector-neutral `K=3` at -4.42 bp per active day. The data
therefore support only a long-side hypothesis, not a symmetric long/short rule.

This is not promotion evidence. The snapshot applies current S&P 500 membership
backward, excludes delisted former members, and lacks historical 15:45 quotes,
timestamped earnings metadata, and contemporaneous borrow availability. It is
watermarked `SURVIVORSHIP_BIASED`, its drawdowns are severe, no ML study was run
in this artifact, and the holdout remains untouched. The numerical result is a
candidate for a new point-in-time campaign—not a list of stocks to trade.

The statistical safeguards follow Bailey and López de Prado (2014) for DSR and
Bailey et al. (2017) for PBO; full citations are in
[REFERENCES.md](../REFERENCES.md).
