# Reading EdgeStack reports

Start with the campaign state, not the equity curve. A report is interpretable only
alongside its campaign ID, data snapshot, bias tier, and gate history. The complete
research definitions are in [methodology.md](methodology.md), and the supporting
literature is in [REFERENCES.md](../REFERENCES.md).

## A five-step reading order

1. **Check `provisional` and the gate summary.** A provisional result has never seen
   the final holdout. A failed or blocked gate prevents promotion even if a row
   farther down looks attractive.
2. **Check `execution_status`.** `DATA_UNAVAILABLE`, `INVALID`, and `UNDERPOWERED`
   describe whether a declaration could be tested. They are not losing trades.
3. **Read `verdict` and `reasons` together.** The reason list identifies the first
   frozen policy failures; never infer a pass from a blank chart or missing metric.
4. **Inspect net, OOS, decay, placebo, and cost evidence.** Gross mean alone is not a
   promotion metric.
5. **Read limitations last but treat them as binding.** In particular,
   `SURVIVORSHIP_BIASED` means current constituents were projected backward.

## Status and verdict vocabulary

| Field value | Meaning |
|---|---|
| `TESTED` | Enough valid input existed to apply the statistical policy. |
| `UNDERPOWERED` | Fewer than the required independent date observations existed. |
| `DATA_UNAVAILABLE` | Required timestamped data were not configured; no effect was estimated. |
| `INVALID` | The declaration was contradictory, empty, or violated a contract. |
| `WORKS` | Every applicable frozen statistical, OOS, cost, confirmation, decay, and holdout condition passed. |
| `WEAK` | Evidence exists but power, costs, OOS, stability, confirmation, regime, or holdout was insufficient. |
| `DEAD` | Earlier significance followed by the preregistered recent-window collapse. |
| `FALSE_POSITIVE` | The trial failed hard discovery/multiple-testing controls or is a placebo. |

`WORKS` is a historical evidence classification, not a guarantee or instruction to
trade. A provisional `WORKS` can become `WEAK` after the one-time holdout without
any rule having changed.

## Core evidence fields

| Field | How to read it |
|---|---|
| `sample_size` | Independent portfolio dates after filtering, not raw stock rows. |
| `gross_mean` | Directed mean before costs. Useful for diagnosing cost drag only. |
| `net_mean` | Directed mean after the frozen baseline costs; must be positive. |
| `hac_t` / `p_value` | Newey-West mean test with overlap-aware lags. Discovery uses the preregistered directed threshold, not an unadjusted visual rule. |
| `mean_ci` | Stationary-bootstrap interval for the mean; dependence is retained through blocks. |
| `sharpe` | Annualized sample mean-to-volatility ratio. It is unstable in small/non-normal samples. |
| `probabilistic_sharpe` | Asymptotic reliability against a benchmark Sharpe, adjusted for sample size and higher moments. |
| `deflated_sharpe_probability` | Sharpe reliability after accounting for trial multiplicity. It is not the probability the next trade wins. |
| `hit_rate` | Fraction of finite date-level returns strictly above zero. It ignores payoff magnitude. |
| `max_drawdown` | Largest historical peak-to-trough loss under the reported convention. It is sample-dependent. |
| `turnover` / `exposure` | Inputs to cost and matched-control diagnostics. |
| `oos_t` | HAC statistic on the stitched walk-forward net OOS stream. |
| `oos_positive_fraction` | Fraction of eligible test years with favorable directed mean; it must exceed one half. |
| `stability_score` | Same-sign fraction across the frozen fixed/rolling windows. |
| `pbo` | Estimated probability of selecting an in-sample winner that ranks poorly OOS. Blank may mean undefined, never zero. |
| `confirmation_difference_bps` | Difference from the independent event-driven implementation in basis points per trade. |
| `holdout_mean` | Net directed mean on the one-time three-year holdout. It appears only in final evidence. |

Confidence intervals and p-values answer different questions. An interval describes
sampling dispersion under the chosen bootstrap; a multiple-testing-adjusted gate
controls selection risk across the declared family. Neither measures data quality,
survivorship bias, or implementation risk.

## Filter funnel and trial controls

The funnel should reconcile the count of all declared hypotheses through valid
data, minimum sample, positive net mean, hard HAC threshold, BH FDR, DSR, SPA,
walk-forward, stability/PBO, cost, and confirmation stages. Rows do not disappear:
failed and unavailable declarations remain in the complete CSV.

Compare real trials with both control families. Shuffled-date controls test whether
the calendar alignment itself matters. Turnover/exposure-matched random signals
test whether apparent performance is a mechanical consequence of trading
intensity. Any placebo reported as final `WORKS` is a campaign-level failure, not an
interesting exception.

## Cost panels

Use 1× as the registered baseline. The 0.5× case is optimistic and the 2×/4× cases
are stress tests. Break-even cost is the additional friction that would reduce the
estimated mean to zero; it is not a prediction of obtainable execution. Large gross
returns with small break-even cost indicate a fragile, high-turnover effect.

Short rows include the 0.3% annual easy-to-borrow research assumption. In the paper
assistant `borrow_verified=false` means exactly that availability, locate fees, and
recall risk were not verified.

## Equity curves and drawdowns

Equity curves are descriptive. Verify whether a chart is gross or net, in-sample,
walk-forward OOS, or final holdout. Overlapping holding periods can make a curve look
smoother while reducing independent information; HAC and purging address this in
inference. Do not compare curves with different exposure, leverage, or missing-date
policies without normalization.

## Decay and regime panels

`STABLE` requires both broad sign consistency and adequate recent magnitude.
`DECAYING` is not deployable merely because its full-history mean is positive.
`DEAD` requires the exact earlier-significance/latest-two-windows rule. A
`REGIME_DEPENDENT` effect is usable only if the interaction survived FDR, the active
regime passed its threshold, and that same regime is currently active.

## Stack and confidence

The stack table shows raw and empirical-Bayes-shrunk means, correlation cluster,
and final frozen weight. Equal cluster weighting prevents several variants of one
effect from outvoting independent effects. Shrunk means support estimation; they do
not rescue edges that failed promotion gates.

The daily `confidence` score is ordinal:

```text
round(100 * composite DSR reliability
          * direction-specific forecast-magnitude percentile)
```

A score of 80 ranks above 65 under the same frozen model. It does **not** imply an
80% success probability and is not comparable across independently defined model
versions without calibration evidence.

## Overlay table

An enabled overlay must improve incremental *net* return through the complete
gauntlet and have a neighboring-parameter plateau. Read disabled rows too: an
overlay that looks best at one isolated RSI threshold or stop multiple is rejected
as brittle. No enabled overlay is a normal successful outcome.

Exploratory intraday/VWAP/limit-fill results are labeled low-power and cannot modify
the daily model. They should never be presented as validated execution alpha.

## Provisional versus final reports

The provisional report is created before holdout access and may be regenerated from
pre-holdout artifacts. `--finalize-holdout` consumes a one-use authorization and
evaluates the frozen individual edges and frozen composite together. Later final
reports replay the sealed result.

The final three-year interval may include zero without violating the protocol; the
frozen requirement is positive net holdout mean, because three years may not support
a meaningful significance test. If an edge or enabled overlay fails, it is not
removed and the old composite is not reweighted. Any redesign becomes a distinct
paper-only version awaiting new OOS data.

## Daily paper-assistant output

Actionable LONG and SHORT tables contain at most five names per direction with
confidence at least 60. `SKIP` rows live in a separate audit table so absence is
explainable. Read the complete entry plan: order type, trigger/current value,
earliest execution, limit policy, expiry action, validity window, stop, size,
timestamps, and rationale.

`ACT_NOW`, `WAIT_UNTIL`, `WAIT_FOR_TRIGGER`, and `SKIP` are paper timing states. An
alert transition is not a broker order. Stable recommendation/event/revision IDs
allow external duplicate deliveries to be deduplicated after retries or restart.

## Empty results are complete results

A report with zero survivors, no stack, no enabled overlays, or no actionable daily
names is not incomplete. Confirm that the funnel reconciles and the diagnostic
artifacts exist; then report the empty conclusion without lowering thresholds.

