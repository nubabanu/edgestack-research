# Frozen research methodology

This document is the human-readable research protocol. Configuration and immutable
campaign manifests are the machine-readable authority. If prose, code, and a
frozen manifest disagree, the campaign must stop and record the discrepancy; it
must not choose whichever definition produces the better result.

The statistical methods follow the literature collected in
[REFERENCES.md](../REFERENCES.md), especially Newey and West (1987), Politis and
Romano (1994), White (2000), Hansen (2005), Bailey and López de Prado (2014), and
Bailey et al. (2017).

## 1. Scope and unit of inference

EdgeStack asks whether a predeclared, directed return stream remains positive after
realistic baseline costs and survives multiple-testing, out-of-sample, stability,
and independent-confirmation gates. It does not optimize a trading strategy until a
backtest looks attractive.

For a cross-sectional hypothesis, stock-level returns on a date are first combined
into one portfolio return. The date, not the stock-date pair, is the independent
unit used for the sample count and inference. This prevents a broad market move
across 500 correlated stocks from masquerading as 500 observations.

All return signs are expressed in the declared direction. A short observation is
the negative of the underlying long return before costs, with borrow charged
separately. “Positive” therefore always means favorable to the declared trade.

## 2. Preregistration and campaign identity

Before campaign data are downloaded, the following are resolved and hashed:

- sample endpoints and the final three-calendar-year holdout;
- universe definition and the exact nine ETFs (`SPY`, `QQQ`, `IWM`, `XLK`, `XLF`,
  `XLE`, `XLV`, `XLY`, and `XLI`);
- features, predicates, interactions, directions, sessions, and holding periods;
- costs, thresholds, bootstrap counts, validation geometry, and random seed;
- source-tree, dependency-lock, and resolved-configuration identities.

Changing one of these creates a new campaign. A clean empirical miss is not a
reason to amend the active campaign.

## 3. Data, adjustments, and known biases

The no-key research universe is the current S&P 500 snapshot plus the nine ETFs.
Current membership is not a point-in-time constituent history. Every affected row,
ranking, and alert is therefore stamped `SURVIVORSHIP_BIASED`; that label does not
prevent a numerical `WORKS` verdict, but it materially limits interpretation.

Daily provider fallback is whole-series only. Raw OHLCV and adjusted total-return
series are cached separately with their source capabilities. Splits, dividends,
stale prices, zero volume, outliers, missing eligible sessions, and cross-provider
differences are audited without silently rewriting raw evidence. Coverage starts at
verified listing or first observation and ends at delisting or last observation.

Every row carries `event_time` and `available_at`. Features and joins use only data
available at the decision timestamp. PEAD is declared `DATA_UNAVAILABLE` unless
timestamped announcement, consensus, and standardized-unexpected-earnings inputs
are configured. Recent intraday evidence is exploratory and may not change the
frozen daily model or authorize VWAP/limit-fill claims.

## 4. Return and feature definitions

Let `P_t`, `O_t`, and `C_t` denote adjusted price, session open, and session close.
Simple returns are used unless an artifact explicitly says `log`; log overnight and
intraday returns are used for exact additive decompositions:

```text
close-to-close: C_t / C_(t-1) - 1
overnight:      O_t / C_(t-1) - 1
intraday:       C_t / O_t - 1
log identity:   log(O_t/C_(t-1)) + log(C_t/O_t) = log(C_t/C_(t-1))
```

Signals available at a close cannot fill at that close. Research execution selects
the next eligible bar strictly after availability. Auction MOC/LOC instructions are
allowed only when their signal was available before the applicable auction cutoff.

The calendar grammar contains weekday, month, turn-of-month/rest, pre/post holiday,
FOMC day-before/day-of/event-week, option-expiry week, and current sector. Event week
means the ISO week containing the scheduled announcement. `ANY` is an explicit
baseline. Pairwise interactions are generated only for compatible families;
contradictory and empty conjunctions are invalid, not zero-return trials.

Predicates expand over close-to-close holds of 1, 3, 5, and 21 sessions, overnight
and intraday conventions, and long and short directions. The registered separate
cross-sectional families are:

- 12-minus-1 momentum: 252-session lookback, latest 21 sessions skipped, monthly
  rebalance (Jegadeesh and Titman, 1993);
- short-term reversal: five-session lookback and hold (Lehmann, 1990);
- low volatility: trailing 252-session realized volatility (Ang et al., 2006);
- 52-week-high proximity: price divided by the trailing 252-session high (George
  and Hwang, 2004).

Canonical JSON, including every parameter and predicate, is SHA-256 hashed into the
hypothesis ID. Each real hypothesis receives deterministic shuffled-date and
turnover/exposure-matched random-signal controls.

## 5. Costs

Costs are applied during discovery. The baseline has zero commission, a quoted full
spread of 1 bp for ETFs or 3 bps for equities with half charged per fill, and per-side
slippage

```text
min(50, 1 + 10 * sqrt(order_dollars / ADV_dollars)) basis points.
```

Paper shorts pay 0.3% annual easy-to-borrow cost on ACT/365 and always state that
borrow availability is unverified. Model selection adds 1 bp per 100% one-way
turnover. Every finalist reports 0.5×, 1×, 2×, and 4× cost sensitivity and its
break-even cost.

## 6. Discovery inference

At least 100 independent date observations are required. The Newey-West/Bartlett
HAC lag is

```text
min(T - 1, max(holding_period - 1, floor(4 * (T / 100) ** (2 / 9)))).
```

A discovery survivor has directed net mean greater than zero, directed HAC
`t > 3`, global Benjamini-Hochberg FDR at `q = 0.05`, and deflated-Sharpe
probability greater than 0.95. Bonferroni is reported as a conservative reference.
If more than 5% of tested hypotheses survive the `t + FDR` filters, the campaign
stops for a frozen diagnostic audit rather than celebrating an implausibly broad
signal.

Stationary-bootstrap confidence intervals use 2,000 shared deterministic draws for
eligible hypotheses and 10,000 for finalists. Shared paths preserve dependence
between strategies. White Reality Check and Hansen SPA use 10,000 draws. PSR and DSR
adjust for sample size, skewness, kurtosis, and the number/distribution of trials;
they are model-risk diagnostics, not posterior probabilities of profit.

## 7. Validation and overfitting controls

Expanding walk-forward validation begins with at least five training years, then
tests one year and advances one year. The stitched net OOS stream requires HAC
`t > 2`, favorable means in more than half of eligible test years, and stability of
at least 75%.

Combinatorial purged cross-validation uses six chronological groups, two held out,
and a 21-session purge and embargo. Candidate-set probability of backtest
overfitting must be below 20% when it is mathematically defined. A missing PBO due
to an inadequate candidate set is labeled undefined, never imputed as zero.

No placebo can receive final `WORKS`; at most 0.5% may survive provisionally. An
independent Zipline confirmation must agree on trade timestamps and counts and on
net mean within 1 bp per trade. The adapter executes a real Zipline 3.1.1
`TradingAlgorithm` over canonical adjusted OHLCV. Daily bars can establish close
fills for close-to-close and overnight returns; a next-open intraday finalist
fails timestamp agreement unless independent minute/auction data are supplied.

## 8. Decay and regime policy

- `STABLE`: same sign in at least 75% of fixed/rolling periods and the recent effect
  is at least 50% of its earlier median.
- `DECAYING`: the recent effect remains positive but is below 50% of its earlier
  median.
- `DEAD`: an earlier rolling window had `t > 3`, while both latest five-year windows
  have `|t| < 1` and effect no greater than 25% of the earlier significant median.
- `REGIME_DEPENDENT`: an FDR-adjusted interaction has `p < 0.05` and the active
  regime has `t > 2`. It is deployable only while that validated regime is active.

These labels describe evidence evolution. They do not override discovery, cost, or
OOS failures.

## 9. Verdict precedence

`INVALID` and `DATA_UNAVAILABLE` are execution statuses and receive no fabricated
verdict. `UNDERPOWERED` receives `WEAK`. Among tested hypotheses, a nonpositive net
mean, failure of hard `t`, BH, DSR, SPA, or placebo status produces
`FALSE_POSITIVE` first. A historically qualifying effect meeting the frozen death
rule is `DEAD`. Remaining failures of costs, OOS, stability, PBO, confirmation,
deployable decay/regime, or final holdout are `WEAK`. Only an edge passing every
applicable frozen condition is `WORKS`.

## 10. Shrinkage, clustering, and confidence

For edge estimates `mu_i` with sampling variances `s_i²`, empirical-Bayes shrinkage
uses

```text
tau² = max(0, sample_variance(mu) - mean(s_i²))
shrunk_mu_i = mu_i * tau² / (tau² + s_i²).
```

Return streams connected at absolute correlation at least 0.70 form a cluster.
Clusters receive equal weight and members receive equal weight within their
cluster. This prevents a family of near-duplicates from dominating merely through
enumeration. No surviving edges means no composite, which is a valid outcome.

Daily confidence is ordinal, not a calibrated probability:

```text
round(100 * composite_DSR_reliability
          * direction_specific_forecast_magnitude_percentile).
```

It ranks current signals under the frozen model; “80” does not mean an 80% chance
of profit.

## 11. Timing overlays

The predeclared neighborhoods are RSI(2) thresholds 5/10/15, Bollinger `%B`
thresholds 0.1/0.2/0.3, expiries 3/5/7, breakout windows 20/63/252, ATR stop
multipliers 1.5/2.0/2.5, MA 200, and VIX regimes 15/25. An overlay is enabled only
when its incremental *net* return passes the full validation gauntlet and exhibits
a plateau: at least two adjacent settings have the same sign and are within 20% of
the best incremental Sharpe. Disabled overlays remain visible in the report.

## 12. Final holdout ceremony

The holdout is the final three calendar years ending on the last completed NYSE
session. Before access, one manifest freezes provisional edges, costs, weights,
score mapping, overlays, thresholds, data identity, configuration, and hashes. One
atomic job evaluates all individual edges and the composite. Positive holdout net
mean is required; a three-year significance threshold is deliberately not required,
although a confidence interval is reported. Enabled overlays must remain
incrementally nonnegative.

The authorization cannot be consumed twice. After observation, the model is not
rebuilt. If a frozen component loses `WORKS`, its composite is not promoted; a
revision is a new paper-only model that must await genuinely new OOS observations.

## 13. Frozen replication gate

All six checks must pass on clean data before discovery begins:

| Replication | Frozen pass condition |
|---|---|
| Turn of month | TOM-minus-rest mean is positive with one-sided HAC `t > 2`. |
| FOMC proxy | Prior-close to meeting-day close is positive with `t > 2` in 1994–2013. This no-key daily proxy is explicitly labeled as post-announcement contaminated. |
| Session decomposition | For SPY and QQQ, cumulative overnight log return exceeds intraday and supplies at least 75% of total log return. |
| Momentum crash | Momentum spread is positive overall and its rolling 63-session loss in 2009 is at least 20%. |
| Reversal and costs | Gross reversal is positive; net mean is nonpositive or no more than 50% of gross. |
| Monday decay | Pre-1975 Monday-minus-other mean is negative with `t < -2`; post-1990 has `|t| < 1.96` and absolute mean difference below 5 bp/day. |

The FOMC event-week feature means the week containing the scheduled announcement;
it is distinct from the contaminated daily replication proxy.

## 14. Interpreting a stopped or empty campaign

Stopping is a result. A data disagreement, failed replication, excessive survivor
rate, placebo leakage, failed confirmation, or nonpositive holdout must remain in
the artifact trail. Likewise, zero `WORKS` edges, an empty composite, and zero
enabled overlays are acceptable research conclusions. The system is designed to
make “nothing survived” easier to report than to rationalize away.
