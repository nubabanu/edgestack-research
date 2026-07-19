# Study round 2026-07-19 — seasonal, low-vol, pairs, PEAD

Four documented anomaly families were considered in this round. Three were
preregistered (commit `bdd7bff`, before any evaluation ran) and evaluated
once each through the shared gauntlet (`edgestack/edges/_study_common.py`)
on the sealed panel `full-stooq-literature-v2-20260715-001` (daily bars
through 2026-07-14, survivorship-biased and stamped as such). One was
formally blocked. **All three evaluated families FAILED preholdout.** No
holdout was consumed; each forward window (2026-07-19 → 2028-07-19) accrues
untouched in case a future protocol revision re-tests a family as a new
campaign.

## Verdicts

| Family | Campaign | Verdict | One-line reason |
| --- | --- | --- | --- |
| Halloween seasonality (9 ETF trials) | `seasonal-study-v1-20260719-001` | **FAIL** | Family-level tests fired (SPA p=0.0024) but no trial cleared the individual bar; SPY came closest at HAC t=3.73 vs the preregistered 3.8 |
| Low-volatility / low-beta (3 trials) | `lowvol-study-v1-20260719-001` | **FAIL** | Placebo survival 33% (cap 0.5%): shuffled controls also cleared t≥3.8, exposing the "edge" as unconditional drift on a survivorship-flattered panel; 4x-cost sensitivity also negative |
| ETF pairs, distance method (3 trials) | `pairs-study-v1-20260719-001` | **FAIL** | Outright money-losing net of declared costs: all trials negative (terminal wealth 0.54–0.69 over ~26 years), SPA p=0.95 |
| Post-earnings announcement drift | `configs/pead-study-v1.yaml` | **BLOCKED** | No free point-in-time earnings timestamps/consensus; approximating them would defeat the causality gates |

## What the details teach

**Seasonal** is the closest miss this repo has recorded. Every one of the
nine trials was positive net of costs (5–9 bp per in-season day), eight of
nine passed walk-forward, all survived the 4x cost ladder, CPCV PBO was
0.13, and placebo survival was 0%. The family-level White/Hansen tests
rejected the zero-mean null at p=0.0024. What failed is the preregistered
*individual* bar: best DSR 0.88 (needs 0.95), best t 3.73 (needs 3.8).
Under the honesty contract that is a FAIL, full stop — the bar cannot be
lowered after seeing the results. Context that matters either way: holding
cash May–October forfeited most of three decades of compounding (terminal
wealth 8.8x vs 31.2x buy-and-hold on SPY), so even a validated Halloween
edge would be a risk-management overlay, not an alpha machine.

**Low-vol** is a textbook catch by the placebo control. Two trials looked
excellent in isolation (t≈4.1–4.2, DSR 0.99, walk-forward pass), but date-
shuffled and matched-random controls cleared the same t bar a third of the
time — permutation preserves an unconditional mean, so the t-stat was
measuring "low-vol stocks that survived into today's S&P 500 drifted up,"
not a harvestable timing rule. The 4x cost sensitivity was independently
negative (by 0.01 bp for vol_252 — costs consume the entire margin). A
credible low-vol test needs a point-in-time universe with delisted names.

**Pairs** confirms Do & Faff (2010): distance-method pairs on liquid,
highly-arbitraged ETFs lose money after realistic per-leg costs and a flat
100 bp/yr borrow fee. Every configuration was net-negative over ~26 years.
This family is closed at this data tier.

## Bookkeeping

- Gate rows: `edge_preholdout = FAIL` recorded in the catalog for all three
  campaigns; result documents with full per-trial evidence live at
  `artifacts/campaigns/<campaign_id>/preholdout/result.json`.
- The preregistration configs stay byte-identical to their declared form;
  any re-test requires a new campaign id and config.
- The validated-edge tier is unchanged: SPY turn-of-month and the five-name
  reversal basket remain the only gauntlet survivors.
