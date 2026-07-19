# Overnight-session edge — preregistration record

Status: **DECLARED, NOT YET EVALUATED.** The binding declaration is
[configs/overnight-edge-v1.yaml](../configs/overnight-edge-v1.yaml),
committed on 2026-07-17 before any evaluation code for this family was
written or run. Any change to the declared family after first evaluation is
a new campaign.

## Why this family, and why now

The full-grid discovery campaign (`full-lit-v2-ci-20260716-002`, 24,684
declared trials) tripped its survivor-fraction guard and was stopped. The
mandated audit of its 818 t+FDR passers found the overnight session as the
strongest structure that is not the already-promoted reversal family: LONG
overnight carried HAC t of 10.7 on the ANY condition net of declared costs,
consistent across sector partitions (t 10.1–11.1). Nothing survived deflated
Sharpe against 24,684 trials — which is the lesson, not a failure: the
promoted reversal edge only passed the gauntlet as a small preregistered
family. This campaign applies the same recipe to the overnight effect.

Economic grounding: the US equity overnight/intraday return split (Cooper,
Cliff, Gulen 2008) and clientele persistence (Lou, Polk, Skouras 2019) —
returns concentrate close-to-open, compensated overnight inventory risk and
open-auction retail flow being the standard explanations.

## The declared family (24 real trials, 72 with placebos)

Two instruments (SPY; equal-weight current S&P 500 equities) × six
conditions (ANY, turn-of-month, month-end window, Friday, market above
SMA200, market high-vol tercile) × two directions. One overnight interval
per entry: MOC buy (15:45 ET decision freeze), market-on-open sell.

## What is expected to kill it

Costs. An overnight strategy pays both auctions every session, so the
declared cost ladder (0.5×–4×) is the binding gate; the preregistration
explicitly records that a cost-negative verdict is a valid, reportable
outcome. Placebos, LITERATURE_V2 thresholds, walk-forward, CPCV/PBO,
causality checks, and independent confirmation all apply unchanged.

## Holdout

Forward-only: 2026-07-17 → 2028-07-17, single consumption, CI_V2-style
evaluator (mean and bootstrap CI lower bound both positive). No historical
window is reserved because earlier campaigns already observed the past
windows analytically; the verdict accrues on data that did not exist at
declaration time. Paper only throughout.
