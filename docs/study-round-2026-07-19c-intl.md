# Study round 2026-07-19c — the international retests

Two frozen rules, twelve US-listed country ETFs each (9 European, Korea,
Japan, Australia), sealed panel 1995→2026 (`artifacts/intl/manifest.json`,
SHA-256 pinned), preregistered in commit `41c6c49` before evaluation.
**Both families FAILED their preregistered gauntlets.** Placebos were 0%
in both; the verdicts are about the effects, not the machinery.

## Halloween international (`seasonal-intl-v1-20260719-001`) — FAIL, and the program's most remarkable failure

The frozen US rule, applied unchanged to instruments the near-miss never
touched, produced the same picture a second, independent time:

- **All 12 of 12 trials positive**, +3.7 to +11.0 bp per in-season day net
  of costs; **all 12 pass BH-FDR and Romano-Wolf; all 12 survive 4x
  costs**; 11 of 12 pass walk-forward; family SPA p=0.0007.
- Best trials: Sweden t=3.71, Switzerland t=3.43, **Korea t=3.30 with the
  largest effect size (+11.0 bp/day)** — Europe and Korea strongest,
  Japan weakest, exactly the literature's ordering.
- And still no survivor: the best t (3.71) sits under the 3.8 bar and the
  best DSR (0.86) under 0.95 — **almost exactly where the US family
  landed (t=3.73, DSR 0.88)** — and CPCV PBO 0.73 fails the <0.20 gate
  (twelve near-identical trials have unstable in-sample ranks, which the
  gate cannot distinguish from overfitting).

Two independent continents, one frozen rule, the same magnitude and the
same shortfall. The honest reading: the Halloween effect is very likely a
real, cost-robust ~6–10 bp/day phenomenon whose effect size sits just
below a discovery bar calibrated for data-mined hypothesis zoos. Under
the honesty contract the verdict is FAIL, the bar does not move, and both
evaluations are now spent. The remaining honest paths to promotion are
the two forward holdouts accruing to 2028 (US and international) and
genuinely new data — nothing on this panel may be re-tested.

## Turn-of-month international (`tom-intl-v1-20260719-001`) — FAIL; the validated edge does not travel at these costs

The frozen SPY LAST1_FIRST3 window on the same 12 wrappers:

- Directionally present nearly everywhere (family SPA p=0.0028; 11 of 12
  positive net at 1x costs; **Korea strongest at +13.9 bp/window-day,
  t=2.83**, Japan flat), but individually weak — t mostly 1.3–2.3 — and
  **every trial fails the 4x-cost stress**: two fills per month against a
  4-session window is affordable on SPY's spreads and fatal on country-
  wrapper spreads.
- CPCV PBO 0.27 also fails the gate.

The mechanism appears global, the implementation is only investable on
the cheapest instrument in the world — which is the one already
validated. SPY TOM's status is unchanged; it gains a diagnostic
cross-market echo, not a second act.

## Bookkeeping

- Gate rows `edge_preholdout = FAIL` recorded for both campaigns; full
  per-trial evidence under `artifacts/campaigns/<campaign_id>/preholdout/`.
- Panel and configs stay byte-identical to their declared form; any
  re-test requires new campaigns on new data.
- Program scoreboard after this round: **2 validated edges, 13 closed
  families**, with the Halloween effect now the best-documented
  almost-edge in the repo — confirmed in shape on two continents, never
  promoted, never hidden.
