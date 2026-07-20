# Live alerts — what pushes to your phone, and when

All alerts are DIAGNOSTIC, paper-only, and delivered over Telegram plus the
Android companion's read-only views. The alert stack, in the order a day
unfolds:

| Alert | When | What it says |
| --- | --- | --- |
| `PRE_CLOSE` | Weekdays ~15:35 ET (scheduled task, 10 min before the decision freeze) | Today's session clears the ~15 bp alignment bar outside an estimated earnings window; includes a 15-minute-delayed quote and `[peers healing: N/M]` context |
| Nightly summary | After the close (~16:40 ET) | Reversal basket, entry/exit MOC sessions, forward-ledger fills |
| `ENTRY_SIGNAL` | Inside the nightly summary | Tomorrow's session clears the bar for a calendar symbol, with regime/dip/peer context |
| `EARNINGS_PRINTED` | Inside the nightly summary, the evening a real 8-K Item 2.02 lands on EDGAR | The estimated blackout is lifted; entry rules re-armed on actual, not estimated, information |

## Five-minute Telegram setup

1. In Telegram, message **@BotFather** → `/newbot` → copy the token.
2. Message your new bot once (any text), then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy your chat id.
3. `setx EDGESTACK_TELEGRAM_TOKEN "<token>"` and
   `setx EDGESTACK_TELEGRAM_CHAT "<chat id>"` (never put tokens in files).
4. Open a NEW terminal and run
   `python -m edgestack.agenttools telegram-test` → expect `SENT`.
5. Register the scheduled jobs once: `scripts/install-autostart.ps1`
   (Mobile API at logon, Post-Close 22:40, Pre-Close 21:35 local; both times
   are parameters — shift them when Europe/US daylight-saving drift changes
   the offset).

Until credentials exist every sender reports `SKIPPED_NO_CREDENTIALS` and
nothing breaks.

## How the earnings logic stays honest

The blackout windows in `entry-check` and the alert suppression are
ESTIMATES projected from each company's historical EDGAR 8-K cadence and are
stamped `EARNINGS_WINDOW_ESTIMATED_NOT_CONFIRMED`. The nightly job then
checks EDGAR for the *actual* filing; only a real acceptance timestamp emits
`EARNINGS_PRINTED` and re-arms entries. Live detections append to
`artifacts/earnings/live-announcements.parquet` — the sealed crawl that fed
the PEAD campaign stays byte-identical.

## The GO score that is deliberately NOT here

A 0-100 composite "GO score" (blending alignment, regime, dip, and peer
signals into one number) was built and evaluated in the sister research
program — and **failed its preregistered backtest gate**: days scoring 60+
showed no better forward returns than other days (slightly worse on CTSH).
Per the pre-committed rule it therefore fires no alerts anywhere, and this
repo does not implement it at all. The individually validated and
individually diagnosed triggers above are the only alert sources. A
composite that scores well in the dashboard but fails the data is exactly
the kind of plausible-looking idea this system exists to refuse.
