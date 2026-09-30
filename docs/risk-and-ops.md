# Risk & Operations

## Portfolio Floor Check

### Formula

The floor check computes mark-to-market (MTM) portfolio value:

```
MTM = cash + sum(live_price[symbol] × quantity[symbol]  for each open position)
```

If `MTM < floor_usd`, all trading activity halts immediately.

The floor is configured as a dollar amount, not a percentage. It represents
the minimum acceptable portfolio value before human review is required.

### Data-Outage Fail-Safe

If the live price for any open position cannot be fetched (API error, network
failure, auth expiry, etc.), the floor check assumes worst-case:

```
MTM_failsafe = cash + sum(stop_price[symbol] × quantity[symbol])
```

The stop price is the lowest realizable value if all positions were stopped
out immediately. This is always ≤ the true mark-to-market value.

**If any price is unavailable, trading halts regardless of the MTM estimate.**

This fail-safe exists because the alternative — continuing to trade while
blind to actual portfolio value — is unacceptable. "Fail safe, never fail
silent."

### What Triggers a Halt

1. MTM < floor_usd (direct floor breach)
2. Any live price unavailable (fail-safe)
3. Watchdog detects extended bot silence (indirect — watchdog alerts but does
   not directly halt trading)

After a halt, the system requires manual review and restart. There is no
automatic recovery.

---

## Watchdog

### Why a Separate Process?

The watchdog runs as an independent scheduled process (macOS launchd, or
equivalent cron job on other systems). It shares no code, no imports, and
no authentication with the trading process.

**Motivation:** If the watchdog were part of the trading process, any failure
that silences the trading process would also silence the watchdog. The whole
point of a dead-man's switch is that it fires *because* the main process
stopped. Coupling them defeats the purpose.

The watchdog's entire dependency graph:
- Python standard library (`json`, `subprocess`, `datetime`, `zoneinfo`)
- `curl` (for webhook delivery, avoids SSL cert issues with Python requests)
- Read access to `positions.json` (written by the bot)

### Market-Hours Gating

The watchdog checks the time in US Eastern time before evaluating:

```
Market hours: Monday–Friday, 09:30–16:15 ET
Grace period: first 45 min after open (bot cron may start at 10:00 AM)
```

Outside market hours, the watchdog exits immediately without evaluating or
alerting. This prevents false positives on weekends and overnight.

### Throttle

Once an alert is sent, the watchdog suppresses further alerts for 2 hours
(configurable via `RETHROTTLE_MINUTES`). This prevents alert spam during
extended outages — a single alert is sufficient to notify the operator.

### Dead-Man's-Switch Pattern

Two independent alert paths are used:

1. **External (healthchecks.io or similar):** The trading process pings a URL
   on every successful run. If the ping stops, the external service sends an
   alert after a grace period. This fires even if the machine is down.

2. **Local (watchdog):** Runs on the same machine, reads `positions.json`,
   sends a Discord webhook alert if the bot has been silent for >45 min during
   market hours. This fires when the machine is up but the bot is silent.

Both paths are required. The external path covers machine failures; the local
path catches bot crashes that don't take the machine down.

---

## Outage Handling

### The 8-Day Auth Expiry Outage

During the initial shadow-mode period, the trading agent's authentication to
the AI provider expired. The agent could not run for approximately 8 days.

**What happened:**
- The bot stopped executing at the normal cadence.
- The watchdog correctly detected the silence and sent alerts.
- The external dead-man's switch also fired after its grace period.
- No trades were executed during the outage.

**How the shadow window was handled:**
- The 8-day gap was excluded from the shadow-mode statistics.
- The shadow window start date was not moved; the gap is documented as an
  operational outage rather than a strategy failure.
- Any open simulated positions at the time of the outage were treated as
  "held at stop price" for the MTM calculation — conservative, matching the
  fail-safe logic.

**Lesson:** Auth expiry is an operational risk that requires monitoring. The
watchdog now specifically includes guidance to check agent auth status as a
first-step remediation when alerts fire.

---

## Versioning

### Frozen Baselines

Each major model version is captured as a frozen baseline:

- `quant/data/snapshot_id.txt` — hash of the data snapshot used for training
- `quant/models/` — versioned model files (e.g., `entry_model_daily.pkl`)
- `quant/data/configs_tried.json` — count of configurations evaluated (for
  Deflated Sharpe Ratio calculation; prevents the DSR from understating the
  multiple-testing penalty)

Baselines are never overwritten. New versions get new version tags.

### Correction Files vs. History Rewrites

When errors in the trade log are discovered and corrected:

- `trade_log.jsonl` — original, unmodified log. Preserved forever.
- `trade_log_corrected.jsonl` — amended records with `correction_reason` and
  `corrected_at` fields. Treated as an additive overlay.

This is intentional. Rewriting history makes auditing impossible. Correction
files allow anyone to reconstruct the original, understand what was changed,
and verify that the correction was applied correctly.

The same principle applies to backtest results: correction notes are added to
`RESULTS.md` rather than silently updating the numbers.

---

## Summary of Risk Controls

| Control | Trigger | Action |
|---|---|---|
| Portfolio floor | MTM < floor_usd | Halt all trading |
| Data-outage fail-safe | Any price unavailable | Halt (assume worst case) |
| Position cap | ≥5 open positions | Skip new signals |
| Per-position cap | Position USD > max | Size down |
| Stop loss | Price ≤ stop_price | Exit at next open |
| Take profit | Price ≥ TP (15%) | Exit at next open |
| Watchdog | Bot silent >45 min | Discord alert |
| External dead-man | Ping missed | External alert (machine-independent) |
