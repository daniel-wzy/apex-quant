# Architecture — Data Flow in Depth

## Overview

The Buffett Quant System is a multi-stage pipeline. Every component is designed
with a single invariant in mind: **the same code path runs in training and in
live execution**. Deviation between the two is the leading cause of production
alpha decay in indicator-driven systems.

---

## 1. OHLCV Ingestion

**Provider:** Moomoo OpenD (local API proxy)  
**Timeframes:** 30m, 1h, 4h, daily  
**Universe:** US equities, curated list reviewed per-symbol for liquidity and
survivorship. SPY is fetched as a market-context reference for all symbols.

**Ingestion protocol:**
- Bars are fetched with `history_start = 2023-01-01` as a target ceiling.
- Moomoo paginates; the fetcher accumulates pages until the target is reached.
- Data is stored in Parquet format in `quant/data/raw/`.
- Each fetch is timestamped; the data snapshot ID is recorded in
  `quant/data/snapshot_id.txt` and embedded in every Scorecard for auditability.

---

## 2. Indicator Layer

Each indicator is a Python module implementing `indicators.interface.Indicator`:

```python
class Indicator(ABC):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame: ...
    @property
    def signal_columns(self) -> list[str]: ...
```

`compute(df)` takes OHLCV bars and returns the same DataFrame extended with
one or more **boolean (0/1) signal columns**. All signals are computed from
data available at or before each bar's close. No lookahead.

The live system uses 8 indicator families covering momentum, trend, oscillator,
and volume-flow regimes. Each produces 2–6 binary signals (buy or sell direction).
Total: ~40 boolean columns registered in `feature_store.SIGNAL_COLUMNS`.

**Multi-timeframe:** The pipeline runs all indicators independently on each of
the 4 timeframes. Cross-timeframe confluence counts (`buy_tfs_count`,
`sell_tfs_count`) are computed as engineered features before the model sees them.

---

## 3. Confluence Scoring

Before the XGBoost gate, a rule-based **confluence score** filters candidates:

```
confluence_score = buy_pts − sell_pts
```

Each signal column is assigned a weight in `SIGNAL_POINTS`. Context bonuses
(`CONTEXT_POINTS`) adjust based on trend-state columns (HMA direction, position
relative to 180-bar MA). The confluence score is itself a feature fed to the
model — it carries the human-readable interpretation of "how many indicators agree."

**Indicator-combo gate (`is_weak_entry_combo`):** An entry whose ONLY firing
buy-indicator family is in `BLOCK_ALONE_FAMILIES` is blocked regardless of the
model score. These families were identified as net losers when acting alone
(win rate 11–37%) in per-symbol review. A second confirming family clears the gate.

---

## 4. Feature Engineering (`feature_store.py`)

`feature_store.build_feature_frame()` is the **single source of truth** for
turning OHLCV bars into the model's input vector. It is imported by both the
training pipeline and the live decision path.

**Engineered features include:**

| Feature | Description |
|---|---|
| `confluence_score` | Net buy minus sell points from all indicators |
| `num_buy_indicators` | Count of distinct buy-indicator families firing |
| `num_sell_indicators` | Count of distinct sell-indicator families firing |
| `buy_tfs_count` / `sell_tfs_count` | Cross-timeframe agreement count |
| `hma_state` | HMA trend direction: 0=bearish, 1=neutral, 2=bullish |
| `above_180` | Price above 180-bar MA (1/0) |
| `vol_ratio` | Current bar volume / 5-bar trailing mean |
| `price_vs_stop` | (close − stop) / close — distance to support |
| `stock_change_pct` | Bar % change (close vs prev close) |
| `spy_change_pct` | SPY bar % change, aligned merge-asof |
| `relative_strength` | stock_change_pct − spy_change_pct |
| `time_of_day` | Minutes since 09:30 open |
| `day_of_week` | Integer 0–4 (Mon–Fri) |

**Train/live parity guarantee:** `build_feature_frame()` is called identically
in training (over historical bars) and live (over the current bar). If the live
path used a different code path for any feature, the model would receive a
distribution shift. This guarantee is enforced by the module structure — there
is no separate "live feature builder."

---

## 5. XGBoost Gate

**Model:** `XGBClassifier` wrapped in `QuantModel` (see `quant/model.py`).

Each timeframe has its own model. A pooled model was tested but found to simply
learn `time_of_day` as a timeframe proxy, reducing signal quality.

**Training pipeline:**
1. `build_feature_frame()` produces one row per bar per symbol per timeframe
2. Label module applies forward-looking TP/stop rules to produce `label_tp15`
   (15% TP, 5% stop, 20-bar horizon) — the primary entry label
3. Only bars where ≥1 buy signal is active are used as training candidates
4. Walk-forward CV (36 folds, 2023-07 → 2026-06, ≥20-bar embargo) validates OOF AUC
5. Final model trained on all data, saved to `quant/models/`

**Calibration:** Raw XGBoost probabilities are not calibrated by default. The
pipeline adds an isotonic regression calibration layer (`calibrator_utils.py`).
This step is non-optional: without it, "threshold=0.75" has no stable meaning —
the same raw score means different things across different trees. The calibrated
probability is what the live path thresholds against `ENTRY_THRESHOLD`. The
validated configuration uses τ = 0.50; the live bot currently runs a stricter
threshold of τ = 0.75.

---

## 6. Vol-Targeted Sizing

Position size is risk-based, not fixed-dollar:

```
shares = (risk_pct × notional) / (entry − stop)
shares = min(shares, max_position_usd / entry)
```

With `risk_pct = 0.60%` of notional (illustrative), each trade risks approximately the same
dollar amount regardless of price or stop distance. Large-stop trades get
smaller positions; tight-stop trades get larger ones.

**Concurrent-slot cap:** At most 5 positions open simultaneously. New signals
that arrive when 5 slots are full are skipped (not queued).

---

## 7. Risk Controls

| Control | Implementation |
|---|---|
| Portfolio floor | `quant/floor_check.py` — MTM vs configurable floor |
| Data-outage fail-safe | Floor check uses `stop_price` as worst-case if price fetch fails |
| Position cap | Max 5 concurrent, max \$N per position |
| Stop loss | 5% hard stop (initial stop = prior 5-bar low) |
| Take profit | 15% (from OOF backtest parameter sweep) |

---

## 8. Watchdog (Independent Process)

The watchdog runs as a separate process on a schedule (e.g., launchd/cron),
completely independent of the trading process and the main agent.

**Why separate?** If the watchdog shared any state, code, or auth with the
trading process, a crash in the trading process could silence the watchdog too.
The watchdog's only dependencies are:
- `positions.json` (written by the trading bot, read-only for watchdog)
- A Discord webhook URL in `.env`
- Standard library + `curl`

The watchdog is the sole alert path for local failures (machine is up, bot is
silent). Alerts are delivered via raw webhook.

---

## Data Flow Diagram

```
OHLCV Data (Moomoo OpenD, multi-timeframe)
         │
         ▼
  Indicator Layer
  (N indicator modules, each returns boolean signal columns)
         │
         ▼
  feature_store.build_feature_frame()
  (OHLCV + signals + engineered context features)
         │
         ▼
  Confluence Score + Indicator-Combo Gate
  (rule-based pre-filter, removes weak/solo-family entries)
         │
         ▼
  XGBoost Gate
  (per-timeframe classifier, isotonic calibration)
  P(profitable) ≥ 0.50 to pass (live bot: ≥ 0.75)
         │
         ▼
  Vol-Targeted Sizing
  (risk_pct × notional / (entry − stop), capped at max_position_usd)
         │
         ▼
  Position Manager
  (5-slot concurrent cap, T+1 entry on next bar open)
         │
         ▼
  Risk Controls / Floor Check
  (MTM vs floor, data-outage fail-safe, watchdog alert)
```
