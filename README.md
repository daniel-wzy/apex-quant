# Buffett Quant System

> Autonomous equity trading system: technical indicator confluence → XGBoost meta-labeling gate → vol-targeted execution.

---

## Architecture

```mermaid
flowchart LR
    A["OHLCV Data\n(multi-timeframe:\n30m, 1h, 4h, daily)"]
    B["Indicator Layer\n(N indicators,\nboolean signals)"]
    C["Confluence Scoring\n+ Combo Gate"]
    D["XGBoost Gate\n(calibrated P(profitable)\n≥ 0.75 to pass)"]
    E["Vol-Targeted Sizing\nrisk_pct × notional\n÷ (entry − stop)"]
    F["Position Manager\n(5-slot cap, T+1 entry)"]
    G["Risk Controls\n/ Floor Check\n(MTM + fail-safe)"]

    A --> B --> C --> D --> E --> F --> G
```

---

## "The Backtest I Didn't Believe"

The first full backtest returned a Sharpe ratio above 4. That number was not
celebrated — it was treated as a bug report.

A real edge in a liquid equity market with realistic costs doesn't produce
Sharpe >4. Either the system has a genuine, enormous advantage (unlikely on
first attempt), or something is wrong with the measurement. Spoiler: seven
things were wrong with the measurement.

**What the rebuild caught:**

- **Label leakage:** Forward labels overlapped the validation window — the model
  saw the future during training. Purged + embargoed CV brought AUC from
  0.843 → 0.659. The lower number is real.
- **Uncalibrated gate:** Raw XGBoost probabilities are ordered but not scaled to
  true probabilities. "τ = 0.75" meant nothing without calibration. Added
  isotonic regression calibration; the threshold now means what it says.
- **Survivorship bias:** The initial universe was hand-picked from names that had
  "worked." Expanded universe + per-symbol review + pruning of losers corrected
  the inflated baseline.
- **Drawdown diluted ~4×:** The drawdown denominator used a hardcoded initial
  capital constant instead of the running peak. Fixed to mark-to-market.
  Reported drawdown grew ~4× to its correct value.
- **Tail concentration:** Most P&L came from 2–3 names. The aggregate looked
  fine; per-symbol review exposed the concentration. Pruned the losers; the
  aggregate became more honest.
- **Disabled safety floor:** The portfolio floor was evaluated against realized
  PnL, not mark-to-market. An unrealized 40% drawdown in open positions would
  not have triggered it. Rebuilt as MTM with a data-outage fail-safe.
- **Partial-exit logging bug:** A quantity-weighting error fabricated losses on
  partial exits. Corrected with an auditable correction file — the original log
  is preserved. History was not rewritten.

The final out-of-sample expectancy is ~1.3R/trade, with a 95% CI lower bound
above zero. This is plausible for a confluence-based discretionary-style signal
with realistic costs applied. The initial Sharpe of >4 was not.

Full postmortems for each bug: [docs/validation.md](docs/validation.md)

---

## Validation Methodology

### Walk-Forward Cross-Validation

Random splits on financial time series leak the future. This system uses
purged/embargoed walk-forward CV throughout:

- **Train window:** 6 months
- **Validation window:** 1 month (out-of-sample)
- **Step:** 1 month
- **Embargo:** 5 bars (matching the maximum forward-label horizon) are removed
  from the end of each training set. This prevents labels computed over
  overlapping forward windows from contaminating the validation fold.

Result: no bar's label ever reflects price data from the validation period.

### Isotonic Calibration

XGBoost produces well-ordered probabilities but uncalibrated scales. The
pipeline adds isotonic regression calibration on a held-out set. A reliability
diagram confirms that P(profitable | model score ≥ τ) matches the actual
positive rate within 5 percentage points across threshold buckets.

### Trade-Level Evaluator with Bootstrap CI

`quant/evaluator.py` computes all metrics from trade-level data: no simulation
state, no equity-curve tricks. Primary gates:

- **Expectancy R:** Mean net trade result in R-units (where R = entry − stop)
- **Bootstrap CI95 lower bound:** 10,000 bootstrap resamples; must be > 0
- **Profit factor:** Gross profit R / gross loss R

Win rate is reported as a diagnostic only, never used as a gate. A system
with a 40% win rate and 3R average wins beats a system with 60% win rate and
0.5R average wins.

### Cost and Slippage Realism

Every trade includes round-trip costs from a realistic model:

- **Commission:** $0.0049/share, min $0.99/trade (Moomoo US pricing)
- **Spread:** % of price, scaled by liquidity tier (low/mid/high)
- **Market impact:** Flat per-share amount, tier-scaled

Costs are tracked explicitly in every Scorecard (`total_costs_paid_pct`).
There is no "assume zero costs" option — this was a deliberate design choice
after finding that costs reduced expectancy by ~0.2R in early tests.

### 2022 Bear Market Stress Test

The out-of-sample period includes 2022 (S&P −19%, Nasdaq −33%). Results:
positive expectancy was maintained. The per-symbol review found that several
names which looked good in 2020–2021 became consistent losers in 2022; they
were pruned from the universe following this discovery.

### Per-Symbol Review

Every symbol in the universe is evaluated independently using the same Evaluator
metrics. Any symbol with:
- Negative expectancy in its per-symbol backtest, OR
- 95% CI lower bound ≤ 0, OR
- Fewer than 10 trades (insufficient sample)

…is removed from the universe. This check catches what aggregate statistics hide.

---

## Results

### Out-of-Sample Backtest (Pruned Universe, v2.2 Methodology)

| Metric | Value |
|---|---|
| Trades (n) | ~312 |
| Expectancy R | ~1.34R/trade |
| Bootstrap CI95 lower bound | > 0 (consistently) |
| Win rate | ~55% (diagnostic only) |
| Profit factor | ~3.79 |
| 2022 bear stress | Positive expectancy maintained |

These numbers reflect a pruned, survivorship-corrected universe with realistic
costs, purged CV, and corrected drawdown calculation.

### Live (Shadow Validation Window)

The system has been running in shadow mode (signals computed and logged but not
all executed live due to operational constraints).

**Honestly stated:**
- The shadow window has ~16 live-executed trades at time of writing (small sample)
- The indicator-only era (before the XGBoost gate) was roughly break-even
- The quant-gated period is positive
- The edge is promising but unproven at scale
- An 8-day auth expiry outage interrupted the shadow period; that gap is excluded

The live sample is too small to draw strong conclusions. Ongoing shadow
validation continues.

---

## Risk & Operations

**Vol-targeted sizing:** Each trade risks 0.75% of notional. Position size =
risk_usd / (entry − stop), capped at max position USD. Tight-stop trades get
larger positions; wide-stop trades get smaller ones.

**Concurrent position cap:** Maximum 5 simultaneous positions. New signals
arriving when 5 slots are full are skipped, not queued.

**MTM floor with fail-safe:** The portfolio floor is evaluated against current
mark-to-market value, not cost basis. If any live price is unavailable (API
error, auth expiry, network failure), the floor check assumes worst-case
(stop price for all open positions) and halts trading regardless.

**Watchdog:** An independent scheduled process (no shared code or auth with
the bot) checks that the bot has run within the expected window during market
hours. Alerts via raw Discord webhook. A second, external dead-man's-switch
service covers the case where the machine itself is down.

Operational details: [docs/risk-and-ops.md](docs/risk-and-ops.md)

---

## Lessons Learned

> **"A number that looks too good is a bug until proven otherwise."**  
> The initial Sharpe >4 was a bug report, not a win.

> **"Measurement bugs outnumbered strategy bugs 4:1."**  
> Four of the seven bugs were in the evaluator, CV, or reporting — not in the
> strategy logic itself.

> **"Fix the validator before you trust the result."**  
> The unit test suite for the Evaluator was written before any strategy was
> evaluated. Without a verified foundation, nothing else is meaningful.

> **"Fail safe, but never fail silent."**  
> Every failure mode has an explicit handler. Missing prices → worst-case
> assumption + halt. Auth expiry → dead-man's switch fires.

> **"Per-symbol review catches what aggregate stats hide."**  
> The aggregate backtest looked fine. The per-symbol breakdown showed 2–3 names
> carrying the entire P&L. Pruning the losers made the aggregate more honest.

> **"Correction files beat history rewrites — auditability matters."**  
> When the partial-exit logging bug was found, a correction file was created,
> not a log rewrite. The original is preserved. Anyone can audit the change.

---

## Tech Stack

- **Python 3.11+** — core language
- **XGBoost** — meta-labeling gate classifier
- **scikit-learn** — StandardScaler, isotonic calibration, CV utilities
- **pandas / NumPy / SciPy** — data manipulation and statistics
- **Moomoo OpenD API** — live OHLCV data and order execution
- **macOS launchd** — scheduling (watchdog, bot cadence)
- **Discord webhooks** — alerting (raw curl, no library dependency)
- **pytest** — test suite

---

## Project Structure

```
buffett-showcase/
├── README.md
├── Makefile
├── requirements.txt
├── .gitignore
│
├── quant/                     # Core quantitative pipeline
│   ├── __init__.py
│   ├── qconfig.py             # Paths, thresholds, walk-forward parameters
│   ├── feature_store.py       # Single source of truth: OHLCV → feature vector
│   ├── model.py               # QuantModel wrapper (scaler + XGBClassifier)
│   ├── train_walkforward.py   # Purged/embargoed walk-forward CV
│   ├── evaluator.py           # Trade-level PnL evaluator (primary gate)
│   ├── calibrator_utils.py    # Isotonic calibration load/predict
│   ├── cost_model.py          # Realistic commission + slippage model
│   ├── gen_sim_trades.py      # Simulate portfolio from synthetic signals
│   └── floor_check.py         # MTM floor check with data-outage fail-safe
│
├── indicators/                # Indicator layer
│   ├── __init__.py
│   ├── interface.py           # Abstract base class (the contract)
│   └── example_indicator.py  # MA crossover demo (implements the interface)
│
├── watchdog.py                # Independent process: bot health monitor
│
├── tests/                     # Test suite
│   ├── test_evaluator.py      # Unit tests with known-expectancy synthetic trades
│   ├── test_purged_cv.py      # Walk-forward CV produces no train/test overlap
│   └── test_floor_check.py    # MTM floor check and fail-safe scenarios
│
├── data/
│   └── synthetic/
│       └── generate.py        # Generate synthetic OHLCV + signal data
│
└── docs/
    ├── architecture.md        # Data flow in depth
    ├── validation.md          # Postmortems on seven bugs
    └── risk-and-ops.md        # Operational design and outage handling
```

---

## Running the Synthetic Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
make synth
# or: python data/synthetic/generate.py

# Run tests
make test
# or: python -m pytest tests/ -v

# Lint check
make lint
```

The synthetic pipeline exercises the full code path — feature store,
cost model, evaluator, and portfolio simulator — against generated data with
no live market dependencies.

---

## Disclaimer

Personal project. Parameters in this repository (indicator weights, threshold
values, universe lists) are **illustrative** — they are representative of the
system's structure, not the exact live configuration. Not financial advice.
Past backtest performance does not guarantee future results.
