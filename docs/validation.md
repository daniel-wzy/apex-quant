# Validation — Postmortems on Seven Bugs

> "A number that looks too good is a bug until proven otherwise."

The initial backtest produced a Sharpe ratio above 4. Rather than treating this
as a success, it was treated as a red flag. The following is a postmortem for
each bug that was found and fixed during the rebuild. Together they reduced the
reported Sharpe from >4 to ~1.3 — and made the remaining number trustworthy.

---

## Bug 1: Label Leakage — Overlapping Forward Labels

**Symptom**  
AUC during walk-forward CV was implausibly high (≈ 0.84). The model appeared
to generalize across time windows with minimal degradation. Too good.

**How it was caught**  
Added a manual check: for each train/validation split, verified that no bar
index appeared in both the training label set and the validation set. Found
that forward labels computed over a 20-bar horizon were overlapping — the last
training bars had labels that "looked into" the validation window.

**Root cause**  
The labeling function applied a rolling forward window (20-bar horizon: +15% TP
before −5% stop) over the entire dataset before splitting. Bars near the
train/val boundary had labels that incorporated price data from the validation
period.

**Fix**  
- Implemented purged/embargoed walk-forward CV (see `quant/train_walkforward.py`).
- An embargo of ≥ 20 bars (≥ the label horizon) is removed from the end of
  each training set before fitting.
- Walk-forward: 36 folds, 2023-07 → 2026-06.
- Labels are recomputed per fold using only data available in the training window.

**Impact on metrics**  
AUC dropped from 0.843 → 0.659. This was expected and welcomed — 0.659
OOF AUC on a hard classification problem with real transaction costs is
consistent with a real edge, not a leakage artefact.

---

## Bug 2: Uncalibrated XGBoost Probabilities

**Symptom**  
Setting the entry threshold to 0.75 (intended to select "high-confidence"
trades) didn't produce noticeably better win rates than 0.60. The threshold
wasn't buying much precision.

**How it was caught**  
Plotted a reliability diagram (calibration curve). Raw XGBoost probabilities
clustered in a narrow range (0.55–0.70) regardless of the true positive rate.
The model was not well-calibrated — "0.75" on the raw scale didn't mean
75% probability of a profitable trade.

**Root cause**  
XGBoost outputs raw scores from the final tree ensemble. These are
well-ordered (higher = more likely positive) but not necessarily calibrated
to true probabilities. The scale shifts with the training class balance,
number of estimators, and regularization.

**Fix**  
Added isotonic regression calibration via `quant/calibrator_utils.py`. The
calibrator is fit on a held-out calibration set (separate from both training
and validation). Calibrated probabilities are stored alongside predictions
in the OOF file.

**Impact on metrics**  
The threshold now has a stable semantic meaning. Precision at τ=0.75
improved from ~55% to ~63% on the validation set. The entry-threshold sweep
(comparing 0.60 vs 0.65 vs 0.75) became meaningful only after calibration.

---

## Bug 3: Survivorship Bias in Universe Construction

**Symptom**  
Backtest win rates for individual symbols were uniformly high. Almost no
symbols looked bad in isolation.

**How it was caught**  
Noticed that the initial symbol list was assembled by hand from names that
had "worked" during manual observation. Added a check: compare per-symbol
backtest win rates to live performance for the same period. The correlation
was low.

**Root cause**  
The initial universe was built from names that were already known to produce
interesting signals. Symbols that failed to produce signals were naturally
excluded — because they weren't observed. Classic survivorship bias.

**Fix**  
- Expanded the universe to include names that were not pre-selected.
- Added a per-symbol review step: each symbol is evaluated individually using
  the same Evaluator metrics (expectancy, profit factor, CVaR).
- Symbols are excluded only when the entire 95% CI is negative on a real
  backtest sample, or for data/liquidity issues.
- The per-symbol review tool has a validation requirement: when applied to the
  full universe, the aggregate must reproduce the baseline.
- The pruned universe is frozen at a versioned checkpoint.

**Impact on metrics**  
Win rate and profit factor dropped across the board. The overall expectancy
settled at a lower but more reliable value, consistent with a realistic edge.

---

## Bug 4: Drawdown Denominator Bug (Hidden Base-Equity Offset)

**Symptom**  
Reported maximum drawdown was suspiciously low (≈ 8%) for a strategy with
significant losing streaks visible in the equity curve.

**How it was caught**  
Eyeballed the equity curve against the reported drawdown number — they
didn't match. Traced the drawdown computation and found the denominator
was using a hidden $100k base constant for a ~$20k account, understating
drawdown approximately 4×.

**Root cause**  
The legacy drawdown function computed `(equity − peak) / initial_capital`
instead of `(equity − peak) / peak`. With a large base constant relative
to the actual account size, the denominator inflated, shrinking the reported
drawdown percentage approximately 4×.

**Fix**  
Rewrote `_max_drawdown_pct()` in `quant/evaluator.py` to use the running
peak of the equity curve as the denominator. Added a unit test with a known
equity curve and a known expected drawdown.

**Impact on metrics**  
Reported drawdown increased approximately 4×. The strategy still passed the
drawdown gate, but the corrected number is now trustworthy and comparable
to industry-standard reporting.

---

## Bug 5: Tail Concentration — Most P&L from 2–3 Names

**Symptom**  
The aggregate backtest looked good. Per-symbol analysis revealed that the
worst-5% of trades came from 5 micro-caps; the rest were break-even or losing.

**How it was caught**  
Added a per-symbol statistics tool (`run_bucket_breakdown.py`) that reports
expectancy, profit factor, and trade count broken down by symbol. The
concentration was immediately obvious.

**Root cause**  
Aggregate metrics hide per-symbol heterogeneity. A strategy that works well
for NVDA and MSFT but loses on 12 other names still shows positive aggregate
expectancy — but that expectancy doesn't generalize to new names or to
changes in the "hero" names' behavior.

**Fix**  
- Per-symbol review is now a required step before any model is promoted.
- Of the 5 micro-caps: BW and EOSE had no statistical edge (CI not clearly positive; EOSE produced the 3 worst trades). ALOY showed positive edge but was pruned because its stop data was broken and risk sizing never engaged (thin OTC name). CIFR and BBAI were kept.
- The pruned universe is treated as a frozen baseline.

**Impact on metrics**  
Aggregate expectancy and profit factor declined but became more robust.
The pruned universe has more consistent per-symbol performance, reducing
the risk that aggregate metrics mask a few outliers.

---

## Bug 6: Disabled Safety Floor

**Symptom**  
The portfolio floor was disabled in code — the mechanism existed in config but
was not being evaluated. It was checked against cost-basis entry price rather
than MTM equity.

**How it was caught**  
Code review of the floor check logic. Found that it read `pnl_realized`
from the positions file rather than computing `sum(price × qty) + cash`.

**Root cause**  
The floor was implemented during an early phase when only realized PnL was
tracked. When unrealized positions were added, the floor check was not
updated.

**Fix**  
Rebuilt as `quant/floor_check.py` — a self-contained module that:
1. Fetches live prices for all open positions
2. Computes MTM value as `sum(price × qty) + cash`
3. Compares against the configured floor
4. Activates the data-outage fail-safe if any price fetch returns None

**Impact on metrics**  
The floor now triggers correctly during drawdowns. The fail-safe ensures
that a data outage never silently bypasses the floor check.

---

## Bug 7: Partial-Exit Logging Bug (Fabricated Losses)

**Symptom**  
Some trades in the log showed large, implausible losses on the exit leg only.
The entry prices looked correct but exit prices were sometimes far below the
stop price — which should have been impossible given the stop logic.

**How it was caught**  
Auditing the trade log against intraday price data. The suspicious entries
were partial exits that had been logged with an incorrect exit price
(quantity-weighted average over two legs was applied to only one leg).

**Root cause**  
The partial-exit handler multiplied quantity by price correctly for the
first lot but wrote the full quantity's dollar value as the per-share price
for the second lot. The result was an arithmetic "loss" on paper that never
occurred in the market.

**Fix**  
- Corrected the per-share exit price calculation in the exit handler.
- Created `trade_log_corrected.jsonl` — a corrected version of the affected
  records with an audit trail (`correction_reason`, `corrected_at` fields).
- Did NOT rewrite the original log. The original `trade_log.jsonl` is
  preserved; the corrected file is an additive amendment.

**Impact on metrics**  
Removing the fabricated losses improved expectancy for the affected period.
The correction was conservative: only records with verifiable price data
were corrected; ambiguous records were left as-is.

> **Principle:** Correction files beat history rewrites. Auditability matters.
> Anyone reviewing the log can see exactly what was changed, when, and why.
