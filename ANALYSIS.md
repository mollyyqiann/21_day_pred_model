# Performance Review & Validated Improvements

> **Superseded (2026-08).** The 21-day-gainer section below — "hits its number",
> "top-5 picks hit 58%" — did not survive a later revalidation that corrected for
> survivorship and look-ahead bias. See the status section at the top of
> [README.md](README.md). This document is kept as a record of what was believed
> in June 2026, including the useful part: that validation overturned two of
> three first instincts. The burst and drop findings have not been re-examined
> under the same controls and should be read with the same suspicion.


A backtested reflection on the three models' live predictions, and improvement
proposals that were **validated before any change** — notably, two of three
first-instinct fixes were overturned by the backtest.

## 1. Track records (live predictions, graded against realized outcomes)

### Burst — a real edge, regime-dependent
- Flagged names hit the burst event **48.7%** of the time; mean **+5.0%** at 5 days
  (median +3.4%, 64% positive).
- Heavily regime-dependent: hit rate **30% in a choppy month vs 68% in a trending
  month**.
- The `v4` universe is strongest (56.6% hit, the only predictive regression head,
  corr +0.46). Highest-confidence calls (prob ≥ 0.80) delivered the best returns.

### Drop — no usable edge in the tested (bull) regime
- Drop event hit rate **31.4%**; flagged names returned **+1.2%** at 5 days on
  average (only 49% actually fell). Probability showed **no calibration**.
- Root cause (see §3): the target predicts a transient down-window, which is
  decoupled from net return — effectively a **volatility detector**.

### 21-day gainer — hits its number, driven by a fat tail
- Across all 28 names ever surfaced, **8 touched +30% (29%)** — in line with the
  ~30–37% backtested top-5 rate.
- The edge lives entirely in the **top-5**: top-5 picks hit **58%**, top-15-only
  names hit just **6%**.
- Returns are fat-tailed: the top-3 names averaged **+72%** at window end, the other
  25 averaged **+2%**, and 12/28 ended underwater. One May semis rally
  (MU +90%, SMCI +83%, AMD +45%) carried the result.
- Half of every peak gain was given back by window end (mean peak +22% → end +10%).

## 2. Validated improvements

| Model | Change | Before → After |
|---|---|---|
| **21-day gainer** | Sell **½** at +30% touch, let the rest run (not a full lock-in) | risk-adjusted return (mean/std) 0.50 → **0.55**; median +8.8% → **+19.4%**; round-trippers rescued, fat-tail kept |
| | Buy **top-5 only**, top-15 → watch | ranks 6-15 contributed **0/5** winners at half the conviction |
| **Burst** | Add **market-regime features** (skip recalibration) | model is regime-blind: hit rate 5.5% → **7.6%** with up-momentum, but predicted prob stays flat |
| | Keep **v4** as primary universe | best hit rate + only predictive regressor |
| **Drop** | **Re-specify the target** to net downside | economic spread (flagged − safe, fwd 5d) **+0.39% → −1.29% / −2.43%** (correct sign) |

## 3. The honest part: validation overturned 2 of 3 first instincts

| First instinct | What the backtest showed | Corrected fix |
|---|---|---|
| 21d: lock in at +30% | a **full** lock-in *loses* money (+13.4% → +9.8%) by capping the fat tail | sell **half**, let half run |
| Burst: recalibrate probabilities | base model is **already calibrated** (test ECE 1.5%) | add **regime features** instead |
| Drop: add regime features | didn't help (AUC 0.724 → 0.711, economics still wrong-signed) | **re-specify the target** |

For drop, re-specifying from "transient −3%/day window" to **net 5-day downside**
flips the economic spread negative (flagged names finally underperform), at the cost
of a lower but more honest AUC (~0.62–0.65) — predicting *direction* is harder than
predicting *volatility*, but only direction correlates with money.

> Caveats: the 21-day exit sample is small (one bull window); the drop re-spec was
> tested only on a bull tape and needs down-tape validation before production use.
