# 21-Day Predictive Model

A gradient-boosted classifier that ranks U.S. stocks by their probability of touching a **+30% return inside the next 21 trading days**, plus the live paper/real execution layer built around it.

> **Disclaimer.** Research code, not financial advice. Past performance — including everything in this repo — is no guarantee of future results. Predictions are model probabilities, not recommendations. Do not trade off this without doing your own work.

---

## ⚠️ Status: read this before the numbers

**A revalidation in August 2026 found no demonstrated edge in this model.** The headline backtest figures further down (AUC 0.893, 14.2× precision lift, +12.4% top-5 return) are preserved for the record, but they did **not** survive proper controls. What the corrected work found:

| finding | number |
|---|---|
| The model is, mechanically, a **volatility sort** | `df.nlargest(5, 'atr_pct')` reproduces its picks |
| That volatility sort, corrected for survivorship + look-ahead | **−1.59pp** |
| Live top-5, graded against realized outcomes | **−4.82%** vs **+2.68%** for the index over the same dates |
| The model's own confidence is **anti**-predictive | high-conviction picks touched +30% **3.9%** of the time vs **17.6%** for low-conviction |

Two biases account for most of the gap between the backtest and live results:

- **Survivorship.** The universe was built from the *current* S&P 500. 61 index-leavers were missing from the panel — and they sat at the 87th percentile of volatility with a −35% median return, i.e. exactly the names a volatility sort would have picked and lost on. A point-in-time membership table is required to backtest this honestly.
- **Look-ahead.** Features and universe construction both peeked at information not available on the decision date.

There is also a **statistical power** problem independent of the biases: at K=5 picks per day, distinguishing this strategy's effect size from zero would take roughly **23 years** of data. Two months and ~25 trades cannot settle it, and several results found in that search *reversed* when properly controlled.

[`ANALYSIS.md`](ANALYSIS.md) predates this work (June 2026) and its 21-day-gainer section — "hits its number", "top-5 picks hit 58%" — is superseded by the above. It is kept as a record of what was believed at the time.

**Treat this repo as a worked example of a model that looked good and wasn't**, and as the plumbing for testing such a thing honestly. That is its actual value.

---

## What it predicts

For each `(ticker, date)` in the universe:

```
y21 = 1 if max(close[t+1 .. t+21]) / close[t] >= 1.30 else 0
```

i.e. "does this stock at any point *touch* +30% within the next 21 trading days." Lenient touch definition, not a hold-to-end-of-window return. Note that a *touch* target is satisfied by volatility alone — a name that spikes +30% and round-trips to flat is a positive label. This is the design decision most responsible for the model learning a volatility detector.

## Original held-out backtest — superseded, kept for the record

Model: `GradientBoostingClassifier(n_estimators=400, max_depth=3, lr=0.03, subsample=0.8)`, isotonic-calibrated on val. Test fold = Nov 2025 → Mar 2026 (92 trading days, 2.07% positive rate — strong-bull regime).

| metric | value | base rate | lift |
|---|---:|---:|---:|
| AUC | 0.893 | — | — |
| PR-AUC | 0.174 | 0.0207 | 8.4× |
| Precision @ top-5 per day | 29.3% | 2.07% | 14.2× |
| Precision @ top-10 per day | 24.6% | 2.07% | 11.9× |
| Long-only sim: top-5, end-of-21d ret | +12.4% | +1.5% (univ avg) | +10.9 pp |
| Long-only sim: top-5, max-in-window ret | +21.7% | +7.2% | +14.5 pp |

**These numbers are not reproducible under a point-in-time universe.** They are what the model scored before the survivorship and look-ahead corrections described above. Full original investigation and ablations in [REPORT.md](REPORT.md).

## What the model actually learned

- ~68% of feature importance is `rv_60 + atr_pct` — it is a **high-vol stock detector** with a thin timing layer on top.
- Within the top vol quartile (`rv_60` Q4) the stratified AUC was 0.786, so there *is* some within-bucket ordering. It is not large enough to overcome the costs of trading it.
- For low/mid-vol stocks (Q1–Q3) and Real Estate, the signal is noise.
- ~23% of fresh +30% events have **no detectable pre-event catalyst** (price, news, sector, earnings) — a hard floor on any price-only model.
- **Concentration:** 33% of all positives come from 10 tickers (CVNA, APP, COIN, SMCI, HOOD, LITE, SNDK, PLTR, VRT, COHR). The model "succeeds" partly by re-picking these.

## Features (37 total)

- **16 base** price/vol: `rsi_14, macd, macd_sig, macd_hist, bb_z20, atr_pct, range_pct, vol_z, vol_5d, rv_60, ma_stack, up_streak, up_bigdays_20d, dist_ma60_atr, ma60_slope_60d, run_length`
- **7 regime**: `spy_ret_5d, spy_ret_20d, spy_rv_20, spy_rv_60, vix, vix_chg_5d, fng`
- **10 catalyst** (v2): FinBERT news-spike features + earnings/M&A keyword counts + same-sector co-move
- **4 cross-sectional ranks** (v3): universe-wide percentile of `rsi_14`, `rv_60`, `ma60_slope_60d`, trailing-20d return

`overnight_gap` is excluded as forward-looking. Max `|Spearman(feature, y21)|` on train = 0.149.

## Live execution layer

The model is wired to a real brokerage account through a two-stage daily pair. This is the part of the repo most likely to be useful to someone else, because the *timing* results below are measured and hold regardless of whether the model has an edge.

```
15:45 ET  com.user.mgcloseplan  →  code/123_mg_close_scorer.py
          rescore on the near-complete session, apply the extension
          indicator and exit rules, write trade_plans/latest.{txt,json},
          push the plan to Telegram.  Places no orders.

15:55 ET  com.user.mgexecute    →  daily/run_mg_execute.sh
          read latest.json, submit exactly those orders via the Robinhood
          MCP connector, reconcile fills with 124_mg_reconcile.py,
          send one execution receipt.  Places real orders.
```

**Why 15:45 / 15:55 and not the next morning.** These picks gap overnight. Measured over 18,142 ticker-days:

| fill point | slip vs official close | sd |
|---|---:|---:|
| 15:55 same day | **−0.04%** | 0.39% |
| 09:30 next day | +0.70% | 3.47% |

Scoring ten minutes early costs almost nothing — the 15:45 provisional bar agrees with the true close on **97.7%** of FAVOURABLE/not calls and **94.4%** of extension-flag counts. So the decision is stable across that window while the fill quality is not. Dollar-based fractional orders are regular-hours only, which is what pins the whole thing to before 16:00.

**Trading days only.** [`code/125_trading_day.py`](code/125_trading_day.py) is a dependency-free NYSE calendar — weekends, the ten full holidays (including Good Friday, which is not a federal holiday), and the 13:00 ET early closes, which matter here because the 15:45/15:55 pair would otherwise run *after* the close. Unscheduled closures are caught separately: the executor also requires the plan's `asof` date (the freshest market bar) to be today, so a day with no bar stands the system down.

**Order entry is a Claude Code session, not a script.** Robinhood exposes no scriptable client here; the connector is an MCP server reachable from an agent session, so the session *is* the order-entry client. [`daily/run_mg_execute.sh`](daily/run_mg_execute.sh) runs `claude -p` with a fixed on-disk prompt ([`daily/mg_execute_prompt.md`](daily/mg_execute_prompt.md)) and a narrow `--allowedTools` list. In `-p` mode anything unlisted is denied outright with no prompt, so that list — four Robinhood endpoints and two local scripts — is the real safety boundary.

**Encoded rules.** Entry: top-5 by `raw_margin`, first appearance of that ticker only, extension indicator FAVOURABLE (0 of 3 flags), free slot. Exit: +30% take profit, or out of the top-15 for 2 consecutive publication days, or 21 trading days. Sizing: 8 equal slots, fractional dollar orders. (+30% rather than +12%: the lower cap only won under an unrealistic same-close entry; under every realistic fill from 09:30 to 15:55, TP30 > TP20 > TP12.)

**To run it yourself** you need `config/mg_execution.json` (see [`config/mg_execution.example.json`](config/mg_execution.example.json)) naming the brokerage account, and `config/burst_daily.json` with Telegram credentials. Both are untracked. **You should not run it.** See the status section — this automates a signal with no demonstrated edge.

## Repo layout

```
21_day_pred_model/
├── README.md / REPORT.md / ANALYSIS.md / data_schema.md
├── code/
│   ├── 80_..._panel.py, 85_..._panel.py     ← labeled panels
│   ├── 87_catalyst_live_features.py
│   ├── 93_train_v3.py, 110_train_v3_universe_blind.py
│   ├── 101_refresh_score_today.py           ← daily rescore
│   ├── 102_sunday_check.py, 104_daily_exit_monitor.py
│   ├── 121_mg_trade_plan.py                 ← paper plan generator
│   ├── 123_mg_close_scorer.py               ← 15:45 same-day scorer  ★
│   ├── 124_mg_reconcile.py                  ← fills → position state ★
│   ├── 125_trading_day.py                   ← NYSE calendar guard    ★
│   ├── mg_account.py, notify.py             ← account + Telegram     ★
│   ├── 10..79_burst_*, 36..69_drop_*        ← sibling burst/drop models
│   ├── regime_features.py, extension_classifier.py
│   └── score_new_data.py                    ← self-contained inference example
├── daily/
│   ├── run_mg_execute.sh                    ← 15:55 executor          ★
│   └── mg_execute_prompt.md                 ← its fixed prompt        ★
├── launchd/                                 ← the two schedules       ★
├── models/          monthly_gainer_v3_{sp500,smallcap,combined}.joblib
├── config/          *.example.json only (real config is untracked)
├── docs/            per-system READMEs (burst_morning, drop_alert)
└── recent_picks/    scored output snapshots
```

★ = the execution layer, added 2026-09.

## Using a trained model on new data

Each `.joblib` is a **dict** bundling the calibrator, raw GBC, feature order, and training-set medians:

```python
import joblib, pandas as pd

bundle = joblib.load("models/monthly_gainer_v3_combined.joblib")
feats = bundle["feats"]                     # 37 feature names in model order
medians = bundle["impute_medians"]

X = pd.read_csv("your_features.csv")
for col in feats:
    if col not in X.columns:
        X[col] = medians[col]
X = X[feats].fillna(value=medians)

prob = bundle["calibrator"].predict_proba(X)[:, 1]
```

Remember that `prob` is anti-predictive in live use (see the status section) — higher is not better.

- [data_schema.md](data_schema.md) — the exact 37 features, dtypes, formulas, and required raw data.
- [`code/score_new_data.py`](code/score_new_data.py) — self-contained end-to-end example: pulls yfinance + ^VIX, computes all 37 features, prints top picks over an example 50-ticker universe. `pip install -r requirements.txt` and run.

## Training from scratch

Assuming you have `data/burst_panel_v8.csv`, `data/burst_universe_v7.csv`, and a Finnhub news cache + FinBERT scores:

```bash
python code/80_monthly_gainer_panel.py        # SP500 labeled panel
python code/84_monthly_gainer_smallcap_universe.py
python code/85_monthly_gainer_smallcap_panel.py
python code/87_catalyst_live_features.py      # catalyst features
python code/93_train_v3.py sp500              # → models/monthly_gainer_v3_sp500.joblib
python code/93_train_v3.py smallcap
python code/110_train_v3_universe_blind.py    # → ..._combined.joblib
python code/111_combined_predictions.py       # → recent_picks/combined_top25.csv
```

**If you are rebuilding this, fix the universe first.** Build a point-in-time membership table and include index leavers before training anything, or you will reproduce the survivorship result above.

`83_monthly_gainer_train.py` (v1, 23 features), `88_..._v2.py` (v2, 33 features), and `92_train_drop15.py` are kept for context.

## Honest caveats

- **No demonstrated edge.** See the status section. Everything below is secondary to that.
- **Test fold is bull-regime.** 2025-Q4 → 2026 base rate = 2.07% vs long-run 1.28%.
- **Survivorship bias in the published numbers.** Universe = *current* S&P 500 + current Robinhood-tradable smallcaps; 61 index leavers are missing and they were high-vol, deeply negative names.
- **Underpowered.** K=5 needs ~23 years to separate from zero. Two months of live trading proves nothing either way.
- **Same-name recurrence.** Top-5 picks change slowly because features change slowly — a candidate screener, not an entry timer.
- **Real Estate / Cons. Staples / low-vol large-caps:** no useful signal.
- **The execution layer places real orders.** It is scheduled and unattended. Read `daily/run_mg_execute.sh` in full before enabling anything.

See [REPORT.md §5](REPORT.md) for the original assessment, including the ~0.92–0.94 AUC ceiling estimate from catalyst-attribution analysis — itself computed pre-correction.

## License

No license attached — all rights reserved. If you want to use this for anything beyond reading, open an issue.
