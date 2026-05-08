# 21-Day Predictive Model

A gradient-boosted classifier that ranks U.S. stocks by their probability of touching a **+30% return inside the next 21 trading days**. Trained on ~3 years of daily price/volume + cross-sectional rank + regime features, evaluated chronologically on a held-out 2025-Q4 → 2026-Q1 fold.

> **Disclaimer.** Research code, not financial advice. Past performance — including everything in this repo — is no guarantee of future results. Predictions are model probabilities, not recommendations. Survivorship bias is present. Do not trade off this without doing your own work.

---

## What it predicts

For each `(ticker, date)` in the universe:

```
y21 = 1 if max(close[t+1 .. t+21]) / close[t] >= 1.30 else 0
```

i.e. "does this stock at any point *touch* +30% within the next 21 trading days." Lenient touch definition, not a hold-to-end-of-window return.

## Headline numbers (held-out test fold)

Model: `GradientBoostingClassifier(n_estimators=400, max_depth=3, lr=0.03, subsample=0.8)`, isotonic-calibrated on val. Test fold = Nov 2025 → Mar 2026 (92 trading days, 2.07% positive rate — strong-bull regime).

| metric | value | base rate | lift |
|---|---:|---:|---:|
| AUC | **0.893** | — | — |
| PR-AUC | **0.174** | 0.0207 | **8.4×** |
| Precision @ top-5 per day | **29.3%** | 2.07% | **14.2×** |
| Precision @ top-10 per day | 24.6% | 2.07% | 11.9× |
| Long-only sim: top-5, end-of-21d ret | **+12.4%** | +1.5% (univ avg) | +10.9 pp |
| Long-only sim: top-5, max-in-window ret | +21.7% | +7.2% | +14.5 pp |

**Smallcap variant** (1,388-ticker broader universe) — lower lift but higher absolute hit rate: precision @ top-5 = **49.5%**, end-of-window top-5 return **+29.3%**.

Full investigation, ablations, and honest assessment in [REPORT.md](REPORT.md).

## Why it works (and where it doesn't)

- ~68% of feature importance is `rv_60 + atr_pct` — the model is fundamentally a **high-vol stock detector + within-vol-bucket timing layer**.
- Within the top vol quartile (`rv_60` Q4) it adds real information: stratified AUC = 0.786.
- For low/mid-vol stocks (Q1–Q3) and Real Estate, signal is essentially noise.
- ~23% of fresh +30% events have **no detectable pre-event catalyst** (price, news, sector, earnings) — a hard floor on what any price-only model can do.

## Features (37 total)

- **16 base** price/vol: `rsi_14, macd, macd_sig, macd_hist, bb_z20, atr_pct, range_pct, vol_z, vol_5d, rv_60, ma_stack, up_streak, up_bigdays_20d, dist_ma60_atr, ma60_slope_60d, run_length`
- **7 regime**: `spy_ret_5d, spy_ret_20d, spy_rv_20, spy_rv_60, vix, vix_chg_5d, fng`
- **10 catalyst** (v2): FinBERT news-spike features + earnings/M&A keyword counts + same-sector co-move
- **4 cross-sectional ranks** (v3): universe-wide percentile of `rsi_14`, `rv_60`, `ma60_slope_60d`, trailing-20d return — captures "stands out vs peers today"

`overnight_gap` is excluded as forward-looking. Max `|Spearman(feature, y21)|` on train = 0.149 (clean, no leakage).

## Repo layout

```
21_day_pred_model/
├── README.md                  ← this file
├── REPORT.md                  ← full investigation (504 lines, all ablations)
├── code/                      ← 32 pipeline scripts + 2 helpers
│   ├── 80_..._panel.py        ← build labeled SP500 panel
│   ├── 84_..._smallcap_universe.py + 85_..._panel.py
│   ├── 87_catalyst_live_features.py
│   ├── 93_train_v3.py         ← v3 trainer (per-universe)
│   ├── 110_train_v3_universe_blind.py  ← v3 combined trainer
│   ├── 111_combined_predictions.py     ← produces top-25
│   ├── 89_predict_today.py / 101_refresh_score_today.py
│   ├── 102_sunday_check.py / 104_daily_exit_monitor.py
│   ├── regime_features.py     ← SPY/VIX/FNG attach helper
│   └── extension_classifier.py ← "is this run already extended?" filter
├── models/
│   ├── monthly_gainer_v3_sp500.joblib    (757 KB)
│   ├── monthly_gainer_v3_smallcap.joblib (764 KB)
│   └── monthly_gainer_v3_combined.joblib (762 KB)  ← universe-blind
└── recent_picks/
    ├── combined_top25.csv               ← top-25 picks across full universe
    ├── about_to_rise_{sp500,smallcap}_full.csv
    ├── hold_until_signal_drops_{sp500,smallcap}.csv
    ├── exit_monitor_latest.md
    ├── sunday_verdict_latest.md
    ├── model_metrics.json / smallcap_metrics.json
    ├── baserate.json / baserate_table.md
    └── multi_horizon.json
```

## Using a trained model

The `.joblib` file is a `CalibratedClassifierCV` wrapping a `GradientBoostingClassifier`. Given a feature matrix in the right column order:

```python
import joblib
import pandas as pd

model = joblib.load("models/monthly_gainer_v3_combined.joblib")
# model.feature_names_in_ holds the expected feature order

X = pd.read_csv("your_features.csv")[model.feature_names_in_]
prob = model.predict_proba(X)[:, 1]   # P(touch +30% within 21 trading days)
```

To reproduce features end-to-end you also need the upstream daily price/volume panel (`burst_panel_v8`-shaped: OHLCV + the 16 base features per ticker-day) plus FinBERT-scored news. Those upstream feeds are not in this repo — building them is its own pipeline.

## Training from scratch

Assuming you have `data/burst_panel_v8.csv`, `data/burst_universe_v7.csv`, and a Finnhub news cache + FinBERT scores:

```bash
python code/80_monthly_gainer_panel.py        # SP500 labeled panel
python code/84_monthly_gainer_smallcap_universe.py
python code/85_monthly_gainer_smallcap_panel.py
python code/87_catalyst_live_features.py      # build catalyst features
python code/93_train_v3.py sp500              # → models/monthly_gainer_v3_sp500.joblib
python code/93_train_v3.py smallcap           # → models/monthly_gainer_v3_smallcap.joblib
python code/110_train_v3_universe_blind.py    # → models/monthly_gainer_v3_combined.joblib
python code/111_combined_predictions.py       # → recent_picks/combined_top25.csv
```

`code/83_monthly_gainer_train.py` (v1, 23 features), `88_monthly_gainer_train_v2.py` (v2, 33 features), and `92_train_drop15.py` (the −15%-drop sibling head) are kept for context.

## Honest caveats

- **Test fold is bull-regime.** 2025-Q4 → 2026 base rate = 2.07% vs long-run 1.28%. Numbers in flat/bear conditions would be lower.
- **Survivorship bias.** Universe = current S&P 500 + current Robinhood-tradable smallcaps. Delisted takeover targets — which historically had lots of +30% pops — are excluded.
- **Concentration.** 33% of all positives come from 10 tickers (CVNA, APP, COIN, SMCI, HOOD, LITE, SNDK, PLTR, VRT, COHR). The model "succeeds" partly by re-picking these.
- **Same-name recurrence.** Top-5 picks change slowly day-to-day because features change slowly — this is a candidate-screener, not an entry-timer.
- **Real Estate / Cons. Staples / low-vol large-caps:** essentially no useful signal. Skip those buckets.

See [REPORT.md §5](REPORT.md) for the full assessment, including the ~0.92–0.94 AUC ceiling estimate from catalyst-attribution analysis.

## License

No license attached — all rights reserved. If you want to use this for anything beyond reading, open an issue.
