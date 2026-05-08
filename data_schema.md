# Data Schema

This document specifies the exact inputs the trained models expect, so you can run them on new data without rebuilding the upstream pipeline.

There are 37 features. They split into four groups by what raw data they need:

| Group | Count | Raw input needed | Required for inference? |
|---|---:|---|---|
| Base price/volume | 16 | OHLCV daily bars per ticker | **Yes** |
| Regime (universe-wide) | 7 | SPY + VIX + Fear & Greed daily | Recommended (model imputes median if missing) |
| Catalyst | 10 | News headlines + FinBERT sentiment | Optional (zeros are a valid default — they just leave the model with no news signal) |
| Cross-sectional rank (x-rank) | 4 | Same OHLCV but with **multiple tickers on the same date** | Recommended; degenerates for a single stock |

The model will accept any subset and impute missing values with `bundle["impute_medians"]`. But the further you stray from the full feature set, the more the model is just predicting from defaults.

---

## Joblib bundle layout

Each `.joblib` is a **dict**, not a model directly. Load like this:

```python
import joblib

bundle = joblib.load("models/monthly_gainer_v3_combined.joblib")

bundle["feats"]               # list[str], 37 feature names in model order
bundle["calibrator"]          # CalibratedClassifierCV — the thing that scores
bundle["raw_gbc"]             # the underlying GradientBoostingClassifier (uncalibrated)
bundle["impute_medians"]      # dict[feat -> float], training-set medians for missing values
bundle["v1_features"]         # list[str], 23 features (base 16 + regime 7)
bundle["catalyst_features"]   # list[str], 10
bundle["xrank_features"]      # list[str], 4
bundle["feature_importance"]  # dict[feat -> float], summing to 1.0
bundle["metrics"]             # dict, training-time metrics (auc, ap, p_at_top5, ...)
bundle["universe"]            # str: "sp500" | "smallcap" | "combined"
```

Score new data:

```python
import joblib, pandas as pd

bundle = joblib.load("models/monthly_gainer_v3_combined.joblib")
feats = bundle["feats"]
medians = bundle["impute_medians"]

X = pd.read_csv("your_features.csv")
for col in feats:
    if col not in X.columns:
        X[col] = medians[col]               # add missing columns
X = X[feats].fillna(value=medians)          # fill NaNs

prob = bundle["calibrator"].predict_proba(X)[:, 1]
# prob[i] = calibrated P(ticker_i touches +30% within next 21 trading days)
```

Three trained bundles are included:

| File | Universe trained on | Test base rate | When to use |
|---|---|---:|---|
| `monthly_gainer_v3_sp500.joblib` | S&P 500 only (~503 tickers) | 2.07% | If you're scoring large-cap U.S. stocks |
| `monthly_gainer_v3_smallcap.joblib` | Smallcap (1,388 tickers) | ~7% | If you're scoring smaller / more volatile names |
| `monthly_gainer_v3_combined.joblib` | Both, universe-blind (1,891 tickers) | 4.69% | Default — handles mid-caps that fall through the SP500/smallcap split |

The probabilities are calibrated against each training fold's base rate, so they're not directly comparable across the three models. Within a single model, higher = more likely.

---

## Feature definitions

### A. Base price/volume (16 features)

All computed from a single ticker's OHLCV history. Implemented in `code/30_burst_v7_pipeline.py` and `code/35_burst_v8_trend_features.py`. All values are at the close of trading day `t` and use only data through `t` (no leakage).

| Feature | Dtype | Typical range | Definition |
|---|---|---|---|
| `rsi_14` | float | 0–100 | Wilder RSI on close, 14 periods |
| `macd` | float | ~ ±0.05 | (EMA₁₂(close) − EMA₂₆(close)) / close |
| `macd_sig` | float | ~ ±0.05 | EMA₉(macd) |
| `macd_hist` | float | ~ ±0.02 | macd − macd_sig |
| `bb_z20` | float | ~ ±3 | (close − rolling20mean(close)) / rolling20std(close) |
| `atr_pct` | float | 0.005–0.20 | ATR(14) / close, where ATR uses true range = max(H−L, \|H−PrevClose\|, \|L−PrevClose\|) |
| `range_pct` | float | 0–0.20 | (high − low) / close, intraday range |
| `vol_z` | float | ~ ±5 | (volume − rolling30mean(volume)) / rolling30std(volume) |
| `vol_5d` | float | ~0–5 | rolling5mean(volume) / rolling30mean(volume) |
| `rv_60` | float | 0.05–2.0 | std(daily_return, 60) × √252, annualized realized vol |
| `ma_stack` | int (0/1) | 0 or 1 | 1 if MA₅(close) > MA₂₀(close) > MA₆₀(close), else 0 |
| `up_streak` | int | 0–30 | Consecutive up-days streak (capped at 30) |
| `up_bigdays_20d` | int | 0–20 | Count of daily returns > +3% in trailing 20 trading days |
| `dist_ma60_atr` | float | ~ ±10 | (close − MA₆₀(close)) / ATR(14), price extension in ATR units |
| `ma60_slope_60d` | float | ~ ±0.5 | (MA₆₀[t] − MA₆₀[t−60]) / close, persistence of trend |
| `run_length` | int | 0–120 | Consecutive days closing > MA₂₀ (capped at 120) |

You need at least **120 trading days of history** before features are warm. The 60-period ones (`rv_60`, `ma60_slope_60d`) are NaN before then.

### B. Regime (7 features, universe-wide per date)

These are the same value for every ticker on a given date. Implemented in `code/regime_features.py`. They need three external daily series:

| Source | What | Where to get it |
|---|---|---|
| SPY ETF close | for `spy_ret_5d`, `spy_ret_20d`, `spy_rv_20`, `spy_rv_60` | yfinance: `yf.Ticker("SPY").history(...)` |
| ^VIX close | for `vix`, `vix_chg_5d` | yfinance: `yf.Ticker("^VIX").history(...)` |
| CNN Fear & Greed | for `fng` | scrape https://production.dataviz.cnn.io/index/fearandgreed/graphdata or skip and use median (51.4) |

| Feature | Dtype | Typical range | Definition |
|---|---|---|---|
| `spy_ret_5d` | float | ~ ±0.10 | SPY.close.pct_change(5) |
| `spy_ret_20d` | float | ~ ±0.20 | SPY.close.pct_change(20) |
| `spy_rv_20` | float | 0.05–0.5 | std(SPY return, 20) × √252 |
| `spy_rv_60` | float | 0.08–0.5 | std(SPY return, 60) × √252 |
| `vix` | float | 10–60 | Daily VIX close |
| `vix_chg_5d` | float | ~ ±10 | vix[t] − vix[t−5] |
| `fng` | float | 0–100 | CNN Fear & Greed index, 0=extreme fear, 100=extreme greed |

The model's training-set median for `fng` is 51.4 (basically neutral). If you skip the CNN scrape, that default is fine — `fng` only carries 0.012 of total feature importance.

### C. Catalyst (10 features)

These need a per-ticker news cache + FinBERT sentiment scores. Computed in `code/87_catalyst_live_features.py`. **All ten are strictly causal** (look at `[t−5..t−1]` or `[t−20..t−1]`, never `[t..]`).

| Feature | Dtype | Definition |
|---|---|---|
| `finbert_max_5d` | float, ~ −1..+1 | max(daily FinBERT max sentiment) over `[t−5, t−1]` |
| `finbert_max_20d` | float, ~ −1..+1 | same, 20-day window |
| `finbert_mean_5d` | float, ~ −1..+1 | mean of daily FinBERT mean sentiment over `[t−5, t−1]` |
| `news_n_5d` | int | total article count in `[t−5, t−1]` |
| `news_n_20d` | int | total article count in `[t−20, t−1]` |
| `earn_news_5d` | int (0/1) | 1 if any earnings-keyword headline in `[t−5, t−1]` |
| `earn_news_20d` | int (0/1) | same, 20-day window |
| `ma_news_5d` | int (0/1) | 1 if any M&A/takeover-keyword headline in `[t−5, t−1]` |
| `ma_news_20d` | int (0/1) | same, 20-day window |
| `sector_pop_5d` | int | count of same-sector tickers with `close[t−1] / close[t−6] − 1 ≥ 10%` |

**Easy default**: set all ten to **0**. The model's training-set medians are all `0.0` for these features (most ticker-days have no news), so this is what the model expects when news is absent. You'll lose ~5–10% of model lift on news-driven names but the rest of the signal still works.

To do it properly: see `code/87_catalyst_live_features.py` for keyword regexes and aggregation logic. You need (per ticker) a daily series of `(finbert_max, finbert_mean, finbert_n, headline_text)` and a sector mapping for the universe.

### D. Cross-sectional ranks (4 features)

Computed across the **whole universe on each date**. Implemented in `code/93_train_v3.py:add_xrank_features`. These are crucial — they carry ~30% of feature importance in v3 (`rv_60_xrank` alone is 29.8%).

| Feature | Dtype | Range | Definition |
|---|---|---|---|
| `rsi_14_xrank` | float | 0–1 | percentile rank of `rsi_14` across all tickers that date |
| `rv_60_xrank` | float | 0–1 | percentile rank of `rv_60` across all tickers that date |
| `ma60_slope_xrank` | float | 0–1 | percentile rank of `ma60_slope_60d` across all tickers that date |
| `ret_20d_xrank` | float | 0–1 | percentile rank of trailing 20d return (`close[t]/close[t−20] − 1`) across all tickers that date |

Computed as `df.groupby("date")[col].rank(pct=True)`.

**Caveat**: x-ranks need a real cross-section. With a single ticker, every rank is `0.5` and the model degenerates. Use **at least 50 liquid tickers** per scoring date to get a meaningful cross-section. The v3 combined model was trained against a 1,891-ticker universe; the smaller your live universe, the more the rank features drift from training-time semantics. Falling back to the median (0.5) is what the imputer does and is reasonable when the cross-section is too thin.

---

## Putting it together

The expected DataFrame for `predict_proba` is, schematically:

```
columns: ticker, date, [37 feature columns], (any extras you carry along)
dtype:   str,    datetime, all float (ints OK — sklearn coerces)
shape:   (n_tickers × n_dates, 39+)
```

Order doesn't matter as long as you select `X[bundle["feats"]]` before scoring. Index doesn't matter. NaN is OK if you `.fillna(bundle["impute_medians"])` first.

A complete worked example — pulling yfinance data, computing all features (catalyst defaulted to zeros), and scoring the model — is in [`code/score_new_data.py`](code/score_new_data.py). Run it as `python code/score_new_data.py` to get top picks for today across a small example universe.

---

## What's NOT included in this repo

- `data/burst_panel_v8.csv` — the precomputed feature panel (every base feature × every ticker × every date). Build it with `code/35_burst_v8_trend_features.py` style logic on top of yfinance bulk downloads.
- `data/sp500_daily.csv`, `data/vix_daily.csv`, `data/fear_greed.csv` — the regime input series. Build with yfinance and (optionally) the CNN F&G endpoint.
- `data/finnhub_news/{TICKER}.jsonl` — raw news cache. Get from Finnhub, Polygon, or any news API; one line per article with `{"datetime": unix_ts, "headline": "..."}`.
- `data/finbert_scores.csv` — FinBERT sentiment scored over the news. Run [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) on each headline; aggregate to daily `(ticker, date, finbert_max, finbert_mean, finbert_n)`.
- `data/burst_universe_v7.csv`, `data/monthly_gainer_panel*.csv`, `data/catalyst_features_*.csv` — derived intermediates produced by the pipeline scripts in `code/`.

If you only want **inference on new data**, you don't need any of these — you just need to compute the 37 features for whatever tickers/dates you care about. See `code/score_new_data.py` for a runnable end-to-end example.
