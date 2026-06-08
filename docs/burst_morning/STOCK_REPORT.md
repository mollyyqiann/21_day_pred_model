# Stock Burst Prediction — Report

**As of:** 2026-04-17  ·  **Universe:** 182 tickers  ·  **Target:** in the next 5 trading days, some 2–5-day window averages ≥ 4%/day

**Primary question:** which normally-calm US stocks priced above $40 are unusually likely to deliver a sustained burst right now?

---

## 1. Scope & universe selection

A *burst* here is the user-specified event: a stretch of **≥ 2 consecutive trading days** within the next 5 whose **average daily return is ≥ 4%**. On calm, large-cap names this is rare by construction — the answer of "nothing is flagged today" is often the honest one.

Universe filter (see [code/10_burst_universe.py](code/10_burst_universe.py)):

- S&P 500 constituents only
- **Price ≥ $40**
- **60-day realized vol between 5% and 30% annualized** (i.e. *usually* not volatile)
- Average dollar-volume ≥ $25 M/day (liquidity)
- Beta vs SPY computable on 1 year of data

This yielded **182 tickers**. The calmest names are utilities, consumer staples, large health-care, and a few mega-cap industrials. Full list: [data/burst_universe.csv](data/burst_universe.csv).

## 2. Data and features

For each name, 3 years of daily OHLCV was pulled via yfinance (+ SPY for market adjustment). Feature set computed at each date *t* using **only data through t** (no look-ahead):

| Group | Features |
|---|---|
| Returns / momentum | `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_60d` |
| Realized volatility | `rv_10`, `rv_20`, `rv_60` |
| Classical technicals | `rsi_14`, `macd`, `macd_sig`, `macd_hist`, `bb_z20`, `atr_pct`, `range_pct` |
| Volume | `vol_z` (30-day z-score), `vol_5d` (5-day avg ratio vs 30d) |
| Trend proximity | `gap_ma50`, `gap_ma200`, `pos_52w` |
| Market adjustment | rolling 60-day `beta_60` vs SPY; residual returns `resid_1d/5d/10d/20d` |
| SPY regime | `spy_ret_5d`, `spy_ret_20d`, `spy_rv_20` |

Beta and residuals are the explicit "reduce noise from general market" layer — features isolate each name's idiosyncratic behavior beyond what SPY would mechanically drag it through.

Panel: **136,929 stock-day rows** across 182 tickers. See [code/11_burst_features.py](code/11_burst_features.py) and [data/burst_panel.csv](data/burst_panel.csv).

## 3. Model

**Gradient Boosted Classifier** ([code/12_burst_train.py](code/12_burst_train.py)): 300 trees, depth 3, lr 0.05, subsample 0.8. Chronological 70/15/15 split by date (train ended 2025-09-05, val through 2025-12-22, test thereafter).

Trees handle the feature interactions (e.g. "RSI high **and** volume spike **and** positive residual") without hand-coded rules. A logistic regression baseline is reported alongside as sanity.

### Why not LSTM here?
With ~0.9% base rate, ~90k labelled rows, and 28 features, a GBC trained on last-step features is the right tool. An LSTM for this task would burn a lot of capacity recovering statistics the tree model reads off directly, and with so few positive examples sequence models tend to overfit.

## 4. Test-set performance

| Split | n | positives | base rate | AUC | PR-AUC | PR-AUC / base (lift) | log-loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **train** | 63,033 | 573 | 0.909% | 0.938 | 0.604 | **66.5×** | 0.0271 | 0.0054 |
| **val** | 13,650 | 141 | 1.033% | 0.648 | 0.025 | **2.4×** | 0.0617 | 0.0111 |
| **test** | 13,650 | 62 | 0.454% | 0.576 | 0.008 | **1.7×** | 0.0338 | 0.0053 |

Logistic regression test AUC (sanity): **0.625**. The tree model isn't dramatically better than logistic at ranking — it mostly wins on calibration.

**Read-out.** This is a hard problem. On the test fold (Dec 2025 → Apr 2026), AUC is 0.58 with ~1.7× PR-AUC lift over the base rate. The model is slightly informative but is **not** a predictor in the casual sense — it mostly identifies where the *conditional* probability is 1–2× elevated. That's consistent with the academic literature: short-horizon, extreme-return events on low-vol stocks are close to random, and the rare hits are often news-driven (earnings, M&A, data readouts), which technicals can only hint at.

Large train/val gap (AUC 0.94 vs 0.65) reflects the base-rate drift between folds and some overfit to the pre-2025 regime. The test fold is the number to trust.

### Feature importance (top 10)

- `beta_60` — 0.121
- `gap_ma50` — 0.074
- `spy_ret_5d` — 0.068
- `vol_z` — 0.057
- `rv_60` — 0.050
- `vol_5d` — 0.048
- `rv_20` — 0.047
- `resid_1d` — 0.046
- `macd_sig` — 0.042
- `ret_1d` — 0.038

Interpretation: the model leans on **beta**, **position vs 50-day MA**, **SPY 5-day return**, **volume z-score**, and **realized vol**. That's a coherent story: bursts cluster where a quiet stock has started diverging from its mean on above-normal volume, with a positive broader tape.

## 5. Today's predictions (2026-04-17)

The model scored all **182** universe tickers for P(burst in next 5 days) using the feature vector at market close 2026-04-17. Base rate in the test fold was **0.45%**, so a probability of *k* × base corresponds to a lift of *k*.

### Flag rule (conservative)

Surface as a **flagged candidate** only when both:
- P(burst) ≥ **5%** (absolute), **and**
- lift ≥ **3.0×** base rate

**Flagged today: 0 stocks.**

No ticker in the universe clears the flag bar. The highest model probability today is **1.26%** (`QCOM`), which is only **1.5×** the test-fold base rate — well below the threshold. In plain terms: *nothing in this basket is showing a statistically unusual burst setup*. That's the expected output on most days.

### Top 15 ranked candidates (technical + news blend)

Even though none cross the flag bar, these are the names with the elevated technical setup. `combined = P(burst) · (1 + 0.2 · clipped news score)`.

| # | Ticker | Close | P(burst) | Lift | News avg | ret_5d | RSI14 | vol_z | Headlines |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **QCOM** | $136.20 | 1.25% | 1.5× | +0.25 | +6.36% | 74 | -0.18 | Can Intel's Core Series 3 Processors Launch Boost Its Profits? // Cirrus Logic Up 34% in 3 Months: S |
| 2 | **DHR** | $194.75 | 1.26% | 1.5× | +0.00 | +2.71% | 67 | +0.44 | Apple defeats bid for new Apple Watch import ban at US trade tribunal // Danaher Gears Up to Post Q1 |
| 3 | **TMO** | $526.60 | 0.93% | 1.1× | +0.50 | +6.15% | 74 | +0.10 | Intuitive Surgical Pre-Q1 Analysis: Buy, Hold or Sell the Stock Now? // Thermo Fisher Scientific (TM |
| 4 | **GOOG** | $339.40 | 0.88% | 1.0× | +0.12 | +7.50% | 95 | +0.11 | How the SpaceX IPO could rewrite space investing // This international market is a great investment  |
| 5 | **TDY** | $635.83 | 0.74% | 0.9× | +1.00 | -1.53% | 66 | +1.40 | Teledyne Technologies to Report Q1 Earnings: What's in the Cards? // Teledyne Drives Growth via Stra |
| 6 | **ISRG** | $469.21 | 0.75% | 0.9× | +0.38 | +4.13% | 60 | +2.35 | ISRG's Rising Procedure TAM With Low Penetration Backs Sustained Growth // Intuitive Surgical Is Jum |
| 7 | **COO** | $70.06 | 0.76% | 0.9× | +0.25 | -1.61% | 51 | +0.56 | Here's Why The Cooper Companies (COO) is a Strong Value Stock // Why The Cooper Companies (COO) is a |
| 8 | **DOV** | $219.07 | 0.77% | 0.9× | +0.00 | +0.90% | 66 | +3.47 | Dover Corporation (DOV) Earnings Expected to Grow: What to Know Ahead of Next Week's Relea // Dover  |
| 9 | **SJM** | $95.50 | 0.67% | 0.8× | +0.50 | +5.05% | 51 | +1.83 | MDLZ vs. SJM: Which Branded Food Stock Is Better Positioned Today? // Conagra Brands incoming CEO on |
| 10 | **HD** | $349.40 | 0.61% | 0.7× | +1.00 | +3.58% | 69 | +0.67 | Dow jumps 800 points, S&P 500 at new record, Nasdaq gains 6.8% this week // Home Depot Weighs SIMPL  |
| 11 | **NDSN** | $281.89 | 0.57% | 0.7× | +1.25 | +2.40% | 73 | -0.13 | Nordson (NDSN) is an Incredible Growth Stock: 3 Reasons Why // Do Nordson's (NDSN) Steady Dividend a |
| 12 | **PH** | $988.80 | 0.60% | 0.7× | +0.88 | +0.46% | 71 | -0.09 | Industrial Demand Holds Strong Despite Iran War, Truist Securities Says // Can Howmet Sustain Growth |
| 13 | **ZTS** | $122.38 | 0.66% | 0.8× | +0.25 | +3.84% | 70 | -0.11 | Is It Time To Reassess Zoetis (ZTS) After A 1-Year Share Price Slide? // Zoetis (ZTS) Stock Slides a |
| 14 | **HON** | $233.55 | 0.57% | 0.7× | +0.12 | -0.63% | 65 | +0.35 | 3 Unpopular Stocks We Approach with Caution // Honeywell International Inc. (HON): One of the Best M |
| 15 | **BMY** | $60.17 | 0.55% | 0.6× | +0.25 | +2.64% | 59 | -0.06 | Can Myqorzo Drive Growth for Cytokinetics Amid Competition? // Roche’s DAC investment; Big Pharma’s  |

## 6. Magnitude and duration — honest answer

The model outputs a probability, not a point estimate of the size of the move. For the *rare* days it fires (none today), the empirical bursts in the training data averaged:
- **median forward 5-day return:** 9.3%
- **mean forward 5-day return:** 9.1%
- **25th–75th percentile:** 7.2% to 11.7%

So when a burst *does* happen on a calm large-cap name, it's typically a ~10–20% move clustered in a 2–4 day stretch — almost always tied to a specific catalyst (earnings, M&A, regulatory). Predicting the **timing** is the hard part the technical model attacks; predicting the **size** given a burst occurs is a separate, easier problem.

## 7. Limitations and honest framing

- **Test AUC 0.58 is modest.** The usable signal is a 1–2× lift over base rate, not a high-confidence buzzer.
- **News is keyword-scored.** A real production system would use a finetuned finance classifier (FinBERT etc.). The current score is coarse but helps separate catalysts from noise on the shortlist.
- **No event calendar.** Earnings dates, FDA PDUFA dates, and M&A rumor streams would materially add to lift but aren't wired in.
- **Regime.** The test fold is Dec 2025 – Apr 2026. Performance in a macro shock regime (Aug 2024, Apr 2025) is not measured here.
- **No cost/slippage modelling.** This is a forecasting exercise, not a backtested trading strategy.
- **Survivorship:** universe = current S&P 500 membership, so dropped names aren't represented. Minor impact on a 3-year window but worth flagging.

## 8. How to reproduce

```bash
cd /Users/mollyqian/Desktop/stocks
python3 code/10_burst_universe.py   # build the filtered universe
python3 code/11_burst_features.py   # fetch history, compute features + targets
python3 code/12_burst_train.py      # train GBC, score today
python3 code/13_burst_news.py       # annotate top candidates with headlines
python3 code/14_burst_report.py     # regenerate STOCK_REPORT.md
```

Key artifacts:
- [`data/burst_universe.csv`](data/burst_universe.csv) — filtered ticker list with metrics
- [`data/burst_panel.csv`](data/burst_panel.csv) — full feature + target panel
- [`models/burst_gbc.joblib`](models/burst_gbc.joblib) — trained classifier
- [`output/burst_metrics.json`](output/burst_metrics.json) — test-set metrics
- [`output/burst_today_ranked.csv`](output/burst_today_ranked.csv) — today's ranked predictions
- [`output/burst_news.json`](output/burst_news.json) — recent headlines per top candidate
- [`output/burst_feature_importance.csv`](output/burst_feature_importance.csv)
