# Stock Burst Prediction V3 — Report

**As of:** 2026-04-17  ·  **Universe:** 41 tickers (S&P 500, episode-pattern filter)

**Target:** within the next 5 trading days, some 2–5-day window averages ≥ 4%/day on a stock that is *normally calm*.

---

## 1. What changed vs v1 / v2

| Aspect | v1 | v2 | **v3** |
|---|---|---|---|
| Universe filter | current 60d vol ≤ 30% | historical 60d vol ≤ 30% (lagged 90d) | calm → vol → calm episode within **6 × episode-length** of today |
| Features | 28 | 37 (adds regime ratios, big-day counts) | same 37 used in ablation |
| Model(s) | single GBC | single GBC | **5-way comparison: 3 GBC feature sets, vanilla LSTM, stock-modified hybrid** |
| Catalyst analysis | none | none | earnings proximity per historical burst |

The universe filter in v3 answers your objection directly: SNDK-style names with a recent vol episode are now **kept** (whereas v1 dropped them the moment they started moving). See § 4 for who qualifies.

## 2. Model comparison — what features/model actually work

Trained on the v2 panel (110335 train rows... ) with a chronological 70/15/15 split (train < 2025, val through end-2025, test Jan–Apr 2026).


| Model | test AUC | test PR-AUC | PR-AUC / base | log-loss |
|---|---:|---:|---:|---:|
| `GBC_A_base` | 0.689 | 0.053 | **2.57×** | 0.1002 | ← **winner**
| `GBC_B_plus_mom` | 0.663 | 0.048 | **2.32×** | 0.1065 |
| `LSTM_vanilla` | 0.673 | 0.042 | **2.06×** | 0.7217 |
| `GBC_C_plus_reg` | 0.660 | 0.041 | **2.01×** | 0.1070 |
| `LSTM_hybrid` | 0.672 | 0.036 | **1.77×** | 0.4801 |

### Read-out

1. **Feature bloat hurts.** The 9-feature classical-technical set (`A_base`: RSI, MACD/MACD-signal/MACD-hist, BB-z, ATR%, range%, vol-z, vol-5d) beats the 18-feature (`B_plus_mom`) and 37-feature (`C_plus_reg`) variants on every metric. Momentum and regime-shift aggregates — which sounded useful — add more noise than signal on this rare-event problem. A reproducible lesson: **on ~1% base-rate events with ~100k rows, 9 well-chosen features > 37 features**.

2. **Vanilla LSTM doesn't win.** On raw 30-day sequences (`ret_1d`, `vol_z`, `resid_vs_spy`, `range_pct`, `rv_20`), AUC is 0.673 and PR-AUC lift 2.1× — competitive, but strictly worse than the tiny GBC on every metric. The log-loss is an order of magnitude higher because the pos-weighted BCE produces uncalibrated probabilities.

3. **The stock-modified hybrid loses, gracefully.** The hybrid (LSTM branch over market-adjusted sequences **concatenated** with an MLP branch over the 37 regime-shift aggregates) was designed to combine path and regime. On AUC it's within noise of vanilla LSTM; on AP it's *worse*, because the extra parameters over-fit the ~1,500 training positives. Bigger data or better regularisation (focal loss, hard-negative mining) would be the next experiments.

4. **Why the simple tabular model wins here.** Burst events on calm large-caps are fundamentally rare, noisy, and largely news-driven. Classical technicals (RSI, MACD, ATR, vol-z) already *are* compressed summaries of sequence behavior — an LSTM would have to rediscover them from raw OHLCV, using gradient updates of a ~1% positive class. With only ~1.5k positive examples across all tickers, the sequence model is data-starved relative to a gradient-boosting tree that reads those same summaries directly.

**Headline:** the v1/v2 report told you "use 28 features and an LSTM." The honest v3 answer is "use 9 features and a gradient boost." The LSTM is retained as a baseline so future model changes can be measured against a sequence model, not just claimed to be better.

## 3. Why do bursts happen? Catalyst attribution

For every historical burst event in the v3 universe (316 events), we computed the calendar-day distance to the nearest scheduled earnings release and tagged events within ±5 days as `earnings`, otherwise `other`.

| Tag | Share of bursts | Reading |
|---|---:|---|
| **earnings** | 51.3% | event adjacent to a scheduled earnings release |
| **other**   | 48.7%   | M&A / regulatory / macro / rumor |

Tickers with the most non-earnings bursts (signal-to-catalyst of interest):

| Ticker | bursts | earnings-adjacent | non-earnings |
|---|---:|---:|---:|
| **CAT** | 38 | 12 | 26 |
| **CVS** | 21 | 5 | 16 |
| **ISRG** | 28 | 13 | 15 |
| **HSIC** | 18 | 5 | 13 |
| **CBRE** | 26 | 14 | 12 |
| **MSCI** | 21 | 10 | 11 |
| **JBHT** | 15 | 5 | 10 |
| **DIS** | 18 | 10 | 8 |
| **TMO** | 10 | 3 | 7 |
| **HSY** | 9 | 3 | 6 |

CAT and CVS are the cleanest examples of "burst machines" where roughly 3 out of 4 bursts aren't earnings-related — they reflect macro/cyclical narrative shifts (CAT: global industrial / infra / China cycle) and policy risk (CVS: drug pricing, PBM policy, rate rumors). The model picking up elevated probability on these names should be read as "a non-earnings catalyst is *possible* given the setup", not "a release is scheduled."

## 4. V3 universe ("calm → vol → calm, recent enough to matter")

A stock qualifies if, within its 3-year history, it has at least one **volatility episode** (rolling 20-day annualized vol > max(2 × its own baseline, 35%), lasting ≥ 3 trading days) that was (a) preceded by a calm regime, (b) followed by a calm regime or is the current tail, and (c) ended **within 6 × episode-length trading days of today**.

The 6× recency rule is your filter: a 20-day episode 100 days ago qualifies (5× length); a 20-day episode 150 days ago does not (7.5× length).

**41 S&P 500 tickers qualify.** Grouped by sector:

- Industrials: 9
- Health Care: 7
- Consumer Staples: 6
- Consumer Discretionary: 6
- Communication Services: 5
- Financials: 3
- Real Estate: 2
- Information Technology: 1
- Materials: 1
- Utilities: 1

## 5. Today's predictions (2026-04-17)

Scored with the winning model (`GBC_A_base`). Test-fold base rate was **2.05%**; lift is prob / base rate.

**Flagged (prob ≥ 5% AND lift ≥ 3.0×): 0.**

Highest prob today is **TDG** at 3.32% (1.6× base). Below the flag bar — consistent with the expected null output on typical days.

### Top-15 ranked (v3 universe)

| # | Ticker | Close | Prob | Lift | Sector | Most recent episode | Days since | Peak rv20 | RSI14 | ret_5d |
|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| 1 | **TDG** | $1265.88 | 3.32% | 1.6× | Industrials | 2026-02-03 (20d) | 32 | 0.40 | 67 | +4.86% |
| 2 | **SYY** | $76.27 | 2.79% | 1.4× | Consumer Staples | 2026-03-30 (14d) | 0 | 0.61 | 40 | +4.74% |
| 3 | **SJM** | $95.50 | 2.76% | 1.3× | Consumer Staples | 2026-03-03 (12d) | 21 | 0.43 | 51 | +5.05% |
| 4 | **CAT** | $794.65 | 2.33% | 1.1× | Industrials | 2025-10-29 (20d) | 97 | 0.53 | 72 | +0.50% |
| 5 | **AVY** | $172.48 | 2.32% | 1.1× | Materials | 2025-10-22 (20d) | 102 | 0.42 | 57 | +0.77% |
| 6 | **ISRG** | $469.21 | 2.31% | 1.1× | Health Care | 2025-10-22 (20d) | 102 | 0.57 | 60 | +4.13% |
| 7 | **MSCI** | $568.55 | 1.98% | 1.0× | Financials | 2026-02-10 (16d) | 31 | 0.50 | 72 | +5.98% |
| 8 | **ALLE** | $144.32 | 1.83% | 0.9× | Industrials | 2026-02-17 (20d) | 23 | 0.42 | 50 | -0.34% |
| 9 | **CVS** | $77.30 | 1.65% | 0.8× | Health Care | 2026-01-27 (20d) | 37 | 0.60 | 72 | -2.56% |
| 10 | **TMO** | $526.60 | 1.52% | 0.7× | Health Care | 2025-10-01 (20d) | 117 | 0.46 | 74 | +6.15% |
| 11 | **HSIC** | $78.83 | 1.51% | 0.7× | Health Care | 2025-11-04 (20d) | 93 | 0.46 | 80 | +4.63% |
| 12 | **KMB** | $98.84 | 1.49% | 0.7× | Consumer Staples | 2025-11-03 (20d) | 94 | 0.57 | 50 | +1.60% |
| 13 | **STZ** | $162.28 | 1.40% | 0.7× | Consumer Staples | 2026-04-09 (7d) | 0 | 0.36 | 67 | -2.33% |
| 14 | **DIS** | $106.29 | 1.25% | 0.6× | Communication Services | 2026-02-06 (16d) | 33 | 0.44 | 91 | +7.18% |
| 15 | **GWW** | $1162.94 | 1.22% | 0.6× | Industrials | 2026-02-12 (13d) | 32 | 0.40 | 76 | -0.78% |

## 6. SNDK case study — why even v3 misses it

You flagged SanDisk as the type of burst you'd want caught. Honest accounting:

- SNDK was spun out of Western Digital on 2025-02-13; it has only ~295 trading days of history.
- Its **post-spin vol was always high** (first-60-day realized vol ran >50% annualized). It never satisfied the "calm baseline" condition, so the v3 episode filter does not mark it as a calm→vol→calm pattern either — it's a *structural re-rating* story, not a regime-shift story.
- It did register the target event multiple times (3-day avg daily: +14.4% on 2026-01-06, +10.5% on 2025-09-05, etc.), but those are part of one ongoing multi-month ramp, not discrete "episodes" bracketed by calm periods.

**Where this pipeline can't help**, and what would: names with <6 months of history or in a structurally new regime need a different playbook — fundamental / thematic screens (memory cycle, AI data-center demand, post-spin re-rating comps) rather than technical regime-shift pattern recognition. The v3 universe explicitly gates on "used to be calm", which is the right constraint for the modeling question but eliminates structural re-rating situations by design.

## 7. Limitations and next steps

- **Absolute probabilities are still low.** Top prob today is ~3%, ~1.6× base rate. That's genuine signal but not a high-conviction flag.
- **No earnings calendar feature.** Given 51% of bursts are earnings-adjacent, injecting `days_to_next_earnings` as a direct feature should materially lift both AUC and calibration. Next pipeline change.
- **No paid news feed.** Catalyst classification is binary (earnings / other); a real "why" attribution would require event labels (M&A, downgrade, guidance cut).
- **No ensembling.** A stacked average of GBC_A_base + Hybrid produced marginal gains in quick tests; not pursued here to keep the comparison clean.
- **Single seed LSTM.** Sequence models benchmark is the one-seed reading.

## 8. How to reproduce

```bash
cd /Users/mollyqian/Desktop/stocks
python3 code/17_burst_universe_v3.py     # calm->vol->calm universe
python3 code/16_burst_features_v2.py     # features + burst targets panel
python3 code/18_burst_models.py          # train/compare 5 models, score today
python3 code/19_burst_catalysts.py       # earnings vs other catalyst analysis
python3 code/20_burst_report_v3.py       # regenerate this report
```

Artifacts:
- [`data/burst_universe_v3.csv`](data/burst_universe_v3.csv) — v3 tickers with episode metadata
- [`output/burst_models_compare.csv`](output/burst_models_compare.csv) — full comparison table
- [`output/burst_models_compare.json`](output/burst_models_compare.json) — raw per-split metrics
- [`output/burst_today_v3.csv`](output/burst_today_v3.csv) — today's ranked predictions
- [`output/burst_catalysts.csv`](output/burst_catalysts.csv) — historical bursts tagged earnings vs other
- [`output/burst_catalyst_summary.json`](output/burst_catalyst_summary.json) — aggregates
- [`models/burst_gbc_v2.joblib`](models/burst_gbc_v2.joblib) — trained full-feature GBC (not the winner)
- [`models/burst_lstm_vanilla.pt`](models/burst_lstm_vanilla.pt), [`models/burst_lstm_hybrid.pt`](models/burst_lstm_hybrid.pt) — sequence-model weights