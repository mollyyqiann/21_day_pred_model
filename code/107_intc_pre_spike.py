"""Did v3 flag {TICKER} as a top pick BEFORE its run?

Usage: python 107_intc_pre_spike.py [TICKER]


Replays INTC's daily v3 score over the last ~60 days and checks if it
ever entered the daily top-15 by raw_margin BEFORE the catalyst.

Also shows what its prob_cal and raw_margin looked like day by day to
diagnose whether the model identified accumulation, or whether it only
caught up AFTER the spike.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

sys.path.insert(0, str(ROOT / "code"))
from regime_features import attach_regime  # noqa: E402

XRANK = ["rsi_14_xrank", "rv_60_xrank", "ma60_slope_xrank", "ret_20d_xrank"]


def main():
    TICKER = sys.argv[1] if len(sys.argv) > 1 else "INTC"
    print(f"=== {TICKER} pre-run signal trajectory ===")
    panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    cat = pd.read_csv(DATA / "catalyst_features_sp500.csv", parse_dates=["date"])
    panel = panel.merge(cat, on=["ticker", "date"], how="left")
    panel = attach_regime(panel)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["close_20d_ago"] = panel.groupby("ticker")["close"].shift(20)
    panel["ret_20d_lag"] = panel["close"] / panel["close_20d_ago"] - 1.0
    panel["close_5d_ago"] = panel.groupby("ticker")["close"].shift(5)
    panel["ret_5d_lag"] = panel["close"] / panel["close_5d_ago"] - 1.0
    panel["rsi_14_xrank"] = panel.groupby("date")["rsi_14"].rank(pct=True)
    panel["rv_60_xrank"] = panel.groupby("date")["rv_60"].rank(pct=True)
    panel["ma60_slope_xrank"] = panel.groupby("date")["ma60_slope_60d"].rank(pct=True)
    panel["ret_20d_xrank"] = panel.groupby("date")["ret_20d_lag"].rank(pct=True)
    for c in ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
              "news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
              "ma_news_5d", "ma_news_20d", "sector_pop_5d"]:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0)
        else:
            panel[c] = 0.0
    for c in XRANK:
        panel[c] = panel[c].fillna(0.5)

    art = joblib.load(MODELS / "monthly_gainer_v3_sp500.joblib")
    feats = art["feats"]; med = pd.Series(art["impute_medians"])
    cal = art["calibrator"]; gbc = art["raw_gbc"]
    sub = panel.dropna(subset=feats).copy()
    X = sub[feats].fillna(med).values
    sub["prob_cal"] = cal.predict_proba(X)[:, 1]
    sub["raw_margin"] = gbc.decision_function(X)

    # INTC trajectory last 60 days of panel
    intc = sub[sub["ticker"] == TICKER].sort_values("date").tail(60).copy()
    print("INTC daily v3 signal — last 60 trading days of panel")
    print(f"{'date':<12} {'close':>9} {'5d':>7} {'20d':>7} {'margin':>8} {'prob':>6} {'rank':>6} {'top15?':>7}")

    # For each INTC day, compute its rank among same-day SP500 stocks
    rank_data = []
    for d, dgrp in sub.groupby("date"):
        rk = dgrp["raw_margin"].rank(method="min", ascending=False)
        intc_row = dgrp[dgrp["ticker"] == TICKER]  # noqa: var name kept
        if intc_row.empty:
            continue
        intc_rank = int(rk[intc_row.index[0]])
        rank_data.append({"date": d, "intc_rank": intc_rank,
                            "n_stocks": len(dgrp)})
    rank_df = pd.DataFrame(rank_data)
    intc = intc.merge(rank_df, on="date", how="left")

    for _, r in intc.iterrows():
        in_top15 = "✅" if r["intc_rank"] <= 15 else ""
        print(f"{r['date'].date()!s:<12} ${r['close']:>7.2f} "
              f"{r['ret_5d_lag']:>+7.1%} {r['ret_20d_lag']:>+7.1%} "
              f"{r['raw_margin']:>+8.2f} {r['prob_cal']:>6.2f} "
              f"#{r['intc_rank']:<5} {in_top15:>7}")

    print(f"\nDays in top-15 (last 60d): {(intc['intc_rank']<=15).sum()}")
    print(f"Days in top-5 (last 60d): {(intc['intc_rank']<=5).sum()}")
    pre_spike = intc[intc["date"] < "2026-04-24"]
    post_spike = intc[intc["date"] >= "2026-04-24"]
    print(f"\nPRE-SPIKE (before 4/24): days in top-15 = {(pre_spike['intc_rank']<=15).sum()} of {len(pre_spike)}")
    print(f"POST-SPIKE (4/24+): days in top-15 = {(post_spike['intc_rank']<=15).sum()} of {len(post_spike)}")
    print(f"\nPRE-SPIKE prob_cal: mean={pre_spike['prob_cal'].mean():.3f}, max={pre_spike['prob_cal'].max():.3f}")
    print(f"PRE-SPIKE raw_margin: mean={pre_spike['raw_margin'].mean():+.2f}, max={pre_spike['raw_margin'].max():+.2f}")


if __name__ == "__main__":
    main()
