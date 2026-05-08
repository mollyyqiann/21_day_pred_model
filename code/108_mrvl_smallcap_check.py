"""Check MRVL's signal trajectory in the SMALLCAP model."""

import sys; sys.stdout.reconfigure(line_buffering=True)
import warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "code"))
from regime_features import attach_regime
from extension_classifier import classify_extension

XRANK = ["rsi_14_xrank", "rv_60_xrank", "ma60_slope_xrank", "ret_20d_xrank"]


def main():
    TICKER = sys.argv[1] if len(sys.argv) > 1 else "MRVL"
    panel = pd.read_csv(DATA / "monthly_gainer_panel_smallcap.csv", parse_dates=["date"])
    cat = pd.read_csv(DATA / "catalyst_features_smallcap.csv", parse_dates=["date"])
    panel = panel.merge(cat, on=["ticker", "date"], how="left")
    panel = attach_regime(panel)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["close_20d_ago"] = panel.groupby("ticker")["close"].shift(20)
    panel["ret_20d_lag"] = panel["close"] / panel["close_20d_ago"] - 1.0
    panel["close_5d_ago"] = panel.groupby("ticker")["close"].shift(5)
    panel["ret_5d_lag"] = panel["close"] / panel["close_5d_ago"] - 1.0
    panel["close_60d_ago"] = panel.groupby("ticker")["close"].shift(60)
    panel["ret_60d_lag"] = panel["close"] / panel["close_60d_ago"] - 1.0
    panel["rsi_14_xrank"] = panel.groupby("date")["rsi_14"].rank(pct=True)
    panel["rv_60_xrank"] = panel.groupby("date")["rv_60"].rank(pct=True)
    panel["ma60_slope_xrank"] = panel.groupby("date")["ma60_slope_60d"].rank(pct=True)
    panel["ret_20d_xrank"] = panel.groupby("date")["ret_20d_lag"].rank(pct=True)
    for c in ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
              "news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
              "ma_news_5d", "ma_news_20d", "sector_pop_5d"]:
        if c in panel.columns: panel[c] = panel[c].fillna(0)
        else: panel[c] = 0.0
    for c in XRANK:
        panel[c] = panel[c].fillna(0.5)

    art = joblib.load(MODELS / "monthly_gainer_v3_smallcap.joblib")
    feats = art["feats"]; med = pd.Series(art["impute_medians"])
    cal = art["calibrator"]; gbc = art["raw_gbc"]
    sub = panel.dropna(subset=feats).copy()
    X = sub[feats].fillna(med).values
    sub["prob_cal"] = cal.predict_proba(X)[:, 1]
    sub["raw_margin"] = gbc.decision_function(X)

    tk_data = sub[sub["ticker"] == TICKER].sort_values("date").tail(60).copy()
    print(f"{TICKER} smallcap-model signal — last {len(tk_data)} days")
    print(f"{'date':<12} {'close':>9} {'5d':>7} {'20d':>7} {'60d':>7} {'margin':>8} {'prob':>6} {'rank':>6} {'top15?':>7}")

    rank_data = []
    for d, dgrp in sub.groupby("date"):
        rk = dgrp["raw_margin"].rank(method="min", ascending=False)
        tk_row = dgrp[dgrp["ticker"] == TICKER]
        if tk_row.empty: continue
        rank_data.append({"date": d, "rank": int(rk[tk_row.index[0]]),
                            "n": len(dgrp)})
    rank_df = pd.DataFrame(rank_data)
    if not rank_df.empty:
        tk_data = tk_data.merge(rank_df, on="date", how="left")
    else:
        tk_data["rank"] = 999

    for _, r in tk_data.iterrows():
        in_top15 = "✅" if r["rank"] <= 15 else ""
        r60_s = f"{r.get('ret_60d_lag', float('nan')):+7.1%}" if pd.notna(r.get('ret_60d_lag')) else "    n/a"
        print(f"{r['date'].date()!s:<12} ${r['close']:>7.2f} "
              f"{r['ret_5d_lag']:>+7.1%} {r['ret_20d_lag']:>+7.1%} {r60_s} "
              f"{r['raw_margin']:>+8.2f} {r['prob_cal']:>6.2f} "
              f"#{r['rank']:<5} {in_top15:>7}")

    print(f"\nDays in top-15 (last 60d): {(tk_data['rank']<=15).sum()}")
    print(f"Days in top-5 (last 60d): {(tk_data['rank']<=5).sum()}")


if __name__ == "__main__":
    main()
