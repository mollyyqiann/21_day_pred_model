"""Two conditional tests for the drop model, per user spec:

TEST 1 (random volatile days): pick random historical days where the stock was
        already experiencing elevated volatility (rv_60 > 40% OR up_streak > 5).
        Does the drop model correctly separate those that drop from those that
        don't?

TEST 2 (after our burst recommendation): for every (ticker, date) where the
        BURST model scored in the top-5 of its universe, look at days t+1..t+10.
        On each of those days, does the DROP model warn in time?
        Primary metric: for bursts that did actually drop within 10 days, what
        fraction were flagged by the drop model BEFORE the drop?

Outputs:
  output/drop_test1_volatile_days.csv
  output/drop_test2_after_burst_rec.csv
  output/drop_conditional_summary.json
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"; OUT = ROOT / "output"

V7_FEATS = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
            "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap"]
V8_FEATS = V7_FEATS + ["ma_stack", "up_streak", "up_bigdays_20d",
                        "dist_ma60_atr", "ma60_slope_60d", "run_length"]


def main():
    panel = pd.read_csv(DATA / "drop_panel_v1.csv", parse_dates=["date"])
    print(f"loaded {len(panel):,} panel rows")

    drop_gbc = joblib.load(MODELS / "drop_gbc_v1.joblib")["gbc"]
    burst_gbc = joblib.load(MODELS / "burst_gbc_v7_augmented.joblib")["gbc"]
    # burst uses V7 feats
    p = panel.dropna(subset=V8_FEATS + ["y_drop"]).copy()
    p = p[p["y_drop"].isin([0, 1])]   # drop rows where y_drop is -1 (unknown/live)
    # score everything
    p["p_drop"] = drop_gbc.predict_proba(p[V8_FEATS].values)[:, 1]
    p["p_burst"] = burst_gbc.predict_proba(p[V7_FEATS].values)[:, 1]

    # use test fold
    dates = np.sort(p["date"].unique()); d2 = dates[int(0.85 * len(dates))]
    te = p[p["date"] >= d2].copy()
    print(f"test fold: {len(te):,} rows  drop-base={te['y_drop'].mean():.4%}  ")

    # ---------- TEST 1: random volatile days ----------
    print("\n=== TEST 1: drop model on volatile days ===")
    vol_mask = (te["rv_60"] > 0.40) | (te["up_streak"] > 5)
    vol = te[vol_mask].copy()
    print(f"  volatile subset: {len(vol):,} rows  drop-rate={vol['y_drop'].mean():.4%}")
    if vol["y_drop"].sum() > 0:
        auc = roc_auc_score(vol["y_drop"], vol["p_drop"])
        ap = average_precision_score(vol["y_drop"], vol["p_drop"])
        print(f"  on volatile days:  AUC={auc:.3f}  AP={ap:.3f}  lift={ap/vol['y_drop'].mean():.2f}x")
        # precision @ top 10% of volatile days
        k = max(50, len(vol)//10)
        top = vol.sort_values("p_drop", ascending=False).head(k)
        print(f"  top-{k} by drop prob:  precision={top['y_drop'].mean():.1%}  "
              f"({int(top['y_drop'].sum())} real drops out of {k})")
        vol[["ticker", "date", "p_drop", "y_drop",
             "rv_60", "up_streak", "overnight_gap"]].to_csv(
            OUT / "drop_test1_volatile_days.csv", index=False)

    # ---------- TEST 2: after-burst-recommendation ----------
    print("\n=== TEST 2: after a burst recommendation, does drop model catch the drawdown? ===")
    # find days where a ticker was in top-5 by burst prob
    burst_picks = []
    for d, g in te.groupby("date"):
        top5 = g.sort_values("p_burst", ascending=False).head(5)
        for _, r in top5.iterrows():
            burst_picks.append({
                "date": d, "ticker": r["ticker"],
                "p_burst": r["p_burst"], "p_drop_same_day": r["p_drop"],
            })
    bp = pd.DataFrame(burst_picks)
    print(f"  total burst-picks in test fold: {len(bp)}")

    # for each burst pick, pull the next 10 trading days from panel
    # and compute: (a) did ANY next-10-day rolling window drop >= -3%/day avg 2+ days?
    # (b) did the drop model ever fire prob>=0.3 during those days BEFORE the drop?

    # build index for quick lookup of (ticker, date)
    p_idx = p.set_index(["ticker", "date"]).sort_index()
    total = 0; realized_drop = 0; caught_before = 0
    drop_after_rec_rows = []
    for _, pick in bp.iterrows():
        t = pick["ticker"]; d = pick["date"]
        future = panel[(panel["ticker"] == t) & (panel["date"] > d)].sort_values("date").head(10)
        if len(future) < 3: continue
        # realized drop in next 5 days window
        realized = future["y_drop"].head(5).max() == 1 if "y_drop" in future.columns else False
        # drop model fires on day after recommendation?
        day_after = future.head(1)
        try:
            pa = p_idx.loc[(t, day_after["date"].iloc[0])]
            drop_prob_day_after = float(pa["p_drop"])
        except Exception:
            drop_prob_day_after = float("nan")
        total += 1
        if realized: realized_drop += 1
        if realized and drop_prob_day_after >= 0.20: caught_before += 1
        drop_after_rec_rows.append({
            "date": d.date() if hasattr(d, "date") else d,
            "ticker": t,
            "p_burst": pick["p_burst"],
            "p_drop_next_day": drop_prob_day_after,
            "realized_drop_next_5d": bool(realized),
        })

    rar = pd.DataFrame(drop_after_rec_rows)
    rar.to_csv(OUT / "drop_test2_after_burst_rec.csv", index=False)
    print(f"  burst picks with >=3d forward data: {total}")
    print(f"  of those, realized a drop within 5 days: {realized_drop}  "
          f"({realized_drop/max(1,total):.1%})")
    print(f"  drop model flagged (p_drop_next_day >= 0.20) AND realized drop: "
          f"{caught_before}  (recall on caught-before-drop: "
          f"{caught_before/max(1,realized_drop):.1%})")

    # distribution check: was p_drop higher for drops-after-rec vs no-drop?
    if realized_drop > 0 and (total - realized_drop) > 0:
        m_drop = rar[rar["realized_drop_next_5d"]]["p_drop_next_day"].mean()
        m_nodrop = rar[~rar["realized_drop_next_5d"]]["p_drop_next_day"].mean()
        print(f"  mean p_drop (drop happened): {m_drop:.3f}  "
              f"vs (no drop): {m_nodrop:.3f}  "
              f"separation: {m_drop - m_nodrop:+.3f}")

    # save summary
    summary = {
        "test1_volatile_days": {
            "n": int(len(vol)),
            "drop_rate": float(vol["y_drop"].mean()) if len(vol) else 0.0,
        },
        "test2_after_burst_rec": {
            "n_picks": int(total),
            "pct_realized_drop": float(realized_drop / max(1, total)),
            "caught_before_recall": float(caught_before / max(1, realized_drop)),
        },
    }
    (OUT / "drop_conditional_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nsaved: output/drop_test1_volatile_days.csv")
    print("       output/drop_test2_after_burst_rec.csv")
    print("       output/drop_conditional_summary.json")


if __name__ == "__main__":
    main()
