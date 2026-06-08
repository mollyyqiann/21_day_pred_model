"""Compare v1 vs v2 predictions on the last 60 live trading days.

Consumes output/drop_v2/walk_forward_predictions.csv (out-of-sample predictions
from both v1 and v2 per row). Produces:

  output/drop_v2/head_to_head.json
    - prob distributions (histogram edges + counts) for each model
    - number of rows crossing 0.30 / 0.40 / 0.50 thresholds
    - realized drop rate among rows crossing each threshold (precision)
    - top-5 per day: how often did the top-5 actually drop
    - per-day alert counts over time

  output/drop_v2/head_to_head_table.csv
    - date, alerts_v1, alerts_v2, hits_v1, hits_v2, precision_v1, precision_v2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / "output" / "drop_v2" / "walk_forward_predictions.csv"
OUT_DIR = ROOT / "output" / "drop_v2"
THRESHOLDS = [0.20, 0.25, 0.30, 0.40, 0.50]
RECENT_DAYS = 60


def _hist(probs: np.ndarray, edges: list[float]) -> dict:
    counts, _ = np.histogram(probs, bins=edges)
    return {f"{edges[i]:.2f}-{edges[i+1]:.2f}": int(counts[i])
            for i in range(len(edges) - 1)}


def main():
    if not WF.exists():
        print(f"walk-forward predictions missing at {WF}; run 39_drop_v2_backtest first")
        return
    df = pd.read_csv(WF, parse_dates=["date"])
    print(f"[h2h] loaded {len(df)} walk-forward predictions "
          f"spanning {df['date'].min().date()} .. {df['date'].max().date()}")

    # Recent window
    cutoff = df["date"].max() - pd.Timedelta(days=RECENT_DAYS)
    recent = df[df["date"] >= cutoff].copy()
    print(f"[h2h] recent window (last {RECENT_DAYS} days): {len(recent)} rows, "
          f"pos_rate={recent['y_drop'].mean():.4f}")

    edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70, 1.0]
    summary = {
        "date_range_recent": [recent["date"].min().strftime("%Y-%m-%d"),
                              recent["date"].max().strftime("%Y-%m-%d")],
        "n_rows_recent": int(len(recent)),
        "pos_rate_recent": float(recent["y_drop"].mean()),
        "histograms": {
            "v1": _hist(recent["p_v1"].values, edges),
            "v2_raw": _hist(recent["p_v2_raw"].values, edges),
            "v2_recal": _hist(recent["p_v2_recal"].values, edges),
        },
        "thresholds": {},
    }

    # Alerts-at-threshold and realized-drop rate
    for thr in THRESHOLDS:
        t_summary = {}
        for pcol in ["p_v1", "p_v2_raw", "p_v2_recal"]:
            mask = recent[pcol] >= thr
            n_alerts = int(mask.sum())
            if n_alerts == 0:
                realized_drop_rate = None
            else:
                realized_drop_rate = float(recent.loc[mask, "y_drop"].mean())
            t_summary[pcol] = {
                "n_alerts": n_alerts,
                "realized_drop_rate": realized_drop_rate,
                "alerts_per_day": round(n_alerts / max(recent["date"].nunique(), 1), 2),
            }
        summary["thresholds"][f"{thr:.2f}"] = t_summary

    # Top-5 per day precision (proxy for the "fresh alerts" list)
    def _top5_precision(group: pd.DataFrame, pcol: str) -> tuple[int, int]:
        top5 = group.nlargest(5, pcol)
        return int(top5["y_drop"].sum()), len(top5)

    per_day = []
    for d, g in recent.groupby("date"):
        if len(g) < 5:
            continue
        hits_v1, _ = _top5_precision(g, "p_v1")
        hits_v2r, _ = _top5_precision(g, "p_v2_raw")
        hits_v2c, _ = _top5_precision(g, "p_v2_recal")
        mean_p_v1 = float(g.nlargest(5, "p_v1")["p_v1"].mean())
        mean_p_v2r = float(g.nlargest(5, "p_v2_raw")["p_v2_raw"].mean())
        mean_p_v2c = float(g.nlargest(5, "p_v2_recal")["p_v2_recal"].mean())
        per_day.append({
            "date": d.strftime("%Y-%m-%d"),
            "n_tickers": len(g),
            "pos_rate_day": float(g["y_drop"].mean()),
            "top5_mean_p_v1": round(mean_p_v1, 4),
            "top5_mean_p_v2_raw": round(mean_p_v2r, 4),
            "top5_mean_p_v2_recal": round(mean_p_v2c, 4),
            "top5_hits_v1": hits_v1,
            "top5_hits_v2_raw": hits_v2r,
            "top5_hits_v2_recal": hits_v2c,
        })

    per_day_df = pd.DataFrame(per_day)
    per_day_df.to_csv(OUT_DIR / "head_to_head_per_day.csv", index=False)

    summary["top5_per_day"] = {
        "days": int(len(per_day_df)),
        "total_top5_predictions": int(len(per_day_df) * 5) if len(per_day_df) else 0,
        "v1": {
            "hits": int(per_day_df["top5_hits_v1"].sum()) if len(per_day_df) else 0,
            "precision": float(per_day_df["top5_hits_v1"].sum() / max(len(per_day_df) * 5, 1))
                         if len(per_day_df) else None,
            "mean_prob_of_top5": float(per_day_df["top5_mean_p_v1"].mean())
                                 if len(per_day_df) else None,
        },
        "v2_raw": {
            "hits": int(per_day_df["top5_hits_v2_raw"].sum()) if len(per_day_df) else 0,
            "precision": float(per_day_df["top5_hits_v2_raw"].sum() / max(len(per_day_df) * 5, 1))
                         if len(per_day_df) else None,
            "mean_prob_of_top5": float(per_day_df["top5_mean_p_v2_raw"].mean())
                                 if len(per_day_df) else None,
        },
        "v2_recal": {
            "hits": int(per_day_df["top5_hits_v2_recal"].sum()) if len(per_day_df) else 0,
            "precision": float(per_day_df["top5_hits_v2_recal"].sum() / max(len(per_day_df) * 5, 1))
                         if len(per_day_df) else None,
            "mean_prob_of_top5": float(per_day_df["top5_mean_p_v2_recal"].mean())
                                 if len(per_day_df) else None,
        },
    }

    (OUT_DIR / "head_to_head.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
