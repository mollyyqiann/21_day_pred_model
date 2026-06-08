"""Backtest vanilla vs augmented models against ACTUAL historical bursts.

Two complementary views:

  (A) Daily simulation over the test fold (last ~15% of history).
      For each trading day d: score every ticker, take top-K by prob, check
      which of those top-K actually had a burst (y==1 means at least one
      2-5 day average-return window >= 4% in the next 5 trading days).
      Reports precision@K, recall@K, and the mean prob of catchers vs missers.

  (B) Per-burst attribution.
      For every y==1 event in the test fold, compute its rank among all
      tickers on that day. Distribution of ranks tells us: how often did
      the model actually put a real burst in the top-5 / top-20 / top-50?

Done separately for v6b (v4 / >$40), v6 (v5 / upside-asymmetric), v7 (full S&P 500),
and for both vanilla and augmented classifiers so the overnight-gap uplift is
visible at the event level, not just as a single AUC number.

Outputs:
  output/backtest_summary.json
  output/backtest_precision_recall.csv
  output/backtest_burst_ranks.csv
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"; OUT = ROOT / "output"

A_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
          "atr_pct", "range_pct", "vol_z", "vol_5d"]
V4_FEATS = A_BASE + ["rv_60"]
V5_FEATS = V4_FEATS + ["skew_60d", "semivol_ratio_60d", "up_bigdays_60d"]
V6_FEATS  = V5_FEATS + ["overnight_gap"]            # augmented for v5 universe
V6B_FEATS = V4_FEATS + ["overnight_gap"]            # augmented for v4 universe
V7_V_FEATS = V4_FEATS                               # vanilla feats for v7
V7_A_FEATS = V4_FEATS + ["overnight_gap"]           # augmented feats for v7

K_LIST = [5, 10, 20, 50]


BURST_WINDOW = 5; BURST_MIN_LEN = 2; BURST_THRESH = 0.04

def recompute_y(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute burst target y per ticker from the close column.
    y = 1 iff within next 5 trading days a contiguous 2-5-day window has
    average daily return >= 4%."""
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    y_all = np.full(len(panel), -1, dtype=np.int8)
    for t, g in panel.groupby("ticker", sort=False):
        idx = g.index.values
        c = g["close"].values
        r = np.concatenate([[0.0], np.diff(c) / c[:-1]])
        n = len(r)
        y = np.zeros(n, dtype=np.int8)
        for i in range(n - BURST_WINDOW):
            fut = r[i+1:i+1+BURST_WINDOW]; best = 0.0
            for L in range(BURST_MIN_LEN, BURST_WINDOW + 1):
                for s in range(0, BURST_WINDOW - L + 1):
                    m = fut[s:s+L].mean()
                    if m > best: best = m
            if best >= BURST_THRESH: y[i] = 1
        y[-BURST_WINDOW:] = -1   # unknown — live rows
        y_all[idx] = y
    panel["y"] = y_all
    return panel


def test_fold_mask(panel: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """Return bool mask for the chronological 85%-100% (test) fold."""
    lab = panel[panel["y"] >= 0].dropna(subset=feats).copy()
    dates = np.sort(lab["date"].unique())
    d2 = dates[int(0.85 * len(dates))]
    return (panel["y"] >= 0) & panel[feats].notna().all(axis=1) & (panel["date"] >= d2)


def run_backtest(panel: pd.DataFrame, vanilla_path: Path, augmented_path: Path,
                 v_feats: list[str], a_feats: list[str], tag: str):
    van = joblib.load(vanilla_path)["gbc"]
    aug = joblib.load(augmented_path)["gbc"]

    # use rows with ALL features present (so we can score both)
    test_mask = test_fold_mask(panel, a_feats)
    te = panel[test_mask].copy()
    if len(te) == 0:
        return {"error": "empty test fold"}
    te["prob_vanilla"] = van.predict_proba(te[v_feats].values)[:, 1]
    te["prob_augmented"] = aug.predict_proba(te[a_feats].values)[:, 1]

    # --- view (B): per-burst attribution ---
    burst_rows = []
    for d, g in te.groupby("date"):
        g = g.copy()
        g["rank_vanilla"]   = g["prob_vanilla"].rank(ascending=False, method="min")
        g["rank_augmented"] = g["prob_augmented"].rank(ascending=False, method="min")
        g["n_universe"] = len(g)
        for _, r in g[g["y"] == 1].iterrows():
            burst_rows.append({
                "universe": tag, "date": d.date().isoformat(), "ticker": r["ticker"],
                "n_universe": int(r["n_universe"]),
                "prob_vanilla": float(r["prob_vanilla"]),
                "prob_augmented": float(r["prob_augmented"]),
                "rank_vanilla": int(r["rank_vanilla"]),
                "rank_augmented": int(r["rank_augmented"]),
                "overnight_gap": float(r.get("overnight_gap", float("nan"))),
            })

    # --- view (A): precision @ K and recall @ K per day, then averaged ---
    days = te["date"].unique()
    agg = {"vanilla": {k: {"tp": 0, "fp": 0, "fn": 0, "day_rows": []} for k in K_LIST},
           "augmented": {k: {"tp": 0, "fp": 0, "fn": 0, "day_rows": []} for k in K_LIST}}
    total_positives = 0
    for d, g in te.groupby("date"):
        npos = int(g["y"].sum())
        total_positives += npos
        for kind, col in [("vanilla", "prob_vanilla"), ("augmented", "prob_augmented")]:
            gs = g.sort_values(col, ascending=False)
            for k in K_LIST:
                topk = gs.head(k)
                tp = int(topk["y"].sum())
                fp = k - tp if len(topk) >= k else len(topk) - tp
                fn = npos - tp
                agg[kind][k]["tp"] += tp
                agg[kind][k]["fp"] += fp
                agg[kind][k]["fn"] += fn
                agg[kind][k]["day_rows"].append({"date": d, "tp": tp, "npos": npos})

    summary = {"universe": tag, "n_test_days": int(len(days)),
               "n_test_rows": int(len(te)),
               "n_test_bursts": int(total_positives),
               "base_rate": float(total_positives / len(te))}
    for kind in ("vanilla", "augmented"):
        for k in K_LIST:
            a = agg[kind][k]
            p = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) else 0.0
            r = a["tp"] / total_positives if total_positives else 0.0
            summary[f"precision@{k}_{kind}"] = float(p)
            summary[f"recall@{k}_{kind}"] = float(r)
            summary[f"tp@{k}_{kind}"] = int(a["tp"])

    # delta: augmented advantage
    for k in K_LIST:
        summary[f"delta_precision@{k}"] = (
            summary[f"precision@{k}_augmented"] - summary[f"precision@{k}_vanilla"])
        summary[f"delta_recall@{k}"] = (
            summary[f"recall@{k}_augmented"] - summary[f"recall@{k}_vanilla"])

    return summary, burst_rows


def main():
    configs = [
        ("v4", DATA / "burst_panel_v6b.csv",
         MODELS / "burst_gbc_v6b_vanilla.joblib",
         MODELS / "burst_gbc_v6b_augmented.joblib",
         V4_FEATS, V6B_FEATS),
        ("v5", DATA / "burst_panel_v6.csv",
         MODELS / "burst_gbc_v6_vanilla.joblib",
         MODELS / "burst_gbc_v6_augmented.joblib",
         V5_FEATS, V6_FEATS),
        ("v7", DATA / "burst_panel_v7.csv",
         MODELS / "burst_gbc_v7_vanilla.joblib",
         MODELS / "burst_gbc_v7_augmented.joblib",
         V7_V_FEATS, V7_A_FEATS),
    ]
    all_summary = []; all_bursts = []
    for tag, panel_path, vp, ap, vf, af in configs:
        print(f"\n=== backtesting {tag} ===")
        panel = pd.read_csv(panel_path, parse_dates=["date"])
        if "y" not in panel.columns or (panel["y"].dropna().eq(-1).all() if "y" in panel.columns else True):
            print(f"[{tag}] panel missing y column — recomputing from close")
            panel = recompute_y(panel)
        out = run_backtest(panel, vp, ap, vf, af, tag)
        if isinstance(out, tuple):
            summary, bursts = out
            all_summary.append(summary); all_bursts.extend(bursts)
            print(f"[{tag}] test days: {summary['n_test_days']}  "
                  f"bursts: {summary['n_test_bursts']}  "
                  f"base: {summary['base_rate']:.3%}")
            print(f"[{tag}]   precision@5  vanilla={summary['precision@5_vanilla']:.1%}  "
                  f"augmented={summary['precision@5_augmented']:.1%}  "
                  f"Δ={summary['delta_precision@5']:+.1%}")
            print(f"[{tag}]   precision@10 vanilla={summary['precision@10_vanilla']:.1%}  "
                  f"augmented={summary['precision@10_augmented']:.1%}  "
                  f"Δ={summary['delta_precision@10']:+.1%}")
            print(f"[{tag}]   precision@20 vanilla={summary['precision@20_vanilla']:.1%}  "
                  f"augmented={summary['precision@20_augmented']:.1%}  "
                  f"Δ={summary['delta_precision@20']:+.1%}")
            print(f"[{tag}]   recall@5     vanilla={summary['recall@5_vanilla']:.1%}  "
                  f"augmented={summary['recall@5_augmented']:.1%}  "
                  f"Δ={summary['delta_recall@5']:+.1%}")
            print(f"[{tag}]   recall@20    vanilla={summary['recall@20_vanilla']:.1%}  "
                  f"augmented={summary['recall@20_augmented']:.1%}")
            print(f"[{tag}]   recall@50    vanilla={summary['recall@50_vanilla']:.1%}  "
                  f"augmented={summary['recall@50_augmented']:.1%}")

    (OUT / "backtest_summary.json").write_text(json.dumps(all_summary, indent=2))
    pd.DataFrame(all_bursts).to_csv(OUT / "backtest_burst_ranks.csv", index=False)

    # Ranks summary per universe
    b = pd.DataFrame(all_bursts)
    if len(b):
        print("\n=== rank distribution of ACTUAL bursts (augmented model) ===")
        for tag in ["v4", "v5", "v7"]:
            sub = b[b["universe"] == tag]
            if len(sub) == 0: continue
            pct = lambda k: (sub["rank_augmented"] <= k).mean()
            print(f"  {tag} ({len(sub)} bursts, mean universe size "
                  f"{int(sub['n_universe'].mean())}): "
                  f"top-5 {pct(5)*100:.1f}%  top-10 {pct(10)*100:.1f}%  "
                  f"top-20 {pct(20)*100:.1f}%  top-50 {pct(50)*100:.1f}%  "
                  f"median rank {int(sub['rank_augmented'].median())}")

        print("\n=== vanilla -> augmented rank lift for actual bursts ===")
        for tag in ["v4", "v5", "v7"]:
            sub = b[b["universe"] == tag]
            if len(sub) == 0: continue
            promoted = (sub["rank_augmented"] < sub["rank_vanilla"]).sum()
            demoted = (sub["rank_augmented"] > sub["rank_vanilla"]).sum()
            same = (sub["rank_augmented"] == sub["rank_vanilla"]).sum()
            med_lift = (sub["rank_vanilla"] - sub["rank_augmented"]).median()
            print(f"  {tag}: promoted {promoted}  same {same}  demoted {demoted}  "
                  f"median rank_lift {int(med_lift):+d}")


if __name__ == "__main__":
    main()
