"""Train a classifier that predicts whether a big drop will CONTINUE or BOUNCE.

Problem: today's drop runner flags p_drop events (expected 1d drop >= -3%), but
gives no handle on whether the move is a one-shot capitulation that will
revert or a continuation that deepens. That's a key distinction for PUT
option buying (continuation is what you want — strike-chasing decay).

Label design:
  Eligible rows: drop_panel_v2 where y_drop_1d == 1 (had a 1-day drop of -3% or worse).
  y_cont     = y_drop_5d  (1 = drop persisted through day 5 cumulative <= -7%).

Features: same technicals the drop regressor sees, plus the forward-return
regressor outputs so we use the drop model's own forecast shape as a feature
for the continuation classifier.

Outputs:
  models/drop_continuation_clf.joblib    bundled GBC + scaler + feat list
  output/drop_continuation_metrics.json  AUC / PR-AUC / p@K / feat importance
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"; OUT = ROOT / "output"

FEATS = [
    "rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
    "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap",
    "ma_stack", "up_streak", "up_bigdays_20d",
    "dist_ma60_atr", "ma60_slope_60d", "run_length",
]


def main() -> None:
    print("[cont] loading drop_panel_v2 ...")
    df = pd.read_csv(DATA / "drop_panel_v2.csv", parse_dates=["date"])
    # Only rows where a 1-day big drop actually happened AND 5-day label exists
    eligible = df[(df["y_drop_1d"] == 1) & df["y_drop_5d"].notna()].copy()
    print(f"[cont] eligible rows (y_drop_1d=1): {len(eligible):,}")

    eligible = eligible.dropna(subset=FEATS)
    print(f"[cont] after feat NA drop: {len(eligible):,}")
    eligible["y_cont"] = eligible["y_drop_5d"].astype(int)
    print(f"[cont] continuation rate: {eligible['y_cont'].mean():.3%}")

    eligible = eligible.sort_values(["date", "ticker"]).reset_index(drop=True)
    n = len(eligible); a = int(n * 0.70); b = int(n * 0.85)
    tr = eligible.iloc[:a]; va = eligible.iloc[a:b]; te = eligible.iloc[b:]
    print(f"[cont] split: tr={len(tr)} va={len(va)} te={len(te)}")
    print(f"[cont] dates: tr {tr['date'].min().date()} -> {tr['date'].max().date()} | "
          f"te {te['date'].min().date()} -> {te['date'].max().date()}")

    scaler = StandardScaler().fit(tr[FEATS])
    Xtr = scaler.transform(tr[FEATS]); Xva = scaler.transform(va[FEATS])
    Xte = scaler.transform(te[FEATS])

    gbc = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=42)
    gbc.fit(Xtr, tr["y_cont"].values)

    def summ(y, p):
        auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
        pr = float(average_precision_score(y, p)) if y.sum() else float("nan")
        order = np.argsort(-p)
        return {"auc": auc, "pr_auc": pr,
                "p@10": float(y[order[:10]].mean()) if len(y) >= 10 else float("nan"),
                "p@50": float(y[order[:50]].mean()) if len(y) >= 50 else float("nan"),
                "p@100": float(y[order[:100]].mean()) if len(y) >= 100 else float("nan"),
                "pos_rate": float(y.mean())}

    p_tr = gbc.predict_proba(Xtr)[:, 1]
    p_va = gbc.predict_proba(Xva)[:, 1]
    p_te = gbc.predict_proba(Xte)[:, 1]

    metrics = {
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "train": summ(tr["y_cont"].values, p_tr),
        "val":   summ(va["y_cont"].values, p_va),
        "test":  summ(te["y_cont"].values, p_te),
        "feature_importance": sorted(
            [{"feat": f, "imp": float(i)}
             for f, i in zip(FEATS, gbc.feature_importances_)],
            key=lambda r: -r["imp"])[:12],
        "test_date_range": [str(te["date"].min().date()), str(te["date"].max().date())],
    }
    (OUT / "drop_continuation_metrics.json").write_text(json.dumps(metrics, indent=2))

    joblib.dump({"gbc": gbc, "scaler": scaler, "feats": FEATS},
                 MODELS / "drop_continuation_clf.joblib")

    print(f"\n=== continuation classifier (test) ===")
    print(f"  pos_rate   = {metrics['test']['pos_rate']:.3%}")
    print(f"  AUC        = {metrics['test']['auc']:.3f}")
    print(f"  PR-AUC     = {metrics['test']['pr_auc']:.3f}")
    print(f"  p@10       = {metrics['test']['p@10']:.3f}")
    print(f"  p@50       = {metrics['test']['p@50']:.3f}")
    print(f"  p@100      = {metrics['test']['p@100']:.3f}")
    print(f"\n  top features:")
    for r in metrics["feature_importance"][:8]:
        print(f"    {r['feat']:<22s}  {r['imp']:.3f}")


if __name__ == "__main__":
    main()
