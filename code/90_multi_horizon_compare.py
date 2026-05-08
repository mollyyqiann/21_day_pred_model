"""Phase H: Multi-horizon accuracy comparison.

Compares 21d-touch>=30% (the primary task) against 4 alternatives, all trained
identically with v1's 23 features:

  shorter horizon: 5d-touch>=30%, 10d-touch>=30%
  longer horizon:  60d-touch>=30%
  easier mag:      21d-touch>=15%
  harder mag:      21d-touch>=50%

Reports AUC, PR-AUC, lift, precision@top-5 per task. Goal: tell the user how
prediction quality changes with window length and magnitude.

Outputs: output/monthly_gainer/multi_horizon.json + console table
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output" / "monthly_gainer"

sys.path.insert(0, str(ROOT / "code"))
from regime_features import REGIME_FEATS, attach_regime  # noqa: E402

V1_FEATURES = [
    "rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
    "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60",
    "ma_stack", "up_streak", "up_bigdays_20d",
    "dist_ma60_atr", "ma60_slope_60d", "run_length",
] + REGIME_FEATS  # 23


def add_multi_horizon_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Add y10_30, y60_30, y21_15, y21_50 labels per ticker."""
    out = []
    for tk, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True).copy()
        c = g["close"].values.astype(float)
        n = len(c)
        # 10d, 60d touch >= 30%; 21d touch >= 15%, 50%
        y10 = np.full(n, -1, dtype=np.int8)
        y60 = np.full(n, -1, dtype=np.int8)
        y21_15 = np.full(n, -1, dtype=np.int8)
        y21_50 = np.full(n, -1, dtype=np.int8)
        for t in range(n - 1):
            ct = c[t]
            if ct <= 0 or not np.isfinite(ct):
                continue
            if t + 10 < n:
                ret10 = c[t+1:t+11].max() / ct - 1.0
                y10[t] = 1 if ret10 >= 0.30 else 0
            if t + 60 < n:
                ret60 = c[t+1:t+61].max() / ct - 1.0
                y60[t] = 1 if ret60 >= 0.30 else 0
            if t + 21 < n:
                ret21 = c[t+1:t+22].max() / ct - 1.0
                y21_15[t] = 1 if ret21 >= 0.15 else 0
                y21_50[t] = 1 if ret21 >= 0.50 else 0
        g["y10_30"] = y10
        g["y60_30"] = y60
        g["y21_15"] = y21_15
        g["y21_50"] = y21_50
        out.append(g)
    return pd.concat(out, ignore_index=True)


def time_split(df, train_frac=0.70, val_frac=0.15):
    dates = np.sort(df["date"].unique())
    t1 = dates[int(len(dates) * train_frac)]
    t2 = dates[int(len(dates) * (train_frac + val_frac))]
    return (df[df["date"] < t1].copy(),
            df[(df["date"] >= t1) & (df["date"] < t2)].copy(),
            df[df["date"] >= t2].copy())


def sample_weights(y, target_pos_frac=0.10):
    p = y.mean()
    if p <= 0 or p >= 1:
        return np.ones(len(y))
    w_pos = target_pos_frac / p
    w_neg = (1 - target_pos_frac) / (1 - p)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def fit_and_eval(panel: pd.DataFrame, label_col: str, name: str) -> dict:
    df = panel[panel[label_col] >= 0].dropna(subset=V1_FEATURES).reset_index(drop=True)
    train, val, test = time_split(df)
    medians = train[V1_FEATURES].median(numeric_only=True)
    Xtr = train[V1_FEATURES].fillna(medians).values
    Xva = val[V1_FEATURES].fillna(medians).values
    Xte = test[V1_FEATURES].fillna(medians).values
    ytr = train[label_col].values.astype(np.int8)
    yva = val[label_col].values.astype(np.int8)
    yte = test[label_col].values.astype(np.int8)
    sw = sample_weights(ytr, target_pos_frac=min(0.20, max(0.04, ytr.mean() * 5)))

    gbc = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42)
    gbc.fit(Xtr, ytr, sample_weight=sw)
    cal = CalibratedClassifierCV(estimator=gbc, method="isotonic", cv="prefit")
    cal.fit(Xva, yva)
    p_te = cal.predict_proba(Xte)[:, 1]

    base = float(yte.mean())
    auc = float(roc_auc_score(yte, p_te)) if len(np.unique(yte)) > 1 else float("nan")
    ap = float(average_precision_score(yte, p_te)) if yte.sum() > 0 else float("nan")
    lift = ap / base if base > 0 else float("nan")

    # precision@top-5 per day
    test_eval = test.copy()
    test_eval["prob"] = p_te
    test_eval["y"] = yte
    precs = []
    for d, g in test_eval.groupby("date"):
        if len(g) < 5:
            continue
        topk = g.nlargest(5, "prob")
        precs.append(topk["y"].mean())
    p5 = float(np.mean(precs)) if precs else float("nan")

    return {
        "name": name,
        "label": label_col,
        "n_test": int(len(test)),
        "pos_test": int(yte.sum()),
        "base_rate": base,
        "auc": auc,
        "ap": ap,
        "ap_lift": lift,
        "p_at_top5": p5,
        "p_at_top5_lift": p5 / base if base > 0 else float("nan"),
    }


def main():
    print("[90] loading panel + adding multi-horizon labels ...")
    panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    panel = attach_regime(panel)
    panel = add_multi_horizon_labels(panel)
    print(f"[90] panel: {len(panel):,} rows")

    tasks = [
        ("y5_touch", "5d-touch >= 30%"),
        ("y10_30",  "10d-touch >= 30%"),
        ("y21",     "21d-touch >= 30% (primary)"),
        ("y60_30",  "60d-touch >= 30%"),
        ("y21_15",  "21d-touch >= 15% (easier mag)"),
        ("y21_50",  "21d-touch >= 50% (harder mag)"),
    ]

    results = []
    for col, name in tasks:
        if col not in panel.columns:
            continue
        print(f"\n[90] === {name} ===")
        t0 = time.time()
        m = fit_and_eval(panel, col, name)
        print(f"[90] {name}: base={m['base_rate']:.3%}  AUC={m['auc']:.3f}  "
              f"PR-AUC={m['ap']:.4f}  lift={m['ap_lift']:.2f}x  "
              f"p@top5={m['p_at_top5']:.2%}  p@5_lift={m['p_at_top5_lift']:.2f}x  "
              f"({time.time() - t0:.0f}s)")
        results.append(m)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "multi_horizon.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[90] saved {OUT / 'multi_horizon.json'}")

    print("\n[90] === SUMMARY TABLE ===")
    print(f"{'task':<35} {'base':>8} {'AUC':>6} {'PR-AUC':>8} {'lift':>6} {'p@top5':>8} {'lift':>6}")
    for r in results:
        print(f"{r['name']:<35} {r['base_rate']:>8.3%} {r['auc']:>6.3f} "
              f"{r['ap']:>8.4f} {r['ap_lift']:>6.2f}x {r['p_at_top5']:>8.2%} "
              f"{r['p_at_top5_lift']:>6.2f}x")


if __name__ == "__main__":
    main()
