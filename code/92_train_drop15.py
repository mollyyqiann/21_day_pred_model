"""Train a 21d-touch-down model: P(min(close[t+1..t+21]) / close[t] - 1 <= -0.15).

Used as a RISK FILTER on top of the up-prob model: a stock with high up-prob
AND low drop-prob is a cleaner buy signal. Crucially, this lets us KEEP
long-run-length stocks that don't show drop risk (instead of filtering all
long runs out blindly).

Same pipeline as v1 (23 features) but inverted label.

Usage:
  python 92_train_drop15.py sp500
  python 92_train_drop15.py smallcap

Outputs:
  models/monthly_gainer_drop15_{universe}.joblib
  output/monthly_gainer/drop15_{universe}_metrics.json
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output" / "monthly_gainer"

sys.path.insert(0, str(ROOT / "code"))
from regime_features import REGIME_FEATS, attach_regime  # noqa: E402

V1_FEATURES = [
    "rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
    "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60",
    "ma_stack", "up_streak", "up_bigdays_20d",
    "dist_ma60_atr", "ma60_slope_60d", "run_length",
] + REGIME_FEATS  # 23


def add_y21_drop15(panel: pd.DataFrame) -> pd.DataFrame:
    """min(close[t+1..t+21]) / close[t] - 1 <= -0.15 → y_drop15 = 1."""
    out = []
    for tk, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True).copy()
        c = g["close"].values.astype(float)
        n = len(c)
        y = np.full(n, -1, dtype=np.int8)
        max_drawdown = np.full(n, np.nan, dtype=np.float32)
        for t in range(n - 1):
            ct = c[t]
            if ct <= 0 or not np.isfinite(ct):
                continue
            if t + 21 < n:
                drawdown = c[t+1:t+22].min() / ct - 1.0
                max_drawdown[t] = drawdown
                y[t] = 1 if drawdown <= -0.15 else 0
        g["y_drop15"] = y
        g["max_fwd21_drawdown"] = max_drawdown
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _time_split(df, train_frac=0.70, val_frac=0.15):
    dates = np.sort(df["date"].unique())
    t1 = dates[int(len(dates) * train_frac)]
    t2 = dates[int(len(dates) * (train_frac + val_frac))]
    return (df[df["date"] < t1].copy(),
            df[(df["date"] >= t1) & (df["date"] < t2)].copy(),
            df[df["date"] >= t2].copy())


def _sample_weights(y, target_pos_frac):
    p = y.mean()
    if p <= 0 or p >= 1:
        return np.ones(len(y))
    w_pos = target_pos_frac / p
    w_neg = (1 - target_pos_frac) / (1 - p)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def evaluate(y, p):
    base = float(np.mean(y))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    ap = float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")
    ll = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
    br = float(brier_score_loss(y, p))
    return {"n": int(len(y)), "pos": int(y.sum()),
            "base_rate": base, "auc": auc, "ap": ap,
            "ap_lift": ap / base if base > 0 else float("nan"),
            "log_loss": ll, "brier": br}


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "sp500"
    assert universe in ("sp500", "smallcap")

    panel_path = (DATA / "monthly_gainer_panel.csv" if universe == "sp500"
                  else DATA / "monthly_gainer_panel_smallcap.csv")
    print(f"[92] training drop15 for {universe}")
    t0 = time.time()
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    panel = attach_regime(panel)
    panel = add_y21_drop15(panel)

    lab = panel[panel["y_drop15"] >= 0].dropna(subset=V1_FEATURES).reset_index(drop=True)
    print(f"[92] labeled+complete: {len(lab):,}  base={lab['y_drop15'].mean():.4%}")

    train, val, test = _time_split(lab)
    print(f"[92] split: train={len(train):,} val={len(val):,} test={len(test):,}  "
          f"test_base={test['y_drop15'].mean():.4%}")

    medians = train[V1_FEATURES].median(numeric_only=True)
    Xtr = train[V1_FEATURES].fillna(medians).values
    Xva = val[V1_FEATURES].fillna(medians).values
    Xte = test[V1_FEATURES].fillna(medians).values
    ytr = train["y_drop15"].values.astype(np.int8)
    yva = val["y_drop15"].values.astype(np.int8)
    yte = test["y_drop15"].values.astype(np.int8)

    sw = _sample_weights(ytr, target_pos_frac=min(0.20, max(0.04, ytr.mean() * 3)))

    print("[92] fitting GBC ...")
    gbc = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42)
    gbc.fit(Xtr, ytr, sample_weight=sw)
    cal = CalibratedClassifierCV(estimator=gbc, method="isotonic", cv="prefit")
    cal.fit(Xva, yva)
    p_te = cal.predict_proba(Xte)[:, 1]

    metrics = evaluate(yte, p_te)
    print(f"[92] test: {metrics}")

    imp = pd.Series(gbc.feature_importances_, index=V1_FEATURES).sort_values(ascending=False)
    print(f"[92] top-10 importance:\n{imp.head(10).to_string()}")

    artifact = {
        "raw_gbc": gbc, "calibrator": cal,
        "feats": V1_FEATURES, "impute_medians": medians.to_dict(),
        "metrics": metrics, "feature_importance": imp.to_dict(),
        "universe": universe, "label": "y_drop15",
        "label_def": "min(close[t+1..t+21]) / close[t] - 1 <= -0.15",
    }
    joblib.dump(artifact, MODELS / f"monthly_gainer_drop15_{universe}.joblib")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_only = {k: v for k, v in artifact.items() if k not in ("raw_gbc", "calibrator")}
    (OUT / f"drop15_{universe}_metrics.json").write_text(
        json.dumps(metrics_only, indent=2, default=str))
    print(f"[92] saved {MODELS / f'monthly_gainer_drop15_{universe}.joblib'}")
    print(f"[92] elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
