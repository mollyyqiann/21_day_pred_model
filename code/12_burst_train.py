"""Train a classifier to predict burst events (see 11_burst_features.py for target).

Chronological 70/15/15 split by date. Primary model: Gradient Boosting Classifier
(scikit-learn). Also computes a simple Logistic Regression as a sanity check.

Outputs:
  models/burst_gbc.joblib
  output/burst_metrics.json
  output/burst_feature_importance.csv
  output/burst_today_scores.csv   -> per-ticker prob of burst in next 5 days, as of
                                      the latest fully-featured date (== today when
                                      run after the market close).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output"
MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42


def main() -> None:
    meta = json.loads((DATA / "burst_meta.json").read_text())
    feat_cols = meta["feature_cols"]
    df = pd.read_csv(DATA / "burst_panel.csv", parse_dates=["date"])
    print(f"[train] panel rows: {len(df)}  features: {len(feat_cols)}")

    # The 'live' rows (unknown target) are y == -1
    live = df[df["y"] == -1].copy()
    lab = df[df["y"] >= 0].copy()

    # drop rows with NaN in features (warmup period)
    lab = lab.dropna(subset=feat_cols).reset_index(drop=True)
    print(f"[train] labelled rows after NA drop: {len(lab)}")
    print(f"[train] burst base rate: {lab['y'].mean():.4%}")

    # chronological split on date
    dates_sorted = np.sort(lab["date"].unique())
    n = len(dates_sorted)
    d1 = dates_sorted[int(0.70 * n)]
    d2 = dates_sorted[int(0.85 * n)]
    print(f"[train] split dates: train<{d1}  val<{d2}  test>=val_end")

    train = lab[lab["date"] < d1]
    val = lab[(lab["date"] >= d1) & (lab["date"] < d2)]
    test = lab[lab["date"] >= d2]
    print(f"[train] sizes  train={len(train)}  val={len(val)}  test={len(test)}")

    Xtr, ytr = train[feat_cols].values, train["y"].values
    Xv, yv = val[feat_cols].values, val["y"].values
    Xte, yte = test[feat_cols].values, test["y"].values

    # Scale for logistic only; GBC doesn't need it.
    scaler = StandardScaler().fit(Xtr)

    # ---------------- Gradient Boosting ----------------
    print("[train] fitting GradientBoostingClassifier ...")
    gbc = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gbc.fit(Xtr, ytr)

    def scores(name, X, y):
        p = gbc.predict_proba(X)[:, 1]
        pos = y.sum()
        base = y.mean() if y.mean() > 0 else float("nan")
        try:
            auc = roc_auc_score(y, p)
        except ValueError:
            auc = float("nan")
        try:
            ap = average_precision_score(y, p)
        except ValueError:
            ap = float("nan")
        try:
            ll = log_loss(y, p, labels=[0, 1])
        except ValueError:
            ll = float("nan")
        bs = brier_score_loss(y, p)
        print(f"[train] {name}: n={len(y)} pos={int(pos)} base={base:.4%} "
              f"AUC={auc:.3f} AP={ap:.3f} (lift x{ap/base:.1f})  "
              f"logloss={ll:.4f} brier={bs:.4f}")
        return {"n": int(len(y)), "pos": int(pos), "base": float(base),
                "auc": float(auc), "ap": float(ap),
                "ap_lift": float(ap / base) if base and base > 0 else float("nan"),
                "log_loss": float(ll), "brier": float(bs)}

    metrics = {
        "train": scores("train", Xtr, ytr),
        "val":   scores("val",   Xv,  yv),
        "test":  scores("test",  Xte, yte),
    }

    # Logistic as sanity
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(scaler.transform(Xtr), ytr)
    p_lr_te = lr.predict_proba(scaler.transform(Xte))[:, 1]
    metrics["test_logistic_auc"] = float(roc_auc_score(yte, p_lr_te))
    print(f"[train] test logistic AUC (sanity): {metrics['test_logistic_auc']:.3f}")

    # Feature importance
    imp = pd.Series(gbc.feature_importances_, index=feat_cols).sort_values(ascending=False)
    imp.to_csv(OUT / "burst_feature_importance.csv", header=["importance"])
    print("[train] top 10 features:")
    print(imp.head(10).to_string())

    # Save artifacts
    joblib.dump({"gbc": gbc, "scaler": scaler, "feat_cols": feat_cols},
                MODELS / "burst_gbc.joblib")
    (OUT / "burst_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # ---------------- Score today ----------------
    # For each ticker, take the LAST row with full features (even if y is -1).
    scored = df.dropna(subset=feat_cols).copy()
    latest = scored.sort_values(["ticker", "date"]).groupby("ticker").tail(1).copy()
    latest["prob_burst"] = gbc.predict_proba(latest[feat_cols].values)[:, 1]

    # threshold for "actionable flag": require probability AND recent setup conditions
    # (positive momentum residual or unusual volume)
    base_rate = float(lab["y"].mean())
    latest["lift"] = latest["prob_burst"] / base_rate

    # decile relative to all test-set predictions (honest out-of-sample distribution)
    p_test = gbc.predict_proba(Xte)[:, 1]
    latest["test_percentile"] = latest["prob_burst"].apply(
        lambda v: float((p_test <= v).mean())
    )

    # constant useful context columns
    latest = latest[[
        "ticker", "date", "close", "prob_burst", "lift", "test_percentile",
        "ret_1d", "ret_5d", "ret_20d", "rv_20", "rsi_14", "bb_z20",
        "resid_5d", "resid_20d", "vol_z", "gap_ma50", "pos_52w",
    ]]
    latest = latest.sort_values("prob_burst", ascending=False).reset_index(drop=True)

    out_path = OUT / "burst_today_scores.csv"
    latest.to_csv(out_path, index=False)
    print(f"[train] scored {len(latest)} tickers -> {out_path}")
    print("[train] top 15 by prob_burst:")
    print(latest.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
