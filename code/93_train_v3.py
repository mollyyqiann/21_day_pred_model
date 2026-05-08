"""V3: v2 (33 feats) + 4 cross-sectional rank features.

The v1/v2 model is mostly a per-stock vol+momentum classifier. Cross-sectional
ranks add "is this stock unusually strong/loud TODAY relative to peers" — the
signal that captures AMD-class moves where the stock isn't yet in extreme
absolute vol but stands out vs sector/universe.

New features (per (ticker, date), no leakage):
  rsi_14_xrank      — universe-wide percentile of rsi_14
  rv_60_xrank       — universe-wide percentile of rv_60
  ma60_slope_xrank  — universe-wide percentile of ma60_slope_60d
  ret_20d_xrank     — universe-wide percentile of trailing 20d return

All ranks computed across same-date rows only (cross-sectional, not time series),
so they cannot leak future info.

Usage:
  python 93_train_v3.py sp500
  python 93_train_v3.py smallcap

Outputs:
  models/monthly_gainer_v3_{universe}.joblib
  output/monthly_gainer/v3_{universe}_metrics.json
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

CATALYST_FEATURES = [
    "finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
    "news_n_5d", "news_n_20d",
    "earn_news_5d", "earn_news_20d",
    "ma_news_5d", "ma_news_20d",
    "sector_pop_5d",
]

XRANK_FEATURES = [
    "rsi_14_xrank", "rv_60_xrank",
    "ma60_slope_xrank", "ret_20d_xrank",
]

ALL_FEATURES = V1_FEATURES + CATALYST_FEATURES + XRANK_FEATURES  # 37


def add_xrank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional percentile rank within each date."""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["close_20d_ago"] = df.groupby("ticker")["close"].shift(20)
    df["ret_20d_lag"] = df["close"] / df["close_20d_ago"] - 1.0

    df["rsi_14_xrank"] = df.groupby("date")["rsi_14"].rank(pct=True)
    df["rv_60_xrank"] = df.groupby("date")["rv_60"].rank(pct=True)
    df["ma60_slope_xrank"] = df.groupby("date")["ma60_slope_60d"].rank(pct=True)
    df["ret_20d_xrank"] = df.groupby("date")["ret_20d_lag"].rank(pct=True)
    return df


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


def precision_at_topk_per_day(test_df, prob_col, k):
    precs = []
    for d, g in test_df.groupby("date"):
        if len(g) < k:
            continue
        topk = g.nlargest(k, prob_col)
        precs.append(topk["y21"].mean())
    return float(np.mean(precs)) if precs else float("nan")


def long_only_sim(test_df, prob_col, k=5):
    rows = []
    for d, g in test_df.groupby("date"):
        if len(g) < k:
            continue
        topk = g.nlargest(k, prob_col)
        rows.append({
            "topk_max": topk["max_fwd21_ret"].mean(),
            "topk_end": topk["end_of_window_ret"].mean(),
            "univ_max": g["max_fwd21_ret"].mean(),
            "univ_end": g["end_of_window_ret"].mean(),
            "topk_pos": topk["y21"].mean(),
        })
    if not rows:
        return None
    s = pd.DataFrame(rows)
    return {
        "topk_max_ret": float(s["topk_max"].mean()),
        "topk_end_ret": float(s["topk_end"].mean()),
        "univ_max_ret": float(s["univ_max"].mean()),
        "univ_end_ret": float(s["univ_end"].mean()),
        "topk_pos_rate": float(s["topk_pos"].mean()),
    }


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "sp500"
    assert universe in ("sp500", "smallcap")

    if universe == "sp500":
        panel_path = DATA / "monthly_gainer_panel.csv"
        cat_path = DATA / "catalyst_features_sp500.csv"
        target_pos_frac = 0.06
    else:
        panel_path = DATA / "monthly_gainer_panel_smallcap.csv"
        cat_path = DATA / "catalyst_features_smallcap.csv"
        target_pos_frac = 0.08

    print(f"[93] training v3 for universe={universe}")
    t0 = time.time()
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    cat = pd.read_csv(cat_path, parse_dates=["date"])
    print(f"[93] panel: {len(panel):,}  catalyst: {len(cat):,}")

    panel = panel.merge(cat, on=["ticker", "date"], how="left")
    panel = attach_regime(panel)
    panel = add_xrank_features(panel)

    lab = panel[panel["y21"] >= 0].dropna(subset=V1_FEATURES).reset_index(drop=True)
    for c in ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d"]:
        lab[c] = lab[c].fillna(0.0)
    for c in ["news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
              "ma_news_5d", "ma_news_20d", "sector_pop_5d"]:
        lab[c] = lab[c].fillna(0).astype(float)
    for c in XRANK_FEATURES:
        lab[c] = lab[c].fillna(0.5)
    print(f"[93] labeled+complete: {len(lab):,}  base={lab['y21'].mean():.4%}")

    train, val, test = _time_split(lab)
    print(f"[93] split: train={len(train):,} val={len(val):,} test={len(test):,}  "
          f"test_base={test['y21'].mean():.4%}")

    medians = train[ALL_FEATURES].median(numeric_only=True)
    Xtr = train[ALL_FEATURES].fillna(medians).values
    Xva = val[ALL_FEATURES].fillna(medians).values
    Xte = test[ALL_FEATURES].fillna(medians).values
    ytr = train["y21"].values.astype(np.int8)
    yva = val["y21"].values.astype(np.int8)
    yte = test["y21"].values.astype(np.int8)

    sw = _sample_weights(ytr, target_pos_frac)

    print("[93] training v3 (33 + 4 = 37 feats) ...")
    gbc = GradientBoostingClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42)
    gbc.fit(Xtr, ytr, sample_weight=sw)
    cal = CalibratedClassifierCV(estimator=gbc, method="isotonic", cv="prefit")
    cal.fit(Xva, yva)
    p_te = cal.predict_proba(Xte)[:, 1]

    test_v3 = test.copy()
    test_v3["prob"] = p_te
    metrics = evaluate(yte, p_te)
    metrics["p_at_top5"] = precision_at_topk_per_day(test_v3, "prob", 5)
    metrics["p_at_top10"] = precision_at_topk_per_day(test_v3, "prob", 10)
    metrics["sim"] = long_only_sim(test_v3, "prob", 5)
    print(f"[93] v3: {metrics}")

    imp = pd.Series(gbc.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    print(f"[93] v3 top-10 importance:\n{imp.head(10).to_string()}")
    xrank_share = float(imp[XRANK_FEATURES].sum())
    print(f"[93] xrank importance share: {xrank_share:.1%}")

    artifact = {
        "raw_gbc": gbc, "calibrator": cal,
        "feats": ALL_FEATURES,
        "v1_features": V1_FEATURES, "catalyst_features": CATALYST_FEATURES,
        "xrank_features": XRANK_FEATURES,
        "impute_medians": medians.to_dict(),
        "metrics": metrics,
        "feature_importance": imp.to_dict(),
        "xrank_importance_share": xrank_share,
        "universe": universe,
    }
    joblib.dump(artifact, MODELS / f"monthly_gainer_v3_{universe}.joblib")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_only = {k: v for k, v in artifact.items() if k not in ("raw_gbc", "calibrator")}
    (OUT / f"v3_{universe}_metrics.json").write_text(
        json.dumps(metrics_only, indent=2, default=str))
    print(f"[93] saved {MODELS / f'monthly_gainer_v3_{universe}.joblib'}")
    print(f"[93] elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
