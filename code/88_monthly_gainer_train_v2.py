"""Phase F.2/F.3: Train v2 monthly-gainer model with catalyst features added.

Loads {sp500, smallcap} panel + corresponding catalyst_features CSV, merges,
trains GBC with the v1 23 features PLUS 10 catalyst features (33 total).
Reports delta vs v1.

Usage:
  python 88_monthly_gainer_train_v2.py sp500
  python 88_monthly_gainer_train_v2.py smallcap

Outputs (per universe):
  models/monthly_gainer_v2_{universe}.joblib
  output/monthly_gainer/v2_{universe}_metrics.json
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
]  # 10
ALL_FEATURES = V1_FEATURES + CATALYST_FEATURES  # 33


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
    base = float(np.mean(y)) if len(y) else float("nan")
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    ap = float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")
    ll = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
    br = float(brier_score_loss(y, p))
    return {"n": int(len(y)), "pos": int(np.sum(y)),
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


def stratified_auc(test_df, prob_col, buckets):
    out = {}
    for lbl, mask in buckets:
        sub = test_df[mask]
        if sub["y21"].sum() < 5 or sub["y21"].nunique() < 2:
            out[lbl] = {"n": int(len(sub)), "pos": int(sub["y21"].sum()), "auc": None, "ap": None}
            continue
        out[lbl] = {
            "n": int(len(sub)), "pos": int(sub["y21"].sum()),
            "auc": float(roc_auc_score(sub["y21"], sub[prob_col])),
            "ap": float(average_precision_score(sub["y21"], sub[prob_col])),
        }
    return out


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
    assert universe in ("sp500", "smallcap"), "universe must be sp500 or smallcap"

    if universe == "sp500":
        panel_path = DATA / "monthly_gainer_panel.csv"
        cat_path = DATA / "catalyst_features_sp500.csv"
        target_pos_frac = 0.06
    else:
        panel_path = DATA / "monthly_gainer_panel_smallcap.csv"
        cat_path = DATA / "catalyst_features_smallcap.csv"
        target_pos_frac = 0.08

    print(f"[88] training v2 for universe={universe}")
    t0 = time.time()
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    cat = pd.read_csv(cat_path, parse_dates=["date"])
    print(f"[88] panel: {len(panel):,}  catalyst: {len(cat):,}")

    panel = panel.merge(cat, on=["ticker", "date"], how="left")
    panel = attach_regime(panel)

    lab = panel[panel["y21"] >= 0].dropna(subset=V1_FEATURES).reset_index(drop=True)
    # catalyst features may have NaNs (no news data) — fill with sentinels
    for c in ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d"]:
        lab[c] = lab[c].fillna(0.0)
    for c in ["news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
              "ma_news_5d", "ma_news_20d", "sector_pop_5d"]:
        lab[c] = lab[c].fillna(0).astype(float)
    print(f"[88] labeled+complete: {len(lab):,}  base={lab['y21'].mean():.4%}")

    train, val, test = _time_split(lab)
    print(f"[88] split: train={len(train):,} val={len(val):,} test={len(test):,}  "
          f"test_base={test['y21'].mean():.4%}")

    medians = train[ALL_FEATURES].median(numeric_only=True)
    Xtr = train[ALL_FEATURES].fillna(medians).values
    Xva = val[ALL_FEATURES].fillna(medians).values
    Xte = test[ALL_FEATURES].fillna(medians).values
    ytr = train["y21"].values.astype(np.int8)
    yva = val["y21"].values.astype(np.int8)
    yte = test["y21"].values.astype(np.int8)

    sw = _sample_weights(ytr, target_pos_frac)

    # ===== v1 baseline (23 feats) for delta computation =====
    print("[88] training v1-baseline (23 feats) ...")
    Xtr_v1 = train[V1_FEATURES].fillna(medians).values
    Xva_v1 = val[V1_FEATURES].fillna(medians).values
    Xte_v1 = test[V1_FEATURES].fillna(medians).values
    gbc_v1 = GradientBoostingClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42)
    gbc_v1.fit(Xtr_v1, ytr, sample_weight=sw)
    cal_v1 = CalibratedClassifierCV(estimator=gbc_v1, method="isotonic", cv="prefit")
    cal_v1.fit(Xva_v1, yva)
    p_te_v1 = cal_v1.predict_proba(Xte_v1)[:, 1]

    test_v1 = test.copy()
    test_v1["prob"] = p_te_v1
    metrics_v1 = evaluate(yte, p_te_v1)
    metrics_v1["p_at_top5"] = precision_at_topk_per_day(test_v1, "prob", 5)
    metrics_v1["p_at_top10"] = precision_at_topk_per_day(test_v1, "prob", 10)
    metrics_v1["sim"] = long_only_sim(test_v1, "prob", 5)
    print(f"[88] v1: {metrics_v1}")

    # ===== v2 (33 feats) =====
    print("[88] training v2 (23 base + 10 catalyst feats) ...")
    gbc_v2 = GradientBoostingClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42)
    gbc_v2.fit(Xtr, ytr, sample_weight=sw)
    cal_v2 = CalibratedClassifierCV(estimator=gbc_v2, method="isotonic", cv="prefit")
    cal_v2.fit(Xva, yva)
    p_te_v2 = cal_v2.predict_proba(Xte)[:, 1]

    test_v2 = test.copy()
    test_v2["prob"] = p_te_v2
    metrics_v2 = evaluate(yte, p_te_v2)
    metrics_v2["p_at_top5"] = precision_at_topk_per_day(test_v2, "prob", 5)
    metrics_v2["p_at_top10"] = precision_at_topk_per_day(test_v2, "prob", 10)
    metrics_v2["sim"] = long_only_sim(test_v2, "prob", 5)
    print(f"[88] v2: {metrics_v2}")

    delta = {
        "auc": metrics_v2["auc"] - metrics_v1["auc"],
        "ap": metrics_v2["ap"] - metrics_v1["ap"],
        "ap_lift": metrics_v2["ap_lift"] - metrics_v1["ap_lift"],
        "p_at_top5": metrics_v2["p_at_top5"] - metrics_v1["p_at_top5"],
        "p_at_top10": metrics_v2["p_at_top10"] - metrics_v1["p_at_top10"],
        "topk_end_ret": (metrics_v2["sim"]["topk_end_ret"]
                          - metrics_v1["sim"]["topk_end_ret"]) if metrics_v1["sim"] and metrics_v2["sim"] else None,
    }
    print(f"\n[88] DELTA v2-v1: {delta}")

    # rv_60 stratified for v2
    rv_q = train["rv_60"].quantile([0.25, 0.5, 0.75]).values
    buckets = [
        ("Q1", test_v2["rv_60"] < rv_q[0]),
        ("Q2", (test_v2["rv_60"] >= rv_q[0]) & (test_v2["rv_60"] < rv_q[1])),
        ("Q3", (test_v2["rv_60"] >= rv_q[1]) & (test_v2["rv_60"] < rv_q[2])),
        ("Q4", test_v2["rv_60"] >= rv_q[2]),
    ]
    rv_strat_v2 = stratified_auc(test_v2, "prob", buckets)
    print(f"[88] v2 rv_60 strat: {rv_strat_v2}")

    # feature importance for v2 (not from calibrator — use raw GBC)
    imp = pd.Series(gbc_v2.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    print(f"[88] v2 top-10 importance:\n{imp.head(10).to_string()}")
    catalyst_imp_share = float(imp[CATALYST_FEATURES].sum())
    print(f"[88] catalyst features importance share: {catalyst_imp_share:.1%}")

    # save
    artifact = {
        "raw_gbc": gbc_v2, "calibrator": cal_v2,
        "feats": ALL_FEATURES,
        "v1_features": V1_FEATURES, "catalyst_features": CATALYST_FEATURES,
        "impute_medians": medians.to_dict(),
        "v1_metrics": metrics_v1, "v2_metrics": metrics_v2,
        "delta_v2_v1": delta,
        "rv_strat_v2": rv_strat_v2,
        "feature_importance": imp.to_dict(),
        "catalyst_importance_share": catalyst_imp_share,
        "universe": universe,
    }
    joblib.dump(artifact, MODELS / f"monthly_gainer_v2_{universe}.joblib")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_only = {k: v for k, v in artifact.items() if k not in ("raw_gbc", "calibrator")}
    (OUT / f"v2_{universe}_metrics.json").write_text(
        json.dumps(metrics_only, indent=2, default=str))
    print(f"[88] saved {MODELS / f'monthly_gainer_v2_{universe}.joblib'}")
    print(f"[88] elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
