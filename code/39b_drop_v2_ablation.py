"""Ablation: toggle each v2 improvement OFF vs. the full v2 on a fixed test fold.

Uses the same 70/15/15 time split as 36_drop_v2_pipeline.py. For each ablation
config, trains a HistGradientBoostingClassifier on the specified feature subset
(and with/without sample weights etc.) and reports val+test metrics.

The FULL config is `all_on` (all features + weights). Each ablation turns off
one lever and reports the delta.

Output: output/drop_v2/ablation.json + ablation_table.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "drop_panel_v2.csv"
OUT_DIR = ROOT / "output" / "drop_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V1_FEATURES = [
    "rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
    "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap",
    "ma_stack", "up_streak", "up_bigdays_20d",
    "dist_ma60_atr", "ma60_slope_60d", "run_length",
]
RANK_FEATURES = [f + "_rank" for f in [
    "rsi_14", "macd", "macd_hist", "bb_z20",
    "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60",
    "overnight_gap",
    "up_bigdays_20d", "dist_ma60_atr", "ma60_slope_60d",
    "run_length", "up_streak",
]]
REGIME_FEATURES = [
    "vix", "vix_5d_chg", "vix_z60",
    "spy_20d_dd", "spy_rv_20",
    "aaii_bb_spread", "aaii_bb_z52",
    "fng", "fng_5d_chg",
    "tnx_10y", "tnx_5d_chg", "dxy", "dxy_5d_chg", "gold_5d_ret",
    "qqq_iwm_5d_spread", "xlk_xlf_5d_spread",
    "breadth_ma_pos",
]
EDGAR_META_FEATURES = [
    "edgar_8k_cnt_5d", "edgar_8k_cnt_30d",
    "edgar_10q_days", "edgar_10k_days",
    "edgar_amend_cnt_30d", "edgar_nt_flag_90d", "edgar_13d_cnt_60d",
    "edgar_total_cnt_5d",
]
EDGAR_FB_FEATURES = ["edgar_fb_mean_30d", "edgar_fb_min_30d"]


def _sample_weights(y, target_pos_frac=0.20):
    y = np.asarray(y, dtype=np.int8)
    p = y.mean()
    if p <= 0 or p >= 1:
        return np.ones(len(y))
    w_pos = target_pos_frac / p
    w_neg = (1 - target_pos_frac) / (1 - p)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def _fit_and_score(features, tr, va, te, *, use_weights=True, learner="hgbc"):
    X_tr = tr[features].fillna(tr[features].median(numeric_only=True)).values
    X_va = va[features].fillna(tr[features].median(numeric_only=True)).values
    X_te = te[features].fillna(tr[features].median(numeric_only=True)).values
    y_tr = tr["y_drop"].values
    y_va = va["y_drop"].values
    y_te = te["y_drop"].values

    sw = _sample_weights(y_tr) if use_weights else None
    if learner == "hgbc":
        mdl = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=0.1,
            early_stopping=True, n_iter_no_change=25, validation_fraction=0.1,
            random_state=42)
        mdl.fit(X_tr, y_tr, sample_weight=sw)
    elif learner == "gbc":
        mdl = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                         learning_rate=0.05, subsample=0.8,
                                         random_state=42)
        mdl.fit(X_tr, y_tr, sample_weight=sw)
    else:
        raise ValueError(learner)
    p_te = mdl.predict_proba(X_te)[:, 1]
    return {
        "auc": roc_auc_score(y_te, p_te),
        "ap": average_precision_score(y_te, p_te),
        "logloss": log_loss(y_te, np.clip(p_te, 1e-7, 1 - 1e-7)),
        "brier": brier_score_loss(y_te, p_te),
        "mean_p": float(p_te.mean()),
        "mean_p_pos": float(p_te[y_te == 1].mean()),
        "n_features": len(features),
    }


def main():
    t0 = time.time()
    df = pd.read_csv(PANEL, parse_dates=["date"])
    df = df[df["y_drop"].isin([0, 1])].copy()
    df["y_drop"] = df["y_drop"].astype(np.int8)

    dates = np.sort(df["date"].unique())
    n = len(dates)
    t1 = dates[int(n * 0.70)]
    t2 = dates[int(n * 0.85)]
    tr = df[df["date"] < t1]
    va = df[(df["date"] >= t1) & (df["date"] < t2)]
    te = df[df["date"] >= t2]
    print(f"[abl] train={len(tr)}  val={len(va)}  test={len(te)} "
          f"(pos_tr={tr['y_drop'].mean():.4f}  pos_te={te['y_drop'].mean():.4f})")

    ALL = V1_FEATURES + RANK_FEATURES + REGIME_FEATURES + EDGAR_META_FEATURES + EDGAR_FB_FEATURES

    # Ablation configs: label -> (feature list, use_weights, learner)
    configs = {
        "v1_baseline (GBC, 17 feat, no weights)":
            (V1_FEATURES, False, "gbc"),
        "all_on (full v2)":
            (ALL, True, "hgbc"),
        "minus_sample_weights":
            (ALL, False, "hgbc"),
        "minus_rank_features":
            (V1_FEATURES + REGIME_FEATURES + EDGAR_META_FEATURES + EDGAR_FB_FEATURES, True, "hgbc"),
        "minus_regime_features":
            (V1_FEATURES + RANK_FEATURES + EDGAR_META_FEATURES + EDGAR_FB_FEATURES, True, "hgbc"),
        "minus_edgar_metadata":
            (V1_FEATURES + RANK_FEATURES + REGIME_FEATURES + EDGAR_FB_FEATURES, True, "hgbc"),
        "minus_edgar_finbert":
            (V1_FEATURES + RANK_FEATURES + REGIME_FEATURES + EDGAR_META_FEATURES, True, "hgbc"),
        "minus_all_edgar":
            (V1_FEATURES + RANK_FEATURES + REGIME_FEATURES, True, "hgbc"),
        "learner_gbc_instead_of_hgbc":
            (ALL, True, "gbc"),
        "v1_features_with_v2_learner (HGBC + weights, old 17 features)":
            (V1_FEATURES, True, "hgbc"),
    }

    rows = []
    for name, (feats, weights, learner) in configs.items():
        print(f"\n[abl] {name}  features={len(feats)}  weights={weights}  learner={learner}")
        try:
            m = _fit_and_score(feats, tr, va, te,
                               use_weights=weights, learner=learner)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        m.update({"config": name})
        rows.append(m)
        print(f"  AUC={m['auc']:.4f} AP={m['ap']:.4f} logloss={m['logloss']:.4f} "
              f"mean_p={m['mean_p']:.4f} mean_p|pos={m['mean_p_pos']:.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "ablation_table.csv", index=False)
    (OUT_DIR / "ablation.json").write_text(
        json.dumps(rows, indent=2, default=str))

    # Print summary delta vs all_on
    baseline = next((r for r in rows if r["config"] == "all_on (full v2)"), None)
    if baseline:
        print(f"\n[abl] === DELTAS vs all_on (full v2) ===")
        print(f"  all_on: AUC={baseline['auc']:.4f} AP={baseline['ap']:.4f} "
              f"logloss={baseline['logloss']:.4f} mean_p|pos={baseline['mean_p_pos']:.4f}")
        for r in rows:
            if r["config"] == "all_on (full v2)":
                continue
            print(f"  {r['config']:<55} "
                  f"ΔAUC={r['auc']-baseline['auc']:+.4f}  "
                  f"ΔAP={r['ap']-baseline['ap']:+.4f}  "
                  f"Δmean_p|pos={r['mean_p_pos']-baseline['mean_p_pos']:+.4f}")

    print(f"\n[abl] done in {time.time()-t0:.0f}s; wrote {OUT_DIR}/ablation*.csv/json")


if __name__ == "__main__":
    main()
