"""Phase E.3: Train + evaluate monthly gainer model on smallcap and union panels.

Two evaluations:
  (1) "transfer test": apply the SP500-trained model directly to smallcap test fold.
      Tells us how well the SP500 features generalize off-universe.
  (2) "union retrain": retrain on the union (SP500 + smallcap) and re-evaluate.
      Same train/val/test chronological split.

Reports the same metrics as Phase C: AUC, PR-AUC, lift, precision@top-k, etc.

Outputs:
  models/monthly_gainer_smallcap_v1.joblib
  models/monthly_gainer_union_v1.joblib
  output/monthly_gainer/smallcap_metrics.json
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

FEATURES_V1 = [
    "rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
    "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60",
    "ma_stack", "up_streak", "up_bigdays_20d",
    "dist_ma60_atr", "ma60_slope_60d", "run_length",
]
ALL_FEATURES = FEATURES_V1 + REGIME_FEATS  # 23


def _time_split(df, train_frac=0.70, val_frac=0.15):
    dates = np.sort(df["date"].unique())
    t1 = dates[int(len(dates) * train_frac)]
    t2 = dates[int(len(dates) * (train_frac + val_frac))]
    return (df[df["date"] < t1].copy(),
            df[(df["date"] >= t1) & (df["date"] < t2)].copy(),
            df[df["date"] >= t2].copy())


def _sample_weights(y, target_pos_frac=0.10):
    p = y.mean()
    if p <= 0 or p >= 1:
        return np.ones(len(y))
    w_pos = target_pos_frac / p
    w_neg = (1 - target_pos_frac) / (1 - p)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def evaluate(y, p, label=""):
    base = float(np.mean(y)) if len(y) else float("nan")
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y, p))
    except ValueError:
        ap = float("nan")
    ll = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
    br = float(brier_score_loss(y, p))
    return {
        "label": label, "n": int(len(y)), "pos": int(np.sum(y)),
        "base_rate": base, "auc": auc, "ap": ap,
        "ap_lift": ap / base if base > 0 else float("nan"),
        "log_loss": ll, "brier": br,
    }


def precision_at_topk_per_day(test_df, prob_col, k):
    precs = []
    for d, g in test_df.groupby("date"):
        if len(g) < k:
            continue
        topk = g.nlargest(k, prob_col)
        precs.append(topk["y21"].mean())
    return float(np.mean(precs)) if precs else float("nan"), len(precs)


def stratified_auc(test_df, prob_col, buckets):
    out = {}
    for lbl, mask in buckets:
        sub = test_df[mask]
        if sub["y21"].sum() < 5 or sub["y21"].nunique() < 2:
            out[lbl] = {"n": int(len(sub)), "pos": int(sub["y21"].sum()), "auc": None, "ap": None}
            continue
        try:
            a = float(roc_auc_score(sub["y21"], sub[prob_col]))
            ap = float(average_precision_score(sub["y21"], sub[prob_col]))
        except ValueError:
            a = ap = None
        out[lbl] = {"n": int(len(sub)), "pos": int(sub["y21"].sum()), "auc": a, "ap": ap}
    return out


def long_only_sim(test_df, prob_col, k=5):
    rows = []
    for d, g in test_df.groupby("date"):
        if len(g) < k:
            continue
        topk = g.nlargest(k, prob_col)
        rows.append({
            "topk_max_ret": float(topk["max_fwd21_ret"].mean()),
            "topk_end_ret": float(topk["end_of_window_ret"].mean()),
            "univ_max_ret": float(g["max_fwd21_ret"].mean()),
            "univ_end_ret": float(g["end_of_window_ret"].mean()),
            "topk_pos_rate": float(topk["y21"].mean()),
        })
    if not rows:
        return None
    sim = pd.DataFrame(rows)
    return {
        "n_days": int(len(sim)),
        "topk_max_ret": float(sim["topk_max_ret"].mean()),
        "topk_end_ret": float(sim["topk_end_ret"].mean()),
        "univ_max_ret": float(sim["univ_max_ret"].mean()),
        "univ_end_ret": float(sim["univ_end_ret"].mean()),
        "topk_pos_rate": float(sim["topk_pos_rate"].mean()),
    }


def full_eval(test_df, label=""):
    metrics = evaluate(test_df["y21"].values, test_df["prob"].values, label)
    p5, n5 = precision_at_topk_per_day(test_df, "prob", 5)
    p10, _ = precision_at_topk_per_day(test_df, "prob", 10)
    p25, _ = precision_at_topk_per_day(test_df, "prob", 25)
    sim = long_only_sim(test_df, "prob", 5)
    return {**metrics,
            "p_at_top5": p5, "p_at_top10": p10, "p_at_top25": p25,
            "n_test_days": n5, "long_only_sim": sim}


def main():
    t0 = time.time()
    print("[86] loading panels ...")
    sp_panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    sc_panel = pd.read_csv(DATA / "monthly_gainer_panel_smallcap.csv", parse_dates=["date"])
    print(f"[86] sp500 panel: {len(sp_panel):,} rows, {sp_panel['ticker'].nunique()} tickers")
    print(f"[86] smallcap panel: {len(sc_panel):,} rows, {sc_panel['ticker'].nunique()} tickers")

    # Attach regime to both
    sp_panel = attach_regime(sp_panel)
    sc_panel = attach_regime(sc_panel)

    # Filter complete + labeled
    sp_lab = sp_panel[sp_panel["y21"] >= 0].dropna(subset=ALL_FEATURES).reset_index(drop=True)
    sc_lab = sc_panel[sc_panel["y21"] >= 0].dropna(subset=ALL_FEATURES).reset_index(drop=True)
    print(f"[86] sp500 labeled: {len(sp_lab):,} ({sp_lab['y21'].mean():.4%} pos)")
    print(f"[86] smallcap labeled: {len(sc_lab):,} ({sc_lab['y21'].mean():.4%} pos)")

    union = pd.concat([sp_lab.assign(univ="sp500"),
                       sc_lab.assign(univ="smallcap")],
                      ignore_index=True)
    print(f"[86] union labeled: {len(union):,} ({union['y21'].mean():.4%} pos)")

    # ===========  EVAL 1: TRANSFER TEST (sp500 model -> smallcap test fold)  ===========
    print("\n[86] === EVAL 1: TRANSFER TEST (sp500 model -> smallcap) ===")
    art = joblib.load(MODELS / "monthly_gainer_v1.joblib")
    medians = pd.Series(art["impute_medians"])
    sc_train, sc_val, sc_test = _time_split(sc_lab)
    print(f"[86] smallcap split: train={len(sc_train):,} val={len(sc_val):,} test={len(sc_test):,}")
    print(f"[86] smallcap test base rate: {sc_test['y21'].mean():.4%}")

    X_sc_test = sc_test[ALL_FEATURES].fillna(medians).values
    sc_test["prob"] = art["calibrator"].predict_proba(X_sc_test)[:, 1]
    transfer_metrics = full_eval(sc_test, "transfer_smallcap")
    print(f"[86] transfer metrics: {transfer_metrics}")

    # rv_60 stratification on smallcap
    rv_q = sc_train["rv_60"].quantile([0.25, 0.5, 0.75]).values
    buckets_sc = [
        ("Q1", sc_test["rv_60"] < rv_q[0]),
        ("Q2", (sc_test["rv_60"] >= rv_q[0]) & (sc_test["rv_60"] < rv_q[1])),
        ("Q3", (sc_test["rv_60"] >= rv_q[1]) & (sc_test["rv_60"] < rv_q[2])),
        ("Q4", sc_test["rv_60"] >= rv_q[2]),
    ]
    transfer_rv_strat = stratified_auc(sc_test, "prob", buckets_sc)
    print(f"[86] transfer rv_60 stratified: {transfer_rv_strat}")

    # ===========  EVAL 2: SMALLCAP-ONLY RETRAIN  ===========
    print("\n[86] === EVAL 2: SMALLCAP-ONLY RETRAIN ===")
    sct, scv, scte = sc_train, sc_val, sc_test
    medians_sc = sct[ALL_FEATURES].median(numeric_only=True)
    Xtr = sct[ALL_FEATURES].fillna(medians_sc).values
    Xva = scv[ALL_FEATURES].fillna(medians_sc).values
    Xte = scte[ALL_FEATURES].fillna(medians_sc).values
    ytr = sct["y21"].values.astype(np.int8)
    yva = scv["y21"].values.astype(np.int8)
    yte = scte["y21"].values.astype(np.int8)
    sw = _sample_weights(ytr, target_pos_frac=0.08)  # smallcap base ~3-5%, gentler weight
    gbc_sc = GradientBoostingClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42)
    gbc_sc.fit(Xtr, ytr, sample_weight=sw)
    cal_sc = CalibratedClassifierCV(estimator=gbc_sc, method="isotonic", cv="prefit")
    cal_sc.fit(Xva, yva)
    p_te = cal_sc.predict_proba(Xte)[:, 1]
    scte["prob"] = p_te
    smallcap_metrics = full_eval(scte, "smallcap_retrain")
    print(f"[86] smallcap-retrain metrics: {smallcap_metrics}")
    smallcap_rv_strat = stratified_auc(scte, "prob", buckets_sc)
    print(f"[86] smallcap-retrain rv_60 strat: {smallcap_rv_strat}")

    joblib.dump({
        "raw_gbc": gbc_sc, "calibrator": cal_sc,
        "feats": ALL_FEATURES,
        "impute_medians": medians_sc.to_dict(),
        "metrics": smallcap_metrics, "rv_strat": smallcap_rv_strat,
    }, MODELS / "monthly_gainer_smallcap_v1.joblib")
    print(f"[86] saved {MODELS / 'monthly_gainer_smallcap_v1.joblib'}")

    # ===========  EVAL 3: UNION RETRAIN  ===========
    print("\n[86] === EVAL 3: UNION RETRAIN ===")
    ut, uv, ute = _time_split(union)
    print(f"[86] union split: train={len(ut):,} val={len(uv):,} test={len(ute):,}")
    medians_u = ut[ALL_FEATURES].median(numeric_only=True)
    Xtr_u = ut[ALL_FEATURES].fillna(medians_u).values
    Xva_u = uv[ALL_FEATURES].fillna(medians_u).values
    Xte_u = ute[ALL_FEATURES].fillna(medians_u).values
    ytr_u = ut["y21"].values.astype(np.int8)
    yva_u = uv["y21"].values.astype(np.int8)
    yte_u = ute["y21"].values.astype(np.int8)
    sw_u = _sample_weights(ytr_u, target_pos_frac=0.10)
    gbc_u = GradientBoostingClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42)
    gbc_u.fit(Xtr_u, ytr_u, sample_weight=sw_u)
    cal_u = CalibratedClassifierCV(estimator=gbc_u, method="isotonic", cv="prefit")
    cal_u.fit(Xva_u, yva_u)
    p_te_u = cal_u.predict_proba(Xte_u)[:, 1]
    ute["prob"] = p_te_u
    union_metrics = full_eval(ute, "union")
    print(f"[86] union metrics: {union_metrics}")

    # Per-univ breakdown of union test
    union_breakdown = {}
    for uval in ["sp500", "smallcap"]:
        sub = ute[ute["univ"] == uval]
        if len(sub) < 100:
            continue
        union_breakdown[uval] = full_eval(sub, f"union_{uval}_subset")
    print(f"[86] union breakdown by source: {union_breakdown}")

    # rv_60 strat on union test
    rv_q_u = ut["rv_60"].quantile([0.25, 0.5, 0.75]).values
    buckets_u = [
        ("Q1", ute["rv_60"] < rv_q_u[0]),
        ("Q2", (ute["rv_60"] >= rv_q_u[0]) & (ute["rv_60"] < rv_q_u[1])),
        ("Q3", (ute["rv_60"] >= rv_q_u[1]) & (ute["rv_60"] < rv_q_u[2])),
        ("Q4", ute["rv_60"] >= rv_q_u[2]),
    ]
    union_rv_strat = stratified_auc(ute, "prob", buckets_u)
    print(f"[86] union rv_60 strat: {union_rv_strat}")

    joblib.dump({
        "raw_gbc": gbc_u, "calibrator": cal_u,
        "feats": ALL_FEATURES,
        "impute_medians": medians_u.to_dict(),
        "metrics": union_metrics,
        "breakdown": union_breakdown,
        "rv_strat": union_rv_strat,
    }, MODELS / "monthly_gainer_union_v1.joblib")
    print(f"[86] saved {MODELS / 'monthly_gainer_union_v1.joblib'}")

    # ===========  SAVE METRICS JSON  ===========
    summary = {
        "transfer_smallcap": transfer_metrics,
        "transfer_rv_strat": transfer_rv_strat,
        "smallcap_retrain": smallcap_metrics,
        "smallcap_retrain_rv_strat": smallcap_rv_strat,
        "union_retrain": union_metrics,
        "union_breakdown": union_breakdown,
        "union_rv_strat": union_rv_strat,
        "panel_sizes": {
            "sp500_labeled": int(len(sp_lab)),
            "smallcap_labeled": int(len(sc_lab)),
            "union_labeled": int(len(union)),
        },
        "smallcap_base_rate": float(sc_lab["y21"].mean()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smallcap_metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[86] saved {OUT / 'smallcap_metrics.json'}")
    print(f"[86] elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
