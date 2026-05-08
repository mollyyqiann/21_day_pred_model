"""Phase C: Predictability test — train baseline GBC for 21d-touch>=30%.

Primary head: y21 binary classifier
Auxiliary heads:
  - y5_touch (5d-touch >=30%) ablation: how much is timing vs window?
  - max_fwd21_ret regression: ranking flexibility

Reports test metrics:
  AUC, PR-AUC, log_loss, Brier, lift over base rate
  precision@top-{5,10,25} per day (cross-sectional, averaged)
  stratified AUC by rv_60 quartile (the vol-detector check)
  per-year breakdown
  sector-stratified precision@top-5
  dedup PR-AUC (one event per consecutive-positive run, 7d cooldown)
  long-only signal sanity-check (top-5 daily picks, hold 21d, vs SPY)
  no-regime ablation

Saves: models/monthly_gainer_v1.joblib
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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss,
    mean_absolute_error, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output" / "monthly_gainer"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from regime_features import REGIME_FEATS, attach_regime  # noqa: E402

FEATURES_V1 = [
    "rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
    "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60",
    "ma_stack", "up_streak", "up_bigdays_20d",
    "dist_ma60_atr", "ma60_slope_60d", "run_length",
]  # 16 base — overnight_gap excluded (forward-looking)
ALL_FEATURES = FEATURES_V1 + REGIME_FEATS  # 23

GRID = [
    {"max_depth": 3, "learning_rate": 0.05},
    {"max_depth": 3, "learning_rate": 0.03},
    {"max_depth": 4, "learning_rate": 0.05},
    {"max_depth": 4, "learning_rate": 0.03},
]
BASE_PARAMS = dict(n_estimators=400, subsample=0.8, random_state=42)


def _time_split(df, train_frac=0.70, val_frac=0.15):
    dates = np.sort(df["date"].unique())
    t1 = dates[int(len(dates) * train_frac)]
    t2 = dates[int(len(dates) * (train_frac + val_frac))]
    return (df[df["date"] < t1].copy(),
            df[(df["date"] >= t1) & (df["date"] < t2)].copy(),
            df[df["date"] >= t2].copy())


def _sample_weights(y, target_pos_frac=0.06):
    """1.28% base rate → target=0.06 ≈ 5x pos weight, ~0.95x neg."""
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
    """For each test date, take top-k by prob_col, compute mean realized y21.
    Average over days. Returns (mean_precision, n_days_with_at_least_k_rows)."""
    precs = []
    for d, g in test_df.groupby("date"):
        if len(g) < k:
            continue
        topk = g.nlargest(k, prob_col)
        precs.append(topk["y21"].mean())
    return float(np.mean(precs)) if precs else float("nan"), len(precs)


def stratified_auc(test_df, prob_col, strat_col, buckets):
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


def dedup_pr_auc(test_df, prob_col, cooldown_days=7):
    """Keep only the first row per ticker within a cooldown window of any
    prior positive event of the same ticker. Reduces overlap inflation."""
    df = test_df.sort_values(["ticker", "date"]).copy()
    df["keep"] = True
    for tk, g in df.groupby("ticker"):
        last_pos = None
        for idx, r in g.iterrows():
            if r["y21"] == 1:
                if last_pos is not None and (r["date"] - last_pos).days <= cooldown_days:
                    df.at[idx, "keep"] = False
                else:
                    last_pos = r["date"]
    sub = df[df["keep"]]
    if sub["y21"].sum() < 2:
        return None
    return {
        "n": int(len(sub)), "pos": int(sub["y21"].sum()),
        "base_rate": float(sub["y21"].mean()),
        "auc": float(roc_auc_score(sub["y21"], sub[prob_col])),
        "ap": float(average_precision_score(sub["y21"], sub[prob_col])),
    }


def long_only_sim(test_df, prob_col, k=5):
    """Each test date, pick top-k stocks by prob_col, hold 21 trading days,
    realize the return = max_fwd21_ret. Compare to SPY equivalent (proxy:
    average max_fwd21_ret across all stocks that day)."""
    rows = []
    for d, g in test_df.groupby("date"):
        if len(g) < k:
            continue
        topk = g.nlargest(k, prob_col)
        rows.append({
            "date": d,
            "topk_mean_max_ret": float(topk["max_fwd21_ret"].mean()),
            "topk_mean_end_ret": float(topk["end_of_window_ret"].mean()),
            "univ_mean_max_ret": float(g["max_fwd21_ret"].mean()),
            "univ_mean_end_ret": float(g["end_of_window_ret"].mean()),
            "topk_pos_rate": float(topk["y21"].mean()),
        })
    if not rows:
        return None
    sim = pd.DataFrame(rows)
    return {
        "n_days": int(len(sim)),
        "topk_mean_max_ret": float(sim["topk_mean_max_ret"].mean()),
        "topk_mean_end_ret": float(sim["topk_mean_end_ret"].mean()),
        "univ_mean_max_ret": float(sim["univ_mean_max_ret"].mean()),
        "univ_mean_end_ret": float(sim["univ_mean_end_ret"].mean()),
        "lift_max_ret": float(sim["topk_mean_max_ret"].mean() - sim["univ_mean_max_ret"].mean()),
        "lift_end_ret": float(sim["topk_mean_end_ret"].mean() - sim["univ_mean_end_ret"].mean()),
        "topk_avg_pos_rate": float(sim["topk_pos_rate"].mean()),
    }


def fit_grid(X_tr, y_tr, X_va, y_va, sw_tr):
    best = None
    for params in GRID:
        gbc = GradientBoostingClassifier(**BASE_PARAMS, **params)
        gbc.fit(X_tr, y_tr, sample_weight=sw_tr)
        p_va = gbc.predict_proba(X_va)[:, 1]
        ap = average_precision_score(y_va, p_va)
        auc = roc_auc_score(y_va, p_va)
        print(f"  grid {params}: val AUC={auc:.4f} PR-AUC={ap:.4f}")
        if best is None or ap > best["val_ap"]:
            best = {"model": gbc, "params": params, "val_ap": ap, "val_auc": auc}
    print(f"  best: {best['params']} val PR-AUC={best['val_ap']:.4f}")
    return best


def main():
    t0 = time.time()
    panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    print(f"[83] loaded panel: {len(panel):,} rows")

    # attach regime
    print("[83] attaching regime features ...")
    panel = attach_regime(panel)

    # filter labeled rows + complete features
    lab = panel[panel["y21"] >= 0].dropna(subset=ALL_FEATURES).reset_index(drop=True)
    print(f"[83] labeled+complete: {len(lab):,} rows  base_rate={lab['y21'].mean():.4%}")

    train, val, test = _time_split(lab)
    print(f"[83] split: train={len(train):,} ({train['date'].min().date()}..{train['date'].max().date()})  "
          f"val={len(val):,} ({val['date'].min().date()}..{val['date'].max().date()})  "
          f"test={len(test):,} ({test['date'].min().date()}..{test['date'].max().date()})")
    print(f"[83] base rates: train={train['y21'].mean():.4%}  val={val['y21'].mean():.4%}  "
          f"test={test['y21'].mean():.4%}")

    # impute medians from train
    medians = train[ALL_FEATURES].median(numeric_only=True)
    X_tr = train[ALL_FEATURES].fillna(medians).values
    X_va = val[ALL_FEATURES].fillna(medians).values
    X_te = test[ALL_FEATURES].fillna(medians).values
    y_tr = train["y21"].values.astype(np.int8)
    y_va = val["y21"].values.astype(np.int8)
    y_te = test["y21"].values.astype(np.int8)

    sw_tr = _sample_weights(y_tr, target_pos_frac=0.06)

    # ============ PRIMARY HEAD ============
    print("\n[83] === PRIMARY HEAD: y21 ===")
    best = fit_grid(X_tr, y_tr, X_va, y_va, sw_tr)
    raw = best["model"]

    # isotonic calibrate
    cal = CalibratedClassifierCV(estimator=raw, method="isotonic", cv="prefit")
    cal.fit(X_va, y_va)

    p_va = cal.predict_proba(X_va)[:, 1]
    p_te = cal.predict_proba(X_te)[:, 1]

    metrics_val = evaluate(y_va, p_va, "val")
    metrics_test = evaluate(y_te, p_te, "test")
    print(f"\n[83] val   {metrics_val}")
    print(f"[83] test  {metrics_test}")

    # ============ TEST DIAGNOSTICS ============
    test_eval = test.copy()
    test_eval["prob"] = p_te

    # Precision @ top-k per day
    p5, n5 = precision_at_topk_per_day(test_eval, "prob", 5)
    p10, _ = precision_at_topk_per_day(test_eval, "prob", 10)
    p25, _ = precision_at_topk_per_day(test_eval, "prob", 25)
    print(f"\n[83] precision@top-5 per day: {p5:.3%}  (across {n5} test days, "
          f"lift vs base = {p5/metrics_test['base_rate']:.2f}x)")
    print(f"[83] precision@top-10 per day: {p10:.3%}  "
          f"(lift {p10/metrics_test['base_rate']:.2f}x)")
    print(f"[83] precision@top-25 per day: {p25:.3%}  "
          f"(lift {p25/metrics_test['base_rate']:.2f}x)")

    # Stratified AUC by rv_60 quartile
    rv_q = train["rv_60"].quantile([0.25, 0.5, 0.75]).values
    buckets = [
        ("Q1_low_vol",   test_eval["rv_60"] < rv_q[0]),
        ("Q2",           (test_eval["rv_60"] >= rv_q[0]) & (test_eval["rv_60"] < rv_q[1])),
        ("Q3",           (test_eval["rv_60"] >= rv_q[1]) & (test_eval["rv_60"] < rv_q[2])),
        ("Q4_high_vol",  test_eval["rv_60"] >= rv_q[2]),
    ]
    rv_strat = stratified_auc(test_eval, "prob", "rv_60", buckets)
    print(f"\n[83] stratified AUC by rv_60 quartile (vol-detector check):")
    for lbl, m in rv_strat.items():
        a = m["auc"] if m["auc"] is not None else float("nan")
        ap = m["ap"] if m["ap"] is not None else float("nan")
        print(f"   {lbl:<14} n={m['n']:,} pos={m['pos']:>4} AUC={a:.3f} PR-AUC={ap:.3f}")

    # Per-year breakdown on test
    test_eval["year"] = test_eval["date"].dt.year
    year_strat = {}
    for y, g in test_eval.groupby("year"):
        if g["y21"].nunique() < 2 or g["y21"].sum() < 5:
            year_strat[int(y)] = {"n": int(len(g)), "pos": int(g["y21"].sum()), "auc": None, "ap": None}
        else:
            year_strat[int(y)] = {
                "n": int(len(g)), "pos": int(g["y21"].sum()),
                "auc": float(roc_auc_score(g["y21"], g["prob"])),
                "ap": float(average_precision_score(g["y21"], g["prob"])),
            }
    print(f"\n[83] per-year test:")
    for y, m in year_strat.items():
        a = m["auc"] if m["auc"] is not None else float("nan")
        ap = m["ap"] if m["ap"] is not None else float("nan")
        print(f"   {y}: n={m['n']:,} pos={m['pos']:>4} AUC={a:.3f} PR-AUC={ap:.3f}")

    # Sector-stratified precision@top-5 per sector (across all test days)
    sec_prec = {}
    for s, g in test_eval.groupby("sector"):
        if len(g) < 50 or g["y21"].sum() < 5:
            continue
        # daily top-5 within sector
        precs = []
        for d, sg in g.groupby("date"):
            if len(sg) < 3:
                continue
            k = min(3, len(sg))
            topk = sg.nlargest(k, "prob")
            precs.append(topk["y21"].mean())
        if precs:
            sec_prec[s] = {"n": len(g), "pos": int(g["y21"].sum()),
                           "base": float(g["y21"].mean()),
                           "p_at_top3": float(np.mean(precs))}
    print(f"\n[83] sector-stratified precision@top-3 per day:")
    for s, m in sorted(sec_prec.items(), key=lambda kv: -kv[1]["p_at_top3"]):
        print(f"   {s:<24} base={m['base']:.3%}  p@3={m['p_at_top3']:.3%}  lift={m['p_at_top3']/max(m['base'],1e-9):.2f}x")

    # Dedup PR-AUC
    dedup = dedup_pr_auc(test_eval, "prob")
    print(f"\n[83] dedup (event-level, 7d cooldown): {dedup}")

    # Long-only sim
    sim = long_only_sim(test_eval, "prob", k=5)
    print(f"\n[83] long-only top-5 sim: {sim}")

    # ============ NO-REGIME ABLATION ============
    print("\n[83] === NO-REGIME ABLATION (base 16 features only) ===")
    medians_base = train[FEATURES_V1].median(numeric_only=True)
    Xtr_b = train[FEATURES_V1].fillna(medians_base).values
    Xva_b = val[FEATURES_V1].fillna(medians_base).values
    Xte_b = test[FEATURES_V1].fillna(medians_base).values
    gbc_b = GradientBoostingClassifier(**BASE_PARAMS, **best["params"])
    gbc_b.fit(Xtr_b, y_tr, sample_weight=sw_tr)
    cal_b = CalibratedClassifierCV(estimator=gbc_b, method="isotonic", cv="prefit")
    cal_b.fit(Xva_b, y_va)
    p_te_b = cal_b.predict_proba(Xte_b)[:, 1]
    metrics_test_b = evaluate(y_te, p_te_b, "test_no_regime")
    print(f"[83] no-regime test: AUC={metrics_test_b['auc']:.3f}  "
          f"PR-AUC={metrics_test_b['ap']:.3f}  "
          f"(delta vs full: AUC={metrics_test_b['auc']-metrics_test['auc']:+.3f}, "
          f"PR-AUC={metrics_test_b['ap']-metrics_test['ap']:+.3f})")

    # ============ AUX HEAD: 5d-touch ABLATION ============
    print("\n[83] === AUX HEAD: y5_touch (timing test) ===")
    lab5 = panel[panel["y5_touch"] >= 0].dropna(subset=ALL_FEATURES).reset_index(drop=True)
    tr5, va5, te5 = _time_split(lab5)
    medians5 = tr5[ALL_FEATURES].median(numeric_only=True)
    Xtr5 = tr5[ALL_FEATURES].fillna(medians5).values
    Xva5 = va5[ALL_FEATURES].fillna(medians5).values
    Xte5 = te5[ALL_FEATURES].fillna(medians5).values
    ytr5 = tr5["y5_touch"].values.astype(np.int8)
    yva5 = va5["y5_touch"].values.astype(np.int8)
    yte5 = te5["y5_touch"].values.astype(np.int8)
    sw5 = _sample_weights(ytr5, target_pos_frac=0.02)  # base 0.08% — gentler weight
    gbc5 = GradientBoostingClassifier(**BASE_PARAMS, **best["params"])
    gbc5.fit(Xtr5, ytr5, sample_weight=sw5)
    cal5 = CalibratedClassifierCV(estimator=gbc5, method="isotonic", cv="prefit")
    cal5.fit(Xva5, yva5)
    p_te5 = cal5.predict_proba(Xte5)[:, 1]
    metrics_5d = evaluate(yte5, p_te5, "test_5d_touch")
    print(f"[83] 5d-touch test: {metrics_5d}")
    print(f"[83] 21d AUC={metrics_test['auc']:.3f} vs 5d AUC={metrics_5d['auc']:.3f}: "
          f"{'21d signal is mostly window-vol' if metrics_5d['auc'] < 0.6 else 'real timing skill present'}")

    # ============ REGRESSION HEAD: max_fwd21_ret ============
    print("\n[83] === REGRESSION HEAD: max_fwd21_ret ===")
    reg_data = panel[panel["max_fwd21_ret"].notna()].dropna(subset=ALL_FEATURES).reset_index(drop=True)
    rtr, rva, rte = _time_split(reg_data)
    Xrtr = rtr[ALL_FEATURES].fillna(medians).values
    Xrte = rte[ALL_FEATURES].fillna(medians).values
    yrtr = rtr["max_fwd21_ret"].values
    yrte = rte["max_fwd21_ret"].values
    reg = GradientBoostingRegressor(loss="absolute_error", n_estimators=400,
                                    max_depth=best["params"]["max_depth"],
                                    learning_rate=best["params"]["learning_rate"],
                                    subsample=0.8, random_state=42)
    reg.fit(Xrtr, yrtr)
    pred_r = reg.predict(Xrte)
    mae = float(mean_absolute_error(yrte, pred_r))
    # rank correlation between predicted max return and binary touch
    rte["pred_max"] = pred_r
    p_at_top5_r, _ = precision_at_topk_per_day(rte.assign(prob=pred_r).rename(columns={"prob": "prob_r"}), "prob_r", 5) \
        if False else (None, None)
    # actually just compute it directly
    precs = []
    for d, g in rte.groupby("date"):
        if len(g) < 5:
            continue
        topk = g.nlargest(5, "pred_max")
        if "y21" in topk.columns:
            precs.append(topk["y21"].mean())
    p_at_top5_r = float(np.mean(precs)) if precs else None
    print(f"[83] regression test: MAE={mae:.4f} (mean |error| in ret units)  "
          f"top-5 daily picks by pred_max touch-rate={p_at_top5_r}")

    # ============ FEATURE IMPORTANCE ============
    imp = pd.Series(raw.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    print(f"\n[83] feature importance (top 10):")
    print(imp.head(10).to_string())

    # ============ LEAKAGE CHECK: rank corr between each feature and y21 on train ============
    from scipy.stats import spearmanr
    leak = {}
    for f in ALL_FEATURES:
        x = train[f].fillna(medians[f]).values
        try:
            r, _ = spearmanr(x, y_tr)
            leak[f] = float(r) if r is not None else None
        except Exception:
            leak[f] = None
    max_abs_corr = max((abs(v) for v in leak.values() if v is not None), default=0.0)
    print(f"\n[83] leakage check: max |spearman(feat, y21)| on train = {max_abs_corr:.3f}")
    if max_abs_corr > 0.6:
        print(f"[83] WARNING: high feature-label correlation, suspicious")

    # ============ SAVE ARTIFACTS ============
    artifact = {
        "raw_gbc": raw, "calibrator": cal,
        "feats": ALL_FEATURES,
        "impute_medians": medians.to_dict(),
        "best_params": best["params"],
        "train_pos_rate": float(y_tr.mean()),
        "val_pos_rate": float(y_va.mean()),
        "test_pos_rate": float(y_te.mean()),
        "metrics": {
            "val": metrics_val, "test": metrics_test,
            "test_no_regime": metrics_test_b,
            "test_5d_touch": metrics_5d,
            "regression_mae": mae,
            "regression_top5_touch_rate": p_at_top5_r,
        },
        "test_precisions": {
            "p_at_top5": p5, "p_at_top10": p10, "p_at_top25": p25,
            "n_test_days": n5,
        },
        "rv_strat": rv_strat,
        "year_strat": year_strat,
        "sector_precision": sec_prec,
        "dedup_metrics": dedup,
        "long_only_sim": sim,
        "feature_importance": imp.to_dict(),
        "leakage_max_abs_spearman": max_abs_corr,
    }

    joblib.dump(artifact, MODELS / "monthly_gainer_v1.joblib")
    # also save metrics summary as json (joblib not great for human-reading)
    metrics_json = {k: v for k, v in artifact.items()
                    if k not in ("raw_gbc", "calibrator")}
    (OUT / "model_metrics.json").write_text(json.dumps(metrics_json, indent=2, default=str))
    print(f"\n[83] saved {MODELS / 'monthly_gainer_v1.joblib'}")
    print(f"[83] saved {OUT / 'model_metrics.json'}")
    print(f"[83] elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
