"""V2 drop15 — explore two residualization styles vs v1.

Why: drop15 v1 has Spearman 0.96 with prob_up_v3 on smallcap → not a useful
risk filter. On SP500, v1 has a different failure mode: calibrator saturates
at ~0.09 because top features are all macro (identical across SP500 names on
a given date) → no top-end ranking. v2 adds cross-sectional ranks +
drawdown features that vary within a date, plus residualizes vs prob_up_v3.

Variants:
  v2a: adds prob_up_v3 as a feature so the GBC can route around it.
  v2b: trains on residualized label (y - baseline_drop_rate_by_prob_up_decile).

Outputs:
  models/monthly_gainer_drop15_v2{a,b}_{universe}.joblib
  output/monthly_gainer/drop15_v2_compare_{universe}.json

Usage:
  python 95_train_drop15_v2.py smallcap
  python 95_train_drop15_v2.py sp500
"""

import sys; sys.stdout.reconfigure(line_buffering=True)
import json, time, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              log_loss, roc_auc_score)
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output" / "monthly_gainer"

sys.path.insert(0, str(ROOT / "code"))
from regime_features import REGIME_FEATS, attach_regime  # noqa

# ---- features ----------------------------------------------------------------

V1_TECH = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
           "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60",
           "ma_stack", "up_streak", "up_bigdays_20d",
           "dist_ma60_atr", "ma60_slope_60d", "run_length"]
CATALYST = ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
            "news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
            "ma_news_5d", "ma_news_20d", "sector_pop_5d"]
XRANK = ["rsi_14_xrank", "rv_60_xrank", "ma60_slope_xrank",
         "ret_20d_xrank", "atr_pct_xrank", "dist_ma60_atr_xrank",
         "ret_60d_xrank"]
DRAWDOWN = ["dd_60d", "dist_52w_high", "weeks_below_ma60"]

BASE_FEATURES = V1_TECH + REGIME_FEATS + CATALYST + XRANK + DRAWDOWN  # 39
V2A_FEATURES = BASE_FEATURES + ["prob_up_v3"]                          # 40
V2B_FEATURES = BASE_FEATURES                                            # 39


def add_y21_drop15(panel):
    out = []
    for tk, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True).copy()
        c = g["close"].values.astype(float); n = len(c)
        y = np.full(n, -1, dtype=np.int8)
        max_dd = np.full(n, np.nan, dtype=np.float32)
        for t in range(n - 1):
            ct = c[t]
            if ct <= 0 or not np.isfinite(ct): continue
            if t + 21 < n:
                dd = c[t+1:t+22].min() / ct - 1.0
                max_dd[t] = dd
                y[t] = 1 if dd <= -0.15 else 0
        g["y_drop15"] = y
        g["max_fwd21_drawdown"] = max_dd
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_features(panel: pd.DataFrame, catalyst: pd.DataFrame) -> pd.DataFrame:
    """Attach regime, catalyst, xrank, drawdown features."""
    panel = attach_regime(panel)
    panel = panel.merge(catalyst, on=["ticker", "date"], how="left")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Lagged returns needed for xranks + drawdown
    panel["close_5d"] = panel.groupby("ticker")["close"].shift(5)
    panel["ret_5d_lag"] = panel["close"] / panel["close_5d"] - 1.0
    panel["close_20d"] = panel.groupby("ticker")["close"].shift(20)
    panel["ret_20d_lag"] = panel["close"] / panel["close_20d"] - 1.0
    panel["close_60d"] = panel.groupby("ticker")["close"].shift(60)
    panel["ret_60d_lag"] = panel["close"] / panel["close_60d"] - 1.0

    # Cross-sectional ranks (per-date pct rank — ties broken)
    for raw, name in [("rsi_14", "rsi_14_xrank"), ("rv_60", "rv_60_xrank"),
                      ("ma60_slope_60d", "ma60_slope_xrank"),
                      ("ret_20d_lag", "ret_20d_xrank"),
                      ("atr_pct", "atr_pct_xrank"),
                      ("dist_ma60_atr", "dist_ma60_atr_xrank"),
                      ("ret_60d_lag", "ret_60d_xrank")]:
        panel[name] = panel.groupby("date")[raw].rank(pct=True)

    # Drawdown / "wounded" features
    panel["max_252d"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.rolling(252, min_periods=60).max())
    panel["max_60d"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.rolling(60, min_periods=20).max())
    panel["min_60d"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.rolling(60, min_periods=20).min())
    panel["dd_60d"] = panel["min_60d"] / panel["max_60d"] - 1.0  # negative
    panel["dist_52w_high"] = panel["close"] / panel["max_252d"] - 1.0  # negative

    # Weeks below MA60: rolling count of (close < ma60) over 60d
    panel["ma60"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.rolling(60, min_periods=10).mean())
    panel["below_ma60"] = (panel["close"] < panel["ma60"]).astype(np.int8)
    panel["weeks_below_ma60"] = panel.groupby("ticker")["below_ma60"].transform(
        lambda s: s.rolling(60, min_periods=10).sum() / 5.0)  # → "weeks"

    # Catalyst NaN → 0 (means "no news")
    for c in CATALYST:
        if c not in panel.columns:
            panel[c] = 0.0
        panel[c] = panel[c].fillna(0)

    # xrank NaN → 0.5
    for c in XRANK:
        panel[c] = panel[c].fillna(0.5)

    return panel


def score_prob_up(panel: pd.DataFrame, up_model: dict) -> np.ndarray:
    feats = up_model["feats"]
    med = pd.Series(up_model["impute_medians"])
    # ensure all expected columns exist
    for c in feats:
        if c not in panel.columns:
            panel[c] = med.get(c, 0.0)
    X = panel[feats].fillna(med).values
    return up_model["calibrator"].predict_proba(X)[:, 1]


def time_split(df, train_frac=0.70, val_frac=0.15):
    dates = np.sort(df["date"].unique())
    t1 = dates[int(len(dates) * train_frac)]
    t2 = dates[int(len(dates) * (train_frac + val_frac))]
    return (df[df["date"] < t1].copy(),
            df[(df["date"] >= t1) & (df["date"] < t2)].copy(),
            df[df["date"] >= t2].copy())


def sample_weights(y, target_pos_frac=0.15):
    p = y.mean()
    if p <= 0 or p >= 1: return np.ones(len(y))
    w_pos = target_pos_frac / p
    w_neg = (1 - target_pos_frac) / (1 - p)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def evaluate_classifier(y, p):
    base = float(np.mean(y))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    ap = float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")
    ll = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
    br = float(brier_score_loss(y, p))
    return {"n": int(len(y)), "pos": int(y.sum()),
            "base_rate": base, "auc": auc, "ap": ap,
            "ap_lift": ap / base if base > 0 else float("nan"),
            "log_loss": ll, "brier": br}


def conditional_metrics(p_drop, prob_up, y, name):
    """Realized drop rate among top-up names with high vs low new drop signal."""
    df = pd.DataFrame({"p_drop": p_drop, "p_up": prob_up, "y": y})
    top_up = df[df["p_up"] >= df["p_up"].quantile(0.90)]
    if len(top_up) < 50:
        return {"name": name, "note": "too few top-up rows"}
    base_top_up = top_up["y"].mean()
    hi_drop = top_up[top_up["p_drop"] >= top_up["p_drop"].quantile(0.50)]
    lo_drop = top_up[top_up["p_drop"] < top_up["p_drop"].quantile(0.50)]
    return {
        "name": name,
        "n_top_up_decile": int(len(top_up)),
        "base_drop_in_top_up": float(base_top_up),
        "drop_rate_top_up_AND_hi_drop": float(hi_drop["y"].mean()) if len(hi_drop) else float("nan"),
        "drop_rate_top_up_AND_lo_drop": float(lo_drop["y"].mean()) if len(lo_drop) else float("nan"),
        "filter_lift": (float(hi_drop["y"].mean()) / base_top_up) if base_top_up > 0 else float("nan"),
        "spearman_p_drop_p_up": float(spearmanr(p_drop, prob_up)[0]),
    }


def inspect_calibrator_plateau(cal_model):
    """Return the # of distinct calibrated values + max."""
    ic = cal_model.calibrated_classifiers_[0]
    iso = ic.calibrators[0]
    n_unique = len(np.unique(np.round(iso.y_thresholds_, 5)))
    return {"isotonic_unique_values": int(n_unique),
            "isotonic_max": float(iso.y_thresholds_.max()),
            "isotonic_n_knots": int(len(iso.y_thresholds_))}


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "smallcap"
    assert universe in ("smallcap", "sp500"), f"unknown universe: {universe}"
    suffix = "_smallcap" if universe == "smallcap" else ""

    t0 = time.time()
    print(f"[95] universe={universe}")
    print(f"[95] loading panel + catalyst")
    panel = pd.read_csv(DATA / f"monthly_gainer_panel{suffix}.csv", parse_dates=["date"])
    cat_path = DATA / f"catalyst_features{suffix or '_sp500'}.csv"
    catalyst = pd.read_csv(cat_path, parse_dates=["date"])

    print("[95] building features")
    panel = build_features(panel, catalyst[["ticker", "date"] + CATALYST])
    panel = add_y21_drop15(panel)

    print("[95] scoring prob_up_v3 for residualization")
    up_model = joblib.load(MODELS / f"monthly_gainer_v3_{universe}.joblib")
    panel["prob_up_v3"] = np.nan
    needed = up_model["feats"]
    have = panel.dropna(subset=[c for c in needed if c in panel.columns]).index
    panel.loc[have, "prob_up_v3"] = score_prob_up(panel.loc[have].copy(), up_model)
    print(f"[95] prob_up_v3: scored {panel['prob_up_v3'].notna().sum():,} / {len(panel):,} rows")

    # Labeled set: drop unlabeled, drop rows missing any base feature
    lab = panel[panel["y_drop15"] >= 0].dropna(
        subset=BASE_FEATURES + ["prob_up_v3"]
    ).reset_index(drop=True)
    print(f"[95] labeled+complete: {len(lab):,}  base_drop={lab['y_drop15'].mean():.4%}")

    train, val, test = time_split(lab)
    print(f"[95] split: train={len(train):,} val={len(val):,} test={len(test):,}  "
          f"test_base={test['y_drop15'].mean():.4%}")

    # ----- v2a: prob_up as feature, binary classifier ------------------------
    print("\n" + "=" * 60)
    print("[v2a] training: BASE + prob_up_v3 as feature, binary GBC")
    print("=" * 60)
    feats_a = V2A_FEATURES
    med_a = train[feats_a].median(numeric_only=True)
    Xtr = train[feats_a].fillna(med_a).values
    Xva = val[feats_a].fillna(med_a).values
    Xte = test[feats_a].fillna(med_a).values
    ytr = train["y_drop15"].values.astype(np.int8)
    yva = val["y_drop15"].values.astype(np.int8)
    yte = test["y_drop15"].values.astype(np.int8)
    sw = sample_weights(ytr, 0.15)

    gbc_a = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                        learning_rate=0.03, subsample=0.8,
                                        random_state=42)
    gbc_a.fit(Xtr, ytr, sample_weight=sw)
    cal_a = CalibratedClassifierCV(estimator=gbc_a, method="isotonic", cv="prefit")
    cal_a.fit(Xva, yva)
    p_te_a = cal_a.predict_proba(Xte)[:, 1]
    metrics_a = evaluate_classifier(yte, p_te_a)
    plateau_a = inspect_calibrator_plateau(cal_a)
    cond_a = conditional_metrics(p_te_a, test["prob_up_v3"].values, yte, "v2a")

    print(f"[v2a] test: {metrics_a}")
    print(f"[v2a] calibrator: {plateau_a}")
    print(f"[v2a] conditional: {cond_a}")
    imp_a = pd.Series(gbc_a.feature_importances_, index=feats_a).sort_values(ascending=False)
    print(f"[v2a] top-10 importances:\n{imp_a.head(10).to_string()}")
    prob_up_imp = float(imp_a.get("prob_up_v3", 0.0))
    print(f"[v2a] prob_up_v3 importance: {prob_up_imp:.4f}")

    art_a = {"raw_gbc": gbc_a, "calibrator": cal_a, "feats": feats_a,
             "impute_medians": med_a.to_dict(),
             "metrics": metrics_a, "plateau": plateau_a,
             "conditional": cond_a, "feature_importance": imp_a.to_dict(),
             "universe": universe, "label": "y_drop15",
             "label_def": "min(close[t+1..t+21]) / close[t] - 1 <= -0.15",
             "variant": "v2a_prob_up_as_feature"}
    joblib.dump(art_a, MODELS / f"monthly_gainer_drop15_v2a_{universe}.joblib")

    # ----- v2b: residualized regression --------------------------------------
    print("\n" + "=" * 60)
    print("[v2b] training: BASE features, regress y_resid (y - baseline[prob_up_decile])")
    print("=" * 60)
    feats_b = V2B_FEATURES
    med_b = train[feats_b].median(numeric_only=True)

    # Baseline drop rate by prob_up decile (from train only)
    train["pu_decile"] = pd.qcut(train["prob_up_v3"], q=10, labels=False, duplicates="drop")
    baseline = train.groupby("pu_decile")["y_drop15"].mean()
    print(f"[v2b] baseline drop rate by prob_up decile (train):")
    print(baseline.round(4).to_string())

    def assign_decile(p_up_arr, ref_quantiles):
        edges = ref_quantiles
        return np.searchsorted(edges, p_up_arr, side="right") - 1

    # Use train's quantile edges as reference for val/test
    edges = train["prob_up_v3"].quantile(np.linspace(0, 1, 11)).values
    edges[0], edges[-1] = -np.inf, np.inf
    train["pu_dec_ref"] = np.clip(np.searchsorted(edges, train["prob_up_v3"]) - 1, 0, 9)
    val["pu_dec_ref"] = np.clip(np.searchsorted(edges, val["prob_up_v3"]) - 1, 0, 9)
    test["pu_dec_ref"] = np.clip(np.searchsorted(edges, test["prob_up_v3"]) - 1, 0, 9)
    base_arr = baseline.reindex(range(10), fill_value=baseline.mean()).values

    train["y_resid"] = train["y_drop15"] - base_arr[train["pu_dec_ref"].values]
    val["y_resid"] = val["y_drop15"] - base_arr[val["pu_dec_ref"].values]
    test["y_resid"] = test["y_drop15"] - base_arr[test["pu_dec_ref"].values]

    Xtr_b = train[feats_b].fillna(med_b).values
    Xva_b = val[feats_b].fillna(med_b).values
    Xte_b = test[feats_b].fillna(med_b).values

    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                     learning_rate=0.03, subsample=0.8,
                                     loss="squared_error", random_state=42)
    gbr.fit(Xtr_b, train["y_resid"].values)
    resid_te = gbr.predict(Xte_b)
    p_te_b = np.clip(base_arr[test["pu_dec_ref"].values] + resid_te, 0.0, 1.0)
    metrics_b = evaluate_classifier(yte, p_te_b)
    cond_b = conditional_metrics(p_te_b, test["prob_up_v3"].values, yte, "v2b")
    print(f"[v2b] test: {metrics_b}")
    print(f"[v2b] conditional: {cond_b}")
    imp_b = pd.Series(gbr.feature_importances_, index=feats_b).sort_values(ascending=False)
    print(f"[v2b] top-10 importances:\n{imp_b.head(10).to_string()}")

    art_b = {"raw_gbr": gbr, "feats": feats_b,
             "impute_medians": med_b.to_dict(),
             "baseline_by_pu_decile": baseline.to_dict(),
             "pu_decile_edges": edges.tolist(),
             "metrics": metrics_b, "conditional": cond_b,
             "feature_importance": imp_b.to_dict(),
             "universe": universe, "label": "y_drop15",
             "label_def": "min(close[t+1..t+21]) / close[t] - 1 <= -0.15",
             "variant": "v2b_residual_regression"}
    joblib.dump(art_b, MODELS / f"monthly_gainer_drop15_v2b_{universe}.joblib")

    # ----- v1 baseline (rescore for fair comparison) -------------------------
    v1 = joblib.load(MODELS / f"monthly_gainer_drop15_{universe}.joblib")
    v1_feats = v1["feats"]
    v1_med = pd.Series(v1["impute_medians"])
    Xte_v1 = test[v1_feats].fillna(v1_med).values
    p_te_v1 = v1["calibrator"].predict_proba(Xte_v1)[:, 1]
    metrics_v1 = evaluate_classifier(yte, p_te_v1)
    cond_v1 = conditional_metrics(p_te_v1, test["prob_up_v3"].values, yte, "v1")
    plateau_v1 = inspect_calibrator_plateau(v1["calibrator"])
    print(f"\n[v1 baseline] test: {metrics_v1}")
    print(f"[v1 baseline] calibrator: {plateau_v1}")
    print(f"[v1 baseline] conditional: {cond_v1}")

    # ----- compare ------------------------------------------------------------
    summary = {
        "v1": {"metrics": metrics_v1, "plateau": plateau_v1, "conditional": cond_v1},
        "v2a": {"metrics": metrics_a, "plateau": plateau_a, "conditional": cond_a,
                "prob_up_importance": prob_up_imp},
        "v2b": {"metrics": metrics_b, "conditional": cond_b},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"drop15_v2_compare_{universe}.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD")
    print("=" * 60)
    print(f"{'metric':<35}{'v1':>12}{'v2a':>12}{'v2b':>12}")
    for k in ["auc", "ap", "ap_lift", "brier"]:
        print(f"{k:<35}{metrics_v1[k]:>12.4f}{metrics_a[k]:>12.4f}{metrics_b[k]:>12.4f}")
    print(f"{'spearman_with_prob_up':<35}"
          f"{cond_v1['spearman_p_drop_p_up']:>12.4f}"
          f"{cond_a['spearman_p_drop_p_up']:>12.4f}"
          f"{cond_b['spearman_p_drop_p_up']:>12.4f}")
    print(f"{'top-up & hi-drop realized rate':<35}"
          f"{cond_v1['drop_rate_top_up_AND_hi_drop']:>12.4f}"
          f"{cond_a['drop_rate_top_up_AND_hi_drop']:>12.4f}"
          f"{cond_b['drop_rate_top_up_AND_hi_drop']:>12.4f}")
    print(f"{'top-up & lo-drop realized rate':<35}"
          f"{cond_v1['drop_rate_top_up_AND_lo_drop']:>12.4f}"
          f"{cond_a['drop_rate_top_up_AND_lo_drop']:>12.4f}"
          f"{cond_b['drop_rate_top_up_AND_lo_drop']:>12.4f}")
    print(f"{'filter lift (hi/base in top-up)':<35}"
          f"{cond_v1['filter_lift']:>12.4f}"
          f"{cond_a['filter_lift']:>12.4f}"
          f"{cond_b['filter_lift']:>12.4f}")
    print(f"\n[95] elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
