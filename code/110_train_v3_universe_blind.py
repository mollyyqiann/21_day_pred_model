"""Train v3 universe-blind: combined SP500 + Smallcap panels.

The current setup splits 1,891 Robinhood-tradable tickers into two
artificial universes (SP500 503 + smallcap 1,388). Mid-caps like MRVL
fall through the cracks — too small for SP500, too calm for the
smallcap model's vol-dominated ranking.

This script combines the panels, re-computes cross-sectional xrank
features ACROSS THE FULL UNIVERSE per date, and trains v3.

Output:
   models/monthly_gainer_v3_combined.joblib
   output/monthly_gainer/v3_combined_metrics.json
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
] + REGIME_FEATS

CATALYST = [
    "finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
    "news_n_5d", "news_n_20d",
    "earn_news_5d", "earn_news_20d",
    "ma_news_5d", "ma_news_20d", "sector_pop_5d",
]

XRANK = ["rsi_14_xrank", "rv_60_xrank", "ma60_slope_xrank", "ret_20d_xrank"]
ALL_FEATS = V1_FEATURES + CATALYST + XRANK


def time_split(df, train_frac=0.70, val_frac=0.15):
    dates = np.sort(df["date"].unique())
    t1 = dates[int(len(dates) * train_frac)]
    t2 = dates[int(len(dates) * (train_frac + val_frac))]
    return (df[df["date"] < t1].copy(),
            df[(df["date"] >= t1) & (df["date"] < t2)].copy(),
            df[df["date"] >= t2].copy())


def sample_weights(y, target_pos_frac):
    p = y.mean()
    if p <= 0 or p >= 1:
        return np.ones(len(y))
    w_pos = target_pos_frac / p
    w_neg = (1 - target_pos_frac) / (1 - p)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def evaluate(y, p):
    base = float(y.mean())
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    ap = float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")
    return {"n": int(len(y)), "pos": int(y.sum()),
            "base": base, "auc": auc, "ap": ap,
            "ap_lift": ap / base if base > 0 else float("nan"),
            "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
            "brier": float(brier_score_loss(y, p))}


def precision_topk(test, k):
    rs = []
    for d, g in test.groupby("date"):
        if len(g) < k:
            continue
        rs.append(g.nlargest(k, "prob")["y21"].mean())
    return float(np.mean(rs)) if rs else float("nan")


def long_only_sim(test, k=5):
    rows = []
    for d, g in test.groupby("date"):
        if len(g) < k:
            continue
        topk = g.nlargest(k, "prob")
        rows.append({
            "topk_max": topk["max_fwd21_ret"].mean(),
            "topk_end": topk["end_of_window_ret"].mean(),
            "univ_max": g["max_fwd21_ret"].mean(),
            "univ_end": g["end_of_window_ret"].mean(),
            "topk_pos": topk["y21"].mean(),
        })
    if not rows: return None
    s = pd.DataFrame(rows)
    return {
        "topk_max_ret": float(s["topk_max"].mean()),
        "topk_end_ret": float(s["topk_end"].mean()),
        "univ_max_ret": float(s["univ_max"].mean()),
        "univ_end_ret": float(s["univ_end"].mean()),
        "topk_pos_rate": float(s["topk_pos"].mean()),
    }


def main():
    print("[110] loading + merging panels ...")
    sp = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    sc = pd.read_csv(DATA / "monthly_gainer_panel_smallcap.csv", parse_dates=["date"])
    print(f"     SP500: {len(sp):,} rows, {sp['ticker'].nunique()} tickers")
    print(f"     Smallcap: {len(sc):,} rows, {sc['ticker'].nunique()} tickers")

    # Make schemas compatible: smallcap doesn't have sector / certain cols
    common = list(set(sp.columns) & set(sc.columns))
    sp = sp[common].copy()
    sc = sc[common].copy()
    panel = pd.concat([sp, sc], ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"     COMBINED: {len(panel):,} rows, {panel['ticker'].nunique()} tickers")

    # Catalyst merge
    cat_sp = pd.read_csv(DATA / "catalyst_features_sp500.csv", parse_dates=["date"])
    cat_sc = pd.read_csv(DATA / "catalyst_features_smallcap.csv", parse_dates=["date"])
    cat = pd.concat([cat_sp, cat_sc], ignore_index=True)
    panel = panel.merge(cat, on=["ticker", "date"], how="left")
    panel = attach_regime(panel)

    # Lag returns + xrank cross-sectional on COMBINED universe
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["close_5d_ago"] = panel.groupby("ticker")["close"].shift(5)
    panel["ret_5d_lag"] = panel["close"] / panel["close_5d_ago"] - 1.0
    panel["close_20d_ago"] = panel.groupby("ticker")["close"].shift(20)
    panel["ret_20d_lag"] = panel["close"] / panel["close_20d_ago"] - 1.0
    panel["close_60d_ago"] = panel.groupby("ticker")["close"].shift(60)
    panel["ret_60d_lag"] = panel["close"] / panel["close_60d_ago"] - 1.0
    panel["close_180d_ago"] = panel.groupby("ticker")["close"].shift(180)
    panel["ret_180d_lag"] = panel["close"] / panel["close_180d_ago"] - 1.0
    panel["rsi_14_xrank"] = panel.groupby("date")["rsi_14"].rank(pct=True)
    panel["rv_60_xrank"] = panel.groupby("date")["rv_60"].rank(pct=True)
    panel["ma60_slope_xrank"] = panel.groupby("date")["ma60_slope_60d"].rank(pct=True)
    panel["ret_20d_xrank"] = panel.groupby("date")["ret_20d_lag"].rank(pct=True)

    # Fill catalyst NaNs (no news data) with 0
    for c in CATALYST:
        if c not in panel.columns:
            panel[c] = 0.0
        else:
            panel[c] = panel[c].fillna(0)
    for c in XRANK:
        panel[c] = panel[c].fillna(0.5)

    # Train
    lab = panel[panel["y21"].isin([0, 1])].dropna(subset=V1_FEATURES).reset_index(drop=True)
    print(f"     labeled: {len(lab):,}, base rate: {lab['y21'].mean():.4%}")

    train, val, test = time_split(lab)
    print(f"     split: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    print(f"     test_base: {test['y21'].mean():.4%}")

    medians = train[ALL_FEATS].median(numeric_only=True)
    Xtr = train[ALL_FEATS].fillna(medians).values
    Xva = val[ALL_FEATS].fillna(medians).values
    Xte = test[ALL_FEATS].fillna(medians).values
    ytr = train["y21"].values.astype(np.int8)
    yva = val["y21"].values.astype(np.int8)
    yte = test["y21"].values.astype(np.int8)

    # Use a target pos frac that's the mean of the two universe targets (0.07)
    sw = sample_weights(ytr, target_pos_frac=0.07)

    print("[110] training v3-combined GBC (n_estimators=400) ...")
    t0 = time.time()
    gbc = GradientBoostingClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42)
    gbc.fit(Xtr, ytr, sample_weight=sw)
    print(f"     fit done in {time.time()-t0:.0f}s")

    cal = CalibratedClassifierCV(estimator=gbc, method="isotonic", cv="prefit")
    cal.fit(Xva, yva)
    p_te = cal.predict_proba(Xte)[:, 1]
    metrics = evaluate(yte, p_te)

    test_eval = test.copy()
    test_eval["prob"] = p_te
    metrics["p_at_top5"] = precision_topk(test_eval, 5)
    metrics["p_at_top10"] = precision_topk(test_eval, 10)
    metrics["sim"] = long_only_sim(test_eval, 5)
    print(f"\n[110] METRICS: {metrics}")

    imp = pd.Series(gbc.feature_importances_, index=ALL_FEATS).sort_values(ascending=False)
    print(f"\n[110] top-10 importance:\n{imp.head(10).to_string()}")

    artifact = {
        "raw_gbc": gbc, "calibrator": cal,
        "feats": ALL_FEATS, "v1_features": V1_FEATURES,
        "catalyst_features": CATALYST, "xrank_features": XRANK,
        "impute_medians": medians.to_dict(),
        "metrics": metrics,
        "feature_importance": imp.to_dict(),
        "universe": "combined",
    }
    joblib.dump(artifact, MODELS / "monthly_gainer_v3_combined.joblib")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_only = {k: v for k, v in artifact.items() if k not in ("raw_gbc", "calibrator")}
    (OUT / "v3_combined_metrics.json").write_text(json.dumps(metrics_only, indent=2, default=str))
    print(f"\n[110] saved {MODELS / 'monthly_gainer_v3_combined.joblib'}")


if __name__ == "__main__":
    main()
