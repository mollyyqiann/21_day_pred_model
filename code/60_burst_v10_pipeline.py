"""Burst v10 training pipeline — regime features + HistGradientBoosting + class weighting.

Produces artifacts compatible with the daily runner's loader (each artifact is
a dict with the key "gbc" or "reg", so `joblib.load(p)["gbc"]` still works).

Inputs (read-only):
    data/burst_panel_v6.csv   (v5 universe, upside-asymmetric)
    data/burst_panel_v6b.csv  (v4 universe, >$40)
    data/burst_panel_v7.csv   (v7 universe, full S&P 500)
    data/sp500_daily.csv, data/vix_daily.csv, data/fear_greed.csv  (regime)

Outputs:
    models/burst_gbc_v10_v4.joblib      (classifier for v4 universe)
    models/burst_gbc_v10_v5.joblib      (classifier for v5 universe)
    models/burst_gbc_v10_v7.joblib      (classifier for v7 universe)
    models/burst_reg_v10_v4_fwd_{1d,3d,5d}.joblib
    models/burst_reg_v10_v5_fwd_{1d,3d,5d}.joblib
    models/burst_reg_v10_v7_fwd_{1d,3d,5d}.joblib
    models/burst_v10_meta.json  (feature list, hyperparams, metrics)

The naming scheme (`_v4/_v5/_v7`) deliberately matches the universe keys the
daily runner uses, so the loader can pick model by universe with no renaming.
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output"
MODELS.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from regime_features import load_regime_frame, attach_regime, REGIME_FEATS  # noqa: E402

from sklearn.ensemble import (HistGradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             log_loss, brier_score_loss,
                             mean_absolute_error)


BURST_WINDOW = 5
BURST_MIN_LEN = 2
BURST_THRESH = 0.04

V7_FEATS = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
            "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap"]

# v5 universe uses 14 features — preserve the extras the prod v5 model uses.
V5_EXTRA = ["skew_60d", "semivol_ratio_60d", "up_bigdays_60d"]

# Per-universe base feature lists (must match production to keep apples-to-apples)
BASE_FEATS = {
    "v4": V7_FEATS,                     # panel v6b: 11 feats
    "v5": V7_FEATS[:-1] + V5_EXTRA + ["overnight_gap"],  # panel v6: 14 feats, same as prod
    "v7": V7_FEATS,                     # panel v7: 11 feats
}


def v10_feats(uni: str) -> list[str]:
    return BASE_FEATS[uni] + REGIME_FEATS


# ---------- labels ----------

def build_target(c: pd.Series) -> pd.Series:
    r = c.pct_change().fillna(0).values
    n = len(r)
    y = np.zeros(n, dtype=np.int8)
    for t in range(n - BURST_WINDOW):
        fut = r[t+1:t+1+BURST_WINDOW]
        best = 0.0
        for L in range(BURST_MIN_LEN, BURST_WINDOW + 1):
            for s in range(0, BURST_WINDOW - L + 1):
                m = fut[s:s+L].mean()
                if m > best:
                    best = m
        if best >= BURST_THRESH:
            y[t] = 1
    out = pd.Series(y, index=c.index)
    out.iloc[-BURST_WINDOW:] = -1
    return out


def attach_labels_and_fwd(p: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    y_parts = []
    for _, g in p.groupby("ticker", sort=False):
        y_parts.append(build_target(g["close"]))
    p["y"] = pd.concat(y_parts).reindex(p.index).values
    p["fwd_1d"] = p.groupby("ticker")["close"].shift(-1) / p["close"] - 1
    p["fwd_3d"] = p.groupby("ticker")["close"].shift(-3) / p["close"] - 1
    p["fwd_5d"] = p.groupby("ticker")["close"].shift(-5) / p["close"] - 1
    return p


# ---------- chronological split helpers ----------

def _split_dates(lab: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = np.sort(lab["date"].unique())
    d1 = dates[int(0.70 * len(dates))]
    d2 = dates[int(0.85 * len(dates))]
    return d1, d2


# ---------- train one universe ----------

def train_one(panel_path: Path, uni_tag: str, cls_grid: list[tuple]) -> dict:
    """Train v10 classifier + 3 regressors for one universe."""
    print(f"\n=== train v10 ({uni_tag}) — {panel_path.name} ===")
    t0 = time.time()
    v10feats = v10_feats(uni_tag)
    base_feats = BASE_FEATS[uni_tag]
    p = pd.read_csv(panel_path, parse_dates=["date"])
    # keep rows where all base (non-regime) features are present
    p = p[p[base_feats].notna().all(axis=1)].copy()
    p = attach_regime(p)
    p = attach_labels_and_fwd(p)

    lab = p[p["y"] >= 0].dropna(subset=v10feats).reset_index(drop=True)
    d1, d2 = _split_dates(lab)
    tr = lab[lab["date"] <  d1]
    va = lab[(lab["date"] >= d1) & (lab["date"] < d2)]
    te = lab[lab["date"] >= d2]
    print(f"  rows labelled: {len(lab):,}  base rate: {lab['y'].mean():.4%}")
    print(f"  train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    X_tr, y_tr = tr[v10feats].values, tr["y"].values
    X_va, y_va = va[v10feats].values, va["y"].values
    X_te, y_te = te[v10feats].values, te["y"].values

    # Class weighting: upweight positives to inverse frequency
    pos = float(y_tr.mean())
    w = np.where(y_tr == 1, (1 - pos) / pos, 1.0)

    # Small grid on val PR-AUC
    best = None
    for (lr, md, mi) in cls_grid:
        m = HistGradientBoostingClassifier(max_iter=mi, max_depth=md,
                                           learning_rate=lr, random_state=42)
        m.fit(X_tr, y_tr, sample_weight=w)
        pv = m.predict_proba(X_va)[:, 1]
        s = average_precision_score(y_va, pv)
        if best is None or s > best[0]:
            best = (s, m, (lr, md, mi))
    lr, md, mi = best[2]
    cls = best[1]
    print(f"  best params lr={lr} max_depth={md} max_iter={mi}  "
          f"val PR_AUC={best[0]:.4f}")

    # Test metrics
    pt = cls.predict_proba(X_te)[:, 1]
    cls_metrics = {
        "n_test": int(len(y_te)),
        "pos_rate": float(np.mean(y_te)),
        "AUC": float(roc_auc_score(y_te, pt)),
        "PR_AUC": float(average_precision_score(y_te, pt)),
        "log_loss": float(log_loss(y_te, np.clip(pt, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "brier": float(brier_score_loss(y_te, pt)),
    }
    # precision@K per day
    te_df = te.assign(_p=pt)
    hits5, hits10, fwd5 = [], [], []
    for _, g in te_df.groupby("date", sort=False):
        if len(g) < 10:
            continue
        top5 = g.sort_values("_p", ascending=False).head(5)
        top10 = g.sort_values("_p", ascending=False).head(10)
        hits5.append(top5["y"].mean())
        hits10.append(top10["y"].mean())
        f = top5["fwd_5d"].dropna()
        if len(f):
            fwd5.append(float(f.mean()))
    cls_metrics["prec@5"] = float(np.mean(hits5)) if hits5 else float("nan")
    cls_metrics["prec@10"] = float(np.mean(hits10)) if hits10 else float("nan")
    cls_metrics["mean_fwd5_top5"] = float(np.mean(fwd5)) if fwd5 else float("nan")
    cls_metrics["n_test_days"] = int(len(hits5))
    print(f"  prec@5={cls_metrics['prec@5']:.4f}  prec@10={cls_metrics['prec@10']:.4f}  "
          f"AUC={cls_metrics['AUC']:.4f}  fwd5_top5={cls_metrics['mean_fwd5_top5']:.4f}")

    # Save classifier — wrap in a dict so the runner's `["gbc"]` loader works
    cls_path = MODELS / f"burst_gbc_v10_{uni_tag}.joblib"
    joblib.dump({"gbc": cls, "feats": v10feats, "version": "v10",
                 "metrics": cls_metrics, "best_params": {"learning_rate": lr,
                 "max_depth": md, "max_iter": mi},
                 "universe": uni_tag}, cls_path)
    print(f"  wrote {cls_path.name}")

    # --- regressors (fwd_1d / fwd_3d / fwd_5d) ---
    reg_metrics = {}
    rr = p.dropna(subset=v10feats + ["fwd_1d", "fwd_3d", "fwd_5d"]).reset_index(drop=True)
    rd1, rd2 = _split_dates(rr)
    rtr = rr[rr["date"] <  rd1]
    rte = rr[rr["date"] >= rd2]
    for h in ["fwd_1d", "fwd_3d", "fwd_5d"]:
        reg = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=42)
        reg.fit(rtr[v10feats].values, rtr[h].values)
        pr = reg.predict(rte[v10feats].values)
        rmae = float(mean_absolute_error(rte[h].values, pr))
        rmae_zero = float(np.mean(np.abs(rte[h].values)))
        dir_hit = float(np.mean(np.sign(pr) == np.sign(rte[h].values)))
        reg_metrics[h] = {"n_test": int(len(rte)), "MAE": rmae,
                          "MAE_vs_zero": rmae_zero, "dir_hit": dir_hit}
        rp = MODELS / f"burst_reg_v10_{uni_tag}_{h}.joblib"
        joblib.dump({"reg": reg, "feats": v10feats, "version": "v10",
                     "horizon": h, "universe": uni_tag,
                     "metrics": reg_metrics[h]}, rp)
        print(f"  {h}: MAE={rmae:.4f} (vs zero {rmae_zero:.4f})  dir_hit={dir_hit:.3f}   -> {rp.name}")

    dt = time.time() - t0
    return {
        "universe": uni_tag,
        "panel": panel_path.name,
        "n_test": cls_metrics["n_test"],
        "cls_metrics": cls_metrics,
        "reg_metrics": reg_metrics,
        "best_params": {"learning_rate": lr, "max_depth": md, "max_iter": mi},
        "runtime_sec": round(dt, 1),
    }


# ---------- main ----------

def main():
    # Small grid — same options that won Exp 3
    cls_grid = [(0.03, None, 400), (0.03, 6, 400), (0.05, None, 400),
                (0.05, 6, 400), (0.08, None, 300), (0.05, 8, 300)]

    # Universe tags match what the daily runner expects (v4, v5, v7)
    jobs = [
        ("v4", DATA / "burst_panel_v6b.csv"),  # >$40 universe, shipped as v4
        ("v5", DATA / "burst_panel_v6.csv"),   # upside-asym, shipped as v5
        ("v7", DATA / "burst_panel_v7.csv"),   # full S&P 500
    ]

    results = {}
    t0 = time.time()
    for uni_tag, path in jobs:
        results[uni_tag] = train_one(path, uni_tag, cls_grid)

    meta = {
        "version": "v10",
        "base_feats_per_universe": BASE_FEATS,
        "regime_feats": REGIME_FEATS,
        "label": {"window": BURST_WINDOW, "min_len": BURST_MIN_LEN, "thresh": BURST_THRESH},
        "total_runtime_sec": round(time.time() - t0, 1),
        "per_universe": results,
    }
    (MODELS / "burst_v10_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\n[v10] wrote {MODELS/'burst_v10_meta.json'}")
    print(f"[v10] total runtime: {meta['total_runtime_sec']}s")

    # Quick summary table
    print("\n=== v10 SUMMARY ===")
    for uni, r in results.items():
        m = r["cls_metrics"]
        print(f"  {uni}: prec@5={m['prec@5']:.4f}  prec@10={m['prec@10']:.4f}  "
              f"AUC={m['AUC']:.4f}  fwd5_top5={m['mean_fwd5_top5']:+.4f}")


if __name__ == "__main__":
    main()
