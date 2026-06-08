"""Burst v11 — v10 base/regime + pre-market & intraday features merged from
data/intraday_daily_features.csv (built by 71_intraday_features.py).

Same training shape as v10: HistGradientBoostingClassifier + 3 regressors per
universe (v4/v5/v7). The classifier already handled NaN; the regressors are
switched from GradientBoostingRegressor to HistGradientBoostingRegressor so
they tolerate NaN intraday values too.

Inputs:
    data/burst_panel_v6b.csv, burst_panel_v6.csv, burst_panel_v7.csv
    data/intraday_daily_features.csv  (per-ticker per-date intraday features)
    data/sp500_daily.csv, vix_daily.csv, fear_greed.csv  (regime)

Outputs:
    models/burst_gbc_v11_{v4,v5,v7}.joblib
    models/burst_reg_v11_{v4,v5,v7}_fwd_{1d,3d,5d}.joblib
    models/burst_v11_meta.json   (head-to-head metrics vs v10)

Note: intraday features only cover the recent ~28 days right now (yfinance 1m
ceiling). Most training rows will have NaN for these — HGB ignores them. The
classifier and regressor will pick up signal as the local intraday store grows
(70_intraday_collect.py runs daily).
"""
from __future__ import annotations

import json
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
MODELS.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from regime_features import attach_regime, REGIME_FEATS  # noqa: E402

from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             log_loss, brier_score_loss,
                             mean_absolute_error)


BURST_WINDOW = 5
BURST_MIN_LEN = 2
BURST_THRESH = 0.04

V7_FEATS = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
            "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap"]
V5_EXTRA = ["skew_60d", "semivol_ratio_60d", "up_bigdays_60d"]

BASE_FEATS = {
    "v4": V7_FEATS,
    "v5": V7_FEATS[:-1] + V5_EXTRA + ["overnight_gap"],
    "v7": V7_FEATS,
}

# Pre-market volume ratio dropped: yfinance pre-market 1m bars carry volume=0,
# making the feature constant. Will be re-added once the Finnhub WS recorder
# (72_finnhub_ws_recorder.py) has accumulated pre-market trades and a separate
# loader plugs it in.
INTRADAY_FEATS = [
    "pm_gap_pct", "pm_range_pct", "pm_drift",
    "ret_first_30m", "ret_last_30m", "intraday_rv_5m",
    "vwap_close_dist", "intraday_dd", "vol_late_share", "close_strength",
]


def v11_feats(uni: str) -> list[str]:
    return BASE_FEATS[uni] + REGIME_FEATS + INTRADAY_FEATS


def attach_intraday(p: pd.DataFrame) -> pd.DataFrame:
    feat_path = DATA / "intraday_daily_features.csv"
    if not feat_path.exists():
        for c in INTRADAY_FEATS:
            p[c] = np.nan
        return p
    f = pd.read_csv(feat_path, parse_dates=["date"])
    f["date"] = f["date"].dt.normalize()
    keep = ["ticker", "date"] + [c for c in INTRADAY_FEATS if c in f.columns]
    f = f[keep]
    p = p.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    return p.merge(f, on=["ticker", "date"], how="left")


# ---------- labels (same as v10) ----------

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


def _split_dates(lab: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = np.sort(lab["date"].unique())
    d1 = dates[int(0.70 * len(dates))]
    d2 = dates[int(0.85 * len(dates))]
    return d1, d2


# ---------- train one universe ----------

def train_one(panel_path: Path, uni_tag: str, cls_grid: list[tuple]) -> dict:
    print(f"\n=== train v11 ({uni_tag}) — {panel_path.name} ===")
    t0 = time.time()
    feats11 = v11_feats(uni_tag)
    base_feats = BASE_FEATS[uni_tag]
    required = base_feats + REGIME_FEATS  # intraday is optional (NaN ok for HGB)

    p = pd.read_csv(panel_path, parse_dates=["date"])
    p = p[p[base_feats].notna().all(axis=1)].copy()
    p = attach_regime(p)
    p = attach_intraday(p)
    p = attach_labels_and_fwd(p)

    # report intraday coverage
    intra_cov = p[INTRADAY_FEATS[0]].notna().mean()
    intra_rows = int(p[INTRADAY_FEATS[0]].notna().sum())
    print(f"  intraday coverage: {intra_cov:.2%} ({intra_rows:,} rows)")

    lab = p[p["y"] >= 0].dropna(subset=required).reset_index(drop=True)
    d1, d2 = _split_dates(lab)
    tr = lab[lab["date"] <  d1]
    va = lab[(lab["date"] >= d1) & (lab["date"] < d2)]
    te = lab[lab["date"] >= d2]
    print(f"  rows labelled: {len(lab):,}  base rate: {lab['y'].mean():.4%}")
    print(f"  train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    X_tr, y_tr = tr[feats11].values, tr["y"].values
    X_va, y_va = va[feats11].values, va["y"].values
    X_te, y_te = te[feats11].values, te["y"].values

    pos = float(y_tr.mean())
    w = np.where(y_tr == 1, (1 - pos) / pos, 1.0)

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
    print(f"  best params lr={lr} max_depth={md} max_iter={mi}  val PR_AUC={best[0]:.4f}")

    pt = cls.predict_proba(X_te)[:, 1]
    cls_metrics = {
        "n_test": int(len(y_te)),
        "pos_rate": float(np.mean(y_te)),
        "AUC": float(roc_auc_score(y_te, pt)),
        "PR_AUC": float(average_precision_score(y_te, pt)),
        "log_loss": float(log_loss(y_te, np.clip(pt, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "brier": float(brier_score_loss(y_te, pt)),
    }
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
          f"AUC={cls_metrics['AUC']:.4f}  fwd5_top5={cls_metrics['mean_fwd5_top5']:+.4f}")

    # Isotonic calibrator: fit on val-set raw probabilities -> labels. The raw
    # HGB probs are systematically inflated (calibration analysis showed even
    # p>=0.90 has only 0.45 precision), so we expose a calibrated probability
    # to the daily runner for the high-precision list.
    pv_for_cal = cls.predict_proba(X_va)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pv_for_cal, y_va)
    pt_cal = iso.transform(pt)
    cal_metrics = {
        "val_brier_raw": float(brier_score_loss(y_va, pv_for_cal)),
        "val_brier_cal": float(brier_score_loss(y_va, iso.transform(pv_for_cal))),
        "test_brier_raw": float(brier_score_loss(y_te, pt)),
        "test_brier_cal": float(brier_score_loss(y_te, pt_cal)),
        # Precision at calibrated thresholds — useful for picking the high-prec gate.
        "test_prec_at_cal_thr": {f"{t:.2f}": float(np.mean(y_te[pt_cal >= t])) if (pt_cal >= t).any() else None
                                 for t in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)},
        "test_n_at_cal_thr": {f"{t:.2f}": int((pt_cal >= t).sum())
                              for t in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)},
    }
    print(f"  brier raw->cal: val {cal_metrics['val_brier_raw']:.4f}->{cal_metrics['val_brier_cal']:.4f}  "
          f"test {cal_metrics['test_brier_raw']:.4f}->{cal_metrics['test_brier_cal']:.4f}")

    cls_path = MODELS / f"burst_gbc_v11_{uni_tag}.joblib"
    joblib.dump({"gbc": cls, "feats": feats11, "version": "v11",
                 "metrics": cls_metrics,
                 "best_params": {"learning_rate": lr, "max_depth": md, "max_iter": mi},
                 "universe": uni_tag,
                 "intraday_coverage_train": float(tr[INTRADAY_FEATS[0]].notna().mean()),
                 "intraday_coverage_test": float(te[INTRADAY_FEATS[0]].notna().mean()),
                 "calibrator": iso,
                 "calibration_metrics": cal_metrics},
                cls_path)
    print(f"  wrote {cls_path.name}")

    # --- regressors (HistGradientBoosting handles NaN) ---
    reg_metrics = {}
    rr = p.dropna(subset=required + ["fwd_1d", "fwd_3d", "fwd_5d"]).reset_index(drop=True)
    rd1, rd2 = _split_dates(rr)
    rtr = rr[rr["date"] <  rd1]
    rte = rr[rr["date"] >= rd2]
    for h in ["fwd_1d", "fwd_3d", "fwd_5d"]:
        reg = HistGradientBoostingRegressor(max_iter=300, max_depth=3,
                                            learning_rate=0.05, random_state=42)
        reg.fit(rtr[feats11].values, rtr[h].values)
        pr = reg.predict(rte[feats11].values)
        rmae = float(mean_absolute_error(rte[h].values, pr))
        rmae_zero = float(np.mean(np.abs(rte[h].values)))
        dir_hit = float(np.mean(np.sign(pr) == np.sign(rte[h].values)))
        reg_metrics[h] = {"n_test": int(len(rte)), "MAE": rmae,
                          "MAE_vs_zero": rmae_zero, "dir_hit": dir_hit}
        rp = MODELS / f"burst_reg_v11_{uni_tag}_{h}.joblib"
        joblib.dump({"reg": reg, "feats": feats11, "version": "v11",
                     "horizon": h, "universe": uni_tag,
                     "metrics": reg_metrics[h]}, rp)
        print(f"  {h}: MAE={rmae:.4f} (vs zero {rmae_zero:.4f})  dir_hit={dir_hit:.3f}   -> {rp.name}")

    dt = time.time() - t0
    return {
        "universe": uni_tag, "panel": panel_path.name,
        "n_test": cls_metrics["n_test"],
        "cls_metrics": cls_metrics,
        "reg_metrics": reg_metrics,
        "best_params": {"learning_rate": lr, "max_depth": md, "max_iter": mi},
        "intraday_rows": intra_rows, "intraday_coverage": intra_cov,
        "runtime_sec": round(dt, 1),
    }


def _load_v10_metrics() -> dict:
    p = MODELS / "burst_v10_meta.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _delta_table(v11: dict, v10: dict) -> str:
    rows = ["", "=== v11 vs v10 (test set) ==="]
    rows.append(f"{'uni':<5}{'metric':<22}{'v10':>10}{'v11':>10}{'delta':>10}")
    for uni in ("v4", "v5", "v7"):
        if uni not in v11:
            continue
        a = v11[uni]["cls_metrics"]
        b = v10.get("per_universe", {}).get(uni, {}).get("cls_metrics", {})
        for k in ("AUC", "PR_AUC", "prec@5", "prec@10", "mean_fwd5_top5", "log_loss", "brier"):
            v = a.get(k); u = b.get(k)
            if v is None or u is None:
                continue
            d = v - u
            rows.append(f"{uni:<5}{k:<22}{u:>10.4f}{v:>10.4f}{d:>+10.4f}")
        for h in ("fwd_1d", "fwd_3d", "fwd_5d"):
            ra = v11[uni]["reg_metrics"].get(h, {})
            rb = b and v10.get("per_universe", {}).get(uni, {}).get("reg_metrics", {}).get(h, {})
            if not ra or not rb:
                continue
            for k in ("MAE", "dir_hit"):
                v = ra.get(k); u = rb.get(k)
                if v is None or u is None:
                    continue
                d = v - u
                rows.append(f"{uni:<5}{('reg ' + h + ' ' + k):<22}{u:>10.4f}{v:>10.4f}{d:>+10.4f}")
    return "\n".join(rows)


def main():
    cls_grid = [(0.03, None, 400), (0.03, 6, 400), (0.05, None, 400),
                (0.05, 6, 400), (0.08, None, 300), (0.05, 8, 300)]

    jobs = [
        ("v4", DATA / "burst_panel_v6b.csv"),
        ("v5", DATA / "burst_panel_v6.csv"),
        ("v7", DATA / "burst_panel_v7.csv"),
    ]

    results = {}
    t0 = time.time()
    for uni_tag, path in jobs:
        results[uni_tag] = train_one(path, uni_tag, cls_grid)

    meta = {
        "version": "v11",
        "base_feats_per_universe": BASE_FEATS,
        "regime_feats": REGIME_FEATS,
        "intraday_feats": INTRADAY_FEATS,
        "label": {"window": BURST_WINDOW, "min_len": BURST_MIN_LEN, "thresh": BURST_THRESH},
        "total_runtime_sec": round(time.time() - t0, 1),
        "per_universe": results,
    }
    (MODELS / "burst_v11_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\n[v11] wrote {MODELS/'burst_v11_meta.json'}")
    print(f"[v11] total runtime: {meta['total_runtime_sec']}s")

    print("\n=== v11 SUMMARY ===")
    for uni, r in results.items():
        m = r["cls_metrics"]
        print(f"  {uni}: prec@5={m['prec@5']:.4f}  prec@10={m['prec@10']:.4f}  "
              f"AUC={m['AUC']:.4f}  fwd5_top5={m['mean_fwd5_top5']:+.4f}  "
              f"intra_cov={r['intraday_coverage']:.2%}")

    v10 = _load_v10_metrics()
    if v10:
        print(_delta_table(results, v10))


if __name__ == "__main__":
    main()
