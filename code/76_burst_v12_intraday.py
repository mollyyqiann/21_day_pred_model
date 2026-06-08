"""Burst v12 — INTRADAY-target retrain

Trains regressor heads (and a fresh classifier) with targets anchored to the
trader's actual entry — open of the prediction day — instead of close-to-close
from the prior session.

Why
---
Live grading on 135 picks revealed: 89% of v11's predicted close-to-close
return was the overnight gap that already happened before the 8:45 ET signal
fires. Trader's mean realized return at entry was +0.03% vs +3.10% c2c —
basically zero alpha. Direction-hit rate flipped from 76% (c2c) to 41% (entry).

Targets (per row, with row anchored at end-of-day t-1 features):
    open_t   = open price on the prediction day
    close_tN = close on day t + (N-1)   (N=1: same-day close; N=3: 3rd session close; N=5: 5th)
    fwd_intra_1d = close_t1 / open_t - 1   (held intraday, exit at close)
    fwd_intra_3d = close_t3 / open_t - 1   (entered open day t, exit close day t+2)
    fwd_intra_5d = close_t5 / open_t - 1

Features: same as v11 (rsi/macd/bb/etc + regime + intraday). intraday_daily_features
covers only the last ~28 days of training rows (yfinance ceiling) — HGB tolerates NaN.

Classifier
----------
Also retrained: y = 1 iff some 2-5d window of intraday-from-open returns averages
≥ 4%/day. This identifies stocks whose burst happens INTRADAY (capturable),
not as overnight gaps (not capturable).

Outputs
-------
    models/burst_gbc_v12_{v4,v5,v7}.joblib
    models/burst_reg_v12_{v4,v5,v7}_intra_{1d,3d,5d}.joblib
    models/burst_v12_meta.json
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

import joblib, numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "code"))
from regime_features import attach_regime, REGIME_FEATS  # noqa

from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             log_loss, brier_score_loss, mean_absolute_error)

V7_FEATS = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
            "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap"]
V5_EXTRA = ["skew_60d", "semivol_ratio_60d", "up_bigdays_60d"]

BASE_FEATS = {
    "v4": V7_FEATS,
    "v5": V7_FEATS[:-1] + V5_EXTRA + ["overnight_gap"],
    "v7": V7_FEATS,
}

INTRADAY_FEATS = [
    "pm_gap_pct", "pm_range_pct", "pm_drift",
    "ret_first_30m", "ret_last_30m", "intraday_rv_5m",
    "vwap_close_dist", "intraday_dd", "vol_late_share", "close_strength",
]

# Intraday burst label: any 2..5 day window of (close[t+i+1..]/open[t+i+1..])
# averages >= 4%/day. (Same shape as v11 c2c label, but using intraday returns.)
WINDOW = 5; MIN_LEN = 2; THRESH = 0.04


def v12_feats(uni: str) -> list[str]:
    return BASE_FEATS[uni] + REGIME_FEATS + INTRADAY_FEATS


def attach_open_and_intra(p: pd.DataFrame) -> pd.DataFrame:
    """Adds:
        open_next   : open price on the NEXT trading day (the entry price)
        close_t1..5 : close on +1, +3, +5 sessions
        intra_1d/3d/5d : close_tN / open_next - 1
        intra_daily : per-row daily intraday return (close_t / open_t) for label-building
    """
    op = pd.read_csv(DATA / "open_prices.csv", parse_dates=["date"])
    op["date"] = op["date"].dt.normalize()
    p = p.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p = p.merge(op, on=["ticker", "date"], how="left")
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)

    g = p.groupby("ticker", sort=False)
    # The row at date t carries close[t]. The trader using this row enters at
    # open[t+1] and exits at close[t+1] (1d), close[t+3] (3d), close[t+5] (5d).
    p["open_next"]  = g["open"].shift(-1)
    p["close_t1"]   = g["close"].shift(-1)
    p["close_t3"]   = g["close"].shift(-3)
    p["close_t5"]   = g["close"].shift(-5)
    p["fwd_intra_1d"] = p["close_t1"] / p["open_next"] - 1
    p["fwd_intra_3d"] = p["close_t3"] / p["open_next"] - 1
    p["fwd_intra_5d"] = p["close_t5"] / p["open_next"] - 1
    # Per-day intraday return for label construction
    p["intra_today"] = p["close"] / p["open"] - 1
    return p


def build_intraday_burst_label(intra: pd.Series) -> pd.Series:
    """y = 1 iff any 2..5 day window of FUTURE intraday-from-open returns
    averages >= 4%/day. Anchored such that the row at date t looks at intraday
    returns of days t+1, t+2, ..., t+5."""
    r = intra.fillna(0).values
    n = len(r)
    y = np.zeros(n, dtype=np.int8)
    for t in range(n - WINDOW):
        fut = r[t+1:t+1+WINDOW]
        best = 0.0
        for L in range(MIN_LEN, WINDOW + 1):
            for s in range(0, WINDOW - L + 1):
                m = fut[s:s+L].mean()
                if m > best: best = m
        if best >= THRESH: y[t] = 1
    out = pd.Series(y, index=intra.index)
    out.iloc[-WINDOW:] = -1
    return out


def attach_intraday_feats(p: pd.DataFrame) -> pd.DataFrame:
    fp = DATA / "intraday_daily_features.csv"
    if not fp.exists():
        for c in INTRADAY_FEATS:
            if c not in p.columns: p[c] = np.nan
        return p
    f = pd.read_csv(fp, parse_dates=["date"])
    f["date"] = f["date"].dt.normalize()
    keep = ["ticker", "date"] + [c for c in INTRADAY_FEATS if c in f.columns]
    return p.merge(f[keep], on=["ticker", "date"], how="left")


def attach_labels(p: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in p.groupby("ticker", sort=False):
        parts.append(build_intraday_burst_label(g["intra_today"]))
    p["y"] = pd.concat(parts).reindex(p.index).values
    return p


def _split_dates(d: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = np.sort(d.unique())
    return dates[int(0.70 * len(dates))], dates[int(0.85 * len(dates))]


def train_one(panel_path: Path, uni_tag: str, cls_grid: list[tuple]) -> dict:
    print(f"\n=== v12 ({uni_tag}) — {panel_path.name} ===")
    t0 = time.time()
    feats = v12_feats(uni_tag)
    base = BASE_FEATS[uni_tag]
    required = base + REGIME_FEATS

    p = pd.read_csv(panel_path, parse_dates=["date"])
    p = p[p[base].notna().all(axis=1)].copy()
    p = attach_regime(p)
    p = attach_open_and_intra(p)
    p = attach_intraday_feats(p)
    p = attach_labels(p)
    intra_cov = p[INTRADAY_FEATS[0]].notna().mean()
    print(f"  intraday coverage: {intra_cov:.2%}")
    print(f"  open coverage:     {p['open_next'].notna().mean():.2%}")

    # Classifier: drop label-future rows (y == -1) and rows missing core feats
    lab = p[p["y"] >= 0].dropna(subset=required).reset_index(drop=True)
    d1, d2 = _split_dates(lab["date"])
    tr = lab[lab["date"] <  d1]
    va = lab[(lab["date"] >= d1) & (lab["date"] < d2)]
    te = lab[lab["date"] >= d2]
    print(f"  rows labelled: {len(lab):,}  base rate: {lab['y'].mean():.4%}")
    print(f"  cls train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    X_tr, y_tr = tr[feats].values, tr["y"].values
    X_va, y_va = va[feats].values, va["y"].values
    X_te, y_te = te[feats].values, te["y"].values
    pos = float(y_tr.mean()); w = np.where(y_tr == 1, (1-pos)/pos, 1.0)
    best = None
    for (lr, md, mi) in cls_grid:
        m = HistGradientBoostingClassifier(max_iter=mi, max_depth=md,
                                           learning_rate=lr, random_state=42)
        m.fit(X_tr, y_tr, sample_weight=w)
        pv = m.predict_proba(X_va)[:, 1]
        s = average_precision_score(y_va, pv)
        if best is None or s > best[0]: best = (s, m, (lr, md, mi))
    lr, md, mi = best[2]; cls = best[1]
    pt = cls.predict_proba(X_te)[:, 1]

    cls_metrics = {
        "n_test": int(len(y_te)), "pos_rate": float(np.mean(y_te)),
        "AUC": float(roc_auc_score(y_te, pt)),
        "PR_AUC": float(average_precision_score(y_te, pt)),
        "log_loss": float(log_loss(y_te, np.clip(pt, 1e-7, 1-1e-7), labels=[0, 1])),
        "brier": float(brier_score_loss(y_te, pt)),
    }
    # prec@K and mean fwd_intra_5d on top-K per date
    te_df = te.assign(_p=pt)
    hits5, hits10, fwd5 = [], [], []
    for _, gd in te_df.groupby("date", sort=False):
        if len(gd) < 10: continue
        t5 = gd.sort_values("_p", ascending=False).head(5)
        hits5.append(t5["y"].mean())
        hits10.append(gd.sort_values("_p", ascending=False).head(10)["y"].mean())
        f = t5["fwd_intra_5d"].dropna()
        if len(f): fwd5.append(float(f.mean()))
    cls_metrics["prec@5"] = float(np.mean(hits5)) if hits5 else float("nan")
    cls_metrics["prec@10"] = float(np.mean(hits10)) if hits10 else float("nan")
    cls_metrics["mean_fwd_intra_5d_top5"] = float(np.mean(fwd5)) if fwd5 else float("nan")
    cls_metrics["n_test_days"] = int(len(hits5))
    print(f"  prec@5={cls_metrics['prec@5']:.4f}  prec@10={cls_metrics['prec@10']:.4f}  "
          f"AUC={cls_metrics['AUC']:.4f}  fwd_intra_5d_top5={cls_metrics['mean_fwd_intra_5d_top5']:+.4f}")

    pv_for_cal = cls.predict_proba(X_va)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pv_for_cal, y_va)

    cls_path = MODELS / f"burst_gbc_v12_{uni_tag}.joblib"
    joblib.dump({"gbc": cls, "feats": feats, "version": "v12",
                 "metrics": cls_metrics, "calibrator": iso,
                 "best_params": {"learning_rate": lr, "max_depth": md, "max_iter": mi},
                 "universe": uni_tag,
                 "intraday_coverage_train": float(tr[INTRADAY_FEATS[0]].notna().mean()),
                 "intraday_coverage_test": float(te[INTRADAY_FEATS[0]].notna().mean()),
                 "label_def": {"window": WINDOW, "min_len": MIN_LEN, "thresh": THRESH,
                                "anchor": "intraday_open_to_close"}},
                cls_path)
    print(f"  wrote {cls_path.name}")

    # Regressors — intraday targets
    reg_metrics = {}
    rr = p.dropna(subset=required + ["fwd_intra_1d", "fwd_intra_3d", "fwd_intra_5d"]).reset_index(drop=True)
    rd1, rd2 = _split_dates(rr["date"])
    rtr = rr[rr["date"] <  rd1]; rte = rr[rr["date"] >= rd2]
    print(f"  reg train={len(rtr):,}  test={len(rte):,}")
    for h in ["fwd_intra_1d", "fwd_intra_3d", "fwd_intra_5d"]:
        reg = HistGradientBoostingRegressor(max_iter=300, max_depth=3,
                                             learning_rate=0.05, random_state=42)
        reg.fit(rtr[feats].values, rtr[h].values)
        pr = reg.predict(rte[feats].values)
        rmae = float(mean_absolute_error(rte[h].values, pr))
        rmae_zero = float(np.mean(np.abs(rte[h].values)))
        dir_hit = float(np.mean(np.sign(pr) == np.sign(rte[h].values)))
        reg_metrics[h] = {"n_test": int(len(rte)), "MAE": rmae,
                           "MAE_vs_zero": rmae_zero, "dir_hit": dir_hit,
                           "skill_vs_zero": 1 - rmae / rmae_zero if rmae_zero > 0 else float("nan")}
        rp = MODELS / f"burst_reg_v12_{uni_tag}_{h.replace('fwd_intra_', 'intra_')}.joblib"
        joblib.dump({"reg": reg, "feats": feats, "version": "v12",
                     "horizon": h, "universe": uni_tag,
                     "metrics": reg_metrics[h]}, rp)
        print(f"  {h:>15s}: MAE={rmae:.4f} (vs zero {rmae_zero:.4f}, skill {(1-rmae/rmae_zero)*100:+.1f}%)  dir={dir_hit:.3f}")

    return {"universe": uni_tag, "panel": panel_path.name,
            "cls_metrics": cls_metrics, "reg_metrics": reg_metrics,
            "best_cls_params": {"learning_rate": lr, "max_depth": md, "max_iter": mi},
            "runtime_sec": round(time.time() - t0, 1)}


def main():
    grids = [(0.05, 5, 300), (0.05, 7, 400), (0.03, 7, 600)]
    out = {"version": "v12", "label_def": {"window": WINDOW, "min_len": MIN_LEN,
                                            "thresh": THRESH,
                                            "anchor": "intraday_open_to_close"},
           "feats": {u: v12_feats(u) for u in ("v4", "v5", "v7")},
           "per_universe": {}}
    panels = [(DATA / "burst_panel_v6b.csv", "v4"),
              (DATA / "burst_panel_v6.csv",  "v5"),
              (DATA / "burst_panel_v7.csv",  "v7")]
    for path, uni in panels:
        out["per_universe"][uni] = train_one(path, uni, grids)
    (MODELS / "burst_v12_meta.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {MODELS/'burst_v12_meta.json'}")


if __name__ == "__main__":
    main()
