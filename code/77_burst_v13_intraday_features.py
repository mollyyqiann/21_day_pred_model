"""Burst v13 — INTRADAY-target retrain with proper YESTERDAY-INTRADAY features

The morning model fires PRE-OPEN at 8:45 ET. Information available at signal
time: full historical OHLCV through yesterday's close, plus today's pre-market
snapshot. Target: today's intraday return + multi-day intraday continuation
(NOT close-to-close, because the overnight gap fires before the signal is
tradeable — see live grading: 89% of v11's pred c2c return was the gap that
already happened, mean entry-frame return was +0.03%).

v12 fixed the target framing but kept shallow close-based features. v13
deepens the feature set with YESTERDAY-INTRADAY characteristics derivable
from daily OHLC (5y of full coverage):

  Yesterday-intraday features (from data/ohlc_prices.csv):
    yest_oc_ret           (close - open) / open    — yesterday's intraday return
    yest_close_in_range   (close - low) / (high - low)  — close strength 0..1
    yest_pullback_pct     (high - close) / close   — pullback from intraday high
    yest_open_to_low      (open - low) / open      — intraday drawdown vs open
    yest_high_low_pct     (high - low) / open      — yesterday's intraday range
    yest_volume_zscore    (vol - 20d_mean) / 20d_std
    yest_oc_streak_5d     # of positive oc days in last 5
    yest_oc_mean_5d       mean of oc returns over last 5 sessions
    yest_oc_vol_20d       std of oc returns over 20 sessions  (intraday vol)
    yest_close_vs_vwap    (close - 20d_avg_close) / 20d_avg_close   (proxy)

  Today pre-market (existing):
    overnight_gap         pm_last / close_prev - 1
    + INTRADAY_FEATS where available (low coverage but HGB tolerates NaN)

  Plus existing close-based stack: rsi/macd/bb/atr_pct/range_pct/vol_z/rv_60
  Plus regime: spy_ret_5d/20d, spy_rv_20/60, vix, vix_chg_5d, fng

Target (anchored at signal day t+1, where row's date = t = previous close):
    open_next  = open price on day t+1 (the entry price)
    fwd_intra_1d = close[t+1] / open_next - 1
    fwd_intra_3d = close[t+3] / open_next - 1
    fwd_intra_5d = close[t+5] / open_next - 1

Outputs:
    models/burst_gbc_v13_{v4,v5,v7}.joblib
    models/burst_reg_v13_{v4,v5,v7}_intra_{1d,3d,5d}.joblib
    models/burst_v13_meta.json
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
    "v4": V7_FEATS, "v5": V7_FEATS[:-1] + V5_EXTRA + ["overnight_gap"], "v7": V7_FEATS,
}

# New: yesterday-intraday features derived from daily OHLC (full historical coverage).
YEST_INTRA_FEATS = [
    "yest_oc_ret", "yest_close_in_range", "yest_pullback_pct",
    "yest_open_to_low", "yest_high_low_pct", "yest_volume_zscore",
    "yest_oc_streak_5d", "yest_oc_mean_5d", "yest_oc_vol_20d",
    "yest_close_vs_ma20",
]

# Existing intraday features from minute-bar store (sparse coverage but useful where present).
INTRADAY_FEATS = [
    "pm_gap_pct", "pm_range_pct", "pm_drift",
    "ret_first_30m", "ret_last_30m", "intraday_rv_5m",
    "vwap_close_dist", "intraday_dd", "vol_late_share", "close_strength",
]

WINDOW = 5; MIN_LEN = 2; THRESH = 0.04


def v13_feats(uni: str) -> list[str]:
    return BASE_FEATS[uni] + YEST_INTRA_FEATS + REGIME_FEATS + INTRADAY_FEATS


def _build_yest_intra(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Per-(ticker, date) yesterday-intraday features. All causal."""
    o = ohlc.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    g = o.groupby("ticker", sort=False)
    rng = (o["high"] - o["low"]).replace(0, np.nan)
    oc = (o["close"] - o["open"]) / o["open"]
    o["yest_oc_ret"] = oc
    o["yest_close_in_range"] = (o["close"] - o["low"]) / rng
    o["yest_pullback_pct"] = (o["high"] - o["close"]) / o["close"]
    o["yest_open_to_low"] = (o["open"] - o["low"]) / o["open"]
    o["yest_high_low_pct"] = (o["high"] - o["low"]) / o["open"]
    o["yest_oc_mean_5d"] = g["yest_oc_ret"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    o["yest_oc_vol_20d"] = g["yest_oc_ret"].transform(lambda s: s.rolling(20, min_periods=10).std())
    o["yest_oc_streak_5d"] = g["yest_oc_ret"].transform(
        lambda s: s.rolling(5, min_periods=3).apply(lambda w: float((w > 0).sum()), raw=False))
    vmean20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    vstd20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).std())
    o["yest_volume_zscore"] = (o["volume"] - vmean20) / vstd20.replace(0, np.nan)
    cma20 = g["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    o["yest_close_vs_ma20"] = (o["close"] - cma20) / cma20
    return o[["ticker", "date"] + YEST_INTRA_FEATS + ["open", "high", "low", "close", "volume"]]


def attach_ohlc_yest_targets(p: pd.DataFrame, ohlc_yi: pd.DataFrame) -> pd.DataFrame:
    """Adds: yesterday-intraday features (joined on row's own date) +
    open_next + close_t1/t3/t5 + fwd_intra_*.
    The row at date t carries close[t] features; target uses open[t+1] entry."""
    p = p.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p = p.merge(ohlc_yi, on=["ticker", "date"], how="left", suffixes=("", "_oh"))
    # If the panel already has close (it does), don't double-write — keep panel's close.
    # But we need the OHLC's open/high/low/volume for current-day intraday and shifted target.
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False)
    # The row at t carries features through close[t]. The trader using this row
    # enters at open[t+1] and exits at close[t+1] (1d), close[t+3] (3d), close[t+5] (5d).
    p["open_next"] = g["open"].shift(-1)
    p["close_t1"] = g["close"].shift(-1)
    p["close_t3"] = g["close"].shift(-3)
    p["close_t5"] = g["close"].shift(-5)
    p["fwd_intra_1d"] = p["close_t1"] / p["open_next"] - 1
    p["fwd_intra_3d"] = p["close_t3"] / p["open_next"] - 1
    p["fwd_intra_5d"] = p["close_t5"] / p["open_next"] - 1
    # Per-day intraday return for label-building (uses today's open/close).
    p["intra_today"] = (p["close"] - p["open"]) / p["open"]
    return p


def build_intraday_burst_label(intra: pd.Series) -> pd.Series:
    """y = 1 iff some 2..5 day window of FUTURE intraday-from-open returns
    averages >= 4%/day. Anchored: row t looks at t+1..t+5."""
    r = intra.fillna(0).values
    n = len(r); y = np.zeros(n, dtype=np.int8)
    for t in range(n - WINDOW):
        fut = r[t+1:t+1+WINDOW]; best = 0.0
        for L in range(MIN_LEN, WINDOW + 1):
            for s in range(0, WINDOW - L + 1):
                m = fut[s:s+L].mean()
                if m > best: best = m
        if best >= THRESH: y[t] = 1
    out = pd.Series(y, index=intra.index)
    out.iloc[-WINDOW:] = -1
    return out


def attach_labels(p: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in p.groupby("ticker", sort=False):
        parts.append(build_intraday_burst_label(g["intra_today"]))
    p["y"] = pd.concat(parts).reindex(p.index).values
    return p


def attach_intraday_minute_feats(p: pd.DataFrame) -> pd.DataFrame:
    fp = DATA / "intraday_daily_features.csv"
    if not fp.exists():
        for c in INTRADAY_FEATS:
            if c not in p.columns: p[c] = np.nan
        return p
    f = pd.read_csv(fp, parse_dates=["date"])
    f["date"] = f["date"].dt.normalize()
    keep = ["ticker", "date"] + [c for c in INTRADAY_FEATS if c in f.columns]
    return p.merge(f[keep], on=["ticker", "date"], how="left")


def _split_dates(d: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = np.sort(d.unique())
    return dates[int(0.70 * len(dates))], dates[int(0.85 * len(dates))]


def train_one(panel_path: Path, uni_tag: str, ohlc_yi: pd.DataFrame,
              cls_grid: list[tuple]) -> dict:
    print(f"\n=== v13 ({uni_tag}) — {panel_path.name} ===")
    t0 = time.time()
    feats = v13_feats(uni_tag)
    base = BASE_FEATS[uni_tag]
    required = base + YEST_INTRA_FEATS + REGIME_FEATS  # intraday minute feats optional

    p = pd.read_csv(panel_path, parse_dates=["date"])
    p = p[p[base].notna().all(axis=1)].copy()
    p = attach_regime(p)
    p = attach_ohlc_yest_targets(p, ohlc_yi)
    p = attach_intraday_minute_feats(p)
    p = attach_labels(p)
    print(f"  open_next coverage: {p['open_next'].notna().mean():.2%}")
    yi_cov = p[YEST_INTRA_FEATS].notna().all(axis=1).mean()
    print(f"  yest-intraday feats coverage: {yi_cov:.2%}")
    intra_cov = p[INTRADAY_FEATS[0]].notna().mean()
    print(f"  minute-bar intraday coverage: {intra_cov:.2%}")

    # Classifier
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
    te_df = te.assign(_p=pt)
    hits5, hits10, fwd5, fwd3, fwd1 = [], [], [], [], []
    for _, gd in te_df.groupby("date", sort=False):
        if len(gd) < 10: continue
        t5 = gd.sort_values("_p", ascending=False).head(5)
        hits5.append(t5["y"].mean())
        hits10.append(gd.sort_values("_p", ascending=False).head(10)["y"].mean())
        for lst, col in [(fwd5, "fwd_intra_5d"), (fwd3, "fwd_intra_3d"), (fwd1, "fwd_intra_1d")]:
            f = t5[col].dropna()
            if len(f): lst.append(float(f.mean()))
    cls_metrics["prec@5"] = float(np.mean(hits5)) if hits5 else float("nan")
    cls_metrics["prec@10"] = float(np.mean(hits10)) if hits10 else float("nan")
    cls_metrics["mean_fwd_intra_5d_top5"] = float(np.mean(fwd5)) if fwd5 else float("nan")
    cls_metrics["mean_fwd_intra_3d_top5"] = float(np.mean(fwd3)) if fwd3 else float("nan")
    cls_metrics["mean_fwd_intra_1d_top5"] = float(np.mean(fwd1)) if fwd1 else float("nan")
    cls_metrics["n_test_days"] = int(len(hits5))
    print(f"  prec@5={cls_metrics['prec@5']:.4f}  AUC={cls_metrics['AUC']:.4f}  "
          f"top5: 1d={cls_metrics['mean_fwd_intra_1d_top5']*100:+.2f}%  "
          f"3d={cls_metrics['mean_fwd_intra_3d_top5']*100:+.2f}%  "
          f"5d={cls_metrics['mean_fwd_intra_5d_top5']*100:+.2f}%")

    pv_for_cal = cls.predict_proba(X_va)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pv_for_cal, y_va)

    cls_path = MODELS / f"burst_gbc_v13_{uni_tag}.joblib"
    joblib.dump({"gbc": cls, "feats": feats, "version": "v13",
                 "metrics": cls_metrics, "calibrator": iso,
                 "best_params": {"learning_rate": lr, "max_depth": md, "max_iter": mi},
                 "universe": uni_tag,
                 "yest_intra_coverage_train": float(tr[YEST_INTRA_FEATS].notna().all(axis=1).mean()),
                 "label_def": {"window": WINDOW, "min_len": MIN_LEN, "thresh": THRESH,
                                "anchor": "intraday_open_to_close"}},
                cls_path)
    print(f"  wrote {cls_path.name}")

    reg_metrics = {}
    rr = p.dropna(subset=required + ["fwd_intra_1d", "fwd_intra_3d", "fwd_intra_5d"]).reset_index(drop=True)
    rd1, rd2 = _split_dates(rr["date"])
    rtr = rr[rr["date"] <  rd1]; rte = rr[rr["date"] >= rd2]
    print(f"  reg train={len(rtr):,}  test={len(rte):,}")
    for h in ["fwd_intra_1d", "fwd_intra_3d", "fwd_intra_5d"]:
        reg = HistGradientBoostingRegressor(max_iter=400, max_depth=4,
                                             learning_rate=0.04, random_state=42)
        reg.fit(rtr[feats].values, rtr[h].values)
        pr = reg.predict(rte[feats].values)
        rmae = float(mean_absolute_error(rte[h].values, pr))
        rmae_zero = float(np.mean(np.abs(rte[h].values)))
        dir_hit = float(np.mean(np.sign(pr) == np.sign(rte[h].values)))
        reg_metrics[h] = {"n_test": int(len(rte)), "MAE": rmae,
                           "MAE_vs_zero": rmae_zero, "dir_hit": dir_hit,
                           "skill_vs_zero": 1 - rmae / rmae_zero if rmae_zero > 0 else float("nan")}
        rp = MODELS / f"burst_reg_v13_{uni_tag}_{h.replace('fwd_intra_', 'intra_')}.joblib"
        joblib.dump({"reg": reg, "feats": feats, "version": "v13",
                     "horizon": h, "universe": uni_tag,
                     "metrics": reg_metrics[h]}, rp)
        print(f"  {h:>15s}: MAE={rmae:.4f} (vs zero {rmae_zero:.4f}, skill {(1-rmae/rmae_zero)*100:+.1f}%)  dir={dir_hit:.3f}")

    return {"universe": uni_tag, "panel": panel_path.name,
            "cls_metrics": cls_metrics, "reg_metrics": reg_metrics,
            "best_cls_params": {"learning_rate": lr, "max_depth": md, "max_iter": mi},
            "runtime_sec": round(time.time() - t0, 1)}


def main():
    print("[v13] loading OHLC + building yest-intraday features ...")
    ohlc = pd.read_csv(DATA / "ohlc_prices.csv", parse_dates=["date"])
    ohlc["date"] = ohlc["date"].dt.normalize()
    yi = _build_yest_intra(ohlc)
    print(f"[v13] yest-intraday rows: {len(yi):,}, tickers: {yi['ticker'].nunique()}")

    grids = [(0.05, 5, 300), (0.05, 7, 400), (0.03, 7, 600)]
    out = {"version": "v13", "label_def": {"window": WINDOW, "min_len": MIN_LEN,
                                            "thresh": THRESH,
                                            "anchor": "intraday_open_to_close"},
           "feats": {u: v13_feats(u) for u in ("v4", "v5", "v7")},
           "yest_intra_feats": YEST_INTRA_FEATS,
           "per_universe": {}}
    panels = [(DATA / "burst_panel_v6b.csv", "v4"),
              (DATA / "burst_panel_v6.csv",  "v5"),
              (DATA / "burst_panel_v7.csv",  "v7")]
    for path, uni in panels:
        out["per_universe"][uni] = train_one(path, uni, yi, grids)
    (MODELS / "burst_v13_meta.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {MODELS/'burst_v13_meta.json'}")


if __name__ == "__main__":
    main()
