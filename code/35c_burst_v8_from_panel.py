"""V8 (fast): reads burst_panel_v7.csv, adds 6 trend features derived from
`close` + existing `atr_pct`, recomputes burst target, trains, backtests.
No yfinance re-download."""

import sys; sys.stdout.reconfigure(line_buffering=True)

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"; OUT = ROOT / "output"

BURST_WINDOW = 5; BURST_MIN_LEN = 2; BURST_THRESH = 0.04

A_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
          "atr_pct", "range_pct", "vol_z", "vol_5d"]
V7_FEATS = A_BASE + ["rv_60", "overnight_gap"]
TREND = ["ma_stack", "up_streak", "up_bigdays_20d",
         "dist_ma60_atr", "ma60_slope_60d", "run_length"]
V8_FEATS = V7_FEATS + TREND


def add_trend_and_y(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    c = g["close"]
    r = c.pct_change()
    ma5 = c.rolling(5).mean(); ma20 = c.rolling(20).mean(); ma60 = c.rolling(60).mean()
    atr = g["atr_pct"] * c
    g["ma_stack"] = ((ma5 > ma20) & (ma20 > ma60)).astype(int)
    up = (r > 0).astype(int)
    grp = (up != up.shift()).cumsum()
    streak = up.groupby(grp).cumsum().where(up == 1, 0)
    g["up_streak"] = streak.clip(upper=30)
    g["up_bigdays_20d"] = (r > 0.03).rolling(20).sum()
    g["dist_ma60_atr"] = (c - ma60) / atr.replace(0, np.nan)
    g["ma60_slope_60d"] = (ma60 - ma60.shift(60)) / c
    above20 = (c > ma20).astype(int)
    grp2 = (above20 != above20.shift()).cumsum()
    run = above20.groupby(grp2).cumsum().where(above20 == 1, 0)
    g["run_length"] = run.clip(upper=120)
    # recompute y
    arr = r.fillna(0).values
    n = len(arr); y = np.zeros(n, dtype=np.int8)
    for t in range(n - BURST_WINDOW):
        fut = arr[t+1:t+1+BURST_WINDOW]; best = 0.0
        for L in range(BURST_MIN_LEN, BURST_WINDOW + 1):
            for s in range(0, BURST_WINDOW - L + 1):
                m = fut[s:s+L].mean()
                if m > best: best = m
        if best >= BURST_THRESH: y[t] = 1
    y[-BURST_WINDOW:] = -1
    g["y"] = y
    return g


def evaluate(y, p):
    base = float(np.mean(y)) if len(y) else float("nan")
    try: auc = roc_auc_score(y, p)
    except ValueError: auc = float("nan")
    try: ap = average_precision_score(y, p)
    except ValueError: ap = float("nan")
    ll = log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])
    bs = brier_score_loss(y, p)
    return {"n": int(len(y)), "pos": int(y.sum()), "base": base,
            "auc": float(auc), "ap": float(ap),
            "ap_lift": float(ap/base) if base > 0 else float("nan"),
            "log_loss": float(ll), "brier": float(bs)}


def main():
    panel = pd.read_csv(DATA / "burst_panel_v7.csv", parse_dates=["date"])
    print(f"[v8-fast] loaded {len(panel):,} panel rows, {panel['ticker'].nunique()} tickers")

    # add trend + y per ticker
    print("[v8-fast] computing trend features + target per ticker ...")
    out = []
    for t, g in panel.groupby("ticker", sort=False):
        out.append(add_trend_and_y(g))
    panel = pd.concat(out, ignore_index=True)
    panel.to_csv(DATA / "burst_panel_v8.csv", index=False)
    print(f"[v8-fast] wrote burst_panel_v8.csv  base rate: "
          f"{panel[panel['y']>=0]['y'].mean():.4%}")

    lab = panel[panel["y"] >= 0].dropna(subset=V8_FEATS).reset_index(drop=True)
    dates = np.sort(lab["date"].unique())
    d1 = dates[int(0.70 * len(dates))]; d2 = dates[int(0.85 * len(dates))]
    tr = lab[lab["date"] < d1]; va = lab[(lab["date"] >= d1) & (lab["date"] < d2)]; te = lab[lab["date"] >= d2]
    print(f"[v8-fast] split sizes: {len(tr):,}/{len(va):,}/{len(te):,}")

    def fit(feats, label):
        m = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                       learning_rate=0.05, subsample=0.8, random_state=42)
        m.fit(tr[feats].values, tr["y"].values)
        p_te = m.predict_proba(te[feats].values)[:, 1]
        met = evaluate(te["y"].values, p_te)
        print(f"[v8-fast] {label:<14} test AUC={met['auc']:.3f}  AP={met['ap']:.3f}  "
              f"lift={met['ap_lift']:.2f}x  log_loss={met['log_loss']:.4f}")
        return m, met

    print("[v8-fast] training v7 baseline ...")
    m7, met7 = fit(V7_FEATS, "v7 baseline")
    print("[v8-fast] training v8 (+trend) ...")
    m8, met8 = fit(V8_FEATS, "v8 +trend")
    print(f"[v8-fast] delta: AUC={met8['auc']-met7['auc']:+.4f}  AP={met8['ap']-met7['ap']:+.4f}")

    joblib.dump({"gbc": m8, "feats": V8_FEATS}, MODELS / "burst_gbc_v8_augmented.joblib")
    imp = pd.Series(m8.feature_importances_, index=V8_FEATS).sort_values(ascending=False)
    print("\n[v8-fast] feature importance:")
    print(imp.to_string())

    # stratified precision by run_length
    te2 = te.copy()
    te2["prob_v7"] = m7.predict_proba(te[V7_FEATS].values)[:, 1]
    te2["prob_v8"] = m8.predict_proba(te[V8_FEATS].values)[:, 1]
    print("\n[v8-fast] precision stratified by run_length (top 10% of bucket):")
    print(f"  {'bin':<8} {'n':>7} {'truth%':>8} {'P_v7':>8} {'P_v8':>8}  {'gain':>7}")
    for lo, hi, lbl in [(0, 5, "0-5"), (5, 20, "5-20"), (20, 60, "20-60"), (60, 999, "60+")]:
        sub = te2[(te2["run_length"] >= lo) & (te2["run_length"] < hi)]
        if len(sub) < 20: continue
        k = max(5, len(sub)//10)
        p7 = sub.sort_values("prob_v7", ascending=False).head(k)["y"].mean()
        p8 = sub.sort_values("prob_v8", ascending=False).head(k)["y"].mean()
        print(f"  {lbl:<8} {len(sub):>7} {sub['y'].mean()*100:>7.1f}% {p7*100:>7.1f}% {p8*100:>7.1f}%  {100*(p8-p7):>+6.1f}pp")

    # regression heads
    print("\n[v8-fast] training 1d/3d/5d regressors ...")
    p = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False)["close"]
    p["fwd_1d"] = g.shift(-1) / p["close"] - 1
    p["fwd_3d"] = g.shift(-3) / p["close"] - 1
    p["fwd_5d"] = g.shift(-5) / p["close"] - 1
    pr = p.dropna(subset=V8_FEATS + ["fwd_1d", "fwd_3d", "fwd_5d"]).reset_index(drop=True)
    rdates = np.sort(pr["date"].unique())
    rd1 = rdates[int(0.70 * len(rdates))]; rd2 = rdates[int(0.85 * len(rdates))]
    rtr = pr[pr["date"] < rd1]; rte = pr[pr["date"] >= rd2]
    reg_metrics = {}
    for h in ["fwd_1d", "fwd_3d", "fwd_5d"]:
        reg = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                        learning_rate=0.05, subsample=0.8, random_state=42)
        reg.fit(rtr[V8_FEATS].values, rtr[h].values)
        pte = reg.predict(rte[V8_FEATS].values)
        mae = float(np.mean(np.abs(rte[h].values - pte)))
        dir_acc = float(np.mean(np.sign(rte[h].values) == np.sign(pte)))
        reg_metrics[h] = {"mae": mae, "dir": dir_acc}
        joblib.dump({"reg": reg, "feats": V8_FEATS}, MODELS / f"burst_reg_v8_{h}.joblib")
        print(f"   {h}: MAE={mae:.4f}  dir={dir_acc:.3f}")

    (OUT / "burst_metrics_v8.json").write_text(json.dumps({
        "v7": met7, "v8": met8, "reg": reg_metrics,
        "features": V8_FEATS,
    }, indent=2))
    print("[v8-fast] done")


if __name__ == "__main__":
    main()
