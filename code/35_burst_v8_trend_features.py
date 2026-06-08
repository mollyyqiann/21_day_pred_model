"""V8 — add trend-continuation features to v7 to fix the fade on sustained
multi-week / multi-month runs (SNDK, MSTR style).

New features on top of V7 (A_base + rv_60 + overnight_gap):
  - ma_stack        : 1 if MA5 > MA20 > MA60 (bullish stacking), else 0
  - up_streak       : consecutive days closing up (capped at 30)
  - up_bigdays_20d  : count of daily returns > +3% over trailing 20 trading days
  - dist_ma60_atr   : (close - MA60) / ATR(14) — extension normalized by own vol
  - ma60_slope_60d  : (MA60[t] - MA60[t-60]) / close — persistence of trend
  - run_length      : consecutive days with close > 20d MA (capped 120)

= 17 features total.

Training: same v7 universe (full S&P 500, 502 tickers). Backtest on the same
test fold and explicitly stratify precision by run_length bucket.
"""

from __future__ import annotations

import io
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"; OUT = ROOT / "output"

MIN_OBS = 120
BURST_WINDOW = 5; BURST_MIN_LEN = 2; BURST_THRESH = 0.04

A_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
          "atr_pct", "range_pct", "vol_z", "vol_5d"]
V7_FEATS = A_BASE + ["rv_60", "overnight_gap"]      # 11
TREND = ["ma_stack", "up_streak", "up_bigdays_20d",
         "dist_ma60_atr", "ma60_slope_60d", "run_length"]
V8_FEATS = V7_FEATS + TREND                         # 17


def rsi(x, n=14):
    d = x.diff(); u = d.clip(lower=0).rolling(n).mean(); dd = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + u / dd.replace(0, np.nan))
def macd(x, f=12, s=26, sig=9):
    ef = x.ewm(span=f, adjust=False).mean(); es = x.ewm(span=s, adjust=False).mean()
    line = ef - es; signal = line.ewm(span=sig, adjust=False).mean()
    return line / x, signal / x, (line - signal) / x
def bb_z(x, n=20): return (x - x.rolling(n).mean()) / x.rolling(n).std()
def atr(h, l, c, n=14):
    pc = c.shift(1); tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def fetch_sp500():
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0][["Symbol", "Security", "GICS Sector"]].copy()
    t.columns = ["ticker", "name", "sector"]
    t["ticker"] = t["ticker"].str.replace(".", "-", regex=False)
    return t


def build_features(df):
    out = pd.DataFrame(index=df.index)
    c, h, l, v, o = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]
    out["rsi_14"] = rsi(c); ml, ms, mh = macd(c)
    out["macd"], out["macd_sig"], out["macd_hist"] = ml, ms, mh
    out["bb_z20"] = bb_z(c)
    atr14 = atr(h, l, c, 14)
    out["atr_pct"] = atr14 / c
    out["range_pct"] = (h - l) / c
    vm = v.rolling(30).mean(); vs = v.rolling(30).std().replace(0, np.nan)
    out["vol_z"] = (v - vm) / vs; out["vol_5d"] = v.rolling(5).mean() / vm
    r = c.pct_change(); out["rv_60"] = r.rolling(60).std() * math.sqrt(252)
    out["overnight_gap"] = (o.shift(-1) / c) - 1.0

    # === trend features (new) ===
    ma5 = c.rolling(5).mean(); ma20 = c.rolling(20).mean(); ma60 = c.rolling(60).mean()
    out["ma_stack"] = ((ma5 > ma20) & (ma20 > ma60)).astype(int)
    up = (r > 0).astype(int)
    grp = (up != up.shift()).cumsum()
    out["up_streak"] = up.groupby(grp).cumsum().where(up == 1, 0).clip(upper=30)
    out["up_bigdays_20d"] = (r > 0.03).rolling(20).sum()
    out["dist_ma60_atr"] = (c - ma60) / atr14.replace(0, np.nan)
    out["ma60_slope_60d"] = (ma60 - ma60.shift(60)) / c
    above20 = (c > ma20).astype(int)
    grp2 = (above20 != above20.shift()).cumsum()
    out["run_length"] = above20.groupby(grp2).cumsum().where(above20 == 1, 0).clip(upper=120)
    return out


def build_target(c):
    r = c.pct_change().fillna(0).values; n = len(r); y = np.zeros(n, dtype=np.int8)
    for t in range(n - BURST_WINDOW):
        fut = r[t+1:t+1+BURST_WINDOW]; best = 0.0
        for L in range(BURST_MIN_LEN, BURST_WINDOW + 1):
            for s in range(0, BURST_WINDOW - L + 1):
                m = fut[s:s+L].mean()
                if m > best: best = m
        if best >= BURST_THRESH: y[t] = 1
    s = pd.Series(y, index=c.index); s.iloc[-BURST_WINDOW:] = -1
    return s


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
    sp = fetch_sp500()
    print(f"[v8] SP500: {len(sp)}")
    print("[v8] bulk downloading 3y daily ...")
    raw = yf.download(sp["ticker"].tolist(), period="3y", interval="1d",
                      group_by="ticker", auto_adjust=True, threads=True, progress=False)

    panels = []; uni_rows = []
    for t in sp["ticker"]:
        try: sub = raw[t].dropna()
        except KeyError: continue
        if len(sub) < MIN_OBS: continue
        feats = build_features(sub)
        feats["y"] = build_target(sub["Close"])
        feats["close"] = sub["Close"]
        feats["ticker"] = t
        feats = feats.reset_index().rename(columns={"Date": "date"})
        panels.append(feats)
        uni_rows.append({"ticker": t, "price": float(sub["Close"].iloc[-1])})

    uni = pd.DataFrame(uni_rows).merge(sp[["ticker", "sector"]], on="ticker", how="left")
    uni.to_csv(DATA / "burst_universe_v8.csv", index=False)
    panel = pd.concat(panels, ignore_index=True)
    panel.to_csv(DATA / "burst_panel_v8.csv", index=False)
    print(f"[v8] universe: {len(uni)}  panel rows: {len(panel):,}")

    lab = panel[panel["y"] >= 0].dropna(subset=V8_FEATS).reset_index(drop=True)
    print(f"[v8] labelled rows: {len(lab):,}  base rate: {lab['y'].mean():.4%}")
    dates = np.sort(lab["date"].unique())
    d1 = dates[int(0.70 * len(dates))]; d2 = dates[int(0.85 * len(dates))]
    tr = lab[lab["date"] < d1]; va = lab[(lab["date"] >= d1) & (lab["date"] < d2)]; te = lab[lab["date"] >= d2]

    # === classifiers ===
    def fit(feats, label):
        m = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=42)
        m.fit(tr[feats].values, tr["y"].values)
        p_te = m.predict_proba(te[feats].values)[:, 1]
        met = evaluate(te["y"].values, p_te)
        print(f"[v8] {label:<12} test: AUC={met['auc']:.3f}  AP={met['ap']:.3f}  "
              f"lift={met['ap_lift']:.2f}x  log_loss={met['log_loss']:.4f}")
        return m, met

    print("\n[v8] training v7 (baseline) vs v8 (with trend features)")
    m_v7, met_v7 = fit(V7_FEATS, "v7 baseline")
    m_v8, met_v8 = fit(V8_FEATS, "v8 trend")
    auc_gain = met_v8["auc"] - met_v7["auc"]; ap_gain = met_v8["ap"] - met_v7["ap"]
    print(f"[v8] delta: AUC={auc_gain:+.4f}  AP={ap_gain:+.4f}")

    joblib.dump({"gbc": m_v8, "feats": V8_FEATS}, MODELS / "burst_gbc_v8_augmented.joblib")

    imp = pd.Series(m_v8.feature_importances_, index=V8_FEATS).sort_values(ascending=False)
    print("\n[v8] feature importance:")
    print(imp.to_string())

    # === stratified precision by run_length on test fold ===
    te2 = te.copy()
    te2["prob_v7"] = m_v7.predict_proba(te[V7_FEATS].values)[:, 1]
    te2["prob_v8"] = m_v8.predict_proba(te[V8_FEATS].values)[:, 1]

    print("\n[v8] precision stratified by run_length (days above MA20) — "
          "top-10% of each bucket:")
    print(f"  {'bin':<12} {'n':>6} {'truth%':>8} {'P@top10_v7':>12} {'P@top10_v8':>12}")
    for lo, hi, lbl in [(0, 5, "0-5"), (5, 20, "5-20"), (20, 60, "20-60"),
                         (60, 999, "60+")]:
        sub = te2[(te2["run_length"] >= lo) & (te2["run_length"] < hi)]
        if len(sub) < 20: continue
        k = max(5, len(sub)//10)
        p7 = sub.sort_values("prob_v7", ascending=False).head(k)["y"].mean()
        p8 = sub.sort_values("prob_v8", ascending=False).head(k)["y"].mean()
        print(f"  {lbl:<12} {len(sub):>6} {sub['y'].mean()*100:>7.1f}% {p7*100:>11.1f}% {p8*100:>11.1f}%")

    # === regression heads (1d/3d/5d) ===
    print("\n[v8] training 1d/3d/5d regressors with v8 features ...")
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
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=42)
        reg.fit(rtr[V8_FEATS].values, rtr[h].values)
        p_te = reg.predict(rte[V8_FEATS].values)
        mae = float(np.mean(np.abs(rte[h].values - p_te)))
        dir_acc = float(np.mean(np.sign(rte[h].values) == np.sign(p_te)))
        reg_metrics[h] = {"mae": mae, "direction_acc": dir_acc}
        joblib.dump({"reg": reg, "feats": V8_FEATS},
                    MODELS / f"burst_reg_v8_{h}.joblib")
        print(f"    {h}: MAE={mae:.4f}  dir={dir_acc:.3f}")

    (OUT / "burst_metrics_v8.json").write_text(json.dumps({
        "vanilla_v7": met_v7, "v8": met_v8,
        "auc_gain": auc_gain, "ap_gain": ap_gain,
        "features": V8_FEATS, "reg": reg_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
