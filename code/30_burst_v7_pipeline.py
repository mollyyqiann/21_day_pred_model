"""V7 — full S&P 500 universe (no price/liquidity/asymmetry filters).

Features: A_base (9) + rv_60 + overnight_gap = 11 (same as v6b).
Training protocol mirrors v6b.

Outputs:
  data/burst_universe_v7.csv
  data/burst_panel_v7.csv
  models/burst_gbc_v7_vanilla.joblib
  models/burst_gbc_v7_augmented.joblib
  models/burst_reg_v7_fwd_{1d,3d,5d}.joblib
  output/burst_metrics_v7.json
  output/burst_reg_metrics_v7.json
  output/burst_today_v7.csv
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

MIN_OBS = 120  # minimum history so 60d rolling features warm up
BURST_WINDOW = 5; BURST_MIN_LEN = 2; BURST_THRESH = 0.04

A_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
          "atr_pct", "range_pct", "vol_z", "vol_5d"]
V4_FEATS = A_BASE + ["rv_60"]
V7_FEATS = V4_FEATS + ["overnight_gap"]    # same 11 as v6b


def rsi(x, n=14):
    d = x.diff(); u = d.clip(lower=0).rolling(n).mean(); dd = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + u / dd.replace(0, np.nan))
def macd(x, f=12, s=26, sig=9):
    ef = x.ewm(span=f, adjust=False).mean(); es = x.ewm(span=s, adjust=False).mean()
    line = ef - es; signal = line.ewm(span=sig, adjust=False).mean()
    return line / x, signal / x, (line - signal) / x
def bb_z(x, n=20):
    return (x - x.rolling(n).mean()) / x.rolling(n).std()
def atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
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
    out["bb_z20"] = bb_z(c); out["atr_pct"] = atr(h, l, c) / c
    out["range_pct"] = (h - l) / c
    vm = v.rolling(30).mean(); vs = v.rolling(30).std().replace(0, np.nan)
    out["vol_z"] = (v - vm) / vs; out["vol_5d"] = v.rolling(5).mean() / vm
    r = c.pct_change(); out["rv_60"] = r.rolling(60).std() * math.sqrt(252)
    out["overnight_gap"] = (o.shift(-1) / c) - 1.0
    return out


def build_target(c):
    r = c.pct_change().fillna(0).values
    n = len(r); y = np.zeros(n, dtype=np.int8)
    for t in range(n - BURST_WINDOW):
        fut = r[t+1:t+1+BURST_WINDOW]; best = 0.0
        for L in range(BURST_MIN_LEN, BURST_WINDOW + 1):
            for s in range(0, BURST_WINDOW - L + 1):
                m = fut[s:s+L].mean()
                if m > best: best = m
        if best >= BURST_THRESH: y[t] = 1
    out = pd.Series(y, index=c.index); out.iloc[-BURST_WINDOW:] = -1
    return out


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
    print(f"[v7] SP500: {len(sp)}")

    print("[v7] bulk downloading 3y daily ...")
    raw = yf.download(sp["ticker"].tolist(), period="3y", interval="1d",
                      group_by="ticker", auto_adjust=True, threads=True, progress=False)

    uni_rows = []; panels = []
    for t in sp["ticker"]:
        try: sub = raw[t].dropna()
        except KeyError: continue
        if len(sub) < MIN_OBS: continue
        price = float(sub["Close"].iloc[-1])
        if price <= 0: continue
        adv = float((sub["Close"] * sub["Volume"]).tail(30).mean())
        rv60 = float(sub["Close"].pct_change().rolling(60).std().iloc[-1] * math.sqrt(252))
        feats = build_features(sub)
        feats["y"] = build_target(sub["Close"])
        feats["close"] = sub["Close"]
        feats["ticker"] = t
        feats = feats.reset_index().rename(columns={"Date": "date"})
        panels.append(feats)
        uni_rows.append({"ticker": t, "price": price, "adv_usd": adv,
                         "rv_60_now": rv60, "n_obs": len(sub)})

    uni = pd.DataFrame(uni_rows).merge(sp[["ticker", "sector"]], on="ticker", how="left")
    uni.to_csv(DATA / "burst_universe_v7.csv", index=False)
    panel = pd.concat(panels, ignore_index=True)
    panel.to_csv(DATA / "burst_panel_v7.csv", index=False)
    print(f"[v7] universe: {len(uni)}  panel rows: {len(panel):,}")

    lab = panel[panel["y"] >= 0].dropna(subset=V7_FEATS).reset_index(drop=True)
    print(f"[v7] labelled rows: {len(lab):,}  base rate: {lab['y'].mean():.4%}")
    dates = np.sort(lab["date"].unique())
    d1 = dates[int(0.70 * len(dates))]; d2 = dates[int(0.85 * len(dates))]
    tr = lab[lab["date"] < d1]; va = lab[(lab["date"] >= d1) & (lab["date"] < d2)]; te = lab[lab["date"] >= d2]

    # === classifier (vanilla vs augmented) ===
    def fit_clf(feats):
        m = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=42)
        m.fit(tr[feats].values, tr["y"].values)
        return m, {"test": evaluate(te["y"].values, m.predict_proba(te[feats].values)[:, 1])}

    print("[v7] training vanilla ...")
    gbc_v, mv = fit_clf(V4_FEATS)
    print(f"    test: AUC={mv['test']['auc']:.3f}  AP={mv['test']['ap']:.3f}  "
          f"lift={mv['test']['ap_lift']:.2f}x")

    print("[v7] training augmented ...")
    gbc_a, ma = fit_clf(V7_FEATS)
    print(f"    test: AUC={ma['test']['auc']:.3f}  AP={ma['test']['ap']:.3f}  "
          f"lift={ma['test']['ap_lift']:.2f}x")

    auc_gain = ma["test"]["auc"] - mv["test"]["auc"]
    ap_gain = ma["test"]["ap"] - mv["test"]["ap"]
    passes = auc_gain > 0 and ap_gain > 0
    print(f"[v7] delta: AUC={auc_gain:+.4f}  AP={ap_gain:+.4f}  "
          f"verdict: {'ACCEPT' if passes else 'REJECT'}")

    joblib.dump({"gbc": gbc_v, "feats": V4_FEATS}, MODELS / "burst_gbc_v7_vanilla.joblib")
    joblib.dump({"gbc": gbc_a, "feats": V7_FEATS}, MODELS / "burst_gbc_v7_augmented.joblib")
    (OUT / "burst_metrics_v7.json").write_text(json.dumps({
        "vanilla": mv, "augmented": ma,
        "auc_gain": auc_gain, "ap_gain": ap_gain,
        "passes_vs_vanilla": bool(passes),
    }, indent=2))

    # === regression heads ===
    print("[v7] training 1d/3d/5d regressors ...")
    p = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False)["close"]
    p["fwd_1d"] = g.shift(-1) / p["close"] - 1
    p["fwd_3d"] = g.shift(-3) / p["close"] - 1
    p["fwd_5d"] = g.shift(-5) / p["close"] - 1
    pr = p.dropna(subset=V7_FEATS + ["fwd_1d", "fwd_3d", "fwd_5d"]).reset_index(drop=True)
    rdates = np.sort(pr["date"].unique())
    rd1 = rdates[int(0.70 * len(rdates))]; rd2 = rdates[int(0.85 * len(rdates))]
    rtr = pr[pr["date"] < rd1]; rte = pr[pr["date"] >= rd2]

    reg_metrics = {}
    for h in ["fwd_1d", "fwd_3d", "fwd_5d"]:
        reg = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=42)
        reg.fit(rtr[V7_FEATS].values, rtr[h].values)
        p_te = reg.predict(rte[V7_FEATS].values)
        mae = float(np.mean(np.abs(rte[h].values - p_te)))
        dir_acc = float(np.mean(np.sign(rte[h].values) == np.sign(p_te)))
        reg_metrics[h] = {"mae": mae, "direction_acc": dir_acc, "n_test": int(len(rte))}
        joblib.dump({"reg": reg, "feats": V7_FEATS},
                    MODELS / f"burst_reg_v7_{h}.joblib")
        print(f"    {h}: MAE={mae:.4f}  dir={dir_acc:.3f}")

    (OUT / "burst_reg_metrics_v7.json").write_text(json.dumps(reg_metrics, indent=2))

    # === score today ===
    scored = panel.dropna(subset=V4_FEATS).copy()
    latest = scored.sort_values(["ticker", "date"]).groupby("ticker").tail(1).copy()
    latest["prob_vanilla"] = gbc_v.predict_proba(latest[V4_FEATS].values)[:, 1]
    aug = latest.copy(); aug["overnight_gap"] = aug["overnight_gap"].fillna(0.0)
    latest["prob_augmented_zero_gap"] = gbc_a.predict_proba(aug[V7_FEATS].values)[:, 1]
    latest = latest.merge(uni[["ticker", "sector", "rv_60_now"]], on="ticker", how="left")
    latest = latest.sort_values("prob_augmented_zero_gap", ascending=False).reset_index(drop=True)
    cols = ["ticker", "date", "close", "prob_vanilla", "prob_augmented_zero_gap",
            "rv_60_now", "rsi_14", "bb_z20", "atr_pct", "sector"]
    latest[cols].to_csv(OUT / "burst_today_v7.csv", index=False)
    print(f"\n[v7] top 15:")
    print(latest[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
