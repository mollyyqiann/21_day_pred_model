"""V4 — broad universe pipeline.

Drop the 'historically calm' filter. Include all liquid S&P 500 names above $40,
regardless of volatility regime. Keep the winning 9-feature A_base set and add
one extra feature so the model can condition on the ticker's volatility state:

  extended_features = A_base (9 features) + rv_60 (1 feature) = 10 features

This is the minimal architectural change that makes the model 'volatility-aware'
without re-introducing the feature bloat that hurt v2.

Stages:
  1. Pull full S&P 500 list (503 tickers).
  2. Bulk download 3y daily history.
  3. Keep tickers with current price >= $40, 30d ADV >= $25M, >= 150 trading days.
  4. Compute features + burst target (same 2-5 day / 4%/day rule).
  5. Chronological 70/15/15 split; train GradientBoostingClassifier.
  6. Score every ticker for today; save ranked predictions.

Outputs:
  data/burst_universe_v4.csv
  data/burst_panel_v4.csv
  models/burst_gbc_v4.joblib
  output/burst_metrics_v4.json
  output/burst_today_v4.csv
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output"
MODELS.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

PRICE_MIN = 40.0
ADV_MIN_USD = 25_000_000
MIN_OBS = 150
PERIOD = "3y"

BURST_WINDOW = 5
BURST_MIN_LEN = 2
BURST_THRESH = 0.04

A_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
          "atr_pct", "range_pct", "vol_z", "vol_5d"]
FEATS = A_BASE + ["rv_60"]   # 10 features


# ---------- helpers ----------

def fetch_sp500():
    html = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0][["Symbol", "Security", "GICS Sector"]].copy()
    t.columns = ["ticker", "name", "sector"]
    t["ticker"] = t["ticker"].str.replace(".", "-", regex=False)
    return t


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
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def build_features(df):
    out = pd.DataFrame(index=df.index)
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    out["rsi_14"] = rsi(c, 14)
    ml, ms, mh = macd(c); out["macd"], out["macd_sig"], out["macd_hist"] = ml, ms, mh
    out["bb_z20"] = bb_z(c, 20)
    out["atr_pct"] = atr(h, l, c, 14) / c
    out["range_pct"] = (h - l) / c
    vm = v.rolling(30).mean(); vs = v.rolling(30).std().replace(0, np.nan)
    out["vol_z"] = (v - vm) / vs
    out["vol_5d"] = v.rolling(5).mean() / vm
    r = c.pct_change()
    out["rv_60"] = r.rolling(60).std() * math.sqrt(252)
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
        if best >= BURST_THRESH:
            y[t] = 1
    out = pd.Series(y, index=c.index)
    out.iloc[-BURST_WINDOW:] = -1
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
    print("[v4] fetching SP500 list ...")
    sp = fetch_sp500()
    print(f"[v4] {len(sp)} constituents")

    print("[v4] bulk downloading 3y daily history ...")
    raw = yf.download(sp["ticker"].tolist(), period=PERIOD, interval="1d",
                      group_by="ticker", auto_adjust=True, threads=True, progress=False)

    uni_rows = []; panels = []; skipped_reason = {"no_data": 0, "too_short": 0,
                                                    "price": 0, "adv": 0}
    for t in sp["ticker"]:
        try:
            sub = raw[t].dropna()
        except KeyError:
            skipped_reason["no_data"] += 1; continue
        if len(sub) < MIN_OBS:
            skipped_reason["too_short"] += 1; continue
        price = float(sub["Close"].iloc[-1])
        if price < PRICE_MIN:
            skipped_reason["price"] += 1; continue
        adv = float((sub["Close"] * sub["Volume"]).tail(30).mean())
        if adv < ADV_MIN_USD:
            skipped_reason["adv"] += 1; continue

        # metrics for universe
        rv60_now = float(sub["Close"].pct_change().rolling(60).std().iloc[-1] * math.sqrt(252))
        uni_rows.append({"ticker": t, "price": price, "adv_usd": adv,
                         "rv_60_now": rv60_now, "n_obs": len(sub)})

        feats = build_features(sub)
        feats["y"] = build_target(sub["Close"])
        feats["close"] = sub["Close"]
        feats["ticker"] = t
        feats = feats.reset_index().rename(columns={"Date": "date"})
        panels.append(feats)

    print(f"[v4] skipped: {skipped_reason}")
    uni = pd.DataFrame(uni_rows).merge(sp[["ticker", "sector"]], on="ticker", how="left")
    uni = uni.sort_values("rv_60_now", ascending=False).reset_index(drop=True)
    uni.to_csv(DATA / "burst_universe_v4.csv", index=False)
    print(f"[v4] universe size: {len(uni)}")

    panel = pd.concat(panels, ignore_index=True)
    panel.to_csv(DATA / "burst_panel_v4.csv", index=False)
    print(f"[v4] panel rows: {len(panel):,}")
    lab_base = panel[panel["y"] >= 0]["y"].mean()
    print(f"[v4] burst base rate (labelled, panel-wide): {lab_base:.4%}")

    # train
    lab = panel[panel["y"] >= 0].dropna(subset=FEATS).reset_index(drop=True)
    dates = np.sort(lab["date"].unique())
    d1 = dates[int(0.70 * len(dates))]; d2 = dates[int(0.85 * len(dates))]
    tr = lab[lab["date"] < d1]; va = lab[(lab["date"] >= d1) & (lab["date"] < d2)]; te = lab[lab["date"] >= d2]
    print(f"[v4] split sizes train/val/test: {len(tr):,} / {len(va):,} / {len(te):,}")

    gbc = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=42)
    gbc.fit(tr[FEATS].values, tr["y"].values)

    p_tr = gbc.predict_proba(tr[FEATS].values)[:, 1]
    p_va = gbc.predict_proba(va[FEATS].values)[:, 1]
    p_te = gbc.predict_proba(te[FEATS].values)[:, 1]
    metrics = {
        "train": evaluate(tr["y"].values, p_tr),
        "val":   evaluate(va["y"].values, p_va),
        "test":  evaluate(te["y"].values, p_te),
        "feature_cols": FEATS,
        "n_features": len(FEATS),
    }
    (OUT / "burst_metrics_v4.json").write_text(json.dumps(metrics, indent=2))
    print(f"[v4] test: AUC={metrics['test']['auc']:.3f}  "
          f"AP_lift={metrics['test']['ap_lift']:.2f}x  "
          f"log_loss={metrics['test']['log_loss']:.4f}")

    joblib.dump({"gbc": gbc, "feats": FEATS}, MODELS / "burst_gbc_v4.joblib")

    # feature importance
    imp = pd.Series(gbc.feature_importances_, index=FEATS).sort_values(ascending=False)
    print(f"\n[v4] feature importance:")
    print(imp.to_string())

    # score today
    scored = panel.dropna(subset=FEATS).copy()
    latest = (scored.sort_values(["ticker", "date"]).groupby("ticker").tail(1).copy())
    latest["prob"] = gbc.predict_proba(latest[FEATS].values)[:, 1]
    latest["base_rate_test"] = metrics["test"]["base"]
    latest["lift"] = latest["prob"] / latest["base_rate_test"]

    # merge sector + rv_60_now
    latest = latest.merge(uni[["ticker", "adv_usd", "rv_60_now", "sector"]],
                           on="ticker", how="left")
    latest = latest.sort_values("prob", ascending=False).reset_index(drop=True)
    out_cols = ["ticker", "date", "close", "prob", "lift", "rv_60_now", "sector",
                "rsi_14", "macd", "bb_z20", "atr_pct", "vol_z", "vol_5d"]
    latest[out_cols].to_csv(OUT / "burst_today_v4.csv", index=False)
    print(f"\n[v4] scored {len(latest)} tickers -> output/burst_today_v4.csv")
    print("\n[v4] top 25:")
    print(latest[out_cols].head(25).to_string(index=False))

    # Where is SNDK?
    if (latest["ticker"] == "SNDK").any():
        sndk = latest[latest["ticker"] == "SNDK"].iloc[0]
        rank = int(latest.index[latest["ticker"] == "SNDK"][0]) + 1
        print(f"\n[SNDK] rank {rank}/{len(latest)}  prob={sndk['prob']:.3%}  "
              f"lift={sndk['lift']:.1f}x  rv_60_now={sndk['rv_60_now']:.3f}")
    else:
        print("\n[SNDK] NOT in scored output -- check universe filters")


if __name__ == "__main__":
    main()
