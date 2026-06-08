"""V5 — build panel for the upside-asymmetric universe, train, and score.

Feature set = the winning A_base (9 features) + rv_60 (v4's 10th) + 3 NEW
dynamic asymmetry features computed per-date on a 60-day rolling window:

  skew_60d            - rolling 60d skewness of daily returns
  semivol_ratio_60d   - rolling 60d upside-semivol / downside-semivol
  up_bigdays_60d      - count of days with ret > +3% in last 60 days

Rationale: the v5 universe filter selects stocks with upside-asymmetric return
distributions over the full 3y window. The 3 new features let the model
recognize *when* that asymmetry is currently showing up in the recent window,
not just on a lifetime basis.

Stages: panel -> chronological split -> GBC -> feature importance -> score today.

Outputs:
  data/burst_panel_v5.csv
  models/burst_gbc_v5.joblib
  output/burst_metrics_v5.json
  output/burst_today_v5.csv
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

BURST_WINDOW = 5; BURST_MIN_LEN = 2; BURST_THRESH = 0.04

A_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
          "atr_pct", "range_pct", "vol_z", "vol_5d"]
ASYM = ["skew_60d", "semivol_ratio_60d", "up_bigdays_60d"]
FEATS = A_BASE + ["rv_60"] + ASYM    # 13 features


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


def rolling_semivol_ratio(r: pd.Series, n: int = 60) -> pd.Series:
    up_sq = (r.clip(lower=0) ** 2).rolling(n).mean()
    dn_sq = (r.clip(upper=0) ** 2).rolling(n).mean()
    return np.sqrt(up_sq) / np.sqrt(dn_sq.replace(0, np.nan))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
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
    # asymmetry features (dynamic, rolling)
    out["skew_60d"] = r.rolling(60).skew()
    out["semivol_ratio_60d"] = rolling_semivol_ratio(r, 60)
    out["up_bigdays_60d"] = (r > 0.03).rolling(60).sum()
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
    uni = pd.read_csv(DATA / "burst_universe_v5.csv")
    tickers = uni["ticker"].tolist()
    print(f"[v5] universe: {len(tickers)} upside-asymmetric SP500 names")

    print("[v5] bulk downloading 3y daily ...")
    raw = yf.download(tickers, period="3y", interval="1d",
                      group_by="ticker", auto_adjust=True, threads=True, progress=False)

    panels = []
    for t in tickers:
        try:
            sub = raw[t].dropna()
        except KeyError:
            continue
        if len(sub) < 120:
            continue
        feats = build_features(sub)
        feats["y"] = build_target(sub["Close"])
        feats["close"] = sub["Close"]
        feats["ticker"] = t
        feats = feats.reset_index().rename(columns={"Date": "date"})
        panels.append(feats)

    panel = pd.concat(panels, ignore_index=True)
    panel.to_csv(DATA / "burst_panel_v5.csv", index=False)
    print(f"[v5] panel rows: {len(panel):,}")
    print(f"[v5] burst base rate: {panel[panel['y']>=0]['y'].mean():.4%}")

    lab = panel[panel["y"] >= 0].dropna(subset=FEATS).reset_index(drop=True)
    print(f"[v5] labelled rows post-NA: {len(lab):,}")
    dates = np.sort(lab["date"].unique())
    d1 = dates[int(0.70 * len(dates))]; d2 = dates[int(0.85 * len(dates))]
    tr = lab[lab["date"] < d1]; va = lab[(lab["date"] >= d1) & (lab["date"] < d2)]; te = lab[lab["date"] >= d2]
    print(f"[v5] train/val/test: {len(tr):,} / {len(va):,} / {len(te):,}")

    gbc = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=42)
    gbc.fit(tr[FEATS].values, tr["y"].values)
    p_te = gbc.predict_proba(te[FEATS].values)[:, 1]
    p_va = gbc.predict_proba(va[FEATS].values)[:, 1]
    p_tr = gbc.predict_proba(tr[FEATS].values)[:, 1]
    metrics = {"train": evaluate(tr["y"].values, p_tr),
               "val":   evaluate(va["y"].values, p_va),
               "test":  evaluate(te["y"].values, p_te),
               "feature_cols": FEATS, "n_features": len(FEATS)}
    (OUT / "burst_metrics_v5.json").write_text(json.dumps(metrics, indent=2))
    print(f"[v5] test: AUC={metrics['test']['auc']:.3f}  AP_lift={metrics['test']['ap_lift']:.2f}x  "
          f"log_loss={metrics['test']['log_loss']:.4f}")

    joblib.dump({"gbc": gbc, "feats": FEATS}, MODELS / "burst_gbc_v5.joblib")

    # feature importance
    imp = pd.Series(gbc.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("\n[v5] feature importance:")
    print(imp.to_string())

    # score today
    scored = panel.dropna(subset=FEATS).copy()
    latest = scored.sort_values(["ticker", "date"]).groupby("ticker").tail(1).copy()
    latest["prob"] = gbc.predict_proba(latest[FEATS].values)[:, 1]
    base = metrics["test"]["base"]
    latest["lift"] = latest["prob"] / base
    latest = latest.merge(uni[["ticker", "skew", "semivol_ratio",
                                "run_ratio", "big_ratio", "reasons", "sector"]],
                          on="ticker", how="left")
    latest = latest.sort_values("prob", ascending=False).reset_index(drop=True)
    out_cols = ["ticker", "date", "close", "prob", "lift",
                "rv_60", "rsi_14", "bb_z20", "skew_60d", "semivol_ratio_60d",
                "up_bigdays_60d", "skew", "run_ratio", "sector", "reasons"]
    latest[out_cols].to_csv(OUT / "burst_today_v5.csv", index=False)

    print(f"\n[v5] scored {len(latest)} tickers -> output/burst_today_v5.csv")
    print("\n[v5] top 25:")
    print(latest[out_cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
