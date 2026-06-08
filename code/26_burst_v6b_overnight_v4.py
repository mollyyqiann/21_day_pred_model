"""V6b — overnight-information feature applied to the v4 (>$40) universe.

Same protocol as v6:
  vanilla:    v4 features (10)           = A_base + rv_60
  augmented:  v4 + overnight_gap (11)

Accept only if augmented strictly beats vanilla on AUC AND AP.

Outputs:
  data/burst_panel_v6b.csv
  models/burst_gbc_v6b_vanilla.joblib
  models/burst_gbc_v6b_augmented.joblib
  output/burst_metrics_v6b.json
  output/burst_today_v6b.csv
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

PRICE_MIN = 40.0
ADV_MIN_USD = 25_000_000
MIN_OBS = 150

BURST_WINDOW = 5; BURST_MIN_LEN = 2; BURST_THRESH = 0.04

A_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
          "atr_pct", "range_pct", "vol_z", "vol_5d"]
V4_FEATS = A_BASE + ["rv_60"]              # 10
V6B_FEATS = V4_FEATS + ["overnight_gap"]   # 11


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


def train_and_eval(feats, tr, va, te):
    gbc = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=42)
    gbc.fit(tr[feats].values, tr["y"].values)
    return gbc, {"train": evaluate(tr["y"].values, gbc.predict_proba(tr[feats].values)[:, 1]),
                 "val":   evaluate(va["y"].values, gbc.predict_proba(va[feats].values)[:, 1]),
                 "test":  evaluate(te["y"].values, gbc.predict_proba(te[feats].values)[:, 1])}


def main():
    sp = fetch_sp500()
    print(f"[v6b] {len(sp)} SP500 constituents")

    print("[v6b] bulk downloading 3y daily ...")
    raw = yf.download(sp["ticker"].tolist(), period="3y", interval="1d",
                      group_by="ticker", auto_adjust=True, threads=True, progress=False)

    panels = []; kept = []
    for t in sp["ticker"]:
        try: sub = raw[t].dropna()
        except KeyError: continue
        if len(sub) < MIN_OBS: continue
        price = float(sub["Close"].iloc[-1])
        if price < PRICE_MIN: continue
        adv = float((sub["Close"] * sub["Volume"]).tail(30).mean())
        if adv < ADV_MIN_USD: continue
        feats = build_features(sub)
        feats["y"] = build_target(sub["Close"])
        feats["close"] = sub["Close"]
        feats["ticker"] = t
        feats = feats.reset_index().rename(columns={"Date": "date"})
        panels.append(feats)
        kept.append({"ticker": t, "price": price, "adv_usd": adv})

    uni = pd.DataFrame(kept).merge(sp[["ticker", "sector"]], on="ticker", how="left")
    panel = pd.concat(panels, ignore_index=True)
    panel.to_csv(DATA / "burst_panel_v6b.csv", index=False)
    print(f"[v6b] universe: {len(uni)}  panel rows: {len(panel):,}")

    lab = panel[panel["y"] >= 0].dropna(subset=V6B_FEATS).reset_index(drop=True)
    print(f"[v6b] labelled rows post-NA (needs overnight_gap): {len(lab):,}  "
          f"base rate: {lab['y'].mean():.4%}")
    dates = np.sort(lab["date"].unique())
    d1 = dates[int(0.70 * len(dates))]; d2 = dates[int(0.85 * len(dates))]
    tr = lab[lab["date"] < d1]; va = lab[(lab["date"] >= d1) & (lab["date"] < d2)]; te = lab[lab["date"] >= d2]
    print(f"[v6b] train/val/test: {len(tr):,} / {len(va):,} / {len(te):,}")

    print("\n[v6b] === VANILLA (v4 features, 10) ===")
    gbc_v, met_v = train_and_eval(V4_FEATS, tr, va, te)
    print(f"    test: AUC={met_v['test']['auc']:.3f}  AP={met_v['test']['ap']:.3f}  "
          f"lift={met_v['test']['ap_lift']:.2f}x  log_loss={met_v['test']['log_loss']:.4f}")

    print("\n[v6b] === AUGMENTED (v4 + overnight_gap, 11) ===")
    gbc_a, met_a = train_and_eval(V6B_FEATS, tr, va, te)
    print(f"    test: AUC={met_a['test']['auc']:.3f}  AP={met_a['test']['ap']:.3f}  "
          f"lift={met_a['test']['ap_lift']:.2f}x  log_loss={met_a['test']['log_loss']:.4f}")

    auc_gain = met_a["test"]["auc"] - met_v["test"]["auc"]
    ap_gain = met_a["test"]["ap"] - met_v["test"]["ap"]
    ll_gain = met_v["test"]["log_loss"] - met_a["test"]["log_loss"]
    passes = auc_gain > 0 and ap_gain > 0
    print(f"\n[v6b] delta: AUC={auc_gain:+.4f}  AP={ap_gain:+.4f}  "
          f"log_loss_reduction={ll_gain:+.4f}")
    print(f"[v6b] verdict: {'ACCEPT' if passes else 'REJECT'} "
          f"(augmented {'beats' if passes else 'did NOT beat'} vanilla on both AUC and AP)")

    joblib.dump({"gbc": gbc_v, "feats": V4_FEATS}, MODELS / "burst_gbc_v6b_vanilla.joblib")
    joblib.dump({"gbc": gbc_a, "feats": V6B_FEATS}, MODELS / "burst_gbc_v6b_augmented.joblib")
    json.dump({
        "vanilla": met_v, "augmented": met_a,
        "auc_gain": auc_gain, "ap_gain": ap_gain, "log_loss_reduction": ll_gain,
        "passes_vs_vanilla": bool(passes),
        "vanilla_feats": V4_FEATS, "augmented_feats": V6B_FEATS,
    }, open(OUT / "burst_metrics_v6b.json", "w"), indent=2)

    print("\n[v6b] augmented feature importance:")
    imp = pd.Series(gbc_a.feature_importances_, index=V6B_FEATS).sort_values(ascending=False)
    print(imp.to_string())

    # Score today
    scored = panel.dropna(subset=V4_FEATS).copy()
    latest = scored.sort_values(["ticker", "date"]).groupby("ticker").tail(1).copy()
    latest["prob_vanilla"] = gbc_v.predict_proba(latest[V4_FEATS].values)[:, 1]
    aug = latest.copy(); aug["overnight_gap"] = aug["overnight_gap"].fillna(0.0)
    latest["prob_augmented_zero_gap"] = gbc_a.predict_proba(aug[V6B_FEATS].values)[:, 1]
    sens = latest.copy(); sens["overnight_gap"] = 0.03
    latest["prob_augmented_gap+3pct"] = gbc_a.predict_proba(sens[V6B_FEATS].values)[:, 1]
    latest = latest.merge(uni[["ticker", "sector"]], on="ticker", how="left")
    latest = latest.sort_values("prob_augmented_zero_gap", ascending=False).reset_index(drop=True)
    cols = ["ticker", "date", "close", "prob_vanilla", "prob_augmented_zero_gap",
            "prob_augmented_gap+3pct", "rv_60", "rsi_14", "bb_z20", "atr_pct", "sector"]
    latest[cols].to_csv(OUT / "burst_today_v6b.csv", index=False)
    print(f"\n[v6b] wrote output/burst_today_v6b.csv ({len(latest)} rows)")
    print("\n[v6b] top 20:")
    print(latest[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
