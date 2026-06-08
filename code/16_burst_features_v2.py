"""Panel builder v2: same burst target as v1, new feature set that explicitly
encodes regime shift (recent vol vs long-term vol, max recent move, big-day counts).

Outputs:
  data/burst_panel_v2.csv
  data/burst_meta_v2.json
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

PERIOD = "3y"
BURST_WINDOW = 5
BURST_MIN_LEN = 2
BURST_THRESH = 0.04


# ---- same technical helpers ----

def rsi(x, n=14):
    d = x.diff(); u = d.clip(lower=0).rolling(n).mean(); dd = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + u / dd.replace(0, np.nan))


def macd(x, f=12, s=26, sig=9):
    ef = x.ewm(span=f, adjust=False).mean()
    es = x.ewm(span=s, adjust=False).mean()
    line = ef - es; signal = line.ewm(span=sig, adjust=False).mean()
    return line / x, signal / x, (line - signal) / x


def bb_z(x, n=20):
    return (x - x.rolling(n).mean()) / x.rolling(n).std()


def atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def build_features(df, spy, hist_rv_median):
    out = pd.DataFrame(index=df.index)
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    out["ret_1d"] = c.pct_change()
    out["ret_5d"] = c.pct_change(5)
    out["ret_10d"] = c.pct_change(10)
    out["ret_20d"] = c.pct_change(20)
    out["ret_60d"] = c.pct_change(60)

    r = out["ret_1d"]
    out["rv_10"] = r.rolling(10).std() * math.sqrt(252)
    out["rv_20"] = r.rolling(20).std() * math.sqrt(252)
    out["rv_60"] = r.rolling(60).std() * math.sqrt(252)
    out["rv_120"] = r.rolling(120).std() * math.sqrt(252)

    # ---- regime shift features (new) ----
    out["rv_ratio_20_60"] = out["rv_20"] / out["rv_60"]
    out["rv_ratio_60_120"] = out["rv_60"] / out["rv_120"]
    out["rv_ratio_vs_hist"] = out["rv_60"] / hist_rv_median   # vs the pre-period baseline
    out["max_ret_20d"] = r.rolling(20).max()
    out["max_ret_60d"] = r.rolling(60).max()
    out["num_3pct_days_60d"] = (r.abs() >= 0.03).rolling(60).sum()
    out["num_5pct_days_60d"] = (r.abs() >= 0.05).rolling(60).sum()
    out["max_3d_avg_60d"] = r.rolling(3).mean().rolling(60).max()

    # ---- classical indicators ----
    out["rsi_14"] = rsi(c, 14)
    ml, ms, mh = macd(c)
    out["macd"] = ml; out["macd_sig"] = ms; out["macd_hist"] = mh
    out["bb_z20"] = bb_z(c, 20)
    out["atr_pct"] = atr(h, l, c, 14) / c
    out["range_pct"] = (h - l) / c

    vm = v.rolling(30).mean(); vs = v.rolling(30).std().replace(0, np.nan)
    out["vol_z"] = (v - vm) / vs
    out["vol_5d"] = v.rolling(5).mean() / vm

    ma50 = c.rolling(50).mean(); ma200 = c.rolling(200).mean()
    out["gap_ma50"] = c / ma50 - 1
    out["gap_ma200"] = c / ma200 - 1
    hi52 = c.rolling(252).max(); lo52 = c.rolling(252).min()
    out["pos_52w"] = (c - lo52) / (hi52 - lo52)
    out["pct_from_52w_high"] = c / hi52 - 1

    # ---- beta / residuals vs SPY ----
    sc = spy["Close"].reindex(c.index).ffill()
    sr = sc.pct_change()
    cov = r.rolling(60).cov(sr); var = sr.rolling(60).var()
    beta = cov / var.replace(0, np.nan)
    resid = r - beta * sr
    out["beta_60"] = beta
    out["resid_1d"] = resid
    out["resid_5d"] = resid.rolling(5).sum()
    out["resid_20d"] = resid.rolling(20).sum()

    # ---- SPY regime ----
    out["spy_ret_5d"] = sc.pct_change(5)
    out["spy_ret_20d"] = sc.pct_change(20)
    out["spy_rv_20"] = sr.rolling(20).std() * math.sqrt(252)
    return out


def build_target(c):
    r = c.pct_change().fillna(0).values
    n = len(r); y = np.zeros(n, dtype=np.int8)
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


def main() -> None:
    uni = pd.read_csv(DATA / "burst_universe_v2.csv")
    tickers = uni["ticker"].tolist()
    hist_rv_by_tkr = dict(zip(uni["ticker"], uni["hist_rv_median"]))
    print(f"[v2-features] downloading {PERIOD} history for {len(tickers)} tickers + SPY ...")
    raw = yf.download(tickers + ["SPY"], period=PERIOD, interval="1d",
                      group_by="ticker", auto_adjust=True, threads=True, progress=False)
    spy = raw["SPY"].dropna()

    panels = []; skipped = 0
    for i, t in enumerate(tickers, 1):
        try:
            sub = raw[t].dropna()
        except KeyError:
            skipped += 1; continue
        if len(sub) < 400:
            skipped += 1; continue
        feats = build_features(sub, spy, hist_rv_by_tkr[t])
        feats["y"] = build_target(sub["Close"])
        feats["close"] = sub["Close"]
        feats["ticker"] = t
        feats = feats.reset_index().rename(columns={"Date": "date"})
        panels.append(feats)
        if i % 50 == 0:
            print(f"[v2-features]   {i}/{len(tickers)}")

    panel = pd.concat(panels, ignore_index=True)
    print(f"[v2-features] rows: {len(panel)}  (skipped {skipped})")
    print(f"[v2-features] burst base rate (labelled): {panel[panel['y']>=0]['y'].mean():.4%}")

    panel.to_csv(DATA / "burst_panel_v2.csv", index=False)
    meta = {
        "tickers": sorted(panel["ticker"].unique().tolist()),
        "burst_window": BURST_WINDOW,
        "burst_min_len": BURST_MIN_LEN,
        "burst_thresh": BURST_THRESH,
        "n_rows": int(len(panel)),
        "feature_cols": [c for c in panel.columns
                         if c not in {"date", "ticker", "y", "close"}],
    }
    (DATA / "burst_meta_v2.json").write_text(json.dumps(meta, indent=2))
    print(f"[v2-features] wrote burst_panel_v2.csv and burst_meta_v2.json")


if __name__ == "__main__":
    main()
