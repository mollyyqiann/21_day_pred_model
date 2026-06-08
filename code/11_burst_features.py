"""Download 3-year daily history for each universe ticker + SPY, compute features,
and build burst targets.

Burst target (binary): at date t, within the NEXT 5 trading days [t+1 .. t+5],
does there exist a contiguous window of length L in {2,3,4,5} whose average daily
return is >= 4%? That matches the user's definition of "averaging more than
4%/day for longer than 2 days".

Outputs:
  data/burst_panel.parquet  (long-format) or .csv fallback
  data/burst_meta.json
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent / "data"

PERIOD = "3y"
BURST_WINDOW = 5           # look in next 5 days
BURST_MIN_LEN = 2          # at least 2 days
BURST_THRESH = 0.04        # 4% per day average


# ---------- feature helpers ----------

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal = line.ewm(span=sig, adjust=False).mean()
    hist = line - signal
    return line / series, signal / series, hist / series  # normalized


def bb_z(series: pd.Series, n: int = 20) -> pd.Series:
    m = series.rolling(n).mean()
    s = series.rolling(n).std()
    return (series - m) / s


def atr(high, low, close, n: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def build_features(df: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """df has columns: Open High Low Close Volume (DatetimeIndex). Returns feature frame."""
    out = pd.DataFrame(index=df.index)
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    out["ret_1d"] = c.pct_change()
    out["ret_5d"] = c.pct_change(5)
    out["ret_10d"] = c.pct_change(10)
    out["ret_20d"] = c.pct_change(20)
    out["ret_60d"] = c.pct_change(60)

    out["rv_10"] = out["ret_1d"].rolling(10).std() * math.sqrt(252)
    out["rv_20"] = out["ret_1d"].rolling(20).std() * math.sqrt(252)
    out["rv_60"] = out["ret_1d"].rolling(60).std() * math.sqrt(252)

    out["rsi_14"] = rsi(c, 14)
    m_line, m_sig, m_hist = macd(c)
    out["macd"] = m_line
    out["macd_sig"] = m_sig
    out["macd_hist"] = m_hist
    out["bb_z20"] = bb_z(c, 20)

    out["atr_pct"] = atr(h, l, c, 14) / c
    out["range_pct"] = (h - l) / c

    # volume features
    vol_m = v.rolling(30).mean()
    vol_s = v.rolling(30).std().replace(0, np.nan)
    out["vol_z"] = (v - vol_m) / vol_s
    out["vol_5d"] = v.rolling(5).mean() / vol_m

    # moving average gaps
    ma50 = c.rolling(50).mean()
    ma200 = c.rolling(200).mean()
    out["gap_ma50"] = c / ma50 - 1
    out["gap_ma200"] = c / ma200 - 1

    # 52-week position
    hi52 = c.rolling(252).max()
    lo52 = c.rolling(252).min()
    out["pos_52w"] = (c - lo52) / (hi52 - lo52)

    # beta residuals vs SPY
    spy_c = spy["Close"].reindex(c.index).ffill()
    spy_r = spy_c.pct_change()
    s_ret = out["ret_1d"]
    # rolling 60-day beta
    cov = s_ret.rolling(60).cov(spy_r)
    var = spy_r.rolling(60).var()
    beta = cov / var.replace(0, np.nan)
    resid = s_ret - beta * spy_r
    out["beta_60"] = beta
    out["resid_1d"] = resid
    out["resid_5d"] = resid.rolling(5).sum()
    out["resid_10d"] = resid.rolling(10).sum()
    out["resid_20d"] = resid.rolling(20).sum()

    # SPY regime features (same for all tickers, but cheap to include)
    out["spy_ret_5d"] = spy_c.pct_change(5)
    out["spy_ret_20d"] = spy_c.pct_change(20)
    out["spy_rv_20"] = spy_r.rolling(20).std() * math.sqrt(252)
    return out


def build_burst_target(c: pd.Series) -> pd.Series:
    """For each date t, does max-L-avg of future 1d returns in [t+1..t+5] meet the burst bar?"""
    r = c.pct_change().fillna(0).values  # daily return vector; r[i] is return FROM i-1 to i
    n = len(r)
    y = np.zeros(n, dtype=np.int8)
    for t in range(n - BURST_WINDOW):
        # future returns indices (t+1 .. t+BURST_WINDOW)
        fut = r[t + 1 : t + 1 + BURST_WINDOW]
        best = 0.0
        for L in range(BURST_MIN_LEN, BURST_WINDOW + 1):
            for start in range(0, BURST_WINDOW - L + 1):
                avg = fut[start : start + L].mean()
                if avg > best:
                    best = avg
        if best >= BURST_THRESH:
            y[t] = 1
    s = pd.Series(y, index=c.index)
    # last BURST_WINDOW rows are "live" (unknown target)
    s.iloc[-BURST_WINDOW:] = -1
    return s


def main() -> None:
    uni = pd.read_csv(DATA / "burst_universe.csv")
    tickers = uni["ticker"].tolist()
    print(f"[features] fetching {PERIOD} daily history for {len(tickers)} tickers + SPY ...")

    raw = yf.download(
        tickers + ["SPY"],
        period=PERIOD,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    spy = raw["SPY"].dropna()

    panels = []
    skipped = 0
    for i, t in enumerate(tickers, 1):
        try:
            sub = raw[t].dropna()
        except KeyError:
            skipped += 1
            continue
        if len(sub) < 260:
            skipped += 1
            continue
        feats = build_features(sub, spy)
        y = build_burst_target(sub["Close"])
        feats["y"] = y
        feats["close"] = sub["Close"]
        feats["ticker"] = t
        feats = feats.reset_index().rename(columns={"Date": "date", "index": "date"})
        panels.append(feats)
        if i % 25 == 0:
            print(f"[features]   {i}/{len(tickers)}")

    panel = pd.concat(panels, ignore_index=True)
    print(f"[features] total rows: {len(panel)}  (skipped {skipped})")
    print(f"[features] positive burst rate (labelled rows): "
          f"{(panel[panel['y'] >= 0]['y']).mean():.4%}")

    # save (csv; parquet optional but keep dependencies minimal)
    out = DATA / "burst_panel.csv"
    panel.to_csv(out, index=False)
    print(f"[features] wrote {out}  ({len(panel)} rows, {panel.shape[1]} cols)")

    meta = {
        "tickers": sorted(panel["ticker"].unique().tolist()),
        "burst_window": BURST_WINDOW,
        "burst_min_len": BURST_MIN_LEN,
        "burst_thresh": BURST_THRESH,
        "n_rows": int(len(panel)),
        "feature_cols": [c for c in panel.columns
                         if c not in {"date", "ticker", "y", "close"}],
    }
    (DATA / "burst_meta.json").write_text(json.dumps(meta, indent=2))
    print("[features] done")


if __name__ == "__main__":
    main()
