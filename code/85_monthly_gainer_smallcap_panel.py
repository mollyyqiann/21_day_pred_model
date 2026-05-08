"""Phase E.2: Build smallcap panel matching v8 features + monthly_gainer labels.

Downloads ~1388 non-S&P-500 tickers via yfinance bulk, computes v8 features
(rsi, macd, bb, atr, vol_z, vol_5d, rv_60, ma_stack, up_streak, up_bigdays_20d,
dist_ma60_atr, ma60_slope_60d, run_length), then computes y21 / max_fwd21_ret /
argmax_day_fwd21 / end_of_window_ret / y5_touch.

Output: data/monthly_gainer_panel_smallcap.csv

Notes:
  - overnight_gap is NOT computed (it's forward-looking and excluded from
    the model anyway).
  - Survivorship bias: rh universe is *current* tickers only.
  - Some tickers will fail download — recorded in the success summary.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MIN_OBS = 120
WINDOW_21 = 21
WINDOW_5 = 5
TOUCH_THRESH = 0.30


def rsi(x, n=14):
    d = x.diff(); u = d.clip(lower=0).rolling(n).mean(); dd = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + u / dd.replace(0, np.nan))


def macd_(x, f=12, s=26, sig=9):
    ef = x.ewm(span=f, adjust=False).mean()
    es = x.ewm(span=s, adjust=False).mean()
    line = ef - es
    signal = line.ewm(span=sig, adjust=False).mean()
    return line / x, signal / x, (line - signal) / x


def bb_z(x, n=20):
    return (x - x.rolling(n).mean()) / x.rolling(n).std()


def atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def build_v8_features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame with v8 features + close. Date is index."""
    out = pd.DataFrame(index=df.index)
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    out["rsi_14"] = rsi(c)
    ml, ms, mh = macd_(c)
    out["macd"], out["macd_sig"], out["macd_hist"] = ml, ms, mh
    out["bb_z20"] = bb_z(c)
    out["atr_pct"] = atr(h, l, c) / c
    out["range_pct"] = (h - l) / c
    vm = v.rolling(30).mean()
    vs = v.rolling(30).std().replace(0, np.nan)
    out["vol_z"] = (v - vm) / vs
    out["vol_5d"] = v.rolling(5).mean() / vm
    r = c.pct_change()
    out["rv_60"] = r.rolling(60).std() * math.sqrt(252)
    # trend features (v8)
    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    atr_dollars = out["atr_pct"] * c
    out["ma_stack"] = ((ma5 > ma20) & (ma20 > ma60)).astype(int)
    up = (r > 0).astype(int)
    grp = (up != up.shift()).cumsum()
    streak = up.groupby(grp).cumsum().where(up == 1, 0)
    out["up_streak"] = streak.clip(upper=30)
    out["up_bigdays_20d"] = (r > 0.03).rolling(20).sum()
    out["dist_ma60_atr"] = (c - ma60) / atr_dollars.replace(0, np.nan)
    out["ma60_slope_60d"] = (ma60 - ma60.shift(60)) / c
    above20 = (c > ma20).astype(int)
    grp2 = (above20 != above20.shift()).cumsum()
    run = above20.groupby(grp2).cumsum().where(above20 == 1, 0)
    out["run_length"] = run.clip(upper=120)
    out["close"] = c
    return out


def add_forward_labels(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    c = g["close"].values.astype(float)
    n = len(c)
    y21 = np.full(n, -1, dtype=np.int8)
    max_ret = np.full(n, np.nan)
    argmax_day = np.full(n, -1, dtype=np.int16)
    end_ret = np.full(n, np.nan)
    y5 = np.full(n, -1, dtype=np.int8)
    for t in range(n - 1):
        c_t = c[t]
        if c_t <= 0 or not np.isfinite(c_t):
            continue
        if t + WINDOW_21 < n:
            fut = c[t + 1: t + 1 + WINDOW_21]
            ret = fut / c_t - 1.0
            mx = float(np.max(ret))
            max_ret[t] = mx
            argmax_day[t] = int(np.argmax(ret) + 1)
            end_ret[t] = float(c[t + WINDOW_21] / c_t - 1.0)
            y21[t] = 1 if mx >= TOUCH_THRESH else 0
        if t + WINDOW_5 < n:
            fut5 = c[t + 1: t + 1 + WINDOW_5]
            mx5 = float(np.max(fut5 / c_t - 1.0))
            y5[t] = 1 if mx5 >= TOUCH_THRESH else 0
    g["y21"] = y21
    g["max_fwd21_ret"] = max_ret
    g["argmax_day_fwd21"] = argmax_day
    g["end_of_window_ret"] = end_ret
    g["y5_touch"] = y5
    return g


def main():
    t0 = time.time()
    universe = pd.read_csv(DATA / "monthly_gainer_universe_smallcap.csv")
    tickers = universe["ticker"].tolist()
    print(f"[85] downloading 3y daily for {len(tickers)} smallcap tickers ...")

    raw = yf.download(tickers, period="3y", interval="1d",
                      group_by="ticker", auto_adjust=True, threads=True, progress=False)
    print(f"[85] download done in {time.time() - t0:.1f}s")

    sector_map = dict(zip(universe["ticker"], universe["sector"]))

    panels = []
    n_ok = 0
    n_skip_obs = 0
    n_skip_err = 0
    for tk in tickers:
        try:
            sub = raw[tk].dropna()
        except Exception:
            n_skip_err += 1
            continue
        if len(sub) < MIN_OBS:
            n_skip_obs += 1
            continue
        feats = build_v8_features(sub)
        feats = feats.reset_index().rename(columns={"Date": "date"})
        feats["ticker"] = tk
        feats["sector"] = sector_map.get(tk)
        feats = add_forward_labels(feats)
        panels.append(feats)
        n_ok += 1
    print(f"[85] {n_ok} OK / {n_skip_obs} skipped (low obs) / {n_skip_err} download errors")

    panel = pd.concat(panels, ignore_index=True)
    out_path = DATA / "monthly_gainer_panel_smallcap.csv"
    panel.to_csv(out_path, index=False)

    lab = panel[panel["y21"] >= 0]
    print(f"[85] panel rows: {len(panel):,}  labeled: {len(lab):,}  "
          f"positives: {int(lab['y21'].sum()):,}  base_rate: {lab['y21'].mean():.4%}")
    lab5 = panel[panel["y5_touch"] >= 0]
    print(f"[85] 5d-touch base rate: {lab5['y5_touch'].mean():.4%} "
          f"(positives: {int(lab5['y5_touch'].sum()):,})")
    print(f"[85] wrote {out_path}")
    print(f"[85] total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
