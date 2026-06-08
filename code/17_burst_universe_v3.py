"""Universe v3 — the 'calm → volatile → calm' pattern filter.

A ticker is included iff, within its history, we find at least one
**volatility episode** that:
  1. Was preceded by a CALM regime (baseline-like vol before start)
  2. Was followed by a CALM regime (vol returned to baseline-like levels)
     -- OR the episode is the most recent one and we're already past the
        "recency gate" (handled explicitly below)
  3. Ended within **6 × episode_length** trading days of today. (User rule.)

Definitions (computed per-ticker from its own history):
  rv20 = 20-day rolling annualized vol of daily returns.
  baseline = 30th percentile of rv20 over the stock's full available history
             (typical calm-regime vol for THIS stock).
  threshold = max(2.0 × baseline, 0.35)
             (elevated vol: at least double the stock's normal AND above 35% ann.)

  Episode = contiguous run of length >= MIN_EP_LEN where rv20 > threshold.

Calm checks (before / after):
  Look at a 20-day window adjacent to the episode; require MEDIAN(rv20) in that
  window to be <= 1.3 × baseline.

Also keep metrics for downstream filtering (price, ADV, current vs baseline vol).
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).resolve().parent.parent / "data"

UA = {"User-Agent": "Mozilla/5.0"}

PRICE_MIN = 40.0
ADV_MIN_USD = 25_000_000
BASELINE_PCTL = 30          # lower-tercile vol == "calm for this name"
THRESH_MULT = 2.0           # elevated vol threshold multiplier
THRESH_ABS_MIN = 0.35       # absolute floor on threshold
MIN_EP_LEN = 3              # episode must be >= 3 trading days
CALM_WINDOW = 20            # pre/post calm check window
CALM_MULT = 1.3             # calm == rv20 median <= baseline * CALM_MULT
RECENCY_MULT = 6            # episode must have ended within 6 x length ago


def fetch_sp500() -> pd.DataFrame:
    html = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=UA, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0][["Symbol", "Security", "GICS Sector"]].copy()
    t.columns = ["ticker", "name", "sector"]
    t["ticker"] = t["ticker"].str.replace(".", "-", regex=False)
    return t


def find_episodes(rv: pd.Series, thresh: float) -> list[tuple[int, int]]:
    above = (rv > thresh).fillna(False).values
    eps, in_ep, start = [], False, 0
    for i, a in enumerate(above):
        if a and not in_ep:
            start = i; in_ep = True
        elif not a and in_ep:
            if i - start >= MIN_EP_LEN:
                eps.append((start, i - 1))
            in_ep = False
    if in_ep and (len(above) - start) >= MIN_EP_LEN:
        eps.append((start, len(above) - 1))
    return eps


def is_calm(rv: pd.Series, i_start: int, i_end: int, baseline: float) -> bool:
    if i_start < 0 or i_end >= len(rv) or i_end <= i_start:
        return False
    w = rv.iloc[i_start:i_end].dropna()
    if len(w) < CALM_WINDOW // 2:
        return False
    return float(w.median()) <= baseline * CALM_MULT


def analyze(sub: pd.DataFrame):
    """Return dict of metrics + qualifying-episode info, or None if unusable."""
    sub = sub.dropna()
    if len(sub) < 350:
        return None
    c = sub["Close"]; v = sub["Volume"]
    r = c.pct_change()
    rv20 = r.rolling(20).std() * math.sqrt(252)
    rv_full = rv20.dropna()
    if len(rv_full) < 120:
        return None

    baseline = float(np.nanpercentile(rv_full, BASELINE_PCTL))
    thresh = max(THRESH_MULT * baseline, THRESH_ABS_MIN)
    eps = find_episodes(rv20, thresh)

    last_idx = len(rv20) - 1
    qualifying = []
    for (s, e) in eps:
        length = e - s + 1
        # recency gate (user rule): end within 6x length of today
        if (last_idx - e) > RECENCY_MULT * length:
            continue
        before_ok = is_calm(rv20, max(0, s - CALM_WINDOW), s, baseline)
        after_ok = is_calm(rv20, e + 1, min(last_idx + 1, e + 1 + CALM_WINDOW), baseline)
        # if episode is the current tail (no "after" yet), treat after as OK
        if e >= last_idx - 2:
            after_ok = True
        if before_ok and after_ok:
            qualifying.append({
                "start_idx": s, "end_idx": e, "length": length,
                "start_date": sub.index[s].date().isoformat(),
                "end_date": sub.index[e].date().isoformat(),
                "days_since_end": last_idx - e,
                "peak_rv": float(rv20.iloc[s:e+1].max()),
            })

    if not qualifying:
        return None

    rv60 = r.rolling(60).std() * math.sqrt(252)
    return {
        "price": float(c.iloc[-1]),
        "adv_usd": float((c * v).tail(30).mean()),
        "baseline_rv20": baseline,
        "episode_threshold": thresh,
        "n_episodes_total": len(eps),
        "n_episodes_qualifying": len(qualifying),
        "most_recent_ep": qualifying[-1],
        "rv_60_now": float(rv60.iloc[-1]) if not np.isnan(rv60.iloc[-1]) else float("nan"),
    }


def main():
    sp = fetch_sp500()
    print(f"[v3] SP500: {len(sp)}")
    print("[v3] bulk downloading 3y daily ...")
    px = yf.download(sp["ticker"].tolist() + ["SPY"], period="3y", interval="1d",
                     group_by="ticker", auto_adjust=True, threads=True, progress=False)

    rows = []
    for t in sp["ticker"]:
        try:
            sub = px[t]
        except KeyError:
            continue
        r = analyze(sub)
        if r is None:
            continue
        r["ticker"] = t
        rows.append(r)
    met = pd.DataFrame(rows)
    if len(met) == 0:
        print("[v3] no qualifying tickers"); return
    met = met.merge(sp[["ticker", "sector"]], on="ticker", how="left")

    flt = met[(met["price"] >= PRICE_MIN)
              & (met["adv_usd"] >= ADV_MIN_USD)].copy()
    flt = flt.sort_values("n_episodes_qualifying", ascending=False).reset_index(drop=True)

    out = DATA / "burst_universe_v3.csv"
    # flatten most_recent_ep for csv
    flt["mre_start"] = flt["most_recent_ep"].apply(lambda d: d["start_date"])
    flt["mre_end"] = flt["most_recent_ep"].apply(lambda d: d["end_date"])
    flt["mre_length"] = flt["most_recent_ep"].apply(lambda d: d["length"])
    flt["mre_days_since"] = flt["most_recent_ep"].apply(lambda d: d["days_since_end"])
    flt["mre_peak_rv"] = flt["most_recent_ep"].apply(lambda d: d["peak_rv"])
    flt = flt.drop(columns=["most_recent_ep"])
    flt.to_csv(out, index=False)
    print(f"[v3] kept {len(flt)} tickers with qualifying calm->vol->calm episode")
    print("[v3] top 20 (most episodes):")
    print(flt[["ticker", "price", "baseline_rv20", "rv_60_now",
               "n_episodes_qualifying", "mre_start", "mre_length",
               "mre_days_since", "mre_peak_rv", "sector"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
