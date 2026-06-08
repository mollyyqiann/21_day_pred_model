"""Universe v2: stocks that were HISTORICALLY calm, regardless of current state.

Motivation: the v1 filter excluded any stock whose current 60d vol > 30%, which
meant SNDK-style names (calm for years, then enter a sustained ramp) dropped
out the moment they started running. We now key the filter off a LAGGED
historical vol window so we keep names that *used to be* calm.

Filter:
  - S&P 500 constituent
  - Price >= $40 today
  - `hist_rv` <= 30%: median of rolling 60d annualized vol, computed over the
    window ending ~90 days before today (i.e. the quiet-baseline period)
  - 30d ADV >= $25M

Also keeps, for each name:
  - `rv_60_now`: current 60d annualized vol
  - `rv_ratio`: rv_60_now / hist_rv  -- the regime-shift signal
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
HIST_RV_MAX = 0.30       # was calm historically
HIST_RV_MIN = 0.05
ADV_MIN_USD = 25_000_000
LAG_DAYS = 90            # how far back to START the "calm baseline" window
HIST_LOOKBACK = 2 * 252  # use ~2 years ending at (today - LAG_DAYS)


def fetch_sp500() -> pd.DataFrame:
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers=UA, timeout=30).text
    df = pd.read_html(io.StringIO(html))[0][["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["ticker", "name", "sector"]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df


def compute_row(sub: pd.DataFrame) -> dict | None:
    sub = sub.dropna()
    if len(sub) < 400:
        return None
    c = sub["Close"]
    v = sub["Volume"]
    r = c.pct_change().dropna()

    # rolling 60d ann vol series
    rv60 = r.rolling(60).std() * math.sqrt(252)

    # lagged window: rows ending LAG_DAYS ago, lookback HIST_LOOKBACK
    end_idx = len(rv60) - LAG_DAYS
    start_idx = max(0, end_idx - HIST_LOOKBACK)
    hist_slice = rv60.iloc[start_idx:end_idx].dropna()
    if len(hist_slice) < 60:
        return None
    hist_rv = float(hist_slice.median())
    hist_rv_max = float(hist_slice.max())

    rv60_now = float(rv60.iloc[-1]) if not np.isnan(rv60.iloc[-1]) else float("nan")
    price = float(c.iloc[-1])
    adv = float((c * v).tail(30).mean())

    return {
        "price": price,
        "adv_usd": adv,
        "hist_rv_median": hist_rv,
        "hist_rv_max": hist_rv_max,
        "rv_60_now": rv60_now,
        "rv_ratio_now_vs_hist": rv60_now / hist_rv if hist_rv > 0 else float("nan"),
        "n_obs": len(sub),
    }


def main() -> None:
    print("[v2-universe] fetching SP500 list ...")
    sp = fetch_sp500()
    print(f"[v2-universe] {len(sp)} constituents")

    tickers = sp["ticker"].tolist() + ["SPY"]
    print("[v2-universe] bulk downloading 3y daily ...")
    px = yf.download(tickers, period="3y", interval="1d",
                     group_by="ticker", auto_adjust=True,
                     threads=True, progress=False)

    rows = []
    for t in sp["ticker"]:
        try:
            sub = px[t]
        except KeyError:
            continue
        r = compute_row(sub)
        if r is None:
            continue
        r["ticker"] = t
        rows.append(r)

    met = pd.DataFrame(rows)
    met = met.merge(sp[["ticker", "sector"]], on="ticker", how="left")
    print(f"[v2-universe] pre-filter: {len(met)}")

    flt = met[
        (met["price"] >= PRICE_MIN)
        & (met["hist_rv_median"] >= HIST_RV_MIN)
        & (met["hist_rv_median"] <= HIST_RV_MAX)
        & (met["adv_usd"] >= ADV_MIN_USD)
    ].copy()
    flt = flt.sort_values("rv_ratio_now_vs_hist", ascending=False).reset_index(drop=True)

    out = DATA / "burst_universe_v2.csv"
    flt.to_csv(out, index=False)
    print(f"[v2-universe] kept {len(flt)} tickers -> {out}")
    print("[v2-universe] top 15 by recent / historical vol ratio (regime shift):")
    print(flt[["ticker", "price", "hist_rv_median", "rv_60_now",
               "rv_ratio_now_vs_hist", "sector"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
