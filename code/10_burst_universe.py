"""Build the candidate universe: S&P 500 constituents that are typically NON-volatile,
priced above $40, with sufficient liquidity.

Outputs:
  data/burst_universe.csv  - columns: ticker, price, rv60_ann, adv_usd, beta, sector
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
}

# Volatility filter: usually NOT volatile.
# A rough cut-off: 60-day realised annualised vol below 30% and above 5%.
RV_MAX = 0.30
RV_MIN = 0.05
PRICE_MIN = 40.0
ADV_MIN_USD = 25_000_000  # liquidity floor


def fetch_sp500_tickers() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers=UA, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["ticker", "name", "sector"]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df


def bulk_history(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    # yfinance bulk returns a multi-index frame; pull Close & Volume
    px = yf.download(
        tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    return px


def compute_metrics(px: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            sub = px[t].dropna()
        except KeyError:
            continue
        if len(sub) < 80:
            continue
        close = sub["Close"]
        vol = sub["Volume"]
        ret = close.pct_change().dropna()
        rv60 = ret.tail(60).std() * np.sqrt(252)
        price = float(close.iloc[-1])
        adv = float((close * vol).tail(30).mean())
        rows.append({
            "ticker": t,
            "price": price,
            "rv60_ann": float(rv60),
            "adv_usd": adv,
            "n_obs": len(sub),
        })
    return pd.DataFrame(rows)


def beta_vs_spy(px: pd.DataFrame, tickers: list[str], spy_ret: pd.Series) -> dict:
    betas = {}
    for t in tickers:
        try:
            r = px[t]["Close"].pct_change().dropna()
        except KeyError:
            continue
        common = r.index.intersection(spy_ret.index)
        if len(common) < 60:
            continue
        x = spy_ret.reindex(common).values
        y = r.reindex(common).values
        var = np.var(x)
        if var <= 0:
            continue
        betas[t] = float(np.cov(x, y, ddof=0)[0, 1] / var)
    return betas


def main() -> None:
    print("[universe] pulling S&P 500 constituent list ...")
    sp = fetch_sp500_tickers()
    print(f"[universe] {len(sp)} constituents")

    # Include SPY for beta calc
    tickers = sp["ticker"].tolist() + ["SPY"]
    print("[universe] bulk downloading 1y daily history for filtering ...")
    px = bulk_history(tickers, period="1y")

    spy = px["SPY"]["Close"].dropna()
    spy_ret = spy.pct_change().dropna()

    print("[universe] computing metrics ...")
    met = compute_metrics(px, sp["ticker"].tolist())
    betas = beta_vs_spy(px, sp["ticker"].tolist(), spy_ret)
    met["beta"] = met["ticker"].map(betas)
    met = met.merge(sp[["ticker", "sector"]], on="ticker", how="left")

    print(f"[universe] pre-filter: {len(met)}")

    flt = met[
        (met["price"] >= PRICE_MIN)
        & (met["rv60_ann"] <= RV_MAX)
        & (met["rv60_ann"] >= RV_MIN)
        & (met["adv_usd"] >= ADV_MIN_USD)
        & met["beta"].notna()
    ].copy()
    flt = flt.sort_values("rv60_ann").reset_index(drop=True)

    out = DATA / "burst_universe.csv"
    flt.to_csv(out, index=False)
    print(f"[universe] kept {len(flt)} tickers -> {out}")
    print(flt.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
