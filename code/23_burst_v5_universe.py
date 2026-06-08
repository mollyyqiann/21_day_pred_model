"""V5 — all S&P 500 names (no price filter), filtered for UPSIDE-asymmetric
volatility.

The instinct we're encoding: we want stocks that, *when* they move, tend to
move up more violently than they move down — OR that produce multi-day
sustained up-moves (not two-way chop). A stock that's symmetrically volatile
(think pure crypto proxy) is NOT what we want; it'll burn as often as it rips.

Per-ticker asymmetry metrics (computed over 3y of daily returns):
  - up_semivol:   sqrt(mean(r^2 * (r>0)))   (upside-only realised vol)
  - dn_semivol:   sqrt(mean(r^2 * (r<0)))   (downside-only realised vol)
  - semivol_ratio = up_semivol / dn_semivol
  - skew:         sample skewness of daily returns
  - up_big / dn_big: count of days with ret > +3% vs ret < -3%
  - big_ratio = up_big / max(1, dn_big)
  - max_3d_up_run: max 3-day cumulative return (positive)
  - min_3d_dn_run: min 3-day cumulative return (negative, as magnitude)
  - run_ratio = max_3d_up_run / abs(min_3d_dn_run)

Qualifying filter (inclusive - pass if ANY of these hold, to catch multiple
flavors of upside asymmetry):
  1. semivol_ratio >= 1.10   (upside realised vol at least 10% bigger)
  2. skew >= 0.30            (positive daily-return skew, distinct upside tail)
  3. run_ratio >= 1.20       (best 3d up run beats worst 3d down run by >=20%)
  4. big_ratio >= 1.30       (at least 30% more big-up days than big-down days)

Additionally, we exclude pure sideways chop (require realised annual vol >= 15%)
and require basic liquidity: 30d ADV >= $5M (relaxed from v4's $25M because
we're explicitly including smaller names).

Outputs:
  data/burst_universe_v5.csv
"""

from __future__ import annotations

import io
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0"}

ADV_MIN = 5_000_000
MIN_ANN_VOL = 0.15   # exclude sideways chop (ann vol < 15%)
MIN_OBS = 250


def fetch_sp500():
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers=UA, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0][["Symbol", "Security", "GICS Sector"]].copy()
    t.columns = ["ticker", "name", "sector"]
    t["ticker"] = t["ticker"].str.replace(".", "-", regex=False)
    return t


def asymmetry_metrics(r: pd.Series) -> dict:
    r = r.dropna().values
    if len(r) < MIN_OBS:
        return {}
    up = r[r > 0]; dn = r[r < 0]
    up_semi = float(np.sqrt((up * up).mean()) * math.sqrt(252)) if len(up) else 0.0
    dn_semi = float(np.sqrt((dn * dn).mean()) * math.sqrt(252)) if len(dn) else 0.0
    skew = float(pd.Series(r).skew())
    big_up = int((r > 0.03).sum()); big_dn = int((r < -0.03).sum())
    # 3-day cumulative return series
    cum3 = pd.Series(r).rolling(3).apply(lambda x: (1 + x).prod() - 1, raw=True)
    max_up = float(cum3.max()) if cum3.notna().any() else 0.0
    min_dn = float(cum3.min()) if cum3.notna().any() else 0.0
    return {
        "up_semivol": up_semi,
        "dn_semivol": dn_semi,
        "semivol_ratio": (up_semi / dn_semi) if dn_semi > 0 else float("nan"),
        "skew": skew,
        "up_big": big_up,
        "dn_big": big_dn,
        "big_ratio": (big_up / max(1, big_dn)),
        "max_3d_up_run": max_up,
        "min_3d_dn_run": min_dn,
        "run_ratio": (max_up / abs(min_dn)) if min_dn < 0 else float("nan"),
        "ann_vol": float(np.std(r) * math.sqrt(252)),
    }


def passes(row: dict) -> tuple[bool, list[str]]:
    reasons = []
    if row.get("semivol_ratio", 0) >= 1.10:
        reasons.append("semivol")
    if row.get("skew", 0) >= 0.30:
        reasons.append("skew")
    if row.get("run_ratio", 0) >= 1.20:
        reasons.append("run")
    if row.get("big_ratio", 0) >= 1.30:
        reasons.append("bigdays")
    return (len(reasons) >= 1, reasons)


def main():
    print("[v5] fetching SP500 list ...")
    sp = fetch_sp500()
    print(f"[v5] {len(sp)} constituents")

    print("[v5] bulk downloading 3y daily ...")
    raw = yf.download(sp["ticker"].tolist(), period="3y", interval="1d",
                      group_by="ticker", auto_adjust=True, threads=True,
                      progress=False)

    rows = []
    for t in sp["ticker"]:
        try:
            sub = raw[t].dropna()
        except KeyError:
            continue
        if len(sub) < MIN_OBS:
            continue
        c, v = sub["Close"], sub["Volume"]
        r = c.pct_change()
        metrics = asymmetry_metrics(r)
        if not metrics:
            continue
        if metrics["ann_vol"] < MIN_ANN_VOL:
            continue
        adv = float((c * v).tail(30).mean())
        if adv < ADV_MIN:
            continue
        ok, reasons = passes(metrics)
        if not ok:
            continue
        rows.append({
            "ticker": t,
            "price": float(c.iloc[-1]),
            "adv_usd": adv,
            "n_obs": int(len(sub)),
            "reasons": "|".join(reasons),
            **metrics,
        })

    uni = pd.DataFrame(rows).merge(sp[["ticker", "sector"]], on="ticker", how="left")
    uni = uni.sort_values(["run_ratio", "semivol_ratio"], ascending=False).reset_index(drop=True)

    out = DATA / "burst_universe_v5.csv"
    uni.to_csv(out, index=False)
    print(f"[v5] qualifying tickers: {len(uni)} (of {len(sp)})")
    print("[v5] top 25 by run_ratio (best-3d-up-run / worst-3d-down-run):")
    cols = ["ticker", "price", "semivol_ratio", "skew", "up_big", "dn_big",
            "big_ratio", "max_3d_up_run", "min_3d_dn_run", "run_ratio",
            "ann_vol", "reasons", "sector"]
    print(uni[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
