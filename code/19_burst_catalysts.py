"""Classify historical burst events in the v3 universe by catalyst.

For each ticker in the v3 universe, pull its historical earnings dates via
yfinance.Ticker.earnings_dates and compute, for every y==1 row in the panel,
the distance (in trading days) to the nearest earnings release. Classify:

  "earnings" - |days to nearest earnings| <= 3 trading days
  "other"    - otherwise

Also for each qualifying 'most recent episode', try to fetch current news
headlines for the ticker and emit a human-scannable report row.

Outputs:
  output/burst_catalysts.csv
  output/burst_catalyst_summary.json
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

EARN_WINDOW = 3  # +/- trading days


def nearest_earnings_distance(burst_date: pd.Timestamp, earnings_ts: list) -> int | None:
    if not earnings_ts:
        return None
    diffs = [abs((pd.Timestamp(e) - burst_date).days) for e in earnings_ts]
    return min(diffs)


def main() -> None:
    uni = pd.read_csv(DATA / "burst_universe_v3.csv")
    panel = pd.read_csv(DATA / "burst_panel_v2.csv", parse_dates=["date"])
    tickers = uni["ticker"].tolist()
    print(f"[catalysts] analyzing {len(tickers)} v3 tickers ...")

    rows = []
    per_ticker_summary = []
    for i, t in enumerate(tickers, 1):
        bursts = panel[(panel["ticker"] == t) & (panel["y"] == 1)].copy()
        if len(bursts) == 0:
            continue

        # pull earnings history
        earn_ts = []
        try:
            ed = yf.Ticker(t).earnings_dates
            if ed is not None and len(ed) > 0:
                earn_ts = [pd.Timestamp(x).tz_localize(None)
                           if hasattr(x, "tz_localize") else pd.Timestamp(x)
                           for x in ed.index]
        except Exception:
            pass

        n_earn_driven = 0
        for _, r in bursts.iterrows():
            bd = pd.Timestamp(r["date"])
            d = nearest_earnings_distance(bd, earn_ts)
            # translate calendar days to ~trading days (<=5 calendar days ~ <=3 trading days in most cases)
            is_earn = (d is not None) and (d <= 5)
            if is_earn:
                n_earn_driven += 1
            rows.append({
                "ticker": t,
                "burst_date": bd.date().isoformat(),
                "close": r["close"],
                "nearest_earnings_calendar_days": d,
                "catalyst": "earnings" if is_earn else "other",
                "ret_5d_at_event": r.get("ret_5d", np.nan),
                "rv_20_at_event": r.get("rv_20", np.nan),
                "rv_ratio_vs_hist_at_event": r.get("rv_ratio_vs_hist", np.nan),
            })

        per_ticker_summary.append({
            "ticker": t,
            "n_bursts": len(bursts),
            "n_earnings_driven": n_earn_driven,
            "pct_earnings": n_earn_driven / max(1, len(bursts)),
        })
        if i % 10 == 0:
            print(f"[catalysts]   {i}/{len(tickers)}")
        time.sleep(0.12)  # be nice to the API

    df = pd.DataFrame(rows).sort_values(["ticker", "burst_date"])
    df.to_csv(OUT / "burst_catalysts.csv", index=False)

    sdf = pd.DataFrame(per_ticker_summary).sort_values("n_bursts", ascending=False)
    overall = {
        "total_bursts": int(df.shape[0]),
        "pct_earnings": float((df["catalyst"] == "earnings").mean()) if len(df) else 0.0,
        "pct_other":    float((df["catalyst"] == "other").mean()) if len(df) else 0.0,
        "by_ticker": sdf.to_dict(orient="records"),
    }
    (OUT / "burst_catalyst_summary.json").write_text(json.dumps(overall, indent=2))

    print("\n=== catalyst breakdown (v3 tickers historical bursts) ===")
    print(f"total bursts: {overall['total_bursts']}")
    print(f"% earnings-driven (|d| <= 5 cal days): {overall['pct_earnings']:.1%}")
    print(f"% other-catalyst: {overall['pct_other']:.1%}")
    print("\nper-ticker:")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
