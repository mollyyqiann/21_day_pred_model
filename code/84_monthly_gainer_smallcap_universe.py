"""Phase E.1: Build "broader than S&P 500" universe for monthly gainer test.

Uses burst_universe_rh.csv (Robinhood-tradable, price>$20) MINUS burst_universe_v7
(S&P 500 we already have data for) to get the *new* tickers we need to download.

This is a Robinhood-tradable broader universe, NOT strictly Russell 2000 — to be
honest, we don't have PIT R2K constituent data and yfinance bulk download for
2000 small caps would have heavy survivorship + data-quality issues. The rh set
spans mid-and-larger small caps and is a fair proxy for testing whether the
model generalizes off the S&P 500.

Output: data/monthly_gainer_universe_smallcap.csv
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main():
    rh = pd.read_csv(DATA / "burst_universe_rh.csv")
    sp = pd.read_csv(DATA / "burst_universe_v7.csv")
    print(f"[84] rh universe: {len(rh)} tickers")
    print(f"[84] v7 (S&P 500) universe: {len(sp)} tickers")

    new_t = sorted(set(rh["ticker"]) - set(sp["ticker"]))
    smallcap = rh[rh["ticker"].isin(new_t)].copy().reset_index(drop=True)
    print(f"[84] new tickers (rh \\ v7): {len(smallcap)}")
    print(f"[84] median rv_60_now: rh={rh['rv_60_now'].median():.3f}  "
          f"sp={sp['rv_60_now'].median():.3f}  smallcap={smallcap['rv_60_now'].median():.3f}")
    print(f"[84] median price: rh={rh['price'].median():.0f}  "
          f"sp={sp['price'].median():.0f}  smallcap={smallcap['price'].median():.0f}")
    print(f"[84] median adv_usd: rh={rh['adv_usd'].median():,.0f}  "
          f"sp={sp['adv_usd'].median():,.0f}  smallcap={smallcap['adv_usd'].median():,.0f}")

    smallcap.to_csv(DATA / "monthly_gainer_universe_smallcap.csv", index=False)
    print(f"[84] wrote {DATA / 'monthly_gainer_universe_smallcap.csv'}")


if __name__ == "__main__":
    main()
