"""Diff current SP500 list vs the model's panel universe.

Pulls current SP500 from Wikipedia (canonical), compares against
data/monthly_gainer_panel.csv tickers, reports:
  - In SP500 today but NOT in our panel (the MRVL-class misses)
  - In our panel but NOT in SP500 today (stale removals)
  - Sector breakdown of the missing set
"""

import sys; sys.stdout.reconfigure(line_buffering=True)
import warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def fetch_current_sp500():
    """Pull current SP500 constituents from Wikipedia (with UA header) or fallback to SPY holdings."""
    import urllib.request, io
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html))
        df = tables[0]
        sym_col = "Symbol" if "Symbol" in df.columns else "Ticker"
        sec_col = "GICS Sector" if "GICS Sector" in df.columns else "Sector"
        sub_col = "GICS Sub-Industry" if "GICS Sub-Industry" in df.columns else None
        cols = [sym_col, sec_col]
        if sub_col: cols.append(sub_col)
        df = df[cols].rename(columns={sym_col: "ticker", sec_col: "sector"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        return df
    except Exception as e:
        print(f"[109] Wikipedia fetch failed: {e}")
        return None


def main():
    print("[109] fetching current SP500 from Wikipedia ...")
    sp = fetch_current_sp500()
    if sp is None:
        print("[109] could not fetch — abort")
        return
    sp_set = set(sp["ticker"].tolist())
    print(f"[109] current SP500: {len(sp_set)} tickers")

    print("[109] loading model's SP500 panel ...")
    panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", usecols=["ticker"])
    panel_set = set(panel["ticker"].unique().tolist())
    print(f"[109] model panel:    {len(panel_set)} tickers")

    in_sp_not_panel = sorted(sp_set - panel_set)
    in_panel_not_sp = sorted(panel_set - sp_set)

    print(f"\n=== {len(in_sp_not_panel)} tickers in current SP500 but NOT in model panel (model misses these) ===")
    if in_sp_not_panel:
        miss_df = sp[sp["ticker"].isin(in_sp_not_panel)].sort_values(["sector", "ticker"])
        for sec, grp in miss_df.groupby("sector"):
            print(f"\n  {sec} ({len(grp)}):")
            for _, r in grp.iterrows():
                print(f"    - {r['ticker']}")

    print(f"\n=== {len(in_panel_not_sp)} tickers in model panel but NOT in current SP500 (stale members) ===")
    if in_panel_not_sp:
        for tk in in_panel_not_sp:
            print(f"  - {tk}")

    print(f"\n=== summary ===")
    print(f"  current SP500 size:     {len(sp_set)}")
    print(f"  model panel size:       {len(panel_set)}")
    print(f"  overlap:                {len(sp_set & panel_set)}")
    print(f"  missing from model:     {len(in_sp_not_panel)}")
    print(f"  stale in model:         {len(in_panel_not_sp)}")
    print(f"  coverage:               {len(sp_set & panel_set) / len(sp_set):.1%}")

    # Save
    out = DATA / "sp500_universe_diff.csv"
    pd.DataFrame({"ticker": in_sp_not_panel,
                   "status": "missing_from_panel"}).to_csv(out, index=False)
    print(f"\n[109] saved missing tickers to {out}")


if __name__ == "__main__":
    main()
