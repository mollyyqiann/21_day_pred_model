"""Reflection: how did the smallcap top-20 from 2026-05-01 actually do?

Pulls fresh yfinance for each of the top-20 names predicted on 2026-05-01,
shows realized return since then, and computes basket-level performance vs
the SP500 (and vs MRNA/SMCI/INTC user holdings).

Goal: validate or invalidate the EXTREME-flag decision to skip these names.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)
import warnings
import pandas as pd
import yfinance as yf
warnings.filterwarnings("ignore")

# 2026-05-01 closing prices (the prediction-date snapshot)
TOP20_SC_BY_RANK = [
    ("ERAS",  10.17,  "EXTREME"),
    ("CAR",   181.59, "FRESH"),     # the only "safe" pick
    ("SIMO",  227.99, "EXTREME"),
    ("ALMU",  24.56,  "EXTREME"),
    ("MXL",   76.91,  "EXTREME"),
    ("USAR",  26.18,  "EXTREME"),
    ("GLXY",  28.88,  "EXTREME"),
    ("NBIS",  154.93, "EXTREME"),
    ("BNAI",  27.22,  "EXTREME"),
    ("CRDO",  180.02, "EXTREME"),
    ("INBX",  125.57, "EXTREME"),
    ("WATT",  33.16,  "EXTREME"),
    ("AAOI",  187.83, "EXTREME"),
    ("BE",    287.03, "EXTREME"),
    ("BETR",  43.02,  "EXTREME"),
    ("ALAB",  201.91, "EXTREME"),
    ("TTMI",  157.57, "EXTREME"),
    ("CRCL",  97.75,  "EXTREME"),
    ("APLD",  34.29,  "EXTREME"),
    ("SGML",  21.37,  "EXTREME"),
]

# Holdings + bench
EXTRA = [("INTC", 99.62, "EXTREME"),
         ("SMCI", 27.09, "MILD"),
         ("MRNA", 45.37, "FRESH"),
         ("SPY",  None,  "BENCH")]


def get_current(ticker):
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="10d", auto_adjust=True)
        if h.empty:
            return None, None
        return float(h["Close"].iloc[-1]), str(h.index[-1].date())
    except Exception:
        return None, None


def main():
    print("[112] pulling current prices vs 2026-05-01 reference ...")
    print()
    print(f"{'ticker':<6} {'ext':<8} {'5/01':>9} {'today':>9} {'%chg':>7}  {'verdict'}")
    print('-' * 70)

    rows = []
    for tk, ref, ext in TOP20_SC_BY_RANK:
        cur, asof = get_current(tk)
        if cur is None:
            print(f"{tk:<6} {ext:<8} ${ref:>7.2f}      n/a      n/a")
            continue
        chg = cur / ref - 1.0
        rows.append({"ticker": tk, "ext": ext, "ref": ref, "cur": cur, "chg": chg})
        verdict = "↑↑" if chg > 0.10 else ("↑" if chg > 0 else ("↓↓" if chg < -0.10 else "↓"))
        print(f"{tk:<6} {ext:<8} ${ref:>7.2f} ${cur:>7.2f} {chg:>+7.1%}  {verdict}")

    print()
    print("=== Holdings + bench ===")
    for tk, ref, ext in EXTRA:
        cur, asof = get_current(tk)
        if cur is None or ref is None:
            if tk == "SPY":
                cur, _ = get_current("SPY")
                spy5d = None
                # get 5d ago for bench
                t = yf.Ticker("SPY")
                h = t.history(period="10d", auto_adjust=True)
                if len(h) >= 6:
                    ref_spy = float(h["Close"].iloc[-6])
                    chg = cur / ref_spy - 1.0
                    print(f"{tk:<6} {ext:<8} ${ref_spy:>7.2f} ${cur:>7.2f} {chg:>+7.1%}  (5-day reference)")
                    rows.append({"ticker": tk, "ext": ext, "ref": ref_spy, "cur": cur, "chg": chg})
            continue
        chg = cur / ref - 1.0
        rows.append({"ticker": tk, "ext": ext, "ref": ref, "cur": cur, "chg": chg})
        verdict = "↑↑" if chg > 0.10 else ("↑" if chg > 0 else ("↓↓" if chg < -0.10 else "↓"))
        print(f"{tk:<6} {ext:<8} ${ref:>7.2f} ${cur:>7.2f} {chg:>+7.1%}  {verdict}")

    df = pd.DataFrame(rows)
    sc = df[df["ext"].isin(["EXTREME", "FRESH"]) & ~df["ticker"].isin(["INTC","SMCI","MRNA","SPY"])]
    extreme = sc[sc["ext"] == "EXTREME"]
    safe = sc[sc["ext"] == "FRESH"]
    holdings = df[df["ticker"].isin(["INTC", "SMCI", "MRNA"])]
    spy = df[df["ticker"] == "SPY"]

    print()
    print("=== AGGREGATE 5-day returns (since 2026-05-01) ===")
    print(f"  EXTREME smallcap (n={len(extreme)})  mean={extreme['chg'].mean():+.1%}  "
          f"median={extreme['chg'].median():+.1%}  best={extreme['chg'].max():+.1%}  worst={extreme['chg'].min():+.1%}")
    if len(safe):
        print(f"  FRESH smallcap   (n={len(safe)})  mean={safe['chg'].mean():+.1%}")
    print(f"  USER HOLDINGS    (n={len(holdings)})  mean={holdings['chg'].mean():+.1%}  "
          f"median={holdings['chg'].median():+.1%}")
    if len(spy):
        print(f"  SPY (bench)             5d={spy['chg'].iloc[0]:+.1%}")
    print()
    print(f"  How many EXTREME picks gained?  {(extreme['chg']>0).sum()} / {len(extreme)}")
    print(f"  How many EXTREME picks dropped >5%?  {(extreme['chg']<-0.05).sum()} / {len(extreme)}")
    print(f"  How many EXTREME picks dropped >10%? {(extreme['chg']<-0.10).sum()} / {len(extreme)}")


if __name__ == "__main__":
    main()
