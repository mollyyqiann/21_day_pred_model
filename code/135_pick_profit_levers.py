"""Phase F.6: What actually increases profit on the real MG v3 stock picks?

134 answered "does leveraging the picks help" (no). This answers the useful
question: given the SAME 258 matured live picks, which exit / sizing rule makes
the most money?

Levers swept, all on the identical pick set so they are directly comparable:
  1. take-profit threshold        +5% .. +40%, sell whole position at first touch
  2. half-out take-profit         sell half at first touch, rest to 21d
  3. stop-loss                    -5% .. -30%, whole position
  4. TP x SL grid                 the interaction, which is where the money is
  5. hold length                  5 / 10 / 15 / 21 / 30 trading days
  6. rank within the top-5        is rank 1 better than rank 5?
  7. entry tag                    EXTREME / FRESH / CATALYST / MILD / COOLED

Intraday high/low are used for touch detection, not closes, because a stop or
target is hit intraday in reality. That is deliberately conservative for stops
(they trigger more often) and generous for targets -- both directions are
reported so the bias is visible.

A note on significance: this is a 4-month, 57-date sample with 21-day overlapping
windows, i.e. ~3 genuinely independent periods. Nothing here reaches significance
and this script does not pretend otherwise -- it reports effect sizes and the
share of picks each rule actually changes, which is what makes a rule worth
carrying forward to a longer test.

Run: /Users/mollyqian/anaconda3/bin/python code/135_pick_profit_levers.py
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
m134 = import_module("134_leveraged_overlay_backtest")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "monthly_gainer"
HOLD = 21


def fetch_ohlc(tickers, start, end):
    raw = yf.download(sorted(set(tickers)), start=start, end=end, interval="1d",
                      group_by="ticker", auto_adjust=True, threads=True, progress=False)
    out = {}
    for t in set(tickers):
        try:
            s = raw[t][["Open", "High", "Low", "Close"]].dropna()
        except Exception:
            continue
        if len(s):
            s.index = pd.to_datetime(s.index).tz_localize(None)
            out[t] = s
    return out


def build_paths(picks, px, hold=30):
    """One row per pick with the entry price and the forward OHLC path."""
    recs = []
    for _, p in picks.iterrows():
        tk = p["ticker"]
        if tk not in px:
            continue
        fut = px[tk][px[tk].index > p["pick_date"]]
        if len(fut) < hold + 1:
            continue
        entry = float(fut["Open"].iloc[0])
        if entry <= 0:
            continue
        recs.append({
            "pick_date": p["pick_date"], "ticker": tk, "rank": p["rank"], "tag": p["tag"],
            "entry": entry,
            "high": fut["High"].iloc[:hold].values.astype(float),
            "low": fut["Low"].iloc[:hold].values.astype(float),
            "close": fut["Close"].iloc[:hold].values.astype(float),
        })
    return recs


def simulate(rec, tp=None, sl=None, hold=HOLD, half=False):
    """Walk the path day by day. Stop is checked before target within a day
    (conservative: assumes the adverse level trades first)."""
    e = rec["entry"]
    hi = rec["high"][:hold] / e - 1.0
    lo = rec["low"][:hold] / e - 1.0
    cl = rec["close"][:hold] / e - 1.0
    for i in range(len(cl)):
        if sl is not None and lo[i] <= -sl:
            return -sl
        if tp is not None and hi[i] >= tp:
            return 0.5 * tp + 0.5 * cl[-1] if half else tp
    return float(cl[-1])


def summarise(rets, base):
    r = np.asarray(rets, dtype=float)
    return {"mean": r.mean(), "median": float(np.median(r)), "win%": (r > 0).mean(),
            "vs_base": r.mean() - base, "worst": r.min(),
            "changed%": float(np.mean(np.abs(r - base_arr) > 1e-9)) if False else np.nan}


def main():
    picks = m134.load_picks()
    picks = picks[picks["ticker"].isin(m134.PROXY)].copy()
    px = fetch_ohlc(set(picks["ticker"]),
                    picks["pick_date"].min() - pd.Timedelta(days=10),
                    pd.Timestamp.today() + pd.Timedelta(days=1))
    recs = build_paths(picks, px, hold=30)
    print(f"[135] {len(recs)} matured picks with a full 30-day forward path "
          f"({picks['pick_date'].min().date()} -> {picks['pick_date'].max().date()})")

    base = np.array([simulate(r) for r in recs])
    print(f"[135] BASELINE (hold {HOLD}d, no TP, no SL): mean {base.mean():+.2%}  "
          f"median {np.median(base):+.2%}  win {np.mean(base > 0):.1%}  "
          f"worst {base.min():.1%}\n")

    def row(label, rets, n_changed=None):
        r = np.asarray(rets)
        d = {"rule": label, "mean": r.mean(), "median": float(np.median(r)),
             "win%": float((r > 0).mean()), "vs_base": r.mean() - base.mean(),
             "worst": r.min()}
        if n_changed is not None:
            d["fired%"] = n_changed
        return d

    # ---- 1 & 2: take-profit ----
    print("[135] === TAKE-PROFIT SWEEP (sell all at first intraday touch) ===")
    rows = [row("hold 21d (baseline)", base, 0.0)]
    for tp in [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40]:
        rets = [simulate(r, tp=tp) for r in recs]
        fired = np.mean([max(r["high"][:HOLD] / r["entry"] - 1.0) >= tp for r in recs])
        rows.append(row(f"TP +{tp:.1%}", rets, fired))
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "mean": "{:+.2%}".format, "median": "{:+.2%}".format, "win%": "{:.1%}".format,
        "vs_base": "{:+.2%}".format, "worst": "{:.1%}".format, "fired%": "{:.1%}".format}))

    print("\n[135] === HALF-OUT TAKE-PROFIT (sell 50% at touch, rest to 21d) ===")
    rows = [row("hold 21d (baseline)", base, 0.0)]
    for tp in [0.10, 0.15, 0.20, 0.30]:
        rets = [simulate(r, tp=tp, half=True) for r in recs]
        fired = np.mean([max(r["high"][:HOLD] / r["entry"] - 1.0) >= tp for r in recs])
        rows.append(row(f"half out at +{tp:.0%}", rets, fired))
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "mean": "{:+.2%}".format, "median": "{:+.2%}".format, "win%": "{:.1%}".format,
        "vs_base": "{:+.2%}".format, "worst": "{:.1%}".format, "fired%": "{:.1%}".format}))

    # ---- 3: stop-loss ----
    print("\n[135] === STOP-LOSS SWEEP ===")
    rows = [row("hold 21d (baseline)", base, 0.0)]
    for sl in [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]:
        rets = [simulate(r, sl=sl) for r in recs]
        fired = np.mean([min(r["low"][:HOLD] / r["entry"] - 1.0) <= -sl for r in recs])
        rows.append(row(f"SL -{sl:.1%}", rets, fired))
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "mean": "{:+.2%}".format, "median": "{:+.2%}".format, "win%": "{:.1%}".format,
        "vs_base": "{:+.2%}".format, "worst": "{:.1%}".format, "fired%": "{:.1%}".format}))

    # ---- 4: TP x SL grid ----
    print("\n[135] === TP x SL GRID (mean return) ===")
    tps = [0.075, 0.10, 0.125, 0.15, 0.20, 0.30]
    sls = [0.05, 0.075, 0.10, 0.15, 0.20, None]
    grid = pd.DataFrame(index=[f"TP+{t:.1%}" for t in tps],
                        columns=[(f"SL-{s:.1%}" if s else "no SL") for s in sls], dtype=float)
    for t in tps:
        for s in sls:
            grid.loc[f"TP+{t:.1%}", (f"SL-{s:.1%}" if s else "no SL")] = \
                np.mean([simulate(r, tp=t, sl=s) for r in recs])
    print((grid * 100).round(2).to_string())
    best = grid.stack().idxmax()
    print(f"[135] best cell: {best[0]} / {best[1]} = {grid.stack().max():+.2%} "
          f"(baseline {base.mean():+.2%})")

    # ---- 5: hold length ----
    print("\n[135] === HOLD LENGTH (no TP/SL) ===")
    rows = []
    for h in [5, 10, 15, 21, 30]:
        rets = [simulate(r, hold=h) for r in recs]
        rows.append(row(f"hold {h}d", rets))
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "mean": "{:+.2%}".format, "median": "{:+.2%}".format, "win%": "{:.1%}".format,
        "vs_base": "{:+.2%}".format, "worst": "{:.1%}".format}))

    # ---- 6 & 7: rank and tag ----
    df = pd.DataFrame([{"rank": r["rank"], "tag": r["tag"], "ticker": r["ticker"],
                        "ret": b} for r, b in zip(recs, base)])
    print("\n[135] === BY RANK IN THE TOP-5 (hold 21d) ===")
    print(df.groupby("rank")["ret"].agg(n="size", mean="mean", win=lambda s: (s > 0).mean())
          .to_string(formatters={"mean": "{:+.2%}".format, "win": "{:.1%}".format}))
    print("\n[135] === BY ENTRY TAG (hold 21d) ===")
    print(df.groupby("tag")["ret"].agg(n="size", mean="mean", win=lambda s: (s > 0).mean())
          .sort_values("mean", ascending=False)
          .to_string(formatters={"mean": "{:+.2%}".format, "win": "{:.1%}".format}))

    df.to_csv(OUT / "pick_profit_levers.csv", index=False)
    print(f"\n[135] wrote {OUT / 'pick_profit_levers.csv'}")


if __name__ == "__main__":
    main()
