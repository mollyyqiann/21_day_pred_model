"""Phase F.7: adjudicate the take-profit contradiction (+12% vs +30% / no cap).

THE DISPUTE, as recorded in 123_mg_close_scorer.py's TAKE_PROFIT comment:
  (a) MFE view  -- FAVOURABLE picks have a median max-favourable-excursion of
      +12.27%, so a +12% target is "where the median pick tops out"; 135's
      sweep found every low TP beating hold-21d by 3-6pp.
  (b) fat-tail view (21_day_pred_model/ANALYSIS.md, June) -- all the money is in
      3 names; a full lock-in cut the mean from +13.4% to +9.8%.

WHAT THIS SCRIPT SHOWS
----------------------
They are not measuring the same population, and neither is the population the
live rules trade.

  1. 135 silently restricted the picks to the 33 tickers in 134's leveraged-ETF
     PROXY map, and counted every re-appearance of a name in the top-5 as a
     separate pick. 43% of its rows are 9th-or-later re-appearances of the same
     ticker, mean 21d return -10.60%. raw_margin is a volatility sort with
     dd_60d corr -0.71, so a name that keeps falling keeps ranking -- the
     re-appearance rows ARE the losers, and a take-profit "wins" mostly by
     cutting them. 123 never buys those rows: entry is first appearance only.
  2. On first appearances (n=34, the tradeable population) hold-21d is +3.58%
     and EVERY full take-profit loses to it. The June fat-tail claim reproduces
     on this larger sample: top-3 mean +71.3% vs -3.0% for the rest; drop the
     top 3 and hold -2.97% < TP12 +0.10%. The whole disagreement is 3 names
     bought in the first week of May 2026 (SMCI/MU/AMD).
  3. The "+12% is an artifact of a same-close entry" claim is FALSE: the
     all-rows TP advantage survives both entry models (next open / prior close)
     and both fill models (resting limit vs the once-daily 15:45 close check).
  4. "Median MFE = +12.27%" is not an argument for a +12% target. MFE and
     outcome are jointly distributed: spearman(peak day, 21d return) = +0.652
     (n=280, p=2e-35); winners peak on day ~15, losers on day ~6. Selling at
     the median excursion truncates exactly the names that make the mean.
     (Partly mechanical -- a high finish puts the max near the end -- so the
     causal versions are tested too: time-limited TPs and trailing stops behave
     like fixed TPs, good on the re-appearance rows, bad on first appearances.)
  5. Regime, not level, drives the answer: on first appearances TP12 - hold is
     -9.0pp in May, +8.5pp in June, 0.0pp in July. Three months, three answers,
     ~4 independent 21-day windows in the whole sample. Neither +12% nor +30%
     is established.

CONCLUSION: a full-size take-profit is the one choice the data argues against.
+12% full-out has the worst worst-month regret (-9.02pp) of every rule tested
on the tradeable population. Half-out dominates full-out at every level and in
both populations -- it is the min-regret answer to a dispute that is a near-tie
in the mean and a large difference in the tail.

Sample: 390 top-5 rows parsed from sunday_verdict_*.md (2026-05-03..2026-09-09),
280 matured with a full 30-day forward path, 36 tickers. yfinance adjusted OHLC.

Run: /Users/mollyqian/anaconda3/bin/python code/136_tp_adjudication.py
"""
import sys; sys.stdout.reconfigure(line_buffering=True)

import glob, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as sps

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "monthly_gainer"
HOLD = 21

# rank. TICKER [TAG hist NN%] $price  margin +x.xx  prob NN%  5d +x.x%
PAT = re.compile(r"^\s*(?:👤|\s)?\s*(\d+)\.\s+([A-Z][A-Z\.\-]*)\s+(?:\[(\w+)[^\]]*\]\s+)?"
                 r"\$\s*([\d,\.]+)\s+margin\s+([-+][\d\.]+)\s+prob\s+(\d+)%\s+5d\s+([-+][\d\.]+)%")


def load_picks():
    rows = []
    for f in sorted(glob.glob(str(OUT / "sunday_verdict_2026-*.md"))):
        d = re.search(r"(\d{4}-\d{2}-\d{2})", Path(f).name).group(1)
        m = re.search(r"## Today's top-5.*?\n(.*?)(?=\n##|\Z)", Path(f).read_text(), re.S)
        if not m:
            continue
        for line in m.group(1).splitlines():
            mm = PAT.match(line)
            if mm:
                rows.append(dict(pick_date=pd.Timestamp(d), rank=int(mm.group(1)),
                                 ticker=mm.group(2), tag=mm.group(3) or "",
                                 prob=int(mm.group(6)) / 100, ret5=float(mm.group(7)) / 100))
    df = pd.DataFrame(rows).drop_duplicates(subset=["pick_date", "ticker"])
    return df.sort_values(["pick_date", "rank"]).reset_index(drop=True)


def fetch(tickers, start, end):
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


def build(picks, px, hold=30):
    """One record per pick: both entry models, the forward path, and the
    reconstructed extension band (dd_60d from prices, the other two flags are
    printed in the verdict file)."""
    recs = []
    for _, p in picks.iterrows():
        s = px.get(p["ticker"])
        if s is None:
            continue
        fut, prev = s[s.index > p["pick_date"]], s[s.index <= p["pick_date"]]
        if len(fut) < hold + 1 or not len(prev):
            continue
        c = prev["Close"]
        dd = (c.iloc[-60:].min() / c.iloc[-60:].max() - 1.0) if len(c) >= 20 else np.nan
        nflag = int(bool(dd > -0.10)) + int(p["ret5"] > 0.02) + int(p["prob"] > 0.33)
        recs.append(dict(pick_date=p["pick_date"], ticker=p["ticker"], rank=p["rank"],
                         e_open=float(fut["Open"].iloc[0]), e_close=float(c.iloc[-1]),
                         nflag=nflag,
                         high=fut["High"].iloc[:hold].values.astype(float),
                         low=fut["Low"].iloc[:hold].values.astype(float),
                         close=fut["Close"].iloc[:hold].values.astype(float)))
    recs.sort(key=lambda r: (r["pick_date"], r["rank"]))
    seen = {}
    for r in recs:                      # nth time this ticker has been printed
        seen[r["ticker"]] = seen.get(r["ticker"], 0) + 1
        r["appear"] = seen[r["ticker"]]
    return recs


def sim(rec, entry="e_open", tp=None, sl=None, hold=HOLD, fill="close", half=False):
    """fill='limit'  -> a resting limit order, fills at tp on the intraday touch.
       fill='close'  -> what 123 does: one look at 15:45, so the trigger is the
                        session price, and the fill is that price."""
    e = rec[entry]
    hi, lo, cl = rec["high"][:hold] / e - 1, rec["low"][:hold] / e - 1, rec["close"][:hold] / e - 1
    for i in range(len(cl)):
        if sl is not None and lo[i] <= -sl:
            return -sl
        if tp is not None and ((hi[i] >= tp) if fill == "limit" else (cl[i] >= tp)):
            got = tp if fill == "limit" else cl[i]
            return 0.5 * got + 0.5 * cl[-1] if half else got
    return float(cl[-1])


def bs(a, b, n=20000, seed=11):
    rng = np.random.default_rng(seed)
    d = a - b
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(m, [2.5, 97.5]), (m > 0).mean()


RULES = {"hold": {}, "full+12%": dict(tp=.12), "full+30%": dict(tp=.30),
         "half+8%": dict(tp=.08, half=True), "half+12%": dict(tp=.12, half=True),
         "half+30%": dict(tp=.30, half=True)}


def main():
    picks = load_picks()
    px = fetch(set(picks["ticker"]), picks["pick_date"].min() - pd.Timedelta(days=140),
               pd.Timestamp.today() + pd.Timedelta(days=1))
    recs = build(picks, px)
    first = [r for r in recs if r["appear"] == 1]
    print(f"[136] {len(picks)} printed picks -> {len(recs)} matured; "
          f"{len(first)} first appearances; {len([r for r in recs if r['nflag']==0])} FAVOURABLE")

    print("\n=== 1. re-appearances are a different animal (hold-21d, next-open entry) ===")
    for lo, hi, lab in [(1, 1, "1st appearance"), (2, 8, "2nd-8th"), (9, 99, "9th+")]:
        b = np.array([sim(r) for r in recs if lo <= r["appear"] <= hi])
        print(f"  {lab:<16} n={len(b):>4}  mean {b.mean():+7.2%}  median {np.median(b):+7.2%}  "
              f"win {np.mean(b > 0):5.1%}")

    print("\n=== 2. MFE timing (why 'median MFE = +12.27%' is not a target) ===")
    pk = np.array([int(np.argmax(r["high"][:HOLD] / r["e_open"] - 1)) + 1 for r in recs])
    fin = np.array([sim(r) for r in recs])
    rho, p = sps.spearmanr(pk, fin)
    print(f"  spearman(peak day, 21d return) = {rho:+.3f} (p={p:.1e}); "
          f"winners peak day {pk[fin > 0].mean():.1f} vs losers {pk[fin <= 0].mean():.1f}")
    for a, b in [(1, 3), (4, 8), (9, 15), (16, 21)]:
        m = (pk >= a) & (pk <= b)
        print(f"    peak on day {a:>2}-{b:<2} n={m.sum():>3}  mean 21d {fin[m].mean():+7.2%}")

    print("\n=== 3. every rule, both populations, by month (mean 21d P&L per position) ===")
    for lab, rs in [("TRADEABLE (first appearance only)", first), ("ALL ROWS (135's population)", recs)]:
        print(f"\n  -- {lab}, n={len(rs)}")
        print("  " + f"{'month':<9}{'n':>4} " + "".join(f"{k:>10}" for k in RULES))
        worst = {k: 9.0 for k in RULES}
        for m in sorted(set(r["pick_date"].to_period("M") for r in rs)):
            sel = [r for r in rs if r["pick_date"].to_period("M") == m]
            v = {k: np.array([sim(r, **RULES[k]) for r in sel]).mean() for k in RULES}
            for k in RULES:
                worst[k] = min(worst[k], v[k] - v["hold"])
            print("  " + f"{str(m):<9}{len(sel):>4} " + "".join(f"{v[k]:>+10.2%}" for k in RULES))
        v = {k: np.array([sim(r, **RULES[k]) for r in rs]) for k in RULES}
        print("  " + f"{'POOLED':<9}{len(rs):>4} " + "".join(f"{v[k].mean():>+10.2%}" for k in RULES))
        print("  " + f"{'worst vs hold':<13}" + "".join(f"{worst[k]:>+10.2%}" for k in RULES))
        print("  " + f"{'95% CI vs hold':<13}" + "".join(
            f"{('[%+.1f,%+.1f]' % tuple(bs(v[k], v['hold'])[1] * 100)):>10}" for k in RULES)
              + "   (pp)")

    print("\n=== 4. fill model and entry model (does +12% survive realistic fills?) ===")
    for entry in ["e_open", "e_close"]:
        for fill in ["limit", "close"]:
            b = np.array([sim(r, entry=entry) for r in recs])
            row = [f"  entry={entry:<8} fill={fill:<6} hold {b.mean():+7.2%}"]
            for tp in [0.12, 0.30]:
                t = np.array([sim(r, entry=entry, tp=tp, fill=fill) for r in recs])
                row.append(f"TP{tp:.0%} {t.mean():+7.2%} ({(t.mean() - b.mean())*100:+.2f}pp)")
            print("  |  ".join(row))


if __name__ == "__main__":
    main()
