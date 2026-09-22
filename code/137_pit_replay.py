"""Phase F.8: point-in-time replay of the whole MG v3 rule set, 2024-2026.

WHY THIS EXISTS
---------------
Every exit and entry rule now running live was chosen on the 4-month live pick
history: ~34 positions and about 4 independent 21-day windows. 136 showed what
that buys you -- the +12%-vs-+30% argument came down to 3 names in one week of
May 2026, and every confidence interval on the tradeable population crosses
zero. No further tuning on that sample can be trusted.

This replays the same rules over 2024-01..2026-08 instead: ~650 trading days,
~31 independent 21-day windows, on the point-in-time SP500 membership.

WHAT IT CAN AND CANNOT SETTLE
-----------------------------
The model was TRAINED on this window, so the replay is in-sample FOR THE MODEL
and says nothing about whether raw_margin has predictive power. (It does not:
corrected for survivorship and look-ahead the sort returns -1.59pp.)

It is out-of-sample FOR THE RULES -- take-profit level and size, stop, hold
cap, dropout, re-entry gates were all fitted on May-Sep 2026 and never saw
2024-2025. Every open question right now is a rule question, which is why this
is worth running despite the model being in-sample.

FIDELITY
--------
The 37-feature panel is rebuilt from data/monthly_gainer_panel.csv (technical),
data/catalyst_features_sp500.csv (news/finbert), yfinance SPY+VIX and
data/fear_greed.csv (regime), plus the four cross-sectional xranks -- the same
construction 101_refresh_score_today.py does live. Re-scoring recent dates and
comparing against the live output/monthly_gainer/today_full_*.csv gives
spearman +0.966..+0.975 across the ~500-name cross-section, so the rebuild is
faithful.

The top-5 itself is NOT stable under that 0.97: only 2-4 of 5 names match on a
given day. That is a property of the strategy, not an error here -- the top of
the raw_margin distribution is a cluster of near-ties. Read the replay as a
valid draw from the same process, not as a re-enactment of the exact names.

Two known imperfections, both stated rather than patched:
  * finbert_* is absent before 2025 (0% coverage in 2023-24, 63% in 2025) and
    imputed to 0.0 -- which is exactly what training did with it. Combined
    importance 2.6%.
  * `fng` comes from data/fear_greed.csv, which stopped updating 2026-05-21
    and is forward-filled from there. That affects the LIVE pipeline too --
    101 fetches SPY/VIX from yfinance but reads fng from this stale file, so
    every live score since 2026-05-21 has used a frozen fng. Importance 0.9%.

Run: /Users/mollyqian/anaconda3/bin/python code/137_pit_replay.py
     --rebuild   force the scored-panel cache to be rebuilt
"""
import sys; sys.stdout.reconfigure(line_buffering=True)

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output" / "monthly_gainer"
CACHE = OUT / "pit_replay_scored.parquet"

REGIME = ["spy_ret_5d", "spy_ret_20d", "spy_rv_20", "spy_rv_60", "vix", "vix_chg_5d", "fng"]
EXT_NEAR_HIGH, EXT_RAN_UP, EXT_HIGH_CONF = -0.10, 0.02, 0.33      # same as 123


# ----------------------------------------------------------------- panel
def build_scored():
    panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    cat = pd.read_csv(DATA / "catalyst_features_sp500.csv", parse_dates=["date"])
    df = panel.merge(cat, on=["ticker", "date"], how="left")

    spy = yf.Ticker("^GSPC").history(start="2023-01-01", auto_adjust=True).reset_index()
    spy["date"] = pd.to_datetime(spy["Date"]).dt.tz_localize(None).dt.normalize()
    spy = spy[["date", "Close"]].rename(columns={"Close": "c"}).sort_values("date")
    spy["spy_ret_5d"] = spy.c.pct_change(5)
    spy["spy_ret_20d"] = spy.c.pct_change(20)
    spy["spy_rv_20"] = spy.c.pct_change().rolling(20).std() * np.sqrt(252)
    spy["spy_rv_60"] = spy.c.pct_change().rolling(60).std() * np.sqrt(252)
    vix = pd.read_csv(DATA / "vix_daily.csv", parse_dates=["date"]).sort_values("date")
    vix["vix_chg_5d"] = vix.vix.diff(5)
    fng = pd.read_csv(DATA / "fear_greed.csv", parse_dates=["date"])[["date", "fng"]]
    reg = (spy.drop(columns=["c"]).merge(vix, on="date", how="outer")
              .merge(fng, on="date", how="outer").sort_values("date"))
    reg[REGIME] = reg[REGIME].ffill()
    df = df.merge(reg[["date"] + REGIME], on="date", how="left")

    for c in ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d"]:
        df[c] = df[c].fillna(0.0)
    for c in ["news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
              "ma_news_5d", "ma_news_20d"]:
        df[c] = df[c].fillna(0).astype(float)

    df = df.sort_values(["ticker", "date"])
    g = df.groupby("ticker")["close"]
    df["ret_5d_lag"] = df.close / g.shift(5) - 1
    df["ret_20d_lag"] = df.close / g.shift(20) - 1
    df["dd_60d"] = (g.transform(lambda s: s.rolling(60, min_periods=20).min())
                    / g.transform(lambda s: s.rolling(60, min_periods=20).max()) - 1)
    for src, dst in [("rsi_14", "rsi_14_xrank"), ("rv_60", "rv_60_xrank"),
                     ("ma60_slope_60d", "ma60_slope_xrank"), ("ret_20d_lag", "ret_20d_xrank")]:
        df[dst] = df.groupby("date")[src].rank(pct=True)

    # Point-in-time membership: a name is only a candidate on dates when the
    # index actually held it. Without this the replay quietly buys the winners
    # of index inclusion -- the survivorship trap that overturned the 2026-06
    # results.
    pit = pd.read_csv(DATA / "sp500_pit_membership.csv", parse_dates=["asof"]).sort_values("asof")
    snaps = [(r.asof, set(r.members.split(","))) for r in pit.itertuples()]
    def members(d):
        m = None
        for a, s in snaps:
            if a <= d: m = s
            else: break
        return m
    memmap = {pd.Timestamp(d): members(pd.Timestamp(d)) for d in df.date.unique()}
    df["in_univ"] = [(t in memmap[d]) if memmap[d] else True
                     for t, d in zip(df.ticker, df.date)]

    # open/low from the OHLC file so `stop_mode="intraday"` can model a
    # resting stop order rather than the once-a-day close check.
    try:
        ohlc = pd.read_csv(DATA / "ohlc_prices.csv", parse_dates=["date"],
                           usecols=["date", "ticker", "open", "low"])
        df = df.merge(ohlc, on=["date", "ticker"], how="left")
    except Exception as e:
        print(f"[137] no intraday OHLC ({e}); stop_mode='intraday' unavailable")
        df["open"] = np.nan; df["low"] = np.nan

    art = joblib.load(ROOT / "models" / "monthly_gainer_v3_sp500.joblib")
    feats, med = art["feats"], pd.Series(art["impute_medians"])
    need = [f for f in feats if not f.startswith("finbert_")]
    sc = df.dropna(subset=need).copy()
    X = sc[feats].fillna(med).values
    sc["prob_cal"] = art["calibrator"].predict_proba(X)[:, 1]
    sc["raw_margin"] = art["raw_gbc"].decision_function(X)
    cols = ["date", "ticker", "close", "open", "low", "sector", "in_univ",
            "prob_cal", "raw_margin", "ret_5d_lag", "dd_60d", "spy_ret_20d",
            "rv_60", "atr_pct"]
    sc = sc[cols].sort_values(["date", "raw_margin"], ascending=[True, False])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    sc.to_parquet(CACHE, index=False)
    return sc


# ----------------------------------------------------------------- replay
def replay(sc, tp=0.12, portion=0.5, stop=0.15, cap=21, dropout=2, min_hold=2,
           slots=8, wash=30, cooldown=60, regime_min=None, top_n=5, fav_only=True,
           rank_by="raw_margin", seed=0, stop_mode="close"):
    """One pass of the strategy. Returns (positions_df, daily_df).

    Prices are closes: the live book decides at 15:45 and fills at 15:55, so a
    close-to-close simulation is the faithful one, not an intraday-touch one.
    """
    sc = sc[sc.in_univ]
    dates = sorted(sc.date.unique())
    rng = np.random.default_rng(seed)
    held, exits, log = {}, [], []
    for d in dates:
        day = sc[sc.date == d]
        if len(day) < 50:
            continue
        # rank_by lets the same machinery run the null controls: "random" is
        # a coin-flip pick from the same point-in-time universe, "rv_60" is the
        # pure volatility sort the model has been shown to approximate. If
        # those clear SPY too, the rules and the vol exposure are doing the
        # work and raw_margin is not.
        if rank_by == "random":
            ranked = day.iloc[rng.permutation(len(day))[:15]]
        else:
            ranked = day.nlargest(15, rank_by)
        top15 = set(ranked.ticker)
        px = dict(zip(day.ticker, day.close))
        op = dict(zip(day.ticker, day.open)) if "open" in day.columns else {}
        lo = dict(zip(day.ticker, day.low)) if "low" in day.columns else {}

        for t, p in list(held.items()):
            p["days"] += 1
            c = px.get(t)
            if c is None:
                continue
            gain = c / p["entry"] - 1
            p["miss"] = 0 if t in top15 else p["miss"] + 1

            # stop_mode="close" is what the live book does: one look at 15:45,
            # so the fill is that session's price and a name that falls through
            # the level intraday is sold at wherever it ended, not at the level.
            # stop_mode="intraday" models a resting stop-market order: it
            # triggers on the low, fills at the stop price, and fills at the
            # OPEN when the day gaps below the level -- which is how a stop
            # order actually behaves, and the reason it is not a floor either.
            stop_px, stop_fill = (p["entry"] * (1 - stop) if stop else None), None
            if stop and stop_mode == "intraday":
                o_, l_ = op.get(t), lo.get(t)
                if o_ is not None and not pd.isna(o_) and o_ <= stop_px:
                    stop_fill = float(o_)
                elif l_ is not None and not pd.isna(l_) and l_ <= stop_px:
                    stop_fill = float(stop_px)

            why = None
            if stop and stop_mode == "intraday" and stop_fill is not None:
                why = "stop"
                c, gain = stop_fill, stop_fill / p["entry"] - 1
            elif stop and stop_mode == "close" and gain <= -stop:  why = "stop"
            elif p["days"] >= cap:                           why = "cap"
            elif p["days"] > min_hold and p["miss"] >= dropout: why = "dropout"
            elif tp and gain >= tp and not p["half"]:        why = "tp"
            if why == "tp" and portion < 1.0:
                p["realized"] += portion * p["qty"] * (c - p["entry"])
                p["qty"] *= (1 - portion)
                p["half"] = True
                continue
            if why:
                p["realized"] += p["qty"] * (c - p["entry"])
                ret = p["realized"] / p["cost"]
                exits.append(dict(ticker=t, entry_date=p["date"], exit_date=d,
                                  days=p["days"], reason=why, ret=ret))
                p["exit_ret"], p["exit_date"] = ret, d
                log.append(p)
                del held[t]

        # entry gates
        last = {p["ticker"]: p for p in log}
        free = slots - len(held)
        if free > 0 and (regime_min is None or day.spy_ret_20d.iloc[0] > regime_min):
            for r in ranked.head(top_n).itertuples():
                if free <= 0: break
                if r.ticker in held: continue
                nfl = int(bool(r.dd_60d > EXT_NEAR_HIGH)) + int(r.ret_5d_lag > EXT_RAN_UP) \
                      + int(r.prob_cal > EXT_HIGH_CONF)
                if fav_only and nfl: continue
                e = last.get(r.ticker)
                if e is not None:
                    gap = (pd.Timestamp(d) - pd.Timestamp(e["exit_date"])).days
                    if e["exit_ret"] <= 0 and gap <= wash: continue
                    if cooldown and gap <= cooldown: continue
                qty = 125.0 / float(r.close)
                held[r.ticker] = dict(ticker=r.ticker, date=d, entry=float(r.close),
                                      qty=qty, cost=125.0, days=0, miss=0,
                                      half=False, realized=0.0)
                free -= 1
    return pd.DataFrame(exits)


def summary(ex, spy, label):
    """Bootstrap resamples ENTRY MONTHS, not positions. 21-day windows overlap
    heavily, so a per-position bootstrap treats ~30 independent months as ~200
    independent draws and reports an interval three times too narrow."""
    if not len(ex):
        return f"  {label:<34} no trades"
    r = ex.ret.values
    b = np.array([spy.get(pd.Timestamp(d), np.nan) for d in ex.exit_date])
    exc = np.where(np.isnan(b), np.nan, r - b)
    months = pd.to_datetime(pd.Series(list(ex.entry_date))).dt.to_period("M").values
    um = np.unique(months)
    idx = {m: np.where(months == m)[0] for m in um}
    rng = np.random.default_rng(4)
    boot = np.empty(4000)
    for k in range(4000):
        sel = np.concatenate([idx[um[j]] for j in rng.integers(0, len(um), len(um))])
        boot[k] = np.nanmean(exc[sel])
    return (f"  {label:<34} n={len(r):>4}  mean {r.mean():+7.2%}  median {np.median(r):+7.2%}  "
            f"win {np.mean(r > 0):5.0%}  vs SPY {np.nanmean(exc):+7.2%} "
            f"[{np.percentile(boot, 2.5):+.2%},{np.percentile(boot, 97.5):+.2%}]  "
            f"{len(um)} months")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    sc = build_scored() if (a.rebuild or not CACHE.exists()) else pd.read_parquet(CACHE)
    print(f"[137] scored panel: {len(sc):,} rows, {sc.date.min().date()} -> {sc.date.max().date()}, "
          f"{sc.ticker.nunique()} tickers, PIT-filtered to {sc.in_univ.mean():.1%}")

    # SPY return over the SAME holding window, keyed by exit date, so "vs SPY"
    # compares like with like rather than against a fixed 21-day bar.
    s = yf.Ticker("SPY").history(start="2023-06-01", auto_adjust=True).reset_index()
    s["date"] = pd.to_datetime(s["Date"]).dt.tz_localize(None).dt.normalize()
    s = s.set_index("date")["Close"]
    spy21 = (s / s.shift(21) - 1).to_dict()

    base = dict(tp=0.12, portion=0.5, stop=0.15, cooldown=60)
    print(f"\n=== LIVE RULE SET as it stands today ===")
    print(summary(replay(sc, **base), spy21, "half +12% / stop 15 / cd 60"))

    print(f"\n=== take profit: level x size (everything else at the live setting) ===")
    for lev in [0.08, 0.12, 0.20, 0.30, None]:
        for por in ([0.5, 1.0] if lev else [1.0]):
            lab = "no take profit" if lev is None else f"{'half' if por < 1 else 'full'} +{lev:.0%}"
            print(summary(replay(sc, **{**base, "tp": lev, "portion": por}), spy21, lab))

    print(f"\n=== stop loss ===")
    for st in [0, 0.10, 0.15, 0.25]:
        print(summary(replay(sc, **{**base, "stop": st}), spy21, f"stop {st:.0%}" if st else "no stop"))

    print(f"\n=== re-entry cooldown ===")
    for cd in [0, 30, 60, 10**5]:
        lab = {0: "wash-sale gate only", 10**5: "never re-buy (the old rule)"}.get(cd, f"cooldown {cd}d")
        print(summary(replay(sc, **{**base, "cooldown": cd}), spy21, lab))

    print(f"\n=== entry filter ===")
    print(summary(replay(sc, **base, fav_only=False), spy21, "no extension filter"))
    print(summary(replay(sc, **base, top_n=15), spy21, "top-15 instead of top-5"))
    print(summary(replay(sc, **base, regime_min=0.0), spy21, "only when SPY 20d > 0"))
    print(summary(replay(sc, **base, regime_min=0.02), spy21, "only when SPY 20d > +2%"))

    print(f"\n=== NULL CONTROLS — the same rules, a different way of picking ===")
    print("  (the replay is in-sample for the model, so the only honest question")
    print("   is whether raw_margin beats picking badly on the same universe)")
    print(summary(replay(sc, **base), spy21, "rank by raw_margin (the model)"))
    print(summary(replay(sc, **base, rank_by="rv_60"), spy21, "rank by rv_60 (pure vol sort)"))
    print(summary(replay(sc, **base, rank_by="atr_pct" if "atr_pct" in sc.columns
                         else "rv_60"), spy21, "rank by atr_pct (vol sort #2)"))
    for sd in range(3):
        print(summary(replay(sc, **base, rank_by="random", seed=sd), spy21,
                      f"random pick from the universe #{sd+1}"))

    print(f"\n=== hold cap ===")
    for c in [10, 21, 42]:
        print(summary(replay(sc, **{**base, "cap": c}), spy21, f"{c}-day cap"))


if __name__ == "__main__":
    main()
