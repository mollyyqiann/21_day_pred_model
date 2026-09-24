"""Pull latest bars for SP500 universe, append to panel, score with v3.

Goal: get fresh top-5 for the most recent trading day without rebuilding
the entire panel. Approach:

1. Load existing monthly_gainer_panel.csv (whatever date it currently ends).
2. Fetch 260 days of bars per ticker from yfinance -- CHUNKED, 60 tickers per
   request with a pause between chunks (see fetch_recent; 2026-09-23). 260
   because ma60_slope_60d first validates at bar 120, and the old 120-bar
   window meant that feature only ever existed on the final bar. A fetch
   covering <80% of the universe is a FAILED RUN (exit 1, nothing written).
3. Recompute all rolling features from the fresh fetch per ticker.
4. Recompute v8 features (rsi, macd, atr, etc.) for the appended rows.
5. Refresh SPY + VIX for regime; compute regime features.
6. Compute xrank cross-sectionally on the latest day.
7. Score with v3 model.
8. Output top-5 by raw_margin.

Run with system Python (has yfinance): /usr/bin/python3
"""

import os, sys; sys.stdout.reconfigure(line_buffering=True)

import warnings
import time
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ImportError:
    yf = None  # only needed for fetch stage

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output" / "monthly_gainer"

sys.path.insert(0, str(ROOT / "code"))
cat87 = import_module("87_catalyst_live_features")  # live catalyst-feature helpers


def compute_v8_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute v7+v8 features for one ticker. Expects sorted by date with
    columns: open, high, low, close, volume."""
    g = g.sort_values("date").reset_index(drop=True).copy()
    c = g["close"]
    o = g["open"]; h = g["high"]; l = g["low"]; v = g["volume"]
    r = c.pct_change()

    # RSI 14
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    g["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9) — normalized by close to match v3/v7 training (script 85's `line/x, signal/x, hist/x`)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig_line = macd_line.ewm(span=9, adjust=False).mean()
    g["macd"] = macd_line / c
    g["macd_sig"] = macd_sig_line / c
    g["macd_hist"] = (macd_line - macd_sig_line) / c

    # Bollinger z (20)
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    g["bb_z20"] = (c - ma20) / sd20

    # ATR (14)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    g["atr_pct"] = atr / c
    g["range_pct"] = (h - l) / c

    # Volume features
    vol_ma20 = v.rolling(20).mean()
    vol_std20 = v.rolling(20).std()
    g["vol_z"] = (v - vol_ma20) / vol_std20
    g["vol_5d"] = v / v.rolling(5).mean()

    # Realized vol 60d (annualized)
    g["rv_60"] = r.rolling(60).std() * np.sqrt(252)

    # Overnight gap (forward-looking; not used)
    g["overnight_gap"] = o.shift(-1) / c - 1.0

    # Trend features (v8)
    ma5 = c.rolling(5).mean()
    ma60 = c.rolling(60).mean()
    g["ma_stack"] = ((ma5 > ma20) & (ma20 > ma60)).astype(int)

    up = (r > 0).astype(int)
    grp = (up != up.shift()).cumsum()
    streak = up.groupby(grp).cumsum().where(up == 1, 0)
    g["up_streak"] = streak.clip(upper=30)

    g["up_bigdays_20d"] = (r > 0.03).rolling(20).sum()

    atr_abs = g["atr_pct"] * c
    g["dist_ma60_atr"] = (c - ma60) / atr_abs.replace(0, np.nan)
    g["ma60_slope_60d"] = (ma60 - ma60.shift(60)) / c

    above20 = (c > ma20).astype(int)
    grp2 = (above20 != above20.shift()).cumsum()
    run = above20.groupby(grp2).cumsum().where(above20 == 1, 0)
    g["run_length"] = run.clip(upper=120)

    # Lag returns for diagnostics
    g["close_5d_ago"] = c.shift(5)
    g["ret_5d_lag"] = c / g["close_5d_ago"] - 1.0
    g["close_20d_ago"] = c.shift(20)
    g["ret_20d_lag"] = c / g["close_20d_ago"] - 1.0

    # 60d max drawdown
    rmax = c.rolling(60, min_periods=20).max()
    rmin = c.rolling(60, min_periods=20).min()
    g["dd_60d"] = rmin / rmax - 1.0

    return g


def _normalize_ohlcv(df, tk):
    df = df.reset_index().rename(columns={"Date": "date", "Open": "open",
                                            "High": "high", "Low": "low",
                                            "Close": "close", "Volume": "volume"})
    df["ticker"] = tk
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def fetch_recent(tickers, days=120):
    """Chunked fetch (60/request + pause) + sequential retry for stragglers.

    The threaded yf.download contends on yfinance's SQLite cache (cookies.db,
    tkr-tz.db). On busy runs (esp. ~500 tickers) the cache hits intermittent
    "OperationalError: unable to open database file" and yfinance silently
    drops those tickers from the result. They succeed immediately on
    single-ticker sequential retry.
    """
    # Chunked, not one 500-ticker blast -- changed 2026-09-24 during the rate
    # limit spiral. Yahoo's limiter reacts to burst shape as much as to
    # volume: a single 501-name threaded download is exactly the profile that
    # trips it (and then the whole run dies at once). ~60 names per request
    # with a pause between chunks keeps the same total volume but a flat
    # profile, and a limiter that bites mid-run now costs one chunk, not the
    # day. MG_FETCH_CHUNK / MG_FETCH_PAUSE override.
    chunk = int(os.environ.get("MG_FETCH_CHUNK") or 60)
    pause = float(os.environ.get("MG_FETCH_PAUSE") or 12)
    n_chunks = (len(tickers) + chunk - 1) // chunk
    print(f"[101] yfinance chunked download: {len(tickers)} tickers, "
          f"{n_chunks} x {chunk}, {pause:.0f}s between chunks, {days}d window ...")
    t0 = time.time()
    rows = []
    missing = []
    for ci in range(n_chunks):
        batch = tickers[ci*chunk:(ci+1)*chunk]
        try:
            data = yf.download(batch, period=f"{days}d", interval="1d",
                               auto_adjust=True, threads=True, progress=False,
                               group_by="ticker")
        except Exception as e:
            print(f"[101]   chunk {ci+1}/{n_chunks}: download error {type(e).__name__}; "
                  f"marking {len(batch)} missing")
            missing.extend(batch)
            time.sleep(pause)
            continue
        got = 0
        for tk in batch:
            try:
                df = data[tk] if isinstance(data.columns, pd.MultiIndex) else data
            except Exception:
                missing.append(tk); continue
            df = df.dropna(subset=["Close"])
            if df.empty:
                missing.append(tk); continue
            rows.append(_normalize_ohlcv(df, tk))
            got += 1
        if got < len(batch):
            print(f"[101]   chunk {ci+1}/{n_chunks}: {got}/{len(batch)}")
        if ci + 1 < n_chunks:
            time.sleep(pause)
    print(f"[101] chunked download in {time.time()-t0:.0f}s")

    if missing:
        print(f"[101] retrying {len(missing)} bulk-failed tickers sequentially ...")
        t1 = time.time()
        recovered = 0
        still_missing = []
        for tk in missing:
            try:
                h = yf.Ticker(tk).history(period=f"{days}d", auto_adjust=True)
                if h.empty or h["Close"].notna().sum() == 0:
                    still_missing.append(tk); continue
                rows.append(_normalize_ohlcv(h.dropna(subset=["Close"]), tk))
                recovered += 1
            except Exception:
                still_missing.append(tk)
        print(f"[101] sequential retry recovered {recovered}/{len(missing)} "
              f"in {time.time()-t1:.0f}s")
        if still_missing:
            print(f"[101] still missing after retry ({len(still_missing)}): "
                  f"{still_missing[:15]}{'...' if len(still_missing) > 15 else ''}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fetch_spy_vix(days=400):
    print(f"[101] fetching SPY + VIX (last {days}d) ...")
    spy = yf.Ticker("^GSPC").history(period=f"{days}d", auto_adjust=True)
    vix = yf.Ticker("^VIX").history(period=f"{days}d", auto_adjust=True)
    spy = spy.reset_index().rename(columns={"Date": "date", "Close": "close"})
    spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None).dt.normalize()
    spy = spy[["date", "close"]].sort_values("date").reset_index(drop=True)
    spy["spy_ret_5d"] = spy["close"].pct_change(5)
    spy["spy_ret_20d"] = spy["close"].pct_change(20)
    spy["spy_rv_20"] = spy["close"].pct_change().rolling(20).std() * np.sqrt(252)
    spy["spy_rv_60"] = spy["close"].pct_change().rolling(60).std() * np.sqrt(252)
    spy = spy.drop(columns=["close"])

    vix = vix.reset_index().rename(columns={"Date": "date", "Close": "vix"})
    vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None).dt.normalize()
    vix = vix[["date", "vix"]].sort_values("date").reset_index(drop=True)
    vix["vix_chg_5d"] = vix["vix"].diff(5)

    return spy.merge(vix, on="date", how="outer")




# Historical outcome by extension band, measured on 253 matured MG v3 top-5 picks
# (2026-05-03..2026-07-27, 21-trading-day hold, re-scored 2026-08-26).
# dd_60d = close / 60-day-high - 1, i.e. how far below its recent peak the name is.
EXTENSION_BANDS = [
    (-1.00, -0.25, ">25% below high",   +0.16, 53),
    (-0.25, -0.15, "15-25% below",      +1.00, 41),
    (-0.15, -0.08, "8-15% below",       -6.29, 31),
    (-0.08, -0.03, "3-8% below",        -9.22, 20),
    (-0.03,  1.00, "within 3% of high", -10.65, 27),
]


def extension_indicator(scored: pd.DataFrame, k: int = 5) -> str:
    """Flag how EXTENDED each pick is -- the strongest outcome separator found.

    Investigation 2026-08-26: across 253 matured top-5 picks, distance from the
    60-day high separated winners from losers better than anything else tested,
    including the model's own probability:

        >25% below high   +0.16%   win 53%
        15-25% below      +1.00%   win 41%
        8-15% below       -6.29%   win 31%
        3-8% below        -9.22%   win 20%
        within 3% of high -10.65%  win 27%

        spearman(dd_60d, 21d return) = -0.351, p<1e-5, and the sign held in
        ALL THREE months tested -- the only candidate that was sign-stable.

    Why it works: the model is a volatility/momentum ranker (71.6% of its feature
    importance is volatility), so it systematically buys names that have already
    run. Those mean-revert. `raw_margin` and `prob_cal` do NOT separate outcomes
    -- prob_cal is INVERTED (highest-confidence quintile returned -10.54% and
    touched +30% only 3.9% of the time, vs +2.61% / 17.6% for the lowest).

    CAVEATS -- read before acting:
      * 253 picks over 3 months. The SIGN is stable; the magnitudes will drift.
      * This was selected by searching ~20 features on this same sample, so the
        numbers are optimistic. Treat the band ORDERING as the signal.
      * It only discriminates when the list actually spreads across bands. When
        every pick is deeply drawn down (as on 2026-08-26), it says nothing.
      * It does NOT make the model profitable. It separates bad from less-bad.
    """
    if "dd_60d" not in scored.columns:
        return "[101] extension indicator: unavailable (dd_60d missing)"
    top = scored.nlargest(k, "raw_margin")
    lines = [f"[101] === EXTENSION INDICATOR (top-{k}) ===",
             "[101]   how far below its 60-day high each pick is; historically the",
             "[101]   strongest separator of good picks from bad."]
    for r in top.itertuples():
        dd = getattr(r, "dd_60d", float("nan"))
        band, hist, win = "n/a", float("nan"), float("nan")
        for lo, hi, name, h, w in EXTENSION_BANDS:
            if lo <= dd < hi:
                band, hist, win = name, h, w
                break
        mark = " <<" if hist == hist and hist < -5 else ""
        lines.append(f"[101]   {r.ticker:<6} {dd*100:+6.1f}%  {band:<18} "
                     f"hist {hist:+6.2f}%  win {win:3.0f}%{mark}")
    bands = []
    for r in top.itertuples():
        dd = getattr(r, "dd_60d", float("nan"))
        for lo, hi, name, h, w in EXTENSION_BANDS:
            if lo <= dd < hi:
                bands.append(name); break
    if len(set(bands)) <= 1 and bands:
        lines.append(f"[101]   >> all {k} picks are in the same band ({bands[0]}) -- "
                     "the indicator cannot discriminate today.")
    n_bad = sum(1 for b in bands if b in ("3-8% below", "within 3% of high"))
    if n_bad:
        lines.append(f"[101]   >> {n_bad} of {k} are near their highs -- historically the "
                     "worst band (-9% to -11%, win 20-27%).")
    return "\n".join(lines)



# --- extension indicator -------------------------------------------------
# Three entry-time flags, each independently significant on 253 matured
# top-5 picks (2026-05..07). All three say the same thing: the model buys
# names that have already moved, and those do worst.
#
#   flag                     flagged      clean      p
#   within 10% of 60d high   -9.07%      -1.70%   0.00087
#   up >2% over prior 5d     -8.41%      -2.34%   0.00611
#   prob_cal > 0.33         -11.15%      -1.96%   0.00001   <- INVERTED
#
# Score = number of flags. Outcome is monotone in the score
# (spearman -0.408, p<1e-6):
#   0 flags  +2.23%  win 55%      2 flags  -6.37%  win 29%
#   1 flag   -3.56%  win 36%      3 flags -19.93%  win  6%
#
# Note the third flag: the model's OWN confidence is anti-predictive. Its
# high-conviction picks touch +30% 3.9% of the time vs 17.6% for its
# low-conviction ones. This is an INDICATOR ONLY -- it does not re-rank.
EXT_NEAR_HIGH = -0.10     # dd_60d above this = close to the 60-day high
EXT_RAN_UP    =  0.02     # ret_5d_lag above this = already popped
EXT_HIGH_CONF =  0.33     # prob_cal above this = model over-confident


def extension_flags(row):
    """Return (score, [reasons]) for one scored row. Higher = worse."""
    f = []
    dd = row.get("dd_60d")
    if dd is not None and pd.notna(dd) and dd > EXT_NEAR_HIGH:
        f.append("near 60d high")
    r5 = row.get("ret_5d_lag")
    if r5 is not None and pd.notna(r5) and r5 > EXT_RAN_UP:
        f.append("up >2% in 5d")
    pc = row.get("prob_cal")
    if pc is not None and pd.notna(pc) and pc > EXT_HIGH_CONF:
        f.append("high model conf")
    return len(f), f


def extension_band(score):
    return "FAVOURABLE" if score == 0 else ("NEUTRAL" if score == 1 else "EXTENDED")


def extension_report(top, k=5):
    """Per-pick indicator lines for the top-k. Does NOT change the ranking."""
    lines = ["[101] === EXTENSION INDICATOR (does not re-rank) ===",
             "[101]   historical 21d return by band: "
             "FAVOURABLE +2.23% (win 55%) | NEUTRAL -3.56% (36%) | EXTENDED -9.87% (23%)"]
    for r in top.head(k).to_dict("records"):
        sc, why = extension_flags(r)
        band = extension_band(sc)
        mark = {"FAVOURABLE": "++", "NEUTRAL": " ~", "EXTENDED": "--"}[band]
        lines.append(f"[101]   {mark} {r['ticker']:<6} {band:<11} "
                     f"({sc}/3)" + (f"  flags: {', '.join(why)}" if why else ""))
    return "\n".join(lines)


def vol_exposure_report(scored: pd.DataFrame, k: int = 5) -> str:
    """Surface the volatility-factor exposure the picks are carrying.

    Why this exists (investigation 2026-08-25): MG v3's label is a fixed +30%
    barrier, which is a ~4.9-sigma move for a low-vol name but only ~2.8-sigma
    for a high-vol one. That makes volatility the optimal predictor -- 71.6% of
    the model's feature importance sits on vol features, and the prediction
    decile maps monotonically onto vol quintile (0.20 -> 3.89). Out of sample
    the model is matched by `df.nlargest(5,'rv_60')` and BEATEN by
    `df.nlargest(5,'atr_pct')`, and forcing vol-neutral selection removes ~84%
    of its apparent edge.

    So the daily top-5 is, in substance, a leveraged long position in the
    high-volatility factor. That bet was never chosen and was invisible in this
    output. Printing it does not fix the model -- it just stops the exposure
    being hidden.
    """
    if "rv_60" not in scored.columns or scored["rv_60"].notna().sum() < 50:
        return "[101] vol exposure: unavailable (rv_60 missing)"
    d = scored.dropna(subset=["rv_60"]).copy()
    d["vol_q"] = pd.qcut(d["rv_60"].rank(method="first"), 5, labels=False)
    sel = d.nlargest(k, "raw_margin")
    mean_q = float(sel["vol_q"].mean())
    top2 = float((sel["vol_q"] >= 3).mean())
    lines = [
        f"[101] === VOLATILITY EXPOSURE OF TOP-{k} ===",
        f"[101]   mean vol quintile : {mean_q:.2f}   (2.00 = vol-neutral, 4.00 = only the most volatile)",
        f"[101]   share in top-2 vol quintiles: {top2:.0%}   (40% would be neutral)",
        f"[101]   median rv_60 picks {sel['rv_60'].median():.1%} vs universe {d['rv_60'].median():.1%}",
    ]
    if mean_q >= 3.0:
        lines.append("[101]   >> These picks are a LONG-VOLATILITY FACTOR BET, not a "
                     "stock-selection signal.")
        lines.append("[101]   >> Out-of-sample this model is beaten by df.nlargest(5,'atr_pct'). "
                     "Size accordingly.")
    return "\n".join(lines)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    cache_path = DATA / "_today_fresh_features.csv"

    if mode in ("all", "fetch"):
        run_fetch_and_features(cache_path)
        if mode == "fetch":
            return
    run_score(cache_path)


def run_fetch_and_features(cache_path):
    print("[101] loading existing panel ...")
    old_panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    print(f"[101] panel: {len(old_panel):,} rows, ends {old_panel['date'].max().date()}")
    tickers = sorted(old_panel["ticker"].unique().tolist())
    print(f"[101] tickers: {len(tickers)}")

    # Sector lookup from existing panel (sector is constant per ticker)
    sec_map = old_panel.dropna(subset=["sector"]).groupby("ticker")["sector"].first().to_dict()

    # Fetch fresh OHLCV
    # 260, not 120 -- fixed 2026-09-23. ma60_slope_60d needs 120 trading bars
    # (rolling(60).mean() then .shift(60)), and a 120d fetch delivers ~120 bars
    # AT BEST: the slope was only ever valid on the final bar, so any ticker
    # missing a single bar lost the feature entirely. On 09-23 that dropped
    # 441/501 names out of "candidates with full features" and the day was
    # ranked inside the surviving 60. With 260d there are ~110 bars of margin.
    fresh = fetch_recent(tickers, days=260)
    # Coverage floor -- added 2026-09-23, the 60-row day. The sequential retry
    # above can itself be rate-limited, and this script used to shrug, score
    # whatever survived, and exit 0: downstream then ranked a "top-15" inside
    # 60 names and nearly sold two healthy holdings as "out of the top-15".
    # A mostly-failed fetch is a FAILED RUN: exit non-zero, write nothing, and
    # let the caller (123's retry loop) back off and try again or stand down.
    _got = fresh["ticker"].nunique() if not fresh.empty and "ticker" in fresh.columns else 0
    _cov = _got / max(1, len(tickers))
    _floor = float(os.environ.get("MG_MIN_FETCH_COVER") or 0.8)
    if _cov < _floor:
        print(f"[101] FATAL: fetched only {_got}/{len(tickers)} tickers ({_cov:.0%} < {_floor:.0%}); "
              f"refusing to score a partial universe")
        sys.exit(1)
    if fresh.empty:
        print("[101] yfinance returned nothing; abort")
        return
    fresh_max = fresh["date"].max()
    print(f"[101] fresh data covers up to {fresh_max.date()}")
    if fresh_max <= old_panel["date"].max():
        print(f"[101] no new dates beyond {old_panel['date'].max().date()}; nothing to add")
        return

    # Compute features per ticker using fresh data only (re-derive everything for safety)
    print("[101] recomputing v8 features per ticker ...")
    out = []
    t0 = time.time()
    for i, (tk, g) in enumerate(fresh.groupby("ticker", sort=False)):
        if i and i % 100 == 0:
            print(f"   ... {i}/{len(tickers)} ({time.time()-t0:.0f}s)")
        gg = compute_v8_features(g)
        gg["sector"] = sec_map.get(tk, "")
        out.append(gg)
    new_panel = pd.concat(out, ignore_index=True)

    # Merge regime
    spy_vix = fetch_spy_vix()
    fng_path = DATA / "fear_greed.csv"
    if fng_path.exists():
        fng = pd.read_csv(fng_path, parse_dates=["date"])[["date", "fng"]]
        regime = spy_vix.merge(fng, on="date", how="left")
    else:
        regime = spy_vix.copy()
        regime["fng"] = np.nan
    regime = regime.sort_values("date").reset_index(drop=True)
    REGIME_FEATS = ["spy_ret_5d", "spy_ret_20d", "spy_rv_20", "spy_rv_60",
                    "vix", "vix_chg_5d", "fng"]
    for c in REGIME_FEATS:
        if c not in regime.columns:
            regime[c] = np.nan
    regime[REGIME_FEATS] = regime[REGIME_FEATS].ffill()

    new_panel = new_panel.merge(regime[["date"] + REGIME_FEATS], on="date", how="left")

    # Live catalyst features (finbert sentiment, news volume, earnings/M&A
    # keyword flags, sector pop) — reuses the same rolling-window logic as
    # the offline panel builder (87_catalyst_live_features.py), computed
    # over the full fetched history (not yet truncated to last 10d) so the
    # 5d/20d rolling windows have enough lookback.
    print("[101] computing live catalyst features ...")
    fb_path = DATA / "finbert_scores.csv"
    if fb_path.exists():
        fb_full = pd.read_csv(fb_path, parse_dates=["date"])
    else:
        fb_full = pd.DataFrame(columns=["ticker", "date", "finbert_max", "finbert_mean", "finbert_n"])

    p = new_panel[["ticker", "date", "close", "sector"]].sort_values(["ticker", "date"]).copy()
    p["close_t-1"] = p.groupby("ticker")["close"].shift(1)
    p["close_t-6"] = p.groupby("ticker")["close"].shift(6)
    p["ret_5d_lag_pop"] = p["close_t-1"] / p["close_t-6"] - 1.0
    p["pop_flag"] = (p["ret_5d_lag_pop"] >= 0.10).astype(int)
    sec_pop = p.groupby(["sector", "date"])["pop_flag"].sum().reset_index()
    sec_pop.columns = ["sector", "date", "sector_pop_5d"]
    new_panel = new_panel.merge(sec_pop, on=["sector", "date"], how="left")
    new_panel["sector_pop_5d"] = new_panel["sector_pop_5d"].fillna(0).astype(int)

    news_rows = []
    for tk, sub in new_panel.groupby("ticker", sort=False):
        sub_dates = sub[["date"]].copy()
        fb_t = fb_full[fb_full["ticker"] == tk][["date", "finbert_max", "finbert_mean", "finbert_n"]]
        kw_t = cat87.parse_news_keywords(tk)
        feat = cat87.compute_per_ticker_news_features(sub_dates, fb_t, kw_t)
        feat["ticker"] = tk
        news_rows.append(feat)
    news_feats = pd.concat(news_rows, ignore_index=True)
    new_panel = new_panel.merge(news_feats, on=["ticker", "date"], how="left")

    # Match training-time imputation (93_train_v3.py): finbert_* -> 0.0, counts/flags -> 0
    for c in ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d"]:
        new_panel[c] = new_panel[c].fillna(0.0)
    for c in ["news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
              "ma_news_5d", "ma_news_20d"]:
        new_panel[c] = new_panel[c].fillna(0).astype(float)

    # XRank features cross-sectional per date
    new_panel["rsi_14_xrank"] = new_panel.groupby("date")["rsi_14"].rank(pct=True)
    new_panel["rv_60_xrank"] = new_panel.groupby("date")["rv_60"].rank(pct=True)
    new_panel["ma60_slope_xrank"] = new_panel.groupby("date")["ma60_slope_60d"].rank(pct=True)
    new_panel["ret_20d_xrank"] = new_panel.groupby("date")["ret_20d_lag"].rank(pct=True)

    # Save features for downstream scoring stage (only last 5 days per ticker — saves disk)
    last_d = new_panel["date"].max()
    keep = new_panel[new_panel["date"] >= (last_d - pd.Timedelta(days=10))].copy()
    _tmp = cache_path.with_suffix(f".tmp{os.getpid()}")
    keep.to_csv(_tmp, index=False); os.replace(_tmp, cache_path)
    print(f"[101] cached features at {cache_path} ({len(keep):,} rows, last 10d)")


def run_score(cache_path):
    new_panel = pd.read_csv(cache_path, parse_dates=["date"])
    print(f"[101] loaded cached features: {len(new_panel):,} rows up to {new_panel['date'].max().date()}")

    # Score
    print("[101] scoring with v3 ...")
    art = joblib.load(MODELS / "monthly_gainer_v3_sp500.joblib")
    feats = art["feats"]
    med = pd.Series(art["impute_medians"])
    cal = art["calibrator"]
    gbc = art["raw_gbc"]

    scor = new_panel.dropna(subset=feats).copy()
    X = scor[feats].fillna(med).values
    scor["prob_cal"] = cal.predict_proba(X)[:, 1]
    scor["raw_margin"] = gbc.decision_function(X)
    last_d = scor["date"].max()
    today = scor[scor["date"] == last_d].copy()

    spy_20d_today = today["spy_ret_20d"].iloc[0] if len(today) else float("nan")
    print(f"\n[101] today = {last_d.date()}  |  SPY 20d = {spy_20d_today:+.1%}  |  "
          f"regime_on = {spy_20d_today > 0}")
    print(f"[101] candidates with full features: {len(today)}")

    # Entry-timing flag (NOT a ranking penalty). Backtest shows the EXTREME
    # tag (deep drawdown + high vol) has the BEST top-5 hit rate of any tag
    # (37%, vs 30% FRESH / 24% CATALYST) — see 102_sunday_check.py's
    # TAG_HITRATE_TOP5 and feedback_extreme_tag_is_best memory. So a still-falling
    # name is NOT excluded or down-ranked here; the ranking is left as-is.
    # What this flags instead is entry timing: a name still down 5%+ over 5d
    # has no confirmed intraday bottom yet, so buying the exact print you see
    # here risks entering mid-drop ("半山腰"). Wait for a flat/green print
    # before entering; don't chase a deeper drop.
    today["entry_note"] = np.where(
        today["ret_5d_lag"] <= -0.05,
        "wait for stabilization (down 5%+/5d, no confirmed bottom yet)", "")

    print("\n[101] === TOP 15 by raw_margin ===")
    top15 = today.nlargest(15, "raw_margin").copy()
    print(top15[["ticker", "sector", "raw_margin", "prob_cal", "close",
                  "ret_5d_lag", "ret_20d_lag", "atr_pct", "dd_60d", "run_length", "entry_note"]]
          .to_string(index=False, formatters={
              "raw_margin": "{:+.2f}".format, "prob_cal": "{:.3f}".format,
              "close": "${:.2f}".format, "ret_5d_lag": "{:+.1%}".format,
              "ret_20d_lag": "{:+.1%}".format, "atr_pct": "{:.1%}".format,
              "dd_60d": "{:.0%}".format, "run_length": "{:.0f}".format,
          }))

    print("\n[101] === TOP 5 (Option 1B entries) ===")
    top5 = today.nlargest(5, "raw_margin")[["ticker", "sector", "raw_margin", "prob_cal",
                                              "close", "ret_5d_lag", "ret_20d_lag",
                                              "atr_pct", "dd_60d", "run_length", "entry_note"]]
    print(top5.to_string(index=False, formatters={
        "raw_margin": "{:+.2f}".format, "prob_cal": "{:.3f}".format,
        "close": "${:.2f}".format, "ret_5d_lag": "{:+.1%}".format,
        "ret_20d_lag": "{:+.1%}".format, "atr_pct": "{:.1%}".format,
        "dd_60d": "{:.0%}".format, "run_length": "{:.0f}".format,
    }))

    print()
    print(extension_report(today.nlargest(15, "raw_margin"), k=5))
    print()
    print(vol_exposure_report(today, k=5))
    print()
    print(extension_indicator(today, k=5))

    OUT.mkdir(parents=True, exist_ok=True)
    # Atomic: on 2026-09-23 a mid-download crash left a 60-row partial here,
    # and downstream ranked the day inside it. tmp+replace means the file is
    # either yesterday's complete score or today's complete score, never a torso.
    _out = OUT / "today_score_fresh_sp500.csv"
    _tmp = _out.with_suffix(f".tmp{os.getpid()}")
    today.to_csv(_tmp, index=False); os.replace(_tmp, _out)
    print(f"\n[101] saved {OUT / 'today_score_fresh_sp500.csv'}")


if __name__ == "__main__":
    main()
# Usage:
#   /usr/bin/python3 code/101_refresh_score_today.py fetch
#   /Users/mollyqian/anaconda3/bin/python code/101_refresh_score_today.py score
