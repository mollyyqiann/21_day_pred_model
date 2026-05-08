"""Pull latest bars for SP500 universe, append to panel, score with v3.

Goal: get fresh top-5 for the most recent trading day without rebuilding
the entire panel. Approach:

1. Load existing monthly_gainer_panel.csv (ends 2026-04-30).
2. yfinance.download SP500 tickers for last ~80 trading days (need lookback
   for rolling features).
3. For each ticker, replace last ~70 days of panel with fresh yfinance data
   (in case anything was off) and append new days.
4. Recompute v8 features (rsi, macd, atr, etc.) for the appended rows.
5. Refresh SPY + VIX for regime; compute regime features.
6. Compute xrank cross-sectionally on the latest day.
7. Score with v3 model.
8. Output top-5 by raw_margin.

Run with system Python (has yfinance): /usr/bin/python3
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import warnings
import time
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


def fetch_recent(tickers, days=120):
    print(f"[101] yfinance bulk download: {len(tickers)} tickers, {days}d window ...")
    t0 = time.time()
    data = yf.download(tickers, period=f"{days}d", interval="1d",
                        auto_adjust=True, threads=True, progress=False, group_by="ticker")
    print(f"[101] downloaded in {time.time()-t0:.0f}s")
    rows = []
    for tk in tickers:
        try:
            df = data[tk] if isinstance(data.columns, pd.MultiIndex) else data
        except Exception:
            continue
        df = df.dropna(subset=["Close"])
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"Date": "date", "Open": "open",
                                                "High": "high", "Low": "low",
                                                "Close": "close", "Volume": "volume"})
        df["ticker"] = tk
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        rows.append(df[["date", "ticker", "open", "high", "low", "close", "volume"]])
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
    fresh = fetch_recent(tickers, days=120)
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

    # Catalyst features: set to 0 (no fresh news pull)
    CATALYST = ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
                "news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
                "ma_news_5d", "ma_news_20d", "sector_pop_5d"]
    for c in CATALYST:
        new_panel[c] = 0.0

    # XRank features cross-sectional per date
    new_panel["rsi_14_xrank"] = new_panel.groupby("date")["rsi_14"].rank(pct=True)
    new_panel["rv_60_xrank"] = new_panel.groupby("date")["rv_60"].rank(pct=True)
    new_panel["ma60_slope_xrank"] = new_panel.groupby("date")["ma60_slope_60d"].rank(pct=True)
    new_panel["ret_20d_xrank"] = new_panel.groupby("date")["ret_20d_lag"].rank(pct=True)

    # Save features for downstream scoring stage (only last 5 days per ticker — saves disk)
    last_d = new_panel["date"].max()
    keep = new_panel[new_panel["date"] >= (last_d - pd.Timedelta(days=10))].copy()
    keep.to_csv(cache_path, index=False)
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

    print("\n[101] === TOP 15 by raw_margin ===")
    top15 = today.nlargest(15, "raw_margin").copy()
    print(top15[["ticker", "sector", "raw_margin", "prob_cal", "close",
                  "ret_5d_lag", "ret_20d_lag", "atr_pct", "dd_60d", "run_length"]]
          .to_string(index=False, formatters={
              "raw_margin": "{:+.2f}".format, "prob_cal": "{:.3f}".format,
              "close": "${:.2f}".format, "ret_5d_lag": "{:+.1%}".format,
              "ret_20d_lag": "{:+.1%}".format, "atr_pct": "{:.1%}".format,
              "dd_60d": "{:.0%}".format, "run_length": "{:.0f}".format,
          }))

    print("\n[101] === TOP 5 (Option 1B entries) ===")
    top5 = today.nlargest(5, "raw_margin")[["ticker", "sector", "raw_margin", "prob_cal",
                                              "close", "ret_5d_lag", "ret_20d_lag",
                                              "atr_pct", "dd_60d", "run_length"]]
    print(top5.to_string(index=False, formatters={
        "raw_margin": "{:+.2f}".format, "prob_cal": "{:.3f}".format,
        "close": "${:.2f}".format, "ret_5d_lag": "{:+.1%}".format,
        "ret_20d_lag": "{:+.1%}".format, "atr_pct": "{:.1%}".format,
        "dd_60d": "{:.0%}".format, "run_length": "{:.0f}".format,
    }))

    OUT.mkdir(parents=True, exist_ok=True)
    today.to_csv(OUT / "today_score_fresh_sp500.csv", index=False)
    print(f"\n[101] saved {OUT / 'today_score_fresh_sp500.csv'}")


if __name__ == "__main__":
    main()
# Usage:
#   /usr/bin/python3 code/101_refresh_score_today.py fetch
#   /Users/mollyqian/anaconda3/bin/python code/101_refresh_score_today.py score
