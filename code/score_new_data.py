"""End-to-end example: pull OHLCV from yfinance, compute the 37 features the
v3 model expects, and score new data.

This is a self-contained demo. Run as:

    pip install yfinance pandas numpy scikit-learn joblib requests
    python code/score_new_data.py

Outputs a ranked table of P(touch +30% in next 21 trading days) for the
example universe.

Defaults:
  - Universe: ~50 liquid US stocks (mix of large-cap + smallcap + meme)
  - Catalyst features: all set to 0 (no news pipeline) — see data_schema.md §C
  - Fear & Greed: skipped, model imputes median (51.4)
  - Model: monthly_gainer_v3_combined.joblib

Edit TICKERS or MODEL below to change.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "monthly_gainer_v3_combined.joblib"

# Example universe — replace with your own. ~50 tickers is enough for the
# cross-sectional rank features to be meaningful.
TICKERS = [
    # large-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "MU",
    # high-vol / momentum
    "PLTR", "COIN", "HOOD", "SMCI", "APP", "CVNA", "MSTR", "AFRM", "RIVN", "LCID",
    # semis / hardware
    "AMAT", "LRCX", "KLAC", "MRVL", "ON", "ARM", "TSM", "AVGO", "QCOM", "ASML",
    # software / SaaS
    "CRM", "NOW", "SNOW", "DDOG", "NET", "MDB", "ZS", "OKTA", "TEAM", "PANW",
    # biotech / pharma
    "MRNA", "BNTX", "GILD", "VRTX", "REGN", "BIIB",
    # other
    "F", "GM", "BA", "DIS",
]

HISTORY_PERIOD = "2y"  # need ≥120 trading days for warmup; 2y is plenty


# ============================================================================
# Feature builders — copied from code/30_burst_v7_pipeline.py +
# code/35_burst_v8_trend_features.py +  code/regime_features.py +
# code/93_train_v3.py to keep this script self-contained.
# ============================================================================


def rsi(x: pd.Series, n: int = 14) -> pd.Series:
    d = x.diff()
    u = d.clip(lower=0).rolling(n).mean()
    dd = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + u / dd.replace(0, np.nan))


def macd(x: pd.Series, f: int = 12, s: int = 26, sig: int = 9):
    ef = x.ewm(span=f, adjust=False).mean()
    es = x.ewm(span=s, adjust=False).mean()
    line = ef - es
    signal = line.ewm(span=sig, adjust=False).mean()
    return line / x, signal / x, (line - signal) / x


def bb_z(x: pd.Series, n: int = 20) -> pd.Series:
    return (x - x.rolling(n).mean()) / x.rolling(n).std()


def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 16 base features for one ticker. df must have OHLCV columns."""
    out = pd.DataFrame(index=df.index)
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    out["rsi_14"] = rsi(c)
    ml, ms, mh = macd(c)
    out["macd"], out["macd_sig"], out["macd_hist"] = ml, ms, mh
    out["bb_z20"] = bb_z(c)
    atr14 = atr(h, l, c, 14)
    out["atr_pct"] = atr14 / c
    out["range_pct"] = (h - l) / c
    vm = v.rolling(30).mean()
    vs = v.rolling(30).std().replace(0, np.nan)
    out["vol_z"] = (v - vm) / vs
    out["vol_5d"] = v.rolling(5).mean() / vm
    r = c.pct_change()
    out["rv_60"] = r.rolling(60).std() * math.sqrt(252)

    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    out["ma_stack"] = ((ma5 > ma20) & (ma20 > ma60)).astype(int)
    up = (r > 0).astype(int)
    grp = (up != up.shift()).cumsum()
    out["up_streak"] = up.groupby(grp).cumsum().where(up == 1, 0).clip(upper=30)
    out["up_bigdays_20d"] = (r > 0.03).rolling(20).sum()
    out["dist_ma60_atr"] = (c - ma60) / atr14.replace(0, np.nan)
    out["ma60_slope_60d"] = (ma60 - ma60.shift(60)) / c
    above20 = (c > ma20).astype(int)
    grp2 = (above20 != above20.shift()).cumsum()
    out["run_length"] = above20.groupby(grp2).cumsum().where(above20 == 1, 0).clip(upper=120)

    return out


def build_regime_frame(period: str = HISTORY_PERIOD) -> pd.DataFrame:
    """Pull SPY + ^VIX, compute 6 of the 7 regime features (fng skipped → imputed)."""
    spy = yf.Ticker("SPY").history(period=period, auto_adjust=False)[["Close"]]
    spy.columns = ["close"]
    spy = spy.reset_index().rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None).dt.normalize()
    spy["spy_ret_5d"] = spy["close"].pct_change(5)
    spy["spy_ret_20d"] = spy["close"].pct_change(20)
    spy["spy_rv_20"] = spy["close"].pct_change().rolling(20).std() * np.sqrt(252)
    spy["spy_rv_60"] = spy["close"].pct_change().rolling(60).std() * np.sqrt(252)
    spy = spy[["date", "spy_ret_5d", "spy_ret_20d", "spy_rv_20", "spy_rv_60"]]

    vix = yf.Ticker("^VIX").history(period=period, auto_adjust=False)[["Close"]]
    vix.columns = ["vix"]
    vix = vix.reset_index().rename(columns={"Date": "date"})
    vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None).dt.normalize()
    vix["vix_chg_5d"] = vix["vix"].diff(5)
    vix = vix[["date", "vix", "vix_chg_5d"]]

    out = spy.merge(vix, on="date", how="outer").sort_values("date").reset_index(drop=True)
    out["fng"] = np.nan  # imputed downstream from bundle["impute_medians"]
    return out


def add_xrank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional percentile ranks of 4 features."""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["close_20d_ago"] = df.groupby("ticker")["close"].shift(20)
    df["ret_20d_lag"] = df["close"] / df["close_20d_ago"] - 1.0
    df["rsi_14_xrank"] = df.groupby("date")["rsi_14"].rank(pct=True)
    df["rv_60_xrank"] = df.groupby("date")["rv_60"].rank(pct=True)
    df["ma60_slope_xrank"] = df.groupby("date")["ma60_slope_60d"].rank(pct=True)
    df["ret_20d_xrank"] = df.groupby("date")["ret_20d_lag"].rank(pct=True)
    return df


def add_catalyst_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Default all 10 catalyst features to 0 (matches model's training-time medians)."""
    for c in [
        "finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
        "news_n_5d", "news_n_20d",
        "earn_news_5d", "earn_news_20d",
        "ma_news_5d", "ma_news_20d",
        "sector_pop_5d",
    ]:
        df[c] = 0
    return df


# ============================================================================
# Main: pull data, build features, score
# ============================================================================


def main() -> int:
    print(f"[1/5] Loading model bundle from {MODEL_PATH.relative_to(ROOT)} ...")
    bundle = joblib.load(MODEL_PATH)
    feats = bundle["feats"]
    medians = bundle["impute_medians"]
    print(f"      universe={bundle['universe']}, n_features={len(feats)}")

    print(f"[2/5] Bulk downloading OHLCV ({HISTORY_PERIOD}) for {len(TICKERS)} tickers ...")
    raw = yf.download(TICKERS, period=HISTORY_PERIOD, interval="1d",
                      auto_adjust=False, progress=False, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        # multi-ticker: per-ticker frames keyed at level 0
        frames = []
        for tk in TICKERS:
            try:
                df = raw[tk].dropna(subset=["Close"]).copy()
            except KeyError:
                print(f"      skipped (no data): {tk}")
                continue
            if len(df) < 120:
                print(f"      skipped (<120 days): {tk}")
                continue
            df.index.name = "date"
            df = df.reset_index()
            df["ticker"] = tk
            frames.append(df)
    else:
        # single-ticker fallback
        raw = raw.dropna(subset=["Close"]).copy()
        raw.index.name = "date"
        raw = raw.reset_index()
        raw["ticker"] = TICKERS[0]
        frames = [raw]

    print(f"      kept {len(frames)} tickers with sufficient history")

    print(f"[3/5] Computing 16 base + 4 cross-sectional rank features ...")
    per_ticker = []
    for f in frames:
        feats_f = build_base_features(f.set_index("date")).reset_index()
        feats_f["ticker"] = f["ticker"].iloc[0]
        feats_f["close"] = f["Close"].values
        per_ticker.append(feats_f)
    panel = pd.concat(per_ticker, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None).dt.normalize()
    panel = add_xrank_features(panel)

    print(f"[4/5] Pulling SPY + ^VIX for regime features ...")
    regime = build_regime_frame(period=HISTORY_PERIOD)
    panel = panel.merge(regime, on="date", how="left")
    panel = panel.sort_values(["ticker", "date"])
    regime_cols = ["spy_ret_5d", "spy_ret_20d", "spy_rv_20", "spy_rv_60",
                   "vix", "vix_chg_5d", "fng"]
    panel[regime_cols] = panel.groupby("ticker")[regime_cols].ffill()

    panel = add_catalyst_zeros(panel)

    # Score the most recent date per ticker
    latest = panel.sort_values("date").groupby("ticker").tail(1).copy()
    print(f"[5/5] Scoring {len(latest)} ticker-rows on most-recent date ...")

    X = latest[feats].fillna(value=medians)
    prob = bundle["calibrator"].predict_proba(X)[:, 1]
    latest["prob"] = prob

    out = latest[["ticker", "date", "close", "prob"]].sort_values("prob", ascending=False)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["close"] = out["close"].round(2)
    out["prob"] = (out["prob"] * 100).round(2).astype(str) + "%"

    print()
    print("=== Top picks: P(touch +30% within 21 trading days) ===")
    print(out.head(15).to_string(index=False))
    print()
    print(f"(Universe: {len(out)} tickers; model: {bundle['universe']}; "
          f"date: {latest['date'].iloc[0].strftime('%Y-%m-%d')})")
    print()
    print("NOTE: Catalyst features defaulted to 0 (no news pipeline). For news-driven")
    print("names you'll under-rank vs the full-pipeline production model. See")
    print("data_schema.md §C for how to add real catalyst features.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
