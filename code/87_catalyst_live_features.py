"""Phase F.1: Strictly-causal catalyst features for live monthly-gainer prediction.

For each (ticker, t) in the SP500 + smallcap panels, compute features that
use ONLY data available at t-1 (i.e., strictly before t). Output one CSV per
universe, joinable on (ticker, date).

Features (10 new):
  finbert_max_5d   : max(finbert_max[t-5..t-1])  — strongest news-day in last 5d
  finbert_max_20d  : max(finbert_max[t-20..t-1])
  finbert_mean_5d  : mean(finbert_mean[t-5..t-1])
  news_n_5d        : sum(finbert_n[t-5..t-1])  — news-volume signal
  news_n_20d       : sum(finbert_n[t-20..t-1])
  earn_news_5d     : 1 if any earnings-keyword headline in [t-5..t-1]
  earn_news_20d    : 1 if any in [t-20..t-1]
  ma_news_5d       : 1 if any M&A-keyword headline in [t-5..t-1]
  ma_news_20d      : 1 if any in [t-20..t-1]
  sector_pop_5d    : # of same-sector tickers with 5d realized ret >= 10%
                     using close[t-1]/close[t-6]-1 (no leakage of t)

All features are NaN/0 when news data is missing — model handles via imputation.

Outputs:
  data/catalyst_features_sp500.csv
  data/catalyst_features_smallcap.csv
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NEWS_DIR = DATA / "finnhub_news"
FINBERT_PATH = DATA / "finbert_scores.csv"

EARN_RE = re.compile(
    r"\b(earning|earnings|eps|reports? q[1-4]|reports? quarterly|"
    r"reported q[1-4]|beats?|misses?|guidance|raises? guidance|cuts? guidance|"
    r"raised guidance|lowered guidance|tops? estimate|q[1-4] result)\b",
    re.IGNORECASE,
)
MA_RE = re.compile(
    r"\b(acquisition|acquire|acquires|acquired|merger|merge|merging|"
    r"takeover|buyout|to be acquired|deal to|in talks to|reportedly|"
    r"stake in|13d|13[- ]?d|tender offer|bid for|bidder|going private)\b",
    re.IGNORECASE,
)


def parse_news_keywords(ticker: str) -> pd.DataFrame:
    """Returns DataFrame[date, is_earn, is_ma] aggregated to daily 'any' flags."""
    path = NEWS_DIR / f"{ticker}.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["date", "is_earn", "is_ma"])
    rows = []
    with path.open("r") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ts = rec.get("datetime")
            hl = rec.get("headline") or ""
            if ts is None or not hl:
                continue
            d = pd.Timestamp.utcfromtimestamp(int(ts)).normalize()
            rows.append({
                "date": d,
                "is_earn": int(bool(EARN_RE.search(hl))),
                "is_ma": int(bool(MA_RE.search(hl))),
            })
    if not rows:
        return pd.DataFrame(columns=["date", "is_earn", "is_ma"])
    df = pd.DataFrame(rows)
    # Daily aggregate (any earnings/MA keyword that day)
    return df.groupby("date").agg({"is_earn": "max", "is_ma": "max"}).reset_index()


def compute_per_ticker_news_features(panel_dates: pd.DataFrame, fb_df: pd.DataFrame,
                                      kw_df: pd.DataFrame) -> pd.DataFrame:
    """For one ticker, compute 10 catalyst features over the panel's date range.

    panel_dates: DataFrame with single column 'date' (target dates we need features for)
    fb_df: DataFrame[date, finbert_max, finbert_mean, finbert_n] for this ticker
    kw_df: DataFrame[date, is_earn, is_ma] for this ticker

    Returns DataFrame[date, finbert_max_5d, ...] aligned to panel_dates.
    """
    pd_dates = panel_dates.sort_values("date").reset_index(drop=True)

    # Reindex fb to a continuous calendar covering panel range with NaN for missing days.
    if len(fb_df) == 0:
        # No finbert data — all NaN/0 placeholders
        out = pd_dates.copy()
        for c in ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d"]:
            out[c] = np.nan
        for c in ["news_n_5d", "news_n_20d"]:
            out[c] = 0
    else:
        fb = fb_df.copy().sort_values("date").set_index("date")
        # rolling on calendar days needs reindex — simpler to use rolling with closed='left'
        # by SHIFT(1) before rolling so [t-5..t-1] = past-5-rows-before t.
        fb_shift = fb.shift(1)  # so row t represents data through t-1
        fb_shift["finbert_max_5d"] = fb_shift["finbert_max"].rolling(5, min_periods=1).max()
        fb_shift["finbert_max_20d"] = fb_shift["finbert_max"].rolling(20, min_periods=1).max()
        fb_shift["finbert_mean_5d"] = fb_shift["finbert_mean"].rolling(5, min_periods=1).mean()
        fb_shift["news_n_5d"] = fb_shift["finbert_n"].rolling(5, min_periods=1).sum()
        fb_shift["news_n_20d"] = fb_shift["finbert_n"].rolling(20, min_periods=1).sum()
        feat = fb_shift[["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
                         "news_n_5d", "news_n_20d"]].reset_index()
        # Note: feat has rows only on news-days. For panel rows with no news on day t,
        # we want the most recent prior news-day's rolling values (not NaN).
        # Achieve by merge_asof on panel dates.
        out = pd.merge_asof(pd_dates, feat, on="date", direction="backward")
        out["news_n_5d"] = out["news_n_5d"].fillna(0)
        out["news_n_20d"] = out["news_n_20d"].fillna(0)

    # Earnings + MA keyword presence — daily indicator, then rolling sum>0
    if len(kw_df) == 0:
        for c in ["earn_news_5d", "earn_news_20d", "ma_news_5d", "ma_news_20d"]:
            out[c] = 0
    else:
        kw = kw_df.copy().sort_values("date").set_index("date")
        kw_shift = kw.shift(1)
        kw_shift["earn_news_5d"] = (kw_shift["is_earn"].rolling(5, min_periods=1).sum() > 0).astype(int)
        kw_shift["earn_news_20d"] = (kw_shift["is_earn"].rolling(20, min_periods=1).sum() > 0).astype(int)
        kw_shift["ma_news_5d"] = (kw_shift["is_ma"].rolling(5, min_periods=1).sum() > 0).astype(int)
        kw_shift["ma_news_20d"] = (kw_shift["is_ma"].rolling(20, min_periods=1).sum() > 0).astype(int)
        kw_feat = kw_shift[["earn_news_5d", "earn_news_20d",
                            "ma_news_5d", "ma_news_20d"]].reset_index()
        out = pd.merge_asof(out, kw_feat, on="date", direction="backward")
        for c in ["earn_news_5d", "earn_news_20d", "ma_news_5d", "ma_news_20d"]:
            out[c] = out[c].fillna(0).astype(int)

    return out


def build_universe_features(panel_path: Path, label: str, fb_full: pd.DataFrame,
                             out_path: Path):
    print(f"\n[87] === {label} ===")
    panel = pd.read_csv(panel_path, parse_dates=["date"])[["ticker", "date", "close", "sector"]]
    print(f"[87] {label}: {len(panel):,} rows, {panel['ticker'].nunique()} tickers")

    # Sector-pop feature: precompute 5d realized return per (ticker, t-1)
    p = panel.sort_values(["ticker", "date"]).copy()
    p["close_t-1"] = p.groupby("ticker")["close"].shift(1)
    p["close_t-6"] = p.groupby("ticker")["close"].shift(6)
    p["ret_5d_lag"] = p["close_t-1"] / p["close_t-6"] - 1.0
    p["pop_flag"] = (p["ret_5d_lag"] >= 0.10).astype(int)
    sec_pop = p.groupby(["sector", "date"])["pop_flag"].sum().reset_index()
    sec_pop.columns = ["sector", "date", "sector_pop_5d"]
    panel = panel.merge(sec_pop, on=["sector", "date"], how="left")
    panel["sector_pop_5d"] = panel["sector_pop_5d"].fillna(0).astype(int)
    print(f"[87] sector_pop_5d nonzero share: {(panel['sector_pop_5d'] > 0).mean():.2%}")

    # Build news features per ticker
    tickers = panel["ticker"].unique()
    print(f"[87] computing news features for {len(tickers)} tickers ...")
    rows_out = []
    t0 = time.time()
    for i, tk in enumerate(tickers):
        if i and i % 200 == 0:
            print(f"   ... {i}/{len(tickers)} ({time.time() - t0:.0f}s)")
        sub_panel = panel[panel["ticker"] == tk][["date"]].copy()
        fb_t = fb_full[fb_full["ticker"] == tk][["date", "finbert_max", "finbert_mean", "finbert_n"]]
        kw_t = parse_news_keywords(tk)
        feat = compute_per_ticker_news_features(sub_panel, fb_t, kw_t)
        feat["ticker"] = tk
        rows_out.append(feat)

    feats = pd.concat(rows_out, ignore_index=True)
    out = panel[["ticker", "date", "sector_pop_5d"]].merge(feats, on=["ticker", "date"], how="left")
    print(f"[87] coverage: finbert_max_5d non-null = {out['finbert_max_5d'].notna().mean():.1%}")
    print(f"[87] earn_news_20d positive share: {out['earn_news_20d'].mean():.2%}")
    print(f"[87] ma_news_20d positive share: {out['ma_news_20d'].mean():.2%}")
    out.to_csv(out_path, index=False)
    print(f"[87] wrote {out_path}")


def main():
    fb = pd.read_csv(FINBERT_PATH, parse_dates=["date"])
    print(f"[87] finbert: {len(fb):,} rows across {fb['ticker'].nunique()} tickers")

    build_universe_features(
        DATA / "monthly_gainer_panel.csv", "SP500", fb,
        DATA / "catalyst_features_sp500.csv")
    build_universe_features(
        DATA / "monthly_gainer_panel_smallcap.csv", "SMALLCAP", fb,
        DATA / "catalyst_features_smallcap.csv")
    print("\n[87] done")


if __name__ == "__main__":
    main()
