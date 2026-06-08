"""Live inference at ~09:15 ET on trading mornings.

Pulls three tight-latency streams and fuses them into a refreshed probability
estimate for every ticker in the v5 and v4 universes:

  1. Pre-market last price (yfinance 1-min bars with prepost=True, ~15 min lag)
     -> overnight_gap = pre_market_last / close_prev - 1
  2. Overnight news (yfinance Ticker.news, timestamp >= close_prev)
     -> keyword-sentiment per ticker -> net_score
  3. Previously-computed v5/v4 features at close_prev

The two augmented models (v6 for v5-universe, v6b for >$40 universe) score each
ticker with the real overnight_gap. The news sentiment is blended as a small
post-hoc bump: `prob_final = prob_model * (1 + 0.15 * clipped_news)`. This is
conservative; news is treated as a ranker tie-breaker, not a primary driver.

Run modes:
  --as-of-close    : treat close_prev's data as current (no pre-market), sets
                     overnight_gap = 0. Use late Sunday / before pre-market opens.
  (default)        : pull live pre-market, use actual gap.

Outputs:
  output/burst_live_v5.csv
  output/burst_live_v4.csv
  output/burst_live_meta.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"; OUT = ROOT / "output"

POS = {"beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
       "jump", "jumps", "upgrade", "upgraded", "outperform", "buy", "record",
       "breakthrough", "approval", "approved", "accelerate", "expand", "expands",
       "growth", "raised", "raises", "guidance", "partnership", "acquire",
       "acquires", "acquisition", "strong", "win", "wins", "contract", "award",
       "dividend", "buyback", "beat estimates", "tops"}
NEG = {"miss", "misses", "drop", "drops", "plunge", "plunges", "downgrade",
       "downgraded", "underperform", "sell", "cut", "cuts", "layoff", "layoffs",
       "fraud", "probe", "investigation", "lawsuit", "sued", "recall", "weak",
       "loss", "losses", "warning", "slump", "slumps", "fall", "falls", "decline",
       "declines", "bankrupt", "restructure", "delay", "delays", "halt", "halts",
       "slashes", "cuts guidance"}


def score_headline(t: str) -> int:
    t = (t or "").lower()
    return sum(1 for w in POS if w in t) - sum(1 for w in NEG if w in t)


def get_premarket_last(ticker: str, close_prev_date: pd.Timestamp) -> tuple[float | None, str]:
    """Return (pre_market_last_price, source_str) or (None, reason)."""
    try:
        h = yf.Ticker(ticker).history(period="2d", interval="1m", prepost=True)
    except Exception as e:
        return None, f"history-err:{e.__class__.__name__}"
    if h is None or len(h) == 0:
        return None, "no-bars"
    # keep only bars strictly AFTER close_prev (09:30-16:00 bar of close_prev_date)
    # practical proxy: bars after 16:00 local on close_prev_date.
    # yfinance returns tz-aware index; normalise to UTC for comparison.
    idx_utc = h.index.tz_convert("UTC")
    cutoff = pd.Timestamp(close_prev_date).tz_localize("America/New_York").tz_convert("UTC") \
             + pd.Timedelta(hours=16)
    after = h[idx_utc > cutoff]
    if len(after) == 0:
        return None, "no-post-close-bars"
    last = float(after["Close"].iloc[-1])
    last_time = after.index[-1].isoformat()
    return last, f"pre-market-or-after-hours bar @ {last_time}"


def get_news_score(ticker: str, since: pd.Timestamp) -> tuple[int, int]:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return 0, 0
    n, s = 0, 0
    for it in items:
        content = it.get("content") or it
        ts = content.get("pubDate") or it.get("providerPublishTime")
        if ts is None:
            continue
        try:
            if isinstance(ts, (int, float)):
                dt = pd.Timestamp(ts, unit="s", tz="UTC")
            else:
                dt = pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tz \
                     else pd.Timestamp(ts).tz_localize("UTC")
        except Exception:
            continue
        if dt < since:
            continue
        title = content.get("title") or it.get("title") or ""
        s += score_headline(title); n += 1
    return s, n


def infer(tag: str, model_path: Path, panel_path: Path, as_of_close: bool):
    art = joblib.load(model_path); gbc, feats = art["gbc"], art["feats"]
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    # keep the latest fully-featured row per ticker (close-day features)
    vanilla_feats = [f for f in feats if f != "overnight_gap"]
    scored = panel.dropna(subset=vanilla_feats).copy()
    latest = scored.sort_values(["ticker", "date"]).groupby("ticker").tail(1).copy()

    # close_prev date per ticker
    close_prev_per = latest.set_index("ticker")["date"]
    since_per = close_prev_per.apply(
        lambda d: pd.Timestamp(d).tz_localize("America/New_York").tz_convert("UTC")
                  + pd.Timedelta(hours=16))

    # collect pre-market data + news
    print(f"[{tag}] fetching pre-market + news for {len(latest)} tickers ...")
    pm_last, pm_source, news_score, news_n = {}, {}, {}, {}
    for i, t in enumerate(latest["ticker"].tolist(), 1):
        if as_of_close:
            pm_last[t], pm_source[t] = None, "as-of-close"
        else:
            pm_last[t], pm_source[t] = get_premarket_last(t, close_prev_per[t])
        s, n = get_news_score(t, since_per[t])
        news_score[t], news_n[t] = s, n
        if i % 50 == 0:
            print(f"[{tag}]   {i}/{len(latest)}")
        time.sleep(0.05)

    latest["pm_last"] = latest["ticker"].map(pm_last)
    latest["pm_source"] = latest["ticker"].map(pm_source)
    latest["news_score"] = latest["ticker"].map(news_score)
    latest["news_n"] = latest["ticker"].map(news_n)
    latest["overnight_gap_live"] = np.where(
        latest["pm_last"].notna() & (latest["close"] > 0),
        latest["pm_last"].astype(float) / latest["close"] - 1.0, np.nan)

    # Build feature matrix — use live overnight_gap if available, else 0.0 fallback
    X = latest.copy()
    if "overnight_gap" in feats:
        X["overnight_gap"] = X["overnight_gap_live"].fillna(0.0)
    latest["prob_model"] = gbc.predict_proba(X[feats].values)[:, 1]

    # news blend (conservative)
    news_clip = latest["news_score"].clip(-3, 3)
    latest["news_avg"] = latest["news_score"] / latest["news_n"].replace(0, np.nan)
    latest["prob_final"] = latest["prob_model"] * (1 + 0.15 * news_clip.fillna(0))

    # Rank
    latest = latest.sort_values("prob_final", ascending=False).reset_index(drop=True)
    cols = ["ticker", "date", "close", "pm_last", "overnight_gap_live",
            "news_score", "news_n", "news_avg",
            "prob_model", "prob_final", "pm_source"]
    out_path = OUT / f"burst_live_{tag}.csv"
    latest[cols].to_csv(out_path, index=False)
    return latest[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of-close", action="store_true",
                    help="Treat close_prev as current (no pre-market); useful "
                         "after-hours / weekends when there's no PM feed.")
    args = ap.parse_args()

    # On weekends yfinance has no pre-market bars to pull. Force as-of-close.
    now_ny = datetime.now(timezone.utc).astimezone(
        tz=timezone(timedelta(hours=-4)))  # approximation of ET
    is_weekend = now_ny.weekday() >= 5
    if is_weekend and not args.as_of_close:
        print("[live] weekend detected; forcing --as-of-close")
        args.as_of_close = True

    v4 = infer("v4", MODELS / "burst_gbc_v6b_augmented.joblib",
               DATA / "burst_panel_v6b.csv", args.as_of_close)
    v5 = infer("v5", MODELS / "burst_gbc_v6_augmented.joblib",
               DATA / "burst_panel_v6.csv", args.as_of_close)

    meta = {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_close_mode": bool(args.as_of_close),
        "v4_rows": int(len(v4)), "v5_rows": int(len(v5)),
        "v4_model": "burst_gbc_v6b_augmented.joblib",
        "v5_model": "burst_gbc_v6_augmented.joblib",
    }
    (OUT / "burst_live_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== v4 (broad >$40) — top 15 ===")
    print(v4.head(15).to_string(index=False))
    print("\n=== v5 (upside-asymmetric) — top 15 ===")
    print(v5.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
