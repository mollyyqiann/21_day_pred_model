"""Fetch recent headlines for the top-N candidates, score them with a simple
keyword-based positive/negative dictionary, and write an annotated predictions
file.

The goal is NOT state-of-the-art NLP; it's to (a) give the user visibility into
what's moving the stock and (b) combine a lightweight news-score with the
technical probability.

Outputs:
  output/burst_news.json           - raw headlines per ticker
  output/burst_today_ranked.csv    - top candidates with news score
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

TOP_N = 30  # fetch news for the top-N by technical prob

POS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
    "jump", "jumps", "upgrade", "upgraded", "outperform", "buy", "record",
    "breakthrough", "approval", "approved", "accelerate", "expand", "expands",
    "growth", "raised", "raises", "guidance", "partnership", "acquire",
    "acquires", "acquisition", "strong", "win", "wins", "contract", "award",
    "dividend", "buyback",
}
NEG = {
    "miss", "misses", "drop", "drops", "plunge", "plunges", "downgrade",
    "downgraded", "underperform", "sell", "cut", "cuts", "layoff", "layoffs",
    "fraud", "probe", "investigation", "lawsuit", "sued", "recall", "weak",
    "loss", "losses", "guidance cut", "warning", "slump", "slumps", "fall",
    "falls", "decline", "declines", "bankrupt", "restructure", "delay",
    "delays", "halt", "halts",
}


def score_headline(title: str) -> int:
    t = (title or "").lower()
    pos = sum(1 for w in POS if w in t)
    neg = sum(1 for w in NEG if w in t)
    return pos - neg


def get_news(ticker: str, limit: int = 8) -> list[dict]:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for it in items[:limit]:
        # yfinance news dicts vary — handle both shapes
        content = it.get("content") or it
        title = content.get("title") or it.get("title") or ""
        publisher = (content.get("provider", {}).get("displayName")
                     if isinstance(content.get("provider"), dict)
                     else content.get("publisher") or it.get("publisher", ""))
        ts = content.get("pubDate") or it.get("providerPublishTime")
        url = (content.get("canonicalUrl", {}).get("url")
               if isinstance(content.get("canonicalUrl"), dict)
               else content.get("link") or it.get("link", ""))
        out.append({
            "title": title,
            "publisher": publisher,
            "timestamp": ts,
            "url": url,
            "score": score_headline(title),
        })
    return out


def main() -> None:
    df = pd.read_csv(OUT / "burst_today_scores.csv")
    top = df.head(TOP_N).copy()
    print(f"[news] fetching headlines for top {len(top)} tickers ...")

    news_all = {}
    sentiment = []
    for i, row in enumerate(top.itertuples(), 1):
        items = get_news(row.ticker)
        news_all[row.ticker] = items
        s = sum(it["score"] for it in items)
        n = len(items)
        sentiment.append({
            "ticker": row.ticker,
            "n_headlines": n,
            "news_score": s,
            "news_score_avg": (s / n) if n else 0.0,
            "top_headlines": " || ".join(it["title"][:90] for it in items[:3]),
        })
        time.sleep(0.15)
        if i % 10 == 0:
            print(f"[news]   {i}/{len(top)}")

    sdf = pd.DataFrame(sentiment)
    merged = top.merge(sdf, on="ticker", how="left")

    # Combined score: technical prob times (1 + 0.2 * clamp(news_score_avg))
    # A modest news bump; technical remains the dominant driver.
    merged["news_score_avg"] = merged["news_score_avg"].fillna(0.0)
    merged["combined"] = merged["prob_burst"] * (1 + 0.2 * merged["news_score_avg"].clip(-2, 2))
    merged = merged.sort_values("combined", ascending=False).reset_index(drop=True)

    # Save
    (OUT / "burst_news.json").write_text(json.dumps(news_all, indent=2, default=str))
    merged.to_csv(OUT / "burst_today_ranked.csv", index=False)
    print(f"[news] wrote output/burst_news.json and burst_today_ranked.csv")
    print(merged[[
        "ticker", "close", "prob_burst", "news_score_avg", "combined",
        "ret_5d", "rsi_14", "vol_z", "top_headlines",
    ]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
