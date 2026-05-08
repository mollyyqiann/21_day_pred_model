"""Phase B: Catalyst attribution for monthly gainer events.

For each positive (ticker, t) event, classifies what drove the +30% touch
into 1+ of these buckets:

  earnings_pre  - earnings-keyword headline in [t-5, t+5]   (pre/early-window)
  earnings_late - earnings-keyword headline in [t+6, t+21]  (late-window, t-feature can't predict)
  ma_keyword    - M&A keyword headline in [t-5, t+k]
  news_pre      - finbert_max>=0.7 or finbert_n>=5 in [t-5, t-1]  (pre-pop signal)
  news_post     - finbert_max>=0.7 or finbert_n>=5 in [t, t+k]    (after-t)
  sector_comove - >=3 same-sector positives in [t-5, t+5]
  in_uptrend    - run_length>=30 and up_bigdays_20d>=5 at t
  no_catalyst   - residual

Reports by-event (deduplicated) AND by-row.

Output: output/monthly_gainer/attribution.json
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output" / "monthly_gainer"

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


def load_news_for_ticker(ticker: str) -> pd.DataFrame:
    """Read TICKER.jsonl, return DataFrame[date, headline_l, is_earn, is_ma]."""
    path = NEWS_DIR / f"{ticker}.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["date", "headline_l", "is_earn", "is_ma"])
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
                "headline_l": hl.lower(),
                "is_earn": bool(EARN_RE.search(hl)),
                "is_ma": bool(MA_RE.search(hl)),
            })
    if not rows:
        return pd.DataFrame(columns=["date", "headline_l", "is_earn", "is_ma"])
    return pd.DataFrame(rows)


def main():
    panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    pos = panel[panel["y21"] == 1].copy().reset_index(drop=True)
    print(f"[82] {len(pos):,} positive rows across {pos['ticker'].nunique()} tickers")

    # finbert
    fb = pd.read_csv(FINBERT_PATH, parse_dates=["date"])
    fb = fb.set_index(["ticker", "date"]).sort_index()
    fb_idx = set(fb.index)
    print(f"[82] finbert: {len(fb):,} ticker-date scores, "
          f"{fb.index.get_level_values('ticker').nunique()} tickers")

    # cache news per ticker
    print(f"[82] preloading news for {pos['ticker'].nunique()} tickers ...")
    news_cache: dict[str, pd.DataFrame] = {}
    for tk in pos["ticker"].unique():
        news_cache[tk] = load_news_for_ticker(tk)

    # news data coverage
    news_cov = sum(1 for tk, df in news_cache.items() if len(df) > 0) / len(news_cache)
    print(f"[82] news coverage: {news_cov:.1%} of positive-tickers have any news on disk")

    # for sector co-move: index of positives per (sector, date)
    pos_by_sector_date = defaultdict(set)
    for _, r in pos.iterrows():
        pos_by_sector_date[(r["sector"], r["date"])].add(r["ticker"])

    # build event labels
    rows_out = []
    for _, r in pos.iterrows():
        tk = r["ticker"]
        d = r["date"]
        peak_k = int(r["argmax_day_fwd21"]) if r["argmax_day_fwd21"] > 0 else 21
        d_peak = d + pd.Timedelta(days=int(peak_k * 1.5))  # cushion: trading→calendar approximation

        nws = news_cache.get(tk)
        # earnings_pre: any earnings headline in [d-5, d+5] cal days
        # earnings_late: in [d+6, d+30]
        earn_pre = earn_late = ma_kw = False
        if nws is not None and len(nws) > 0:
            mask_pre_e = (nws["date"] >= d - pd.Timedelta(days=7)) & (nws["date"] <= d + pd.Timedelta(days=7))
            mask_late_e = (nws["date"] > d + pd.Timedelta(days=7)) & (nws["date"] <= d + pd.Timedelta(days=35))
            mask_ma = (nws["date"] >= d - pd.Timedelta(days=7)) & (nws["date"] <= d_peak + pd.Timedelta(days=2))
            earn_pre = bool(nws.loc[mask_pre_e, "is_earn"].any())
            earn_late = bool(nws.loc[mask_late_e, "is_earn"].any())
            ma_kw = bool(nws.loc[mask_ma, "is_ma"].any())

        # finbert pre and post
        fb_pre = fb_post = False
        for offs in range(-7, 0):
            key = (tk, d + pd.Timedelta(days=offs))
            if key in fb_idx:
                row = fb.loc[key]
                if (row["finbert_max"] >= 0.7) or (row["finbert_n"] >= 5):
                    fb_pre = True
                    break
        for offs in range(0, peak_k + 2):
            key = (tk, d + pd.Timedelta(days=offs))
            if key in fb_idx:
                row = fb.loc[key]
                if (row["finbert_max"] >= 0.7) or (row["finbert_n"] >= 5):
                    fb_post = True
                    break

        # sector co-move: ≥2 OTHER same-sector positives in [d-5, d+5] (cal days)
        sec_co = 0
        for offs in range(-5, 6):
            day = d + pd.Timedelta(days=offs)
            others = pos_by_sector_date.get((r["sector"], day), set()) - {tk}
            sec_co = max(sec_co, len(others))
        sector_comove = sec_co >= 2

        # already in uptrend (price-feature inferred)
        in_uptrend = (r["run_length"] >= 30) and (r["up_bigdays_20d"] >= 5)

        # has_news_data: do we have ANY news for this ticker?
        has_news = len(news_cache.get(tk, [])) > 0

        rows_out.append({
            "ticker": tk, "date": d.isoformat(), "max_fwd21_ret": r["max_fwd21_ret"],
            "argmax_day": int(r["argmax_day_fwd21"]),
            "earnings_pre": earn_pre, "earnings_late": earn_late, "ma_keyword": ma_kw,
            "news_pre": fb_pre, "news_post": fb_post,
            "sector_comove": sector_comove, "in_uptrend": in_uptrend,
            "has_news_data": has_news, "has_finbert_data": (tk in fb.index.get_level_values("ticker")),
        })

    df = pd.DataFrame(rows_out)
    df["date"] = pd.to_datetime(df["date"])

    # event dedup: a "fresh" event = no prior positive within 7 days for the same ticker.
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["fresh"] = True
    for tk, g in df.groupby("ticker"):
        last = None
        for idx, row in g.iterrows():
            if last is not None and (row["date"] - last).days <= 7:
                df.at[idx, "fresh"] = False
            else:
                last = row["date"]
    print(f"[82] fresh events: {int(df['fresh'].sum())}  total positive rows: {len(df)}")

    def summarize(sub: pd.DataFrame, label: str) -> dict:
        n = len(sub)
        if n == 0:
            return {"n": 0}
        out = {"n": n}
        for col in ["earnings_pre", "earnings_late", "ma_keyword",
                    "news_pre", "news_post", "sector_comove", "in_uptrend"]:
            out[f"pct_{col}"] = float(sub[col].mean())
        out["pct_no_catalyst"] = float(
            (~sub[["earnings_pre", "ma_keyword", "news_pre",
                   "sector_comove", "in_uptrend"]].any(axis=1)).mean()
        )
        out["pct_predictable_pre_t"] = float(
            sub[["earnings_pre", "ma_keyword", "news_pre",
                 "sector_comove", "in_uptrend"]].any(axis=1).mean()
        )
        # only earnings_pre/news_pre/in_uptrend are *causally available* at t.
        # ma_keyword can include post-t news, sector_comove uses ±5d (some leakage).
        # so a tighter "available at t" bucket:
        only_pre = sub[["earnings_pre", "news_pre", "in_uptrend"]].any(axis=1)
        out["pct_signal_at_t_only"] = float(only_pre.mean())
        # data coverage
        out["pct_with_news_data"] = float(sub["has_news_data"].mean())
        out["pct_with_finbert_data"] = float(sub["has_finbert_data"].mean())
        return out

    summary = {
        "by_row": summarize(df, "by_row"),
        "by_event": summarize(df[df["fresh"]], "by_event"),
        "overlap_matrix": {},
    }

    # overlap matrix on events
    fresh = df[df["fresh"]].copy()
    bool_cols = ["earnings_pre", "ma_keyword", "news_pre",
                 "sector_comove", "in_uptrend"]
    for a in bool_cols:
        for b in bool_cols:
            if a >= b:
                continue
            both = (fresh[a] & fresh[b]).sum()
            summary["overlap_matrix"][f"{a}__{b}"] = int(both)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attribution.json").write_text(json.dumps(summary, indent=2))
    df.to_csv(OUT / "attribution_events.csv", index=False)
    print(f"[82] wrote {OUT / 'attribution.json'}")
    print(f"[82] wrote {OUT / 'attribution_events.csv'} ({len(df)} rows)")

    print("\n[82] By-event attribution (fresh events only):")
    be = summary["by_event"]
    for k, v in be.items():
        if isinstance(v, float):
            print(f"   {k:<30} {v:.1%}")
        else:
            print(f"   {k:<30} {v}")


if __name__ == "__main__":
    main()
