"""Stage 2: load cached panel, score with v3 model (needs sklearn), generate verdict."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import json, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output" / "monthly_gainer"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = DATA / "_102_stage1_cache.pkl"

PORTFOLIO = ["INTC", "SMCI", "MRNA"]

sys.path.insert(0, str(ROOT / "code"))
from extension_classifier import attach_extension  # noqa

def classify_news(headlines):
    bull_kw = ["beat", "raises", "upgrade", "target raised", "buy rating", "expansion",
                "deal", "approval", "wins", "soars", "surges", "rally"]
    bear_kw = ["miss", "cuts", "downgrade", "target cut", "sell rating", "lawsuit",
                "investigation", "recall", "warning", "plunge", "tumble", "delay",
                "fraud", "loss", "fired", "ousted"]
    score = 0; flagged = []
    for h in headlines:
        t = (h.get("title") or "").lower()
        if not t: continue
        if any(k in t for k in bear_kw):
            score -= 1; flagged.append(("bear", h.get("title")))
        elif any(k in t for k in bull_kw):
            score += 1; flagged.append(("bull", h.get("title")))
    return {"score": score, "flagged": flagged, "n_headlines": len(headlines)}

def build_verdict(last_d, regime_on, spy_20d, futures, news, top15,
                   portfolio_data, rank_map):
    lines = [
        f"# SUNDAY VERDICT — {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}",
        f"Panel last close: {last_d.date()}  |  SPY 20d: {spy_20d:+.1%}  |  Regime: {'ON' if regime_on else 'OFF'}",
        "",
        "## Portfolio status",
    ]
    for tk in PORTFOLIO:
        row = portfolio_data[portfolio_data["ticker"] == tk]
        if row.empty:
            lines.append(f"- **{tk}**: no data"); continue
        r = row.iloc[0]
        rk = rank_map.get(tk, "?")
        in15 = "IN top-15" if rk != "?" and rk <= 15 else f"rank {rk} — NOT in top-15"
        ext_lvl = r.get("ext_level", "")
        ext_tag = f" [{ext_lvl}]" if ext_lvl else ""
        lines.append(f"- **{tk}** ${r['close']:.2f}  margin={r['raw_margin']:+.2f}  prob={r['prob_cal']:.0%}  "
                      f"5d={r['ret_5d_lag']:+.1%}  rank #{rk}  {in15}{ext_tag}")

    lines += ["", "## Futures (Sunday evening levels)"]
    for sym, info in futures.items():
        if info is None or "error" in info:
            lines.append(f"- {sym}: data unavailable"); continue
        chg = info["chg_pct"]
        arrow = "UP" if chg > 0.005 else "DOWN" if chg < -0.005 else "FLAT"
        lines.append(f"- {arrow} **{info['label']} ({sym})**: {info['current']:.2f}  "
                      f"({chg:+.2%} vs Fri close)")

    es = futures.get("ES=F", {}); nq = futures.get("NQ=F", {})
    es_chg = es.get("chg_pct", 0) if es else 0
    nq_chg = nq.get("chg_pct", 0) if nq else 0
    gap_signal = max(abs(es_chg), abs(nq_chg))
    direction = nq_chg
    if abs(direction) < 0.005:
        gap_verdict = "FLAT — Monday open likely calm. Buy at open as planned."
    elif direction > 0.015:
        gap_verdict = "BIG GAP UP — wait for first 30min, then buy on any pullback."
    elif direction > 0.005:
        gap_verdict = "MODERATE GAP UP — consider scaling in 50% open / 50% later."
    elif direction < -0.015:
        gap_verdict = "BIG GAP DOWN — check news. If macro-driven, buy at open. If stock-specific, hold off."
    else:
        gap_verdict = "MILD GAP DOWN — small savings at open. OK to buy."
    lines += ["", f"## Gap signal: {gap_verdict}"]

    lines += ["", "## News flags"]
    portfolio_news_score = {}
    for tk in PORTFOLIO:
        hh = news.get(tk, [])
        cls = classify_news(hh)
        portfolio_news_score[tk] = cls["score"]
        emoji = "BULL" if cls["score"] > 0 else "BEAR" if cls["score"] < 0 else "NEUTRAL"
        if not hh:
            lines.append(f"- [{emoji}] **{tk}**: no recent news")
        else:
            lines.append(f"- [{emoji}] **{tk}**: {cls['n_headlines']} headlines, score {cls['score']:+d}")
            for kind, title in cls["flagged"][:3]:
                lines.append(f"    - [{kind}] {title}")

    lines += ["", "## ACTION per holding"]
    for tk in PORTFOLIO:
        row = portfolio_data[portfolio_data["ticker"] == tk]
        if row.empty:
            lines.append(f"- **{tk}**: no data — HOLD OFF"); continue
        rk = rank_map.get(tk, 999)
        news_score = portfolio_news_score.get(tk, 0)
        if not regime_on:
            verdict = "REGIME OFF — no buys today. Hold cash."
        elif rk > 15:
            verdict = f"DROPPED OUT of top-15 (rank #{rk}). Model conviction weakened — HOLD OFF."
        elif news_score < -1:
            verdict = "NEGATIVE NEWS — HOLD OFF. Monitor for impact."
        elif gap_signal > 0.015 and direction > 0:
            verdict = "GAP UP risk — WAIT for first 30min before buying."
        elif news_score > 0:
            verdict = "BUY at Monday open — bullish news + still in top-15."
        else:
            verdict = "BUY at Monday open — still in top-15, no negative signals."
        lines.append(f"- **{tk}**: {verdict}")

    lines += ["", "## Today's top-5"]
    for i, (_, r) in enumerate(top15.iterrows(), 1):
        marker = "<-" if r["ticker"] in PORTFOLIO else "  "
        ext_lvl = r.get("ext_level", "")
        ext_tag = f" [{ext_lvl}]" if ext_lvl else ""
        lines.append(f"  {i}. {r['ticker']:<5}{ext_tag} ${r['close']:>8.2f}  "
                      f"margin {r['raw_margin']:+.2f}  prob {r['prob_cal']:.0%}  "
                      f"5d {r['ret_5d_lag']:+.1%}  {marker}")

    lines += ["", "---", f"Run: code/102 two-stage | {datetime.now().isoformat()}"]
    return "\n".join(lines)


def main():
    print(f"[S2] Stage 2 started at {datetime.now().isoformat()}")
    cache = pd.read_pickle(CACHE)
    new_panel = cache["new_panel"]
    futures = cache["futures"]
    news = cache["news"]

    art = joblib.load(MODELS / "monthly_gainer_v3_sp500.joblib")
    feats = art["feats"]; med = pd.Series(art["impute_medians"])
    cal = art["calibrator"]; gbc = art["raw_gbc"]

    scor = new_panel.dropna(subset=feats).copy()
    X = scor[feats].fillna(med).values
    scor["prob_cal"] = cal.predict_proba(X)[:, 1]
    scor["raw_margin"] = gbc.decision_function(X)

    last_d = scor["date"].max()
    today = scor[scor["date"] == last_d].copy()
    spy_20d = today["spy_ret_20d"].iloc[0]
    regime_on = spy_20d > 0
    print(f"[S2] panel last date: {last_d.date()}  SPY 20d: {spy_20d:+.1%}  regime_on: {regime_on}")

    top15 = today.nlargest(5, "raw_margin")
    top15 = attach_extension(top15)
    today_full = attach_extension(today)
    portfolio_data = today_full[today_full["ticker"].isin(PORTFOLIO)].copy()
    rank_map = {row["ticker"]: int(today["raw_margin"].rank(method="min", ascending=False)
                                     .loc[row.name]) for _, row in portfolio_data.iterrows()}

    # Fetch news for top-5 too (already have portfolio news from stage 1)
    top5_tickers = top15["ticker"].tolist()
    for tk in top5_tickers:
        if tk not in news:
            news[tk] = []

    verdict = build_verdict(last_d, regime_on, spy_20d, futures, news, top15,
                              portfolio_data, rank_map)

    today_str = datetime.now().strftime("%Y-%m-%d")
    md_path = OUT / f"sunday_verdict_{today_str}.md"
    md_path.write_text(verdict)
    (OUT / "sunday_verdict_latest.md").write_text(verdict)
    print(f"[S2] saved {md_path}")

    status = {
        "run_at": datetime.now().isoformat(),
        "panel_last_date": str(last_d.date()),
        "regime_on": regime_on,
        "spy_20d": float(spy_20d),
        "portfolio_in_top15": {tk: rank_map.get(tk, 999) <= 15 for tk in PORTFOLIO},
        "portfolio_ranks": rank_map,
        "futures": {k: v for k, v in futures.items() if v},
        "news_summary": {tk: classify_news(hh) for tk, hh in news.items()},
    }
    (OUT / f"sunday_status_{today_str}.json").write_text(json.dumps(status, indent=2, default=str))
    print("[S2] Stage 2 done")

if __name__ == "__main__":
    main()
