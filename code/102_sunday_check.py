"""Sunday-night portfolio + Monday-open verdict.

MODEL: Monthly Gainer v3 (`monthly_gainer_v3_sp500.joblib`) — NOT the burst
model. Target is `y21 = max(close[t+1..t+21]) / close[t] >= 1.30` — i.e.
"stock touches +30% within 21 trading days (~1 month)". This is a SHARES
position model with multi-week hold horizon, not a 5d burst trade.

HOLD GUIDANCE (applies to all picks below):
- Minimum hold: 1 week before evaluating exit. Don't day-trade these.
- Full window: 21 trading days (~30 calendar days).
- Exit triggers: +30% touched (target hit), 21d window expired, or stock
  drops out of model's top-15 on a re-score.
- `ret_5d_lag` shown in the verdict is just a feature for context, NOT the
  prediction horizon. The model has already weighed it.

Runs Sunday evening to:
1. Refresh yfinance bulk for SP500 universe (cached daily bars; on Sunday this
   returns Friday's close).
2. Pull weekend / Sunday futures (ES=F, NQ=F, RTY=F) and VIX (^VIX) levels —
   futures market opens Sunday 6pm ET so by 8pm ET we have ~2hr of price action.
3. Pull weekend news for the user's holdings (INTC, SMCI, MRNA) via yfinance.
4. Re-score the latest panel day with Monthly Gainer v3 model.
5. Verify INTC, SMCI, MRNA are still in v3 top-15 by raw_margin.
6. Generate a verdict text:
   - Are these still buys? (Are they still in top-15?)
   - Monday-open gap risk based on futures (gap up / down / flat)
   - Per-name news flags
   - Recommended ENTRY action: BUY at open / WAIT for first 30min / HOLD off / SKIP
   - Hold guidance reminder (min 1 week, max 21 trading days)

Outputs:
   output/monthly_gainer/sunday_verdict_{YYYY-MM-DD}.md
   output/monthly_gainer/sunday_verdict_latest.md  (symlink-ish copy)
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import json
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SEND_TELEGRAM = "--telegram" in sys.argv

try:
    import yfinance as yf
except ImportError:
    yf = None
    print("[102] WARNING: yfinance not available — using cached panel only")

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output" / "monthly_gainer"

sys.path.insert(0, str(ROOT / "code"))
from extension_classifier import attach_extension  # noqa: E402

PORTFOLIO = ["INTC", "SMCI", "MRNA"]


def _is_market_hours(now=None) -> bool:
    """Conservative US market-hours check: Mon-Fri, 09:30-16:00 local time.
    Assumes machine is in ET (true for the user's Mac per the rest of the
    scheduler). Doesn't handle holidays — yfinance will just return stale
    data on holidays which the gate's caller can detect via chg == 0."""
    if now is None:
        now = datetime.now()
    if now.weekday() >= 5:
        return False
    after_open = (now.hour > 9) or (now.hour == 9 and now.minute >= 30)
    before_close = now.hour < 16
    return after_open and before_close


def intraday_gap_gate(ticker: str, ref_close: float, threshold: float = 0.02):
    """Lightweight entry-timing gate. Compares current live price to the
    reference close (typically Friday's close from the panel) and returns one
    of ENTER / WAIT_PULLBACK / WAIT_FALLING / NO_DATA along with the live price
    and the % change.

    Returns NO_DATA outside market hours (yfinance returns the prior close as
    "live" on weekends/after-hours, which would generate misleading ENTER
    signals at chg=0). Only fires Mon-Fri 09:30-16:00 ET.
    """
    if yf is None or ref_close is None or pd.isna(ref_close):
        return ("NO_DATA", None, 0.0)
    if not _is_market_hours():
        return ("NO_DATA", None, 0.0)
    try:
        t = yf.Ticker(ticker)
        cur = None
        if hasattr(t, "fast_info"):
            try:
                cur = float(t.fast_info["lastPrice"])
            except Exception:
                cur = None
        if cur is None or cur <= 0:
            hist = t.history(period="1d", interval="1m")
            if not hist.empty:
                cur = float(hist["Close"].iloc[-1])
        if cur is None or cur <= 0:
            return ("NO_DATA", None, 0.0)
        chg = (cur - ref_close) / ref_close
        if chg > threshold:
            return ("WAIT_PULLBACK", cur, chg)
        if chg < -threshold:
            return ("WAIT_FALLING", cur, chg)
        return ("ENTER", cur, chg)
    except Exception:
        return ("NO_DATA", None, 0.0)


def compute_v8_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True).copy()
    c = g["close"]; o = g["open"]; h = g["high"]; l = g["low"]; v = g["volume"]
    r = c.pct_change()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    g["rsi_14"] = 100 - (100 / (1 + rs))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig_line = macd_line.ewm(span=9, adjust=False).mean()
    g["macd"] = macd_line / c
    g["macd_sig"] = macd_sig_line / c
    g["macd_hist"] = (macd_line - macd_sig_line) / c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    g["bb_z20"] = (c - ma20) / sd20
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    g["atr_pct"] = atr / c
    g["range_pct"] = (h - l) / c
    g["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    g["vol_5d"] = v / v.rolling(5).mean()
    g["rv_60"] = r.rolling(60).std() * np.sqrt(252)
    g["overnight_gap"] = o.shift(-1) / c - 1.0
    ma5 = c.rolling(5).mean(); ma60 = c.rolling(60).mean()
    g["ma_stack"] = ((ma5 > ma20) & (ma20 > ma60)).astype(int)
    up = (r > 0).astype(int)
    grp = (up != up.shift()).cumsum()
    g["up_streak"] = up.groupby(grp).cumsum().where(up == 1, 0).clip(upper=30)
    g["up_bigdays_20d"] = (r > 0.03).rolling(20).sum()
    atr_abs = g["atr_pct"] * c
    g["dist_ma60_atr"] = (c - ma60) / atr_abs.replace(0, np.nan)
    g["ma60_slope_60d"] = (ma60 - ma60.shift(60)) / c
    above20 = (c > ma20).astype(int)
    grp2 = (above20 != above20.shift()).cumsum()
    g["run_length"] = above20.groupby(grp2).cumsum().where(above20 == 1, 0).clip(upper=120)
    g["close_5d_ago"] = c.shift(5)
    g["ret_5d_lag"] = c / g["close_5d_ago"] - 1.0
    g["close_20d_ago"] = c.shift(20)
    g["ret_20d_lag"] = c / g["close_20d_ago"] - 1.0
    g["close_60d_ago"] = c.shift(60)
    g["ret_60d_lag"] = c / g["close_60d_ago"] - 1.0
    g["close_180d_ago"] = c.shift(180)
    g["ret_180d_lag"] = c / g["close_180d_ago"] - 1.0
    g["max_60d"] = c.rolling(60, min_periods=20).max()
    g["min_60d"] = c.rolling(60, min_periods=20).min()
    g["dd_60d"] = g["min_60d"] / g["max_60d"] - 1.0
    return g


def fetch_recent(tickers, days=120):
    print(f"[102] yfinance bulk: {len(tickers)} tickers, {days}d window ...")
    t0 = time.time()
    data = yf.download(tickers, period=f"{days}d", interval="1d",
                        auto_adjust=True, threads=True, progress=False, group_by="ticker")
    print(f"[102] downloaded in {time.time()-t0:.0f}s")
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
    print(f"[102] fetching SPY + VIX ...")
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


def fetch_futures():
    """Returns dict with current futures levels + 5d/Friday-close changes."""
    out = {}
    for sym, label in [("ES=F", "S&P 500 futures"),
                        ("NQ=F", "Nasdaq-100 futures"),
                        ("RTY=F", "Russell 2000 futures"),
                        ("^VIX", "VIX")]:
        try:
            h = yf.Ticker(sym).history(period="10d", interval="1h", auto_adjust=True)
            if h is None or len(h) < 5:
                h = yf.Ticker(sym).history(period="10d", interval="1d", auto_adjust=True)
            if h is None or len(h) == 0:
                out[sym] = None; continue
            cur = float(h["Close"].iloc[-1])
            # Use the Friday close (last weekday before Sunday)
            h_daily = yf.Ticker(sym).history(period="10d", interval="1d", auto_adjust=True)
            fri = float(h_daily["Close"].iloc[-1]) if len(h_daily) else cur
            chg_pct = (cur / fri - 1.0) if fri else 0.0
            out[sym] = {"label": label, "current": cur, "fri_close": fri, "chg_pct": chg_pct}
        except Exception as e:
            out[sym] = {"error": str(e)}
    return out


def fetch_news_for(tickers, days_back=3):
    """Pull headlines from yfinance for the last few days."""
    out = {}
    cutoff = datetime.utcnow().timestamp() - days_back * 86400
    for tk in tickers:
        try:
            news = yf.Ticker(tk).news or []
            recent = []
            for item in news[:10]:
                ts = item.get("providerPublishTime") or item.get("provider_publish_time")
                title = item.get("title", "")
                publisher = item.get("publisher", "")
                if ts and ts >= cutoff:
                    recent.append({"ts": ts, "title": title, "publisher": publisher,
                                    "when": datetime.fromtimestamp(ts).isoformat()})
            out[tk] = recent
        except Exception as e:
            out[tk] = [{"error": str(e)}]
    return out


def classify_news(headlines: list) -> dict:
    """Crude bullish/bearish/neutral classifier on headlines."""
    bull_kw = ["beat", "raises", "upgrade", "target raised", "buy rating", "expansion",
                "deal", "approval", "wins", "soars", "surges", "rally"]
    bear_kw = ["miss", "cuts", "downgrade", "target cut", "sell rating", "lawsuit",
                "investigation", "recall", "warning", "plunge", "tumble", "delay",
                "fraud", "loss", "fired", "ousted"]
    score = 0
    flagged = []
    for h in headlines:
        t = (h.get("title") or "").lower()
        if not t: continue
        if any(k in t for k in bear_kw):
            score -= 1; flagged.append(("bear", h.get("title")))
        elif any(k in t for k in bull_kw):
            score += 1; flagged.append(("bull", h.get("title")))
    return {"score": score, "flagged": flagged, "n_headlines": len(headlines)}


def main():
    print(f"[102] Sunday check started at {datetime.now().isoformat()}")
    if yf is None:
        print("[102] FATAL: yfinance required")
        return

    # Score the SP500 universe
    print("[102] scoring full SP500 universe ...")
    old_panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    tickers = sorted(old_panel["ticker"].unique().tolist())
    sec_map = old_panel.dropna(subset=["sector"]).groupby("ticker")["sector"].first().to_dict()

    fresh = fetch_recent(tickers, days=120)
    if fresh.empty:
        print("[102] yfinance returned nothing")
        return

    # Compute features
    out = []
    for i, (tk, g) in enumerate(fresh.groupby("ticker", sort=False)):
        gg = compute_v8_features(g)
        gg["sector"] = sec_map.get(tk, "")
        out.append(gg)
    new_panel = pd.concat(out, ignore_index=True)

    # Regime
    spy_vix = fetch_spy_vix()
    fng_path = DATA / "fear_greed.csv"
    if fng_path.exists():
        fng = pd.read_csv(fng_path, parse_dates=["date"])[["date", "fng"]]
        regime = spy_vix.merge(fng, on="date", how="left")
    else:
        regime = spy_vix.copy(); regime["fng"] = np.nan
    regime = regime.sort_values("date").reset_index(drop=True)
    REGIME_FEATS = ["spy_ret_5d", "spy_ret_20d", "spy_rv_20", "spy_rv_60",
                    "vix", "vix_chg_5d", "fng"]
    for c in REGIME_FEATS:
        if c not in regime.columns: regime[c] = np.nan
    regime[REGIME_FEATS] = regime[REGIME_FEATS].ffill()
    new_panel = new_panel.merge(regime[["date"] + REGIME_FEATS], on="date", how="left")

    CATALYST = ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
                "news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
                "ma_news_5d", "ma_news_20d", "sector_pop_5d"]
    for c in CATALYST: new_panel[c] = 0.0

    new_panel["rsi_14_xrank"] = new_panel.groupby("date")["rsi_14"].rank(pct=True)
    new_panel["rv_60_xrank"] = new_panel.groupby("date")["rv_60"].rank(pct=True)
    new_panel["ma60_slope_xrank"] = new_panel.groupby("date")["ma60_slope_60d"].rank(pct=True)
    new_panel["ret_20d_xrank"] = new_panel.groupby("date")["ret_20d_lag"].rank(pct=True)

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
    print(f"[102] panel last date: {last_d.date()}  SPY 20d: {spy_20d:+.1%}  regime_on: {regime_on}")

    # Surface top-5 (display) and top-15 (used for finding non-EXTREME alternatives)
    top5 = today.nlargest(5, "raw_margin")
    top5 = attach_extension(top5)
    top15 = today.nlargest(15, "raw_margin")
    top15 = attach_extension(top15)
    today_full = attach_extension(today)
    portfolio_in_top15 = {tk: tk in top15["ticker"].values for tk in PORTFOLIO}
    portfolio_data = today_full[today_full["ticker"].isin(PORTFOLIO)].copy()
    portfolio_data["rank"] = portfolio_data["raw_margin"].rank(method="min", ascending=False).astype(int)
    rank_map = {row["ticker"]: int(today["raw_margin"].rank(method="min", ascending=False)
                                     .loc[row.name]) for _, row in portfolio_data.iterrows()}

    # Futures
    futures = fetch_futures()

    # News
    news = fetch_news_for(PORTFOLIO + list(top5["ticker"][:5]), days_back=3)

    # Build verdict
    verdict = build_verdict(last_d, regime_on, spy_20d, futures, news, top5, top15,
                              portfolio_data, rank_map)

    # Save
    OUT.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    md_path = OUT / f"sunday_verdict_{today_str}.md"
    md_path.write_text(verdict)
    (OUT / "sunday_verdict_latest.md").write_text(verdict)
    print(f"\n[102] saved {md_path}")
    print(f"\n{verdict}")

    # JSON status for downstream
    status = {
        "run_at": datetime.now().isoformat(),
        "panel_last_date": str(last_d.date()),
        "regime_on": regime_on,
        "spy_20d": float(spy_20d),
        "portfolio_in_top15": portfolio_in_top15,
        "portfolio_ranks": rank_map,
        "futures": {k: v for k, v in futures.items() if v},
        "news_summary": {tk: classify_news(hh) for tk, hh in news.items()},
    }
    (OUT / f"sunday_status_{today_str}.json").write_text(json.dumps(status, indent=2, default=str))

    if SEND_TELEGRAM:
        try:
            send_telegram(verdict)
            print("[102] Telegram message sent.")
        except Exception as e:
            print(f"[102] Telegram send failed: {e}", file=sys.stderr)


def send_telegram(body: str) -> None:
    """Telegram delivery using the same creds as daily/daily_report.py."""
    tok = Path("~/.telegram_qqq_token").expanduser().read_text().strip()
    chat = Path("~/.telegram_qqq_chat").expanduser().read_text().strip()
    if len(body) > 4000:
        body = body[:3950] + "\n\n[truncated — see sunday_verdict_latest.md for full]"
    r = subprocess.run(
        [
            "curl", "-s", "-X", "POST",
            f"https://api.telegram.org/bot{tok}/sendMessage",
            "-d", f"chat_id={chat}",
            "--data-urlencode", f"text={body}",
        ],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0 or '"ok":true' not in r.stdout:
        raise RuntimeError(
            f"telegram send failed: rc={r.returncode}  out={r.stdout[:300]}  err={r.stderr[:300]}"
        )


def build_verdict(last_d, regime_on, spy_20d, futures, news, top5, top15,
                   portfolio_data, rank_map) -> str:
    lines = [
        f"# 🔔 SUNDAY VERDICT — {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}",
        f"Model: **Monthly Gainer v3** (target = touch +30% within 21 trading days)  |  Hold horizon: 1 week min, 21 trading days max",
        f"Panel last close: {last_d.date()}  |  SPY 20d: {spy_20d:+.1%}  |  Regime: {'ON' if regime_on else 'OFF'}",
        "",
        "## Portfolio status",
    ]

    for tk in PORTFOLIO:
        row = portfolio_data[portfolio_data["ticker"] == tk]
        if row.empty:
            lines.append(f"- **{tk}**: no data")
            continue
        r = row.iloc[0]
        rk = rank_map.get(tk, "?")
        in15 = "✅ in top-15" if rk != "?" and rk <= 15 else f"⚠️ rank {rk}"
        ext_lvl = r.get("ext_level", "")
        ext_tag = f"[{ext_lvl}]" if ext_lvl else ""
        lines.append(f"- **{tk}** ${r['close']:.2f}  margin={r['raw_margin']:+.2f}  prob={r['prob_cal']:.0%}  "
                      f"5d={r['ret_5d_lag']:+.1%}  rank #{rk}  {in15}  {ext_tag}")

    # Futures section
    lines += ["", "## Futures (Sunday evening levels)"]
    for sym, info in futures.items():
        if info is None or "error" in info:
            lines.append(f"- {sym}: data unavailable")
            continue
        chg = info["chg_pct"]
        arrow = "🟢" if chg > 0.005 else "🔴" if chg < -0.005 else "⚪"
        lines.append(f"- {arrow} **{info['label']} ({sym})**: {info['current']:.2f}  "
                      f"({chg:+.2%} vs Fri close)")

    # Gap-risk classification
    es = futures.get("ES=F", {}); nq = futures.get("NQ=F", {})
    es_chg = es.get("chg_pct", 0) if es else 0
    nq_chg = nq.get("chg_pct", 0) if nq else 0
    gap_signal = max(abs(es_chg), abs(nq_chg))
    direction = nq_chg
    if abs(direction) < 0.005:
        gap_verdict = "FLAT — Monday open likely calm. Buy at open as planned."
    elif direction > 0.015:
        gap_verdict = "BIG GAP UP — Monday will likely gap up >1.5%. WAIT for first 30min, then buy on any pullback. You'd be paying premium at open."
    elif direction > 0.005:
        gap_verdict = "MODERATE GAP UP — slight premium at open but still OK to buy. Consider scaling in 50% open / 50% later."
    elif direction < -0.015:
        gap_verdict = "BIG GAP DOWN — Monday will likely gap down >1.5%. Could be opportunity (better cost basis) OR something is wrong. Check news. If macro-driven, buy at open. If news-driven on a holding, hold off on that name."
    else:
        gap_verdict = "MILD GAP DOWN — small premium savings at open. OK to buy."

    lines += ["", f"## Gap signal: {gap_verdict}"]

    # News
    lines += ["", "## News flags"]
    portfolio_news_score = {}
    for tk in PORTFOLIO:
        hh = news.get(tk, [])
        cls = classify_news(hh)
        portfolio_news_score[tk] = cls["score"]
        emoji = "🟢" if cls["score"] > 0 else "🔴" if cls["score"] < 0 else "⚪"
        if not hh:
            lines.append(f"- {emoji} **{tk}**: no recent news")
        else:
            lines.append(f"- {emoji} **{tk}**: {cls['n_headlines']} headlines, score {cls['score']:+d}")
            for kind, title in cls["flagged"][:3]:
                lines.append(f"    - [{kind}] {title}")

    # Per-stock recommendation
    lines += ["", "## Recommendation per holding"]
    for tk in PORTFOLIO:
        row = portfolio_data[portfolio_data["ticker"] == tk]
        if row.empty:
            lines.append(f"- **{tk}**: no data, hold off until checked manually")
            continue
        rk = rank_map.get(tk, 999)
        news_score = portfolio_news_score.get(tk, 0)

        if not regime_on:
            verdict = "🛑 REGIME OFF — Option 1B says no buys today. Hold cash."
        elif rk > 15:
            verdict = f"⚠️ DROPPED OUT of top-15 (rank #{rk}). Model conviction weakened — HOLD off on adding. If already long, hold the existing position through 1-week min."
        elif news_score < -1:
            verdict = "🛑 NEGATIVE NEWS — hold off on adding. Monitor for impact."
        elif gap_signal > 0.015 and direction > 0:
            verdict = f"⏳ GAP UP risk on broader market — wait for first 30min on Monday before buying. Set a limit order at Friday close +0.5%."
        elif news_score > 0:
            verdict = "✅ BUY at Monday open — bullish news + still in top-15."
        else:
            # Add stabilization guidance for dip names (negative 5d ret on a model that's about MULTI-WEEK upside)
            ret5 = float(row.iloc[0].get("ret_5d_lag", 0))
            if ret5 <= -0.05:
                verdict = "✅ BUY but wait for stabilization — name is in top-15 but down 5%+ over 5d. Wait through first 15-30min Mon for selling pressure to clear, then enter. Don't chase a deeper drop — wait for a flat or green print."
            else:
                verdict = "✅ BUY at Monday open — still in top-15, no negative signals."

        # Intraday gap gate — only meaningful if run during/after market hours
        ref_close = float(row.iloc[0]["close"])
        gate_status, cur_px, chg = intraday_gap_gate(tk, ref_close)
        if gate_status == "WAIT_PULLBACK":
            gate_line = (f"  ⏳ Live: ${cur_px:.2f} ({chg:+.1%} vs Fri close). UP >+2% — being chased. "
                         f"Set limit at -1.5% from current (${cur_px * 0.985:.2f}) OR wait for Tue open.")
        elif gate_status == "WAIT_FALLING":
            gate_line = (f"  ⏳ Live: ${cur_px:.2f} ({chg:+.1%} vs Fri close). DOWN >-2% — selling pressure ongoing. "
                         f"Wait until afternoon for stabilization (green close) OR Tue open.")
        elif gate_status == "ENTER":
            gate_line = (f"  ✅ Live: ${cur_px:.2f} ({chg:+.1%} vs Fri close). Within ±2% — open auction cleared. ENTER NOW.")
        else:
            gate_line = ""

        lines.append(f"- **{tk}**: {verdict}")
        if gate_line:
            lines.append(gate_line)

    # Top 5 today
    lines += ["", "## Today's top-5 (concentrated picks for small-N portfolio)"]
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        marker = "👤" if r["ticker"] in PORTFOLIO else "  "
        ext_lvl = r.get("ext_level", "")
        ext_tag = f" [{ext_lvl}]" if ext_lvl else ""
        lines.append(f"{marker} {i:>2}. {r['ticker']:<5}{ext_tag} ${r['close']:>8.2f}  "
                      f"margin {r['raw_margin']:+.2f}  prob {r['prob_cal']:.0%}  "
                      f"5d {r['ret_5d_lag']:+.1%}")

    # Fresh alternatives — non-EXTREME picks in top-15 with positive margin, excluding portfolio
    alts = top15[~top15["ticker"].isin(PORTFOLIO)].copy()
    alts = alts[alts.get("ext_level", pd.Series([""] * len(alts))) != "EXTREME"]
    alts = alts[alts["raw_margin"] > 0]
    alts = alts.sort_values("raw_margin", ascending=False)
    lines += ["", "## Fresh alternatives in top-15 (non-EXTREME, positive margin, excl. portfolio)"]
    if alts.empty:
        lines.append("- (none — all non-portfolio non-EXTREME picks in top-15 have negative margin)")
    else:
        for _, r in alts.iterrows():
            ext_lvl = r.get("ext_level", "")
            ext_tag = f"[{ext_lvl}]" if ext_lvl else ""
            lines.append(
                f"- **{r['ticker']}** {ext_tag} ${r['close']:.2f}  margin {r['raw_margin']:+.2f}  "
                f"prob {r['prob_cal']:.0%}  5d {r['ret_5d_lag']:+.1%}  20d {r.get('ret_20d_lag', 0):+.1%}"
            )

    # Hold guidance — applies to all positions, not just new entries
    lines += [
        "",
        "## Hold guidance (Monthly Gainer v3 — distinct from burst model)",
        "- **Minimum hold: 1 week** before evaluating exit. The model's edge is over a multi-week window; bailing in 2-3 days throws away expected return.",
        "- **Full target window: 21 trading days** (~30 calendar days). After this, the +30% touch probability has fully bled out — re-evaluate or exit.",
        "- **Exit triggers**: (a) +30% touched (target hit, lock in), (b) 21d window expired, (c) name drops out of top-15 on a re-score AND has been held ≥1 week.",
        "- The 5d return shown above is a feature, not a horizon. The model has already weighed it. Don't day-trade these picks.",
    ]

    lines += ["", "---", "Run: code/102_sunday_check.py | Model: Monthly Gainer v3 (21-trading-day +30% touch target)"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
