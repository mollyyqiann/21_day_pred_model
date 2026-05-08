"""Daily Option 1B exit monitor.

Runs after market close (4:30pm ET) M-F. For each holding:
1. Re-score with v3 using fresh yfinance data.
2. Check prob_cal — if < 0.05 → SELL signal.
3. Check days since entry — if >= 60 → SELL signal (max hold cap).
4. Otherwise → HOLD, log status.

Output:
   output/monthly_gainer/exit_monitor_latest.md  (always written)
   output/monthly_gainer/exit_monitor_{YYYY-MM-DD}.md
   output/monthly_gainer/exit_signals_log.csv  (append per run)

The agent that runs this should output the markdown content as its
final reply if any SELL signal is present. On pure HOLD days the agent
still produces a short status text so the user knows monitoring is alive.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import json
import time
import warnings
from datetime import datetime, date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

warnings.filterwarnings("ignore")

# extension classifier (added so monitor flags newly-extended holdings)
import sys as _s
_s.path.insert(0, str(Path(__file__).resolve().parent))
from extension_classifier import classify_extension  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output" / "monthly_gainer"


def compute_v8_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True).copy()
    c = g["close"]; o = g["open"]; h = g["high"]; l = g["low"]; v = g["volume"]
    r = c.pct_change()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    g["rsi_14"] = 100 - (100 / (1 + rs))
    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean()
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
    print(f"[104] yfinance bulk: {len(tickers)} tickers, {days}d ...")
    t0 = time.time()
    data = yf.download(tickers, period=f"{days}d", interval="1d",
                        auto_adjust=True, threads=True, progress=False, group_by="ticker")
    print(f"[104] downloaded in {time.time()-t0:.0f}s")
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
    print(f"[104] Daily exit monitor at {datetime.now().isoformat()}")

    # Load portfolio config
    holdings_path = DATA / "portfolio_holdings.json"
    cfg = json.loads(holdings_path.read_text())
    holdings = cfg["holdings"]
    er = cfg["exit_rules"]
    threshold = float(er["prob_cal_threshold"])
    stale_review_days = int(er.get("stale_review_days", 60))
    extend_if_profitable = bool(er.get("extend_if_profitable", True))
    stale_sell_if_underwater = bool(er.get("stale_sell_if_underwater", True))
    absolute_max_days = int(er.get("absolute_max_days", 180))
    portfolio_tickers = list(holdings.keys())

    if yf is None:
        print("[104] FATAL: yfinance required")
        return

    # Score the SP500 universe so xrank features are correct cross-sectionally
    print("[104] scoring SP500 universe ...")
    old_panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    tickers = sorted(old_panel["ticker"].unique().tolist())
    sec_map = old_panel.dropna(subset=["sector"]).groupby("ticker")["sector"].first().to_dict()

    fresh = fetch_recent(tickers, days=120)
    if fresh.empty:
        print("[104] yfinance returned nothing")
        return

    out = []
    for tk, g in fresh.groupby("ticker", sort=False):
        gg = compute_v8_features(g)
        gg["sector"] = sec_map.get(tk, "")
        out.append(gg)
    new_panel = pd.concat(out, ignore_index=True)

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

    # Build per-holding status
    today_iso = date.today().isoformat()
    rows = []
    sells = []
    holds = []
    for tk, info in holdings.items():
        sub = scor[(scor["ticker"] == tk) & (scor["date"] == last_d)]
        if sub.empty:
            rows.append({"ticker": tk, "status": "no_data"})
            continue
        r = sub.iloc[0]
        entry_date = pd.Timestamp(info["entry_date"]).normalize()
        days_held = (last_d - entry_date).days
        entry_price = float(info["entry_price_estimate"]) if info.get("entry_price_estimate") else None
        cur_price = float(r["close"])
        pnl_pct = (cur_price / entry_price - 1.0) if entry_price else None

        prob = float(r["prob_cal"])
        margin = float(r["raw_margin"])

        # Extension classification (informational — don't sell on this alone)
        ext = classify_extension(
            ret_5d=r.get("ret_5d_lag"),
            ret_20d=r.get("ret_20d_lag"),
            ret_60d=r.get("ret_60d_lag"),
            ret_180d=r.get("ret_180d_lag"),
            up_bigdays_20d=r.get("up_bigdays_20d"),
        )

        # Sell triggers (Option 1B+, extended-hold version)
        # 1. Always sell when model signal dies
        # 2. At/past stale_review_days, sell ONLY if underwater (cut stale losses)
        # 3. Hard cap at absolute_max_days regardless of profitability
        sell_reason = None
        extended_hold = False
        if prob < threshold:
            sell_reason = f"prob_cal {prob:.3f} < {threshold} (model signal died)"
        elif days_held >= absolute_max_days:
            sell_reason = f"days_held {days_held} >= absolute max {absolute_max_days}"
        elif days_held >= stale_review_days:
            if pnl_pct is not None and pnl_pct < 0 and stale_sell_if_underwater:
                sell_reason = f"day {days_held} (past 60d review): underwater {pnl_pct:+.1%}, cutting stale loss"
            elif extend_if_profitable:
                # Profitable past 60d — continue holding
                extended_hold = True

        row = {
            "ticker": tk, "date": str(last_d.date()),
            "entry_date": str(entry_date.date()), "days_held": days_held,
            "entry_price": entry_price, "cur_price": cur_price,
            "pnl_pct": pnl_pct, "shares": info["shares"],
            "prob_cal": prob, "raw_margin": margin,
            "ret_5d": float(r["ret_5d_lag"]), "ret_20d": float(r["ret_20d_lag"]),
            "sell_signal": sell_reason is not None,
            "sell_reason": sell_reason,
            "extended_hold": extended_hold,
            "ext_level": ext["level"],
            "ext_reason": ext["reason"],
        }
        rows.append(row)
        if sell_reason:
            sells.append(row)
        else:
            holds.append(row)

    # Build verdict markdown
    has_sell = len(sells) > 0
    title_emoji = "🔔🚨 SELL SIGNAL" if has_sell else "📊 Daily monitor"
    lines = [
        f"# {title_emoji} — {today_iso}",
        f"As-of close: {last_d.date()}",
        "",
    ]

    if has_sell:
        lines.append("## ⚠️ SELL TODAY/TOMORROW OPEN")
        for r in sells:
            pnl = f"{r['pnl_pct']:+.1%}" if r["pnl_pct"] is not None else "?"
            lines += [
                f"- **{r['ticker']}** ({r['shares']} shares):",
                f"  - reason: {r['sell_reason']}",
                f"  - entry ${r['entry_price']:.2f} → now ${r['cur_price']:.2f} (P&L {pnl}, held {r['days_held']}d)",
                f"  - prob_cal {r['prob_cal']:.3f}  margin {r['raw_margin']:+.2f}",
                f"  - 5d {r['ret_5d']:+.1%}  20d {r['ret_20d']:+.1%}",
            ]
        lines.append("")

    if holds:
        lines.append("## ✅ HOLD")
        for r in holds:
            pnl = f"{r['pnl_pct']:+.1%}" if r["pnl_pct"] is not None else "?"
            margin_arrow = "↑" if r["raw_margin"] > 0 else "↓"
            ext_tag = "  🟢 EXTENDED HOLD (>60d, profitable)" if r.get("extended_hold") else ""
            ext_lvl = r.get("ext_level", "")
            ext_meta = f"  [{ext_lvl}]" if ext_lvl in ("CATALYST", "GRADUAL", "EXTREME") else ""
            lines.append(
                f"- **{r['ticker']}** {r['shares']}sh ${r['cur_price']:.2f} ({pnl}, {r['days_held']}d held) — "
                f"prob {r['prob_cal']:.0%} {margin_arrow}margin {r['raw_margin']:+.2f}{ext_tag}{ext_meta}"
            )
        lines.append("")

    if not has_sell:
        lines.append("All holdings still meet hold criteria. Continue holding.")

    OUT.mkdir(parents=True, exist_ok=True)
    md = "\n".join(lines)
    (OUT / f"exit_monitor_{today_iso}.md").write_text(md)
    (OUT / "exit_monitor_latest.md").write_text(md)

    # Append to log CSV
    log_path = OUT / "exit_signals_log.csv"
    df = pd.DataFrame(rows)
    if log_path.exists():
        existing = pd.read_csv(log_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(log_path, index=False)

    print(f"\n[104] saved exit_monitor_latest.md ({'SELL' if has_sell else 'HOLD'} signals)")
    print(md)


if __name__ == "__main__":
    main()
