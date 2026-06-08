"""Stage 1: fetch yfinance data + compute features, save to cache for stage 2."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, warnings, json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output" / "monthly_gainer"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = DATA / "_102_stage1_cache.pkl"

PORTFOLIO = ["INTC", "SMCI", "MRNA"]

sys.path.insert(0, str(ROOT / "code"))
from extension_classifier import attach_extension  # noqa

# --- reuse functions from 102 ---
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("m102", ROOT / "code" / "102_sunday_check.py")
m102 = module_from_spec(spec)
# We can't exec the module (sklearn import would fail), so copy the functions we need

def compute_v8_features(g):
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
    print(f"[S1] yfinance bulk: {len(tickers)} tickers, {days}d window ...")
    t0 = time.time()
    data = yf.download(tickers, period=f"{days}d", interval="1d",
                        auto_adjust=True, threads=True, progress=False, group_by="ticker")
    print(f"[S1] downloaded in {time.time()-t0:.0f}s")
    rows = []
    for tk in tickers:
        try:
            df = data[tk] if isinstance(data.columns, pd.MultiIndex) else data
        except Exception:
            continue
        df = df.dropna(subset=["Close"])
        if df.empty: continue
        df = df.reset_index().rename(columns={"Date": "date", "Open": "open",
                                                "High": "high", "Low": "low",
                                                "Close": "close", "Volume": "volume"})
        df["ticker"] = tk
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        rows.append(df[["date", "ticker", "open", "high", "low", "close", "volume"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def fetch_spy_vix(days=400):
    print(f"[S1] fetching SPY + VIX ...")
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
    out = {}
    for sym, label in [("ES=F", "S&P 500 futures"), ("NQ=F", "Nasdaq-100 futures"),
                        ("RTY=F", "Russell 2000 futures"), ("^VIX", "VIX")]:
        try:
            h = yf.Ticker(sym).history(period="10d", interval="1h", auto_adjust=True)
            if h is None or len(h) < 5:
                h = yf.Ticker(sym).history(period="10d", interval="1d", auto_adjust=True)
            if h is None or len(h) == 0:
                out[sym] = None; continue
            cur = float(h["Close"].iloc[-1])
            h_daily = yf.Ticker(sym).history(period="10d", interval="1d", auto_adjust=True)
            fri = float(h_daily["Close"].iloc[-1]) if len(h_daily) else cur
            chg_pct = (cur / fri - 1.0) if fri else 0.0
            out[sym] = {"label": label, "current": cur, "fri_close": fri, "chg_pct": chg_pct}
        except Exception as e:
            out[sym] = {"error": str(e)}
    return out

def fetch_news_for(tickers, days_back=3):
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


def main():
    print(f"[S1] Stage 1 started at {datetime.now().isoformat()}")
    old_panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    tickers = sorted(old_panel["ticker"].unique().tolist())
    sec_map = old_panel.dropna(subset=["sector"]).groupby("ticker")["sector"].first().to_dict()

    fresh = fetch_recent(tickers, days=120)
    if fresh.empty:
        print("[S1] yfinance returned nothing"); return

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

    futures = fetch_futures()
    news = fetch_news_for(PORTFOLIO, days_back=3)

    # Save everything stage 2 needs
    cache = {
        "new_panel": new_panel,
        "futures": futures,
        "news": news,
        "sec_map": sec_map,
    }
    pd.to_pickle(cache, CACHE)
    print(f"[S1] cached to {CACHE}")
    print("[S1] Stage 1 done")

if __name__ == "__main__":
    main()
