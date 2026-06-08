"""Daily burst-prediction job.

Designed to be scheduled by launchd at ~08:45 ET on trading days:
  1. Refreshes the feature panels for both universes from yfinance
  2. Pulls live pre-market price + overnight news for each top candidate
  3. Scores with the augmented classifiers (prob of burst) and the three
     regression heads (expected 1d/3d/5d return)
  4. Logs today's predictions to output/daily_log/<YYYY-MM-DD>.json
  5. If yesterday's log exists, evaluates those predictions against now-
     observable realized returns and computes hit rates + MAE
  6. Formats a compact message and (optionally) sends it via iMessage

Runs on Saturdays / Sundays / holidays in "as-of-close" mode for testing.

Config: config/burst_daily.json (generated with sensible defaults on first run)
  {
    "notifier": "telegram",                // "telegram" | "imessage"
    "send_notifications": false,           // flip true after verifying send
    "top_n_per_universe": 5,
    "universes": ["v4", "v5"]
  }
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"
OUT = ROOT / "output"; LOG_DIR = OUT / "daily_log"
CONFIG_DIR = ROOT / "config"; CONFIG_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "burst_daily.json"

# --asof replay support. None = live mode (datetime.now()).
import sys as _sys_asof
_sys_asof.path.insert(0, str(Path(__file__).resolve().parent))
from _asof import (parse_asof as _parse_asof,                     # noqa: E402
                    asof_today_iso as _asof_today_iso,
                    asof_now_et as _asof_now_et,
                    filter_daily_panel as _asof_filter_panel,
                    filter_intraday as _asof_filter_intraday,
                    filter_news_items as _asof_filter_news)
from _snapshot import (open_snapshot as _open_snapshot,           # noqa: E402
                        open_loader as _open_loader)
_ASOF = None  # type: pd.Timestamp | None
_SNAP = None  # type: object | None  (only set in live mode)
_LOADER = None  # type: object | None  (set in replay when a snapshot resolves)

DEFAULT_CONFIG = {
    "send_notifications": False,
    "top_n_per_universe": 15,             # soft message cap (floor=3, threshold below)
    "burst_alert_min_prob": 0.60,        # prob_final cut for "worth mentioning"
    "burst_alert_min_n": 3,              # always show at least this many per universe
    "universes": ["v4", "v5", "v7"],
    "news_scorer": "lexicon_expanded",   # "lexicon" | "lexicon_expanded" | "finbert"
    "news_blend_weight": 0.10,           # multiplier coefficient on clipped score
    "drop_gate_version": "v2",           # "v2" | "v1" — drop model used by the burst gate
    "catalyst_enabled": True,            # include catalyst-surprise top-N in morning text
    "catalyst_top_n": 5,                 # items from the catalyst model in the text
    "max_per_sector": 3,                 # gentle cap on per-universe reference lists
                                         # (kills extreme clustering; preserves breadth)
    "high_prec_max_per_sector": 1,       # aggressive cap on cross-universe buy shortlist
                                         # (one pick per sector — true diversification when
                                         # buying 1-2 names; can be loosened later)
    "realtime_quote_refresh": True,      # refresh pm_last on selected picks via Finnhub
                                         # /quote (eliminates yfinance 15-min lag on the
                                         # entry-price field; ~5-30 calls per run)
    # ----- 10am --refresh-picks-only knobs -----
    "candidate_pool_per_universe": 60,   # how many top-by-prob_final passing rows to
                                         # save to the daily log for the 10am refresh job
    "refresh_max_picks_total": 120,      # cap on /quote calls in the 10am job (cross-
                                         # universe unique tickers; 60 = 1 min runtime,
                                         # 120 = ~2 min with rate limiting)
    "refresh_rate_limit_per_min": 58,    # 2/min cushion under Finnhub free-tier 60/min
    "refresh_min_prob": 0.80,            # looser min_prob at 10am — picks just below the
                                         # morning 0.85 cut can still qualify if their
                                         # forward-from-entry survives the gap-gate
    "refresh_top_n_per_universe": 10,    # cap on per-universe list size in 10am message
    "send_refresh_notifications": True,  # gate on the 10am Telegram send
}

# Threshold-based selection in `score_universe`. These defaults are used if
# the config does not override them.
BURST_ALERT_MIN_PROB = 0.60
BURST_ALERT_MIN_N = 3


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        print(f"[config] wrote default {CONFIG_PATH}  — edit and re-run")
    return json.loads(CONFIG_PATH.read_text())


# ---------- sector lookup (for sector-cap correlation control) ----------

_SECTOR_LOOKUP: dict[str, str] | None = None


def attach_vol_z_rel(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `vol_z_rel` = vol_z − cross-section median(vol_z) per date.

    Why: raw `vol_z` is per-ticker (today's vol vs the ticker's own 30-day
    history), so a 2σ volume day market-wide (FOMC, mass earnings) inflates
    every ticker equally and weakens vol_z as a discriminator. Subtracting
    the daily median normalizes against that systemic component, leaving
    the *idiosyncratic* volume surge.

    On a 9-day backtest of historical eval rows this feature showed a
    strong split: among top-conviction picks (prob_final > 0.85), those
    with vol_z_rel > 1.0 hit 100% directionally (n=9, mean realized +7.6%)
    vs 33% for picks in the [0, 1) "trap zone" (n=3).

    Adds the column in-place and returns the same DataFrame for chaining.
    """
    if "vol_z" not in panel.columns:
        return panel
    med = panel.groupby("date")["vol_z"].transform("median")
    panel["vol_z_rel"] = panel["vol_z"] - med
    return panel


def load_sector_lookup() -> dict[str, str]:
    """ticker -> GICS sector. First non-empty hit wins across the universe CSVs.

    Reads in size order (rh.csv = ~1,877 names, then v7/v5/v4) so the broadest
    universe seeds the map first. Tickers without a sector entry resolve to
    "Unknown" at lookup time — the cap still groups them together.
    """
    global _SECTOR_LOOKUP
    if _SECTOR_LOOKUP is not None:
        return _SECTOR_LOOKUP
    out: dict[str, str] = {}
    for name in ("burst_universe_rh.csv", "burst_universe_v7.csv",
                 "burst_universe_v5.csv", "burst_universe_v4.csv"):
        p = DATA / name
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, usecols=["ticker", "sector"])
        except Exception:
            continue
        for _, row in df.iterrows():
            t = row["ticker"]; s = row.get("sector")
            if not isinstance(s, str) or not s.strip():
                continue
            if t not in out:
                out[t] = s.strip()
    _SECTOR_LOOKUP = out
    print(f"[sector] loaded {len(out)} ticker→sector mappings")
    return out


# ---------- data refresh ----------

def refresh_panel(universe_csv: Path, panel_path: Path, feat_builder) -> pd.DataFrame:
    """Pull latest bars for all tickers in the universe, rebuild panel rows for
    the last ~400 days (enough for 60-day rolling features to re-warm).

    Sequential 100-ticker chunks (not parallel). Bursting many parallel chunks
    at yfinance reliably trips its rate limiter — on a 1,877-ticker rebuild we
    saw 7×parallel×300 return only 14% coverage. Serial 100-ticker chunks plus
    a one-shot retry round on misses get full coverage for the same universe.

    Refuses to overwrite the cached panel if coverage falls below 80% — the
    caller falls back to the existing file when that happens.
    """
    uni = pd.read_csv(universe_csv)
    tickers = uni["ticker"].tolist()
    print(f"[refresh] {universe_csv.name}: {len(tickers)} tickers")

    def _dl(chunk_tickers):
        t0 = time.time()
        try:
            raw = yf.download(chunk_tickers, period="3y", interval="1d",
                               group_by="ticker", auto_adjust=True,
                               threads=True, progress=False)
            return raw, time.time() - t0
        except Exception as e:
            print(f"    chunk {len(chunk_tickers)} failed: "
                  f"{type(e).__name__}: {e}")
            return None, time.time() - t0

    def _extract(tickers_in, raw):
        ok, bad = [], []
        for t in tickers_in:
            try: sub = raw[t].dropna()
            except Exception:
                bad.append(t); continue
            if len(sub) < 120:
                bad.append(t); continue
            feats = feat_builder(sub)
            feats["close"] = sub["Close"]
            feats["ticker"] = t
            feats = feats.reset_index().rename(columns={"Date": "date"})
            ok.append(feats)
        return ok, bad

    CHUNK = 100
    panels, misses = [], []
    chunks = [tickers[i:i+CHUNK] for i in range(0, len(tickers), CHUNK)]
    for i, chunk in enumerate(chunks):
        raw, dt = _dl(chunk)
        if raw is None:
            misses.extend(chunk); continue
        got, bad = _extract(chunk, raw)
        panels.extend(got); misses.extend(bad)
        if (i + 1) % 5 == 0 or i == len(chunks) - 1:
            print(f"[refresh]   chunk {i+1}/{len(chunks)}: "
                  f"got {len(panels)} total, misses {len(misses)} "
                  f"(last chunk {dt:.1f}s)")
        time.sleep(0.3)

    if misses:
        print(f"[refresh] retrying {len(misses)} missed tickers in 25-batches …")
        still = []
        for i in range(0, len(misses), 25):
            batch = misses[i:i+25]
            raw, _ = _dl(batch)
            if raw is None:
                still.extend(batch); continue
            got, bad = _extract(batch, raw)
            panels.extend(got); still.extend(bad)
            time.sleep(0.5)
        print(f"[refresh]   retry recovered "
              f"{len(misses) - len(still)}/{len(misses)}")

    MIN_COVERAGE = 0.80
    cov = len(panels) / max(1, len(tickers))
    if not panels or cov < MIN_COVERAGE:
        raise ValueError(
            f"refresh coverage {len(panels)}/{len(tickers)} = {cov:.1%} "
            f"below {MIN_COVERAGE:.0%} — refusing to overwrite cached panel")
    panel = pd.concat(panels, ignore_index=True)
    panel.to_csv(panel_path, index=False)
    return panel


# We import the build_features from the pipeline modules to stay consistent.
# Re-coded inline (copy) so this file is standalone and robust to reorgs.

import math as _m

def _rsi(x, n=14):
    d = x.diff(); u = d.clip(lower=0).rolling(n).mean(); dd = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + u / dd.replace(0, np.nan))
def _macd(x, f=12, s=26, sig=9):
    ef = x.ewm(span=f, adjust=False).mean(); es = x.ewm(span=s, adjust=False).mean()
    line = ef - es; signal = line.ewm(span=sig, adjust=False).mean()
    return line / x, signal / x, (line - signal) / x
def _bbz(x, n=20):
    return (x - x.rolling(n).mean()) / x.rolling(n).std()
def _atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def _semi(r, n=60):
    u = (r.clip(lower=0)**2).rolling(n).mean()
    d = (r.clip(upper=0)**2).rolling(n).mean()
    return np.sqrt(u) / np.sqrt(d.replace(0, np.nan))

def feat_v4(df):
    out = pd.DataFrame(index=df.index)
    c, h, l, v, o = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]
    out["rsi_14"] = _rsi(c); ml, ms, mh = _macd(c)
    out["macd"], out["macd_sig"], out["macd_hist"] = ml, ms, mh
    out["bb_z20"] = _bbz(c); out["atr_pct"] = _atr(h, l, c) / c
    out["range_pct"] = (h - l) / c
    vm = v.rolling(30).mean(); vs = v.rolling(30).std().replace(0, np.nan)
    out["vol_z"] = (v - vm) / vs; out["vol_5d"] = v.rolling(5).mean() / vm
    r = c.pct_change(); out["rv_60"] = r.rolling(60).std() * _m.sqrt(252)
    out["overnight_gap"] = (o.shift(-1) / c) - 1.0
    return out

def feat_v5(df):
    out = feat_v4(df)
    c = df["Close"]; r = c.pct_change()
    out["skew_60d"] = r.rolling(60).skew()
    out["semivol_ratio_60d"] = _semi(r, 60)
    out["up_bigdays_60d"] = (r > 0.03).rolling(60).sum()
    return out


# Drop v1 needs the same 11 V7 feats the burst panel already has, plus 6 trend
# feats that v6b/v7 panels don't carry. We compute them per ticker from the
# existing `close` + `atr_pct` columns — no extra yfinance round trip needed.
_DROP_TREND_FEATS = ["ma_stack", "up_streak", "up_bigdays_20d",
                     "dist_ma60_atr", "ma60_slope_60d", "run_length"]


def _add_trend_feats(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    c = g["close"]; r = c.pct_change()
    ma5 = c.rolling(5).mean(); ma20 = c.rolling(20).mean(); ma60 = c.rolling(60).mean()
    atr = g["atr_pct"] * c
    g["ma_stack"] = ((ma5 > ma20) & (ma20 > ma60)).astype(int)
    up = (r > 0).astype(int)
    grp = (up != up.shift()).cumsum()
    g["up_streak"] = up.groupby(grp).cumsum().where(up == 1, 0).clip(upper=30)
    g["up_bigdays_20d"] = (r > 0.03).rolling(20).sum()
    g["dist_ma60_atr"] = (c - ma60) / atr.replace(0, np.nan)
    g["ma60_slope_60d"] = (ma60 - ma60.shift(60)) / c
    above20 = (c > ma20).astype(int)
    grp2 = (above20 != above20.shift()).cumsum()
    g["run_length"] = above20.groupby(grp2).cumsum().where(above20 == 1, 0).clip(upper=120)
    return g


def _build_drop_lookup_v1(panel: pd.DataFrame) -> dict[str, dict]:
    """Legacy 17-feat (V7 + TREND) drop v1 scoring. Used as a fallback when v2
    artifacts / dependencies are missing. Mean p_drop runs ~0.07 in-sample —
    weak discrimination, so v2 is strongly preferred."""
    drop_clf_path = MODELS / "drop_gbc_v1.joblib"
    drop_reg_path = MODELS / "drop_reg_v1_fwd_1d.joblib"
    if not drop_clf_path.exists() or not drop_reg_path.exists():
        print("[daily] drop v1 artifacts missing — skipping drop-risk gate")
        return {}
    clf = joblib.load(drop_clf_path)
    feats_needed = clf["feats"]
    reg = joblib.load(drop_reg_path)["reg"]

    enriched = (panel.groupby("ticker", group_keys=False)
                     .apply(_add_trend_feats))
    latest = (enriched.dropna(subset=feats_needed)
                      .sort_values(["ticker", "date"])
                      .groupby("ticker").tail(1))
    if len(latest) == 0:
        return {}
    X = latest[feats_needed].values
    p_drop = clf["gbc"].predict_proba(X)[:, 1]
    pred_1d = reg.predict(X)
    return {t: {"p_drop": float(p), "pred_1d_drop": float(d)}
            for t, p, d in zip(latest["ticker"], p_drop, pred_1d)}


def _build_drop_lookup_v2(panel: pd.DataFrame) -> dict[str, dict]:
    """Score each ticker's latest row with drop v2 (59-feat: V7+TREND + ranks
    + regime + EDGAR) and apply recency isotonic recal when walk-forward preds
    are available. We skip the per-ticker FinBERT news blend — too expensive
    for all ~500 names and we're using p_drop as a gate, not for ranking.

    Returns {ticker: {"p_drop": float, "pred_1d_drop": float}}.
    """
    feat_list_path = MODELS / "drop_v2_feature_list.json"
    clf_path = MODELS / "drop_gbc_v2_raw.joblib"
    reg_path = MODELS / "drop_reg_v2_fwd_1d.joblib"
    if not (feat_list_path.exists() and clf_path.exists() and reg_path.exists()):
        raise FileNotFoundError("drop v2 artifacts missing")

    # 1. Start from v7 panel, add the 6 TREND feats so it matches the v8 layout
    #    that build_v2_live_features expects. overnight_gap at the latest row
    #    is NaN (feat_v4 computes it as next-day open / today close); fill 0.
    import sys as _s
    _s.path.insert(0, str(ROOT / "code"))
    from features_live_v2 import build_v2_live_features
    from sklearn.isotonic import IsotonicRegression

    enriched = (panel.groupby("ticker", group_keys=False)
                     .apply(_add_trend_feats))
    base_feats = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
                  "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60"] + _DROP_TREND_FEATS
    latest = (enriched.dropna(subset=base_feats)
                      .sort_values(["ticker", "date"])
                      .groupby("ticker").tail(1).copy())
    if len(latest) == 0:
        return {}
    latest["overnight_gap"] = latest.get("overnight_gap", 0.0).fillna(0.0)
    as_of = pd.Timestamp(latest["date"].max())

    # 2. Build the full 59-feature v2 row (ranks + regime + EDGAR)
    latest_v2 = build_v2_live_features(latest, as_of)
    feat_list = json.loads(feat_list_path.read_text())["features"]
    X = latest_v2[feat_list].fillna(0.0).values

    # 3. Raw classifier
    clf = joblib.load(clf_path)
    p_raw = clf.predict_proba(X)[:, 1]

    # 4. Recency isotonic recal — mirrors _score_v2 in 38_drop_daily_runner.py
    p_recal = p_raw.copy()
    wf_path = OUT / "drop_v2" / "walk_forward_predictions.csv"
    if wf_path.exists():
        try:
            wf = pd.read_csv(wf_path, parse_dates=["date"])
            cutoff = as_of - pd.Timedelta(days=60)
            recent = wf[wf["date"] >= cutoff].dropna(subset=["y_drop"])
            if len(recent) >= 500 and recent["y_drop"].sum() > 0:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(recent["p_v2_raw"].values, recent["y_drop"].values)
                p_recal = iso.transform(p_raw)
                print(f"[daily] drop-v2 recal: n={len(recent)}, "
                      f"mean p_raw={p_raw.mean():.3f} -> p_recal={p_recal.mean():.3f}")
        except Exception as e:
            print(f"[daily] drop-v2 recal failed: {type(e).__name__}: {e}; using raw")

    # 5. Regressor ensemble (5 HistGBR models averaged)
    reg_bundle = joblib.load(reg_path)
    reg_X = latest[base_feats + ["overnight_gap"]].fillna(0.0).values
    pred_1d = np.mean([r.predict(reg_X) for r in reg_bundle["models"]], axis=0)

    return {t: {"p_drop": float(p), "pred_1d_drop": float(d)}
            for t, p, d in zip(latest_v2["ticker"], p_recal, pred_1d)}


def build_drop_lookup(panel: pd.DataFrame, version: str = "v2") -> dict[str, dict]:
    """Score every ticker's latest row; returns {ticker: {p_drop, pred_1d_drop}}.
    Missing tickers are treated as p_drop=0 by the consumer (fail open)."""
    if version == "v2":
        try:
            return _build_drop_lookup_v2(panel)
        except Exception as e:
            print(f"[daily] drop v2 lookup failed: {type(e).__name__}: {e}; "
                  f"falling back to v1")
    return _build_drop_lookup_v1(panel)


# ---------- news + pre-market ----------

from news_scorer import get_scorer   # lexicon | lexicon_expanded | finbert

# `score_headline` is now bound at runtime from the config-selected scorer.
# A module-level variable is set in main() before score_universe runs.
score_headline = None   # type: ignore


_FINNHUB_CFG: dict | None = None
_FINNHUB_CALL_TIMES: list[float] = []   # sliding-window timestamps


class FinnhubRateLimiter:
    """Sliding-window rate limiter for Finnhub /quote (free tier = 60/min).

    Uses 58/min by default to leave a 2 req/min cushion for any other code
    in the pipeline that hits Finnhub. Sleeps the minimum required to keep
    the trailing 60-second window under `max_per_min`.
    """
    def __init__(self, max_per_min: int = 58):
        self.max = int(max_per_min)
        self.times: list[float] = []

    def wait(self) -> None:
        import time as _t
        now = _t.time()
        self.times = [t for t in self.times if now - t < 60.0]
        if len(self.times) >= self.max:
            sleep_for = 60.0 - (now - self.times[0]) + 0.05
            if sleep_for > 0:
                _t.sleep(sleep_for)
                now = _t.time()
                self.times = [t for t in self.times if now - t < 60.0]
        self.times.append(now)


def _finnhub_cfg() -> dict | None:
    """Load + cache the Finnhub config. Returns None if missing/invalid so
    callers can short-circuit silently (yfinance fallback path stays valid).
    """
    global _FINNHUB_CFG
    if _FINNHUB_CFG is not None:
        return _FINNHUB_CFG
    p = ROOT / "config" / "finnhub.json"
    if not p.exists():
        return None
    try:
        cfg = json.loads(p.read_text())
        if not cfg.get("api_key"):
            return None
        cfg.setdefault("base_url", "https://finnhub.io/api/v1")
        _FINNHUB_CFG = cfg
        return cfg
    except Exception:
        return None


def finnhub_quote(ticker: str, max_age_seconds: int = 1800,
                   limiter: FinnhubRateLimiter | None = None) -> float | None:
    """Real-time quote via Finnhub /quote endpoint (free tier, 60 req/min).

    Returns the latest price `c`, or None if:
      - config missing
      - HTTP error / non-200
      - quote came back empty (`t == 0` or `c == 0`, used by Finnhub for
        unknown / not-subscribed tickers)
      - quote is older than `max_age_seconds` (default 30 min — keeps us
        safe during weekends / holidays where /quote returns a stale tick)

    If `limiter` is provided, blocks as needed to stay under the rate limit.
    """
    cfg = _finnhub_cfg()
    if cfg is None:
        return None
    if limiter is not None:
        limiter.wait()
    try:
        import requests as _rq
        r = _rq.get(f"{cfg['base_url']}/quote",
                    params={"symbol": ticker, "token": cfg["api_key"]},
                    timeout=5.0)
        if r.status_code != 200:
            return None
        j = r.json()
    except Exception:
        return None
    c = j.get("c"); t = j.get("t") or 0
    if not c or t == 0:
        return None
    # Reject stale ticks (e.g., weekend → still returns last close)
    import time as _time
    if _time.time() - t > max_age_seconds:
        return None
    return float(c)


def refresh_pm_last_finnhub(rows: list[dict],
                             max_unique_tickers: int | None = None,
                             rate_limit_per_min: int = 58) -> int:
    """For each row with a `ticker` and `close_prev`, try Finnhub /quote and
    overwrite `pm_last` (and recompute `overnight_gap_live`) when we get a
    fresh price. Returns the number of rows refreshed.

    Dedup by ticker: a ticker shared across universes only burns one /quote
    call. If `max_unique_tickers` is set, we cap the number of distinct
    tickers we'll touch — picked greedily from the input order (callers
    should pre-sort by priority, e.g. prob_final desc).

    Rate-limited via FinnhubRateLimiter (default 58/min, leaving 2 req/min
    of headroom under Finnhub's 60/min free-tier cap).
    """
    if not rows:
        return 0
    limiter = FinnhubRateLimiter(max_per_min=rate_limit_per_min)
    seen: dict[str, float | None] = {}
    n_refreshed = 0
    for r in rows:
        t = r.get("ticker"); cp = r.get("close_prev")
        if not t or not cp:
            continue
        if t not in seen:
            if max_unique_tickers is not None and len(seen) >= max_unique_tickers:
                # Cap reached — don't fetch new tickers, but if a prior fetch
                # exists for this ticker we already used it via the seen map.
                continue
            seen[t] = finnhub_quote(t, limiter=limiter)
        new_pm = seen[t]
        if new_pm is None:
            continue
        r["pm_last"] = new_pm
        r["pm_last_source"] = "finnhub"
        r["overnight_gap_live"] = new_pm / cp - 1.0
        n_refreshed += 1
    return n_refreshed


def premarket_last(ticker: str, close_date: pd.Timestamp):
    if _LOADER is not None:
        h = _LOADER.intraday("1m", ticker)
    else:
        try:
            h = yf.Ticker(ticker).history(period="2d", interval="1m", prepost=True)
        except Exception:
            return None
    if h is None or len(h) == 0:
        return None
    if _SNAP is not None:
        _SNAP.cache_intraday("1m", ticker, h)
    if _ASOF is not None:
        h = _asof_filter_intraday(h, _ASOF)
        if h is None or len(h) == 0:
            return None
    idx_utc = h.index.tz_convert("UTC")
    cutoff = pd.Timestamp(close_date).tz_localize("America/New_York").tz_convert("UTC") + pd.Timedelta(hours=16)
    after = h[idx_utc > cutoff]
    if len(after) == 0: return None
    return float(after["Close"].iloc[-1])


def news_since(ticker: str, since_utc: pd.Timestamp) -> tuple[int, int, list[str]]:
    if _LOADER is not None:
        items = _LOADER.news(ticker)
    else:
        try: items = yf.Ticker(ticker).news or []
        except Exception: return 0, 0, []
    if _SNAP is not None:
        _SNAP.cache_news(ticker, items)
    if _ASOF is not None:
        items = _asof_filter_news(items, _ASOF)
    n, s, titles = 0, 0, []
    for it in items:
        content = it.get("content") or it
        ts = content.get("pubDate") or it.get("providerPublishTime")
        title = content.get("title") or it.get("title") or ""
        if ts is None: continue
        try:
            dt = pd.Timestamp(ts, unit="s", tz="UTC") if isinstance(ts, (int, float)) \
                 else (pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tz
                       else pd.Timestamp(ts).tz_localize("UTC"))
        except Exception:
            continue
        if dt < since_utc: continue
        s += score_headline(title); n += 1
        titles.append(title[:80])
    return s, n, titles[:3]


# ---------- scoring one universe ----------

def score_universe(tag: str, panel: pd.DataFrame, clf_path: Path,
                    reg_paths: dict, feats: list[str],
                    top_n: int, live: bool,
                    blend_weight: float = 0.15,
                    drop_lookup: dict | None = None,
                    agreement_clf_path: Path | None = None,
                    agreement_feats: list[str] | None = None) -> list[dict]:
    """Score every gate-passing ticker. Returns ALL passing rows (sorted by
    prob_final desc); callers decide which downstream filter to apply.

    Per-row metadata:
        prob_final          primary score (raw v11 prob * news * drop)
        prob_model          raw v11 prob (no news/drop multipliers)
        prob_v11_cal        isotonic-calibrated prob_model (if calibrator
                            present in the v11 bundle)
        prob_v10            agreement model raw prob (if agreement_clf_path
                            provided; same panel, no intraday feats)
        pred_1d/3d/5d       regressor predictions
        pm_last, news_n, news_score, p_drop, ...

    Two-pass to keep FinBERT cheap:
       1. Screen with overnight_gap=0 and no news, keep top news_K.
       2. For those candidates, fetch pre-market + news, compute final score,
          gate on regressor_negative + drop_risk.

    Caller-side filters: see pick_current_list / pick_high_prec_list.
    """
    news_K = max(80, top_n * 16)

    clf = joblib.load(clf_path); gbc = clf["gbc"]
    iso_cal = clf.get("calibrator")  # may be None on older bundles
    clf_version = clf.get("version", "?")
    # Each regressor carries its own `feats` list (may be a subset of the
    # classifier's union when the clf is an ensemble like v13_ens). Use the
    # regressor's own feats at predict time, not the classifier's.
    reg_bundles = {h: joblib.load(p) for h, p in reg_paths.items()}
    regs = {h: b["reg"] for h, b in reg_bundles.items()}
    reg_feats = {h: b.get("feats", feats) for h, b in reg_bundles.items()}
    # The regressor-agreement gate (skip if pred_1d <= 0) was useful for the
    # c2c-trained v11 regressors. v12/v13_ens use intraday regressors with
    # essentially zero MAE skill — their pred_1d is dominated by noise around
    # zero, so the gate would block ~half the universe with no real signal.
    # Disable it for these versions; rely on classifier prob for selection.
    skip_reg_neg_gate = clf_version in ("v12", "v13_ens")

    gbc_v10 = None; v10_feats: list[str] | None = None
    if agreement_clf_path and agreement_feats:
        try:
            v10_bundle = joblib.load(agreement_clf_path)
            gbc_v10 = v10_bundle["gbc"]
            v10_feats = agreement_feats
        except Exception as e:
            print(f"[score {tag}] agreement clf load failed: {e}; skipping")
            gbc_v10 = None

    vanilla_cols = [f for f in feats if f != "overnight_gap"]
    latest = (panel.dropna(subset=vanilla_cols)
                   .sort_values(["ticker", "date"])
                   .groupby("ticker").tail(1).copy())

    # pass 1: screen with overnight_gap=0 and no news
    X = latest.copy()
    if "overnight_gap" in feats:
        X["overnight_gap"] = X.get("overnight_gap", 0.0).fillna(0.0)
    latest["prob_model_screen"] = gbc.predict_proba(X[feats].values)[:, 1]
    top = (latest.sort_values("prob_model_screen", ascending=False)
                  .head(news_K).copy())

    def _score_one(r):
        t = r["ticker"]; close = float(r["close"])
        close_date = pd.Timestamp(r["date"])
        pm = premarket_last(t, close_date)
        since = pd.Timestamp(close_date).tz_localize("America/New_York").tz_convert("UTC") + pd.Timedelta(hours=16)
        sscore, sn, titles = news_since(t, since)
        og_live = (pm / close - 1.0) if pm else 0.0
        feat_row = r[feats].copy()
        feat_row["overnight_gap"] = og_live
        prob = float(gbc.predict_proba([feat_row[feats].values])[:, 1][0])
        preds = {h: float(regs[h].predict([feat_row[reg_feats[h]].values])[0]) for h in regs}
        news_mult = 1 + blend_weight * max(-3, min(3, sscore))

        # Regressor-agreement gate: skip if 1d regressor disagrees.
        # v12/v13_ens regressors have ~0 skill — disable to avoid dropping
        # picks based on noisy pred_1d sign.
        if not skip_reg_neg_gate and preds["fwd_1d"] <= 0:
            return None, "regressor_negative"

        # Drop-risk gate: multiply prob by (1 - p_drop).
        p_drop = 0.0
        if drop_lookup and t in drop_lookup:
            p_drop = max(0.0, min(1.0, drop_lookup[t]["p_drop"]))
        drop_mult = 1.0 - p_drop
        prob_final = prob * news_mult * drop_mult
        gate = "drop_gated" if p_drop >= 0.5 else None

        # Calibrated probability — used only by the high-precision list.
        prob_cal = None
        if iso_cal is not None:
            try:
                prob_cal = float(iso_cal.transform([prob])[0])
            except Exception:
                prob_cal = None

        # Agreement model probability (v10 or whatever was passed).
        prob_v10 = None
        if gbc_v10 is not None and v10_feats:
            try:
                row_v10 = r[v10_feats].copy()
                if "overnight_gap" in v10_feats:
                    row_v10["overnight_gap"] = og_live
                prob_v10 = float(gbc_v10.predict_proba([row_v10[v10_feats].values])[:, 1][0])
            except Exception:
                prob_v10 = None

        return {
            "ticker": t,
            "sector": load_sector_lookup().get(t, "Unknown"),
            "close_prev": close,
            "close_prev_date": close_date.date().isoformat(),
            "pm_last": pm,
            "overnight_gap_live": og_live if pm else None,
            "news_n": int(sn), "news_score": int(sscore),
            "news_titles": titles,
            "prob_model": prob,
            "prob_v11_cal": prob_cal,
            "prob_v10": prob_v10,
            "p_drop": p_drop,
            "prob_final": prob_final,
            # Pred frame depends on the regressor's training target.
            # v11 regressors: c2c (close_prev -> close at +N). Display layer
            #   applies forward-from-pm transform: (1+pred)/(1+gap)-1.
            # v12/v13_ens regressors: intraday (open[t+1] -> close at +N).
            #   Already in the trader's frame (modulo small pm_last vs open
            #   delta). Display should NOT apply the gap transform.
            "_pred_frame": "intraday" if clf_version in ("v12", "v13_ens") else "c2c",
            "_model_version": clf_version,
            "pred_1d": preds["fwd_1d"],
            "pred_3d": preds["fwd_3d"],
            "pred_5d": preds["fwd_5d"],
            "vol_z": float(r.get("vol_z")) if pd.notna(r.get("vol_z")) else None,
            "vol_z_rel": float(r.get("vol_z_rel")) if "vol_z_rel" in r and pd.notna(r.get("vol_z_rel")) else None,
        }, gate

    rows = []
    filtered_stats = {"regressor_negative": 0, "drop_gated": 0}
    tasks = [r for _, r in top.iterrows()]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for row, gate in ex.map(_score_one, tasks):
            if row is None:
                if gate in filtered_stats:
                    filtered_stats[gate] += 1
                continue
            if gate == "drop_gated":
                filtered_stats["drop_gated"] += 1
            rows.append(row)
    print(f"[score {tag}] gates: regressor_negative={filtered_stats['regressor_negative']}, "
          f"drop_gated={filtered_stats['drop_gated']}, passing={len(rows)}")
    rows.sort(key=lambda r: r["prob_final"], reverse=True)
    return rows


def cap_per_sector(rows: list[dict], max_per_sector: int | None) -> list[dict]:
    """Keep at most `max_per_sector` rows per GICS sector, picking by input
    order (callers pass rows pre-sorted by their score, so this naturally
    keeps the highest-scored picks per sector).

    `max_per_sector` <= 0 or None disables the cap. Rows missing a `sector`
    key default to "Unknown" — they're still grouped together, which is the
    right behavior for v7's small-cap tail where sector data may be absent.
    """
    if not max_per_sector or max_per_sector <= 0:
        return list(rows)
    counts: dict[str, int] = {}
    out: list[dict] = []
    for r in rows:
        s = r.get("sector") or "Unknown"
        if counts.get(s, 0) >= max_per_sector:
            continue
        counts[s] = counts.get(s, 0) + 1
        out.append(r)
    return out


def pick_current_list(rows: list[dict], top_n: int,
                       min_prob: float, min_n: int,
                       agreement_min_prob: float | None = None,
                       max_per_sector: int | None = None) -> list[dict]:
    """Apply the existing 'qual' selection plus optional v10 agreement gate.

    Selection rules:
       - prob_final >= min_prob
       - prob_v10  >= agreement_min_prob (if provided AND v10 score available)
       - if fewer than min_n rows clear the gates, fall back to top-min_n by
         prob_final (so the message is never empty on a low-conviction day)
       - cap at max_per_sector picks per GICS sector (if set; correlation control)
       - cap at top_n
    """
    qual = [r for r in rows if r["prob_final"] >= min_prob]
    if agreement_min_prob is not None:
        qual = [r for r in qual
                if (r.get("prob_v10") is None) or (r["prob_v10"] >= agreement_min_prob)]
    if len(qual) < min_n:
        qual = rows[:min_n]
    qual = cap_per_sector(qual, max_per_sector)
    if len(qual) > top_n:
        qual = qual[:top_n]
    return qual


def pick_high_prec_list(all_uni_rows: list[dict], top_n: int,
                         min_cal_prob: float,
                         max_per_sector: int | None = None) -> list[dict]:
    """Cross-universe high-precision picks: calibrated v11 prob >= threshold,
    AND all three regression heads strictly positive *from the trader's actual
    entry price* (pm_last when available, else close_prev).

    The regressors are trained to predict close_prev -> close_{prev+N}. If
    pm_last has already moved, the forward return from entry is
    (1+pred)/(1+gap)-1, which is what the trader will realize. We filter on
    that adjusted return so a name like ORLY (pred +8% but pm gap +8.6%, so
    nothing left) is correctly excluded.

    Dedupes by ticker (keeps the highest cal prob across universes), sorts by
    cal prob desc, caps at top_n. Returns [] if no rows clear the gate.
    """
    eligible = []
    for r in all_uni_rows:
        cp = r.get("prob_v11_cal")
        if cp is None or cp < min_cal_prob:
            continue
        latest = r.get("pm_last")
        close_prev = r.get("close_prev") or 0
        gap = (latest / close_prev - 1.0) if (latest and close_prev) else 0.0
        if abs(1 + gap) < 1e-9:
            continue
        p1 = (1 + r.get("pred_1d", 0)) / (1 + gap) - 1
        p3 = (1 + r.get("pred_3d", 0)) / (1 + gap) - 1
        p5 = (1 + r.get("pred_5d", 0)) / (1 + gap) - 1
        if not (p1 > 0 and p3 > 0 and p5 > 0):
            continue
        eligible.append(r)
    # dedupe by ticker, keep best cal prob
    by_t: dict[str, dict] = {}
    for r in eligible:
        t = r["ticker"]
        if t not in by_t or r["prob_v11_cal"] > by_t[t]["prob_v11_cal"]:
            by_t[t] = r
    out = sorted(by_t.values(), key=lambda r: r["prob_v11_cal"], reverse=True)
    out = cap_per_sector(out, max_per_sector)
    return out[:top_n]


# ---------- accuracy eval ----------

def evaluate_yesterday(yesterday_log: dict, today_panel_v4: pd.DataFrame,
                        today_panel_v5: pd.DataFrame,
                        today_panel_v7: pd.DataFrame | None = None) -> dict:
    """For each ticker in yesterday's log, compute realized 1d/3d/5d return
    from TWO bases:
      - c2c: close on close_prev_date  -> close on close_prev_date + N sessions
        (the model's pure close-to-close skill — what the regressor predicts)
      - entry: pm_last at signal time -> close on close_prev_date + N sessions
        (what a trader who entered at the 8:45 ET pm_last actually got — strips
         out the pre-market gap that fired before the signal was tradeable)
    The displayed prediction in the morning message is "forward from pm_last
    entry" (gap-adjusted), so the entry-based realized return is the apples-to-
    apples eval. The c2c version is kept for back-compat and as the model's
    pure close-skill diagnostic."""
    panels = {"v4": today_panel_v4.sort_values(["ticker", "date"]),
              "v5": today_panel_v5.sort_values(["ticker", "date"])}
    if today_panel_v7 is not None:
        panels["v7"] = today_panel_v7.sort_values(["ticker", "date"])
    report = {k: [] for k in panels}
    for uni_key, preds in yesterday_log.get("predictions", {}).items():
        if uni_key not in panels: continue
        p = panels[uni_key]
        for row in preds:
            t = row["ticker"]; d = pd.Timestamp(row["close_prev_date"])
            g = p[p["ticker"] == t].sort_values("date")
            if g.empty: continue
            base = g[g["date"] == d]
            if base.empty: continue
            base_close = float(base["close"].iloc[0])
            pm_last = row.get("pm_last")
            entry_base = float(pm_last) if pm_last else None
            gap = (entry_base / base_close - 1) if entry_base else 0.0
            after = g[g["date"] > d]
            horizons = {"1d": 1, "3d": 3, "5d": 5}
            realized_c2c, realized_entry = {}, {}
            for hname, nd in horizons.items():
                if len(after) >= nd:
                    fwd_close = float(after["close"].iloc[nd-1])
                    realized_c2c[hname] = fwd_close / base_close - 1
                    realized_entry[hname] = (fwd_close / entry_base - 1) if entry_base else None
                else:
                    realized_c2c[hname] = None
                    realized_entry[hname] = None
            row_out = {"ticker": t, "pm_last": entry_base, "close_prev": base_close,
                       "overnight_gap": gap if entry_base else None,
                       "prob_final": row["prob_final"]}
            for hname in ("1d", "3d", "5d"):
                pred_c2c = row[f"pred_{hname}"]
                # pred_from_entry mirrors the runner's display transform:
                # forward = (1 + pred_c2c) / (1 + gap) - 1
                pred_entry = ((1 + pred_c2c) / (1 + gap) - 1) if entry_base and abs(1 + gap) > 1e-9 else None
                row_out[f"pred_{hname}"] = pred_c2c
                row_out[f"realized_{hname}"] = realized_c2c[hname]
                row_out[f"pred_{hname}_entry"] = pred_entry
                row_out[f"realized_{hname}_entry"] = realized_entry[hname]
            report[uni_key].append(row_out)
    # aggregates: parallel summaries for c2c and entry frames
    summary = {}
    for k, rows in report.items():
        if not rows: continue
        agg = {}
        for h in ["1d", "3d", "5d"]:
            for kind, suf in (("c2c", ""), ("entry", "_entry")):
                vals = [(r[f"pred_{h}{suf}"], r[f"realized_{h}{suf}"]) for r in rows
                        if r.get(f"realized_{h}{suf}") is not None
                        and r.get(f"pred_{h}{suf}") is not None]
                if not vals: continue
                err = [abs(p - a) for (p, a) in vals]
                dir_hit = [int(np.sign(p) == np.sign(a)) for (p, a) in vals]
                d_key = h if kind == "c2c" else f"{h}_entry"
                agg[d_key] = {"n": len(vals), "MAE": float(np.mean(err)),
                              "dir_hit": float(np.mean(dir_hit)),
                              "mean_pred": float(np.mean([p for p, _ in vals])),
                              "mean_realized": float(np.mean([a for _, a in vals]))}
        summary[k] = agg
    return {"summary": summary, "rows": report}


def evaluate_high_prec(yesterday_log: dict, panels: dict[str, pd.DataFrame]) -> dict | None:
    """Score yesterday's cross-universe `high_prec` shortlist (the buy list)
    against now-observable 1d returns.

    Tracked separately from per-universe `eval` so we can decide when to
    raise `high_prec_min_cal_prob` (the gate that promotes a name to the
    buy list). Mirrors evaluate_yesterday's per-row schema but adds
    prob_v11_cal so we can correlate confidence with realized outcome.

    Returns None if there were no high_prec picks yesterday or no panels
    available.
    """
    hp = yesterday_log.get("high_prec") or []
    if not hp or not panels:
        return None
    rows = []
    for row in hp:
        t = row["ticker"]
        d = pd.Timestamp(row["close_prev_date"])
        # Each high_prec row carries _universe; use that panel for lookup.
        uni = row.get("_universe")
        p = panels.get(uni) if uni else None
        if p is None:
            # Fall back: try any panel that has the ticker
            for pn in panels.values():
                if pn is not None and (pn["ticker"] == t).any():
                    p = pn; break
        if p is None: continue
        g = p[p["ticker"] == t].sort_values("date")
        base = g[g["date"] == d]
        if base.empty: continue
        base_close = float(base["close"].iloc[0])
        after = g[g["date"] > d]
        if after.empty: continue
        fwd_close = float(after["close"].iloc[0])
        realized_1d = fwd_close / base_close - 1
        # Trade-realistic frame: realized return for a trader who entered at
        # the 8:45 ET pm_last signal price (not at yesterday's close, which is
        # not actionable). Strips the overnight gap from the realized return.
        pm_last = row.get("pm_last")
        entry_base = float(pm_last) if pm_last else None
        realized_1d_entry = (fwd_close / entry_base - 1) if entry_base else None
        gap = (entry_base / base_close - 1) if entry_base else 0.0
        pred_1d_c2c = row["pred_1d"]
        pred_1d_entry = ((1 + pred_1d_c2c) / (1 + gap) - 1) if entry_base and abs(1 + gap) > 1e-9 else None
        rows.append({
            "ticker": t, "_universe": uni,
            "prob_v11_cal": row.get("prob_v11_cal"),
            "prob_final": row.get("prob_final"),
            "pred_1d": pred_1d_c2c, "realized_1d": realized_1d,
            "pred_1d_entry": pred_1d_entry, "realized_1d_entry": realized_1d_entry,
            "pm_last": entry_base, "close_prev": base_close,
        })
    if not rows:
        return None
    n = len(rows)
    dir_hit = float(np.mean([int(np.sign(r["pred_1d"]) == np.sign(r["realized_1d"]))
                              for r in rows]))
    mae = float(np.mean([abs(r["pred_1d"] - r["realized_1d"]) for r in rows]))
    pos = sum(1 for r in rows if r["realized_1d"] > 0)
    # Trade-realistic aggregates (entry frame). Skip rows where pm_last was
    # missing so the entry stats reflect only picks with a real entry price.
    erows = [r for r in rows if r.get("realized_1d_entry") is not None
                                and r.get("pred_1d_entry") is not None]
    en = len(erows)
    if en:
        e_dir = float(np.mean([int(np.sign(r["pred_1d_entry"]) == np.sign(r["realized_1d_entry"]))
                                for r in erows]))
        e_mae = float(np.mean([abs(r["pred_1d_entry"] - r["realized_1d_entry"]) for r in erows]))
        e_up  = sum(1 for r in erows if r["realized_1d_entry"] > 0) / en
        e_mp  = float(np.mean([r["pred_1d_entry"] for r in erows]))
        e_mr  = float(np.mean([r["realized_1d_entry"] for r in erows]))
    else:
        e_dir = e_mae = e_up = e_mp = e_mr = None
    # Per-universe breakdown for both frames.
    by_universe = {}
    for r in rows:
        uni = r.get("_universe") or "?"
        by_universe.setdefault(uni, []).append(r)
    by_universe_summary = {}
    for uni, urows in by_universe.items():
        un = len(urows)
        e_urows = [r for r in urows if r.get("realized_1d_entry") is not None
                                       and r.get("pred_1d_entry") is not None]
        eun = len(e_urows)
        s = {
            "n": un,
            "dir_hit": float(np.mean([int(np.sign(r["pred_1d"]) == np.sign(r["realized_1d"]))
                                       for r in urows])),
            "MAE": float(np.mean([abs(r["pred_1d"] - r["realized_1d"]) for r in urows])),
            "up_rate": sum(1 for r in urows if r["realized_1d"] > 0) / un,
            "mean_pred": float(np.mean([r["pred_1d"] for r in urows])),
            "mean_realized": float(np.mean([r["realized_1d"] for r in urows])),
        }
        if eun:
            s.update({
                "n_entry": eun,
                "dir_hit_entry": float(np.mean([int(np.sign(r["pred_1d_entry"]) == np.sign(r["realized_1d_entry"]))
                                                 for r in e_urows])),
                "MAE_entry": float(np.mean([abs(r["pred_1d_entry"] - r["realized_1d_entry"]) for r in e_urows])),
                "up_rate_entry": sum(1 for r in e_urows if r["realized_1d_entry"] > 0) / eun,
                "mean_pred_entry": float(np.mean([r["pred_1d_entry"] for r in e_urows])),
                "mean_realized_entry": float(np.mean([r["realized_1d_entry"] for r in e_urows])),
            })
        by_universe_summary[uni] = s
    return {
        "n": n, "dir_hit": dir_hit, "MAE": mae,
        "up_rate": pos / n,
        "mean_pred": float(np.mean([r["pred_1d"] for r in rows])),
        "mean_realized": float(np.mean([r["realized_1d"] for r in rows])),
        "n_entry": en, "dir_hit_entry": e_dir, "MAE_entry": e_mae,
        "up_rate_entry": e_up, "mean_pred_entry": e_mp, "mean_realized_entry": e_mr,
        "rows": rows,
        "by_universe": by_universe_summary,
    }


# ---------- notification backend (telegram / iMessage) ----------

import sys as _sys_notify
_sys_notify.path.insert(0, str(ROOT / "code"))
from notify import send as _notify_send   # noqa: E402
from regime_features import attach_regime, REGIME_FEATS  # noqa: E402
# Importing ensemble_classifier so joblib unpickling at inference time can
# resolve EnsembleClassifier / IdentityCalibrator from the v13_ens bundle.
from ensemble_classifier import EnsembleClassifier, IdentityCalibrator  # noqa: F401, E402


# ---------- model-version dispatch (v7 / v10 / v11) ----------
# Base feature lists per universe (must match what each model was trained on).
_V7_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
            "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap"]
_V5_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
            "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60",
            "skew_60d", "semivol_ratio_60d", "up_bigdays_60d", "overnight_gap"]

# v11 adds pre-market and intraday features built by 71_intraday_features.py
# from the local store at data/intraday_store/. pm_volume_ratio is excluded
# because yfinance pre-market 1m bars carry volume=0.
_INTRADAY_FEATS = ["pm_gap_pct", "pm_range_pct", "pm_drift",
                   "ret_first_30m", "ret_last_30m", "intraday_rv_5m",
                   "vwap_close_dist", "intraday_dd", "vol_late_share",
                   "close_strength"]

# v13 adds yesterday-intraday features derived from daily OHLC (full historical
# coverage). Computed inline at inference time from data/ohlc_prices.csv —
# everything anchored on the row's own date (no look-ahead).
_YEST_INTRA_FEATS = [
    "yest_oc_ret", "yest_close_in_range", "yest_pullback_pct",
    "yest_open_to_low", "yest_high_low_pct", "yest_volume_zscore",
    "yest_oc_streak_5d", "yest_oc_mean_5d", "yest_oc_vol_20d",
    "yest_close_vs_ma20",
]


def attach_intraday(panel: pd.DataFrame) -> pd.DataFrame:
    """Left-join data/intraday_daily_features.csv onto a panel by (ticker, date).
    Missing keys -> NaN (HGB models handle NaN natively)."""
    feat_path = DATA / "intraday_daily_features.csv"
    if not feat_path.exists():
        for c in _INTRADAY_FEATS:
            if c not in panel.columns:
                panel[c] = np.nan
        return panel
    f = pd.read_csv(feat_path, parse_dates=["date"])
    f["date"] = f["date"].dt.normalize()
    keep = ["ticker", "date"] + [c for c in _INTRADAY_FEATS if c in f.columns]
    f = f[keep]
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    return panel.merge(f, on=["ticker", "date"], how="left")


_OHLC_CACHE = {"frame": None, "yi": None}
_HORIZON_LOOKUP_CACHE = {"loaded": False, "data": None}


def horizon_lookup(universe: str, cal_prob: float | None,
                    horizon: str = "5d") -> tuple[float | None, float | None]:
    """Look up (mean realized return, win rate) for (universe, cal_prob) from
    the v13_ens prob-bucket historical-mean table built by
    79_burst_v13_horizon_lookup.py. Replaces the noisy regressor pred_Nd
    display in the morning message — the classifier ranks well, but the
    regressor predictions are zero-skill noise; bucket means are honest.

    Returns (mean, win_rate) as fractions, or (None, None) if no data."""
    if not _HORIZON_LOOKUP_CACHE["loaded"]:
        path = MODELS / "burst_v13_ens_horizon_lookup.json"
        if path.exists():
            _HORIZON_LOOKUP_CACHE["data"] = json.loads(path.read_text())
        _HORIZON_LOOKUP_CACHE["loaded"] = True
    data = _HORIZON_LOOKUP_CACHE["data"]
    if data is None or cal_prob is None: return (None, None)
    buckets = data.get("per_universe", {}).get(universe) or []
    for b in buckets:
        if b["lo"] <= cal_prob < b["hi"]:
            return (b.get(f"{horizon}_mean"), b.get(f"{horizon}_wr"))
    return (None, None)


def attach_yest_intraday(panel: pd.DataFrame) -> pd.DataFrame:
    """Left-join yesterday-intraday features (computed from data/ohlc_prices.csv)
    onto a panel by (ticker, date). All features are anchored on the row's own
    date (so they reflect closing-day intraday character — no look-ahead).
    Missing keys -> NaN."""
    ohlc_path = DATA / "ohlc_prices.csv"
    if not ohlc_path.exists():
        for c in _YEST_INTRA_FEATS:
            if c not in panel.columns:
                panel[c] = np.nan
        return panel
    if _OHLC_CACHE["yi"] is None:
        ohlc = pd.read_csv(ohlc_path, parse_dates=["date"])
        ohlc["date"] = ohlc["date"].dt.normalize()
        ohlc = ohlc.sort_values(["ticker", "date"]).reset_index(drop=True)
        g = ohlc.groupby("ticker", sort=False)
        rng = (ohlc["high"] - ohlc["low"]).replace(0, np.nan)
        oc = (ohlc["close"] - ohlc["open"]) / ohlc["open"]
        ohlc["yest_oc_ret"] = oc
        ohlc["yest_close_in_range"] = (ohlc["close"] - ohlc["low"]) / rng
        ohlc["yest_pullback_pct"] = (ohlc["high"] - ohlc["close"]) / ohlc["close"]
        ohlc["yest_open_to_low"] = (ohlc["open"] - ohlc["low"]) / ohlc["open"]
        ohlc["yest_high_low_pct"] = (ohlc["high"] - ohlc["low"]) / ohlc["open"]
        ohlc["yest_oc_mean_5d"] = g["yest_oc_ret"].transform(lambda s: s.rolling(5, min_periods=3).mean())
        ohlc["yest_oc_vol_20d"] = g["yest_oc_ret"].transform(lambda s: s.rolling(20, min_periods=10).std())
        ohlc["yest_oc_streak_5d"] = g["yest_oc_ret"].transform(
            lambda s: s.rolling(5, min_periods=3).apply(lambda w: float((w > 0).sum()), raw=False))
        vmean20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
        vstd20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).std())
        ohlc["yest_volume_zscore"] = (ohlc["volume"] - vmean20) / vstd20.replace(0, np.nan)
        cma20 = g["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
        ohlc["yest_close_vs_ma20"] = (ohlc["close"] - cma20) / cma20
        _OHLC_CACHE["yi"] = ohlc[["ticker", "date"] + _YEST_INTRA_FEATS]
    yi = _OHLC_CACHE["yi"]
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    return panel.merge(yi, on=["ticker", "date"], how="left")


def _model_spec(uni: str, version: str) -> dict:
    """Return {clf, regs (dict), feats, needs_regime, needs_intraday, needs_yest_intra}
    for one (universe, version)."""
    # v13_ens (default after May 2026): v12+v13 ensemble. v12 = intraday-target
    # GBC with v11 features. v13 = same target, plus yest-intraday OHLC features.
    # Bundle file is built by 78_burst_v13_ens.py and presents a single
    # predict_proba that averages calibrated probs.
    # Regressors: reuse v12's intraday regressors (v13's were no better).
    # Naming convention preserved: regs dict keys are still "fwd_1d/3d/5d" so
    # downstream consumers don't change — but the values are intraday-trained
    # (close at +N sessions / open[t+1] - 1). pred_Nd in the message is now
    # "intraday return (open-of-day-of-pick to close N sessions later)".
    if version == "v13_ens":
        feats_base = {"v4": _V7_BASE, "v5": _V5_BASE, "v7": _V7_BASE}[uni]
        return {
            "clf":  MODELS / f"burst_gbc_v13_ens_{uni}.joblib",
            "regs": {"fwd_1d": MODELS / f"burst_reg_v12_{uni}_intra_1d.joblib",
                     "fwd_3d": MODELS / f"burst_reg_v12_{uni}_intra_3d.joblib",
                     "fwd_5d": MODELS / f"burst_reg_v12_{uni}_intra_5d.joblib"},
            "feats": feats_base + _YEST_INTRA_FEATS + REGIME_FEATS + _INTRADAY_FEATS,
            "needs_regime": True,
            "needs_intraday": True,
            "needs_yest_intra": True,
        }
    # v12: same as v13_ens but without the v13 component. Useful as fallback
    # if ensemble has issues, or for ablation.
    if version == "v12":
        feats_base = {"v4": _V7_BASE, "v5": _V5_BASE, "v7": _V7_BASE}[uni]
        return {
            "clf":  MODELS / f"burst_gbc_v12_{uni}.joblib",
            "regs": {"fwd_1d": MODELS / f"burst_reg_v12_{uni}_intra_1d.joblib",
                     "fwd_3d": MODELS / f"burst_reg_v12_{uni}_intra_3d.joblib",
                     "fwd_5d": MODELS / f"burst_reg_v12_{uni}_intra_5d.joblib"},
            "feats": feats_base + REGIME_FEATS + _INTRADAY_FEATS,
            "needs_regime": True,
            "needs_intraday": True,
            "needs_yest_intra": False,
        }
    if version == "v11":
        feats_base = {"v4": _V7_BASE, "v5": _V5_BASE, "v7": _V7_BASE}[uni]
        return {
            "clf":  MODELS / f"burst_gbc_v11_{uni}.joblib",
            "regs": {h: MODELS / f"burst_reg_v11_{uni}_{h}.joblib"
                     for h in ("fwd_1d", "fwd_3d", "fwd_5d")},
            "feats": feats_base + REGIME_FEATS + _INTRADAY_FEATS,
            "needs_regime": True,
            "needs_intraday": True,
            "needs_yest_intra": False,
        }
    if version == "v10":
        feats_base = {"v4": _V7_BASE, "v5": _V5_BASE, "v7": _V7_BASE}[uni]
        return {
            "clf":  MODELS / f"burst_gbc_v10_{uni}.joblib",
            "regs": {h: MODELS / f"burst_reg_v10_{uni}_{h}.joblib"
                     for h in ("fwd_1d", "fwd_3d", "fwd_5d")},
            "feats": feats_base + REGIME_FEATS,
            "needs_regime": True,
            "needs_intraday": False,
            "needs_yest_intra": False,
        }
    # default/legacy v7 track (production models as of Apr 18)
    legacy = {
        "v4": ("burst_gbc_v6b_augmented", "burst_reg_v6b", _V7_BASE),
        "v5": ("burst_gbc_v6_augmented",  "burst_reg_v6",  _V5_BASE),
        "v7": ("burst_gbc_v7_augmented",  "burst_reg_v7",  _V7_BASE),
    }[uni]
    clf_name, reg_prefix, base = legacy
    return {
        "clf":  MODELS / f"{clf_name}.joblib",
        "regs": {h: MODELS / f"{reg_prefix}_{h}.joblib"
                 for h in ("fwd_1d", "fwd_3d", "fwd_5d")},
        "feats": base,
        "needs_regime": False,
        "needs_intraday": False,
        "needs_yest_intra": False,
    }


def _resolve_model_version(cfg: dict) -> str:
    """Pick version from config with availability fallback.

    Preference order: v11 -> v10 -> v7. Users can force any of these by
    setting `burst_model_version` in config.
    """
    want = cfg.get("burst_model_version")
    if want in ("v7", "v10", "v11", "v12", "v13_ens"):
        return want
    unis = cfg.get("universes", ["v4", "v5", "v7"])
    for cand in ("v13_ens", "v12", "v11", "v10"):
        ok = True
        for uni in unis:
            spec = _model_spec(uni, cand)
            if not spec["clf"].exists() or not all(p.exists() for p in spec["regs"].values()):
                ok = False
                break
        if ok:
            return cand
    return "v7"


# ---------- catalyst-surprise scoring ----------

import sys as _sys_cat
_sys_cat.path.insert(0, str(ROOT / "code"))
from importlib import import_module as _import_module
try:
    _cat_feat_mod = _import_module("62_catalyst_features")
except Exception as _e:
    print(f"[daily] catalyst module not importable: {_e}")
    _cat_feat_mod = None


def _empty_catalyst_features() -> dict:
    out = {"news_n": 0, "news_score": 0, "news_strong": 0,
            "news_primary_n": 0, "pub_div": 0, "novelty": 0.0}
    if _cat_feat_mod is not None and hasattr(_cat_feat_mod, "EVENT_PATTERNS"):
        for k in _cat_feat_mod.EVENT_PATTERNS: out[f"evt_{k}"] = 0
    return out


def _catalyst_news_features_today(ticker: str, market_day) -> dict:
    """Aggregate news attributed to `market_day` for one ticker. Mirrors the
    training-time aggregation in 62_catalyst_features (post-16:00 ET prev day
    + pre-09:30 ET current day). Returns base counts + event-type flags.
    """
    if _cat_feat_mod is None:
        return _empty_catalyst_features()
    n = _cat_feat_mod.load_news(ticker)
    if len(n) == 0:
        return _empty_catalyst_features()
    md = _cat_feat_mod.assign_market_day(n["ts"])
    n = n.assign(market_day=md)
    et = n["ts"].dt.tz_convert("America/New_York")
    preopen = (et.dt.hour < 9) | ((et.dt.hour == 9) & (et.dt.minute < 30))
    n = n[preopen | (et.dt.hour >= 16)]
    today = n[n["market_day"] == market_day].copy()
    if len(today) == 0:
        return _empty_catalyst_features()
    today["toks"] = today["headline"].map(_cat_feat_mod.tok)
    pos = today["toks"].map(lambda s: len(s & _cat_feat_mod.POS_WORDS) > 0).sum()
    neg = today["toks"].map(lambda s: len(s & _cat_feat_mod.NEG_WORDS) > 0).sum()
    strong = today["headline"].map(
        lambda h: bool(_cat_feat_mod.STRONG_RE.search(h or ""))).sum()
    primary = today["source"].isin(_cat_feat_mod.PRIMARY_SOURCES).sum()
    prior_md = sorted(d for d in set(n["market_day"]) if d < market_day)[-5:]
    prior_toks = set()
    for d in prior_md:
        for h in n[n["market_day"] == d]["headline"]:
            prior_toks |= _cat_feat_mod.tok(h or "")
    cur_toks = set().union(*today["toks"].tolist()) if len(today) else set()
    inter = len(cur_toks & prior_toks); union = len(cur_toks | prior_toks) or 1
    novelty = 1.0 - inter / union if cur_toks else 0.0
    out = {
        "news_n": int(len(today)), "news_score": int(pos - neg),
        "news_strong": int(strong), "news_primary_n": int(primary),
        "pub_div": int(today["source"].nunique()), "novelty": float(novelty),
    }
    if hasattr(_cat_feat_mod, "EVENT_PATTERNS"):
        for kind, pat in _cat_feat_mod.EVENT_PATTERNS.items():
            out[f"evt_{kind}"] = int(today["headline"].map(
                lambda h, p=pat: bool(p.search(h or ""))).sum())
    return out


def _score_catalyst_with_bundle(bundle: dict, panel_v7: pd.DataFrame,
                                  market_day) -> pd.DataFrame:
    """Score every ticker with a given catalyst-model bundle. Returns a
    DataFrame with p_catalyst + the features used for display + gates."""
    gbc = bundle["gbc"]; scaler = bundle["scaler"]; feats = bundle["feats"]
    last = (panel_v7.sort_values(["ticker", "date"])
                    .groupby("ticker").tail(1).reset_index(drop=True))
    rows = []
    for r in last.itertuples(index=False):
        t = r.ticker
        nf = _catalyst_news_features_today(t, market_day)
        tg = panel_v7[panel_v7["ticker"] == t].sort_values("date")
        prior_5d_ret = (float(tg["close"].iloc[-1] / tg["close"].iloc[-6] - 1)
                         if len(tg) >= 6 else None)
        if prior_5d_ret is None: continue
        rows.append({
            "ticker": t,
            "close_prev": float(r.close),
            "close_prev_date": pd.Timestamp(r.date).date().isoformat(),
            **nf,
            "prior_5d_ret": prior_5d_ret, "overnight_gap": 0.0,
            "atr_pct": float(r.atr_pct) if pd.notna(r.atr_pct) else 0.0,
            "rv_60": float(r.rv_60) if pd.notna(r.rv_60) else 0.0,
            "rsi_14": float(r.rsi_14) if pd.notna(r.rsi_14) else 50.0,
            "bb_z20": float(r.bb_z20) if pd.notna(r.bb_z20) else 0.0,
            "vol_z": float(r.vol_z) if pd.notna(r.vol_z) else 0.0,
            "macd_hist": float(r.macd_hist) if pd.notna(r.macd_hist) else 0.0,
            # FinBERT placeholders — filled in if feats expect them
            "finbert_max": 0.0, "finbert_mean": 0.0, "finbert_n": 0,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in feats:
        if c not in df.columns: df[c] = 0
    X = scaler.transform(df[feats])
    df["p_catalyst"] = gbc.predict_proba(X)[:, 1]
    return df


def score_catalyst(panel_v7: pd.DataFrame, top_n: int,
                    market_day_override=None,
                    drop_lookup: dict | None = None,
                    apply_gates: bool = True) -> list[dict]:
    """Rank the v7 universe by P(next-day >= +5%). Preferred model is
    catalyst_gbc_best.joblib (trained with event-type features, and/or FinBERT
    if available); falls back to catalyst_gbc.joblib. Optional gates:

      gate_runner : drop picks with news_n == 0 AND prior_5d_ret > +20% —
                    kills momentum-extended names with no news anchor.
      gate_drop   : drop picks where drop model flags the ticker at p_drop
                    >= 0.7 — avoids the "up model vs down model conflict"
                    (today's AXTI was in both baskets; the short won).
    """
    clf_path = MODELS / "catalyst_gbc_best.joblib"
    if not clf_path.exists(): clf_path = MODELS / "catalyst_gbc.joblib"
    if not clf_path.exists() or _cat_feat_mod is None:
        print("[catalyst] model or feature module missing — skipping")
        return []
    bundle = joblib.load(clf_path)
    market_day = (market_day_override if market_day_override is not None
                   else datetime.now().date())
    df = _score_catalyst_with_bundle(bundle, panel_v7, market_day)
    if len(df) == 0: return []

    n_pre = len(df)
    if apply_gates:
        mask_runner = (df["news_n"] == 0) & (df["prior_5d_ret"] > 0.20)
        if drop_lookup:
            mask_drop = df["ticker"].map(
                lambda t: drop_lookup.get(t, {}).get("p_drop", 0.0) if
                    isinstance(drop_lookup.get(t, 0.0), dict)
                    else drop_lookup.get(t, 0.0)) >= 0.7
        else:
            mask_drop = pd.Series([False] * len(df), index=df.index)
        df = df[~(mask_runner | mask_drop)]
        print(f"[catalyst] gates removed {n_pre - len(df)}/{n_pre} tickers "
              f"(runner={int(mask_runner.sum())}, drop={int(mask_drop.sum())}); "
              f"model={clf_path.name}")
    return (df.sort_values("p_catalyst", ascending=False)
              .head(top_n).to_dict(orient="records"))


def score_catalyst_short(panel_v7: pd.DataFrame, top_n: int,
                          market_day_override=None) -> list[dict]:
    """Catalyst-downside bucket: P(fwd_1d <= -5%). Today's ablation showed the
    down head would have correctly flagged CAR, AXTI, ALMU, BNAI — names the
    upside head got wrong. Short candidates."""
    clf_path = MODELS / "catalyst_gbc_down.joblib"
    if not clf_path.exists() or _cat_feat_mod is None:
        return []
    bundle = joblib.load(clf_path)
    market_day = (market_day_override if market_day_override is not None
                   else datetime.now().date())
    df = _score_catalyst_with_bundle(bundle, panel_v7, market_day)
    if len(df) == 0: return []
    df = df.rename(columns={"p_catalyst": "p_down"})
    return (df.sort_values("p_down", ascending=False)
              .head(top_n).to_dict(orient="records"))


# ---------- formatting ----------

def format_message(today_iso: str, by_uni: dict, yest_eval: dict | None,
                    catalyst: list[dict] | None = None,
                    catalyst_eval: dict | None = None,
                    catalyst_short: list[dict] | None = None,
                    high_prec: list[dict] | None = None,
                    high_prec_eval: dict | None = None) -> str:
    L = []
    L.append(f"📈 Burst predictions — {today_iso}")
    # accuracy from yesterday
    if yest_eval and yest_eval.get("summary"):
        L.append("")
        # Entry frame only: realized return is from pm_last (signal price at
        # 8:45 ET) to close. The overnight gap fired BEFORE the signal was
        # tradeable; crediting it to the model would overstate effectiveness.
        L.append("Yesterday's predictions (realized from pm_last entry):")
        for uni, agg in yest_eval["summary"].items():
            for h in ["1d", "3d", "5d"]:
                ae = agg.get(f"{h}_entry")
                if not ae: continue
                L.append(f"  {uni} {h}: dir {ae['dir_hit']*100:.0f}%  "
                         f"pred {ae['mean_pred']*100:+.2f}% / real {ae['mean_realized']*100:+.2f}%  "
                         f"MAE {ae['MAE']*100:.2f}%  (n={ae['n']})")
        if catalyst_eval and catalyst_eval.get("n"):
            L.append(f"  catalyst 1d: hit5%={catalyst_eval['hit_5pct']*100:.0f}%  "
                     f"up={catalyst_eval['up_rate']*100:.0f}%  "
                     f"mean {catalyst_eval['mean_ret']*100:+.2f}% "
                     f"(n={catalyst_eval['n']})")
        if high_prec_eval and high_prec_eval.get("n_entry"):
            L.append(f"  high_prec 1d: dir {high_prec_eval['dir_hit_entry']*100:.0f}%  "
                     f"up={high_prec_eval['up_rate_entry']*100:.0f}%  "
                     f"mean {high_prec_eval['mean_realized_entry']*100:+.2f}%  "
                     f"(n={high_prec_eval['n_entry']})")
            # Per-universe high_prec breakdown — surfaces v7's contribution
            # even when yesterday's predictions[v7] was empty (the common case
            # for picks promoted via the cross-universe cal_prob gate).
            for uni, a in (high_prec_eval.get("by_universe") or {}).items():
                if not a.get("n_entry"): continue
                L.append(f"    {uni}: dir {a['dir_hit_entry']*100:.0f}%  "
                         f"up={a['up_rate_entry']*100:.0f}%  "
                         f"mean {a['mean_realized_entry']*100:+.2f}%  (n={a['n_entry']})")
            # Per-pick reflection: predicted vs realized return from pm_last
            # to close. The ✓/✗ marks what the trader actually got.
            for r in (high_prec_eval.get("rows") or []):
                rent = r.get("realized_1d_entry")
                pent = r.get("pred_1d_entry")
                if rent is None: continue
                tag = "✓" if rent > 0 else "✗"
                pent_s = f"{pent*100:+.1f}%" if pent is not None else "—"
                L.append(f"    {r['ticker']:5s} [{r.get('_universe','?')}] "
                         f"pred {pent_s} / real {rent*100:+.1f}%  {tag}")
    # Show forward-from-entry predictions instead of close_prev→close_next,
    # so already-happened pre-market moves aren't counted in the displayed
    # prediction. Entry price = pm_last (what the trader can actually enter
    # at after the 8:45 text). Forward return from entry:
    #   fwd_Nd = (1 + pred_Nd) / (1 + overnight_gap_live) - 1
    # This is what you'd capture if you bought at pm_last and held to close
    # N sessions later. Toggled off by setting
    # `morning_show_forward_from_entry: false` in burst_daily.json, in which
    # case the raw close-prev anchored prediction is shown.
    cfg_forward = bool(load_config().get(
        "morning_show_forward_from_entry", True))
    L.append("")
    for uni, rows in by_uni.items():
        header = "Top " + uni
        # For v13_ens picks, the displayed 3d/5d are PROB-BUCKET HISTORICAL
        # MEANS (not regressor predictions \u2014 those have ~0 MAE skill). For v11,
        # they're the regressor's forward-from-pm prediction.
        is_v13 = any(r.get("_pred_frame") == "intraday" for r in rows)
        if is_v13:
            header += "  (3d/5d = historical mean for this prob bucket; hold-5d is the design horizon)"
        elif cfg_forward:
            header += "  (pred = forward return from pm_last entry)"
        L.append(header + ":")
        for r in rows:
            latest = r.get("pm_last")
            close_prev = r.get("close_prev") or r.get("close", 0)
            if latest and close_prev:
                gap = latest / close_prev - 1
                price_str = (f"${close_prev:.2f}\u2192${latest:.2f} ({gap*100:+.1f}%)")
            else:
                gap = 0.0
                price_str = f"${close_prev:.2f}"
            pred_frame = r.get("_pred_frame", "c2c")
            if pred_frame == "intraday":
                # Use prob-bucket historical means (drop 1d \u2014 too noisy to act on).
                cp = r.get("prob_v11_cal") or r.get("prob_model") or 0.0
                m3, w3 = horizon_lookup(uni, cp, "3d")
                m5, w5 = horizon_lookup(uni, cp, "5d")
                m3s = f"{m3*100:+.1f}%" if m3 is not None else "\u2014"
                m5s = f"{m5*100:+.1f}%" if m5 is not None else "\u2014"
                w3s = f"{w3*100:.0f}%" if w3 is not None else "\u2014"
                w5s = f"{w5*100:.0f}%" if w5 is not None else "\u2014"
                L.append(f"  {r['ticker']:5s} {price_str}  p={r['prob_final']*100:.0f}%  "
                         f"3d ~{m3s} (wr {w3s}) / 5d ~{m5s} (wr {w5s})  "
                         f"news {r['news_score']:+d}")
            else:
                p1 = r.get("pred_1d", 0.0); p3 = r.get("pred_3d", 0.0); p5 = r.get("pred_5d", 0.0)
                if cfg_forward and latest and abs(1 + gap) > 1e-9:
                    p1 = (1 + p1) / (1 + gap) - 1
                    p3 = (1 + p3) / (1 + gap) - 1
                    p5 = (1 + p5) / (1 + gap) - 1
                L.append(f"  {r['ticker']:5s} {price_str}  p={r['prob_final']*100:.0f}%  "
                         f"1d {p1*100:+.1f}% / 3d {p3*100:+.1f}% / "
                         f"5d {p5*100:+.1f}%  news {r['news_score']:+d}")
    if high_prec:
        L.append("")
        # Header reflects what selection actually does. v13_ens uses
        # cal_prob >= threshold + cross-universe dedupe + sector cap.
        # The "1d/3d/5d all >0" gate was retained but is essentially a
        # coin flip on v12/v13_ens regressors (zero MAE skill).
        is_v13 = any(r.get("_pred_frame") == "intraday" for r in high_prec)
        if is_v13:
            L.append("🎯 High-precision picks (cal_p ≥ thr; 3d/5d shown = bucket historical mean):")
        else:
            L.append("🎯 High-precision picks (calibrated v11 + 1d/3d/5d all >0):")
        for r in high_prec:
            uni_tag = r.get("_universe", "?")
            latest = r.get("pm_last")
            close_prev = r.get("close_prev") or r.get("close", 0)
            if latest and close_prev:
                gap = latest / close_prev - 1
                price_str = f"${close_prev:.2f}\u2192${latest:.2f} ({gap*100:+.1f}%)"
            else:
                gap = 0.0
                price_str = f"${close_prev:.2f}"
            cal = r.get("prob_v11_cal") or 0.0
            vzr = r.get("vol_z_rel")
            # vol_z_rel marker: ✓ if >1 (idiosyncratic surge — historically 100% dir_hit
            # on top-conviction picks), · if in [0,1) "trap zone", or signed value otherwise
            if vzr is None:
                vzr_str = ""
            elif vzr > 1.0:
                vzr_str = f"  vzr=✓{vzr:+.1f}"
            elif 0 <= vzr <= 1.0:
                vzr_str = f"  vzr=·{vzr:+.1f}"
            else:
                vzr_str = f"  vzr={vzr:+.1f}"
            pred_frame = r.get("_pred_frame", "c2c")
            if pred_frame == "intraday":
                # v13_ens: drop 1d, show prob-bucket historical means for 3d/5d.
                m3, w3 = horizon_lookup(uni_tag, cal, "3d")
                m5, w5 = horizon_lookup(uni_tag, cal, "5d")
                m3s = f"{m3*100:+.1f}%" if m3 is not None else "—"
                m5s = f"{m5*100:+.1f}%" if m5 is not None else "—"
                w3s = f"{w3*100:.0f}%" if w3 is not None else "—"
                w5s = f"{w5*100:.0f}%" if w5 is not None else "—"
                L.append(f"  {r['ticker']:5s} [{uni_tag}] {price_str}  cal_p={cal*100:.0f}%  "
                         f"3d ~{m3s} (wr {w3s}) / 5d ~{m5s} (wr {w5s})  "
                         f"news {r.get('news_score', 0):+d}{vzr_str}")
            else:
                p1 = r.get("pred_1d", 0.0); p3 = r.get("pred_3d", 0.0); p5 = r.get("pred_5d", 0.0)
                if cfg_forward and latest and abs(1 + gap) > 1e-9:
                    p1 = (1 + p1) / (1 + gap) - 1
                    p3 = (1 + p3) / (1 + gap) - 1
                    p5 = (1 + p5) / (1 + gap) - 1
                L.append(f"  {r['ticker']:5s} [{uni_tag}] {price_str}  cal_p={cal*100:.0f}%  "
                         f"1d {p1*100:+.1f}% / 3d {p3*100:+.1f}% / "
                         f"5d {p5*100:+.1f}%  news {r.get('news_score', 0):+d}{vzr_str}")

    if catalyst:
        L.append("")
        L.append("Top catalyst surprises (news-driven):")
        for r in catalyst:
            price_str = f"${r.get('close_prev', 0):.2f}"
            L.append(f"  {r['ticker']:5s} {price_str}  p={r['p_catalyst']*100:.1f}%  "
                     f"news n={r['news_n']}/strong={r['news_strong']}/"
                     f"pubs={r['pub_div']} score={r['news_score']:+d}  "
                     f"prior5d {r['prior_5d_ret']*100:+.1f}%")
    if catalyst_short:
        L.append("")
        L.append("Catalyst downside (short candidates):")
        for r in catalyst_short:
            price_str = f"${r.get('close_prev', 0):.2f}"
            L.append(f"  {r['ticker']:5s} {price_str}  p_down={r['p_down']*100:.1f}%  "
                     f"news n={r['news_n']}/strong={r['news_strong']}  "
                     f"prior5d {r['prior_5d_ret']*100:+.1f}%")
    return "\n".join(L)


# ---------- main ----------

def is_weekend_now() -> bool:
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-4)))  # ET approx
    return now.weekday() >= 5


def refresh_picks_only_main(cfg: dict, dry_run: bool = False) -> int:
    """10am refresh job. Reads today's daily log, refreshes pm_last on the
    candidate pool via Finnhub /quote, re-applies pick_current_list and
    pick_high_prec_list with the refreshed prices and a (configurably
    looser) min_prob, formats a tagged "10am refresh" message, sends it,
    and writes a sidecar log to <date>_1000.json.

    Catches gap-and-fade reversals because the regression-from-entry gate
    in pick_high_prec_list re-evaluates against the now-current pm_last.
    """
    today_iso = datetime.now().strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"{today_iso}.json"
    if not log_path.exists():
        print(f"[refresh] no log at {log_path} — morning run hasn't happened, "
              f"nothing to refresh")
        return 1
    log = json.loads(log_path.read_text())

    pool: dict[str, list[dict]] = log.get("candidate_pool") or {}
    if not pool:
        # Backward compat: older logs only have `predictions`.
        pool = {k: list(v) for k, v in (log.get("predictions") or {}).items()}
        print(f"[refresh] candidate_pool missing, falling back to predictions "
              f"({sum(len(v) for v in pool.values())} rows)")

    # Total cap on /quote calls. 60 = comfortable inside 60/min limit;
    # 120 = ~2 min runtime with the rate limiter spreading calls.
    max_total = int(cfg.get("refresh_max_picks_total", 120))
    rate = int(cfg.get("refresh_rate_limit_per_min", 58))

    # Cross-universe pool of unique tickers, ranked by best prob_final
    # across universes (so a name picked in v4+v5+v7 dedupes to one).
    by_tkr: dict[str, dict] = {}
    for uni, rows in pool.items():
        for r in rows:
            t = r.get("ticker")
            if not t: continue
            if t not in by_tkr or (r.get("prob_final") or 0) > (by_tkr[t].get("prob_final") or 0):
                by_tkr[t] = {**r, "_universe": uni}
    # Rank by prob_final desc, take top max_total
    ranked = sorted(by_tkr.values(), key=lambda r: -(r.get("prob_final") or 0))
    target = ranked[:max_total]
    print(f"[refresh] candidate pool size: {len(by_tkr)} unique tickers; "
          f"refreshing top {len(target)} by prob_final (cap={max_total})")

    # Refresh /quote for the target set. The same quote applies to all
    # universe rows for that ticker, so build a map and replay.
    import time as _t
    t0 = _t.time()
    target_tickers = {r["ticker"] for r in target}
    # Refresh on the deduped target list so we get the cap right
    n_ref = refresh_pm_last_finnhub(target, max_unique_tickers=max_total,
                                     rate_limit_per_min=rate)
    elapsed = _t.time() - t0
    print(f"[refresh] /quote refresh: {n_ref}/{len(target)} rows updated "
          f"in {elapsed:.1f}s")

    # Build a price/gap lookup from the refreshed targets, then propagate
    # back to the full candidate pool (so pick_current_list per universe
    # sees current prices on every ticker that got refreshed).
    refreshed = {r["ticker"]: r for r in target if r.get("pm_last_source") == "finnhub"}
    for uni, rows in pool.items():
        for r in rows:
            t = r.get("ticker")
            if t in refreshed:
                rr = refreshed[t]
                r["pm_last"] = rr["pm_last"]
                r["pm_last_source"] = "finnhub"
                r["overnight_gap_live"] = rr["overnight_gap_live"]

    # Re-apply selection gates with refreshed prices. Looser min_prob lets
    # candidates that were just below the morning cut surface here if they
    # still pass the regressor-from-entry gate at 10am pricing.
    refresh_min_prob = float(cfg.get("refresh_min_prob",
                                       cfg.get("burst_alert_min_prob", 0.85) - 0.05))
    refresh_top_n = int(cfg.get("refresh_top_n_per_universe",
                                  cfg.get("top_n_per_universe", 50)))
    refresh_min_n = int(cfg.get("refresh_min_n",
                                  cfg.get("burst_alert_min_n", 0)))
    agreement_min_prob = cfg.get("require_agreement_min_prob_v10")
    if agreement_min_prob is not None:
        agreement_min_prob = float(agreement_min_prob)
    max_per_sector = cfg.get("max_per_sector", 3)
    if max_per_sector is not None: max_per_sector = int(max_per_sector)
    high_prec_max_per_sector = cfg.get("high_prec_max_per_sector", 1)
    if high_prec_max_per_sector is not None:
        high_prec_max_per_sector = int(high_prec_max_per_sector)

    by_uni_refreshed: dict[str, list[dict]] = {}
    full_uni_refreshed: dict[str, list[dict]] = {}
    for uni, rows in pool.items():
        # Re-apply the regressor-from-entry gate: drop rows where any of the
        # 1d/3d/5d forward-from-entry returns went non-positive after the
        # gap update. This is the key gap-and-fade catcher.
        kept = []
        for r in rows:
            pm = r.get("pm_last"); cp = r.get("close_prev")
            if pm and cp and abs(1 + (pm/cp - 1)) > 1e-9:
                gap = pm/cp - 1
                fwd1 = (1 + r.get("pred_1d", 0)) / (1 + gap) - 1
                fwd3 = (1 + r.get("pred_3d", 0)) / (1 + gap) - 1
                fwd5 = (1 + r.get("pred_5d", 0)) / (1 + gap) - 1
                if not (fwd1 > 0 and fwd3 > 0 and fwd5 > 0):
                    continue
            kept.append(r)
        full_uni_refreshed[uni] = sorted(kept, key=lambda r: -(r.get("prob_final") or 0))
        by_uni_refreshed[uni] = pick_current_list(
            full_uni_refreshed[uni],
            top_n=refresh_top_n, min_prob=refresh_min_prob,
            min_n=refresh_min_n,
            agreement_min_prob=agreement_min_prob,
            max_per_sector=max_per_sector)
        print(f"[refresh {uni}] {len(rows)} → {len(full_uni_refreshed[uni])} after gap-gate, "
              f"{len(by_uni_refreshed[uni])} make the list "
              f"(min_prob>={refresh_min_prob})")

    # Cross-universe high-precision shortlist
    all_rows = []
    for uni, rows in full_uni_refreshed.items():
        for r in rows:
            rr = dict(r); rr["_universe"] = uni
            all_rows.append(rr)
    high_prec_top_n = int(cfg.get("high_prec_top_n", 5))
    high_prec_min_cal_prob = float(cfg.get("high_prec_min_cal_prob", 0.40))
    high_prec_rows = pick_high_prec_list(
        all_rows, top_n=high_prec_top_n,
        min_cal_prob=high_prec_min_cal_prob,
        max_per_sector=high_prec_max_per_sector)
    print(f"[refresh high-prec] {len(high_prec_rows)} cross-universe picks")

    # Write refresh log
    refresh_log_path = LOG_DIR / f"{today_iso}_1000.json"
    refresh_log = {
        "date": today_iso, "tag": "10am_refresh",
        "predictions": by_uni_refreshed,
        "high_prec": high_prec_rows,
        "n_quote_refreshed": n_ref,
        "refresh_elapsed_sec": round(elapsed, 2),
        "refresh_min_prob": refresh_min_prob,
        "max_total_quote_calls": max_total,
    }
    refresh_log_path.write_text(json.dumps(refresh_log, indent=2, default=str))
    print(f"[refresh] wrote {refresh_log_path}")

    # Format + send. Reuse format_message but prepend a tag header.
    msg_body = format_message(today_iso, by_uni_refreshed, log.get("eval"),
                                catalyst=None, catalyst_eval=None,
                                catalyst_short=None,
                                high_prec=high_prec_rows,
                                high_prec_eval=log.get("high_prec_eval"))
    msg = (f"🔄 Confirmed entry pricing — 10:00 ET\n"
           f"({n_ref}/{len(target)} prices refreshed via Finnhub /quote, "
           f"{len(high_prec_rows)} high-prec picks survived re-gate)\n\n"
           + msg_body)
    print("\n=== REFRESH MESSAGE ===\n" + msg + "\n=======================\n")

    burst_gate = (cfg.get("send_notifications", False)
                   and cfg.get("send_refresh_notifications", True))
    if burst_gate and not dry_run:
        ok = _notify_send(msg)
        print(f"[notify refresh] sent: {ok}")
    else:
        print(f"[notify refresh] skipped (dry_run={dry_run}, "
              f"send_notifications={cfg.get('send_notifications')})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of-close", action="store_true",
                    help="skip pre-market + news, score with overnight_gap=0")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the message; do not send iMessage even if configured")
    ap.add_argument("--asof", default=None,
                    help="Replay as-of 'YYYY-MM-DD HH:MM' (ET). Filters the "
                         "panel, intraday, and news to before this timestamp; "
                         "skips eval-and-retrain subprocesses.")
    ap.add_argument("--snapshot", default=None,
                    help="Use the named/located snapshot dir instead of "
                         "yfinance live calls (intraday + news). When --asof "
                         "is given, a matching snapshot is auto-resolved if "
                         "this flag is not set.")
    ap.add_argument("--no-notify", action="store_true",
                    help="Run the full pipeline but suppress the morning "
                         "Telegram message. Use for the 8:45 background "
                         "scoring run when the actual buy signal is the "
                         "10am --refresh-picks-only message.")
    ap.add_argument("--refresh-picks-only", action="store_true",
                    help="Skip panel refresh + scoring; load today's daily "
                         "log, refresh pm_last on the broader candidate pool "
                         "via Finnhub /quote (real-time), re-apply the "
                         "selection gates with current prices, and send the "
                         "resulting message tagged as 10am refresh. Catches "
                         "gap-and-fade reversals before you buy.")
    args = ap.parse_args()
    cfg = load_config()

    # Refresh-picks-only mode: short-circuits the heavy pipeline, just
    # re-prices today's candidate pool via Finnhub and re-applies gates.
    if args.refresh_picks_only:
        return refresh_picks_only_main(cfg, dry_run=args.dry_run)

    global _ASOF, _SNAP, _LOADER
    _ASOF = _parse_asof(args.asof)
    if _ASOF is not None:
        print(f"[daily] REPLAY MODE asof={_ASOF}")
        _LOADER = _open_loader(args.snapshot, _ASOF, runner_tag="burst")
        if _LOADER is None:
            print("[daily] no snapshot resolved — replay falls back to live "
                  "yfinance (news/intraday will reflect *current* yf cache)")

    is_weekend_local = (_ASOF.weekday() >= 5) if _ASOF is not None else is_weekend_now()
    live = not (args.as_of_close or is_weekend_local)
    today_iso = _asof_today_iso(_ASOF)

    # In live mode, capture every yfinance round-trip into a snapshot dir so
    # the message can be reproduced later even after yf's news/intraday caches
    # rotate. Asof replays don't snapshot — the data isn't "the data of that
    # asof", it's just current yf trimmed to a past timestamp.
    if _ASOF is None:
        _SNAP = _open_snapshot("burst")
        print(f"[daily] snapshot dir: {_SNAP.dir}")

    # Bind the chosen news scorer into this module's namespace, replacing
    # the None sentinel above. Lazy-loads FinBERT only if requested.
    scorer_name = cfg.get("news_scorer", "lexicon_expanded")
    blend_weight = float(cfg.get("news_blend_weight", 0.15))
    print(f"[daily] news scorer={scorer_name!r}  blend_weight={blend_weight}")
    import sys as _sys
    _sys.modules[__name__].score_headline = get_scorer(scorer_name)

    # refresh panels — v4 (broad >$40), v5 (upside-asym), v7 (full S&P 500).
    # yfinance downloads are network-bound, so run the 3 refreshes concurrently.
    uni_v4_path = DATA / "burst_universe_v4.csv"
    uni_v5_path = DATA / "burst_universe_v5.csv"
    uni_v7_path = DATA / "burst_universe_v7.csv"

    def _try_refresh(uni_path: Path, panel_path: Path, builder, label: str):
        """Refresh via yfinance, falling back to the cached panel on failure.
        yfinance has a known race condition on very large universes; we don't
        want one bad download to kill the morning run."""
        try:
            return refresh_panel(uni_path, panel_path, builder)
        except Exception as e:
            print(f"[daily] {label} refresh failed ({type(e).__name__}: {e}); "
                  f"falling back to cached {panel_path.name}")
            if panel_path.exists():
                return pd.read_csv(panel_path, parse_dates=["date"])
            raise

    def _from_loader(tag: str):
        if _LOADER is None: return None
        p = _LOADER.panel(tag)
        if p is not None:
            print(f"[daily] loaded {tag} panel from snapshot ({len(p)} rows)")
        return p

    def _refresh_v4():
        snap_p = _from_loader("v4")
        if snap_p is not None: return snap_p
        if uni_v4_path.exists():
            print("[daily] refreshing v4 (>$40) panel ...")
            return _try_refresh(uni_v4_path, DATA / "burst_panel_v6b.csv", feat_v4, "v4")
        print("[daily] burst_universe_v4.csv missing — using cached panel")
        return pd.read_csv(DATA / "burst_panel_v6b.csv", parse_dates=["date"])

    def _refresh_v5():
        snap_p = _from_loader("v5")
        if snap_p is not None: return snap_p
        print("[daily] refreshing v5 (upside-asymmetric) panel ...")
        return _try_refresh(uni_v5_path, DATA / "burst_panel_v6.csv", feat_v5, "v5")

    def _refresh_v7():
        snap_p = _from_loader("v7")
        if snap_p is not None: return snap_p
        if uni_v7_path.exists() and "v7" in cfg.get("universes", []):
            print("[daily] refreshing v7 (full S&P 500) panel ...")
            return _try_refresh(uni_v7_path, DATA / "burst_panel_v7.csv", feat_v4, "v7")
        return None

    # Refresh the three universes SEQUENTIALLY (not in parallel). Running them
    # concurrently issued ~2,700 yfinance requests at once and reliably
    # tripped the rate limiter on the 1,877-ticker v7 universe. Sequential
    # refresh is a little slower wall-clock but finishes with full coverage.
    panel_v6b = _refresh_v4()
    panel_v6 = _refresh_v5()
    panel_v7 = _refresh_v7()

    if _SNAP is not None:
        if panel_v6b is not None: _SNAP.save_panel("v4", panel_v6b)
        if panel_v6 is not None: _SNAP.save_panel("v5", panel_v6)
        if panel_v7 is not None: _SNAP.save_panel("v7", panel_v7)

    if _ASOF is not None:
        # Trim each in-memory panel to bars closed before the asof timestamp.
        # The on-disk cache CSV is left as-is; it gets refreshed on the next
        # live run anyway and keeping it untouched avoids cross-replay leakage.
        if panel_v6b is not None:
            panel_v6b = _asof_filter_panel(panel_v6b, _ASOF)
            print(f"[daily] asof: v4 panel filtered to "
                  f"{panel_v6b['date'].max()} ({len(panel_v6b)} rows)")
        if panel_v6 is not None:
            panel_v6 = _asof_filter_panel(panel_v6, _ASOF)
            print(f"[daily] asof: v5 panel filtered to "
                  f"{panel_v6['date'].max()} ({len(panel_v6)} rows)")
        if panel_v7 is not None:
            panel_v7 = _asof_filter_panel(panel_v7, _ASOF)
            print(f"[daily] asof: v7 panel filtered to "
                  f"{panel_v7['date'].max()} ({len(panel_v7)} rows)")

    # Build drop-risk lookup once from the v7 panel (widest — covers v4/v5).
    # v7 panel has V7_FEATS + close; _add_trend_feats computes the extra 6.
    # Used inside score_universe to suppress bursts on names the drop model
    # flags as at-risk of a pullback.
    drop_lookup: dict[str, dict] = {}
    drop_gate_version = cfg.get("drop_gate_version", "v2")
    print(f"[daily] drop_gate_version = {drop_gate_version}")
    try:
        if panel_v7 is not None:
            drop_lookup = build_drop_lookup(panel_v7, version=drop_gate_version)
        elif panel_v6b is not None:
            drop_lookup = build_drop_lookup(panel_v6b, version=drop_gate_version)
        n_gated = sum(1 for v in drop_lookup.values() if v["p_drop"] >= 0.5)
        print(f"[daily] drop lookup: {len(drop_lookup)} tickers, "
              f"{n_gated} with p_drop >= 0.5")
    except Exception as e:
        print(f"[daily] drop lookup failed: {type(e).__name__}: {e}; "
              f"proceeding without drop gate")

    by_uni = {}
    full_uni: dict[str, list[dict]] = {}   # all gate-passing rows per universe
    version = _resolve_model_version(cfg)
    print(f"[daily] burst_model_version = {version}")
    # Selection knobs (config-driven). The current list now uses a tighter
    # top_n + min_prob, plus an optional v10 agreement filter. The high-prec
    # list is built across universes from calibrated probabilities.
    cur_top_n = int(cfg.get("top_n_per_universe", 3))
    cur_min_prob = float(cfg.get("burst_alert_min_prob", 0.85))
    cur_min_n = int(cfg.get("burst_alert_min_n", 3))
    agreement_min_prob = cfg.get("require_agreement_min_prob_v10")  # None disables
    if agreement_min_prob is not None:
        agreement_min_prob = float(agreement_min_prob)
    high_prec_enabled = bool(cfg.get("high_prec_enabled", True))
    high_prec_min_cal_prob = float(cfg.get("high_prec_min_cal_prob", 0.40))
    high_prec_top_n = int(cfg.get("high_prec_top_n", 5))
    # Sector caps: per-universe lists serve reflection/training (gentle cap),
    # cross-universe high_prec serves the actual buy decision (aggressive cap).
    # 0/None disables either. See cap_per_sector docstring.
    max_per_sector = cfg.get("max_per_sector", 3)
    if max_per_sector is not None:
        max_per_sector = int(max_per_sector)
    high_prec_max_per_sector = cfg.get("high_prec_max_per_sector", 1)
    if high_prec_max_per_sector is not None:
        high_prec_max_per_sector = int(high_prec_max_per_sector)
    if (max_per_sector and max_per_sector > 0) or \
       (high_prec_max_per_sector and high_prec_max_per_sector > 0):
        # Pre-warm the lookup so scoring threads don't race to populate it.
        load_sector_lookup()
        print(f"[daily] sector caps: per-universe={max_per_sector or 'off'}, "
              f"high_prec={high_prec_max_per_sector or 'off'}")

    # Attach regime features once per panel if v10/v11/v12/v13_ens need them
    if version in ("v10", "v11", "v12", "v13_ens"):
        regime = None  # load lazily inside attach_regime
        if panel_v6b is not None:
            panel_v6b = attach_regime(panel_v6b, regime)
        if panel_v6 is not None:
            panel_v6 = attach_regime(panel_v6, regime)
        if panel_v7 is not None:
            panel_v7 = attach_regime(panel_v7, regime)
    # Attach market-relative vol_z (logged on each pick; available for
    # future model retrains and surfaced in the morning message). Cheap
    # group-by-date median, no external dependencies.
    if panel_v6b is not None:
        panel_v6b = attach_vol_z_rel(panel_v6b)
    if panel_v6 is not None:
        panel_v6 = attach_vol_z_rel(panel_v6)
    if panel_v7 is not None:
        panel_v7 = attach_vol_z_rel(panel_v7)
    # Attach intraday minute-bar features for v11/v12/v13_ens (per-(ticker,date)
    # merge from data/intraday_daily_features.csv; coverage = whatever the
    # local intraday store has — non-covered rows pass NaN to HGB models.)
    if version in ("v11", "v12", "v13_ens"):
        if panel_v6b is not None:
            panel_v6b = attach_intraday(panel_v6b)
        if panel_v6 is not None:
            panel_v6 = attach_intraday(panel_v6)
        if panel_v7 is not None:
            panel_v7 = attach_intraday(panel_v7)
    # Attach yest-intraday features for v13_ens. Built from data/ohlc_prices.csv
    # (full historical OHLC). Each row's features are anchored on its own date,
    # so they reflect the closing-day's intraday character (no look-ahead).
    if version == "v13_ens":
        if panel_v6b is not None:
            panel_v6b = attach_yest_intraday(panel_v6b)
        if panel_v6 is not None:
            panel_v6 = attach_yest_intraday(panel_v6)
        if panel_v7 is not None:
            panel_v7 = attach_yest_intraday(panel_v7)

    _uni_panel = {"v4": panel_v6b, "v5": panel_v6, "v7": panel_v7}
    _uni_label = {"v4": "v4 (>$40)", "v5": "v5 (upside-asymmetric)",
                  "v7": "v7 (full S&P 500)"}
    for uni in ("v4", "v5", "v7"):
        if uni not in cfg.get("universes", []):
            continue
        pnl = _uni_panel.get(uni)
        if pnl is None:
            continue
        spec = _model_spec(uni, version)
        # If the chosen version's artifacts are missing, fall back per-universe
        # along the preference chain v11 -> v10 -> v7.
        used_version = version
        for fallback in ("v11", "v10", "v7"):
            if not spec["clf"].exists() or not all(p.exists() for p in spec["regs"].values()):
                if fallback == used_version:
                    continue
                print(f"[daily] {uni}: {used_version} artifacts missing, trying {fallback}")
                spec = _model_spec(uni, fallback)
                used_version = fallback
            else:
                break
        print(f"[daily] scoring {_uni_label[uni]} with {used_version} ...")
        # Optional v10 agreement spec — only when running v11 and the v10
        # artifacts exist for this universe (used for the current-list gate).
        agree = None
        agree_feats = None
        if used_version == "v11" and agreement_min_prob is not None:
            v10_spec = _model_spec(uni, "v10")
            if v10_spec["clf"].exists():
                agree = v10_spec["clf"]
                agree_feats = v10_spec["feats"]
        rows = score_universe(
            uni, pnl, spec["clf"], spec["regs"],
            feats=spec["feats"],
            top_n=cur_top_n,
            live=live, blend_weight=blend_weight,
            drop_lookup=drop_lookup,
            agreement_clf_path=agree,
            agreement_feats=agree_feats)
        full_uni[uni] = rows
        by_uni[uni] = pick_current_list(
            rows, top_n=cur_top_n, min_prob=cur_min_prob,
            min_n=cur_min_n,
            agreement_min_prob=agreement_min_prob if agree else None,
            max_per_sector=max_per_sector)
        print(f"[score {uni}] current-list size: {len(by_uni[uni])} "
              f"(threshold p>={cur_min_prob}"
              + (f", v10>={agreement_min_prob}" if agree else "")
              + (f", sector_cap={max_per_sector}" if max_per_sector else "")
              + ")")

    # Cross-universe high-precision list (calibrated v11 prob + multi-horizon
    # regressor agreement). Sourced from full_uni — uses all gate-passing
    # rows, not just the current-list survivors, so a candidate that's just
    # below the current-list threshold can still surface here if it's well
    # calibrated and unanimously positive across 1d/3d/5d.
    high_prec_rows: list[dict] = []
    if high_prec_enabled and full_uni:
        all_rows = []
        for uni, rows in full_uni.items():
            for r in rows:
                rr = dict(r); rr["_universe"] = uni
                all_rows.append(rr)
        high_prec_rows = pick_high_prec_list(
            all_rows, top_n=high_prec_top_n,
            min_cal_prob=high_prec_min_cal_prob,
            max_per_sector=high_prec_max_per_sector)
        print(f"[high-prec] cross-universe size: {len(high_prec_rows)} "
              f"(cal_p>={high_prec_min_cal_prob}, all 3 regressors >0)")

    # Refresh pm_last on the actual pick rows via Finnhub /quote (real-time,
    # 60 req/min free tier). The wide score pass uses yfinance 1m bars which
    # carry ~15-min lag — fine for screening, but the morning text shows
    # pm_last as the *entry price* and that needs to be current. We touch
    # only the surviving picks (per-universe + high_prec), so total /quote
    # cost is ~5-30 calls and well under the rate limit. Disable with
    # `realtime_quote_refresh: false` in burst_daily.json.
    if cfg.get("realtime_quote_refresh", True):
        pick_rows: list[dict] = []
        for uni_rows in by_uni.values():
            pick_rows.extend(uni_rows)
        pick_rows.extend(high_prec_rows)
        n_ref = refresh_pm_last_finnhub(pick_rows)
        if n_ref > 0:
            print(f"[realtime] refreshed pm_last on {n_ref} pick rows via Finnhub /quote")

    # Catalyst-surprise model — runs on the same v7 panel with Finnhub news.
    # Surfaced in the morning text as a separate "Top catalyst surprises" bucket.
    catalyst_rows: list[dict] = []
    catalyst_short_rows: list[dict] = []
    if cfg.get("catalyst_enabled", True) and panel_v7 is not None:
        try:
            print("[daily] scoring catalyst-surprise model (upside) ...")
            catalyst_rows = score_catalyst(
                panel_v7, top_n=int(cfg.get("catalyst_top_n", 5)),
                drop_lookup=drop_lookup,
                apply_gates=cfg.get("catalyst_gates_enabled", True))
            print(f"[catalyst] top {len(catalyst_rows)}: "
                  f"{[r['ticker'] for r in catalyst_rows]}")
        except Exception as e:
            print(f"[catalyst] upside scoring failed: {type(e).__name__}: {e}")
        if cfg.get("catalyst_short_enabled", True):
            try:
                print("[daily] scoring catalyst-downside model (shorts) ...")
                catalyst_short_rows = score_catalyst_short(
                    panel_v7,
                    top_n=int(cfg.get("catalyst_short_top_n",
                                       cfg.get("catalyst_top_n", 5))))
                print(f"[catalyst-short] top {len(catalyst_short_rows)}: "
                      f"{[r['ticker'] for r in catalyst_short_rows]}")
            except Exception as e:
                print(f"[catalyst] downside scoring failed: {type(e).__name__}: {e}")

    # evaluate yesterday
    yest_eval = None
    yest_high_prec_eval = None
    yesterday = None
    yest_log = None
    _eval_anchor = _asof_now_et(_ASOF) if _ASOF is not None else datetime.now()
    # Find the most recent 6 prior log files (~6 trading days, allowing for
    # weekends and one holiday). Preserved order = most-recent first.
    prior_paths: list[tuple[str, Path]] = []
    for delta in range(1, 11):
        y = (_eval_anchor - timedelta(days=delta)).strftime("%Y-%m-%d")
        ypath = LOG_DIR / f"{y}.json"
        if ypath.exists():
            prior_paths.append((y, ypath))
            if len(prior_paths) >= 6: break
    if prior_paths:
        # 1d eval: yesterday's predictions evaluated against today's close.
        yesterday, ypath = prior_paths[0]
        yest_log = json.loads(ypath.read_text())
        yest_eval = evaluate_yesterday(yest_log, panel_v6b, panel_v6, panel_v7)
        yest_high_prec_eval = evaluate_high_prec(
            yest_log,
            {"v4": panel_v6b, "v5": panel_v6, "v7": panel_v7})
        # 3d / 5d eval: predictions made N trading days ago will have N-day
        # forward closes available now. Splice their summary stats into
        # yest_eval so the morning message shows all three horizons.
        eval_sources: dict[str, str] = {"1d": yesterday}
        eval_rows_extra: dict[str, dict] = {}
        for hidx, hkey in ((2, "3d"), (4, "5d")):
            if len(prior_paths) <= hidx: continue
            src_date, src_path = prior_paths[hidx]
            try:
                src_log = json.loads(src_path.read_text())
                src_eval = evaluate_yesterday(src_log, panel_v6b, panel_v6, panel_v7)
            except Exception as e:
                print(f"[eval] {hkey} eval of {src_date} failed: {type(e).__name__}: {e}")
                continue
            eval_sources[hkey] = src_date
            for uni, agg in (src_eval.get("summary") or {}).items():
                if uni not in yest_eval["summary"]:
                    yest_eval["summary"][uni] = {}
                for k, v in agg.items():
                    if k == hkey or k == f"{hkey}_entry":
                        yest_eval["summary"][uni][k] = v
            eval_rows_extra[f"rows_{hkey}"] = src_eval.get("rows") or {}
        yest_eval["sources"] = eval_sources
        yest_eval.update(eval_rows_extra)

    # Catalyst-model eval: for yesterday's catalyst picks, check the realized
    # 1d return in today's panel. Report hit-rate at +5% threshold + up-rate.
    catalyst_eval = None
    if yest_log and yest_log.get("catalyst") and panel_v7 is not None:
        p7 = panel_v7.sort_values(["ticker", "date"])
        hits = []; ups = []; rets = []
        for row in yest_log["catalyst"]:
            t = row["ticker"]; d = pd.Timestamp(row.get("close_prev_date") or row.get("date"))
            g = p7[p7["ticker"] == t]
            base = g[g["date"] == d]
            after = g[g["date"] > d]
            if base.empty or after.empty: continue
            ret1 = float(after["close"].iloc[0]) / float(base["close"].iloc[0]) - 1
            hits.append(ret1 >= 0.05); ups.append(ret1 > 0); rets.append(ret1)
        if rets:
            catalyst_eval = {
                "n": len(rets),
                "hit_5pct": float(np.mean(hits)),
                "up_rate": float(np.mean(ups)),
                "mean_ret": float(np.mean(rets)),
            }

    # Build a broader candidate pool per universe for the 10am refresh job.
    # Keeps top N by prob_final from each universe's full passing-rows set
    # (`full_uni`), so the refresh can re-evaluate names that were just below
    # the 8:45 morning cut. Capped per-universe to keep the JSON reasonable;
    # the refresh-only job applies its own cross-universe cap before /quote.
    candidate_pool: dict[str, list[dict]] = {}
    pool_per_uni = int(cfg.get("candidate_pool_per_universe", 60))
    for uni, rows in full_uni.items():
        candidate_pool[uni] = rows[:pool_per_uni]
    n_pool = sum(len(v) for v in candidate_pool.values())
    print(f"[daily] candidate_pool: {n_pool} rows across {len(candidate_pool)} universes "
          f"(top {pool_per_uni}/universe by prob_final)")

    # log today
    today_log = {"date": today_iso, "live": live, "predictions": by_uni,
                 "high_prec": high_prec_rows,
                 "candidate_pool": candidate_pool,
                 "catalyst": catalyst_rows,
                 "catalyst_short": catalyst_short_rows,
                 "evaluation_for": yesterday, "eval": yest_eval,
                 "high_prec_eval": yest_high_prec_eval,
                 "catalyst_eval": catalyst_eval,
                 "replay": _ASOF is not None,
                 "asof": str(_ASOF) if _ASOF is not None else None}
    (LOG_DIR / f"{today_iso}.json").write_text(json.dumps(today_log, indent=2, default=str))
    print(f"[daily] logged {LOG_DIR/f'{today_iso}.json'}")

    msg = format_message(today_iso, by_uni, yest_eval,
                          catalyst=catalyst_rows, catalyst_eval=catalyst_eval,
                          catalyst_short=catalyst_short_rows,
                          high_prec=high_prec_rows,
                          high_prec_eval=yest_high_prec_eval)
    print("\n=== MESSAGE ===\n" + msg + "\n===============\n")

    # send burst message via configured backend (telegram or iMessage).
    # Two config gates: `send_notifications` is the global master switch
    # (shared with the afternoon AH runner), `send_morning_burst_notifications`
    # is a morning-burst-only override. Both must be truthy to send.
    burst_gate = (cfg.get("send_notifications", False)
                   and cfg.get("send_morning_burst_notifications", True))
    if burst_gate and not args.dry_run and not args.no_notify:
        ok = _notify_send(msg)
        print(f"[notify burst] sent: {ok}")
    else:
        why = []
        if args.no_notify: why.append("--no-notify")
        if args.dry_run: why.append("--dry-run")
        if not cfg.get("send_notifications", False): why.append("send_notifications=false")
        if not cfg.get("send_morning_burst_notifications", True):
            why.append("send_morning_burst_notifications=false")
        print(f"[notify burst] skipped ({', '.join(why) or 'gates failed'})")

    # Persist all yfinance data captured this run (intraday 1m for top picks,
    # news payloads, panels). Lets us replay this exact message later even
    # after yf rotates the news cache.
    if _SNAP is not None:
        _SNAP.update_meta(today_iso=today_iso, live=live,
                          dry_run=bool(args.dry_run),
                          message_text=msg,
                          tickers_in_message=sorted({
                              r["ticker"] for rows in by_uni.values() for r in rows
                          } | {r["ticker"] for r in catalyst_rows}
                            | {r["ticker"] for r in catalyst_short_rows}))
        out = _SNAP.flush()
        print(f"[daily] snapshot flushed -> {out}")

    # Eval only — drop alert was decoupled to its own plist
    # (com.user.dropalert.plist @ 09:20 ET). The drop text needs fresher AH /
    # pre-market data to avoid alerting on drops that already happened, and
    # running it 35 min after the 8:45 morning kickoff gives the extra ticks
    # time to settle. Eval still runs here since it reads old logs only.
    if _ASOF is not None:
        # Replay mode: eval-and-retrain + recap touch live state files
        # (drift_summary, prediction_outcomes) — skip them in replay so we
        # don't pollute live history with backfilled-day artifacts.
        print("[daily] asof replay: skipping eval_and_retrain + recap")
        return
    if not args.dry_run:
        print("\n[daily] running eval_and_retrain ...")
        try:
            subprocess.run(
                [_sys_notify.executable, "-u", str(ROOT / "code" / "43_eval_and_retrain.py")],
                cwd=str(ROOT), check=False, timeout=900)
        except Exception as e:
            print(f"[daily] eval_and_retrain failed: {e}")

        # Recap reads output/drift_summary.txt + prediction_outcomes.csv from
        # the eval step, so it must run after both parallel jobs complete.
        print("\n[daily] sending performance recap ...")
        try:
            subprocess.run(
                [_sys_notify.executable, "-u", str(ROOT / "code" / "45_performance_recap.py"),
                 "--source", "morning"],
                cwd=str(ROOT), check=False, timeout=60)
        except Exception as e:
            print(f"[daily] recap failed: {e}")


if __name__ == "__main__":
    main()
