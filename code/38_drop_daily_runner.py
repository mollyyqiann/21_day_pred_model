"""Drop-alert daily runner — runs AFTER the burst runner.

Reads the day's burst predictions log (output/daily_log/YYYY-MM-DD.json) and:
  1. Scans the *last 7 days* of logs for burst picks whose drop probability is
     now elevated. Sends as "watch list from prior recommendations."
  2. Ranks today's universe by p_drop (v8 features), surfaces top-5 as fresh
     drop alerts.

Sends as a SEPARATE message so the burst and drop signals don't entangle.

Config keys (reuse burst_daily.json):
  "send_drop_notifications": bool (default same as send_notifications)
  "drop_prob_warn_threshold": float (default 0.30) for prior-rec watch list

Outputs:
  output/drop_live_today.csv
  output/daily_log/<date>_drop.json
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import subprocess
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
LOG_DIR = OUT / "daily_log"
CONFIG_PATH = ROOT / "config" / "burst_daily.json"

# --asof replay support (see code/_asof.py).
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

V7_FEATS = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
            "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap"]
TREND = ["ma_stack", "up_streak", "up_bigdays_20d",
         "dist_ma60_atr", "ma60_slope_60d", "run_length"]
FEATS = V7_FEATS + TREND

# Guardrails for stale-panel detection. PRU (2026-04-22) fired a false alarm
# because the panel was 5 business days stale and `overnight_gap` got computed
# against an ancient close. We now fail loud (and attempt self-rebuild) if the
# panel is more than MAX_PANEL_LAG_BD business days behind today.
MAX_PANEL_LAG_BD = 3
# Drift between panel close and yfinance live close that we treat as a warning
# signal (usually means a missed intraday refresh).
PANEL_DRIFT_WARN = 0.03


def is_weekend_now():
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-4)))
    return now.weekday() >= 5


def _business_day_lag(panel_max: pd.Timestamp, today: "datetime.date") -> int:
    """Business days between panel_max and today (exclusive of both ends).
    Zero when panel_max is today or in the future. Weekends/holidays via
    pandas bdate_range (pure weekday — doesn't know NYSE holidays but that's
    fine for a 3-bd threshold)."""
    if pd.Timestamp(panel_max).date() >= today:
        return 0
    rng = pd.bdate_range(pd.Timestamp(panel_max).date() + pd.Timedelta(days=1),
                          today - pd.Timedelta(days=1))
    return len(rng)


def _rebuild_v8_panel_from_yf(quiet: bool = False):
    """Refresh burst_panel_v8.csv via yfinance using burst_universe_v8.csv
    (or a fallback S&P 500 universe). Reuses refresh_panel + feat_v4 +
    _add_trend_feats from 29_burst_daily_runner so behavior stays in sync.

    Returns the rebuilt DataFrame, or None if yfinance coverage fell below
    the refresh_panel coverage floor (refresh_panel raises in that case —
    we don't want to proceed on a truncated panel).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_burst_runner_for_drop", ROOT / "code" / "29_burst_daily_runner.py")
    br = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(br)

    uni_path = DATA / "burst_universe_v8.csv"
    if not uni_path.exists():
        uni_path = DATA / "burst_universe_v7_sp500_backup.csv"
    if not uni_path.exists():
        uni_path = DATA / "burst_universe_v7.csv"
    panel_path = DATA / "burst_panel_v8.csv"
    if not quiet:
        print(f"[drop] rebuilding panel from {uni_path.name} via yfinance ...")
    try:
        panel = br.refresh_panel(uni_path, panel_path, br.feat_v4)
    except Exception as e:
        print(f"[drop] panel rebuild failed: {type(e).__name__}: {e}")
        return None
    # refresh_panel writes V7 features only; v8 schema also has 6 trend feats.
    out = []
    for _, g in panel.groupby("ticker", sort=False):
        out.append(br._add_trend_feats(g))
    panel = pd.concat(out, ignore_index=True)
    panel.to_csv(panel_path, index=False)
    if not quiet:
        print(f"[drop] panel rebuilt: {panel['ticker'].nunique()} tickers, "
              f"latest {pd.Timestamp(panel['date'].max()).date()}")
    return panel


import sys as _sys_notify
_sys_notify.path.insert(0, str(ROOT / "code"))
from notify import send as _notify_send   # noqa: E402


def load_recent_burst_picks(days_back: int = 7) -> list[dict]:
    """Collect burst top-5 picks from the last `days_back` daily logs.
    Only reads the base YYYY-MM-DD.json (burst log); skips _drop/_ah/_catalyst
    siblings which have different schemas (list vs dict of {top_up, top_down}).
    """
    out = []
    SUFFIX_SKIP = ("_drop.json", "_ah.json", "_catalyst.json")
    for logfile in sorted(LOG_DIR.glob("*.json"))[-days_back * 4:]:
        if any(logfile.name.endswith(s) for s in SUFFIX_SKIP):
            continue
        try:
            d = json.loads(logfile.read_text())
        except Exception:
            continue
        # Only the burst log has a "predictions" dict. Non-dict payloads are
        # other logs (catalyst scoring) that slipped through the name filter.
        if not isinstance(d, dict):
            continue
        for uni, preds in (d.get("predictions") or {}).items():
            for r in preds[:5]:
                out.append({
                    "log_date": d.get("date"),
                    "universe": uni,
                    "ticker": r["ticker"],
                    "close_prev_date": r.get("close_prev_date"),
                    "prob_burst": r.get("prob_final") or r.get("prob_model"),
                })
    return out


def _score_v2(latest: pd.DataFrame, as_of: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    """Score `latest` with v2 models. Returns frame with `p_drop` column.

    Pipeline:
      1. Build v2 features (ranks + regime + EDGAR)
      2. Raw HGBC probabilities
      3. Rolling-60-day recency recalibration (isotonic on recent walk-forward
         predictions vs. actuals, if the walk-forward file is present)
      4. Live FinBERT news blend: p_final = p_recal * (1 - w * clip(news_score, -3, 3))
    """
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from features_live_v2 import build_v2_live_features
    from sklearn.isotonic import IsotonicRegression

    # 1. Features
    latest_v2 = build_v2_live_features(latest, as_of)

    # 2. Raw probabilities
    feat_list = json.loads((MODELS / "drop_v2_feature_list.json").read_text())["features"]
    X2 = latest_v2[feat_list].fillna(0.0).values
    primary = joblib.load(MODELS / "drop_gbc_v2_raw.joblib")
    p_raw = primary.predict_proba(X2)[:, 1]

    # 3. Recency recal (optional, only if walk-forward pred log exists)
    wf_pred_path = ROOT / "output" / "drop_v2" / "walk_forward_predictions.csv"
    p_recal = p_raw.copy()
    if wf_pred_path.exists():
        try:
            wf = pd.read_csv(wf_pred_path, parse_dates=["date"])
            cutoff = as_of - pd.Timedelta(days=int(cfg.get("drop_v2_recency_days", 60)))
            recent = wf[wf["date"] >= cutoff].dropna(subset=["y_drop"])
            if len(recent) >= 500 and recent["y_drop"].sum() > 0:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(recent["p_v2_raw"].values, recent["y_drop"].values)
                p_recal = iso.transform(p_raw)
                print(f"[drop-v2] recency recal: {len(recent)} rows, "
                      f"pos_rate={recent['y_drop'].mean():.3f}, "
                      f"mean p_raw={p_raw.mean():.3f} -> p_recal={p_recal.mean():.3f}")
        except Exception as e:
            print(f"[drop-v2] recency recal failed: {type(e).__name__}: {e}; using raw")

    # 4. Live news blend (mirrors burst runner's pattern; sign flipped).
    # Only fetch news for the top-20 candidates by p_recal — blending news on
    # tickers that won't clear any alert threshold is wasteful and triggers
    # yfinance rate limits.
    p_final = p_recal.copy()
    news_scores = np.zeros(len(latest_v2))
    w = float(cfg.get("drop_v2_news_blend_weight", 0.10))
    if w > 0:
        try:
            _s.path.insert(0, str(Path(__file__).resolve().parent))
            from news_scorer import get_scorer
            scorer_name = cfg.get("news_scorer", "lexicon_expanded")
            scorer = get_scorer(scorer_name)
            # Index of top-K by p_recal
            k = int(cfg.get("drop_v2_news_top_k", 20))
            order = np.argsort(-p_recal)
            top_idx = order[:k]
            tickers = latest_v2["ticker"].tolist()
            print(f"[drop-v2] fetching FinBERT news for top-{k} "
                  f"(by p_recal) out of {len(tickers)} candidates")
            for i in top_idx:
                t = tickers[i]
                try:
                    if _LOADER is not None:
                        items = _LOADER.news(t)
                    else:
                        items = yf.Ticker(t).news or []
                    if _SNAP is not None:
                        _SNAP.cache_news(t, items)
                    if _ASOF is not None:
                        items = _asof_filter_news(items, _ASOF)
                    titles = [x.get("title", "") for x in items[:10]]
                    if titles:
                        res = scorer(titles)
                        news_scores[i] = res.get("score", 0)
                except Exception:
                    news_scores[i] = 0
            nclip = np.clip(news_scores, -3, 3)
            # Negative news => multiplier > 1 (lifts drop prob). Sign flip
            # vs burst runner where positive news lifts burst prob.
            mult = 1.0 - w * nclip
            p_final = p_recal * mult
            p_final = np.clip(p_final, 0.0, 1.0)
            latest_v2["news_score"] = news_scores
        except Exception as e:
            print(f"[drop-v2] news blend skipped: {type(e).__name__}: {e}")

    latest_v2["p_drop_raw"] = p_raw
    latest_v2["p_drop_recal"] = p_recal
    latest_v2["p_drop"] = p_final

    # 5. Percent-change predictions pred_1d/3d/5d — v2 regressor ensemble.
    #    These replace the "-0.05 * p_aux" magnitude proxy. The ensemble
    #    joblib stores 5 HistGBR models under "models"; we average them.
    #    The regressors take the original 17 FEATS, not the 59-feature v2 set.
    reg_X = latest[FEATS].fillna(0.0).values
    for horizon in ["1d", "3d", "5d"]:
        reg_path = MODELS / f"drop_reg_v2_fwd_{horizon}.joblib"
        try:
            bundle = joblib.load(reg_path)
            preds = np.mean([r.predict(reg_X) for r in bundle["models"]], axis=0)
            latest_v2[f"pred_{horizon}"] = preds
        except Exception as e:
            print(f"[drop-v2] regressor {horizon} failed: {type(e).__name__}: {e}")
            latest_v2[f"pred_{horizon}"] = 0.0
    # Keep aux-classifier probs for reporting/transparency.
    for aux in ["1d", "3d", "5d"]:
        aux_path = MODELS / f"drop_aux_v2_{aux}.joblib"
        if aux_path.exists():
            try:
                m = joblib.load(aux_path)
                latest_v2[f"p_aux_{aux}"] = m.predict_proba(X2)[:, 1]
            except Exception:
                pass
    return latest_v2


# ---------- accuracy eval ----------

def _eval_drop_log(log_path, panel, horizon_days):
    """Grade a drop log's `fresh_drop_alerts` at a single forward horizon.

    For each pick, the base close is found by matching `ref_close` to the
    panel close on a date strictly before the log's date (within 0.3%
    tolerance). Realized return is then `close[base_date + N trading days]
    / ref_close - 1`. Direction is "down" if realized < 0; the magnitude
    targets are −3% and −5% over the horizon.

    Returns {"summary": {...}, "rows": [...]} or None if the log is missing
    or has no fresh alerts.
    """
    if not log_path.exists():
        return None
    try:
        log = json.loads(log_path.read_text())
    except Exception:
        return None
    fresh = log.get("fresh_drop_alerts") or []
    if not fresh:
        return None
    pick_date = pd.Timestamp(log.get("date"))
    p = panel.sort_values(["ticker", "date"])
    rows_out = []
    for r in fresh:
        t = r.get("ticker"); ref_close = r.get("ref_close")
        if not t or ref_close is None:
            continue
        g = p[(p["ticker"] == t) & (p["date"] < pick_date)]
        if g.empty:
            continue
        # Match base by ref_close (≤0.3% tolerance), fall back to latest pre-pick date
        diff = (g["close"] / float(ref_close) - 1).abs()
        cand = g[diff < 0.003]
        base_row = (cand if not cand.empty else g).sort_values("date").iloc[-1]
        base_date = base_row["date"]
        base_close = float(base_row["close"])
        after = p[(p["ticker"] == t) & (p["date"] > base_date)].sort_values("date")
        if len(after) < horizon_days:
            continue
        fwd_close = float(after["close"].iloc[horizon_days - 1])
        realized = fwd_close / base_close - 1
        pred = r.get(f"pred_{horizon_days}d")
        rows_out.append({
            "ticker": t, "rank": r.get("rank"),
            "p_drop": r.get("p_drop"), "p_continuation": r.get("p_continuation"),
            "drop_shape": r.get("drop_shape"),
            "pred": pred, "realized": realized,
        })
    if not rows_out:
        return None
    real = np.array([r["realized"] for r in rows_out])
    pred = np.array([r["pred"] for r in rows_out if r["pred"] is not None])
    real_with_pred = np.array([r["realized"] for r in rows_out if r["pred"] is not None])
    summary = {
        "n": len(rows_out),
        "mean_real": float(np.mean(real)),
        "median_real": float(np.median(real)),
        "down_rate": float(np.mean(real < 0)),
        "hit_neg_3pct": float(np.mean(real <= -0.03)),
        "hit_neg_5pct": float(np.mean(real <= -0.05)),
    }
    if len(pred):
        summary["n_with_pred"] = int(len(pred))
        summary["mean_pred"] = float(np.mean(pred))
        summary["MAE"] = float(np.mean(np.abs(pred - real_with_pred)))
        summary["dir_hit"] = float(np.mean((pred < 0) == (real_with_pred < 0)))
    return {"summary": summary, "rows": rows_out, "source_date": str(pick_date.date())}


def evaluate_prior_drops(panel, log_dir, eval_anchor):
    """Find the most recent drop log files at 1, 3, and 5 trading-day offsets
    and grade each at its appropriate forward horizon. Returns a unified
    eval dict shaped like the burst runner's `eval` block.
    """
    prior_paths: list[tuple[str, Path]] = []
    for delta in range(1, 11):
        y = (eval_anchor - timedelta(days=delta)).strftime("%Y-%m-%d")
        ypath = log_dir / f"{y}_drop.json"
        if ypath.exists():
            prior_paths.append((y, ypath))
            if len(prior_paths) >= 6:
                break
    if not prior_paths:
        return {}
    sources: dict[str, str] = {}
    summaries: dict[str, dict] = {}
    rows_by_h: dict[str, list[dict]] = {}
    for hidx, hname, hdays in ((0, "1d", 1), (2, "3d", 3), (4, "5d", 5)):
        if len(prior_paths) <= hidx:
            continue
        src_date, src_path = prior_paths[hidx]
        try:
            ev = _eval_drop_log(src_path, panel, hdays)
        except Exception as e:
            print(f"[drop-eval] {hname} eval of {src_date} failed: "
                  f"{type(e).__name__}: {e}")
            continue
        if not ev:
            continue
        sources[hname] = src_date
        summaries[hname] = ev["summary"]
        rows_by_h[hname] = ev["rows"]
    return {"sources": sources, "summary": summaries, "rows": rows_by_h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-rebuild", action="store_true",
                    help="Rebuild burst_panel_v8.csv from yfinance before scoring")
    ap.add_argument("--min-p-drop", type=float, default=None,
                    help="Floor p_drop for fresh alerts (overrides config)")
    ap.add_argument("--soft-cap", type=int, default=None,
                    help="Soft cap on fresh alerts shown (default from config, fallback 15)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute + log outputs but suppress telegram/iMessage send")
    ap.add_argument("--asof", default=None,
                    help="Replay as-of 'YYYY-MM-DD HH:MM' (ET). Filters panel, "
                         "intraday, news to before this timestamp.")
    ap.add_argument("--snapshot", default=None,
                    help="Use the named/located snapshot dir instead of "
                         "yfinance live calls. Auto-resolves from --asof "
                         "if not given.")
    args, _ = ap.parse_known_args()

    global _ASOF, _SNAP, _LOADER
    _ASOF = _parse_asof(args.asof)
    if _ASOF is not None:
        print(f"[drop] REPLAY MODE asof={_ASOF}")
        _LOADER = _open_loader(args.snapshot, _ASOF, runner_tag="drop")
        if _LOADER is None:
            print("[drop] no snapshot resolved — replay falls back to live yf")
    if _ASOF is None:
        _SNAP = _open_snapshot("drop")
        print(f"[drop] snapshot dir: {_SNAP.dir}")

    cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    today_iso = _asof_today_iso(_ASOF)
    model_version = str(cfg.get("drop_model_version", "v1")).lower()
    print(f"[drop] model_version={model_version}")

    # --- Panel load + freshness guard ---------------------------------------
    # Previously we just read burst_panel_v8.csv blindly. When the scheduler
    # ran drop before anyone refreshed that file, `close` was stale and the
    # `overnight_gap` computed against it produced 100% p_drop false alarms
    # (see PRU, 2026-04-22). Now: load, check business-day lag, rebuild
    # via yfinance if stale, and refuse to produce alerts when the panel
    # can't be brought current.
    panel_path = DATA / "burst_panel_v8.csv"
    panel = None
    if _LOADER is not None:
        panel = _LOADER.panel("v8")
        if panel is not None:
            print(f"[drop] loaded v8 panel from snapshot ({len(panel)} rows)")
    if panel is None and args.force_rebuild:
        panel = _rebuild_v8_panel_from_yf()
    if panel is None:
        if not panel_path.exists():
            panel_path = DATA / "drop_panel_v1.csv"
        panel = pd.read_csv(panel_path, parse_dates=["date"])

    if _SNAP is not None:
        _SNAP.save_panel("v8", panel)
    if _ASOF is not None:
        panel = _asof_filter_panel(panel, _ASOF)
        print(f"[drop] asof: panel filtered to {panel['date'].max()} "
              f"({len(panel)} rows)")

    today_et = (_ASOF.date() if _ASOF is not None
                else (datetime.now(timezone.utc)
                      .astimezone(timezone(timedelta(hours=-4)))).date())
    panel_max = pd.Timestamp(panel["date"].max())
    lag = _business_day_lag(panel_max, today_et)
    print(f"[drop] panel: {panel['ticker'].nunique()} tickers, "
          f"latest {panel_max.date()} (lag {lag} bd from {today_et})")
    if lag > MAX_PANEL_LAG_BD:
        print(f"[drop] stale (>{MAX_PANEL_LAG_BD} bd) — rebuilding …")
        rebuilt = _rebuild_v8_panel_from_yf()
        if rebuilt is None:
            raise RuntimeError(
                f"drop panel stale ({panel_max.date()}, {lag} bd lag) "
                f"and yfinance rebuild failed — refusing to produce alerts")
        panel = rebuilt
        if _ASOF is not None:
            panel = _asof_filter_panel(panel, _ASOF)
            print(f"[drop] asof: rebuilt panel filtered to {panel['date'].max()}")
        panel_max = pd.Timestamp(panel["date"].max())
        lag = _business_day_lag(panel_max, today_et)
        if lag > MAX_PANEL_LAG_BD:
            raise RuntimeError(
                f"drop panel still stale after rebuild "
                f"({panel_max.date()}, {lag} bd) — refusing to produce alerts")

    vanilla_cols = [f for f in FEATS if f != "overnight_gap"]
    scored = panel.dropna(subset=vanilla_cols).copy()
    latest = scored.sort_values(["ticker", "date"]).groupby("ticker").tail(1).copy()

    # --- Live AH + reference close (yfinance) ------------------------------
    # Belt-and-suspenders on top of the freshness guard: we always pull two
    # bars from yfinance — the most recent regular-session daily close
    # (reference) AND the latest 1-minute post-market/pre-market tick (ah_last).
    # overnight_gap is then ref→ah, independent of the panel's cached close.
    # If yfinance is down we fall back to the panel close. If the panel close
    # disagrees with the yfinance ref close by >PANEL_DRIFT_WARN we log it —
    # that's exactly the stale-panel signature we want to surface.
    from concurrent.futures import ThreadPoolExecutor

    # ah_last sanity: reject a single outlier pre/post-market tick that differs
    # from the rest of the 1-minute series by a large margin. yfinance has
    # shown ghost prints (zero-volume, ~40% dislocation) that corrupt the
    # overnight_gap feature and produce false p_drop=100% calls. We take the
    # median of the last ~20 minutes of prints instead of the final tick when
    # the final tick disagrees with that median by > 15%.
    AH_OUTLIER_PCT = 0.15
    AH_TAIL = 20

    def _live_refs(row):
        t = row["ticker"]
        panel_close = float(row["close"])
        ref_close, ah_last = panel_close, None
        try:
            if _LOADER is not None:
                d = _LOADER.intraday("ref_5d", t)
            else:
                d = yf.Ticker(t).history(period="5d", interval="1d")
            if d is not None and len(d) > 0:
                if _SNAP is not None:
                    _SNAP.cache_intraday("ref_5d", t, d)
                if _ASOF is not None:
                    d = _asof_filter_intraday(d, _ASOF)
                if d is not None and len(d) > 0:
                    ref_close = float(d["Close"].iloc[-1])
        except Exception:
            pass
        try:
            if _LOADER is not None:
                h = _LOADER.intraday("1m", t)
            else:
                h = yf.Ticker(t).history(period="2d", interval="1m", prepost=True)
            if _SNAP is not None and h is not None and len(h) > 0:
                _SNAP.cache_intraday("1m", t, h)
            if _ASOF is not None and h is not None and len(h) > 0:
                h = _asof_filter_intraday(h, _ASOF)
            if h is not None and len(h) > 0:
                last_tick = float(h["Close"].iloc[-1])
                # Robust fallback: median of the last AH_TAIL prints
                tail = h["Close"].dropna().tail(AH_TAIL)
                median_tail = float(tail.median()) if len(tail) else last_tick
                if median_tail > 0:
                    dislocation = abs(last_tick / median_tail - 1.0)
                else:
                    dislocation = 0.0
                if dislocation > AH_OUTLIER_PCT:
                    # treat last tick as a ghost print; use median of the tail
                    ah_last = median_tail
                else:
                    ah_last = last_tick
        except Exception:
            pass
        # Second-layer sanity: if ah_last still disagrees with ref_close by
        # more than 25%, the series itself is suspect — fall back to no gap.
        if ah_last and ref_close and abs(ah_last / ref_close - 1.0) > 0.25:
            ah_last = ref_close
        gap = (ah_last / ref_close - 1.0) if ah_last else 0.0
        drift = (ref_close / panel_close - 1.0) if panel_close else 0.0
        return gap, ah_last, ref_close, drift

    tasks = list(latest.to_dict("records"))
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_live_refs, tasks))
    latest["overnight_gap"] = [r[0] for r in results]
    latest["ah_last"] = [r[1] for r in results]
    latest["ref_close"] = [r[2] for r in results]
    panel_drift = np.array([abs(r[3]) for r in results])
    n_drift = int((panel_drift > PANEL_DRIFT_WARN).sum())
    if n_drift > 0:
        max_drift = float(panel_drift.max())
        print(f"[drop] panel→live close drift >{PANEL_DRIFT_WARN:.0%}: "
              f"{n_drift} tickers (max {max_drift*100:.1f}%) — using yfinance live "
              f"close as reference")

    if model_version == "v2":
        as_of = pd.Timestamp(latest["date"].max())
        latest = _score_v2(latest, as_of, cfg)
        # For v2, rank by p_drop directly (no pred_5d filter). Still expose
        # legacy cols for the rest of the pipeline.
        # Optional post-hoc calibration on the magnitude heads. The raw v2
        # regressors over-predict drop magnitude in BULL regimes (live picks:
        # pred_5d -6.8% vs realized -3.2%) and may need different correction
        # in BEAR regimes. Calibration is regime-conditional: use today's
        # universe-mean 20d return to pick BEAR/NEUTRAL/BULL, then apply
        # that regime's per-horizon (alpha, beta).
        # Set `drop_reg_calibration_version: "v2"` in config to enable.
        # Only rows where raw_pred is past the decision threshold are touched
        # (predictions near zero stay raw).
        cal_version = str(cfg.get("drop_reg_calibration_version", "")).lower()
        if cal_version:
            cal_path = MODELS / f"drop_reg_{cal_version}_calibration.joblib"
            if cal_path.exists():
                cal = joblib.load(cal_path)
                # Detect today's regime from the panel
                pcal = panel.sort_values(["ticker", "date"]).copy()
                pcal["_ret_20"] = pcal.groupby("ticker")["close"].pct_change(20)
                mkt_series = pcal.groupby("date")["_ret_20"].mean()
                if len(mkt_series) and pd.notna(mkt_series.iloc[-1]):
                    current_mkt_ret = float(mkt_series.iloc[-1])
                else:
                    current_mkt_ret = float("nan")
                bnd = cal.get("regime_boundaries") or {}
                bear_max = float(bnd.get("bear_max", 0.0))
                bull_min = float(bnd.get("bull_min", 0.05))
                if pd.isna(current_mkt_ret):
                    regime = cal.get("fallback_regime", "NEUTRAL")
                elif current_mkt_ret < bear_max:
                    regime = "BEAR"
                elif current_mkt_ret >= bull_min:
                    regime = "BULL"
                else:
                    regime = "NEUTRAL"
                print(f"[drop] regime: mkt_ret_20={current_mkt_ret*100:+.2f}% "
                      f"(bear<{bear_max*100:+.2f}%, bull≥{bull_min*100:+.2f}%) "
                      f"→ {regime}")

                regime_horizons = (cal.get("regimes") or {}).get(regime, {}).get("horizons") or {}
                applied = []
                for h in ("1d", "3d", "5d"):
                    params = regime_horizons.get(h)
                    if not params:
                        continue
                    alpha = float(params["alpha"])
                    beta  = float(params["beta"])
                    thr   = float(params.get("fit_pred_threshold", -0.03))
                    col   = f"pred_{h}"
                    raw   = latest[col].copy()
                    latest[f"{col}_raw"] = raw
                    mask = raw <= thr
                    latest.loc[mask, col] = alpha * raw[mask] + beta
                    applied.append(f"{h}(α={alpha:.2f},β={beta*100:+.1f}%,n_adj={int(mask.sum())})")
                print(f"[drop] applied {cal_version} {regime} calibration: {', '.join(applied)}")
            else:
                print(f"[drop] WARNING: drop_reg_calibration_version={cal_version} "
                      f"but {cal_path.name} not found; skipping calibration")
    else:
        # v1 path (unchanged)
        clf = joblib.load(MODELS / "drop_gbc_v1.joblib")["gbc"]
        reg_1d = joblib.load(MODELS / "drop_reg_v1_fwd_1d.joblib")["reg"]
        reg_3d = joblib.load(MODELS / "drop_reg_v1_fwd_3d.joblib")["reg"]
        reg_5d = joblib.load(MODELS / "drop_reg_v1_fwd_5d.joblib")["reg"]
        X = latest[FEATS].values
        latest["p_drop"] = clf.predict_proba(X)[:, 1]
        latest["pred_1d"] = reg_1d.predict(X)
        latest["pred_3d"] = reg_3d.predict(X)
        latest["pred_5d"] = reg_5d.predict(X)

    # 1. Fresh short candidates —
    #    v1 policy: rank by pred_5d (weak 56.5% directional signal), filter p_burst<0.35.
    #    v2 policy: rank by p_drop directly (strong signal), keep burst filter as
    #               a soft exclusion only.
    burst_clf = joblib.load(MODELS / "burst_gbc_v7_augmented.joblib")
    V7_FEATS = [c for c in FEATS if c not in ["ma_stack","up_streak","up_bigdays_20d",
                                                "dist_ma60_atr","ma60_slope_60d","run_length"]]
    latest["p_burst"] = burst_clf["gbc"].predict_proba(latest[V7_FEATS].values)[:, 1]

    # Threshold-based selection (no hard .head(5) cap). Keep all names that
    # clear the p_drop floor and burst exclusion; cap only for message
    # readability. Tunable via CLI (--min-p-drop, --soft-cap) or config keys
    # drop_alert_min_p_drop / drop_alert_soft_cap.
    soft_cap = args.soft_cap if args.soft_cap is not None else \
               int(cfg.get("drop_alert_soft_cap", 15))
    if model_version == "v2":
        drop_floor = args.min_p_drop if args.min_p_drop is not None else \
                     float(cfg.get("drop_alert_min_p_drop", 0.30))
        shortable = latest[(latest["p_drop"] >= drop_floor) &
                           (latest["p_burst"] < 0.50)].copy()
        top = shortable.sort_values("p_drop", ascending=False).head(soft_cap).copy()
    else:
        drop_floor = args.min_p_drop if args.min_p_drop is not None else \
                     float(cfg.get("drop_alert_min_p_drop", 0.25))
        shortable = latest[(latest["p_drop"] >= drop_floor) &
                           (latest["p_burst"] < 0.35)].copy()
        top = shortable.sort_values("pred_5d", ascending=True).head(soft_cap).copy()

    # Manual watchlist: tickers pinned to the drop alert regardless of score.
    # Useful when the model previously misfired (e.g., a yfinance ghost tick)
    # or you specifically want eyes on a name going forward. Tagged [pinned]
    # so they're visually distinct from threshold-passers.
    manual_tickers = [t.upper() for t in cfg.get("drop_watchlist", []) if t]
    if manual_tickers:
        pinned = latest[latest["ticker"].isin(manual_tickers)].copy()
        for_pin = [t for t in manual_tickers
                    if t not in set(top["ticker"]) and t in set(pinned["ticker"])]
        if for_pin:
            add = pinned[pinned["ticker"].isin(for_pin)].copy()
            add["_pinned"] = True
            top["_pinned"] = False
            top = pd.concat([top, add], ignore_index=True)
            print(f"[drop] pinned {len(for_pin)} watchlist ticker(s): {for_pin}")
    # Forward-from-entry filter — exclude drops that already mostly happened.
    # pred_1d is a close-to-close 1-day return. The entry price a live trader
    # can actually use is `ah_last` (post AH / pre-open tick). So the edge you
    # capture is `(close_prev * (1 + pred_1d)) / ah_last - 1`. If that's
    # non-negative, the predicted move is already in the price and there's
    # nothing to short. Require at least -0.5% implied-forward drop.
    # Pinned tickers bypass this filter.
    FWD_FLOOR = float(cfg.get("drop_min_fwd_from_entry", -0.005))
    def _fwd(row):
        entry = row.get("ah_last") or row.get("close") or 0.0
        c0 = row.get("close") or 0.0
        p1 = row.get("pred_1d") or 0.0
        if entry <= 0 or c0 <= 0:
            return 0.0
        return (c0 * (1.0 + p1)) / entry - 1.0
    top["fwd_from_entry"] = top.apply(_fwd, axis=1)
    before = len(top)
    is_pinned = top["_pinned"] if "_pinned" in top.columns else False
    keep = (top["fwd_from_entry"] <= FWD_FLOOR) | is_pinned
    top = top[keep].copy()
    print(f"[drop] fwd-from-entry filter (<= {FWD_FLOOR*100:+.2f}%): "
          f"{before} -> {len(top)}")

    print(f"[drop] passing p_drop>={drop_floor:.0%} with burst filter: "
          f"{len(shortable)} tickers (showing up to {soft_cap})")
    display_cols = ["ticker", "close", "ref_close", "ah_last", "p_drop", "p_burst",
                     "pred_1d", "pred_3d", "pred_5d",
                     "rv_60", "rsi_14", "run_length", "up_streak",
                     "overnight_gap", "fwd_from_entry"]
    if "_pinned" in top.columns:
        display_cols = display_cols + ["_pinned"]
    fresh_rows = top[display_cols].to_dict("records")

    # ---- continuation classifier + PUT candidate filter ----
    # Score each fresh alert with the continuation classifier (models/
    # drop_continuation_clf.joblib). Tag as CONTINUATION or ONE_SHOT and
    # separate the PUT-worthy subset: high p_drop, pred past target with
    # margin, and classifier says the drop will persist through day 5.
    put_rows: list[dict] = []
    # `drop_continuation_clf_version`:
    #   "v1" (default) -> drop_continuation_clf.joblib (legacy, anti-predictive
    #         in 2026-04 live; tagged CONTINUATION rallied +2.97% mean 5d).
    #   "v2"           -> drop_continuation_clf_v2.joblib (path-based labels:
    #         CONTINUATION = real_3d<=-3% AND real_5d<=real_3d. Holdout AUC
    #         0.678 vs v1's 0.624; on the 2026-04 live picks the new top-5
    #         drops by p_cont averaged -4.9% real_5d / 80% down vs the old
    #         top-5 at +7.0% / 20% down).
    cont_version = str(cfg.get("drop_continuation_clf_version", "v1")).lower()
    cont_filename = ("drop_continuation_clf.joblib" if cont_version == "v1"
                     else f"drop_continuation_clf_{cont_version}.joblib")
    cont_path = MODELS / cont_filename
    if cont_path.exists() and len(fresh_rows) > 0:
        cont = joblib.load(cont_path)
        cont_feats = cont["feats"]; cont_gbc = cont["gbc"]; cont_sc = cont["scaler"]
        print(f"[drop] continuation classifier: {cont_filename} "
              f"(version={cont_version})")
        # Need to pull the features from `top` (same dataframe, same indexing)
        X_cont = cont_sc.transform(top[cont_feats].fillna(0.0).values)
        p_cont = cont_gbc.predict_proba(X_cont)[:, 1]
        for r, pc in zip(fresh_rows, p_cont):
            r["p_continuation"] = float(pc)
            r["drop_shape"] = "CONTINUATION" if pc >= 0.5 else "ONE_SHOT"

        # PUT filter:
        #   - p_drop           >= 0.65  (high-conviction drop)
        #   - p_continuation   >= 0.50  (classifier says it persists)
        #   - pred is "far past target" for at least one horizon (target + ~1pt
        #     margin so we're definitively past, not borderline):
        #         pred_1d <= -4%  (past y_drop_1d's -3%)
        #      OR pred_3d <= -6%  (past y_drop_3d's -5%)
        #      OR pred_5d <= -8%  (past y_drop_5d's -7%)
        #
        # No overnight_gap exclusion: the continuation classifier already
        # factors in whether a big pre-open gap will extend or bounce (it's
        # the top-importance feature, weight 0.42). Let that judgment stand.
        def _is_put(r: dict) -> bool:
            if r["p_drop"] < 0.65: return False
            if r.get("p_continuation", 0.0) < 0.50: return False
            far = (r.get("pred_1d", 0.0) <= -0.04
                    or r.get("pred_3d", 0.0) <= -0.06
                    or r.get("pred_5d", 0.0) <= -0.08)
            return far

        put_rows = [r for r in fresh_rows if _is_put(r)]
        # Rank PUTs by p_continuation * |pred_5d| (continuation + magnitude)
        for r in put_rows:
            r["put_rank_score"] = float(
                r["p_continuation"] * abs(r.get("pred_5d", 0.0)))
        put_rows.sort(key=lambda r: r["put_rank_score"], reverse=True)
        print(f"[drop] PUT candidates: {len(put_rows)} passing "
              f"(p_drop>=65%, p_cont>=50%, 1d<=-4% OR 3d<=-6% OR 5d<=-8%)")

    # 2. Watch list: prior burst picks whose p_drop is now elevated
    thr = float(cfg.get("drop_prob_warn_threshold", 0.30))
    prior = load_recent_burst_picks(days_back=7)
    watch_rows = []
    idx = latest.set_index("ticker")
    for pick in prior:
        t = pick["ticker"]
        if t not in idx.index:
            continue
        r = idx.loc[t]
        p_drop = float(r["p_drop"])
        if p_drop >= thr:
            # Use ref_close (yfinance live) not panel close — avoids showing
            # a stale price when the panel didn't refresh intraday.
            close_now = float(r.get("ref_close") or r["close"])
            watch_rows.append({
                **pick, "p_drop_now": p_drop,
                "close_now": close_now,
            })
    # dedup: keep highest p_drop per ticker
    if watch_rows:
        wdf = pd.DataFrame(watch_rows).sort_values("p_drop_now", ascending=False)
        wdf = wdf.drop_duplicates(subset="ticker", keep="first")
        watch_rows = wdf.to_dict("records")

    # save
    pd.DataFrame(fresh_rows).to_csv(OUT / "drop_live_today.csv", index=False)

    # Full per-ticker scores for the entire universe (1,877 Robinhood-tradable
    # names on the refreshed v8 panel). This gives every ticker a drop score
    # even if it doesn't clear the alert threshold — useful for downstream
    # filtering, hedging, or cross-referencing with positions.
    full_cols = ["ticker", "close", "ref_close", "ah_last",
                 "p_drop", "p_burst",
                 "pred_1d", "pred_3d", "pred_5d",
                 "rv_60", "rsi_14", "macd", "bb_z20", "atr_pct",
                 "run_length", "up_streak", "ma_stack", "dist_ma60_atr",
                 "overnight_gap"]
    # Some columns (p_drop_raw / p_drop_recal / p_aux_*) are v2-only — include
    # them if present.
    for c in ("p_drop_raw", "p_drop_recal", "p_aux_1d", "p_aux_3d", "p_aux_5d",
              "news_score"):
        if c in latest.columns:
            full_cols.append(c)
    full_cols = [c for c in full_cols if c in latest.columns]
    full_df = (latest[full_cols]
                 .sort_values("p_drop", ascending=False)
                 .reset_index(drop=True))
    full_df.to_csv(OUT / "drop_scores_all.csv", index=False)
    print(f"[drop] wrote full per-ticker scores: {len(full_df)} tickers -> "
          f"output/drop_scores_all.csv")

    # Grade prior drop predictions at 1d / 3d / 5d so the morning message
    # can show realized accuracy for picks made 1, 3, and 5 trading days ago.
    eval_anchor = _ASOF if _ASOF is not None else datetime.now()
    if not isinstance(eval_anchor, datetime):
        eval_anchor = datetime.fromisoformat(str(eval_anchor))
    try:
        prior_eval = evaluate_prior_drops(panel, LOG_DIR, eval_anchor)
    except Exception as e:
        print(f"[drop-eval] prior eval failed: {type(e).__name__}: {e}")
        prior_eval = {}

    dlog = {
        "date": today_iso, "threshold": thr,
        "n_scored": int(len(latest)),
        "fresh_drop_alerts": fresh_rows,
        "put_candidates": put_rows,
        "watch_list_from_prior_bursts": watch_rows,
        "eval": prior_eval,
    }
    (LOG_DIR / f"{today_iso}_drop.json").write_text(json.dumps(dlog, indent=2, default=str))

    # compose message
    lines = []
    lines.append(f"⚠️ Drop alerts — {today_iso}")
    lines.append("")
    if prior_eval and prior_eval.get("summary"):
        srcs = prior_eval.get("sources") or {}
        lines.append("Prior drop accuracy (close-to-close):")
        for h in ("1d", "3d", "5d"):
            agg = (prior_eval["summary"] or {}).get(h)
            if not agg:
                continue
            src = srcs.get(h, "?")
            mp = agg.get("mean_pred")
            mp_str = f"pred {mp*100:+.1f}% / " if mp is not None else ""
            lines.append(
                f"  {h} ({src}): {mp_str}"
                f"real {agg['mean_real']*100:+.1f}%  "
                f"down {agg['down_rate']*100:.0f}%  "
                f"≤-3% {agg['hit_neg_3pct']*100:.0f}%  "
                f"≤-5% {agg['hit_neg_5pct']*100:.0f}%  (n={agg['n']})"
            )
        lines.append("")
    if watch_rows:
        lines.append(f"Watch (prev picks, p_drop >= {thr:.0%}):")
        for r in watch_rows[:5]:
            lines.append(f"  {r['ticker']:5s} p_drop={r['p_drop_now']*100:.0f}%  "
                         f"(was rec'd {r['log_date']} in {r['universe']})")
        lines.append("")
    if fresh_rows:
        rank_tag = "ranked by p_drop" if model_version == "v2" else "ranked by expected 5d decline"
        lines.append(f"Short candidates — {len(fresh_rows)} passing (p_drop>={drop_floor:.0%}), {rank_tag}:")
        for r in fresh_rows:
            ref = r.get("ref_close") or r["close"]
            if r.get("ah_last"):
                gap = r["ah_last"] / ref - 1
                price_str = f"${ref:.2f}\u2192${r['ah_last']:.2f} ({gap*100:+.1f}%)"
            else:
                price_str = f"${ref:.2f}"
            shape_tag = ""
            if r.get("drop_shape"):
                shape_tag = (" [CONT]" if r["drop_shape"] == "CONTINUATION"
                              else " [1-SHOT]")
            if r.get("_pinned"):
                shape_tag += " [pinned]"
            lines.append(f"  {r['ticker']:5s} {price_str}  p_drop={r['p_drop']*100:.0f}%  "
                         f"1d {r['pred_1d']*100:+.1f}% / 3d {r['pred_3d']*100:+.1f}% / "
                         f"5d {r['pred_5d']*100:+.1f}%  (RSI {r['rsi_14']:.0f}){shape_tag}")
    else:
        lines.append(f"Short candidates: none meet the bar today "
                     f"(p_drop >= {drop_floor:.0%} AND p_burst < "
                     f"{'50%' if model_version=='v2' else '35%'}).")

    if put_rows:
        lines.append("")
        lines.append(f"🎯 PUT candidates — {len(put_rows)} "
                     f"(p_drop≥65%, p_cont≥50%, past target):")
        for r in put_rows[:10]:
            ref = r.get("ref_close") or r["close"]
            lines.append(f"  {r['ticker']:5s} ${ref:.2f}  p_drop={r['p_drop']*100:.0f}%  "
                         f"p_cont={r['p_continuation']*100:.0f}%  "
                         f"1d {r['pred_1d']*100:+.1f}% / 3d {r['pred_3d']*100:+.1f}% / "
                         f"5d {r['pred_5d']*100:+.1f}%  (RSI {r['rsi_14']:.0f})")
    elif fresh_rows and all("p_continuation" in r for r in fresh_rows):
        # No PUTs passed — explain why by showing the closest near-miss.
        near = sorted(
            fresh_rows,
            key=lambda r: -(r["p_drop"] * r["p_continuation"] *
                             max(abs(r.get("pred_1d",0)), abs(r.get("pred_3d",0)),
                                  abs(r.get("pred_5d",0)))))[:1]
        if near:
            n = near[0]
            lines.append("")
            lines.append(f"🎯 PUT candidates: none qualify today. "
                         f"Closest: {n['ticker']} p_drop={n['p_drop']*100:.0f}% "
                         f"p_cont={n['p_continuation']*100:.0f}% "
                         f"(5d {n['pred_5d']*100:+.1f}%).")
    msg = "\n".join(lines)
    print("\n=== DROP MESSAGE ===\n" + msg + "\n====================\n")

    if args.dry_run:
        print("[notify drop] skipped (--dry-run)")
    elif _ASOF is not None:
        print("[notify drop] skipped (asof replay)")
    elif cfg.get("send_drop_notifications", cfg.get("send_notifications", False)):
        ok = _notify_send(msg)
        print(f"[notify drop] sent: {ok}")
    else:
        print("[notify drop] skipped (send_notifications=false)")

    if _SNAP is not None:
        _SNAP.update_meta(today_iso=today_iso, dry_run=bool(args.dry_run),
                          message_text=msg,
                          tickers_in_message=sorted(
                              {r["ticker"] for r in fresh_rows}
                              | {r["ticker"] for r in put_rows}
                              | {r["ticker"] for r in watch_rows}))
        out = _SNAP.flush()
        print(f"[drop] snapshot flushed -> {out}")


if __name__ == "__main__":
    main()
