"""HOLD UNTIL SIGNAL DROPS — exit driven by the model, not by price.

Strategy:
  ENTRY: same as before — top-5 by v3 raw_margin on day t (with SPY regime filter).
  HOLD:  re-score the stock daily. While signal stays above hold_threshold,
         keep the position.
  EXIT:  at the first day the signal drops below hold_threshold. Sell at
         that day's close.
  CAP:   force-sell if held > max_hold_days.

We sweep across exit thresholds (margin > 0, > -0.2, > -0.5; or prob_cal
> 0.10, > 0.05) and max_hold caps (30, 60, 90 days) to find the
holding rule that earns most while still being a "model-driven" exit.

Reports realized per-pick return distribution and daily-basket Sharpe
when entries are equal-weighted into a rolling portfolio.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output" / "monthly_gainer"

sys.path.insert(0, str(ROOT / "code"))
from regime_features import attach_regime  # noqa: E402

XRANK_FEATURES = ["rsi_14_xrank", "rv_60_xrank", "ma60_slope_xrank", "ret_20d_xrank"]


def add_xrank(df):
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["close_20d_ago"] = df.groupby("ticker")["close"].shift(20)
    df["ret_20d_lag"] = df["close"] / df["close_20d_ago"] - 1.0
    df["close_5d_ago"] = df.groupby("ticker")["close"].shift(5)
    df["ret_5d_lag"] = df["close"] / df["close_5d_ago"] - 1.0
    df["rsi_14_xrank"] = df.groupby("date")["rsi_14"].rank(pct=True)
    df["rv_60_xrank"] = df.groupby("date")["rv_60"].rank(pct=True)
    df["ma60_slope_xrank"] = df.groupby("date")["ma60_slope_60d"].rank(pct=True)
    df["ret_20d_xrank"] = df.groupby("date")["ret_20d_lag"].rank(pct=True)
    return df


def score_full_panel(panel_path, cat_path, model_path):
    """Return DataFrame with cols [ticker, date, close, prob_cal, raw_margin]
    covering ALL rows with complete features (not just labeled)."""
    art = joblib.load(model_path)
    gbc = art["raw_gbc"]; cal = art["calibrator"]
    feats = art["feats"]; med = pd.Series(art["impute_medians"])
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    cat = pd.read_csv(cat_path, parse_dates=["date"])
    panel = panel.merge(cat, on=["ticker", "date"], how="left")
    panel = attach_regime(panel)
    panel = add_xrank(panel)
    for c in ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
              "news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
              "ma_news_5d", "ma_news_20d", "sector_pop_5d"]:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0)
        else:
            panel[c] = 0.0
    for c in XRANK_FEATURES:
        panel[c] = panel[c].fillna(0.5)

    full = panel.dropna(subset=feats).copy()
    X = full[feats].fillna(med).values
    full["prob_cal"] = cal.predict_proba(X)[:, 1]
    full["raw_margin"] = gbc.decision_function(X)
    full = full[["ticker", "date", "close", "prob_cal", "raw_margin",
                  "spy_ret_20d", "y21"]].copy()
    return full.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_ticker_lookup(scores: pd.DataFrame) -> dict:
    """ticker -> ndarray of [date_idx, close, prob_cal, raw_margin]."""
    out = {}
    for tk, g in scores.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        out[tk] = {
            "dates": g["date"].values,
            "close": g["close"].values,
            "prob": g["prob_cal"].values,
            "margin": g["raw_margin"].values,
        }
    return out


def simulate_hold(lookup_t: dict, entry_idx: int, threshold: float,
                  max_hold: int, score_field: str = "margin") -> tuple:
    """Hold from entry_idx forward. Exit when score < threshold or after max_hold days.
    Returns (realized_return, hold_days, exit_reason)."""
    close = lookup_t["close"]
    score = lookup_t[score_field]
    n = len(close)
    entry_close = close[entry_idx]
    if not np.isfinite(entry_close) or entry_close <= 0:
        return None, 0, "bad_entry"
    for offset in range(1, max_hold + 1):
        idx = entry_idx + offset
        if idx >= n:
            # End of data — force sell at last available
            exit_close = close[n - 1]
            return float(exit_close / entry_close - 1.0), n - 1 - entry_idx, "end_of_data"
        if score[idx] < threshold:
            exit_close = close[idx]
            return float(exit_close / entry_close - 1.0), offset, "signal_dropped"
    # Hit max_hold cap
    idx = entry_idx + max_hold
    if idx >= n:
        idx = n - 1
    exit_close = close[idx]
    return float(exit_close / entry_close - 1.0), max_hold, "max_hold"


def get_test_entries(scores: pd.DataFrame, k: int = 5,
                     score_col: str = "raw_margin",
                     regime_filter: bool = True):
    """Top-K daily entries from test fold (last 15% of dates)."""
    dates = np.sort(scores["date"].unique())
    test_start = dates[int(len(dates) * 0.85)]
    test = scores[scores["date"] >= test_start].copy()
    if regime_filter:
        test = test[test["spy_ret_20d"] > 0]
    entries = []
    for d, g in test.groupby("date"):
        if len(g) < k:
            continue
        topk = g.nlargest(k, score_col)
        for _, r in topk.iterrows():
            entries.append({
                "date": r["date"], "ticker": r["ticker"],
                "score": r[score_col], "y21": r["y21"],
            })
    return pd.DataFrame(entries)


def run_config(scores: pd.DataFrame, lookup: dict, label: str,
                threshold: float, max_hold: int,
                score_field: str = "margin", entry_col: str = "raw_margin"):
    entries = get_test_entries(scores, k=5, score_col=entry_col, regime_filter=True)
    rets, holds, reasons = [], [], []
    daily_baskets = {}
    for _, e in entries.iterrows():
        tk = e["ticker"]
        if tk not in lookup:
            continue
        d = e["date"]
        idxs = np.where(lookup[tk]["dates"] == np.datetime64(d))[0]
        if len(idxs) == 0:
            continue
        entry_idx = int(idxs[0])
        ret, hold, reason = simulate_hold(lookup[tk], entry_idx, threshold, max_hold, score_field)
        if ret is None:
            continue
        rets.append(ret); holds.append(hold); reasons.append(reason)
        daily_baskets.setdefault(d, []).append(ret)

    if not rets:
        return None
    rets = np.array(rets); holds = np.array(holds)

    # Daily basket return = mean of picks entered that day
    basket_rets = np.array([np.mean(v) for v in daily_baskets.values()])

    out = {
        "config": label,
        "threshold": threshold,
        "max_hold": max_hold,
        "n_picks": len(rets),
        "hit_rate": float((rets > 0).mean()),
        "mean": float(rets.mean()),
        "median": float(np.median(rets)),
        "std_pick": float(rets.std()),
        "basket_mean": float(basket_rets.mean()),
        "basket_std": float(basket_rets.std()),
        "basket_sharpe": float(basket_rets.mean() / basket_rets.std()) if basket_rets.std() > 0 else float("nan"),
        "loser_rate": float((rets < 0).mean()),
        "big_loss_rate": float((rets < -0.15).mean()),
        "min_pick": float(rets.min()),
        "max_pick": float(rets.max()),
        "avg_hold_days": float(holds.mean()),
        "median_hold_days": float(np.median(holds)),
        "max_hold_days": int(holds.max()),
        "exit_signal_pct": float(np.mean([r == "signal_dropped" for r in reasons])),
        "exit_max_hold_pct": float(np.mean([r == "max_hold" for r in reasons])),
        "exit_eod_pct": float(np.mean([r == "end_of_data" for r in reasons])),
    }
    return out


def main():
    print("[98] scoring SP500 full panel ...")
    sp_scores = score_full_panel(
        DATA / "monthly_gainer_panel.csv",
        DATA / "catalyst_features_sp500.csv",
        MODELS / "monthly_gainer_v3_sp500.joblib")
    print(f"   scored: {len(sp_scores):,}")
    sp_lookup = build_ticker_lookup(sp_scores)

    configs = [
        # (label, threshold, max_hold, score_field)
        # raw_margin thresholds
        ("margin>0_30d",   0.0,  30, "margin"),
        ("margin>0_60d",   0.0,  60, "margin"),
        ("margin>0_90d",   0.0,  90, "margin"),
        ("margin>0_120d",  0.0, 120, "margin"),
        ("margin>-0.2_60d", -0.2, 60, "margin"),
        ("margin>-0.5_60d", -0.5, 60, "margin"),
        ("margin>-0.5_90d", -0.5, 90, "margin"),
        ("margin>0.5_60d", 0.5, 60, "margin"),
        # prob_cal thresholds
        ("prob>0.10_60d",  0.10, 60, "prob"),
        ("prob>0.10_90d",  0.10, 90, "prob"),
        ("prob>0.05_60d",  0.05, 60, "prob"),
        ("prob>0.05_90d",  0.05, 90, "prob"),
        ("prob>0.02_60d",  0.02, 60, "prob"),
        ("prob>0.15_60d",  0.15, 60, "prob"),
        # baseline 21d hold (for comparison)
        ("BASE_21d",      -1e9, 21, "margin"),  # never exits early — held 21d max
        ("BASE_60d",      -1e9, 60, "margin"),
    ]

    print("\n[98] === SP500 (entry: regime + top-5 raw_margin) ===")
    rows = []
    for cfg in configs:
        s = run_config(sp_scores, sp_lookup, cfg[0], cfg[1], cfg[2], cfg[3])
        if s is None:
            continue
        rows.append(s)
        print(f"   {s['config']:<22} n={s['n_picks']:>4}  "
              f"mean={s['mean']:+.1%}  med={s['median']:+.1%}  "
              f"sharpe={s['basket_sharpe']:.2f}  "
              f"hold={s['avg_hold_days']:.0f}d  "
              f"big_loss={s['big_loss_rate']:.0%}  min={s['min_pick']:+.0%}  "
              f"by-signal/cap/eod = {s['exit_signal_pct']:.0%}/{s['exit_max_hold_pct']:.0%}/{s['exit_eod_pct']:.0%}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "hold_until_signal_drops_sp500.csv", index=False)
    print(f"\n[98] saved {OUT / 'hold_until_signal_drops_sp500.csv'}")

    # ===== SMALLCAP — same configs =====
    print("\n[98] scoring SMALLCAP full panel ...")
    sc_scores = score_full_panel(
        DATA / "monthly_gainer_panel_smallcap.csv",
        DATA / "catalyst_features_smallcap.csv",
        MODELS / "monthly_gainer_v3_smallcap.joblib")
    print(f"   scored: {len(sc_scores):,}")
    sc_lookup = build_ticker_lookup(sc_scores)

    sc_configs = [
        ("margin>0_60d",   0.0,  60, "margin"),
        ("margin>-0.5_60d", -0.5, 60, "margin"),
        ("prob>0.30_60d",  0.30, 60, "prob"),
        ("prob>0.20_60d",  0.20, 60, "prob"),
        ("prob>0.15_60d",  0.15, 60, "prob"),
        ("prob>0.15_90d",  0.15, 90, "prob"),
        ("prob>0.10_60d",  0.10, 60, "prob"),
        ("prob>0.05_60d",  0.05, 60, "prob"),
        ("BASE_21d",      -1e9, 21, "margin"),
        ("BASE_60d",      -1e9, 60, "margin"),
    ]
    print("\n[98] === SMALLCAP (entry: regime + top-5 raw_margin) ===")
    sc_rows = []
    for cfg in sc_configs:
        s = run_config(sc_scores, sc_lookup, cfg[0], cfg[1], cfg[2], cfg[3])
        if s is None:
            continue
        sc_rows.append(s)
        print(f"   {s['config']:<22} n={s['n_picks']:>4}  "
              f"mean={s['mean']:+.1%}  med={s['median']:+.1%}  "
              f"sharpe={s['basket_sharpe']:.2f}  "
              f"hold={s['avg_hold_days']:.0f}d  "
              f"big_loss={s['big_loss_rate']:.0%}  min={s['min_pick']:+.0%}  "
              f"by-signal/cap/eod = {s['exit_signal_pct']:.0%}/{s['exit_max_hold_pct']:.0%}/{s['exit_eod_pct']:.0%}")
    pd.DataFrame(sc_rows).to_csv(OUT / "hold_until_signal_drops_smallcap.csv", index=False)


if __name__ == "__main__":
    main()
