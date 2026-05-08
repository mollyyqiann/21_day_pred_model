"""Two analyses:

1. TIEBREAKER VALIDATION — within the saturated calibrated-prob bucket
   (prob_cal >= 0.30), does raw GBC margin predict better outcomes?
   If yes, we can use raw margin to rank within the saturated tier.

2. HEAD-TO-HEAD BACKTEST — SP500 vs SMALLCAP top-K daily picks under v3.
   Computes mean / median per-pick realized return, win rate, std of returns,
   and Sharpe-like ratio. Tells us which universe earns more, and what
   risk-adjusted ratio looks like.
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


def score_test_fold(panel_path, cat_path, model_path):
    art = joblib.load(model_path)
    gbc = art["raw_gbc"]
    cal = art["calibrator"]
    feats = art["feats"]
    med = pd.Series(art["impute_medians"])

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

    lab = panel[panel["y21"].isin([0, 1])].dropna(subset=feats).copy()

    # Last 15% of dates = test fold
    dates = np.sort(lab["date"].unique())
    test_start = dates[int(len(dates) * 0.85)]
    test = lab[lab["date"] >= test_start].copy()

    X = test[feats].fillna(med).values
    test["prob_cal"] = cal.predict_proba(X)[:, 1]
    test["raw_margin"] = gbc.decision_function(X)
    return test


def tiebreaker_validation(test, label, prob_thresh=0.20):
    """Within prob_cal >= prob_thresh, bucket by raw_margin and check
    if higher margin → higher realized return."""
    sat = test[test["prob_cal"] >= prob_thresh].copy()
    if len(sat) < 50:
        prob_thresh_alt = test["prob_cal"].quantile(0.95)
        sat = test[test["prob_cal"] >= prob_thresh_alt].copy()
        print(f"[95] {label}: top bucket too small at {prob_thresh:.2f}; using p95={prob_thresh_alt:.3f}")

    print(f"\n[95] === {label} TIEBREAKER VALIDATION ===")
    print(f"     Saturated bucket (prob_cal >= {prob_thresh:.2f}): n={len(sat)}, hit_rate={sat['y21'].mean():.2%}")

    # Bucket by raw_margin quintiles
    sat["margin_q"] = pd.qcut(sat["raw_margin"], 5, labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"], duplicates="drop")
    summary = sat.groupby("margin_q", observed=True).agg(
        n=("raw_margin", "size"),
        margin_lo=("raw_margin", "min"),
        margin_hi=("raw_margin", "max"),
        prob_cal_mean=("prob_cal", "mean"),
        hit_rate=("y21", "mean"),
        peak_ret=("max_fwd21_ret", "mean"),
        end_ret=("end_of_window_ret", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))

    # Spread: top quintile vs bottom quintile
    q1 = sat[sat["margin_q"] == "Q1 (low)"]
    q5 = sat[sat["margin_q"] == "Q5 (high)"]
    spread = {
        "hit_q5_vs_q1": q5["y21"].mean() - q1["y21"].mean(),
        "peak_q5_vs_q1": q5["max_fwd21_ret"].mean() - q1["max_fwd21_ret"].mean(),
        "end_q5_vs_q1": q5["end_of_window_ret"].mean() - q1["end_of_window_ret"].mean(),
    }
    print(f"     Q5-Q1 spread: hit {spread['hit_q5_vs_q1']:+.1%}, "
          f"peak {spread['peak_q5_vs_q1']:+.1%}, end {spread['end_q5_vs_q1']:+.1%}")
    return summary, spread


def topk_daily_backtest(test, label, k=5, score_col="prob_cal"):
    """Mean per-pick realized return + risk metrics."""
    rows = []
    daily_returns = []
    for d, g in test.groupby("date"):
        if len(g) < k:
            continue
        topk = g.nlargest(k, score_col)
        for _, r in topk.iterrows():
            rows.append({
                "date": d,
                "ticker": r["ticker"],
                "score": r[score_col],
                "y21": r["y21"],
                "peak_ret": r["max_fwd21_ret"],
                "end_ret": r["end_of_window_ret"],
            })
        # daily mean (equal-weighted basket)
        daily_returns.append({
            "date": d,
            "basket_peak": topk["max_fwd21_ret"].mean(),
            "basket_end": topk["end_of_window_ret"].mean(),
            "basket_hit": topk["y21"].mean(),
        })
    picks = pd.DataFrame(rows)
    daily = pd.DataFrame(daily_returns)
    print(f"\n[95] === {label} ({score_col}, top-{k}) BACKTEST ===")
    print(f"     n_picks={len(picks)} ({len(daily)} trading days)")
    print(f"     hit rate (per pick):  mean={picks['y21'].mean():.2%}  median={picks['y21'].median():.2%}")
    print(f"     peak return per pick: mean={picks['peak_ret'].mean():+.2%}  median={picks['peak_ret'].median():+.2%}  std={picks['peak_ret'].std():.2%}")
    print(f"     end return per pick:  mean={picks['end_ret'].mean():+.2%}  median={picks['end_ret'].median():+.2%}  std={picks['end_ret'].std():.2%}")
    # risk metrics on daily basket
    print(f"     daily basket end-ret: mean={daily['basket_end'].mean():+.2%}  std={daily['basket_end'].std():.2%}  "
          f"sharpe={daily['basket_end'].mean()/daily['basket_end'].std():.2f}")
    # downside
    losers = picks[picks["end_ret"] < 0]
    print(f"     loser rate: {len(losers)/len(picks):.2%}  avg loss: {losers['end_ret'].mean():+.2%}")
    big_losers = picks[picks["end_ret"] < -0.15]
    print(f"     big-loss rate (<-15%): {len(big_losers)/len(picks):.2%}  avg: {big_losers['end_ret'].mean() if len(big_losers) else 0:+.2%}")
    return picks, daily


def main():
    print("[95] loading + scoring SP500 test fold ...")
    sp_test = score_test_fold(
        DATA / "monthly_gainer_panel.csv",
        DATA / "catalyst_features_sp500.csv",
        MODELS / "monthly_gainer_v3_sp500.joblib")
    print(f"     SP500 test: {len(sp_test)} rows")

    print("[95] loading + scoring SMALLCAP test fold ...")
    sc_test = score_test_fold(
        DATA / "monthly_gainer_panel_smallcap.csv",
        DATA / "catalyst_features_smallcap.csv",
        MODELS / "monthly_gainer_v3_smallcap.joblib")
    print(f"     SMALLCAP test: {len(sc_test)} rows")

    # === Tiebreaker validation ===
    sp_summary, sp_spread = tiebreaker_validation(sp_test, "SP500", prob_thresh=0.20)
    sc_summary, sc_spread = tiebreaker_validation(sc_test, "SMALLCAP", prob_thresh=0.30)

    # === Backtest: top-K by calibrated prob (baseline) ===
    sp_picks, sp_daily = topk_daily_backtest(sp_test, "SP500", k=5, score_col="prob_cal")
    sc_picks, sc_daily = topk_daily_backtest(sc_test, "SMALLCAP", k=5, score_col="prob_cal")

    # === Backtest: top-K by raw margin (proposed tiebreaker) ===
    sp_picks_m, _ = topk_daily_backtest(sp_test, "SP500", k=5, score_col="raw_margin")
    sc_picks_m, _ = topk_daily_backtest(sc_test, "SMALLCAP", k=5, score_col="raw_margin")

    # === Net comparison ===
    print("\n[95] === FINAL NET COMPARISON ===")
    rows = []
    for label, picks in [("SP500 by prob_cal", sp_picks), ("SP500 by raw_margin", sp_picks_m),
                          ("SMALLCAP by prob_cal", sc_picks), ("SMALLCAP by raw_margin", sc_picks_m)]:
        rows.append({
            "strategy": label,
            "n_picks": len(picks),
            "hit_rate": picks["y21"].mean(),
            "mean_peak": picks["peak_ret"].mean(),
            "mean_end": picks["end_ret"].mean(),
            "median_end": picks["end_ret"].median(),
            "std_end": picks["end_ret"].std(),
            "sharpe": picks["end_ret"].mean() / picks["end_ret"].std() if picks["end_ret"].std() > 0 else float("nan"),
            "loser_rate": (picks["end_ret"] < 0).mean(),
            "big_loss_rate_-15%": (picks["end_ret"] < -0.15).mean(),
        })
    final = pd.DataFrame(rows)
    print(final.to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT / "tiebreaker_backtest_summary.csv", index=False)
    sp_summary.to_csv(OUT / "tiebreaker_sp500_quintiles.csv", index=False)
    sc_summary.to_csv(OUT / "tiebreaker_smallcap_quintiles.csv", index=False)
    print(f"\n[95] saved 3 files in {OUT}")


if __name__ == "__main__":
    main()
