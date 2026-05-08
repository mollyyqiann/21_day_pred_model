"""Backtest: 'partial winner' picks vs 'about to rise' picks.

Splits the Option 1B test-fold entries by their 5d-return at entry:
  - PARTIAL WINNER: 5d_ret in [+5%, +30%) — already moving
  - ABOUT TO RISE : |5d_ret| < 5% — fresh setup
  - DEEP DIP      : 5d_ret < -5% — beaten down
  - TOO HOT       : 5d_ret >= +30% — extreme

For each subset, reports:
  - n picks
  - hit rate (touched +30% in 21d)
  - mean / median realized end-of-window return
  - mean / median peak return
  - Sharpe (per-pick mean / std)
  - big-loss rate (-15%)
  - also: cross-tab with extension (20d return)

Tells the user definitively: do you give up upside by skipping partial winners?
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


def score_test_fold():
    art = joblib.load(MODELS / "monthly_gainer_v3_sp500.joblib")
    gbc = art["raw_gbc"]; cal = art["calibrator"]
    feats = art["feats"]; med = pd.Series(art["impute_medians"])
    panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    cat = pd.read_csv(DATA / "catalyst_features_sp500.csv", parse_dates=["date"])
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
    dates = np.sort(lab["date"].unique())
    test_start = dates[int(len(dates) * 0.85)]
    test = lab[lab["date"] >= test_start].copy()
    X = test[feats].fillna(med).values
    test["prob_cal"] = cal.predict_proba(X)[:, 1]
    test["raw_margin"] = gbc.decision_function(X)
    return test


def stats(rets, label):
    rets = np.asarray(rets)
    if len(rets) == 0:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": int(len(rets)),
        "hit_pos": float((rets > 0).mean()),
        "hit_30pct": float((rets >= 0.30).mean()),
        "hit_50pct": float((rets >= 0.50).mean()),
        "hit_100pct": float((rets >= 1.00).mean()),
        "mean_end": float(rets.mean()),
        "median_end": float(np.median(rets)),
        "std_end": float(rets.std()),
        "sharpe": float(rets.mean() / rets.std()) if rets.std() > 0 else float("nan"),
        "loser_rate": float((rets < 0).mean()),
        "big_loss_15pct": float((rets < -0.15).mean()),
        "big_loss_30pct": float((rets < -0.30).mean()),
        "min": float(rets.min()),
        "max": float(rets.max()),
    }


def main():
    print("[106] scoring SP500 test fold ...")
    test = score_test_fold()

    # Apply Option 1B entry filter: regime ON + top-5 by raw_margin daily
    test = test[test["spy_ret_20d"] > 0]
    rows = []
    for d, g in test.groupby("date"):
        if len(g) < 5:
            continue
        topk = g.nlargest(5, "raw_margin")
        for _, r in topk.iterrows():
            rows.append({
                "date": d, "ticker": r["ticker"],
                "ret_5d": r["ret_5d_lag"], "ret_20d": r["ret_20d_lag"],
                "raw_margin": r["raw_margin"], "prob_cal": r["prob_cal"],
                "max_fwd": r["max_fwd21_ret"], "end": r["end_of_window_ret"],
                "y21": r["y21"],
            })
    df = pd.DataFrame(rows)
    print(f"[106] total entries: {len(df)}")

    # Split by 5d_ret at entry (the "partial winner" vs "about to rise" categorization)
    deep_dip = df[df["ret_5d"] < -0.05]
    fresh = df[df["ret_5d"].abs() < 0.05]
    partial = df[(df["ret_5d"] >= 0.05) & (df["ret_5d"] < 0.30)]
    too_hot = df[df["ret_5d"] >= 0.30]

    print(f"\n=== Realized END-OF-WINDOW return (held to day 21) ===")
    rows = []
    for label, sub in [("ALL entries", df),
                        ("DEEP DIP (5d<-5%)", deep_dip),
                        ("ABOUT TO RISE (|5d|<5%)", fresh),
                        ("PARTIAL WINNER (5d 5-30%)", partial),
                        ("TOO HOT (5d>=30%)", too_hot)]:
        rows.append(stats(sub["end"], label))
    df_end = pd.DataFrame(rows)
    print(df_end[["label", "n", "hit_pos", "mean_end", "median_end", "std_end", "sharpe",
                   "big_loss_15pct", "min", "max"]]
          .to_string(index=False, formatters={
              "hit_pos": "{:.0%}".format, "mean_end": "{:+.1%}".format,
              "median_end": "{:+.1%}".format, "std_end": "{:.1%}".format,
              "sharpe": "{:.2f}".format, "big_loss_15pct": "{:.0%}".format,
              "min": "{:+.0%}".format, "max": "{:+.0%}".format,
          }))

    print(f"\n=== Realized PEAK return (best day of 21d window) ===")
    rows = []
    for label, sub in [("ALL entries", df),
                        ("DEEP DIP (5d<-5%)", deep_dip),
                        ("ABOUT TO RISE (|5d|<5%)", fresh),
                        ("PARTIAL WINNER (5d 5-30%)", partial),
                        ("TOO HOT (5d>=30%)", too_hot)]:
        rows.append(stats(sub["max_fwd"], label))
    df_peak = pd.DataFrame(rows)
    print(df_peak[["label", "n", "hit_30pct", "hit_50pct", "hit_100pct",
                    "mean_end", "median_end", "max"]]
          .to_string(index=False, formatters={
              "hit_30pct": "{:.0%}".format, "hit_50pct": "{:.0%}".format,
              "hit_100pct": "{:.0%}".format, "mean_end": "{:+.1%}".format,
              "median_end": "{:+.1%}".format, "max": "{:+.0%}".format,
          }))

    # Cross-tab: 5d category × 20d extension
    print(f"\n=== Cross-tab: 5d category × 20d return bucket ===")
    df["5d_cat"] = pd.cut(df["ret_5d"], [-1, -0.05, 0.05, 0.30, 10],
                            labels=["dip", "fresh", "partial", "too_hot"])
    df["20d_cat"] = pd.cut(df["ret_20d"], [-1, 0.0, 0.30, 0.60, 10],
                             labels=["20d_neg", "20d_<30", "20d_30-60", "20d_>60"])
    grp = df.groupby(["5d_cat", "20d_cat"], observed=True)["end"].agg(["count", "mean", "median"]).reset_index()
    print(grp.to_string(index=False, formatters={"mean": "{:+.1%}".format, "median": "{:+.1%}".format}))

    # If you SKIP partial winners, what happens?
    print(f"\n=== STRATEGY COMPARISON ===")
    strats = {
        "BASELINE (all top-5)": df,
        "SKIP partial winners (about-to-rise + dip + too_hot only)":
            df[~((df["ret_5d"] >= 0.05) & (df["ret_5d"] < 0.30))],
        "ONLY partial winners (5d 5-30%)": partial,
        "ONLY about-to-rise (|5d|<5%)": fresh,
        "ABOUT TO RISE + DIP (avoid all post-5d>5% pops)":
            df[df["ret_5d"] < 0.05],
        "EXTENSION FILTER (20d <50% only)":
            df[df["ret_20d"] < 0.50],
        "EXTENSION FILTER (20d <30% only)":
            df[df["ret_20d"] < 0.30],
    }
    rows = []
    for label, sub in strats.items():
        rows.append(stats(sub["end"], label))
    print(pd.DataFrame(rows)[["label", "n", "hit_pos", "mean_end", "median_end", "sharpe",
                                "big_loss_15pct", "max"]].to_string(index=False, formatters={
        "hit_pos": "{:.0%}".format, "mean_end": "{:+.1%}".format,
        "median_end": "{:+.1%}".format, "sharpe": "{:.2f}".format,
        "big_loss_15pct": "{:.0%}".format, "max": "{:+.0%}".format,
    }))

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "partial_vs_fresh_picks.csv", index=False)
    print(f"\n[106] saved {OUT / 'partial_vs_fresh_picks.csv'}")


if __name__ == "__main__":
    main()
