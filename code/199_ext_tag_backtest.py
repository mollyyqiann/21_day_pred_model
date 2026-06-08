"""Backtest the MG v3 model's target-hit rate by extension tag.

Re-run periodically (e.g. monthly) to refresh the TAG_HITRATE_TOP5 / TAG_AVG21D_TOP5
constants in 102_sunday_check.py.

Output: output/monthly_gainer/ext_tag_hitrate_top{5,15}.csv with per-tag stats.

Major finding (as of run 2026-06-01, 3yr panel, ~3,060 top-5 picks): EXTREME is
the BEST-performing tag, contradicting the "extension penalty" intuition. The
production scoring file `today_score_extension_aware.csv` applies a penalty that
DESTROYS alpha for this universe. Use raw_margin ranking instead.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from extension_classifier import attach_extension

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "monthly_gainer"


def add_lags(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    for n in (5, 20, 60, 180):
        g[f"ret_{n}d_lag"] = g["close"].pct_change(n)
    for n in (5, 20, 60):
        g[f"close_{n}d_ago"] = g["close"].shift(n)
    return g


def main():
    panel = pd.read_csv(ROOT / "data" / "monthly_gainer_panel.csv", parse_dates=["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel = panel.groupby("ticker", group_keys=False).apply(add_lags)
    for col, src in [("rsi_14_xrank", "rsi_14"), ("rv_60_xrank", "rv_60"),
                      ("ma60_slope_xrank", "ma60_slope_60d"), ("ret_20d_xrank", "ret_20d_lag")]:
        panel[col] = panel.groupby("date")[src].rank(pct=True)

    art = joblib.load(ROOT / "models" / "monthly_gainer_v3_sp500.joblib")
    feats = art["feats"]
    med = pd.Series(art["impute_medians"])
    gbc = art["raw_gbc"]
    for f in feats:
        if f not in panel.columns:
            panel[f] = med.get(f, 0)

    df = panel.dropna(subset=["rsi_14", "macd", "rv_60", "ma60_slope_60d",
                               "ret_20d_lag", "atr_pct", "y21"]).copy()
    df = df[df["y21"] != -1]
    df["raw_margin"] = gbc.decision_function(df[feats].fillna(med).values)

    top15 = df.groupby("date", group_keys=False).apply(lambda g: g.nlargest(15, "raw_margin"))
    top5 = df.groupby("date", group_keys=False).apply(lambda g: g.nlargest(5, "raw_margin"))
    top15 = attach_extension(top15)
    top5 = attach_extension(top5)

    baseline_hit = df["y21"].mean()
    baseline_peak = df["max_fwd21_ret"].mean()
    print(f"UNIVERSE  hit_rate {baseline_hit:.1%}  avg_peak +{baseline_peak:.1%}  rows {len(df):,}")
    print()

    def summarize(picks, label):
        s = picks.groupby("ext_level").agg(
            n=("ticker", "count"),
            hit_rate=("y21", "mean"),
            avg_peak=("max_fwd21_ret", "mean"),
            avg_end=("end_of_window_ret", "mean"),
        ).sort_values("hit_rate", ascending=False)
        s["edge"] = s["hit_rate"] - baseline_hit
        print(f"=== {label} ({len(picks):,} picks) ===")
        print(f"{'tag':<10}{'n':>7}{'%mix':>7}{'hit':>8}{'peak':>9}{'end':>9}{'edge':>9}")
        print("-" * 60)
        total = s["n"].sum()
        for tag, r in s.iterrows():
            print(f"{tag:<10}{int(r['n']):>7}{r['n']/total:>6.0%}{r['hit_rate']:>8.1%}"
                  f"{r['avg_peak']:>+9.1%}{r['avg_end']:>+9.1%}{r['edge']:>+9.1%}")
        print()
        return s

    s15 = summarize(top15, "TOP-15")
    s5 = summarize(top5, "TOP-5")

    s15.to_csv(OUT / "ext_tag_hitrate_top15.csv")
    s5.to_csv(OUT / "ext_tag_hitrate_top5.csv")
    print(f"[saved] {OUT}/ext_tag_hitrate_top{{5,15}}.csv")


if __name__ == "__main__":
    main()
