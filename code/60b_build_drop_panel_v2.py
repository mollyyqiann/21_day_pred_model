"""Assemble data/drop_panel_v2.csv.

Starts from drop_panel_v1.csv (existing features + y_drop label) and adds:
  1. Cross-sectional rank features  (percentile rank within each date)
  2. Regime/macro/cross-asset columns (same value for all tickers on a date)
  3. Multi-horizon labels y_drop_1d / y_drop_3d / y_drop_5d (cumulative-return based)

EDGAR / FinBERT news features are merged in later by a separate step so that
this script can run independently (and quickly) of the EDGAR backfill.

Output: data/drop_panel_v2.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_regime import build_regime_panel  # noqa: E402

V1_PATH = ROOT / "data" / "drop_panel_v1.csv"
OUT_PATH = ROOT / "data" / "drop_panel_v2.csv"

# Features that should get a per-date cross-sectional rank version.
# Skip discrete/degenerate ones (ma_stack is a small-int stack score; run_length
# and up_streak are already rank-like but we add % versions anyway).
RANK_FEATURES = [
    "rsi_14", "macd", "macd_hist", "bb_z20",
    "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60",
    "overnight_gap",
    "up_bigdays_20d", "dist_ma60_atr", "ma60_slope_60d", "run_length",
    "up_streak",
]


def _cum_ret_n(closes: pd.Series, n: int) -> pd.Series:
    """Return cumulative return over next n business days: close[t+n]/close[t] - 1."""
    return closes.shift(-n) / closes - 1.0


def build_multi_horizon_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add y_drop_1d / y_drop_3d / y_drop_5d based on forward cumulative return."""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker", sort=False)["close"]

    # Thresholds chosen so each horizon has roughly 5-10% positive rate,
    # in the same ballpark as the v1 5.87% base rate, but capture different
    # drop flavors (flash -3% day vs slow -7% week).
    fwd_1d = g.transform(lambda s: _cum_ret_n(s, 1))
    fwd_3d = g.transform(lambda s: _cum_ret_n(s, 3))
    fwd_5d = g.transform(lambda s: _cum_ret_n(s, 5))

    df["y_drop_1d"] = (fwd_1d <= -0.03).astype("Int8")
    df["y_drop_3d"] = (fwd_3d <= -0.05).astype("Int8")
    df["y_drop_5d"] = (fwd_5d <= -0.07).astype("Int8")
    # Mark as missing where horizon is out of sample
    df.loc[fwd_1d.isna(), "y_drop_1d"] = pd.NA
    df.loc[fwd_3d.isna(), "y_drop_3d"] = pd.NA
    df.loc[fwd_5d.isna(), "y_drop_5d"] = pd.NA
    return df


def build_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """For each RANK_FEATURE f, add f_rank = percentile_rank(f) within same date."""
    for f in RANK_FEATURES:
        if f not in df.columns:
            print(f"[rank] skipping missing feature: {f}")
            continue
        # pct=True gives percentile in [0, 1]. method='average' handles ties.
        df[f"{f}_rank"] = df.groupby("date")[f].rank(pct=True, method="average")
    return df


def main() -> None:
    print(f"[panel-v2] loading {V1_PATH} ...")
    df = pd.read_csv(V1_PATH, parse_dates=["date"])
    print(f"  shape: {df.shape}")
    print(f"  date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"  y_drop pos rate (v1 target): {(df['y_drop'] == 1).mean():.4f}")

    # 1. Multi-horizon labels
    print("[panel-v2] building multi-horizon labels ...")
    df = build_multi_horizon_labels(df)
    for c in ["y_drop_1d", "y_drop_3d", "y_drop_5d"]:
        pos = (df[c] == 1).sum()
        n = df[c].notna().sum()
        print(f"  {c}: pos={pos} / {n} = {pos / max(n, 1):.4f}")

    # 2. Cross-sectional ranks
    print("[panel-v2] building cross-sectional ranks ...")
    df = build_cross_sectional_ranks(df)

    # 3. Regime panel merge
    print("[panel-v2] loading regime panel ...")
    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")
    regime = build_regime_panel(start, end)
    print(f"  regime shape: {regime.shape}")
    before = len(df)
    df = df.merge(regime, on="date", how="left")
    after = len(df)
    print(f"  rows before: {before}, after: {after} (should match)")

    # Sanity: fraction of rows with at least one regime col NaN
    regime_cols = [c for c in regime.columns if c != "date"]
    n_nan_regime = df[regime_cols].isna().any(axis=1).sum()
    print(f"  rows with any NaN regime col: {n_nan_regime} / {len(df)} "
          f"({n_nan_regime / len(df):.3f})")

    # Save
    df.to_csv(OUT_PATH, index=False)
    print(f"[panel-v2] wrote {OUT_PATH}  rows={len(df)}  cols={len(df.columns)}")


if __name__ == "__main__":
    main()
