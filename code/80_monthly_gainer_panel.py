"""Phase A.1: Build monthly gainer panel.

Reads burst_panel_v8.csv, computes 21-trading-day forward 'touch' label
(y21 = 1 if max(close[t+1..t+21]) / close[t] >= 1.30), plus diagnostic
columns (max_fwd21_ret, argmax_day_fwd21, end_of_window_ret, y5_touch).
Merges sector from burst_universe_v7.csv.

Writes data/monthly_gainer_panel.csv.

The v8 features remain causal (overnight_gap excluded by training script,
not here). y21 = -1 for the last 21 rows per ticker.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

WINDOW_21 = 21
WINDOW_5 = 5
TOUCH_THRESH = 0.30


def add_forward_labels(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    c = g["close"].values.astype(float)
    n = len(c)

    y21 = np.full(n, -1, dtype=np.int8)
    max_fwd21_ret = np.full(n, np.nan)
    argmax_day = np.full(n, -1, dtype=np.int16)
    end_ret = np.full(n, np.nan)
    y5 = np.full(n, -1, dtype=np.int8)

    for t in range(n - 1):
        c_t = c[t]
        if c_t <= 0 or not np.isfinite(c_t):
            continue
        # 21d window
        if t + WINDOW_21 < n:
            fut = c[t + 1: t + 1 + WINDOW_21]
            ret = fut / c_t - 1.0
            mx = float(np.max(ret))
            max_fwd21_ret[t] = mx
            argmax_day[t] = int(np.argmax(ret) + 1)
            end_ret[t] = float(c[t + WINDOW_21] / c_t - 1.0)
            y21[t] = 1 if mx >= TOUCH_THRESH else 0
        # 5d window (auxiliary)
        if t + WINDOW_5 < n:
            fut5 = c[t + 1: t + 1 + WINDOW_5]
            mx5 = float(np.max(fut5 / c_t - 1.0))
            y5[t] = 1 if mx5 >= TOUCH_THRESH else 0

    g["y21"] = y21
    g["max_fwd21_ret"] = max_fwd21_ret
    g["argmax_day_fwd21"] = argmax_day
    g["end_of_window_ret"] = end_ret
    g["y5_touch"] = y5
    return g


def main():
    panel = pd.read_csv(DATA / "burst_panel_v8.csv", parse_dates=["date"])
    print(f"[80] loaded {len(panel):,} v8 rows, {panel['ticker'].nunique()} tickers")

    print("[80] computing forward labels per ticker ...")
    out = []
    for tk, g in panel.groupby("ticker", sort=False):
        out.append(add_forward_labels(g))
    panel = pd.concat(out, ignore_index=True)

    universe = pd.read_csv(DATA / "burst_universe_v7.csv")[["ticker", "sector"]]
    panel = panel.merge(universe, on="ticker", how="left")

    out_path = DATA / "monthly_gainer_panel.csv"
    panel.to_csv(out_path, index=False)

    lab = panel[panel["y21"] >= 0]
    print(f"[80] wrote {out_path.name}")
    print(f"[80] labeled rows: {len(lab):,}  positives: {int(lab['y21'].sum()):,}  "
          f"base rate: {lab['y21'].mean():.4%}")
    lab5 = panel[panel["y5_touch"] >= 0]
    print(f"[80] 5d-touch base rate: {lab5['y5_touch'].mean():.4%} "
          f"(positives: {int(lab5['y5_touch'].sum()):,})")
    print(f"[80] sector coverage: {panel['sector'].notna().mean():.1%} of rows")


if __name__ == "__main__":
    main()
