"""Extension-aware re-ranker for today's top picks.

Loads cached fresh features (from script 101), computes extension metrics,
applies a penalty to raw_margin, and re-ranks. Shows side-by-side with the
unfiltered ranking so you can see what gets flagged.

Extension metric:
  ret_20d_lag    : 1-month return (already in panel)
  ret_60d_lag    : 3-month return (computed below)
  high_60d_pct   : current price / max(close[t-60:t]) - 1  (distance from 60d high; <0 = below high)

Penalty:
  - +0% extension: no penalty
  - +30% extension: -0.05 to raw_margin
  - +50% extension: -0.15 to raw_margin
  - +75% extension: -0.30 to raw_margin
  - +100% extension: -0.50 to raw_margin
  Formula: penalty = max(0, (ret_20d_lag - 0.20)) ** 1.5  * 1.5
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


def extension_penalty(ret_20d, ret_60d=None):
    """Penalty grows superlinearly past +20%."""
    excess = max(0.0, ret_20d - 0.20)
    p1 = (excess ** 1.5) * 1.5
    if ret_60d is not None and not np.isnan(ret_60d):
        excess60 = max(0.0, ret_60d - 0.50)
        p2 = (excess60 ** 1.3) * 0.5
        return p1 + p2
    return p1


def main():
    cache_path = DATA / "_today_fresh_features.csv"
    if not cache_path.exists():
        print(f"[105] no cached features at {cache_path}")
        print("[105] run /usr/bin/python3 code/101_refresh_score_today.py fetch first")
        return

    df = pd.read_csv(cache_path, parse_dates=["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Compute 60d return per ticker
    df["close_60d_ago"] = df.groupby("ticker")["close"].shift(60)
    df["ret_60d_lag"] = df["close"] / df["close_60d_ago"] - 1.0

    # Score with v3
    art = joblib.load(MODELS / "monthly_gainer_v3_sp500.joblib")
    feats = art["feats"]
    med = pd.Series(art["impute_medians"])
    cal = art["calibrator"]
    gbc = art["raw_gbc"]

    sub = df.dropna(subset=feats).copy()
    X = sub[feats].fillna(med).values
    sub["prob_cal"] = cal.predict_proba(X)[:, 1]
    sub["raw_margin"] = gbc.decision_function(X)

    last_d = sub["date"].max()
    today = sub[sub["date"] == last_d].copy()

    # Extension penalty
    today["ext_penalty"] = today.apply(
        lambda r: extension_penalty(r["ret_20d_lag"], r.get("ret_60d_lag", np.nan)),
        axis=1
    )
    today["adj_margin"] = today["raw_margin"] - today["ext_penalty"]

    # Top 15 by ORIGINAL raw_margin
    print(f"[105] today: {last_d.date()}")
    print(f"\n=== TOP 15 by ORIGINAL raw_margin (no extension awareness) ===")
    orig15 = today.nlargest(15, "raw_margin")
    print(orig15[["ticker", "sector", "raw_margin", "prob_cal", "close",
                   "ret_5d_lag", "ret_20d_lag", "ret_60d_lag", "ext_penalty", "adj_margin"]]
          .to_string(index=False, formatters={
              "raw_margin": "{:+.2f}".format, "prob_cal": "{:.3f}".format,
              "close": "${:.2f}".format, "ret_5d_lag": "{:+.0%}".format,
              "ret_20d_lag": "{:+.0%}".format, "ret_60d_lag": "{:+.0%}".format,
              "ext_penalty": "{:.2f}".format, "adj_margin": "{:+.2f}".format,
          }))

    # Top 15 by EXTENSION-ADJUSTED margin
    print(f"\n=== TOP 15 by EXTENSION-ADJUSTED margin (penalty applied) ===")
    adj15 = today.nlargest(15, "adj_margin")
    print(adj15[["ticker", "sector", "raw_margin", "ext_penalty", "adj_margin",
                  "prob_cal", "close", "ret_5d_lag", "ret_20d_lag", "ret_60d_lag"]]
          .to_string(index=False, formatters={
              "raw_margin": "{:+.2f}".format, "ext_penalty": "{:.2f}".format,
              "adj_margin": "{:+.2f}".format, "prob_cal": "{:.3f}".format,
              "close": "${:.2f}".format, "ret_5d_lag": "{:+.0%}".format,
              "ret_20d_lag": "{:+.0%}".format, "ret_60d_lag": "{:+.0%}".format,
          }))

    # Holdings comparison
    print(f"\n=== Your holdings (INTC, SMCI, MRNA) ===")
    holdings = today[today["ticker"].isin(["INTC", "SMCI", "MRNA", "SNDK"])]
    print(holdings[["ticker", "raw_margin", "ext_penalty", "adj_margin",
                     "prob_cal", "close", "ret_5d_lag", "ret_20d_lag", "ret_60d_lag"]]
          .to_string(index=False, formatters={
              "raw_margin": "{:+.2f}".format, "ext_penalty": "{:.2f}".format,
              "adj_margin": "{:+.2f}".format, "prob_cal": "{:.3f}".format,
              "close": "${:.2f}".format, "ret_5d_lag": "{:+.0%}".format,
              "ret_20d_lag": "{:+.0%}".format, "ret_60d_lag": "{:+.0%}".format,
          }))

    OUT.mkdir(parents=True, exist_ok=True)
    today.to_csv(OUT / "today_score_extension_aware.csv", index=False)
    print(f"\n[105] saved {OUT / 'today_score_extension_aware.csv'}")


if __name__ == "__main__":
    main()
