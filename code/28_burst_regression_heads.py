"""Train 1-day / 3-day / 5-day forward-return regression heads on top of the
v6/v6b panels, so daily predictions can report expected % change at three
horizons (not just a burst probability).

For each of the two universes (v6 -> upside-asymmetric 213, v6b -> >$40 450):
  - Compute fwd_1d, fwd_3d, fwd_5d cumulative return per row
  - Fit GradientBoostingRegressor(n_est=300, depth=3, lr=0.05) on the same
    features as the augmented classifier (includes overnight_gap)
  - Report test MAE and direction accuracy (sign hit rate)

Outputs:
  models/burst_reg_v6_{1d,3d,5d}.joblib
  models/burst_reg_v6b_{1d,3d,5d}.joblib
  output/burst_reg_metrics.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"; OUT = ROOT / "output"

A_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
          "atr_pct", "range_pct", "vol_z", "vol_5d"]
V4_FEATS = A_BASE + ["rv_60"]                       # 10
V5_FEATS = V4_FEATS + ["skew_60d", "semivol_ratio_60d", "up_bigdays_60d"]  # 13
V6_FEATS  = V5_FEATS + ["overnight_gap"]            # 14
V6B_FEATS = V4_FEATS + ["overnight_gap"]            # 11


def add_fwd_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = panel.groupby("ticker", sort=False)["close"]
    panel["fwd_1d"] = g.shift(-1) / panel["close"] - 1
    panel["fwd_3d"] = g.shift(-3) / panel["close"] - 1
    panel["fwd_5d"] = g.shift(-5) / panel["close"] - 1
    return panel


def train_one(panel: pd.DataFrame, feats: list[str], tag: str) -> dict:
    p = add_fwd_returns(panel)
    p = p.dropna(subset=feats + ["fwd_1d", "fwd_3d", "fwd_5d"]).reset_index(drop=True)
    dates = np.sort(p["date"].unique())
    d1 = dates[int(0.70 * len(dates))]; d2 = dates[int(0.85 * len(dates))]
    tr = p[p["date"] < d1]; va = p[(p["date"] >= d1) & (p["date"] < d2)]; te = p[p["date"] >= d2]

    out = {}
    for h in ["fwd_1d", "fwd_3d", "fwd_5d"]:
        reg = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=42)
        reg.fit(tr[feats].values, tr[h].values)
        p_te = reg.predict(te[feats].values)
        mae = float(np.mean(np.abs(te[h].values - p_te)))
        naive = float(np.mean(np.abs(te[h].values)))   # always predict 0
        dir_acc = float(np.mean(np.sign(te[h].values) == np.sign(p_te)))
        out[h] = {"mae": mae, "mae_naive_zero": naive,
                  "mae_skill_vs_naive": 1 - mae / naive if naive > 0 else float("nan"),
                  "direction_acc": dir_acc,
                  "n_test": int(len(te))}
        joblib.dump({"reg": reg, "feats": feats},
                    MODELS / f"burst_reg_{tag}_{h}.joblib")
        print(f"[{tag}:{h}]  n_test={len(te):,}  MAE={mae:.4f}  "
              f"vs naive {naive:.4f}  dir_acc={dir_acc:.3f}")
    return out


def main():
    print("[reg] v6 (upside-asymmetric, v5 universe) ...")
    panel_v6 = pd.read_csv(DATA / "burst_panel_v6.csv", parse_dates=["date"])
    m6 = train_one(panel_v6, V6_FEATS, "v6")

    print("\n[reg] v6b (broad >$40) ...")
    panel_v6b = pd.read_csv(DATA / "burst_panel_v6b.csv", parse_dates=["date"])
    m6b = train_one(panel_v6b, V6B_FEATS, "v6b")

    (OUT / "burst_reg_metrics.json").write_text(
        json.dumps({"v6": m6, "v6b": m6b}, indent=2))
    print(f"\n[reg] wrote {OUT/'burst_reg_metrics.json'}")


if __name__ == "__main__":
    main()
