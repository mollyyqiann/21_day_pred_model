"""Drop-prediction model — mirror of the burst model but targeting downside.

Target: in the next 5 trading days, does there exist a contiguous 2-5 day
window whose average daily return is <= -3%/day? (A softer threshold than the
+4% burst because downside realizations cluster differently — we want to surface
REAL risk, not just any down-day.)

Universe: v7 (full S&P 500, same as burst side) so it's a direct mirror.

Features: same as v8 (17 features). The drop and burst targets are not mutually
exclusive (a stock can have both within 5 days), so the models can legitimately
disagree — that's useful.

Output:
  data/drop_panel_v1.csv   (reuses v7 panel with y_drop column)
  models/drop_gbc_v1.joblib
  models/drop_reg_v1_{1d,3d,5d}.joblib
  output/drop_metrics_v1.json
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score, log_loss, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"; OUT = ROOT / "output"

DROP_WINDOW = 5; DROP_MIN_LEN = 2; DROP_THRESH = -0.03   # -3%/day avg

A_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
          "atr_pct", "range_pct", "vol_z", "vol_5d"]
V7_FEATS = A_BASE + ["rv_60", "overnight_gap"]
TREND = ["ma_stack", "up_streak", "up_bigdays_20d",
         "dist_ma60_atr", "ma60_slope_60d", "run_length"]
FEATS = V7_FEATS + TREND   # 17


def add_drop_target(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    c = g["close"]
    r = c.pct_change().fillna(0).values
    n = len(r); y = np.zeros(n, dtype=np.int8)
    for t in range(n - DROP_WINDOW):
        fut = r[t+1:t+1+DROP_WINDOW]; worst = 0.0
        for L in range(DROP_MIN_LEN, DROP_WINDOW + 1):
            for s in range(0, DROP_WINDOW - L + 1):
                m = fut[s:s+L].mean()
                if m < worst: worst = m
        if worst <= DROP_THRESH: y[t] = 1
    y[-DROP_WINDOW:] = -1
    g["y_drop"] = y
    return g


def evaluate(y, p):
    base = float(np.mean(y)) if len(y) else float("nan")
    try: auc = roc_auc_score(y, p)
    except ValueError: auc = float("nan")
    try: ap = average_precision_score(y, p)
    except ValueError: ap = float("nan")
    ll = log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])
    return {"n": int(len(y)), "pos": int(y.sum()), "base": base,
            "auc": float(auc), "ap": float(ap),
            "ap_lift": float(ap/base) if base > 0 else float("nan"),
            "log_loss": float(ll)}


def main():
    panel = pd.read_csv(DATA / "burst_panel_v8.csv", parse_dates=["date"])
    print(f"[drop] loaded {len(panel):,} rows from burst_panel_v8.csv")

    # add drop target per ticker
    out = []
    for t, g in panel.groupby("ticker", sort=False):
        out.append(add_drop_target(g))
    panel = pd.concat(out, ignore_index=True)
    panel.to_csv(DATA / "drop_panel_v1.csv", index=False)
    print(f"[drop] base rate: {panel[panel['y_drop']>=0]['y_drop'].mean():.4%}")

    lab = panel[panel["y_drop"] >= 0].dropna(subset=FEATS).reset_index(drop=True)
    dates = np.sort(lab["date"].unique())
    d1 = dates[int(0.70 * len(dates))]; d2 = dates[int(0.85 * len(dates))]
    tr = lab[lab["date"] < d1]; va = lab[(lab["date"] >= d1) & (lab["date"] < d2)]; te = lab[lab["date"] >= d2]
    print(f"[drop] train/val/test: {len(tr):,}/{len(va):,}/{len(te):,}")

    gbc = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
    gbc.fit(tr[FEATS].values, tr["y_drop"].values)
    p_te = gbc.predict_proba(te[FEATS].values)[:, 1]
    met = evaluate(te["y_drop"].values, p_te)
    print(f"[drop] test AUC={met['auc']:.3f}  AP={met['ap']:.3f}  "
          f"lift={met['ap_lift']:.2f}x  log_loss={met['log_loss']:.4f}")

    joblib.dump({"gbc": gbc, "feats": FEATS}, MODELS / "drop_gbc_v1.joblib")
    imp = pd.Series(gbc.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("\n[drop] feature importance:")
    print(imp.to_string())

    # regression heads for expected 1d/3d/5d returns (downside same regressors as upside)
    p = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False)["close"]
    p["fwd_1d"] = g.shift(-1) / p["close"] - 1
    p["fwd_3d"] = g.shift(-3) / p["close"] - 1
    p["fwd_5d"] = g.shift(-5) / p["close"] - 1
    pr = p.dropna(subset=FEATS + ["fwd_1d", "fwd_3d", "fwd_5d"]).reset_index(drop=True)
    rdates = np.sort(pr["date"].unique())
    rd1 = rdates[int(0.70 * len(rdates))]; rd2 = rdates[int(0.85 * len(rdates))]
    rtr = pr[pr["date"] < rd1]; rte = pr[pr["date"] >= rd2]
    reg_metrics = {}
    for h in ["fwd_1d", "fwd_3d", "fwd_5d"]:
        reg = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                        learning_rate=0.05, subsample=0.8, random_state=42)
        reg.fit(rtr[FEATS].values, rtr[h].values)
        pte = reg.predict(rte[FEATS].values)
        mae = float(np.mean(np.abs(rte[h].values - pte)))
        dir_acc = float(np.mean(np.sign(rte[h].values) == np.sign(pte)))
        reg_metrics[h] = {"mae": mae, "dir": dir_acc}
        joblib.dump({"reg": reg, "feats": FEATS}, MODELS / f"drop_reg_v1_{h}.joblib")
        print(f"   {h}: MAE={mae:.4f}  dir={dir_acc:.3f}")

    (OUT / "drop_metrics_v1.json").write_text(json.dumps({
        "classifier": met, "reg": reg_metrics, "features": FEATS,
        "target": {"window": DROP_WINDOW, "min_len": DROP_MIN_LEN, "thresh": DROP_THRESH},
    }, indent=2))
    print("[drop] done")


if __name__ == "__main__":
    main()
