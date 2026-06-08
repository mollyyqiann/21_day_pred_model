"""Finish the training that 99_rebuild couldn't (GBR on 1.38M rows was too slow).

Order:
  1. v7 regressors (fwd_1d/3d/5d) — swap from GradientBoostingRegressor to
     HistGradientBoostingRegressor. Same feature set, ~20x faster on 1.38M rows,
     comparable MAE in our testing. Saves burst_reg_v10_v7_fwd_{1d,3d,5d}.joblib.
  2. burst_v10_meta.json — record the retrained v7 regressor metrics alongside
     the existing v4/v5 metrics (kept from 60_burst_v10_pipeline.py's output).
  3. drop_panel_v1.csv — rebuild from fresh burst_panel_v8.csv (adds y_drop).
  4. drop_panel_v2.csv — rebuild (60b) from fresh drop_panel_v1.csv + regime.
  5. EDGAR features merged into drop_panel_v2 via 60d (optional, will skip if
     the filings_scored.csv is missing or stale).
  6. drop_gbc_v2_raw.joblib + drop_gbc_v2_calibrated.joblib — retrain via 36.
  7. drop_reg_v2_fwd_{1d,3d,5d}.joblib — retrain via 36c on fresh v1.

Logs: output/rebuild_Cfix.log
"""
from __future__ import annotations

import sys; sys.stdout.reconfigure(line_buffering=True)

import importlib.util
import json
import subprocess
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CODE = ROOT / "code"
MODELS = ROOT / "models"
OUT = ROOT / "output"


def _load_regime():
    sys.path.insert(0, str(CODE))
    from regime_features import load_regime_frame, attach_regime, REGIME_FEATS
    return attach_regime, REGIME_FEATS


def _v10_feats(uni_tag: str, REGIME_FEATS) -> list[str]:
    # same feature set per universe as 60_burst_v10_pipeline.py
    base = {
        "v4": ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
               "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap"],
        "v5": ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
               "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap",
               "skew_60d", "semivol_ratio_60d", "up_bigdays_60d"],
        "v7": ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
               "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap"],
    }
    return base[uni_tag] + list(REGIME_FEATS)


def retrain_v7_regressors():
    print("\n=== v7 regressors (HGBR) ===")
    attach_regime, REGIME_FEATS = _load_regime()
    feats = _v10_feats("v7", REGIME_FEATS)
    base_feats = [f for f in feats if f not in REGIME_FEATS]

    print("[v7reg] loading burst_panel_v7.csv …")
    p = pd.read_csv(DATA / "burst_panel_v7.csv", parse_dates=["date"])
    print(f"  rows: {len(p):,}  tickers: {p['ticker'].nunique()}")
    p = p[p[base_feats].notna().all(axis=1)].copy()
    p = attach_regime(p)
    # fwd return labels
    p["fwd_1d"] = p.groupby("ticker")["close"].shift(-1) / p["close"] - 1
    p["fwd_3d"] = p.groupby("ticker")["close"].shift(-3) / p["close"] - 1
    p["fwd_5d"] = p.groupby("ticker")["close"].shift(-5) / p["close"] - 1

    lab = p.dropna(subset=feats + ["fwd_1d", "fwd_3d", "fwd_5d"]).reset_index(drop=True)
    dates = np.sort(lab["date"].unique())
    d1 = dates[int(0.70 * len(dates))]
    d2 = dates[int(0.85 * len(dates))]
    tr = lab[lab["date"] <  d1]
    te = lab[lab["date"] >= d2]
    print(f"  train={len(tr):,}  test={len(te):,}")

    reg_metrics = {}
    for h in ["fwd_1d", "fwd_3d", "fwd_5d"]:
        t0 = time.time()
        reg = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=300, max_depth=5, learning_rate=0.05,
            l2_regularization=0.0, random_state=42,
        )
        reg.fit(tr[feats].values, tr[h].values)
        pred = reg.predict(te[feats].values)
        mae = float(mean_absolute_error(te[h].values, pred))
        mae_zero = float(np.mean(np.abs(te[h].values)))
        dir_hit = float(np.mean(np.sign(pred) == np.sign(te[h].values)))
        reg_metrics[h] = {"n_test": int(len(te)), "MAE": mae,
                           "MAE_vs_zero": mae_zero, "dir_hit": dir_hit}
        path = MODELS / f"burst_reg_v10_v7_{h}.joblib"
        joblib.dump({"reg": reg, "feats": feats, "version": "v10_hgbr",
                     "horizon": h, "universe": "v7",
                     "metrics": reg_metrics[h]}, path)
        dt = time.time() - t0
        print(f"  {h}: MAE={mae:.4f}  vs-zero {mae_zero:.4f}  dir={dir_hit:.3f}  "
              f"-> {path.name}  ({dt:.1f}s)")
    return reg_metrics


def update_meta(v7_reg_metrics):
    print("\n=== updating burst_v10_meta.json ===")
    meta_path = MODELS / "burst_v10_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"version": "v10"}
    uni = meta.setdefault("universes", {}).setdefault("v7", {})
    uni["regressor_algo"] = "HistGradientBoostingRegressor (absolute_error, max_iter=300, max_depth=5, lr=0.05)"
    uni["regressor_metrics"] = v7_reg_metrics
    uni["retrained_at"] = pd.Timestamp.now().isoformat()
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"  wrote {meta_path}")


def _run(cmd, label):
    print(f"\n[{label}] {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    print(f"[{label}] rc={rc} elapsed {time.time()-t0:.1f}s")
    return rc


def rebuild_drop_panels():
    """Add y_drop to fresh v8 → drop_panel_v1.csv. Then 60b builds v2."""
    print("\n=== drop panels ===")
    DROP_WINDOW, DROP_MIN_LEN, DROP_THRESH = 5, 2, -0.03

    def add_drop_target(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").reset_index(drop=True)
        r = g["close"].pct_change().fillna(0).values
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

    print("[drop-panels] loading burst_panel_v8.csv …")
    panel = pd.read_csv(DATA / "burst_panel_v8.csv", parse_dates=["date"])
    print(f"  {len(panel):,} rows, {panel['ticker'].nunique()} tickers")
    out = []
    for _, g in panel.groupby("ticker", sort=False):
        out.append(add_drop_target(g))
    v1 = pd.concat(out, ignore_index=True)
    v1.to_csv(DATA / "drop_panel_v1.csv", index=False)
    print(f"  wrote drop_panel_v1.csv; base rate: "
          f"{v1[v1['y_drop']>=0]['y_drop'].mean():.4%}")

    # 60b builds drop_panel_v2 from v1
    _run([sys.executable, "-u", str(CODE / "60b_build_drop_panel_v2.py")],
         "60b (drop_panel_v2)")

    # 60d merges EDGAR features into v2 (optional — skip if data missing)
    edgar_path = DATA / "edgar_backfill" / "filings_scored.csv"
    if edgar_path.exists():
        _run([sys.executable, "-u", str(CODE / "60d_edgar_daily_features.py")],
             "60d (EDGAR features)")
    else:
        print("[60d] filings_scored.csv missing — skipping EDGAR merge")


def retrain_drop_v2():
    print("\n=== drop v2 classifier + regressors ===")
    _run([sys.executable, "-u", str(CODE / "36_drop_v2_pipeline.py")],
         "drop v2 classifier")
    _run([sys.executable, "-u", str(CODE / "36c_drop_v2_final.py")],
         "drop v2 regressors (final)")


def main():
    t0 = time.time()
    v7_metrics = retrain_v7_regressors()
    update_meta(v7_metrics)
    rebuild_drop_panels()
    retrain_drop_v2()
    print(f"\n[fast-retrain] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
