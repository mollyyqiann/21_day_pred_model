"""Burst Meta-Model — Day-1 Re-Rank for 3d/5d Hold

WHY THIS EXISTS
---------------
Live grading of v11 burst picks (output/daily_log/*.json) showed:
  - Base v11 fwd_3d/fwd_5d regressors have ~0% MAE skill (no better than predicting 0).
  - But realized day-1 return strongly forecasts day-3/day-5 outcome:
      Spearman(real_1d, real_5d) = +0.32 (p=.0015 over n=97 live picks)
      Picks closing day-1 ≥+2% averaged +4.88% by day-5 (71% win rate, n=45)
      Picks closing day-1 in [-2%, +2%] drift to ~0% by day-5
  - prob_final only sorts in the top quintile (Q5 mean real_5d = +7.98%, Q1-Q4 ≈ 0%)

This script trains a META-regressor that scores yesterday's burst picks at
end-of-day-1 (after the day-1 close is observable) to produce a refined 3d/5d
expected-return — used to decide HOLD-vs-EXIT for the remaining horizon.

TRAINING
--------
Retrospectively score the entire v7 panel with the v11 base classifier +
regressors -> append realized_1d, realized_3d, realized_5d -> attach regime.
Then train HistGradientBoostingRegressor for two targets:
  fwd_3d_from_d1 = (close_t3 - close_t1) / close_t1   (2 more days from day-1 close)
  fwd_5d_from_d1 = (close_t5 - close_t1) / close_t1   (4 more days from day-1 close)

These are *carry-forward* returns from the close of day-1, NOT close-to-close
from day-0. That's the right target for a "should I hold from here?" decision.

INFERENCE (later, in a separate script)
---------------------------------------
End-of-day-1 (~16:05 ET) or next morning pre-open:
  1. Load yesterday's picks from output/daily_log/<yesterday>.json.
  2. Look up day-1 closing price from yfinance / fresh panel.
  3. Score with this meta-model -> expected fwd_3d_from_d1, fwd_5d_from_d1.
  4. Emit hold list (top-K by expected return) + recommended exit picks.

Outputs:
  models/burst_meta_d1_v1_3d.joblib
  models/burst_meta_d1_v1_5d.joblib
  models/burst_meta_d1_v1_meta.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output"
sys.path.insert(0, str(ROOT / "code"))
from regime_features import attach_regime, REGIME_FEATS  # noqa: E402

# Same base features as v11 (matches what the live runner stores per pick)
V7_FEATS = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
            "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap"]

# Meta input features = base classifier output + base regressor outputs +
#                       day-1 realized return + regime + a couple raw technicals.
# We deliberately AVOID intraday features here -- the meta-model fires after
# day-1 close so day-0 intraday is stale.
META_FEATS = [
    # base-model outputs (computed by retro-scoring v11)
    "p_burst_v11",        # raw v11 prob from classifier
    "p_burst_v11_cal",    # isotonic-calibrated v11 prob
    "pred_1d_v11", "pred_3d_v11", "pred_5d_v11",
    # day-1 realized -- THE key new signal
    "real_1d",
    # regime as of day-0 (base_date), forward-fill from regime_frame
    *REGIME_FEATS,
    # cheap raw technicals as redundancy
    "rsi_14", "atr_pct", "vol_z", "rv_60", "overnight_gap",
]


def _load_panel(panel_path: Path) -> pd.DataFrame:
    p = pd.read_csv(panel_path, parse_dates=["date"])
    p = p[p[V7_FEATS].notna().all(axis=1)].copy()
    p = attach_regime(p)
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False)["close"]
    p["close_t1"] = g.shift(-1)
    p["close_t3"] = g.shift(-3)
    p["close_t5"] = g.shift(-5)
    p["real_1d"] = p["close_t1"] / p["close"] - 1
    p["fwd_3d_from_d1"] = p["close_t3"] / p["close_t1"] - 1
    p["fwd_5d_from_d1"] = p["close_t5"] / p["close_t1"] - 1
    return p


def _retro_score(p: pd.DataFrame, uni_tag: str) -> pd.DataFrame:
    """Score every row with v11 classifier + regressors (without intraday — fill NaN)."""
    bundle_cls = joblib.load(MODELS / f"burst_gbc_v11_{uni_tag}.joblib")
    feats_cls = bundle_cls["feats"]
    cls = bundle_cls["gbc"]
    iso = bundle_cls.get("calibrator")

    # Add NaN columns for intraday features that v11 expects but we don't have
    for c in feats_cls:
        if c not in p.columns:
            p[c] = np.nan

    X = p[feats_cls].values
    p["p_burst_v11"] = cls.predict_proba(X)[:, 1]
    p["p_burst_v11_cal"] = iso.transform(p["p_burst_v11"]) if iso is not None else p["p_burst_v11"]

    for h in ("fwd_1d", "fwd_3d", "fwd_5d"):
        bundle_reg = joblib.load(MODELS / f"burst_reg_v11_{uni_tag}_{h}.joblib")
        reg = bundle_reg["reg"]
        feats_reg = bundle_reg["feats"]
        for c in feats_reg:
            if c not in p.columns:
                p[c] = np.nan
        p[f"pred_{h.split('_')[1]}_v11"] = reg.predict(p[feats_reg].values)
    return p


def _train_meta(panel_path: Path, uni_tag: str,
                 select_top_n_per_date: int = 20,
                 select_min_cal_prob: float = 0.10) -> dict:
    """Train meta on a subset that mimics live pick selection.

    Live runner emits top-5 per universe by prob_final (after p_burst_v11_cal gate).
    Training on the full panel washes out the day-1 follow-through signal because
    it only exists in the high-conviction regime. Filter to top-N per date AND
    p_burst_v11_cal >= threshold so the training distribution matches deployment.
    """
    print(f"\n=== meta-d1 training on {panel_path.name} (uni={uni_tag}) ===")
    print(f"  selection filter: top {select_top_n_per_date}/date AND cal_prob >= {select_min_cal_prob}")
    t0 = time.time()
    p = _load_panel(panel_path)
    p = _retro_score(p, uni_tag)

    needed_for_features = [c for c in META_FEATS if c in p.columns]
    p = p.dropna(subset=["real_1d", "p_burst_v11", "pred_3d_v11", "pred_5d_v11"]).reset_index(drop=True)
    n_full = len(p)

    # Subset: per-date top-N by p_burst_v11_cal, and cal_prob >= floor.
    # This is the universe of rows the live runner could plausibly have surfaced.
    p["rank_in_date"] = p.groupby("date")["p_burst_v11_cal"].rank(method="first", ascending=False)
    p = p[(p["rank_in_date"] <= select_top_n_per_date) &
          (p["p_burst_v11_cal"] >= select_min_cal_prob)].copy()
    p = p.drop(columns=["rank_in_date"])
    print(f"  rows after candidate filter: {len(p):,} (from {n_full:,})")

    dates = np.sort(p["date"].unique())
    d1 = dates[int(0.70 * len(dates))]
    d2 = dates[int(0.85 * len(dates))]
    tr = p[p["date"] <  d1]
    va = p[(p["date"] >= d1) & (p["date"] < d2)]
    te = p[p["date"] >= d2]
    print(f"  train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    metrics = {"universe": uni_tag, "panel": panel_path.name}
    for tgt in ("fwd_3d_from_d1", "fwd_5d_from_d1"):
        # only rows where target is known
        tr_t = tr.dropna(subset=[tgt])
        va_t = va.dropna(subset=[tgt])
        te_t = te.dropna(subset=[tgt])
        Xtr = tr_t[needed_for_features].values
        ytr = tr_t[tgt].values
        Xte = te_t[needed_for_features].values
        yte = te_t[tgt].values

        # Light grid: sweep lr / depth with val-set MAE
        best = None
        for lr in (0.03, 0.05, 0.08):
            for md in (3, 5):
                for mi in (300, 500):
                    reg = HistGradientBoostingRegressor(
                        max_iter=mi, max_depth=md,
                        learning_rate=lr, random_state=42,
                        l2_regularization=1.0)
                    reg.fit(Xtr, ytr)
                    pv = reg.predict(va_t[needed_for_features].values)
                    mae_v = mean_absolute_error(va_t[tgt].values, pv)
                    if best is None or mae_v < best[0]:
                        best = (mae_v, reg, (lr, md, mi))
        lr, md, mi = best[2]
        reg = best[1]

        pte = reg.predict(Xte)
        mae = float(mean_absolute_error(yte, pte))
        mae_zero = float(np.mean(np.abs(yte)))
        dir_hit = float(np.mean(np.sign(pte) == np.sign(yte)))

        # Compare to base v11 regressor on same test slice for the closest horizon.
        # Note this isn't a perfect apples-to-apples compare (v11 predicts close-to-close
        # from day-0; meta predicts close-from-day-1) -- but lets us see the lift.
        base_h = "pred_3d_v11" if tgt == "fwd_3d_from_d1" else "pred_5d_v11"
        # rescale base prediction to the from-day-1 frame:
        # base predicts (close_tN - close_t0)/close_t0; we want (close_tN - close_t1)/close_t1.
        # Approx: (1 + base) / (1 + real_1d) - 1
        base_pred_from_d1 = (1 + te_t[base_h].values) / (1 + te_t["real_1d"].values) - 1
        mae_base = float(mean_absolute_error(yte, base_pred_from_d1))
        dir_base = float(np.mean(np.sign(base_pred_from_d1) == np.sign(yte)))

        # Spearman lift
        rho_meta, _ = spearmanr(pte, yte)
        rho_base, _ = spearmanr(base_pred_from_d1, yte)

        print(f"  {tgt}: best lr={lr} md={md} mi={mi}  "
              f"MAE={mae:.4f} (vs zero {mae_zero:.4f}, vs base_v11 {mae_base:.4f})  "
              f"dir={dir_hit:.3f} (base {dir_base:.3f})  "
              f"rho={rho_meta:+.3f} (base {rho_base:+.3f})")

        path = MODELS / f"burst_meta_d1_v1_{uni_tag}_{tgt.split('_')[1]}.joblib"
        joblib.dump({
            "reg": reg, "feats": needed_for_features,
            "version": "meta_d1_v1", "horizon": tgt, "universe": uni_tag,
            "metrics": {
                "n_test": int(len(te_t)),
                "MAE": mae, "MAE_vs_zero": mae_zero, "MAE_vs_base_v11": mae_base,
                "dir_hit": dir_hit, "dir_hit_base_v11": dir_base,
                "spearman": float(rho_meta), "spearman_base_v11": float(rho_base),
                "skill_vs_zero": 1 - mae / mae_zero if mae_zero > 0 else float("nan"),
                "skill_vs_base": 1 - mae / mae_base if mae_base > 0 else float("nan"),
            },
            "best_params": {"learning_rate": lr, "max_depth": md, "max_iter": mi},
        }, path)
        print(f"    wrote {path.name}")

        # Decile lift on test set
        if len(te_t) > 50:
            te2 = te_t.copy()
            te2["pred"] = pte
            te2["q"] = pd.qcut(te2["pred"].rank(method="first"), 5,
                              labels=["Q1","Q2","Q3","Q4","Q5"])
            print(f"    {tgt} quintiles of meta-pred -> mean realized:")
            agg = te2.groupby("q")[tgt].agg(["count","mean"])
            for q,row in agg.iterrows():
                print(f"      {q}: n={int(row['count']):>5d}  mean={row['mean']*100:+.2f}%")

        metrics[tgt] = {
            "MAE": mae, "MAE_vs_zero": mae_zero, "MAE_vs_base_v11": mae_base,
            "dir_hit": dir_hit, "dir_hit_base_v11": dir_base,
            "spearman": float(rho_meta), "spearman_base_v11": float(rho_base),
            "best_params": {"learning_rate": lr, "max_depth": md, "max_iter": mi},
            "n_test": int(len(te_t)),
        }

    metrics["runtime_sec"] = round(time.time() - t0, 1)
    return metrics


def main():
    panels = [
        (DATA / "burst_panel_v6b.csv", "v4"),
        (DATA / "burst_panel_v6.csv",  "v5"),
        (DATA / "burst_panel_v7.csv",  "v7"),
    ]
    out = {"version": "meta_d1_v1", "feats": META_FEATS, "per_universe": {}}
    for path, uni in panels:
        if not path.exists():
            print(f"  skip {path.name} (missing)")
            continue
        out["per_universe"][uni] = _train_meta(path, uni)
    (MODELS / "burst_meta_d1_v1_meta.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {MODELS/'burst_meta_d1_v1_meta.json'}")


if __name__ == "__main__":
    main()
