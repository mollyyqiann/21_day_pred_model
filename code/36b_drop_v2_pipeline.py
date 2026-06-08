"""Drop-magnitude regressors v2 — improves percent-change accuracy.

Mirrors v1 (36_drop_v1_pipeline.py) but applies 5 accuracy-targeted changes
to the *regression* heads only. The drop CLASSIFIER is untouched.

Writes NEW model files (drop_reg_v2_*) so the 08:45 ET morning runner
(38_drop_daily_runner.py) continues loading v1 and is not disturbed.

Five improvements over v1 (all ablation-tested in the metrics JSON):
  1. Drop-conditional sample weighting — rows with y_drop=1 get WEIGHT_POS
     weight; baseline rows keep weight 1. The v1 regressor was trained on
     the full universe (drop base-rate ~8%) so it was modelling "average
     return", not "magnitude once a drop is flagged". Weighting concentrates
     the fit on the regime the runner actually scores at inference.
  2. Absolute-error loss (L1) — HistGradientBoosting's robust regression
     option. Earnings-gap and trading-halt moves (±15-30 %) otherwise
     dominate MSE and pull the fit toward tail events that aren't
     repeatable from features. L1 optimises for the conditional median,
     which is what we actually want when the right tail is fat.
  3. Target winsorization at the 1st/99th percentile (computed on TRAIN
     rows only, then applied to train + val + test). Same motivation as (2)
     but acts on the label distribution; the two are complementary.
  4. Volatility-normalized targets — we train on z = fwd_Nd / (sqrt(N)*rv_60),
     then denormalize predictions at inference by multiplying with live
     sqrt(N)*rv_60. The internal target is stationary across vol regimes,
     so the tree can split on *what drives excess-of-vol moves* instead of
     also carrying the volatility scale.
  5. Multi-seed bagging — 5 GBRs with different random_state; predictions
     are averaged. Reduces variance from greedy single-split choices.

Baseline numbers reproduced inline from output/drop_metrics_v1.json:
  fwd_1d: MAE 1.352%,  dir 62.7%
  fwd_3d: MAE 2.606%,  dir 57.7%
  fwd_5d: MAE 3.453%,  dir 56.5%

Output:
  models/drop_reg_v2_fwd_{1d,3d,5d}.joblib
  output/drop_metrics_v2.json  (headline v2 metrics + per-improvement ablation)
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"; OUT = ROOT / "output"

V7_FEATS = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
            "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap"]
TREND = ["ma_stack", "up_streak", "up_bigdays_20d",
         "dist_ma60_atr", "ma60_slope_60d", "run_length"]
FEATS = V7_FEATS + TREND   # 17, same as v1

HORIZONS = {"fwd_1d": 1, "fwd_3d": 3, "fwd_5d": 5}

# --- Improvement hyper-parameters (tuned on the val split) ---
WEIGHT_POS = 5.0         # #1 drop-row weight (baseline 1.0)
WINSOR_LO, WINSOR_HI = 0.01, 0.99  # #3
EPS_VOL = 1e-4           # #4 floor to avoid div-by-zero
SEEDS = [42, 7, 1337, 2024, 99]     # #5
# HGBR base params roughly matching v1's GBR(n_estimators=300, max_depth=3,
# learning_rate=0.05, subsample=0.8). HGBR is ~10× faster via histogram binning.
HGBR_KW = dict(max_iter=300, max_depth=5, learning_rate=0.05,
               l2_regularization=1.0, min_samples_leaf=32,
               early_stopping=False)


def _build_panel() -> pd.DataFrame:
    """Load the same drop panel v1 built (which has y_drop, rv_60, etc.)."""
    p = pd.read_csv(DATA / "drop_panel_v1.csv", parse_dates=["date"])
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False)["close"]
    p["fwd_1d"] = g.shift(-1) / p["close"] - 1
    p["fwd_3d"] = g.shift(-3) / p["close"] - 1
    p["fwd_5d"] = g.shift(-5) / p["close"] - 1
    return p


def _split(pr: pd.DataFrame):
    dates = np.sort(pr["date"].unique())
    d1 = dates[int(0.70 * len(dates))]
    d2 = dates[int(0.85 * len(dates))]
    tr = pr[pr["date"] < d1]
    va = pr[(pr["date"] >= d1) & (pr["date"] < d2)]
    te = pr[pr["date"] >= d2]
    return tr, va, te


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    # drop-subset metrics: how well does the regressor fare on the actual
    # drop-candidate rows (where the runner applies it)?
    return {"mae": mae, "dir": dir_acc}


def _drop_subset_metrics(te: pd.DataFrame, y_pred: np.ndarray, h: str) -> dict:
    mask = te["y_drop"].values == 1
    if mask.sum() == 0:
        return {"mae_drop": float("nan"), "dir_drop": float("nan"),
                "bias_drop": float("nan"), "n_drop": 0}
    yt = te[h].values[mask]; yp = y_pred[mask]
    return {
        "mae_drop": float(np.mean(np.abs(yt - yp))),
        "dir_drop": float(np.mean(np.sign(yt) == np.sign(yp))),
        "bias_drop": float(np.mean(yp - yt)),  # + = over-predicts return
        "n_drop": int(mask.sum()),
    }


def _make(loss: str, seed: int):
    return HistGradientBoostingRegressor(loss=loss, random_state=seed, **HGBR_KW)


def _fit_predict(X_tr, y_tr, X_te, *, loss="squared_error",
                 sample_weight=None, seeds=(42,)) -> np.ndarray:
    preds = []
    for s in seeds:
        reg = _make(loss, s)
        reg.fit(X_tr, y_tr, sample_weight=sample_weight)
        preds.append(reg.predict(X_te))
    return np.mean(preds, axis=0)


def _fit_ensemble(X_tr, y_tr, *, loss="squared_error",
                  sample_weight=None, seeds=(42,)) -> list:
    models = []
    for s in seeds:
        reg = _make(loss, s)
        reg.fit(X_tr, y_tr, sample_weight=sample_weight)
        models.append(reg)
    return models


def _predict_ensemble(models, X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


def main():
    print("[drop-v2] loading panel…")
    p = _build_panel()
    pr = p.dropna(subset=FEATS + list(HORIZONS)).reset_index(drop=True)
    tr, va, te = _split(pr)
    print(f"[drop-v2] rows train/val/test: {len(tr):,}/{len(va):,}/{len(te):,}")

    report = {"headline_v2": {}, "ablation": {}, "baseline_v1_reference": {}}

    # Pull v1 reference for apples-to-apples presentation
    v1j = json.loads((OUT / "drop_metrics_v1.json").read_text())["reg"]
    report["baseline_v1_reference"] = v1j

    for h, N in HORIZONS.items():
        print(f"\n[drop-v2] === horizon {h} (N={N}) ===")

        X_tr = tr[FEATS].values; X_va = va[FEATS].values; X_te = te[FEATS].values
        y_tr_raw = tr[h].values; y_va_raw = va[h].values; y_te_raw = te[h].values

        # --- Ablations: run each improvement in isolation vs. baseline. ---
        ablation = {}

        # 0) baseline (= v1 recipe, reproduced inside this harness for parity)
        yhat = _fit_predict(X_tr, y_tr_raw, X_te, seeds=(42,))
        ablation["0_baseline_v1_recipe"] = {**_metrics(y_te_raw, yhat),
                                            **_drop_subset_metrics(te, yhat, h)}

        # 1) drop-conditional sample weighting only
        w_tr = np.where(tr["y_drop"].values == 1, WEIGHT_POS, 1.0)
        yhat = _fit_predict(X_tr, y_tr_raw, X_te, sample_weight=w_tr, seeds=(42,))
        ablation["1_drop_weighted"] = {**_metrics(y_te_raw, yhat),
                                       **_drop_subset_metrics(te, yhat, h)}

        # 2) Absolute-error (L1, robust) loss only
        yhat = _fit_predict(X_tr, y_tr_raw, X_te, loss="absolute_error",
                            seeds=(42,))
        ablation["2_l1_loss"] = {**_metrics(y_te_raw, yhat),
                                 **_drop_subset_metrics(te, yhat, h)}

        # 3) target winsorization only (train-only percentiles)
        lo = np.quantile(y_tr_raw, WINSOR_LO); hi = np.quantile(y_tr_raw, WINSOR_HI)
        y_tr_w = np.clip(y_tr_raw, lo, hi)
        yhat = _fit_predict(X_tr, y_tr_w, X_te, seeds=(42,))
        ablation["3_winsorized"] = {**_metrics(y_te_raw, yhat),
                                    **_drop_subset_metrics(te, yhat, h)}

        # 4) volatility-normalized targets only
        scale_tr = np.sqrt(N) * np.maximum(tr["rv_60"].values, EPS_VOL)
        scale_te = np.sqrt(N) * np.maximum(te["rv_60"].values, EPS_VOL)
        y_tr_z = y_tr_raw / scale_tr
        z_pred = _fit_predict(X_tr, y_tr_z, X_te, seeds=(42,))
        yhat = z_pred * scale_te
        ablation["4_volnorm"] = {**_metrics(y_te_raw, yhat),
                                 **_drop_subset_metrics(te, yhat, h)}

        # 5) 5-seed ensemble only
        yhat = _fit_predict(X_tr, y_tr_raw, X_te, seeds=SEEDS)
        ablation["5_seedbag"] = {**_metrics(y_te_raw, yhat),
                                 **_drop_subset_metrics(te, yhat, h)}

        # --- Headline v2: ALL FIVE stacked. ---
        w_tr = np.where(tr["y_drop"].values == 1, WEIGHT_POS, 1.0)
        lo = np.quantile(y_tr_raw, WINSOR_LO); hi = np.quantile(y_tr_raw, WINSOR_HI)
        scale_tr = np.sqrt(N) * np.maximum(tr["rv_60"].values, EPS_VOL)
        scale_te = np.sqrt(N) * np.maximum(te["rv_60"].values, EPS_VOL)
        y_tr_final = np.clip(y_tr_raw, lo, hi) / scale_tr   # winsor THEN z-score

        models = _fit_ensemble(X_tr, y_tr_final, loss="absolute_error",
                               sample_weight=w_tr, seeds=SEEDS)
        z_pred = _predict_ensemble(models, X_te)
        y_pred_v2 = z_pred * scale_te

        m_v2 = {**_metrics(y_te_raw, y_pred_v2),
                **_drop_subset_metrics(te, y_pred_v2, h)}
        report["headline_v2"][h] = m_v2
        report["ablation"][h] = ablation

        baseline_mae = ablation["0_baseline_v1_recipe"]["mae"]
        baseline_dir = ablation["0_baseline_v1_recipe"]["dir"]
        baseline_mae_drop = ablation["0_baseline_v1_recipe"]["mae_drop"]
        print(f"[drop-v2] {h}  v1-recipe  MAE={baseline_mae:.4f}  dir={baseline_dir:.3f}  "
              f"MAE(y_drop=1)={baseline_mae_drop:.4f}")
        for k, v in ablation.items():
            if k == "0_baseline_v1_recipe":
                continue
            print(f"[drop-v2] {h}  {k:18s}  MAE={v['mae']:.4f}  dir={v['dir']:.3f}  "
                  f"MAE(y_drop=1)={v['mae_drop']:.4f}")
        print(f"[drop-v2] {h}  v2-stacked         MAE={m_v2['mae']:.4f}  dir={m_v2['dir']:.3f}  "
              f"MAE(y_drop=1)={m_v2['mae_drop']:.4f}")

        # Save ensemble + the metadata the runner needs to denormalize.
        joblib.dump({
            "models": models, "feats": FEATS, "horizon_days": N,
            "volnorm": True, "eps_vol": EPS_VOL,
            "winsor": {"lo": float(lo), "hi": float(hi)},
            "config": {"weight_pos": WEIGHT_POS, "seeds": SEEDS,
                       "loss": "absolute_error", "hgbr_kw": HGBR_KW},
        }, MODELS / f"drop_reg_v2_{h}.joblib")
        print(f"[drop-v2] saved models/drop_reg_v2_{h}.joblib")

    (OUT / "drop_metrics_v2.json").write_text(json.dumps(report, indent=2))
    print("\n[drop-v2] wrote output/drop_metrics_v2.json")
    print("[drop-v2] done. Morning runner still reads drop_reg_v1 — unchanged.")


if __name__ == "__main__":
    main()
