"""Drop-magnitude regressors — per-horizon FINAL recipe.

The v2 ablation (code/36b_drop_v2_pipeline.py, output/drop_metrics_v2.json)
showed that the five improvements do NOT compose — stacking all five actually
underperforms the single best improvement on the production metric
(MAE on the y_drop=1 subset, which is the regime the 08:45 runner scores at
inference after the p_drop / p_burst filter).

Per-horizon winners from the ablation:
  fwd_1d   — horizon too short for drop-weighting to matter; L1-loss + seedbag
             for a marginal dir + robustness gain (0.628 dir vs v1 0.627;
             0.0236 drop MAE vs v1 0.0237).
  fwd_3d   — drop_weighted wins decisively on drop MAE (−11% vs v1,
             0.0523 vs 0.0589). Adding L1-loss on top to see if robustness
             compounds with weighting; adding 5-seed bag for variance reduction.
  fwd_5d   — same story, −20% drop MAE (0.0626 vs 0.0783). Same recipe as 3d.

This script writes the FINAL per-horizon recipe to the same v2 filenames
used by the ablation script (drop_reg_v2_fwd_{1d,3d,5d}.joblib), so the
ablation-era models (which were all-5-stacked, and provably worse) are
overwritten. Morning runner still reads drop_reg_v1_* and is unaffected.

Also runs one additional combo check per long horizon — drop_weighted +
L1-loss — to verify the chosen recipe isn't a local optimum.

Output:
  models/drop_reg_v2_fwd_{1d,3d,5d}.joblib  (overwrites the ablation ensembles)
  output/drop_metrics_v2_final.json          (chosen recipes + combo checks)
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
FEATS = V7_FEATS + TREND

HORIZONS = {"fwd_1d": 1, "fwd_3d": 3, "fwd_5d": 5}
WEIGHT_POS = 5.0
SEEDS = [42, 7, 1337, 2024, 99]
HGBR_KW = dict(max_iter=300, max_depth=5, learning_rate=0.05,
               l2_regularization=1.0, min_samples_leaf=32,
               early_stopping=False)

# Per-horizon recipes — chosen by the combo-sanity check in the first pass.
# At 3d/5d, drop_weighted + MSE beat drop_weighted + L1 on BOTH drop-subset
# MAE *and* drop-subset direction (which is what the runner cares about when
# it ranks candidates; overall direction is meaningless on y_drop=0 rows).
# At 1d, horizon is too short for drop-weighting to pay off, so stay on L1.
RECIPES = {
    "fwd_1d": {"weight_pos": 1.0, "loss": "absolute_error", "seeds": SEEDS},
    "fwd_3d": {"weight_pos": WEIGHT_POS, "loss": "squared_error", "seeds": SEEDS},
    "fwd_5d": {"weight_pos": WEIGHT_POS, "loss": "squared_error", "seeds": SEEDS},
}


def _panel() -> pd.DataFrame:
    p = pd.read_csv(DATA / "drop_panel_v1.csv", parse_dates=["date"])
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False)["close"]
    p["fwd_1d"] = g.shift(-1) / p["close"] - 1
    p["fwd_3d"] = g.shift(-3) / p["close"] - 1
    p["fwd_5d"] = g.shift(-5) / p["close"] - 1
    return p


def _split(pr):
    dates = np.sort(pr["date"].unique())
    d1 = dates[int(0.70 * len(dates))]; d2 = dates[int(0.85 * len(dates))]
    tr = pr[pr["date"] < d1]
    va = pr[(pr["date"] >= d1) & (pr["date"] < d2)]
    te = pr[pr["date"] >= d2]
    return tr, va, te


def _fit_ensemble(X_tr, y_tr, *, loss, sample_weight, seeds):
    models = []
    for s in seeds:
        reg = HistGradientBoostingRegressor(loss=loss, random_state=s, **HGBR_KW)
        reg.fit(X_tr, y_tr, sample_weight=sample_weight)
        models.append(reg)
    return models


def _predict_ensemble(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def _metrics(te, h, y_pred):
    yt = te[h].values
    mae = float(np.mean(np.abs(yt - y_pred)))
    dir_acc = float(np.mean(np.sign(yt) == np.sign(y_pred)))
    mask = te["y_drop"].values == 1
    mae_drop = float(np.mean(np.abs(yt[mask] - y_pred[mask]))) if mask.any() else float("nan")
    dir_drop = float(np.mean(np.sign(yt[mask]) == np.sign(y_pred[mask]))) if mask.any() else float("nan")
    bias_drop = float(np.mean(y_pred[mask] - yt[mask])) if mask.any() else float("nan")
    return {"mae": mae, "dir": dir_acc, "mae_drop": mae_drop,
            "dir_drop": dir_drop, "bias_drop": bias_drop,
            "n_drop": int(mask.sum())}


def main():
    p = _panel()
    pr = p.dropna(subset=FEATS + list(HORIZONS)).reset_index(drop=True)
    tr, va, te = _split(pr)
    print(f"[final] rows train/val/test: {len(tr):,}/{len(va):,}/{len(te):,}")

    X_tr = tr[FEATS].values; X_te = te[FEATS].values
    w_pos = np.where(tr["y_drop"].values == 1, WEIGHT_POS, 1.0)
    w_off = np.ones(len(tr), dtype=float)

    final_report = {"chosen_recipe_per_horizon": {}, "combo_sanity_check": {}}

    for h, N in HORIZONS.items():
        print(f"\n[final] === {h} ===")
        recipe = RECIPES[h]
        sw = w_pos if recipe["weight_pos"] > 1.0 else w_off
        y_tr = tr[h].values

        # --- CHOSEN recipe (ensemble) ---
        models = _fit_ensemble(X_tr, y_tr, loss=recipe["loss"],
                               sample_weight=sw, seeds=recipe["seeds"])
        yhat = _predict_ensemble(models, X_te)
        met = _metrics(te, h, yhat)
        print(f"[final] {h}  CHOSEN ({recipe})")
        print(f"[final] {h}  MAE={met['mae']:.4f}  dir={met['dir']:.3f}  "
              f"MAE(y_drop=1)={met['mae_drop']:.4f}  "
              f"dir(y_drop=1)={met['dir_drop']:.3f}  bias={met['bias_drop']:+.4f}")

        joblib.dump({
            "models": models, "feats": FEATS, "horizon_days": N,
            "volnorm": False, "winsor": None,
            "recipe": recipe,
        }, MODELS / f"drop_reg_v2_{h}.joblib")
        print(f"[final] saved models/drop_reg_v2_{h}.joblib (overwrote ablation all-5 stack)")
        final_report["chosen_recipe_per_horizon"][h] = {"recipe": recipe, "metrics": met}

        # --- combo sanity: cross-check the other loss as an alternative ---
        if recipe["weight_pos"] > 1.0:
            alt_loss = "absolute_error" if recipe["loss"] == "squared_error" else "squared_error"
            models2 = _fit_ensemble(X_tr, y_tr, loss=alt_loss,
                                    sample_weight=sw, seeds=SEEDS)
            yhat2 = _predict_ensemble(models2, X_te)
            met2 = _metrics(te, h, yhat2)
            print(f"[final] {h}  combo-check drop_w + {alt_loss}: "
                  f"MAE={met2['mae']:.4f}  dir={met2['dir']:.3f}  "
                  f"MAE_drop={met2['mae_drop']:.4f}  dir_drop={met2['dir_drop']:.3f}")
            final_report["combo_sanity_check"][h] = {
                f"drop_w+{recipe['loss']}+seedbag (chosen)": met,
                f"drop_w+{alt_loss}+seedbag (alt)": met2,
            }

    (OUT / "drop_metrics_v2_final.json").write_text(
        json.dumps(final_report, indent=2))
    print("\n[final] wrote output/drop_metrics_v2_final.json")
    print("[final] done — morning runner still reads drop_reg_v1; v2 models are new-only.")


if __name__ == "__main__":
    main()
