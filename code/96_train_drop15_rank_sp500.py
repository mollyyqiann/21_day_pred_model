"""SP500 within-date drop ranker — answers "which SP500 names are most
drop-prone today?" without needing absolute probabilities.

Why: SP500 has only ~5% base rate for -15% touch over 21d, and the dominant
features are macro (identical across SP500 names on a given date), so absolute
classification fails (v1's 0.09 calibrator ceiling; v2a/b were worse).

But within a given date, names DO differ — that's what we model here.

Approach:
  - Target = drawdown_depth = max(0, -max_fwd21_drawdown), continuous.
  - Train GBR on the depth (richer gradient than binary).
  - At inference, rank stocks per-date by predicted depth.

Evaluation (the metrics that actually matter for a ranker):
  - Per-date Spearman corr(predicted_score, realized_drawdown_depth).
    Averaged across test dates.
  - Top-K daily picks: median realized drawdown depth of model's top-K per date,
    vs random pick from same date. K = 5, 10, 20.
  - Compare to v1 (re-ranked within-date) for fair head-to-head.

Outputs:
  models/monthly_gainer_drop15_rank_sp500.joblib
  output/monthly_gainer/drop15_rank_sp500_metrics.json

Usage:
  python 96_train_drop15_rank_sp500.py
"""

import sys; sys.stdout.reconfigure(line_buffering=True)
import json, time, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output" / "monthly_gainer"

sys.path.insert(0, str(ROOT / "code"))
from regime_features import attach_regime  # noqa

# Reuse feature definitions from the v2 trainer
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("_v2", ROOT / "code" / "95_train_drop15_v2.py")
_v2 = module_from_spec(_spec); _spec.loader.exec_module(_v2)
BASE_FEATURES = _v2.BASE_FEATURES
CATALYST = _v2.CATALYST
build_features = _v2.build_features
add_y21_drop15 = _v2.add_y21_drop15
time_split = _v2.time_split


def evaluate_ranker(test_df: pd.DataFrame, score_col: str, label_col: str = "drawdown_depth"):
    """Per-date Spearman + top-K average realized drawdown depth."""
    per_date_spr = []
    topk_depths = {5: [], 10: [], 20: []}
    median_depths = []
    n_dates_used = 0
    for dt, g in test_df.groupby("date"):
        if len(g) < 30:
            continue
        n_dates_used += 1
        # Spearman within date
        if g[score_col].nunique() > 1 and g[label_col].nunique() > 1:
            r, _ = spearmanr(g[score_col].values, g[label_col].values)
            per_date_spr.append(r)
        # Top-K realized drawdown depth (averaged)
        g_sorted = g.sort_values(score_col, ascending=False)
        for k in topk_depths:
            if len(g_sorted) >= k:
                topk_depths[k].append(g_sorted[label_col].iloc[:k].mean())
        median_depths.append(g[label_col].median())
    return {
        "n_test_dates": n_dates_used,
        "per_date_spearman_mean": float(np.mean(per_date_spr)) if per_date_spr else float("nan"),
        "per_date_spearman_med":  float(np.median(per_date_spr)) if per_date_spr else float("nan"),
        "topk_avg_realized_depth": {k: float(np.mean(v)) if v else float("nan")
                                     for k, v in topk_depths.items()},
        "universe_median_depth": float(np.mean(median_depths)) if median_depths else float("nan"),
    }


def lift_vs_median(metrics):
    """Top-K depth / universe-median depth — >1 means model picks deeper drops than typical name."""
    med = metrics["universe_median_depth"]
    if med <= 0:
        return {k: float("nan") for k in metrics["topk_avg_realized_depth"]}
    return {k: v / med for k, v in metrics["topk_avg_realized_depth"].items()}


def main():
    t0 = time.time()
    print("[96] loading sp500 panel + catalyst")
    panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    catalyst = pd.read_csv(DATA / "catalyst_features_sp500.csv", parse_dates=["date"])

    print("[96] building features")
    panel = build_features(panel, catalyst[["ticker", "date"] + CATALYST])
    panel = add_y21_drop15(panel)

    # Target: depth of drawdown (positive number; 0 if didn't drop)
    panel["drawdown_depth"] = np.maximum(0.0, -panel["max_fwd21_drawdown"])

    lab = panel.dropna(
        subset=BASE_FEATURES + ["max_fwd21_drawdown"]
    ).reset_index(drop=True)
    print(f"[96] labeled+complete: {len(lab):,}  mean depth={lab['drawdown_depth'].mean():.4f}  "
          f"med depth={lab['drawdown_depth'].median():.4f}")

    train, val, test = time_split(lab)
    print(f"[96] split: train={len(train):,} val={len(val):,} test={len(test):,}  "
          f"test dates={test['date'].nunique()}")

    med = train[BASE_FEATURES].median(numeric_only=True)
    Xtr = train[BASE_FEATURES].fillna(med).values
    Xte = test[BASE_FEATURES].fillna(med).values
    ytr = train["drawdown_depth"].values

    print("[96] training GBR (regression on drawdown_depth)")
    gbr = GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.8, loss="huber", random_state=42)
    gbr.fit(Xtr, ytr)

    test = test.copy()
    test["pred_depth"] = gbr.predict(Xte)

    rank_metrics = evaluate_ranker(test, "pred_depth")
    rank_lift = lift_vs_median(rank_metrics)
    print(f"\n[96] RANKER (v_rank) metrics:")
    print(f"  per-date Spearman:  mean={rank_metrics['per_date_spearman_mean']:.4f}  "
          f"median={rank_metrics['per_date_spearman_med']:.4f}")
    print(f"  top-K avg realized drawdown depth (negative = deeper drop in our sign):")
    for k, v in rank_metrics["topk_avg_realized_depth"].items():
        print(f"    K={k}: depth={v:.4f}  lift vs universe-median={rank_lift[k]:.3f}x")
    print(f"  universe median depth: {rank_metrics['universe_median_depth']:.4f}")

    # ===== Compare: v1 classifier ranked within-date =====
    print("\n[96] comparing to v1 (binary classifier, ranked within-date)")
    v1 = joblib.load(MODELS / "monthly_gainer_drop15_sp500.joblib")
    v1_feats = v1["feats"]
    v1_med = pd.Series(v1["impute_medians"])
    Xte_v1 = test[v1_feats].fillna(v1_med).values
    test["v1_p_drop"] = v1["calibrator"].predict_proba(Xte_v1)[:, 1]

    v1_metrics = evaluate_ranker(test, "v1_p_drop")
    v1_lift = lift_vs_median(v1_metrics)
    print(f"[v1 as ranker] per-date Spearman: mean={v1_metrics['per_date_spearman_mean']:.4f}  "
          f"median={v1_metrics['per_date_spearman_med']:.4f}")
    print(f"[v1 as ranker] top-K realized depth + lift:")
    for k in (5, 10, 20):
        print(f"    K={k}: depth={v1_metrics['topk_avg_realized_depth'][k]:.4f}  "
              f"lift={v1_lift[k]:.3f}x")

    # Feature importance
    imp = pd.Series(gbr.feature_importances_, index=BASE_FEATURES).sort_values(ascending=False)
    print(f"\n[96] top-15 importances:")
    print(imp.head(15).to_string())

    # Save
    artifact = {
        "raw_gbr": gbr, "feats": BASE_FEATURES,
        "impute_medians": med.to_dict(),
        "label": "drawdown_depth",
        "label_def": "max(0, -min(close[t+1..t+21]) / close[t] + 1)  [depth of 21d drawdown]",
        "universe": "sp500", "variant": "rank_regression",
        "metrics": rank_metrics,
        "lift_vs_median": rank_lift,
        "feature_importance": imp.to_dict(),
        "comparison_v1_as_ranker": {"metrics": v1_metrics, "lift_vs_median": v1_lift},
    }
    joblib.dump(artifact, MODELS / "monthly_gainer_drop15_rank_sp500.joblib")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in artifact.items() if k not in ("raw_gbr",)}
    (OUT / "drop15_rank_sp500_metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD (SP500, as a within-date ranker)")
    print("=" * 60)
    print(f"{'metric':<35}{'v1 (re-ranked)':>20}{'v_rank':>20}")
    print(f"{'per-date Spearman (mean)':<35}"
          f"{v1_metrics['per_date_spearman_mean']:>20.4f}"
          f"{rank_metrics['per_date_spearman_mean']:>20.4f}")
    print(f"{'per-date Spearman (median)':<35}"
          f"{v1_metrics['per_date_spearman_med']:>20.4f}"
          f"{rank_metrics['per_date_spearman_med']:>20.4f}")
    for k in (5, 10, 20):
        print(f"{f'top-{k} realized depth':<35}"
              f"{v1_metrics['topk_avg_realized_depth'][k]:>20.4f}"
              f"{rank_metrics['topk_avg_realized_depth'][k]:>20.4f}")
        print(f"{f'top-{k} lift vs median':<35}"
              f"{v1_lift[k]:>20.3f}{rank_lift[k]:>20.3f}")
    print(f"\n[96] elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
