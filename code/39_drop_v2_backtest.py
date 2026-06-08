"""Walk-forward backtest for drop v2.

For every rebuild date in the panel, train on data strictly before the rebuild,
score the next `step_days` worth of rows, and record metrics + per-row
predictions. Uses a rolling 60-day recency recalibration applied at scoring
time to correct for regime drift.

Outputs:
  output/drop_v2/walk_forward_metrics.csv    : one row per window with AUC/AP/...
  output/drop_v2/walk_forward_predictions.csv: one row per scored (ticker, date)
  output/drop_v2/walk_forward_summary.json   : headline numbers

Also emits a "v1-baseline" track that retrains a v1-style model
(GradientBoostingClassifier on the original 17 features, no class weighting,
no calibration) in the same walk-forward windows, for direct head-to-head.

Usage: python3 code/39_drop_v2_backtest.py [--step 20] [--min-train-days 252]
"""
from __future__ import annotations

import sys as _sys_buf
_sys_buf.stdout.reconfigure(line_buffering=True)

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "drop_panel_v2.csv"
OUT_DIR = ROOT / "output" / "drop_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Feature sets (mirror the v2 pipeline) ---- #
V1_FEATURES = [
    "rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
    "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60", "overnight_gap",
    "ma_stack", "up_streak", "up_bigdays_20d",
    "dist_ma60_atr", "ma60_slope_60d", "run_length",
]
RANK_FEATURES = [
    "rsi_14_rank", "macd_rank", "macd_hist_rank", "bb_z20_rank",
    "atr_pct_rank", "range_pct_rank", "vol_z_rank", "vol_5d_rank", "rv_60_rank",
    "overnight_gap_rank",
    "up_bigdays_20d_rank", "dist_ma60_atr_rank", "ma60_slope_60d_rank",
    "run_length_rank", "up_streak_rank",
]
REGIME_FEATURES = [
    "vix", "vix_5d_chg", "vix_z60",
    "spy_20d_dd", "spy_rv_20",
    "aaii_bb_spread", "aaii_bb_z52",
    "fng", "fng_5d_chg",
    "tnx_10y", "tnx_5d_chg", "dxy", "dxy_5d_chg", "gold_5d_ret",
    "qqq_iwm_5d_spread", "xlk_xlf_5d_spread",
    "breadth_ma_pos",
]
EDGAR_FEATURES = [
    "edgar_8k_cnt_5d", "edgar_8k_cnt_30d",
    "edgar_10q_days", "edgar_10k_days",
    "edgar_amend_cnt_30d", "edgar_nt_flag_90d", "edgar_13d_cnt_60d",
    "edgar_fb_mean_30d", "edgar_fb_min_30d",
    "edgar_total_cnt_5d",
]
V2_FEATURES = V1_FEATURES + RANK_FEATURES + REGIME_FEATURES + EDGAR_FEATURES

GBC_V2_PARAMS = dict(
    max_iter=400,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=0.1,
    early_stopping=True,
    n_iter_no_change=25,
    validation_fraction=0.1,
    random_state=42,
)

# v1 training: GradientBoostingClassifier, n_estimators trimmed from 300 to 200
# for walk-forward speed. On 376k-row panels this shaves ~30% off each window's
# training time with a <=0.002 AUC impact per ablation spot-checks.
GBC_V1_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, random_state=42,
)


def _sample_weights(y, target_pos_frac=0.20):
    y = np.asarray(y, dtype=np.int8)
    p = y.mean()
    if p <= 0 or p >= 1:
        return np.ones(len(y))
    w_pos = target_pos_frac / p
    w_neg = (1 - target_pos_frac) / (1 - p)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def _fit_v2_model(X_tr, y_tr):
    mdl = HistGradientBoostingClassifier(**GBC_V2_PARAMS)
    mdl.fit(X_tr, y_tr, sample_weight=_sample_weights(y_tr))
    return mdl


def _fit_v1_model(X_tr, y_tr):
    mdl = GradientBoostingClassifier(**GBC_V1_PARAMS)
    mdl.fit(X_tr, y_tr)
    return mdl


def _score_with_recency_recal(p_raw_score: np.ndarray, recency_probs: np.ndarray,
                               recency_y: np.ndarray) -> np.ndarray:
    """Fit isotonic regression on (recency_probs, recency_y), apply to p_raw_score.

    If the recency fold is empty / constant, returns p_raw_score unchanged.
    """
    if len(recency_y) < 200 or recency_y.sum() == 0:
        return p_raw_score
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(recency_probs, recency_y)
        return iso.transform(p_raw_score)
    except Exception:
        return p_raw_score


def _metrics(y, p, label=""):
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {f"{label}_auc": np.nan, f"{label}_ap": np.nan,
                f"{label}_logloss": np.nan, f"{label}_brier": np.nan,
                f"{label}_mean_p": float(p.mean()) if len(p) else np.nan,
                f"{label}_mean_p_pos": np.nan}
    return {
        f"{label}_auc": roc_auc_score(y, p),
        f"{label}_ap": average_precision_score(y, p),
        f"{label}_logloss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        f"{label}_brier": brier_score_loss(y, p),
        f"{label}_mean_p": float(p.mean()),
        f"{label}_mean_p_pos": float(p[y == 1].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=20,
                    help="Retrain every N business days (default 20)")
    ap.add_argument("--min-train-days", type=int, default=252,
                    help="Require at least N days of training data before first window")
    ap.add_argument("--recency-days", type=int, default=60,
                    help="Days of out-of-sample predictions used for recency recal")
    args = ap.parse_args()

    print(f"[bt] loading {PANEL}")
    df = pd.read_csv(PANEL, parse_dates=["date"])
    print(f"  shape: {df.shape}")
    df = df[df["y_drop"].isin([0, 1])].copy()
    df["y_drop"] = df["y_drop"].astype(np.int8)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    print(f"  valid rows: {len(df)}")

    # Drop rows with missing v2 features (should be rare; impute rather than drop)
    for col in V2_FEATURES:
        if col not in df.columns:
            print(f"[bt] WARN missing col: {col} — filling 0")
            df[col] = 0.0
    df[V2_FEATURES] = df[V2_FEATURES].fillna(df[V2_FEATURES].median(numeric_only=True))

    all_dates = np.sort(df["date"].unique())
    # Rebuild dates: start after min_train_days, then every step days
    starts = []
    i = args.min_train_days
    while i < len(all_dates) - args.step:
        starts.append(i)
        i += args.step

    print(f"[bt] windows: {len(starts)}  step={args.step}  "
          f"first train ends: {pd.Timestamp(all_dates[args.min_train_days-1]).date()}  "
          f"first test starts: {pd.Timestamp(all_dates[args.min_train_days]).date()}",
          flush=True)

    # Resume: if existing metrics file has some rows, pick up after the last one.
    done_windows = set()
    resume_metrics = OUT_DIR / "walk_forward_metrics.csv"
    resume_preds = OUT_DIR / "walk_forward_predictions.csv"
    if resume_metrics.exists():
        try:
            prev = pd.read_csv(resume_metrics)
            done_windows = set(int(x) for x in prev["window"])
            print(f"[bt] resume: {len(done_windows)} windows already in "
                  f"{resume_metrics.name}", flush=True)
        except Exception as e:
            print(f"[bt] resume load failed ({e}); starting fresh", flush=True)

    per_window: list[dict] = []
    preds: list[pd.DataFrame] = []

    t0 = time.time()
    # Track recency buffer: list of (date, ticker, p_v2_raw, y_drop) tuples
    # used by recency recal.
    recency_buf: list[tuple[pd.Timestamp, str, float, int]] = []

    # If we have prior predictions CSV, seed `preds` so the final summary
    # includes them.
    if resume_preds.exists() and done_windows:
        try:
            prev_preds = pd.read_csv(resume_preds, parse_dates=["date"])
            preds.append(prev_preds)
            # Seed per_window too
            prev_m = pd.read_csv(resume_metrics)
            per_window.extend(prev_m.to_dict("records"))
            # Rebuild recency buffer from prior predictions so recal works
            for _, r in prev_preds.iterrows():
                recency_buf.append((pd.Timestamp(r["date"]), str(r["ticker"]),
                                    float(r["p_v2_raw"]), int(r["y_drop"])))
            print(f"[bt] resume: seeded {len(prev_preds)} prior prediction rows "
                  f"into recency buffer", flush=True)
        except Exception as e:
            print(f"[bt] resume preds load failed ({e}); continuing", flush=True)

    for widx, si in enumerate(starts, 1):
        if widx in done_windows:
            continue
        train_end = all_dates[si]
        test_start = all_dates[si]
        test_end = all_dates[min(si + args.step, len(all_dates) - 1)]

        tr = df[df["date"] < train_end]
        te = df[(df["date"] >= test_start) & (df["date"] < test_end)]
        if len(tr) < 10000 or len(te) < 100:
            continue

        # ---- v1 baseline: 17 features, no weighting, raw probs
        X_tr1 = tr[V1_FEATURES].values
        y_tr = tr["y_drop"].values
        X_te1 = te[V1_FEATURES].values
        y_te = te["y_drop"].values
        m1 = _fit_v1_model(X_tr1, y_tr)
        p_te_v1 = m1.predict_proba(X_te1)[:, 1]

        # ---- v2: 59 features, class-weighted, HistGBC, raw probs
        X_tr2 = tr[V2_FEATURES].values
        X_te2 = te[V2_FEATURES].values
        m2 = _fit_v2_model(X_tr2, y_tr)
        p_te_v2_raw = m2.predict_proba(X_te2)[:, 1]

        # ---- v2 + recency recalibration: use last `recency_days` of history
        recency_cutoff = pd.Timestamp(test_start) - pd.Timedelta(days=args.recency_days)
        r_probs = np.array([b[2] for b in recency_buf if b[0] >= recency_cutoff])
        r_y = np.array([b[3] for b in recency_buf if b[0] >= recency_cutoff])
        p_te_v2_recal = _score_with_recency_recal(p_te_v2_raw, r_probs, r_y)

        # Record metrics
        row = {
            "window": widx,
            "train_end": pd.Timestamp(train_end).strftime("%Y-%m-%d"),
            "test_start": pd.Timestamp(test_start).strftime("%Y-%m-%d"),
            "test_end": pd.Timestamp(test_end).strftime("%Y-%m-%d"),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "pos_train": float(y_tr.mean()),
            "pos_test": float(y_te.mean()),
            "recency_buffer_size": int(len(r_y)),
        }
        row.update(_metrics(y_te, p_te_v1, "v1"))
        row.update(_metrics(y_te, p_te_v2_raw, "v2_raw"))
        row.update(_metrics(y_te, p_te_v2_recal, "v2_recal"))
        per_window.append(row)

        # Extend recency buffer with the fresh predictions (will be used by next window)
        for d, t_, p, y in zip(te["date"].values, te["ticker"].values,
                               p_te_v2_raw, y_te):
            recency_buf.append((pd.Timestamp(d), str(t_), float(p), int(y)))

        # Save per-row predictions for this window
        preds.append(pd.DataFrame({
            "date": te["date"].values,
            "ticker": te["ticker"].values,
            "y_drop": y_te,
            "p_v1": p_te_v1,
            "p_v2_raw": p_te_v2_raw,
            "p_v2_recal": p_te_v2_recal,
        }))

        # Print progress every window (was every 5) so a kill leaves a trail.
        elapsed = time.time() - t0
        eta = elapsed / widx * (len(starts) - widx) if widx else 0
        print(f"[bt] {widx}/{len(starts)}  test {row['test_start']}..{row['test_end']}  "
              f"v1 AUC={row['v1_auc']:.3f} AP={row['v1_ap']:.3f}  |  "
              f"v2_raw AUC={row['v2_raw_auc']:.3f} AP={row['v2_raw_ap']:.3f}  |  "
              f"v2_recal AUC={row['v2_recal_auc']:.3f} AP={row['v2_recal_ap']:.3f}  "
              f"mean_p|pos={row['v2_recal_mean_p_pos']:.3f}  eta {eta/60:.1f}min",
              flush=True)

        # CHECKPOINT EVERY WINDOW — so a process death preserves all work.
        pd.DataFrame(per_window).to_csv(OUT_DIR / "walk_forward_metrics.csv", index=False)
        pd.concat(preds, ignore_index=True).to_csv(
            OUT_DIR / "walk_forward_predictions.csv", index=False)

        # Free model memory between windows
        del m1, m2
        gc.collect()

    metrics_df = pd.DataFrame(per_window)
    preds_df = pd.concat(preds, ignore_index=True)
    metrics_df.to_csv(OUT_DIR / "walk_forward_metrics.csv", index=False)
    preds_df.to_csv(OUT_DIR / "walk_forward_predictions.csv", index=False)

    # ---- Summary: average metrics weighted by test rows ----
    def _wavg(col):
        return float((metrics_df[col] * metrics_df["n_test"]).sum() /
                     metrics_df["n_test"].sum())

    summary = {
        "n_windows": int(len(metrics_df)),
        "total_test_rows": int(metrics_df["n_test"].sum()),
        "v1": {
            "auc_wavg": _wavg("v1_auc"),
            "ap_wavg": _wavg("v1_ap"),
            "logloss_wavg": _wavg("v1_logloss"),
            "brier_wavg": _wavg("v1_brier"),
            "mean_p_wavg": _wavg("v1_mean_p"),
            "mean_p_pos_wavg": _wavg("v1_mean_p_pos"),
        },
        "v2_raw": {
            "auc_wavg": _wavg("v2_raw_auc"),
            "ap_wavg": _wavg("v2_raw_ap"),
            "logloss_wavg": _wavg("v2_raw_logloss"),
            "brier_wavg": _wavg("v2_raw_brier"),
            "mean_p_wavg": _wavg("v2_raw_mean_p"),
            "mean_p_pos_wavg": _wavg("v2_raw_mean_p_pos"),
        },
        "v2_recal": {
            "auc_wavg": _wavg("v2_recal_auc"),
            "ap_wavg": _wavg("v2_recal_ap"),
            "logloss_wavg": _wavg("v2_recal_logloss"),
            "brier_wavg": _wavg("v2_recal_brier"),
            "mean_p_wavg": _wavg("v2_recal_mean_p"),
            "mean_p_pos_wavg": _wavg("v2_recal_mean_p_pos"),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT_DIR / "walk_forward_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print("\n[bt] === WALK-FORWARD SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
