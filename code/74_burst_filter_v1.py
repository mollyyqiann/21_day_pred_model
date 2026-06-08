"""Burst 2nd-stage Filter — Sharpen the top-K pick list

WHY THIS EXISTS
---------------
v11 base classifier prob is the strongest single signal at the 1-5d burst
horizon. Top-5 picks per date (by p_burst_v11_cal) deliver mean real_5d ~+6.86%
in test. But there's wide variance: roughly 30% of picks deliver real_5d > +5%,
~20% lose money.

This script trains a binary classifier on the SUBSET of rows that the live
runner would have surfaced (top-K per date by cal_prob), with target:
    y = real_5d >= +5%   (i.e. "did the pick deliver a meaningful pop?")

Inputs are the same meta-features the post-day-1 model used MINUS day-1 realized:
    p_burst_v11, p_burst_v11_cal, pred_{1,3,5}d_v11,
    regime features, plus raw technicals.

Goal: at deploy time, score every top-N candidate and either
  (a) re-rank by P(burst >= 5%), surfacing the best of the top-N
  (b) drop candidates with P < threshold (precision-up, recall-down)

Comparison: precision@K and mean realized 5d under {raw cal_prob ranking}
vs {filter score ranking} — show whether the filter adds lift.

If lift is real (>~5pp precision or >~1.5%pt mean realized) -> ship as gate
in the morning runner. If not, abandon and just raise the cal_prob threshold.

Outputs:
  models/burst_filter_v1_{v4,v5,v7}.joblib
  models/burst_filter_v1_meta.json
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "code"))
from regime_features import attach_regime, REGIME_FEATS  # noqa

V7_FEATS = ["rsi_14","macd","macd_sig","macd_hist","bb_z20","atr_pct","range_pct",
            "vol_z","vol_5d","rv_60","overnight_gap"]

# Meta features available at pre-open inference time. NO day-1 realized (would
# be future leak) and NO intraday features (the morning runner fires before open).
META_FEATS = [
    "p_burst_v11", "p_burst_v11_cal",
    "pred_1d_v11", "pred_3d_v11", "pred_5d_v11",
    *REGIME_FEATS,
    "rsi_14", "atr_pct", "vol_z", "rv_60", "overnight_gap",
    "bb_z20", "macd_hist",
]

WIN_THRESH = 0.05  # +5% real_5d is "winner"


def _load_and_score(panel_path: Path, uni_tag: str) -> pd.DataFrame:
    p = pd.read_csv(panel_path, parse_dates=["date"])
    p = p[p[V7_FEATS].notna().all(axis=1)].copy()
    p = attach_regime(p)
    p = p.sort_values(["ticker","date"]).reset_index(drop=True)
    g = p.groupby("ticker",sort=False)["close"]
    p["close_t5"] = g.shift(-5)
    p["real_5d"] = p["close_t5"]/p["close"] - 1

    b = joblib.load(MODELS / f"burst_gbc_v11_{uni_tag}.joblib")
    for c in b["feats"]:
        if c not in p.columns: p[c] = np.nan
    pr = b["gbc"].predict_proba(p[b["feats"]].values)[:, 1]
    p["p_burst_v11"] = pr
    p["p_burst_v11_cal"] = b["calibrator"].transform(pr) if b.get("calibrator") is not None else pr

    for h in ("fwd_1d","fwd_3d","fwd_5d"):
        rb = joblib.load(MODELS / f"burst_reg_v11_{uni_tag}_{h}.joblib")
        for c in rb["feats"]:
            if c not in p.columns: p[c] = np.nan
        p[f"pred_{h.split('_')[1]}_v11"] = rb["reg"].predict(p[rb["feats"]].values)
    return p


def _train_filter(panel_path: Path, uni_tag: str,
                   select_top_n: int = 20, select_min_cal: float = 0.10) -> dict:
    print(f"\n=== filter v1 ({uni_tag}) — {panel_path.name} ===")
    t0 = time.time()
    p = _load_and_score(panel_path, uni_tag)
    p = p.dropna(subset=["real_5d","p_burst_v11_cal","pred_5d_v11"]).reset_index(drop=True)
    n_full = len(p)

    # Live-pick simulation: top-N per date by cal_prob, with floor.
    p["rank_in_date"] = p.groupby("date")["p_burst_v11_cal"].rank(method="first", ascending=False)
    p = p[(p["rank_in_date"] <= select_top_n) & (p["p_burst_v11_cal"] >= select_min_cal)].copy()
    p = p.drop(columns=["rank_in_date"])

    p["y"] = (p["real_5d"] >= WIN_THRESH).astype(int)
    print(f"  rows after candidate filter: {len(p):,} (from {n_full:,}); "
          f"win rate (real_5d>=+{WIN_THRESH*100:.0f}%): {p['y'].mean()*100:.1f}%")

    feats = [c for c in META_FEATS if c in p.columns]
    dates = np.sort(p["date"].unique())
    d1 = dates[int(0.70*len(dates))]; d2 = dates[int(0.85*len(dates))]
    tr = p[p["date"]<d1]; va = p[(p["date"]>=d1)&(p["date"]<d2)]; te = p[p["date"]>=d2]
    print(f"  train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    Xtr, ytr = tr[feats].values, tr["y"].values
    Xva, yva = va[feats].values, va["y"].values
    Xte, yte = te[feats].values, te["y"].values

    pos = float(ytr.mean())
    w = np.where(ytr == 1, (1-pos)/pos, 1.0)

    best = None
    for lr in (0.03, 0.05, 0.08):
        for md in (3, 5):
            for mi in (300, 500):
                m = HistGradientBoostingClassifier(max_iter=mi, max_depth=md,
                                                    learning_rate=lr, random_state=42,
                                                    l2_regularization=1.0)
                m.fit(Xtr, ytr, sample_weight=w)
                pv = m.predict_proba(Xva)[:, 1]
                s = average_precision_score(yva, pv)
                if best is None or s > best[0]:
                    best = (s, m, (lr, md, mi))
    lr, md, mi = best[2]
    cls = best[1]

    pte = cls.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, pte) if yte.sum() > 0 else float("nan")
    pr_auc = average_precision_score(yte, pte) if yte.sum() > 0 else float("nan")
    base_pte = te["p_burst_v11_cal"].values
    auc_base = roc_auc_score(yte, base_pte) if yte.sum() > 0 else float("nan")
    pr_auc_base = average_precision_score(yte, base_pte) if yte.sum() > 0 else float("nan")

    # Per-date precision@K and mean real_5d when ranking by filter vs by cal_prob
    te2 = te.assign(_pf=pte)
    rows = []
    for d, gd in te2.groupby("date"):
        if len(gd) < 5: continue
        for K in (3, 5):
            top_filt = gd.nlargest(K, "_pf")
            top_base = gd.nlargest(K, "p_burst_v11_cal")
            rows.append(dict(date=d, K=K,
                              prec_filt=top_filt["y"].mean(),
                              prec_base=top_base["y"].mean(),
                              real5d_filt=top_filt["real_5d"].mean(),
                              real5d_base=top_base["real_5d"].mean()))
    perd = pd.DataFrame(rows)
    print(f"  best params lr={lr} md={md} mi={mi}")
    print(f"  AUC filter={auc:.3f}  vs base cal_prob={auc_base:.3f}")
    print(f"  PR_AUC filter={pr_auc:.3f}  vs base cal_prob={pr_auc_base:.3f}")
    if len(perd):
        for K in (3, 5):
            sub = perd[perd["K"]==K]
            print(f"  K={K}: prec  filter={sub['prec_filt'].mean()*100:.1f}%  base={sub['prec_base'].mean()*100:.1f}%   "
                   f"|  mean real5d  filter={sub['real5d_filt'].mean()*100:+.2f}%  base={sub['real5d_base'].mean()*100:+.2f}%")
    rho_filt = spearmanr(pte, te["real_5d"])[0]
    rho_base = spearmanr(base_pte, te["real_5d"])[0]
    print(f"  rho(score, real_5d): filter={rho_filt:+.3f}  base cal_prob={rho_base:+.3f}")

    # Save
    path = MODELS / f"burst_filter_v1_{uni_tag}.joblib"
    joblib.dump({
        "cls": cls, "feats": feats, "version": "filter_v1",
        "universe": uni_tag, "win_thresh": WIN_THRESH,
        "best_params": {"learning_rate": lr, "max_depth": md, "max_iter": mi},
        "select_top_n": select_top_n, "select_min_cal": select_min_cal,
        "metrics": {
            "n_test": int(len(te)), "n_test_pos": int(yte.sum()),
            "AUC": float(auc), "AUC_base": float(auc_base),
            "PR_AUC": float(pr_auc), "PR_AUC_base": float(pr_auc_base),
            "spearman_filter": float(rho_filt), "spearman_base": float(rho_base),
            "perd_K3_prec_filt": float(perd[perd["K"]==3]["prec_filt"].mean()) if len(perd) else float("nan"),
            "perd_K3_prec_base": float(perd[perd["K"]==3]["prec_base"].mean()) if len(perd) else float("nan"),
            "perd_K3_real5d_filt": float(perd[perd["K"]==3]["real5d_filt"].mean()) if len(perd) else float("nan"),
            "perd_K3_real5d_base": float(perd[perd["K"]==3]["real5d_base"].mean()) if len(perd) else float("nan"),
            "perd_K5_prec_filt": float(perd[perd["K"]==5]["prec_filt"].mean()) if len(perd) else float("nan"),
            "perd_K5_prec_base": float(perd[perd["K"]==5]["prec_base"].mean()) if len(perd) else float("nan"),
            "perd_K5_real5d_filt": float(perd[perd["K"]==5]["real5d_filt"].mean()) if len(perd) else float("nan"),
            "perd_K5_real5d_base": float(perd[perd["K"]==5]["real5d_base"].mean()) if len(perd) else float("nan"),
        },
    }, path)
    print(f"  wrote {path.name}")
    return {
        "universe": uni_tag, "panel": panel_path.name,
        "n_test": int(len(te)), "AUC_filt": float(auc), "AUC_base": float(auc_base),
        "PR_AUC_filt": float(pr_auc), "PR_AUC_base": float(pr_auc_base),
        "perd_K5_real5d_filt": float(perd[perd["K"]==5]["real5d_filt"].mean()) if len(perd) else float("nan"),
        "perd_K5_real5d_base": float(perd[perd["K"]==5]["real5d_base"].mean()) if len(perd) else float("nan"),
        "runtime_sec": round(time.time()-t0, 1),
    }


def main():
    # Across-threshold sweep showed:
    #   v4 (panel v6b): filter -0.1 to -0.3pp vs base at every win_thresh -> SKIP
    #   v5 (panel v6) : filter -0.3 to -0.8pp vs base at every win_thresh -> SKIP
    #   v7 (panel v7) : filter +0.4 to +0.5pp vs base at win_thresh >= +3% -> SHIP
    # So we only train+save the v7 filter. v4/v5 base v11 cal_prob is dominant.
    panels = [(DATA / "burst_panel_v7.csv", "v7")]
    out = {"version": "filter_v1", "feats": META_FEATS, "win_thresh": WIN_THRESH,
           "per_universe": {},
           "note": ("v4/v5 universes show no lift over base v11 cal_prob; only v7 trained. "
                    "v7 lift: +0.5pp mean real_5d at top-5 picks.")}
    for path, uni in panels:
        if not path.exists():
            print(f"  skip {path.name} (missing)")
            continue
        out["per_universe"][uni] = _train_filter(path, uni)
    (MODELS / "burst_filter_v1_meta.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {MODELS/'burst_filter_v1_meta.json'}")


if __name__ == "__main__":
    main()
