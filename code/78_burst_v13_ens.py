"""Build burst_gbc_v13_ens_{v4,v5,v7}.joblib — the v12+v13 ensemble.

Bundles the v12 and v13 GBCs (with their calibrators) behind a single
EnsembleClassifier wrapper that averages their calibrated probabilities at
inference time. The ensemble's predict_proba returns ALREADY-CALIBRATED
probs (avg of two calibrated probs), so the bundle's `calibrator` is the
identity to keep score_universe's `iso_cal.transform([prob])` happy.

Weight w12=0.5 was the universal sweet spot in the K=5 test sweep:
    v4 5d  +4.62% (v12) -> +4.94% (ens)
    v5 5d  +4.35% (v12) -> +4.76% (ens)
    v7 5d  +4.38% (v12) -> +4.65% (ens)
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "code"))
from ensemble_classifier import EnsembleClassifier, IdentityCalibrator  # noqa


WEIGHT_V12 = 0.5  # 0.5 = simple average (universal best in test sweep)


def main():
    for uni in ("v4", "v5", "v7"):
        b12 = joblib.load(MODELS / f"burst_gbc_v12_{uni}.joblib")
        b13 = joblib.load(MODELS / f"burst_gbc_v13_{uni}.joblib")
        # Union of feats, preserving v13's superset order.
        union = list(dict.fromkeys(list(b13["feats"]) + list(b12["feats"])))
        ens = EnsembleClassifier(
            feats_full=union,
            gbc_a=b12["gbc"], feats_a=b12["feats"], cal_a=b12.get("calibrator"),
            gbc_b=b13["gbc"], feats_b=b13["feats"], cal_b=b13.get("calibrator"),
            weight=WEIGHT_V12,
        )
        out = MODELS / f"burst_gbc_v13_ens_{uni}.joblib"
        joblib.dump({
            "gbc": ens,
            "feats": union,
            "calibrator": IdentityCalibrator(),  # ens output already calibrated
            "version": "v13_ens",
            "universe": uni,
            "components": {"v12": str(MODELS / f"burst_gbc_v12_{uni}.joblib"),
                           "v13": str(MODELS / f"burst_gbc_v13_{uni}.joblib")},
            "weight_v12": WEIGHT_V12,
        }, out)
        print(f"  wrote {out.name}  feats_union={len(union)} (v12={len(b12['feats'])}, v13={len(b13['feats'])})")


if __name__ == "__main__":
    main()
