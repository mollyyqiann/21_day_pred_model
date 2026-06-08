"""Build the prob-bucket historical-mean lookup table for v13_ens.

Replaces the noisy regressor pred_Nd display in the morning message with
"at this cal_prob bucket, historical 3d / 5d realized return is X / Y%".
This is honest about what the model knows: the classifier ranks well, the
regressor predictions are zero-skill noise.

Computed on the test split (last 15% of dates) for each universe x cal_prob
bucket. Stores both mean realized and win rate so the display can show both.

Output: models/burst_v13_ens_horizon_lookup.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "code"))
from regime_features import attach_regime  # noqa
import importlib.util
spec = importlib.util.spec_from_file_location("v13", str(ROOT/"code/77_burst_v13_intraday_features.py"))
v13mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(v13mod)

V7=["rsi_14","macd","macd_sig","macd_hist","bb_z20","atr_pct","range_pct","vol_z","vol_5d","rv_60","overnight_gap"]

# Buckets — finer at high prob since that's where picks live.
BUCKETS = [(0.0, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
           (0.50, 0.60), (0.60, 0.70), (0.70, 1.01)]


def main():
    ohlc = pd.read_csv(DATA/"ohlc_prices.csv", parse_dates=["date"])
    ohlc["date"] = ohlc["date"].dt.normalize()
    yi = v13mod._build_yest_intra(ohlc)

    out = {"version": "v13_ens", "buckets": [(lo, hi) for lo, hi in BUCKETS],
           "horizons": ["3d", "5d"], "per_universe": {}}

    for uni, panel in [("v4", "burst_panel_v6b.csv"), ("v5", "burst_panel_v6.csv"), ("v7", "burst_panel_v7.csv")]:
        p = pd.read_csv(DATA/panel, parse_dates=["date"])
        p = p[p[V7].notna().all(axis=1)].copy()
        p = attach_regime(p)
        p = v13mod.attach_ohlc_yest_targets(p, yi)
        p = v13mod.attach_intraday_minute_feats(p)
        b = joblib.load(MODELS / f"burst_gbc_v13_ens_{uni}.joblib")
        for c in b["feats"]:
            if c not in p.columns: p[c] = np.nan
        pr = b["gbc"].predict_proba(p[b["feats"]].values)[:, 1]
        p["cal_p"] = pr
        # Test slice: last 15% of dates (matches v12/v13 training split)
        dates = np.sort(p["date"].unique())
        d2 = pd.Timestamp(dates[int(0.85*len(dates))])
        te = p[(p["date"] >= d2)].dropna(subset=["fwd_intra_3d", "fwd_intra_5d", "cal_p"]).copy()

        per_bucket = []
        for lo, hi in BUCKETS:
            sub = te[(te["cal_p"] >= lo) & (te["cal_p"] < hi)]
            if len(sub) == 0:
                per_bucket.append({"lo": lo, "hi": hi, "n": 0,
                                    "3d_mean": None, "3d_wr": None,
                                    "5d_mean": None, "5d_wr": None})
                continue
            per_bucket.append({
                "lo": lo, "hi": hi, "n": int(len(sub)),
                "3d_mean": float(sub["fwd_intra_3d"].mean()),
                "3d_wr":   float((sub["fwd_intra_3d"] > 0).mean()),
                "5d_mean": float(sub["fwd_intra_5d"].mean()),
                "5d_wr":   float((sub["fwd_intra_5d"] > 0).mean()),
            })
        out["per_universe"][uni] = per_bucket
        print(f"  {uni}: {len(te):,} test rows  ->  bucket counts: " +
              " ".join(f"{b['n']}" for b in per_bucket))

    out_path = MODELS / "burst_v13_ens_horizon_lookup.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
