"""V6 — add an overnight-information feature for pre-market scoring.

The goal: at 09:15 ET on day t+1 (15 min before open), produce a burst
prediction that OUTPERFORMS the post-close-of-day-t scoring from v5.

Idea. At 09:15 ET on day t+1, pre-market already reflects two things that
day-t's close did NOT:
  1. After-hours / overnight news (earnings beats, guidance, M&A, macro)
  2. Pre-market trading (order flow from institutions that digested #1)

The single summary statistic of both is the **open gap** on day t+1 —
`Open[t+1] / Close[t] - 1`. This is the perfect historical proxy for the
pre-market price we'd observe at 09:15 ET. We use it in training and testing
and will substitute the live pre-market price in the inference script.

Head-to-head protocol (MUST outperform vanilla):
  vanilla:     v5 features (13 features)          -> Close[t] only
  augmented:   v5 + `overnight_gap` (14 features) -> adds the overnight info

We accept the augmented model only if:
  - test AUC strictly higher than vanilla, AND
  - test PR-AUC strictly higher than vanilla

Outputs:
  data/burst_panel_v6.csv
  models/burst_gbc_v6_vanilla.joblib
  models/burst_gbc_v6_augmented.joblib
  output/burst_metrics_v6.json
  output/burst_today_v6.csv   (scored with augmented, using the LATEST overnight
                                 gap we can observe — if after-hours data for
                                 any name is available, it's preferred over 0)
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output"

BURST_WINDOW = 5; BURST_MIN_LEN = 2; BURST_THRESH = 0.04

A_BASE = ["rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
          "atr_pct", "range_pct", "vol_z", "vol_5d"]
ASYM = ["skew_60d", "semivol_ratio_60d", "up_bigdays_60d"]
V5_FEATS = A_BASE + ["rv_60"] + ASYM           # 13
V6_FEATS = V5_FEATS + ["overnight_gap"]        # 14


def rsi(x, n=14):
    d = x.diff(); u = d.clip(lower=0).rolling(n).mean(); dd = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + u / dd.replace(0, np.nan))

def macd(x, f=12, s=26, sig=9):
    ef = x.ewm(span=f, adjust=False).mean(); es = x.ewm(span=s, adjust=False).mean()
    line = ef - es; signal = line.ewm(span=sig, adjust=False).mean()
    return line / x, signal / x, (line - signal) / x

def bb_z(x, n=20):
    return (x - x.rolling(n).mean()) / x.rolling(n).std()

def atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rolling_semivol_ratio(r, n=60):
    up_sq = (r.clip(lower=0) ** 2).rolling(n).mean()
    dn_sq = (r.clip(upper=0) ** 2).rolling(n).mean()
    return np.sqrt(up_sq) / np.sqrt(dn_sq.replace(0, np.nan))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c, h, l, v, o = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]
    out["rsi_14"] = rsi(c, 14)
    ml, ms, mh = macd(c); out["macd"], out["macd_sig"], out["macd_hist"] = ml, ms, mh
    out["bb_z20"] = bb_z(c, 20)
    out["atr_pct"] = atr(h, l, c, 14) / c
    out["range_pct"] = (h - l) / c
    vm = v.rolling(30).mean(); vs = v.rolling(30).std().replace(0, np.nan)
    out["vol_z"] = (v - vm) / vs
    out["vol_5d"] = v.rolling(5).mean() / vm
    r = c.pct_change()
    out["rv_60"] = r.rolling(60).std() * math.sqrt(252)
    out["skew_60d"] = r.rolling(60).skew()
    out["semivol_ratio_60d"] = rolling_semivol_ratio(r, 60)
    out["up_bigdays_60d"] = (r > 0.03).rolling(60).sum()

    # OVERNIGHT-GAP feature: Open[t+1] / Close[t] - 1, aligned to row t
    # i.e. "the gap we'll observe at 09:15 on the next session."
    out["overnight_gap"] = (o.shift(-1) / c) - 1.0
    return out


def build_target(c):
    r = c.pct_change().fillna(0).values
    n = len(r); y = np.zeros(n, dtype=np.int8)
    for t in range(n - BURST_WINDOW):
        fut = r[t+1:t+1+BURST_WINDOW]; best = 0.0
        for L in range(BURST_MIN_LEN, BURST_WINDOW + 1):
            for s in range(0, BURST_WINDOW - L + 1):
                m = fut[s:s+L].mean()
                if m > best: best = m
        if best >= BURST_THRESH: y[t] = 1
    out = pd.Series(y, index=c.index); out.iloc[-BURST_WINDOW:] = -1
    return out


def evaluate(y, p):
    base = float(np.mean(y)) if len(y) else float("nan")
    try: auc = roc_auc_score(y, p)
    except ValueError: auc = float("nan")
    try: ap = average_precision_score(y, p)
    except ValueError: ap = float("nan")
    ll = log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])
    bs = brier_score_loss(y, p)
    return {"n": int(len(y)), "pos": int(y.sum()), "base": base,
            "auc": float(auc), "ap": float(ap),
            "ap_lift": float(ap/base) if base > 0 else float("nan"),
            "log_loss": float(ll), "brier": float(bs)}


def train_and_eval(feats, tr, va, te):
    gbc = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=42)
    gbc.fit(tr[feats].values, tr["y"].values)
    p_tr = gbc.predict_proba(tr[feats].values)[:, 1]
    p_va = gbc.predict_proba(va[feats].values)[:, 1]
    p_te = gbc.predict_proba(te[feats].values)[:, 1]
    return gbc, {"train": evaluate(tr["y"].values, p_tr),
                 "val":   evaluate(va["y"].values, p_va),
                 "test":  evaluate(te["y"].values, p_te)}


def main():
    uni = pd.read_csv(DATA / "burst_universe_v5.csv")
    tickers = uni["ticker"].tolist()
    print(f"[v6] universe: {len(tickers)} upside-asymmetric SP500 names")

    print("[v6] bulk downloading 3y daily (Open+HLCV) ...")
    raw = yf.download(tickers, period="3y", interval="1d",
                      group_by="ticker", auto_adjust=True, threads=True,
                      progress=False)

    panels = []
    for t in tickers:
        try:
            sub = raw[t].dropna()
        except KeyError:
            continue
        if len(sub) < 120:
            continue
        feats = build_features(sub)
        feats["y"] = build_target(sub["Close"])
        feats["close"] = sub["Close"]
        feats["open_next"] = sub["Open"].shift(-1)  # kept for inspection
        feats["ticker"] = t
        feats = feats.reset_index().rename(columns={"Date": "date"})
        panels.append(feats)

    panel = pd.concat(panels, ignore_index=True)
    panel.to_csv(DATA / "burst_panel_v6.csv", index=False)
    print(f"[v6] panel rows: {len(panel):,}")

    lab = panel[panel["y"] >= 0].dropna(subset=V6_FEATS).reset_index(drop=True)
    print(f"[v6] labelled rows post-NA (requires overnight_gap): {len(lab):,}")
    print(f"[v6] base rate: {lab['y'].mean():.4%}")

    dates = np.sort(lab["date"].unique())
    d1 = dates[int(0.70 * len(dates))]; d2 = dates[int(0.85 * len(dates))]
    tr = lab[lab["date"] < d1]; va = lab[(lab["date"] >= d1) & (lab["date"] < d2)]; te = lab[lab["date"] >= d2]
    print(f"[v6] train/val/test: {len(tr):,} / {len(va):,} / {len(te):,}")

    print("\n[v6] === VANILLA (v5 features, no overnight gap) ===")
    gbc_v, met_v = train_and_eval(V5_FEATS, tr, va, te)
    print(f"    test: AUC={met_v['test']['auc']:.3f}  "
          f"AP={met_v['test']['ap']:.3f}  lift={met_v['test']['ap_lift']:.2f}x  "
          f"log_loss={met_v['test']['log_loss']:.4f}")

    print("\n[v6] === AUGMENTED (v5 + overnight_gap) ===")
    gbc_a, met_a = train_and_eval(V6_FEATS, tr, va, te)
    print(f"    test: AUC={met_a['test']['auc']:.3f}  "
          f"AP={met_a['test']['ap']:.3f}  lift={met_a['test']['ap_lift']:.2f}x  "
          f"log_loss={met_a['test']['log_loss']:.4f}")

    auc_gain = met_a["test"]["auc"] - met_v["test"]["auc"]
    ap_gain = met_a["test"]["ap"] - met_v["test"]["ap"]
    ll_gain = met_v["test"]["log_loss"] - met_a["test"]["log_loss"]
    print(f"\n[v6] delta (aug - vanilla): "
          f"AUC={auc_gain:+.4f}  AP={ap_gain:+.4f}  log_loss_reduction={ll_gain:+.4f}")

    passes = auc_gain > 0 and ap_gain > 0
    verdict = "ACCEPT (augmented beats vanilla on both AUC and AP)" if passes else \
              "REJECT (augmented did NOT strictly beat vanilla)"
    print(f"[v6] verdict: {verdict}")

    joblib.dump({"gbc": gbc_v, "feats": V5_FEATS}, MODELS / "burst_gbc_v6_vanilla.joblib")
    joblib.dump({"gbc": gbc_a, "feats": V6_FEATS}, MODELS / "burst_gbc_v6_augmented.joblib")
    json.dump({
        "vanilla": met_v, "augmented": met_a,
        "auc_gain": auc_gain, "ap_gain": ap_gain, "log_loss_reduction": ll_gain,
        "passes_vs_vanilla": bool(passes),
        "vanilla_feats": V5_FEATS, "augmented_feats": V6_FEATS,
    }, open(OUT / "burst_metrics_v6.json", "w"), indent=2)

    # Feature importances
    print("\n[v6] augmented feature importance (top 14):")
    imp = pd.Series(gbc_a.feature_importances_, index=V6_FEATS).sort_values(ascending=False)
    print(imp.to_string())

    # Score today: use the latest row per ticker with a known overnight_gap.
    # For historical rows (overnight_gap = next_open/close-1) this is the normal feature.
    # For LIVE (Saturday), use the most recent row that still has an Open_next (i.e.
    # we can't yet observe Monday's open, so we fall back to a vanilla-only score
    # and leave the live-inference script to override with pre-market data).
    scored = panel.dropna(subset=V5_FEATS).copy()   # vanilla row existence
    latest = scored.sort_values(["ticker", "date"]).groupby("ticker").tail(1).copy()
    # vanilla prob (always available)
    latest["prob_vanilla"] = gbc_v.predict_proba(latest[V5_FEATS].values)[:, 1]
    # augmented prob — set overnight_gap = 0 as a neutral placeholder when unknown
    aug_rows = latest.copy()
    aug_rows["overnight_gap"] = aug_rows["overnight_gap"].fillna(0.0)
    latest["prob_augmented_zero_gap"] = gbc_a.predict_proba(aug_rows[V6_FEATS].values)[:, 1]

    # Also sensitivity to a +3% hypothetical gap (for reference)
    sens_rows = latest.copy(); sens_rows["overnight_gap"] = 0.03
    latest["prob_augmented_gap+3pct"] = gbc_a.predict_proba(sens_rows[V6_FEATS].values)[:, 1]

    latest = latest.merge(uni[["ticker", "sector", "reasons"]], on="ticker", how="left")
    latest = latest.sort_values("prob_augmented_zero_gap", ascending=False).reset_index(drop=True)
    out_cols = ["ticker", "date", "close", "prob_vanilla",
                "prob_augmented_zero_gap", "prob_augmented_gap+3pct",
                "rv_60", "rsi_14", "bb_z20", "skew_60d", "semivol_ratio_60d",
                "up_bigdays_60d", "sector", "reasons"]
    latest[out_cols].to_csv(OUT / "burst_today_v6.csv", index=False)
    print(f"\n[v6] wrote output/burst_today_v6.csv with {len(latest)} rows")
    print("\n[v6] top 20 (augmented w/ zero overnight gap):")
    print(latest[out_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
