"""Predict today's stocks most likely to touch +50% in next 21 days.

Trains the same v1 pipeline (23 features) but with the y21_50 label
(max[t+1..t+21]/close[t] >= 1.50). Then scores the latest panel rows
and emits top-K picks.

Note: the +50% model has AUC 0.941 but very low absolute hit rate
(p@top-5 was 0.9% in the multi-horizon test fold, base rate 0.38%).
Treat these as a ranked watchlist of "extreme upside potential" rather
than as confident probabilities.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output" / "monthly_gainer"

sys.path.insert(0, str(ROOT / "code"))
from regime_features import REGIME_FEATS, attach_regime  # noqa: E402

V1_FEATURES = [
    "rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
    "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60",
    "ma_stack", "up_streak", "up_bigdays_20d",
    "dist_ma60_atr", "ma60_slope_60d", "run_length",
] + REGIME_FEATS


def add_y21_50(panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for tk, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True).copy()
        c = g["close"].values.astype(float)
        n = len(c)
        y = np.full(n, -1, dtype=np.int8)
        for t in range(n - 1):
            ct = c[t]
            if ct <= 0 or not np.isfinite(ct):
                continue
            if t + 21 < n:
                ret = c[t+1:t+22].max() / ct - 1.0
                y[t] = 1 if ret >= 0.50 else 0
        g["y21_50"] = y
        out.append(g)
    return pd.concat(out, ignore_index=True)


def time_split(df, train_frac=0.70, val_frac=0.15):
    dates = np.sort(df["date"].unique())
    t1 = dates[int(len(dates) * train_frac)]
    t2 = dates[int(len(dates) * (train_frac + val_frac))]
    return (df[df["date"] < t1].copy(),
            df[(df["date"] >= t1) & (df["date"] < t2)].copy(),
            df[df["date"] >= t2].copy())


def sample_weights(y, target_pos_frac=0.10):
    p = y.mean()
    if p <= 0 or p >= 1:
        return np.ones(len(y))
    w_pos = target_pos_frac / p
    w_neg = (1 - target_pos_frac) / (1 - p)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def train_and_score(panel_path: Path, label: str, k: int = 20):
    print(f"\n[91] === {label} ===")
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    panel = attach_regime(panel)
    panel = add_y21_50(panel)
    print(f"[91] panel: {len(panel):,} rows  pos_share(labeled): {(panel['y21_50']==1).sum() / max(1,(panel['y21_50']>=0).sum()):.3%}")

    # 5d trailing return for diagnostics
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["close_5d_ago"] = panel.groupby("ticker")["close"].shift(5)
    panel["ret_5d_lag"] = panel["close"] / panel["close_5d_ago"] - 1.0
    panel["close_20d_ago"] = panel.groupby("ticker")["close"].shift(20)
    panel["ret_20d_lag"] = panel["close"] / panel["close_20d_ago"] - 1.0

    df = panel[panel["y21_50"] >= 0].dropna(subset=V1_FEATURES).reset_index(drop=True)
    train, val, test = time_split(df)
    medians = train[V1_FEATURES].median(numeric_only=True)
    Xtr = train[V1_FEATURES].fillna(medians).values
    Xva = val[V1_FEATURES].fillna(medians).values
    Xte = test[V1_FEATURES].fillna(medians).values
    ytr = train["y21_50"].values.astype(np.int8)
    yva = val["y21_50"].values.astype(np.int8)
    yte = test["y21_50"].values.astype(np.int8)
    print(f"[91] split: train={len(train):,} val={len(val):,} test={len(test):,} "
          f"test_base={yte.mean():.3%}")

    sw = sample_weights(ytr, target_pos_frac=min(0.20, max(0.04, ytr.mean() * 5)))
    gbc = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                                     subsample=0.8, random_state=42)
    gbc.fit(Xtr, ytr, sample_weight=sw)
    cal = CalibratedClassifierCV(estimator=gbc, method="isotonic", cv="prefit")
    cal.fit(Xva, yva)

    p_te = cal.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p_te)) if len(np.unique(yte)) > 1 else float("nan")
    ap = float(average_precision_score(yte, p_te)) if yte.sum() > 0 else float("nan")
    print(f"[91] test AUC={auc:.3f}  PR-AUC={ap:.4f}  base={yte.mean():.3%}  "
          f"lift={ap/max(yte.mean(),1e-9):.2f}x")

    # Score the FULL panel (including last-21-days unlabeled rows) for today's picks
    full_complete = panel.dropna(subset=V1_FEATURES).copy()
    Xfull = full_complete[V1_FEATURES].fillna(medians).values
    full_complete["prob_y50"] = cal.predict_proba(Xfull)[:, 1]

    last_d = full_complete["date"].max()
    today = full_complete[full_complete["date"] == last_d].copy()
    print(f"\n[91] === {label}: top-{k} +50%-touch picks for {last_d.date()} ===")
    cols = ["ticker", "prob_y50", "rv_60", "atr_pct", "run_length",
            "ret_5d_lag", "ret_20d_lag", "close"]
    if "sector" in today.columns:
        cols.insert(2, "sector")
    top = today.nlargest(k, "prob_y50")[cols]
    print(top.to_string(index=False, formatters={
        "prob_y50": "{:.3f}".format,
        "rv_60": "{:.2f}".format,
        "atr_pct": "{:.2%}".format,
        "ret_5d_lag": "{:+.1%}".format,
        "ret_20d_lag": "{:+.1%}".format,
        "close": "{:.2f}".format,
    }))
    return today.sort_values("prob_y50", ascending=False)


def main():
    sp = train_and_score(DATA / "monthly_gainer_panel.csv", "SP500", k=20)
    sc = train_and_score(DATA / "monthly_gainer_panel_smallcap.csv", "SMALLCAP", k=20)
    OUT.mkdir(parents=True, exist_ok=True)
    sp.to_csv(OUT / "today_picks_sp500_y50.csv", index=False)
    sc.to_csv(OUT / "today_picks_smallcap_y50.csv", index=False)
    print(f"\n[91] saved {OUT / 'today_picks_sp500_y50.csv'}")
    print(f"[91] saved {OUT / 'today_picks_smallcap_y50.csv'}")


if __name__ == "__main__":
    main()
