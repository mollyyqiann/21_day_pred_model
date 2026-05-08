"""Risk-reduction sweep on the v3 test fold.

For each top-K daily basket, walks the actual close-price path over 21
trading days and simulates several exit rules:

  baseline       : hold to end of 21d window
  TP_only(X)     : first touch of +X%, sell at +X. Else hold to end.
  SL_only(Y)     : first touch of -Y%, sell at -Y. Else hold to end.
  TP+SL(X,Y)     : whichever triggers first along the daily path.
  K-larger       : K=10, K=20 to cut tail variance via diversification.
  regime         : require SPY 20d > 0 to enter (no buy when market down).

Reports per-config: hit rate, mean/median end ret, std, Sharpe, big-loss
rate, max-DD, n_picks. Each universe (SP500, smallcap) under raw_margin.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output" / "monthly_gainer"

sys.path.insert(0, str(ROOT / "code"))
from regime_features import attach_regime  # noqa: E402

XRANK_FEATURES = ["rsi_14_xrank", "rv_60_xrank", "ma60_slope_xrank", "ret_20d_xrank"]


def add_xrank(df):
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["close_20d_ago"] = df.groupby("ticker")["close"].shift(20)
    df["ret_20d_lag"] = df["close"] / df["close_20d_ago"] - 1.0
    df["close_5d_ago"] = df.groupby("ticker")["close"].shift(5)
    df["ret_5d_lag"] = df["close"] / df["close_5d_ago"] - 1.0
    df["rsi_14_xrank"] = df.groupby("date")["rsi_14"].rank(pct=True)
    df["rv_60_xrank"] = df.groupby("date")["rv_60"].rank(pct=True)
    df["ma60_slope_xrank"] = df.groupby("date")["ma60_slope_60d"].rank(pct=True)
    df["ret_20d_xrank"] = df.groupby("date")["ret_20d_lag"].rank(pct=True)
    return df


def score_test_fold(panel_path, cat_path, model_path):
    art = joblib.load(model_path)
    gbc = art["raw_gbc"]; cal = art["calibrator"]
    feats = art["feats"]; med = pd.Series(art["impute_medians"])
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    cat = pd.read_csv(cat_path, parse_dates=["date"])
    panel = panel.merge(cat, on=["ticker", "date"], how="left")
    panel = attach_regime(panel)
    panel = add_xrank(panel)
    for c in ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
              "news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
              "ma_news_5d", "ma_news_20d", "sector_pop_5d"]:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0)
        else:
            panel[c] = 0.0
    for c in XRANK_FEATURES:
        panel[c] = panel[c].fillna(0.5)

    lab = panel[panel["y21"].isin([0, 1])].dropna(subset=feats).copy()
    dates = np.sort(lab["date"].unique())
    test_start = dates[int(len(dates) * 0.85)]
    test = lab[lab["date"] >= test_start].copy()
    X = test[feats].fillna(med).values
    test["prob_cal"] = cal.predict_proba(X)[:, 1]
    test["raw_margin"] = gbc.decision_function(X)
    return test, panel


def build_path_lookup(panel: pd.DataFrame) -> dict:
    """Map (ticker, date) -> ndarray of close[t+1..t+21] (length 21)."""
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    paths = {}
    for tk, g in panel.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        c = g["close"].values
        d = g["date"].values
        n = len(c)
        for t in range(n):
            if t + 21 < n and c[t] > 0 and np.isfinite(c[t]):
                paths[(tk, d[t])] = c[t + 1:t + 22] / c[t] - 1.0
    return paths


def simulate_pick(path: np.ndarray, tp: float | None = None,
                   sl: float | None = None) -> float:
    """Simulate exit rules along a 21d return path (length 21).
    Returns realized return. tp=0.20 means take-profit at +20%; sl=0.15
    means stop-loss at -15%."""
    end_ret = path[-1]
    if tp is None and sl is None:
        return float(end_ret)
    for r in path:
        if sl is not None and r <= -sl:
            return float(-sl)
        if tp is not None and r >= tp:
            return float(tp)
    return float(end_ret)


def topk_basket(test: pd.DataFrame, score_col: str, k: int,
                regime_filter: bool = False) -> pd.DataFrame:
    if regime_filter:
        test = test[test["spy_ret_20d"] > 0].copy()
    rows = []
    for d, g in test.groupby("date"):
        if len(g) < k:
            continue
        topk = g.nlargest(k, score_col)
        for _, r in topk.iterrows():
            rows.append({
                "date": d, "ticker": r["ticker"],
                "score": r[score_col], "y21": r["y21"],
                "max_fwd": r["max_fwd21_ret"], "end": r["end_of_window_ret"],
            })
    return pd.DataFrame(rows)


def stats(returns: pd.Series, label: str) -> dict:
    return {
        "config": label,
        "n": len(returns),
        "hit": float((returns > 0).mean()),
        "mean": float(returns.mean()),
        "median": float(returns.median()),
        "std": float(returns.std()),
        "sharpe": float(returns.mean() / returns.std()) if returns.std() > 0 else float("nan"),
        "loser": float((returns < 0).mean()),
        "big_loss": float((returns < -0.15).mean()),
        "min": float(returns.min()),
        "max": float(returns.max()),
    }


def run_universe(test, paths, label, score_col="raw_margin"):
    print(f"\n[96] === {label} ({score_col}) ===")
    configs = [
        # (tag, k, tp, sl, regime)
        ("baseline_k5", 5, None, None, False),
        ("baseline_k10", 10, None, None, False),
        ("baseline_k20", 20, None, None, False),
        ("TP20_k5", 5, 0.20, None, False),
        ("TP25_k5", 5, 0.25, None, False),
        ("TP30_k5", 5, 0.30, None, False),
        ("SL10_k5", 5, None, 0.10, False),
        ("SL15_k5", 5, None, 0.15, False),
        ("SL20_k5", 5, None, 0.20, False),
        ("TP20_SL10_k5", 5, 0.20, 0.10, False),
        ("TP25_SL10_k5", 5, 0.25, 0.10, False),
        ("TP30_SL15_k5", 5, 0.30, 0.15, False),
        ("TP25_SL10_k10", 10, 0.25, 0.10, False),
        ("TP25_SL15_k10", 10, 0.25, 0.15, False),
        ("regime_TP25_SL10_k5", 5, 0.25, 0.10, True),
        ("regime_TP25_SL15_k10", 10, 0.25, 0.15, True),
    ]
    rows = []
    for tag, k, tp, sl, regime in configs:
        picks = topk_basket(test, score_col, k, regime)
        if len(picks) == 0:
            print(f"   {tag}: no qualifying picks")
            continue
        rets = []
        for _, p in picks.iterrows():
            d = p["date"].to_datetime64() if hasattr(p["date"], "to_datetime64") else np.datetime64(p["date"])
            path = paths.get((p["ticker"], d))
            if path is None:
                continue
            rets.append(simulate_pick(path, tp=tp, sl=sl))
        if not rets:
            continue
        s = stats(pd.Series(rets), tag)
        rows.append(s)
        print(f"   {tag:<28} n={s['n']:>4}  hit={s['hit']:.0%}  "
              f"mean={s['mean']:+.1%}  med={s['median']:+.1%}  "
              f"std={s['std']:.1%}  sharpe={s['sharpe']:.2f}  "
              f"big_loss={s['big_loss']:.1%}  min={s['min']:+.0%}")
    return pd.DataFrame(rows)


def main():
    print("[96] loading + scoring SP500 ...")
    sp_test, sp_panel = score_test_fold(
        DATA / "monthly_gainer_panel.csv",
        DATA / "catalyst_features_sp500.csv",
        MODELS / "monthly_gainer_v3_sp500.joblib")
    print(f"   sp500 test: {len(sp_test)}")

    print("[96] building SP500 path lookup ...")
    sp_paths = build_path_lookup(sp_panel)
    print(f"   {len(sp_paths)} (ticker,date) paths cached")

    print("[96] loading + scoring SMALLCAP ...")
    sc_test, sc_panel = score_test_fold(
        DATA / "monthly_gainer_panel_smallcap.csv",
        DATA / "catalyst_features_smallcap.csv",
        MODELS / "monthly_gainer_v3_smallcap.joblib")
    print(f"   smallcap test: {len(sc_test)}")

    print("[96] building SMALLCAP path lookup ...")
    sc_paths = build_path_lookup(sc_panel)
    print(f"   {len(sc_paths)} (ticker,date) paths cached")

    sp_results = run_universe(sp_test, sp_paths, "SP500", score_col="raw_margin")
    sc_results = run_universe(sc_test, sc_paths, "SMALLCAP", score_col="raw_margin")

    OUT.mkdir(parents=True, exist_ok=True)
    sp_results.to_csv(OUT / "risk_reduction_sp500.csv", index=False)
    sc_results.to_csv(OUT / "risk_reduction_smallcap.csv", index=False)
    print(f"\n[96] saved {OUT / 'risk_reduction_sp500.csv'} and smallcap.csv")


if __name__ == "__main__":
    main()
