"""More risk-reduction levers — properly daily-basket-aggregated.

Levers tested (on top of the existing SP500 best: regime + TP25 + SL10 + K=5):
  1. TRAILING STOP — sell at running-max minus X. Captures more upside.
  2. VIX FILTER     — only trade when VIX < threshold (regime + VIX)
  3. EARNINGS EXCL  — skip picks with earn_news_5d == 1
  4. POSITION SIZING — softmax over raw_margin (T=0.5 / 1.0); concentrate
                       capital on stronger signals.

For each config, computes BOTH per-pick Sharpe (sanity) and the
proper PORTFOLIO Sharpe (mean / std of DAILY BASKET RETURNS) — the
metric you'd actually compound on.
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
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    paths = {}
    for tk, g in panel.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        c = g["close"].values; d = g["date"].values
        n = len(c)
        for t in range(n):
            if t + 21 < n and c[t] > 0 and np.isfinite(c[t]):
                paths[(tk, d[t])] = c[t + 1:t + 22] / c[t] - 1.0
    return paths


def simulate_pick(path, tp=None, sl=None, ts=None):
    """Walk path day by day. Order of priority: SL, TP, trailing.
    tp = take-profit (e.g. 0.25), sl = stop-loss (0.10), ts = trailing-stop (0.10).
    Returns realized return."""
    running_max = 0.0
    for r in path:
        if r > running_max:
            running_max = r
        # Hard SL has priority — never let losses exceed -sl
        if sl is not None and r <= -sl:
            return float(-sl)
        # Take-profit
        if tp is not None and r >= tp:
            return float(tp)
        # Trailing stop only kicks in once running_max is positive
        if ts is not None and running_max > 0 and r <= running_max - ts:
            return float(running_max - ts)
    return float(path[-1])


def softmax(x, T=1.0):
    x = np.asarray(x) / T
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def daily_basket_returns(picks_per_day: dict, weighting: str, T: float = 1.0):
    """Aggregate per-pick returns into a daily basket return.
    picks_per_day: dict[date] -> list[(margin, realized_return)]
    weighting: 'equal' or 'softmax'
    """
    rows = []
    for d, items in picks_per_day.items():
        margins = np.array([m for m, _ in items])
        rets = np.array([r for _, r in items])
        if weighting == "equal":
            w = np.ones_like(rets) / len(rets)
        elif weighting == "softmax":
            w = softmax(margins, T=T)
        else:
            raise ValueError(weighting)
        rows.append({"date": d, "ret": float((w * rets).sum()),
                     "n": len(rets), "max_w": float(w.max()),
                     "min_w": float(w.min())})
    return pd.DataFrame(rows)


def basket_stats(daily: pd.DataFrame, label: str) -> dict:
    rets = daily["ret"]
    return {
        "config": label,
        "n_days": len(daily),
        "mean": float(rets.mean()),
        "median": float(rets.median()),
        "std": float(rets.std()),
        "sharpe": float(rets.mean() / rets.std()) if rets.std() > 0 else float("nan"),
        "max_w_avg": float(daily["max_w"].mean()),
        "n_per_day": float(daily["n"].mean()),
        "min_basket": float(rets.min()),
        "max_basket": float(rets.max()),
        "loss_days": float((rets < 0).mean()),
        "big_loss_days": float((rets < -0.05).mean()),  # any day basket loses >5%
    }


def run_config(test, paths, label, k=5, score_col="raw_margin",
                tp=None, sl=None, ts=None,
                regime_spy=False, vix_max=None, exclude_earnings=False,
                weighting="equal", soft_T=1.0):
    """Build picks under filters, compute realized returns, aggregate per day."""
    df = test.copy()
    if regime_spy:
        df = df[df["spy_ret_20d"] > 0]
    if vix_max is not None:
        df = df[df["vix"] < vix_max]
    if exclude_earnings and "earn_news_5d" in df.columns:
        df = df[df["earn_news_5d"] < 0.5]

    picks_per_day = {}
    for d, g in df.groupby("date"):
        if len(g) < k:
            continue
        topk = g.nlargest(k, score_col)
        items = []
        for _, r in topk.iterrows():
            dt = r["date"].to_datetime64() if hasattr(r["date"], "to_datetime64") else np.datetime64(r["date"])
            path = paths.get((r["ticker"], dt))
            if path is None:
                continue
            real = simulate_pick(path, tp=tp, sl=sl, ts=ts)
            items.append((float(r[score_col]), real))
        if items:
            picks_per_day[d] = items
    if not picks_per_day:
        return None
    daily = daily_basket_returns(picks_per_day, weighting=weighting, T=soft_T)
    return basket_stats(daily, label)


def main():
    print("[97] loading SP500 ...")
    sp_test, sp_panel = score_test_fold(
        DATA / "monthly_gainer_panel.csv",
        DATA / "catalyst_features_sp500.csv",
        MODELS / "monthly_gainer_v3_sp500.joblib")
    sp_paths = build_path_lookup(sp_panel)
    print(f"   sp500 test rows={len(sp_test)} paths={len(sp_paths)}")

    print("[97] loading SMALLCAP ...")
    sc_test, sc_panel = score_test_fold(
        DATA / "monthly_gainer_panel_smallcap.csv",
        DATA / "catalyst_features_smallcap.csv",
        MODELS / "monthly_gainer_v3_smallcap.joblib")
    sc_paths = build_path_lookup(sc_panel)
    print(f"   smallcap test rows={len(sc_test)} paths={len(sc_paths)}")

    # ===== SP500 — sweep new levers on top of regime+TP25+SL10+k5 =====
    sp_configs = [
        # (label, k, tp, sl, ts, regime_spy, vix_max, excl_earn, weighting, T)
        ("SP_baseline",                       5, None, None, None,  False, None, False, "equal",   1.0),
        ("SP_REC_TP25_SL10",                  5, 0.25, 0.10, None,  True,  None, False, "equal",   1.0),  # current best
        ("SP_REC_TP25_TS10",                  5, 0.25, None, 0.10,  True,  None, False, "equal",   1.0),  # trailing instead of fixed
        ("SP_REC_TP25_SL10_TS10",             5, 0.25, 0.10, 0.10,  True,  None, False, "equal",   1.0),  # both
        ("SP_REC_TS10_only",                  5, None, None, 0.10,  True,  None, False, "equal",   1.0),  # trailing only
        ("SP_REC_TP25_SL10_VIX25",            5, 0.25, 0.10, None,  True,  25.0, False, "equal",   1.0),
        ("SP_REC_TP25_SL10_VIX22",            5, 0.25, 0.10, None,  True,  22.0, False, "equal",   1.0),
        ("SP_REC_TP25_SL10_NoEarn",           5, 0.25, 0.10, None,  True,  None, True,  "equal",   1.0),
        ("SP_REC_TP25_SL10_VIX25_NoEarn",     5, 0.25, 0.10, None,  True,  25.0, True,  "equal",   1.0),
        ("SP_REC_TP25_SL10_softmaxT1",        5, 0.25, 0.10, None,  True,  None, False, "softmax", 1.0),
        ("SP_REC_TP25_SL10_softmaxT0.5",      5, 0.25, 0.10, None,  True,  None, False, "softmax", 0.5),
        ("SP_REC_TP25_SL10_softmaxT0.3",      5, 0.25, 0.10, None,  True,  None, False, "softmax", 0.3),
        # ALL levers combined
        ("SP_REC_TP25_TS10_VIX25_NoEarn_smT0.5", 5, 0.25, None, 0.10, True, 25.0, True, "softmax", 0.5),
        ("SP_REC_TP25_SL10_TS10_VIX25_NoEarn_smT0.5", 5, 0.25, 0.10, 0.10, True, 25.0, True, "softmax", 0.5),
    ]

    print("\n[97] === SP500 ===")
    sp_rows = []
    for cfg in sp_configs:
        s = run_config(sp_test, sp_paths, cfg[0], k=cfg[1],
                        tp=cfg[2], sl=cfg[3], ts=cfg[4],
                        regime_spy=cfg[5], vix_max=cfg[6], exclude_earnings=cfg[7],
                        weighting=cfg[8], soft_T=cfg[9])
        if s is None:
            continue
        sp_rows.append(s)
        print(f"   {s['config']:<46} days={s['n_days']:>3}  "
              f"mean={s['mean']:+.2%}  med={s['median']:+.2%}  "
              f"std={s['std']:.2%}  sharpe={s['sharpe']:.2f}  "
              f"big_loss_days={s['big_loss_days']:.0%}  min={s['min_basket']:+.0%}")

    # ===== SMALLCAP — see if any combo beats SP500's mean =====
    sc_configs = [
        ("SC_baseline",                       5, None, None, None,  False, None, False, "equal",   1.0),
        ("SC_TP30",                           5, 0.30, None, None,  False, None, False, "equal",   1.0),  # current best
        ("SC_TP30_TS10",                      5, 0.30, None, 0.10,  False, None, False, "equal",   1.0),
        ("SC_TP30_TS15",                      5, 0.30, None, 0.15,  False, None, False, "equal",   1.0),
        ("SC_TP30_TS20",                      5, 0.30, None, 0.20,  False, None, False, "equal",   1.0),
        ("SC_TS15_only",                      5, None, None, 0.15,  False, None, False, "equal",   1.0),
        ("SC_TP30_VIX25",                     5, 0.30, None, None,  False, 25.0, False, "equal",   1.0),
        ("SC_TP30_NoEarn",                    5, 0.30, None, None,  False, None, True,  "equal",   1.0),
        ("SC_TP30_softmaxT1",                 5, 0.30, None, None,  False, None, False, "softmax", 1.0),
        ("SC_TP30_softmaxT0.5",               5, 0.30, None, None,  False, None, False, "softmax", 0.5),
        ("SC_TP30_softmaxT0.3",               5, 0.30, None, None,  False, None, False, "softmax", 0.3),
        ("SC_TP30_TS15_softmaxT0.5",          5, 0.30, None, 0.15,  False, None, False, "softmax", 0.5),
        # max-mean push
        ("SC_baseline_softmaxT0.3",           5, None, None, None,  False, None, False, "softmax", 0.3),
        ("SC_TP50_softmaxT0.5",               5, 0.50, None, None,  False, None, False, "softmax", 0.5),
        ("SC_TP50",                           5, 0.50, None, None,  False, None, False, "equal",   1.0),
    ]

    print("\n[97] === SMALLCAP ===")
    sc_rows = []
    for cfg in sc_configs:
        s = run_config(sc_test, sc_paths, cfg[0], k=cfg[1],
                        tp=cfg[2], sl=cfg[3], ts=cfg[4],
                        regime_spy=cfg[5], vix_max=cfg[6], exclude_earnings=cfg[7],
                        weighting=cfg[8], soft_T=cfg[9])
        if s is None:
            continue
        sc_rows.append(s)
        print(f"   {s['config']:<46} days={s['n_days']:>3}  "
              f"mean={s['mean']:+.2%}  med={s['median']:+.2%}  "
              f"std={s['std']:.2%}  sharpe={s['sharpe']:.2f}  "
              f"big_loss_days={s['big_loss_days']:.0%}  min={s['min_basket']:+.0%}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sp_rows).to_csv(OUT / "risk_levers_sp500.csv", index=False)
    pd.DataFrame(sc_rows).to_csv(OUT / "risk_levers_smallcap.csv", index=False)
    print(f"\n[97] saved 2 files in {OUT}")


if __name__ == "__main__":
    main()
