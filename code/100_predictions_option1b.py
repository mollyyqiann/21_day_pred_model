"""Option 1B prediction lists for Telegram.

Strategy:
  ENTRY: top-15 v3 raw_margin daily (SPY 20d > 0 regime ON)
  HOLD:  while prob_cal >= 0.05
  CAP:   60 days max

For each pick today, attach:
  - raw_margin (entry signal strength)
  - prob_cal (used for hold/exit)
  - 5d / 20d ret (current growth)
  - atr_pct, dd_60d, run_length (volatility context)
  - Option 1B per-prob-bucket projections (mean / median realized
    return on TEST FOLD with this exact strategy)

Splits the 15 into:
  - PARTIAL WINNERS: 5d_ret in [+5%, +30%) -- already running
  - ABOUT TO RISE  : |5d_ret| < 5% -- not yet moving
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
from extension_classifier import attach_extension  # noqa: E402

XRANK_FEATURES = ["rsi_14_xrank", "rv_60_xrank", "ma60_slope_xrank", "ret_20d_xrank"]
PROB_BUCKETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.01]


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


def score_full_panel(panel_path, cat_path, model_path):
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

    # 60d max drawdown
    panel["max_60d"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.rolling(60, min_periods=20).max())
    panel["min_60d"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.rolling(60, min_periods=20).min())
    panel["dd_60d"] = panel["min_60d"] / panel["max_60d"] - 1.0

    full = panel.dropna(subset=feats).copy()
    X = full[feats].fillna(med).values
    full["prob_cal"] = cal.predict_proba(X)[:, 1]
    full["raw_margin"] = gbc.decision_function(X)
    return full.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_lookup(scores: pd.DataFrame) -> dict:
    out = {}
    for tk, g in scores.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        out[tk] = {
            "dates": g["date"].values,
            "close": g["close"].values,
            "prob": g["prob_cal"].values,
        }
    return out


def simulate_hold(lookup_t, entry_idx: int, threshold=0.05, max_hold=60):
    close = lookup_t["close"]; prob = lookup_t["prob"]
    n = len(close)
    entry_close = close[entry_idx]
    if not np.isfinite(entry_close) or entry_close <= 0:
        return None
    for offset in range(1, max_hold + 1):
        i = entry_idx + offset
        if i >= n:
            return float(close[n - 1] / entry_close - 1.0), n - 1 - entry_idx, "eod"
        if prob[i] < threshold:
            return float(close[i] / entry_close - 1.0), offset, "signal"
    i = min(entry_idx + max_hold, n - 1)
    return float(close[i] / entry_close - 1.0), max_hold, "max_hold"


def option1b_calibration_table(scores: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Run Option 1B on test fold, bucket realized return by prob_cal at entry."""
    dates = np.sort(scores["date"].unique())
    test_start = dates[int(len(dates) * 0.85)]
    test = scores[scores["date"] >= test_start]
    test = test[test["spy_ret_20d"] > 0]

    rows = []
    for d, g in test.groupby("date"):
        if len(g) < 5:
            continue
        topk = g.nlargest(5, "raw_margin")
        for _, r in topk.iterrows():
            tk = r["ticker"]
            if tk not in lookup:
                continue
            idx = np.where(lookup[tk]["dates"] == np.datetime64(d))[0]
            if len(idx) == 0:
                continue
            res = simulate_hold(lookup[tk], int(idx[0]), threshold=0.05, max_hold=60)
            if res is None:
                continue
            ret, hold, reason = res
            rows.append({"prob_cal": r["prob_cal"],
                          "raw_margin": r["raw_margin"],
                          "ret": ret, "hold_days": hold})
    df = pd.DataFrame(rows)
    df["bucket"] = pd.cut(df["prob_cal"], bins=PROB_BUCKETS,
                            labels=[f"[{PROB_BUCKETS[i]:.2f},{PROB_BUCKETS[i+1]:.2f})"
                                     for i in range(len(PROB_BUCKETS) - 1)],
                            include_lowest=True)
    tab = df.groupby("bucket", observed=True).agg(
        n=("ret", "size"),
        prob_lo=("prob_cal", "min"),
        prob_hi=("prob_cal", "max"),
        proj_mean=("ret", "mean"),
        proj_median=("ret", "median"),
        avg_hold=("hold_days", "mean"),
        win_rate=("ret", lambda x: (x > 0).mean()),
    ).reset_index()
    return tab


def attach_projection(today: pd.DataFrame, tab: pd.DataFrame) -> pd.DataFrame:
    today = today.copy()
    proj_mean, proj_med, win_rate, avg_hold = [], [], [], []
    for p in today["prob_cal"]:
        match = None
        for _, b in tab.iterrows():
            if b["prob_lo"] <= p <= b["prob_hi"]:
                match = b
                break
        if match is None:
            match = tab.iloc[(tab["prob_lo"] - p).abs().argmin()]
        proj_mean.append(match["proj_mean"])
        proj_med.append(match["proj_median"])
        win_rate.append(match["win_rate"])
        avg_hold.append(match["avg_hold"])
    today["proj_mean"] = proj_mean
    today["proj_median"] = proj_med
    today["win_rate"] = win_rate
    today["avg_hold"] = avg_hold
    return today


def get_today_picks(scores: pd.DataFrame, last_date, k: int = 5) -> pd.DataFrame:
    """Top-K by raw_margin on the latest date, regime ON if SPY 20d > 0.
    Annotates each pick with extension classification."""
    today = scores[scores["date"] == last_date].copy()
    spy_20d = today["spy_ret_20d"].iloc[0] if len(today) else 0.0
    regime_on = spy_20d > 0
    print(f"     today={last_date.date()} | spy_20d={spy_20d:+.1%} | regime_on={regime_on}")
    if not regime_on:
        print(f"     REGIME OFF — Option 1B says no trades today")
        return pd.DataFrame(), regime_on, spy_20d
    # Get top-K by raw_margin, then attach extension classification
    top = today.nlargest(k, "raw_margin").reset_index(drop=True)
    top = attach_extension(top)
    return top, regime_on, spy_20d


def format_pick(row, idx) -> str:
    sec = row.get("sector", "")
    if pd.isna(sec) or sec == "":
        sec = "-"
    sec_short = sec[:6] if isinstance(sec, str) and len(sec) > 6 else sec
    rl = int(row["run_length"]) if not pd.isna(row["run_length"]) else 0
    atr = row.get("atr_pct", float("nan"))
    dd = row.get("dd_60d", float("nan"))
    atr_s = f"{atr:.1%}" if not pd.isna(atr) else "?"
    dd_s = f"{dd:.0%}" if not pd.isna(dd) else "?"
    ext_lvl = row.get("ext_level", "")
    r60 = row.get("ret_60d_lag", float("nan"))
    r60_s = f"60d{r60:+.0%}" if not pd.isna(r60) else ""
    ext_tag = f"[{ext_lvl}] " if ext_lvl else ""
    return (
        f"{idx}. {row['ticker']} {ext_tag}${row['close']:.2f}  "
        f"margin{row['raw_margin']:+.2f}  prob{row['prob_cal']:.0%}  "
        f"5d{row['ret_5d_lag']:+.0%} 20d{row['ret_20d_lag']:+.0%} {r60_s}  "
        f"win{row['win_rate']:.0%} mean{row['proj_mean']:+.0%} med{row['proj_median']:+.0%} hold{row['avg_hold']:.0f}d  "
        f"atr{atr_s} dd{dd_s} rl{rl} {sec_short}"
    )


def build_lists(today: pd.DataFrame, label: str, last_date, spy_20d) -> tuple:
    # Split by extension safety FIRST, then by 5d-momentum.
    # Anything classified EXTREME or GRADUAL is moved to a "DO NOT CHASE" list,
    # NOT a buy recommendation — even if its raw_margin is high.
    safe = today[today["ext_safe"] == True].copy()
    no_chase = today[today["ext_safe"] == False].copy()

    pw = safe[(safe["ret_5d_lag"] >= 0.05) & (safe["ret_5d_lag"] < 0.30)].copy()
    ar = safe[safe["ret_5d_lag"].abs() < 0.05].copy()

    pw_lines = [
        f"PARTIAL WINNERS — {label} ({last_date.date()}) — Option 1B",
        f"Already 5-30% into a possible run. Top picks by v3 raw_margin (regime ON, SPY 20d {spy_20d:+.1%}).",
        f"HOLD RULE: keep while prob_cal >= 5% (cap 60d). EXIT when model says signal gone.",
        f"Legend: margin = entry strength | prob = hold-decision | win/mean/med/hold = TEST-FOLD historical at this prob bucket",
        f"        atr = daily-range %, dd = 60d max drawdown, rl = consecutive up-days",
        "",
    ]
    for i, (_, r) in enumerate(pw.iterrows(), 1):
        pw_lines.append(format_pick(r, i))
    if len(pw) == 0:
        pw_lines.append("(no qualifying stocks)")

    ar_lines = [
        f"ABOUT TO RISE — {label} ({last_date.date()}) — Option 1B",
        f"Top picks not yet moving (|5d ret| < 5%). v3 raw_margin (regime ON, SPY 20d {spy_20d:+.1%}).",
        f"HOLD RULE: keep while prob_cal >= 5% (cap 60d). EXIT when model says signal gone.",
        f"Legend: margin = entry strength | prob = hold-decision | win/mean/med/hold = TEST-FOLD historical at this prob bucket",
        f"        atr = daily-range %, dd = 60d max drawdown, rl = consecutive up-days",
        "",
    ]
    for i, (_, r) in enumerate(ar.iterrows(), 1):
        ar_lines.append(format_pick(r, i))
    if len(ar) == 0:
        ar_lines.append("(no qualifying stocks)")

    # Extended — high raw_margin but already extended; informational, not primary picks.
    nc_lines = []
    if len(no_chase) > 0:
        nc_lines = [
            f"EXTENDED (informational) — {label} ({last_date.date()})",
            f"Still high model conviction but already had a substantial run.",
            f"Historical bucket mean for this extension class: ~+13% vs +29% on fresh setups.",
            f"Lean toward fresh picks above for new entries.",
            "",
        ]
        for i, (_, r) in enumerate(no_chase.iterrows(), 1):
            nc_lines.append(format_pick(r, i))

    return "\n".join(pw_lines), pw, "\n".join(ar_lines), ar, "\n".join(nc_lines), no_chase


def run(universe):
    print(f"\n[100] === {universe} ===")
    if universe == "sp500":
        panel_path = DATA / "monthly_gainer_panel.csv"
        cat_path = DATA / "catalyst_features_sp500.csv"
        model_path = MODELS / "monthly_gainer_v3_sp500.joblib"
    else:
        panel_path = DATA / "monthly_gainer_panel_smallcap.csv"
        cat_path = DATA / "catalyst_features_smallcap.csv"
        model_path = MODELS / "monthly_gainer_v3_smallcap.joblib"

    scores = score_full_panel(panel_path, cat_path, model_path)
    print(f"     scored: {len(scores):,}")
    lookup = build_lookup(scores)

    print(f"     building Option 1B calibration table ...")
    tab = option1b_calibration_table(scores, lookup)
    print(tab.to_string(index=False))

    last_date = scores["date"].max()
    # Top-5 raw_margin (universe-blind safe picks, extension-filtered).
    # Surface fewer picks since small-N investors don't take 15.
    today, regime_on, spy_20d = get_today_picks(scores, last_date, k=5)
    if not regime_on:
        return

    today = attach_projection(today, tab)

    pw_text, pw_df, ar_text, ar_df, nc_text, nc_df = build_lists(today, universe.upper(), last_date, spy_20d)

    print("\n" + "="*80)
    print(pw_text)
    print("\n" + "="*80)
    print(ar_text)
    if nc_text:
        print("\n" + "="*80)
        print(nc_text)
    print("="*80)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"opt1b_partial_winners_{universe}.txt").write_text(pw_text)
    (OUT / f"opt1b_about_to_rise_{universe}.txt").write_text(ar_text)
    if nc_text:
        (OUT / f"opt1b_no_chase_{universe}.txt").write_text(nc_text)
    pw_df.to_csv(OUT / f"opt1b_partial_winners_{universe}.csv", index=False)
    ar_df.to_csv(OUT / f"opt1b_about_to_rise_{universe}.csv", index=False)
    if len(nc_df) > 0:
        nc_df.to_csv(OUT / f"opt1b_no_chase_{universe}.csv", index=False)
    tab.to_csv(OUT / f"opt1b_calibration_{universe}.csv", index=False)


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "sp500"
    run(universe)


if __name__ == "__main__":
    main()
