"""Today's top picks under v3-combined (universe-blind: SP500 + smallcap).

This is what surfaces mid-cap names like MRVL that fall through the
SP500/smallcap split. Output: top-15 across the full 1,891-ticker
Robinhood-tradable universe.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)
import warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output" / "monthly_gainer"
sys.path.insert(0, str(ROOT / "code"))
from regime_features import attach_regime
from extension_classifier import attach_extension

XRANK = ["rsi_14_xrank", "rv_60_xrank", "ma60_slope_xrank", "ret_20d_xrank"]
CATALYST = [
    "finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
    "news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
    "ma_news_5d", "ma_news_20d", "sector_pop_5d",
]


def main():
    print("[111] loading combined panels ...")
    sp = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    sc = pd.read_csv(DATA / "monthly_gainer_panel_smallcap.csv", parse_dates=["date"])
    common = list(set(sp.columns) & set(sc.columns))
    panel = pd.concat([sp[common], sc[common]], ignore_index=True)

    cat_sp = pd.read_csv(DATA / "catalyst_features_sp500.csv", parse_dates=["date"])
    cat_sc = pd.read_csv(DATA / "catalyst_features_smallcap.csv", parse_dates=["date"])
    cat = pd.concat([cat_sp, cat_sc], ignore_index=True)
    panel = panel.merge(cat, on=["ticker", "date"], how="left")
    panel = attach_regime(panel)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["close_5d_ago"] = panel.groupby("ticker")["close"].shift(5)
    panel["ret_5d_lag"] = panel["close"] / panel["close_5d_ago"] - 1.0
    panel["close_20d_ago"] = panel.groupby("ticker")["close"].shift(20)
    panel["ret_20d_lag"] = panel["close"] / panel["close_20d_ago"] - 1.0
    panel["close_60d_ago"] = panel.groupby("ticker")["close"].shift(60)
    panel["ret_60d_lag"] = panel["close"] / panel["close_60d_ago"] - 1.0
    panel["close_180d_ago"] = panel.groupby("ticker")["close"].shift(180)
    panel["ret_180d_lag"] = panel["close"] / panel["close_180d_ago"] - 1.0
    panel["rsi_14_xrank"] = panel.groupby("date")["rsi_14"].rank(pct=True)
    panel["rv_60_xrank"] = panel.groupby("date")["rv_60"].rank(pct=True)
    panel["ma60_slope_xrank"] = panel.groupby("date")["ma60_slope_60d"].rank(pct=True)
    panel["ret_20d_xrank"] = panel.groupby("date")["ret_20d_lag"].rank(pct=True)
    for c in CATALYST:
        if c not in panel.columns: panel[c] = 0.0
        else: panel[c] = panel[c].fillna(0)
    for c in XRANK:
        panel[c] = panel[c].fillna(0.5)

    panel["max_60d"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.rolling(60, min_periods=20).max())
    panel["min_60d"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.rolling(60, min_periods=20).min())
    panel["dd_60d"] = panel["min_60d"] / panel["max_60d"] - 1.0

    art = joblib.load(MODELS / "monthly_gainer_v3_combined.joblib")
    feats = art["feats"]; med = pd.Series(art["impute_medians"])
    cal = art["calibrator"]; gbc = art["raw_gbc"]
    sub = panel.dropna(subset=feats).copy()
    X = sub[feats].fillna(med).values
    sub["prob_cal"] = cal.predict_proba(X)[:, 1]
    sub["raw_margin"] = gbc.decision_function(X)

    last_d = sub["date"].max()
    today = sub[sub["date"] == last_d].copy()
    spy_20d = today["spy_ret_20d"].iloc[0] if len(today) else 0.0
    print(f"\n[111] today={last_d.date()}  spy_20d={spy_20d:+.1%}  regime={'ON' if spy_20d>0 else 'OFF'}")
    print(f"[111] candidates with full features: {len(today)}")

    today = attach_extension(today)

    # Show top-25 by raw_margin
    top25 = today.nlargest(25, "raw_margin").reset_index(drop=True)
    print(f"\n=== UNIVERSE-BLIND TOP-25 ===")
    print(f"{'rank':<5} {'ticker':<7} {'ext':<10} {'close':>9} {'margin':>8} {'prob':>6} "
          f"{'5d':>6} {'20d':>6} {'60d':>6} {'sector'}")
    for i, r in top25.iterrows():
        sec_raw = r.get('sector')
        sec = (sec_raw if isinstance(sec_raw, str) and sec_raw else '?')[:6]
        r60 = r.get('ret_60d_lag', float('nan'))
        r60_s = f"{r60:+.0%}" if pd.notna(r60) else "n/a"
        print(f"{i+1:<5} {r['ticker']:<7} {r.get('ext_level',''):<10} ${r['close']:>7.2f} "
              f"{r['raw_margin']:>+8.2f} {r['prob_cal']:>6.0%} "
              f"{r['ret_5d_lag']:>+6.0%} {r['ret_20d_lag']:>+6.0%} {r60_s:>6} {sec}")

    # Where does MRVL rank?
    mrvl_row = today[today["ticker"] == "MRVL"]
    rank_df = today.copy()
    rank_df["rank"] = rank_df["raw_margin"].rank(method="min", ascending=False)
    if not mrvl_row.empty:
        m = rank_df[rank_df["ticker"] == "MRVL"].iloc[0]
        print(f"\n=== MRVL universe-blind ranking ===")
        print(f"  rank: #{int(m['rank'])} of {len(rank_df)}")
        print(f"  margin: {m['raw_margin']:+.2f}, prob: {m['prob_cal']:.0%}")
        print(f"  close: ${m['close']:.2f}, 5d: {m['ret_5d_lag']:+.1%}, 20d: {m['ret_20d_lag']:+.1%}")
        print(f"  ext: {m.get('ext_level','?')}")

    OUT.mkdir(parents=True, exist_ok=True)
    today.to_csv(OUT / "combined_today_score.csv", index=False)
    top25.to_csv(OUT / "combined_top25.csv", index=False)
    print(f"\n[111] saved {OUT / 'combined_top25.csv'}")


if __name__ == "__main__":
    main()
