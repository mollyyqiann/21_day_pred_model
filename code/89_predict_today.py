"""Phase G: 'Predict today' + SNDK/AMD lookback + partial-winners view.

Answers:
  1. What stocks does the model predict are about to rise (next 21 days)?
     -> Score the latest panel rows with v1 model, return top-K predictions.
  2. Did the model see SNDK and AMD coming?
     -> Walk the last 90 days of those tickers; show prob over time vs realized run.
  3. Which test-fold predictions are 'in progress' (already +5-30% partway)?
     -> Filter test rows where prob >= top-decile AND 5d realized return is in [5%, 30%).

Usage:
  python 89_predict_today.py
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

V1_FEATURES = [
    "rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20",
    "atr_pct", "range_pct", "vol_z", "vol_5d", "rv_60",
    "ma_stack", "up_streak", "up_bigdays_20d",
    "dist_ma60_atr", "ma60_slope_60d", "run_length",
]


def score_panel(panel_path: Path, model_path: Path, label: str) -> pd.DataFrame:
    """Score the entire panel and return a DataFrame with prob attached."""
    art = joblib.load(model_path)
    feats = art["feats"]
    medians = pd.Series(art["impute_medians"])
    cal = art["calibrator"]
    print(f"[89] {label}: loaded {model_path.name}  feats={len(feats)}")

    panel = pd.read_csv(panel_path, parse_dates=["date"])
    panel = attach_regime(panel)
    # need 5d trailing return for partial-winners view
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["close_5d_ago"] = panel.groupby("ticker")["close"].shift(5)
    panel["ret_5d_lag"] = panel["close"] / panel["close_5d_ago"] - 1.0
    panel["close_20d_ago"] = panel.groupby("ticker")["close"].shift(20)
    panel["ret_20d_lag"] = panel["close"] / panel["close_20d_ago"] - 1.0

    # scoring requires complete features (drop warm-up rows only)
    scor = panel.dropna(subset=feats).copy()
    X = scor[feats].fillna(medians).values
    scor["prob"] = cal.predict_proba(X)[:, 1]
    print(f"[89] {label}: scored {len(scor):,} rows out of {len(panel):,}")
    return scor


def show_today_picks(scor: pd.DataFrame, label: str, k: int = 15):
    last_d = scor["date"].max()
    today = scor[scor["date"] == last_d].copy()
    print(f"\n[89] === {label}: top-{k} picks for {last_d.date()} ===")
    cols = ["ticker", "prob", "rv_60", "run_length", "ret_5d_lag", "ret_20d_lag", "close"]
    if "sector" in today.columns:
        cols.insert(2, "sector")
    top = today.nlargest(k, "prob")[cols]
    print(f"  base test rate (proxy): see report; calibrated probs are absolute risk over next 21 trading days")
    print(top.to_string(index=False, formatters={
        "prob": "{:.3f}".format,
        "rv_60": "{:.2f}".format,
        "ret_5d_lag": "{:+.1%}".format,
        "ret_20d_lag": "{:+.1%}".format,
        "close": "{:.2f}".format,
    }))


def show_ticker_history(scor: pd.DataFrame, ticker: str, n_days: int = 60):
    """Show the model's prob trajectory for a ticker over the last n_days."""
    g = scor[scor["ticker"] == ticker].sort_values("date").tail(n_days)
    if len(g) == 0:
        print(f"\n[89] no data for {ticker}")
        return
    print(f"\n[89] === {ticker}: last {len(g)} days ===")
    # peak-from-here: max future close / current close
    g = g.copy()
    closes = g["close"].values
    peaks = np.full(len(g), np.nan)
    for i in range(len(g)):
        peaks[i] = closes[i:].max() / closes[i] - 1.0 if i < len(g) else np.nan
    g["peak_remaining"] = peaks
    cols = ["date", "close", "prob", "rv_60", "run_length", "ret_5d_lag", "peak_remaining"]
    print(g[cols].to_string(index=False, formatters={
        "close": "{:.2f}".format,
        "prob": "{:.3f}".format,
        "rv_60": "{:.2f}".format,
        "ret_5d_lag": "{:+.1%}".format,
        "peak_remaining": "{:+.1%}".format,
    }))


def show_partial_winners(scor: pd.DataFrame, label: str,
                          prob_pctile: float = 0.95,
                          ret_lo: float = 0.05, ret_hi: float = 0.30,
                          k: int = 20):
    """Stocks where the model predicted high prob AND have already rallied
    5-30% over the last 5d (i.e., in-progress, not yet hit +30%)."""
    print(f"\n[89] === {label}: partial winners (prob>=p{int(prob_pctile*100)}, 5d return in [{ret_lo:.0%}, {ret_hi:.0%})) ===")
    # take last available date as "today"
    last_d = scor["date"].max()
    today = scor[scor["date"] == last_d].copy()
    p_thresh = today["prob"].quantile(prob_pctile)
    matches = today[(today["prob"] >= p_thresh)
                    & (today["ret_5d_lag"] >= ret_lo)
                    & (today["ret_5d_lag"] < ret_hi)].copy()
    matches = matches.sort_values("prob", ascending=False).head(k)
    cols = ["ticker", "prob", "rv_60", "run_length", "ret_5d_lag", "ret_20d_lag", "close"]
    if "sector" in matches.columns:
        cols.insert(2, "sector")
    print(matches[cols].to_string(index=False, formatters={
        "prob": "{:.3f}".format,
        "rv_60": "{:.2f}".format,
        "ret_5d_lag": "{:+.1%}".format,
        "ret_20d_lag": "{:+.1%}".format,
        "close": "{:.2f}".format,
    }))


def main():
    print(f"[89] === SP500 universe ===")
    sp_scor = score_panel(DATA / "monthly_gainer_panel.csv",
                           MODELS / "monthly_gainer_v1.joblib", "SP500")
    show_today_picks(sp_scor, "SP500", k=15)
    show_partial_winners(sp_scor, "SP500", prob_pctile=0.90)

    # SNDK / AMD lookback
    show_ticker_history(sp_scor, "SNDK", n_days=80)
    show_ticker_history(sp_scor, "AMD", n_days=80)

    print(f"\n[89] === Smallcap universe ===")
    sc_scor = score_panel(DATA / "monthly_gainer_panel_smallcap.csv",
                           MODELS / "monthly_gainer_smallcap_v1.joblib", "SMALLCAP")
    show_today_picks(sc_scor, "SMALLCAP", k=15)
    show_partial_winners(sc_scor, "SMALLCAP", prob_pctile=0.95)

    # save the scored 'today' rows for both universes
    sp_today = sp_scor[sp_scor["date"] == sp_scor["date"].max()].sort_values("prob", ascending=False)
    sc_today = sc_scor[sc_scor["date"] == sc_scor["date"].max()].sort_values("prob", ascending=False)
    sp_today.to_csv(OUT / "today_picks_sp500.csv", index=False)
    sc_today.to_csv(OUT / "today_picks_smallcap.csv", index=False)
    print(f"\n[89] saved {OUT / 'today_picks_sp500.csv'} ({len(sp_today)} rows)")
    print(f"[89] saved {OUT / 'today_picks_smallcap.csv'} ({len(sc_today)} rows)")


if __name__ == "__main__":
    main()
