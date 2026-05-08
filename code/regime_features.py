"""Shared regime-feature loader used by v10 training and the daily runner.

Loads market-wide daily features (SPY realized vol, returns, VIX, VIX trend,
CNN Fear & Greed) from local CSVs under data/, returns a per-date frame that
can be merged onto any stock panel.

Usage:
    from regime_features import load_regime_frame, REGIME_FEATS, attach_regime

    rf = load_regime_frame()
    panel_with_regime = attach_regime(panel, rf)

The columns produced are (see REGIME_FEATS):
    spy_ret_5d, spy_ret_20d, spy_rv_20, spy_rv_60,
    vix, vix_chg_5d, fng

Design notes:
    * All features are causal (only look back from a given date), so no
      look-ahead leakage when merged on `date`.
    * `attach_regime` forward-fills per ticker so a stock's row on day D picks
      up the most recent regime value as of D (handles the rare case where a
      stock's trading calendar differs from SPY — e.g. halts).
    * Missing local data is tolerated (returns NaNs for that column); downstream
      models trained with HistGradientBoosting handle NaNs natively.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

REGIME_FEATS = [
    "spy_ret_5d", "spy_ret_20d", "spy_rv_20", "spy_rv_60",
    "vix", "vix_chg_5d", "fng",
]


def _safe_read(path: Path, **kw) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return pd.DataFrame()


def load_regime_frame() -> pd.DataFrame:
    """Return a DataFrame with columns ['date'] + REGIME_FEATS, one row per
    trading day (outer join of SPY ∪ VIX ∪ FnG), sorted and forward-filled."""
    spy = _safe_read(DATA / "sp500_daily.csv", parse_dates=["date"])
    if len(spy):
        spy = spy.sort_values("date").reset_index(drop=True)
        spy["spy_ret_5d"]  = spy["close"].pct_change(5)
        spy["spy_ret_20d"] = spy["close"].pct_change(20)
        spy["spy_rv_20"]   = spy["close"].pct_change().rolling(20).std() * np.sqrt(252)
        spy["spy_rv_60"]   = spy["close"].pct_change().rolling(60).std() * np.sqrt(252)
        spy = spy[["date", "spy_ret_5d", "spy_ret_20d", "spy_rv_20", "spy_rv_60"]]
    else:
        spy = pd.DataFrame(columns=["date", "spy_ret_5d", "spy_ret_20d",
                                     "spy_rv_20", "spy_rv_60"])

    vix = _safe_read(DATA / "vix_daily.csv", parse_dates=["date"])
    if len(vix):
        vix = vix.sort_values("date").reset_index(drop=True)
        vix["vix_chg_5d"] = vix["vix"].diff(5)
        vix = vix[["date", "vix", "vix_chg_5d"]]
    else:
        vix = pd.DataFrame(columns=["date", "vix", "vix_chg_5d"])

    fg = _safe_read(DATA / "fear_greed.csv", parse_dates=["date"])
    if len(fg):
        fg = fg.sort_values("date")[["date", "fng"]]
    else:
        fg = pd.DataFrame(columns=["date", "fng"])

    out = spy.merge(vix, on="date", how="outer").merge(fg, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    # Forward-fill so non-trading-day rows (if any slip in) pick up last known
    for c in REGIME_FEATS:
        if c not in out.columns:
            out[c] = np.nan
    out[REGIME_FEATS] = out[REGIME_FEATS].ffill()
    # Drop duplicate dates (some source CSVs have them); keep last
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out[["date"] + REGIME_FEATS]


def attach_regime(panel: pd.DataFrame, regime: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge regime features onto `panel` by date, then forward-fill per
    ticker to patch any gap.

    Returns a copy. `panel` is not modified in place.
    """
    if regime is None:
        regime = load_regime_frame()
    # Normalize date dtype
    p = panel.copy()
    if not np.issubdtype(p["date"].dtype, np.datetime64):
        p["date"] = pd.to_datetime(p["date"])
    r = regime.copy()
    if not np.issubdtype(r["date"].dtype, np.datetime64):
        r["date"] = pd.to_datetime(r["date"])
    p = p.merge(r, on="date", how="left")
    p = p.sort_values(["ticker", "date"])
    # forward fill within ticker to handle any remaining gaps
    p[REGIME_FEATS] = p.groupby("ticker")[REGIME_FEATS].ffill()
    return p.reset_index(drop=True)


if __name__ == "__main__":
    # smoke test
    rf = load_regime_frame()
    print(f"loaded regime frame: {len(rf)} rows, "
          f"date range {rf['date'].min()} → {rf['date'].max()}")
    print(rf.tail().to_string(index=False))
