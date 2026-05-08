"""Score today's panel rows with v3-up + drop15-down models, build the
two ranked lists for WhatsApp delivery:

  1. PARTIAL WINNERS: high prob_up AND already 5–30% into a possible run.
     Filter: prob_up >= 0.10 AND ret_5d_lag in [+5%, +30%).
  2. ABOUT TO RISE: high prob_up AND no recent move.
     Filter: prob_up >= 0.10 AND |ret_5d_lag| < 5%.

NOTE on drop filter: the drop15 calibrator saturates at ~0.093 — the
ceiling shared by all high-vol stocks. So drop_prob doesn't differentiate
among top picks. We REPORT it as info but don't hard-filter on it,
matching the user's instruction "don't penalize long run unless high drop".
A run-length stock with drop_prob at the ceiling is no riskier than any
other top pick.

Projection: prob_up bucketed by absolute value (not deciles, since deciles
collapse the long tail of zeros). Conditional mean max_fwd21_ret and
end_of_window_ret computed from labeled test rows.

Outputs WhatsApp-ready text + CSV per universe.
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
PROB_BUCKETS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.01]


def add_xrank_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    if "close_20d_ago" not in df.columns:
        df["close_20d_ago"] = df.groupby("ticker")["close"].shift(20)
        df["ret_20d_lag"] = df["close"] / df["close_20d_ago"] - 1.0
    df["rsi_14_xrank"] = df.groupby("date")["rsi_14"].rank(pct=True)
    df["rv_60_xrank"] = df.groupby("date")["rv_60"].rank(pct=True)
    df["ma60_slope_xrank"] = df.groupby("date")["ma60_slope_60d"].rank(pct=True)
    df["ret_20d_xrank"] = df.groupby("date")["ret_20d_lag"].rank(pct=True)
    return df


def score_panel(panel_path: Path, cat_path: Path, up_model: dict,
                drop_model: dict, label: str) -> pd.DataFrame:
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    panel = attach_regime(panel)
    if cat_path.exists():
        cat = pd.read_csv(cat_path, parse_dates=["date"])
        panel = panel.merge(cat, on=["ticker", "date"], how="left")

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["close_5d_ago"] = panel.groupby("ticker")["close"].shift(5)
    panel["ret_5d_lag"] = panel["close"] / panel["close_5d_ago"] - 1.0
    panel["close_20d_ago"] = panel.groupby("ticker")["close"].shift(20)
    panel["ret_20d_lag"] = panel["close"] / panel["close_20d_ago"] - 1.0

    # 60d trailing max drawdown: min(close in last 60d) / max(close in last 60d) - 1
    panel["max_60d"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.rolling(60, min_periods=20).max())
    panel["min_60d"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.rolling(60, min_periods=20).min())
    panel["dd_60d"] = panel["min_60d"] / panel["max_60d"] - 1.0  # negative

    panel = add_xrank_features(panel)

    for c in ["finbert_max_5d", "finbert_max_20d", "finbert_mean_5d",
              "news_n_5d", "news_n_20d", "earn_news_5d", "earn_news_20d",
              "ma_news_5d", "ma_news_20d", "sector_pop_5d"]:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0)
        else:
            panel[c] = 0.0
    for c in XRANK_FEATURES:
        panel[c] = panel[c].fillna(0.5)

    up_feats = up_model["feats"]
    up_med = pd.Series(up_model["impute_medians"])
    scor = panel.dropna(subset=up_feats).copy()
    Xup = scor[up_feats].fillna(up_med).values
    scor["prob_up"] = up_model["calibrator"].predict_proba(Xup)[:, 1]

    dn_feats = drop_model["feats"]
    dn_med = pd.Series(drop_model["impute_medians"])
    scor_d = scor.dropna(subset=dn_feats).copy()
    Xdn = scor_d[dn_feats].fillna(dn_med).values
    scor["prob_drop15"] = np.nan
    scor.loc[scor_d.index, "prob_drop15"] = drop_model["calibrator"].predict_proba(Xdn)[:, 1]

    print(f"[94] {label}: scored {len(scor):,} rows; "
          f"prob_up max={scor['prob_up'].max():.3f}  "
          f"prob_drop max={scor['prob_drop15'].max():.3f}")
    return scor


def projection_table(scor: pd.DataFrame) -> pd.DataFrame:
    """Per absolute-prob-bucket, conditional mean max_fwd21_ret and end_ret."""
    labeled = scor[scor["y21"].isin([0, 1])].copy()
    if "max_fwd21_ret" not in labeled.columns:
        return None
    labeled["bucket"] = pd.cut(labeled["prob_up"], bins=PROB_BUCKETS,
                                labels=[f"[{PROB_BUCKETS[i]:.2f},{PROB_BUCKETS[i+1]:.2f})"
                                         for i in range(len(PROB_BUCKETS) - 1)],
                                include_lowest=True)
    tab = labeled.groupby("bucket", observed=True).agg(
        n=("prob_up", "size"),
        prob_lo=("prob_up", "min"),
        prob_hi=("prob_up", "max"),
        proj_peak=("max_fwd21_ret", "mean"),
        proj_end=("end_of_window_ret", "mean"),
        hit_rate=("y21", "mean"),
    ).reset_index()
    return tab


def attach_projection(today: pd.DataFrame, tab: pd.DataFrame) -> pd.DataFrame:
    today = today.copy()
    proj_peak, proj_end, hit_rate = [], [], []
    for p in today["prob_up"]:
        match = None
        for _, b in tab.iterrows():
            if b["prob_lo"] <= p <= b["prob_hi"]:
                match = b
                break
        if match is None:
            match = tab.iloc[(tab["prob_lo"] - p).abs().argmin()]
        proj_peak.append(match["proj_peak"])
        proj_end.append(match["proj_end"])
        hit_rate.append(match["hit_rate"])
    today["proj_peak"] = proj_peak
    today["proj_end"] = proj_end
    today["hit_rate"] = hit_rate
    return today


def format_pick_line(row, idx) -> str:
    sec = row.get("sector", "")
    if pd.isna(sec) or sec == "":
        sec = "-"
    sec_short = sec[:6] if isinstance(sec, str) and len(sec) > 6 else sec
    rl = int(row["run_length"]) if not pd.isna(row["run_length"]) else 0
    atr_pct = row.get("atr_pct", float("nan"))
    dd_60d = row.get("dd_60d", float("nan"))
    atr_str = f"{atr_pct:.1%}" if not pd.isna(atr_pct) else "?"
    dd_str = f"{dd_60d:.0%}" if not pd.isna(dd_60d) else "?"
    return (
        f"{idx}. {row['ticker']} ${row['close']:.2f}  "
        f"prob+{row['prob_up']:.0%}  "
        f"5d{row['ret_5d_lag']:+.0%} 20d{row['ret_20d_lag']:+.0%}  "
        f"hit{row['hit_rate']:.0%} peak{row['proj_peak']:+.0%} end{row['proj_end']:+.0%}  "
        f"atr{atr_str} dd60{dd_str} rl{rl} {sec_short}"
    )


def build_partial_winners(scor: pd.DataFrame, label: str, tab: pd.DataFrame,
                           prob_thresh: float = 0.10, k: int = 12) -> tuple:
    last_d = scor["date"].max()
    today = scor[scor["date"] == last_d].copy()
    matches = today[
        (today["prob_up"] >= prob_thresh)
        & (today["ret_5d_lag"] >= 0.05)
        & (today["ret_5d_lag"] < 0.30)
    ].copy()
    matches = matches.sort_values("prob_up", ascending=False).head(k)
    matches = attach_projection(matches, tab)

    lines = [
        f"PARTIAL WINNERS — {label} ({last_d.date()})",
        f"Already 5-30% into a possible +30% run. prob+ >= {prob_thresh:.0%}.",
        f"Legend: prob+ = P(touch +30% in 21d) | hit/peak/end = historical avg outcome at this prob bucket",
        f"atr = daily-range %, dd60 = max drawdown last 60d, rl = consecutive up-days",
        "",
    ]
    for i, (_, r) in enumerate(matches.iterrows(), 1):
        lines.append(format_pick_line(r, i))
    if len(matches) == 0:
        lines.append("(no qualifying stocks)")
    return "\n".join(lines), matches


def build_about_to_rise(scor: pd.DataFrame, label: str, tab: pd.DataFrame,
                         prob_thresh: float = 0.10, k: int = 12) -> tuple:
    last_d = scor["date"].max()
    today = scor[scor["date"] == last_d].copy()
    matches = today[
        (today["prob_up"] >= prob_thresh)
        & (today["ret_5d_lag"].abs() < 0.05)
    ].copy()
    matches = matches.sort_values("prob_up", ascending=False).head(k)
    matches = attach_projection(matches, tab)

    lines = [
        f"ABOUT TO RISE — {label} ({last_d.date()})",
        f"High prob+ AND no recent move (|5d-ret| < 5%). prob+ >= {prob_thresh:.0%}.",
        f"Legend: prob+ = P(touch +30% in 21d) | hit/peak/end = historical avg outcome at this prob bucket",
        f"atr = daily-range %, dd60 = max drawdown last 60d, rl = consecutive up-days",
        "",
    ]
    for i, (_, r) in enumerate(matches.iterrows(), 1):
        lines.append(format_pick_line(r, i))
    if len(matches) == 0:
        lines.append("(no qualifying stocks)")
    return "\n".join(lines), matches


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "sp500"
    print(f"[94] universe={universe}")

    if universe == "sp500":
        panel_path = DATA / "monthly_gainer_panel.csv"
        cat_path = DATA / "catalyst_features_sp500.csv"
        up_path = MODELS / "monthly_gainer_v3_sp500.joblib"
        drop_path = MODELS / "monthly_gainer_drop15_sp500.joblib"
        prob_thresh = 0.10
    else:
        panel_path = DATA / "monthly_gainer_panel_smallcap.csv"
        cat_path = DATA / "catalyst_features_smallcap.csv"
        up_path = MODELS / "monthly_gainer_v3_smallcap.joblib"
        if not up_path.exists():
            up_path = MODELS / "monthly_gainer_v2_smallcap.joblib"
            print(f"[94] v3 smallcap not found; using v2")
        drop_path = MODELS / "monthly_gainer_drop15_smallcap.joblib"
        prob_thresh = 0.30

    print(f"[94] loading {up_path.name} + {drop_path.name}")
    up_model = joblib.load(up_path)
    drop_model = joblib.load(drop_path)

    scor = score_panel(panel_path, cat_path, up_model, drop_model, universe)
    tab = projection_table(scor)
    print(f"[94] projection table:\n{tab.to_string(index=False)}")

    pw_text, pw_df = build_partial_winners(scor, universe.upper(), tab, prob_thresh)
    ar_text, ar_df = build_about_to_rise(scor, universe.upper(), tab, prob_thresh)

    print("\n" + "="*70)
    print(pw_text)
    print("\n" + "="*70)
    print(ar_text)
    print("="*70)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"whatsapp_partial_winners_{universe}.txt").write_text(pw_text)
    (OUT / f"whatsapp_about_to_rise_{universe}.txt").write_text(ar_text)
    pw_df.to_csv(OUT / f"partial_winners_{universe}_full.csv", index=False)
    ar_df.to_csv(OUT / f"about_to_rise_{universe}_full.csv", index=False)
    print(f"\n[94] saved 4 files in {OUT}")


if __name__ == "__main__":
    main()
