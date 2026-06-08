"""Form 4 (insider transactions) signal spike for the burst v6 model.

Pulls Form 4 filing metadata from SEC EDGAR submissions JSON for every ticker
in burst_panel_v6.csv (3y, 226 tickers), aggregates to per-(ticker, filed_date)
counts, builds rolling 5d/30d features, joins to the panel, and reports:

  - coverage (panel rows with non-zero feature)
  - univariate IC (Spearman) and decile lift vs target y
  - add-to-v6 ablation: train a GBC on v6 features ± Form 4 features and
    compare 5-fold time-series CV log-loss + AUC.

This is a v1 count-based test. Counts mix buys/sells; if there's signal here,
a v2 should parse Form 4 XML for transactionCode (P=buy / S=sell). If counts
show no IC, the XML pass is unnecessary.

Outputs:
  data/burst_panel_v6_form4.csv    — panel + form4_* feature columns
  data/form4_filings.csv           — pulled filings (ticker, filed, accession)
  output/form4_spike_report.txt    — IC + ablation summary

Usage:
  python3 code/130_burst_form4_spike.py              # full run
  python3 code/130_burst_form4_spike.py --skip-pull  # skip SEC fetch, use cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from _sector_common import sec_session  # noqa: E402

PANEL_PATH = ROOT / "data" / "burst_panel_v6.csv"
CIK_MAP_PATH = ROOT / "data" / "edgar_backfill" / "ticker_cik_map.csv"
FORM4_OUT = ROOT / "data" / "form4_filings.csv"
PANEL_OUT = ROOT / "data" / "burst_panel_v6_form4.csv"
REPORT_OUT = ROOT / "output" / "form4_spike_report.txt"


# ---------- pull ----------

def _extract_form4(submissions: dict) -> list[tuple[str, str]]:
    """Return list of (filed_date_str, accession) for form='4' in submissions."""
    out: list[tuple[str, str]] = []
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    for f, d, a in zip(forms, dates, accs):
        if f == "4":
            out.append((d, a))
    return out


def _fetch_older_form4(sess, cik: str, files_meta: list[dict]) -> list[tuple[str, str]]:
    """Walk older 'files' index pages and pull form-4 metadata."""
    out: list[tuple[str, str]] = []
    for entry in files_meta:
        name = entry.get("name")
        if not name:
            continue
        url = f"https://data.sec.gov/submissions/{name}"
        try:
            page = sess.json(url)
        except Exception as e:
            print(f"  [warn] older index {name} failed: {e}")
            continue
        forms = page.get("form", [])
        dates = page.get("filingDate", [])
        accs = page.get("accessionNumber", [])
        for f, d, a in zip(forms, dates, accs):
            if f == "4":
                out.append((d, a))
    return out


def pull_form4_filings(tickers: list[str], cik_map: dict[str, str], min_date: str) -> pd.DataFrame:
    """Pull Form 4 filing metadata for each ticker. Handles older-filings paging.
    Returns DataFrame[ticker, filed, accession]."""
    sess = sec_session()
    rows: list[dict] = []
    min_ts = pd.Timestamp(min_date)
    for i, t in enumerate(tickers):
        cik = cik_map.get(t)
        if not cik:
            continue
        try:
            sub = sess.json(f"https://data.sec.gov/submissions/CIK{cik}.json")
        except Exception as e:
            print(f"[{i+1}/{len(tickers)}] {t} ({cik}) submissions failed: {e}")
            continue
        recent_form4 = _extract_form4(sub)
        files_meta = sub.get("filings", {}).get("files", []) or []
        older_form4 = _fetch_older_form4(sess, cik, files_meta) if files_meta else []
        all_form4 = recent_form4 + older_form4
        kept = 0
        for d, a in all_form4:
            try:
                ts = pd.Timestamp(d)
            except Exception:
                continue
            if ts < min_ts:
                continue
            rows.append({"ticker": t, "filed": d, "accession": a})
            kept += 1
        if (i + 1) % 25 == 0 or i == len(tickers) - 1:
            print(f"[{i+1}/{len(tickers)}] {t}: kept={kept} (recent={len(recent_form4)}, older_idx={len(files_meta)})")
    return pd.DataFrame(rows)


# ---------- features ----------

def build_form4_features(filings: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """For each (ticker, date) in panel, count Form 4 filings in trailing
    1d / 5d / 30d / 90d windows, plus distinct-filing-days in 30d.

    Strictly uses filings filed STRICTLY BEFORE the panel date — pre-open
    features cannot see same-day filings.

    Implementation: pre-sort panel by ticker, run searchsorted per ticker
    against that ticker's sorted filing dates, fill ndarray slices directly.
    """
    print("  preparing inputs...", flush=True)
    f = filings.copy()
    f["filed"] = pd.to_datetime(f["filed"])
    p = panel[["ticker", "date"]].copy()
    p["date"] = pd.to_datetime(p["date"])

    # daily count per (ticker, filed_date)
    daily = f.groupby(["ticker", "filed"]).size().rename("n").reset_index()

    # sort panel by (ticker, date) and track contiguous slices
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    n = len(p)
    tickers = p["ticker"].values
    dates_np = p["date"].values.astype("datetime64[D]")
    # find slice [start, end) per ticker via numpy
    change = np.flatnonzero(np.r_[True, tickers[1:] != tickers[:-1]])
    bounds = np.r_[change, n]

    # group filings by ticker for fast lookup
    filings_by_ticker = {tk: grp.sort_values("filed") for tk, grp in daily.groupby("ticker")}

    wins = (1, 5, 30, 90)
    cnt = {w: np.zeros(n, dtype=np.int32) for w in wins}
    distinct_30 = np.zeros(n, dtype=np.int32)

    print(f"  computing rolling windows for {len(change)} tickers...", flush=True)
    for i, start in enumerate(change):
        end = bounds[i + 1]
        tk = tickers[start]
        slice_dates = dates_np[start:end]
        grp = filings_by_ticker.get(tk)
        if grp is None or len(grp) == 0:
            continue
        filed_arr = grp["filed"].values.astype("datetime64[D]")
        n_arr = grp["n"].values.astype(np.int32)
        cum_n = np.concatenate([[0], np.cumsum(n_arr)])
        cum_d = np.arange(len(n_arr) + 1, dtype=np.int32)  # distinct days = count of filing-day rows
        hi = np.searchsorted(filed_arr, slice_dates - np.timedelta64(1, "D"), side="right")
        for w in wins:
            lo = np.searchsorted(filed_arr, slice_dates - np.timedelta64(w, "D"), side="left")
            cnt[w][start:end] = cum_n[hi] - cum_n[lo]
        lo30 = np.searchsorted(filed_arr, slice_dates - np.timedelta64(30, "D"), side="left")
        distinct_30[start:end] = cum_d[hi] - cum_d[lo30]
        if (i + 1) % 50 == 0 or i == len(change) - 1:
            print(f"    {i+1}/{len(change)}", flush=True)

    out = p.copy()
    for w in wins:
        out[f"form4_cnt_{w}d"] = cnt[w]
    out["form4_distinct_days_30d"] = distinct_30
    return out


# ---------- evaluation ----------

def evaluate_signal(panel_with_features: pd.DataFrame) -> str:
    from scipy.stats import spearmanr
    p = panel_with_features.copy()
    p = p[p["y"].isin([0, 1])].copy()
    lines = []
    lines.append(f"rows: {len(p)}    y=1 rate: {p['y'].mean():.4f}")
    lines.append("")

    # coverage + IC + decile lift
    feat_cols = [c for c in p.columns if c.startswith("form4_")]
    lines.append(f"{'feature':<30s}  {'cov%':>6s}  {'mean':>8s}  {'spearman':>10s}  {'top10%lift':>11s}")
    lines.append("-" * 75)
    for c in feat_cols:
        cov = (p[c] > 0).mean() * 100
        mu = p[c].mean()
        try:
            rho, _ = spearmanr(p[c].values, p["y"].values)
        except Exception:
            rho = np.nan
        # decile lift: top 10% by feature value vs base rate
        thr = p[c].quantile(0.90)
        if thr > 0:
            top = p[p[c] >= thr]
            lift = top["y"].mean() / max(p["y"].mean(), 1e-12)
        else:
            lift = np.nan
        lines.append(f"{c:<30s}  {cov:>5.1f}%  {mu:>8.3f}  {rho:>10.4f}  {lift:>10.2f}x")
    lines.append("")

    # ablation: HistGBC on v6 features ± form4_* (HistGBC is ~10x faster than GBC)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    v6_cols = [
        "rsi_14", "macd", "macd_sig", "macd_hist", "bb_z20", "atr_pct",
        "range_pct", "vol_z", "vol_5d", "rv_60", "skew_60d",
        "semivol_ratio_60d", "up_bigdays_60d", "overnight_gap",
    ]
    df = p.dropna(subset=v6_cols + ["y"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    lines.append(f"ablation rows (after dropna on v6 cols): {len(df)}")

    X_base = df[v6_cols].values
    X_aug = df[v6_cols + feat_cols].values
    y = df["y"].astype(int).values

    tscv = TimeSeriesSplit(n_splits=5)

    def cv(X, label):
        print(f"  fitting {label}...", flush=True)
        ll, auc = [], []
        for fold, (tr, te) in enumerate(tscv.split(X)):
            m = HistGradientBoostingClassifier(
                max_iter=200, max_depth=4, learning_rate=0.05, random_state=0
            )
            m.fit(X[tr], y[tr])
            p_ = m.predict_proba(X[te])[:, 1]
            ll.append(log_loss(y[te], p_, labels=[0, 1]))
            try:
                auc.append(roc_auc_score(y[te], p_))
            except Exception:
                auc.append(np.nan)
            print(f"    fold {fold+1}: ll={ll[-1]:.5f} auc={auc[-1]:.4f}", flush=True)
        return float(np.mean(ll)), float(np.mean(auc))

    ll_b, auc_b = cv(X_base, "baseline (v6 only)")
    ll_a, auc_a = cv(X_aug, "augmented (v6 + form4_*)")
    lines.append("")
    lines.append("5-fold time-series CV (HistGBC, max_iter=200, depth=4, lr=0.05):")
    lines.append(f"  baseline  v6 only          : log_loss={ll_b:.5f}  AUC={auc_b:.4f}")
    lines.append(f"  augmented v6 + form4_*     : log_loss={ll_a:.5f}  AUC={auc_a:.4f}")
    lines.append(f"  Δ log_loss (lower=better)  : {ll_a - ll_b:+.5f}")
    lines.append(f"  Δ AUC      (higher=better) : {auc_a - auc_b:+.5f}")

    # also report per-fold AUC for stability
    return "\n".join(lines)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pull", action="store_true",
                    help="skip SEC fetch; use cached data/form4_filings.csv")
    args = ap.parse_args()

    panel = pd.read_csv(PANEL_PATH)
    cik_map = pd.read_csv(CIK_MAP_PATH, dtype={"cik": str})
    cik_map["cik"] = cik_map["cik"].str.zfill(10)
    cik_dict = dict(zip(cik_map["ticker"], cik_map["cik"]))

    tickers = sorted(set(panel["ticker"].unique()) & set(cik_dict))
    print(f"panel tickers: {panel['ticker'].nunique()}, with CIK: {len(tickers)}")
    min_date = panel["date"].min()  # 3y back from panel

    if args.skip_pull and FORM4_OUT.exists():
        print(f"[skip-pull] loading {FORM4_OUT}")
        filings = pd.read_csv(FORM4_OUT)
    else:
        # back-pad min_date by 90d to give 30d/90d windows valid at panel start
        pull_min = (pd.Timestamp(min_date) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
        print(f"pulling Form 4 filings since {pull_min} for {len(tickers)} tickers...")
        filings = pull_form4_filings(tickers, cik_dict, pull_min)
        FORM4_OUT.parent.mkdir(parents=True, exist_ok=True)
        filings.to_csv(FORM4_OUT, index=False)
        print(f"saved {len(filings)} filings to {FORM4_OUT}")

    print(f"\nbuilding features for {len(panel)} panel rows...")
    feats = build_form4_features(filings, panel)
    panel["date"] = pd.to_datetime(panel["date"])
    feats["date"] = pd.to_datetime(feats["date"])
    merged = panel.merge(feats, on=["ticker", "date"], how="left")
    feat_cols = [c for c in feats.columns if c.startswith("form4_")]
    merged[feat_cols] = merged[feat_cols].fillna(0)
    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PANEL_OUT, index=False)
    print(f"saved augmented panel to {PANEL_OUT}")

    print("\nevaluating signal...")
    report = evaluate_signal(merged)
    print(report)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(report + "\n")
    print(f"\nreport saved to {REPORT_OUT}")


if __name__ == "__main__":
    main()
