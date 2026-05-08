"""Phase A.2: Base rate descriptive study.

Slices the y21 base rate by year, calendar month, sector, VIX regime,
rv_60 quartile, run_length bucket. Computes ticker-concentration metrics
and a "fresh vs continuation" event analysis.

Writes:
  output/monthly_gainer/baserate.json
  output/monthly_gainer/baserate_table.md
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output" / "monthly_gainer"
OUT.mkdir(parents=True, exist_ok=True)


def slice_rate(df, label_col="y21"):
    """Return (n_rows, n_pos, base_rate)."""
    n = len(df)
    pos = int(df[label_col].sum()) if n else 0
    rate = pos / n if n else float("nan")
    return n, pos, rate


def main():
    panel = pd.read_csv(DATA / "monthly_gainer_panel.csv", parse_dates=["date"])
    lab = panel[panel["y21"] >= 0].reset_index(drop=True)

    n_total, n_pos, br = slice_rate(lab)
    print(f"[81] overall: n={n_total:,}  pos={n_pos:,}  base_rate={br:.4%}")

    results = {
        "overall": {"n": n_total, "pos": n_pos, "base_rate": br},
        "by_year": {},
        "by_month": {},
        "by_sector": {},
        "by_vix_regime": {},
        "by_rv60_quartile": {},
        "by_run_length_bucket": {},
        "concentration": {},
        "event_clustering": {},
    }

    # by year
    for y, g in lab.groupby(lab["date"].dt.year):
        n, p, r = slice_rate(g)
        results["by_year"][int(y)] = {"n": n, "pos": p, "base_rate": r}

    # by calendar month (1-12)
    for m, g in lab.groupby(lab["date"].dt.month):
        n, p, r = slice_rate(g)
        results["by_month"][int(m)] = {"n": n, "pos": p, "base_rate": r}

    # by sector
    for s, g in lab.groupby("sector"):
        n, p, r = slice_rate(g)
        top = (g[g["y21"] == 1]["ticker"].value_counts().head(3)).to_dict()
        results["by_sector"][str(s)] = {"n": n, "pos": p, "base_rate": r,
                                        "top_tickers": top}

    # by vix regime (terciles, computed on the labeled subset)
    # need vix attached: load from regime_features
    sys.path.insert(0, str(ROOT / "code"))
    from regime_features import load_regime_frame  # noqa: E402
    rf = load_regime_frame()
    lab_v = lab.merge(rf[["date", "vix"]], on="date", how="left")
    vix_q = lab_v["vix"].quantile([1/3, 2/3]).values
    def _vix_bucket(v):
        if pd.isna(v):
            return "unknown"
        if v < vix_q[0]:
            return "low"
        if v < vix_q[1]:
            return "mid"
        return "high"
    lab_v["vix_bucket"] = lab_v["vix"].apply(_vix_bucket)
    for b, g in lab_v.groupby("vix_bucket"):
        n, p, r = slice_rate(g)
        results["by_vix_regime"][b] = {"n": n, "pos": p, "base_rate": r,
                                       "vix_lo_hi": [float(g["vix"].min()), float(g["vix"].max())]}

    # by rv_60 quartile
    rv_q = lab["rv_60"].quantile([0.25, 0.5, 0.75]).values
    def _rv_bucket(v):
        if pd.isna(v):
            return "unknown"
        if v < rv_q[0]:
            return "Q1"
        if v < rv_q[1]:
            return "Q2"
        if v < rv_q[2]:
            return "Q3"
        return "Q4"
    lab_rv = lab.copy()
    lab_rv["rv_bucket"] = lab_rv["rv_60"].apply(_rv_bucket)
    for b, g in lab_rv.groupby("rv_bucket"):
        n, p, r = slice_rate(g)
        results["by_rv60_quartile"][b] = {
            "n": n, "pos": p, "base_rate": r,
            "rv_lo_hi": [float(g["rv_60"].min()), float(g["rv_60"].max())] if len(g) else None
        }

    # by run_length bucket
    for lo, hi, lbl in [(0, 5, "0-5"), (5, 20, "5-20"), (20, 60, "20-60"), (60, 9999, "60+")]:
        g = lab[(lab["run_length"] >= lo) & (lab["run_length"] < hi)]
        n, p, r = slice_rate(g)
        results["by_run_length_bucket"][lbl] = {"n": n, "pos": p, "base_rate": r}

    # concentration
    pos_rows = lab[lab["y21"] == 1]
    tk_counts = pos_rows["ticker"].value_counts()
    total_pos = int(tk_counts.sum())
    for k in (10, 25, 50, 100):
        share = float(tk_counts.head(k).sum() / total_pos) if total_pos else 0.0
        results["concentration"][f"top_{k}_share"] = share
    results["concentration"]["n_tickers_with_any_positive"] = int((tk_counts > 0).sum())
    results["concentration"]["n_total_tickers"] = int(lab["ticker"].nunique())
    results["concentration"]["top_15_tickers"] = tk_counts.head(15).to_dict()

    # event clustering: of all positive rows, what fraction had another positive
    # in the same ticker in [t-5, t-1]? "continuation" = yes; "fresh" = no.
    cont = 0
    fresh = 0
    for tk, g in pos_rows.groupby("ticker"):
        ds = g["date"].sort_values().values
        for i, d in enumerate(ds):
            recent = (d - ds[:i]).astype("timedelta64[D]").astype(int)
            if (recent > 0).any() and ((recent > 0) & (recent <= 7)).any():
                cont += 1
            else:
                fresh += 1
    results["event_clustering"] = {
        "fresh_events": fresh, "continuation_events": cont,
        "fresh_share": fresh / (fresh + cont) if (fresh + cont) else float("nan"),
    }

    (OUT / "baserate.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"[81] wrote {OUT / 'baserate.json'}")

    # markdown table
    lines = []
    lines.append("# Monthly Gainer Base Rate Study\n")
    lines.append(f"**Overall**: {n_total:,} labeled rows, {n_pos:,} positives, base rate **{br:.4%}**\n")
    lines.append("## By year\n")
    lines.append("| year | n | pos | rate |\n|---|---:|---:|---:|")
    for y, d in sorted(results["by_year"].items()):
        lines.append(f"| {y} | {d['n']:,} | {d['pos']:,} | {d['base_rate']:.3%} |")
    lines.append("\n## By GICS sector\n")
    lines.append("| sector | n | pos | rate | top tickers |\n|---|---:|---:|---:|---|")
    for s, d in sorted(results["by_sector"].items(), key=lambda kv: -kv[1]["base_rate"]):
        top = ", ".join(f"{k}({v})" for k, v in d["top_tickers"].items())
        lines.append(f"| {s} | {d['n']:,} | {d['pos']:,} | {d['base_rate']:.3%} | {top} |")
    lines.append("\n## By VIX regime (terciles)\n")
    lines.append("| regime | vix range | n | pos | rate |\n|---|---|---:|---:|---:|")
    for b in ["low", "mid", "high", "unknown"]:
        if b not in results["by_vix_regime"]:
            continue
        d = results["by_vix_regime"][b]
        rng = d.get("vix_lo_hi", [None, None])
        rng_s = f"{rng[0]:.1f}-{rng[1]:.1f}" if rng[0] is not None else "—"
        lines.append(f"| {b} | {rng_s} | {d['n']:,} | {d['pos']:,} | {d['base_rate']:.3%} |")
    lines.append("\n## By rv_60 quartile (stock-level vol)\n")
    lines.append("| bucket | rv range | n | pos | rate |\n|---|---|---:|---:|---:|")
    for b in ["Q1", "Q2", "Q3", "Q4", "unknown"]:
        if b not in results["by_rv60_quartile"]:
            continue
        d = results["by_rv60_quartile"][b]
        rng = d.get("rv_lo_hi") or [None, None]
        rng_s = f"{rng[0]:.2f}-{rng[1]:.2f}" if rng[0] is not None else "—"
        lines.append(f"| {b} | {rng_s} | {d['n']:,} | {d['pos']:,} | {d['base_rate']:.3%} |")
    lines.append("\n## By run_length bucket (already-trending state)\n")
    lines.append("| bucket | n | pos | rate |\n|---|---:|---:|---:|")
    for lbl in ["0-5", "5-20", "20-60", "60+"]:
        d = results["by_run_length_bucket"][lbl]
        lines.append(f"| {lbl} | {d['n']:,} | {d['pos']:,} | {d['base_rate']:.3%} |")
    lines.append("\n## Concentration\n")
    lines.append(f"- {results['concentration']['n_tickers_with_any_positive']} of "
                 f"{results['concentration']['n_total_tickers']} tickers ever fired (≥1 positive)")
    for k in (10, 25, 50, 100):
        s = results['concentration'][f'top_{k}_share']
        lines.append(f"- top-{k} tickers contribute **{s:.1%}** of all positives")
    lines.append("\n**Top 15 tickers by positive count:**")
    lines.append("| ticker | positives |\n|---|---:|")
    for tk, c in results['concentration']['top_15_tickers'].items():
        lines.append(f"| {tk} | {c} |")
    lines.append("\n## Event clustering\n")
    ec = results["event_clustering"]
    lines.append(f"- Fresh events (no positive in prior 7 days): {ec['fresh_events']:,} ({ec['fresh_share']:.1%})")
    lines.append(f"- Continuation events: {ec['continuation_events']:,}")

    (OUT / "baserate_table.md").write_text("\n".join(lines))
    print(f"[81] wrote {OUT / 'baserate_table.md'}")
    print(f"[81] done")


if __name__ == "__main__":
    main()
