"""Monthly Gainer v3 — daily PAPER trade plan generator.

*** THIS PLACES NO ORDERS. It prints a plan for a human to review and act on. ***

=============================================================================
READ THIS BEFORE USING IT
=============================================================================
The underlying model has NO demonstrated edge. Established 2026-08-25/26 over a
purged walk-forward on 13 years of point-in-time data:
  - the model is a volatility sort; `df.nlargest(5,'atr_pct')` matches it
  - corrected for survivorship + look-ahead, that volatility sort returns -1.59pp
  - its own confidence is ANTI-predictive: high-conviction picks touched +30%
    3.9% of the time vs 17.6% for low-conviction ones
  - live top-5 returned -4.82% vs +2.68% for the index on the same dates

This routine encodes the least-bad configuration found, on ~2 months and ~25
trades, with p-values around 0.05-0.5. Several results in that search REVERSED
when properly controlled. Treat its output as a hypothesis to paper-trade, not
as advice. See scratchpad/target_rework/PLAN.md (22 findings) for the audit.

=============================================================================
THE RULES IT ENCODES
=============================================================================
ENTRY   - name is in today's top-5 by raw_margin, AND
          it is the FIRST time that ticker has appeared (no re-entry), AND
          its extension indicator is FAVOURABLE (0 of 3 flags), AND
          a slot is free.
EXIT    - take profit at +12% (whichever comes first), else
          out of the top-15 for 2 consecutive publication days, else
          21 trading days elapsed.
SIZING  - 8 equal slots. Below ~8 the outcome is driven by which signals
          happened to arrive when capital was free, not by the rules.

Why +12% and not +30%: measured MFE of FAVOURABLE picks is a median +12.27%,
peaking around day 14, and they give back ~10.4pp from peak to close. Capping
near the median peak raised total P&L in BOTH model eras; +20% and +30% caps
were negative in era B. This is about capital turnover, not win rate.

Usage:
    python 121_mg_trade_plan.py            # print today's plan
    python 121_mg_trade_plan.py --json     # machine-readable
State: data/mg_paper_positions.json  (positions you tell it you opened)
"""
import sys, json, argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "monthly_gainer"
STATE = ROOT / "data" / "mg_paper_positions.json"

SLOTS = 8
TAKE_PROFIT = 0.12
MAX_HOLD_DAYS = 21
DROPOUT_DAYS = 2          # consecutive publication days outside the top-15
MIN_HOLD_DAYS = 2

# extension indicator (see 101_refresh_score_today.py for the fitted evidence)
EXT_NEAR_HIGH, EXT_RAN_UP, EXT_HIGH_CONF = -0.10, 0.02, 0.33


def flags(row):
    f = []
    if pd.notna(row.get("dd_60d")) and row["dd_60d"] > EXT_NEAR_HIGH: f.append("near 60d high")
    if pd.notna(row.get("ret_5d_lag")) and row["ret_5d_lag"] > EXT_RAN_UP: f.append("up >2% in 5d")
    if pd.notna(row.get("prob_cal")) and row["prob_cal"] > EXT_HIGH_CONF: f.append("high model conf")
    return f


def band(n): return "FAVOURABLE" if n == 0 else ("NEUTRAL" if n == 1 else "EXTENDED")


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"positions": [], "ever_entered": []}


def save_state(s): STATE.write_text(json.dumps(s, indent=2, default=str))


def build_plan(score_csv=None):
    score_csv = score_csv or (OUT / "today_score_fresh_sp500.csv")
    d = pd.read_csv(score_csv)
    asof = str(d["date"].iloc[0]) if "date" in d.columns else "unknown"
    ranked = d.nlargest(15, "raw_margin").reset_index(drop=True)
    top5 = set(ranked.head(5).ticker)
    top15 = set(ranked.ticker)
    st = load_state()
    open_pos = {p["ticker"]: p for p in st["positions"]}
    ever = set(st["ever_entered"])

    exits = []
    for t, p in open_pos.items():
        held = int(p.get("days_held", 0)) + 1
        miss = int(p.get("days_out_of_top15", 0)) + (0 if t in top15 else 1)
        if t in top15: miss = 0
        why = None
        if held >= MAX_HOLD_DAYS: why = f"21-day cap reached"
        elif held > MIN_HOLD_DAYS and miss >= DROPOUT_DAYS: why = f"out of top-15 for {miss} days"
        if why: exits.append({"ticker": t, "reason": why, "days_held": held,
                              "entry_price": p.get("entry_price")})
        p["days_held"] = held; p["days_out_of_top15"] = miss

    free = SLOTS - (len(open_pos) - len(exits))
    entries = []
    for r in ranked.head(5).to_dict("records"):
        t = r["ticker"]
        if t in open_pos or t in ever: continue
        fl = flags(r); b = band(len(fl))
        if b != "FAVOURABLE": continue
        entries.append({"ticker": t, "band": b, "close": r.get("close"),
                        "raw_margin": r.get("raw_margin"), "sector": r.get("sector")})
    entries = entries[:max(0, free)]

    watch = []
    for r in ranked.head(5).to_dict("records"):
        fl = flags(r)
        watch.append({"ticker": r["ticker"], "band": band(len(fl)), "flags": fl,
                      "held": r["ticker"] in open_pos})
    return {"asof": asof, "slots": SLOTS, "open": len(open_pos), "free_after_exits": free,
            "exits": exits, "entries": entries, "top5": watch,
            "take_profit_pct": TAKE_PROFIT, "state": st}


def render(plan):
    L = [f"MONTHLY GAINER v3 — PAPER TRADE PLAN   as of {plan['asof']}",
         f"{'='*70}",
         "*** NO ORDERS ARE PLACED. Review before acting. The model has no",
         "*** demonstrated edge -- see the docstring in this file.",
         "",
         f"Capital: {plan['slots']} slots | open {plan['open']} | free after exits {plan['free_after_exits']}",
         "",
         "TODAY'S TOP-5 WITH INDICATOR:"]
    for w in plan["top5"]:
        mark = {"FAVOURABLE": "++", "NEUTRAL": " ~", "EXTENDED": "--"}[w["band"]]
        held = "  [HELD]" if w["held"] else ""
        L.append(f"  {mark} {w['ticker']:<6} {w['band']:<11}"
                 + (f"  flags: {', '.join(w['flags'])}" if w["flags"] else "")+held)
    L += ["", f"SELL ({len(plan['exits'])}):"]
    if plan["exits"]:
        for e in plan["exits"]:
            L.append(f"  - {e['ticker']:<6} {e['reason']}  (held {e['days_held']}d)")
    else:
        L.append("  (none)")
    L += ["", f"BUY ({len(plan['entries'])}):"]
    if plan["entries"]:
        for e in plan["entries"]:
            L.append(f"  + {e['ticker']:<6} FAVOURABLE  close ${e['close']:.2f}  "
                     f"margin {e['raw_margin']:+.2f}")
    else:
        L.append("  (none — no FAVOURABLE first-time names, or no free slot)")
    L += ["", f"STANDING EXIT ON EVERY OPEN POSITION: take profit at "
              f"+{plan['take_profit_pct']*100:.0f}%, else out-of-top-15 for 2 days, else 21 days.",
          "", "To record what you actually did:  python 121_mg_trade_plan.py --confirm TICKER PRICE"]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--confirm", nargs=2, metavar=("TICKER", "PRICE"))
    ap.add_argument("--close", nargs=1, metavar="TICKER")
    a = ap.parse_args()
    if a.confirm:
        st = load_state()
        st["positions"].append({"ticker": a.confirm[0].upper(),
                                "entry_price": float(a.confirm[1]),
                                "entry_date": datetime.now().date().isoformat(),
                                "days_held": 0, "days_out_of_top15": 0})
        st["ever_entered"] = sorted(set(st["ever_entered"]) | {a.confirm[0].upper()})
        save_state(st); print(f"recorded position in {a.confirm[0].upper()}"); sys.exit()
    if a.close:
        st = load_state()
        st["positions"] = [p for p in st["positions"] if p["ticker"] != a.close[0].upper()]
        save_state(st); print(f"closed {a.close[0].upper()}"); sys.exit()
    plan = build_plan()
    if a.json:
        plan.pop("state", None); print(json.dumps(plan, indent=2, default=str))
    else:
        txt = render(plan)
        print(txt)
        # persist a dated copy plus a stable "latest" so the 19:00 routine
        # leaves a reviewable record rather than only a log line
        pdir = OUT / "trade_plans"
        pdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d")
        (pdir / f"plan_{stamp}.txt").write_text(txt)
        (pdir / "latest.txt").write_text(txt)
        save_state(plan["state"])
