"""Monthly Gainer v3 — 15:45 ET scorer + same-day trade plan.

*** THIS SCRIPT PLACES NO ORDERS -- but the 15:55 job that reads its output
*** DOES. It writes trade_plans/latest.json, and com.user.mgexecute
*** (daily/run_mg_execute.sh, 15:55 ET) submits exactly those orders into the
*** Agentic Robinhood account. Changing what lands in `entries`/`exits` here
*** changes what gets bought with real money ten minutes later.

WHY 15:45 (measured, not assumed)
---------------------------------
The old 19:00 job produced a plan you could only act on the NEXT morning, and
these picks gap up overnight -- next-open entry cost -1.87pp versus the close
the backtests assumed, with 3.47% of price dispersion you cannot control.

Scoring late in the SAME session removes that:
  fill point            slip vs official close      sd
  15:55 same day               -0.04%             0.39%
  09:30 next day               +0.70%             3.47%

And scoring 15 minutes early barely changes the decision. Across 18,142
ticker-days the 15:45 provisional bar vs the true close agrees on:
  extension flag count      94.39%
  FAVOURABLE vs not         97.72%      (dd60 corr 0.871, p5 corr 0.935)
The 413 disagreements sit on the dd60 = -0.10 boundary, as expected.

But note the 15:45 PRICE runs ~+0.43% above the close on average, so score at
15:45 and FILL at 15:55 -- the decision is stable over those ten minutes while
the entry price improves by roughly that much.

RULES ENCODED
-------------
ENTRY  top-5 by raw_margin, first appearance of that ticker only,
       extension indicator FAVOURABLE (0 of 3 flags), free slot.
EXIT   take profit +30%; or out of top-15 for 2 consecutive publication days
       (minimum 2-day hold); or 21 trading days.
SIZE   8 slots, equal weight, fractional (dollar) orders -- regular hours only.

TP is +30%, not the +12% an earlier pass suggested. That +12% was an artifact
of an unrealistic same-close entry; under every realistic fill tested
(09:30 through 15:55) the ordering was TP30 > TP20 > TP12.

STANDING CAVEAT
---------------
The underlying model has no demonstrated edge. Corrected for survivorship and
look-ahead its volatility sort returns -1.59pp. This configuration was selected
by a long search over ~2 months and ~25 trades; several results in that search
reversed under proper controls. Paper-trade it against the index before funding.

Usage:
    python 123_mg_close_scorer.py             # score now, print the plan
    python 123_mg_close_scorer.py --notify    # also send it to Telegram
Schedule: weekdays 15:45 ET.
"""
import sys, json, subprocess, argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "monthly_gainer"
PLANS = OUT / "trade_plans"
STATE = ROOT / "data" / "mg_paper_positions.json"
PY = sys.executable

SLOTS = 8
SLOT_DOLLARS = 125
TAKE_PROFIT = 0.30
MAX_HOLD_DAYS = 21
DROPOUT_DAYS = 2
MIN_HOLD_DAYS = 2
FILL_WINDOW = "15:55 ET"

# The strategy trades ONLY the dedicated Agentic Robinhood account. The 15:55
# executor (daily/run_mg_execute.sh) refuses to act on any other account, and
# 124_mg_reconcile.py checks the same number. Resolved from
# config/mg_execution.json so the number stays out of the public repo.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mg_account import account as _account
AGENTIC_ACCOUNT = _account()

EXT_NEAR_HIGH, EXT_RAN_UP, EXT_HIGH_CONF = -0.10, 0.02, 0.33


def flags(r):
    f = []
    if pd.notna(r.get("dd_60d")) and r["dd_60d"] > EXT_NEAR_HIGH: f.append("near 60d high")
    if pd.notna(r.get("ret_5d_lag")) and r["ret_5d_lag"] > EXT_RAN_UP: f.append("up >2% in 5d")
    if pd.notna(r.get("prob_cal")) and r["prob_cal"] > EXT_HIGH_CONF: f.append("high model conf")
    return f


def band(n): return "FAVOURABLE" if n == 0 else ("NEUTRAL" if n == 1 else "EXTENDED")


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"positions": [], "ever_entered": []}


def rescore():
    """Refresh the score on the current (near-complete) session."""
    r = subprocess.run([PY, str(ROOT / "code" / "101_refresh_score_today.py")],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"[123] WARNING: rescore exited {r.returncode}; using the existing score file")
    return r.returncode == 0


def build():
    d = pd.read_csv(OUT / "today_score_fresh_sp500.csv")
    asof = str(d["date"].iloc[0]) if "date" in d.columns else "?"
    ranked = d.nlargest(15, "raw_margin").reset_index(drop=True)
    top15 = set(ranked.ticker)
    st = load_state()
    openp = {p["ticker"]: p for p in st["positions"]}
    ever = set(st["ever_entered"])

    exits = []
    for t, p in openp.items():
        held = int(p.get("days_held", 0)) + 1
        miss = 0 if t in top15 else int(p.get("days_out_of_top15", 0)) + 1
        why = None
        if held >= MAX_HOLD_DAYS: why = "21-day cap"
        elif held > MIN_HOLD_DAYS and miss >= DROPOUT_DAYS: why = f"out of top-15 for {miss}d"
        if why: exits.append({"ticker": t, "reason": why, "days_held": held})
        p["days_held"], p["days_out_of_top15"] = held, miss

    free = SLOTS - (len(openp) - len(exits))
    entries, watch = [], []
    for r in ranked.head(5).to_dict("records"):
        fl = flags(r); b = band(len(fl))
        watch.append({"ticker": r["ticker"], "band": b, "flags": fl,
                      "held": r["ticker"] in openp})
        if b == "FAVOURABLE" and r["ticker"] not in openp and r["ticker"] not in ever:
            entries.append(r)
    entries = entries[:max(0, free)]
    return {"asof": asof, "exits": exits, "entries": entries, "watch": watch,
            "free": free, "open": len(openp), "state": st}


def render(p):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [f"MG v3 — SAME-DAY TRADE PLAN   scored {stamp} ET   (data asof {p['asof']})",
         "=" * 68,
         f"*** AUTO-EXECUTES at {FILL_WINDOW}. Model has no proven edge. ***",
         "",
         f"Slots {SLOTS} | open {p['open']} | free after exits {p['free']}",
         "", "TOP-5 + INDICATOR:"]
    for w in p["watch"]:
        m = {"FAVOURABLE": "++", "NEUTRAL": " ~", "EXTENDED": "--"}[w["band"]]
        L.append(f"  {m} {w['ticker']:<6} {w['band']:<11}"
                 + (f"  {', '.join(w['flags'])}" if w["flags"] else "")
                 + ("  [HELD]" if w["held"] else ""))
    L += ["", f"SELL ({len(p['exits'])}):"]
    L += [f"  - {e['ticker']:<6} {e['reason']} (held {e['days_held']}d)" for e in p["exits"]] or ["  (none)"]
    L += ["", f"BUY ({len(p['entries'])}):"]
    if p["entries"]:
        for e in p["entries"]:
            L.append(f"  + {e['ticker']:<6} ~${SLOT_DOLLARS} fractional  (last ${e['close']:.2f}, "
                     f"margin {e['raw_margin']:+.2f})")
    else:
        L.append("  (none — no FAVOURABLE first-time names, or no free slot)")
    L += ["", f"Standing exits: +{TAKE_PROFIT*100:.0f}% take profit | "
              f"2 days out of top-15 | {MAX_HOLD_DAYS}-day cap",
          f"Executor fires at {FILL_WINDOW} and sends its own receipt.",
          "To stop it:  launchctl unload ~/Library/LaunchAgents/com.user.mgexecute.plist"]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--notify", action="store_true")
    ap.add_argument("--skip-rescore", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="score even on a non-trading day (manual use only; "
                         "the plan it writes will NOT be executed -- the 15:55 "
                         "job runs the same calendar check)")
    a = ap.parse_args()

    # Only score on days the NYSE is open through 16:00. On a holiday or a
    # 13:00 early close the "near-complete session" this script assumes is
    # actually the previous one, so the plan would be built on stale bars and
    # then Telegrammed as if it were live. Bailing before anything is written
    # also leaves trade_plans/latest.json on yesterday's date, which is the
    # 15:55 executor's second, independent reason to stand down.
    sys.path.insert(0, str(ROOT / "code"))
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_td", ROOT / "code" / "125_trading_day.py")
    _td = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_td)
    _cal = _td.check(datetime.now().date())
    if not _cal["tradable"] and not a.force:
        print(f"[123] {_cal['date']}: {_cal['reason']} -> no plan written, nothing sent")
        sys.exit(0)
    if not _cal["tradable"]:
        print(f"[123] WARNING: {_cal['reason']} — running anyway (--force)")

    if not a.skip_rescore:
        rescore()
    plan = build()
    txt = render(plan)
    print(txt)
    PLANS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    (PLANS / f"closeplan_{stamp}.txt").write_text(txt)
    (PLANS / "latest.txt").write_text(txt)
    # Machine-readable twin of the text plan. The 15:55 executor reads THIS --
    # parsing the prose would make a rendering tweak silently change what gets
    # traded. Written after the text so the two can never disagree.
    pj = {
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "date": stamp,
        "asof": plan["asof"],
        "account": AGENTIC_ACCOUNT,
        "slots": SLOTS,
        "open": plan["open"],
        "free_after_exits": plan["free"],
        "slot_dollars": SLOT_DOLLARS,
        "sells": [{"ticker": e["ticker"], "reason": e["reason"],
                   "days_held": e["days_held"]} for e in plan["exits"]],
        "buys": [{"ticker": e["ticker"], "dollar_amount": SLOT_DOLLARS,
                  "last": round(float(e["close"]), 2),
                  "raw_margin": round(float(e["raw_margin"]), 4)}
                 for e in plan["entries"]],
    }
    body = json.dumps(pj, indent=2)
    (PLANS / f"closeplan_{stamp}.json").write_text(body)
    (PLANS / "latest.json").write_text(body)
    STATE.write_text(json.dumps(plan["state"], indent=2, default=str))
    if a.notify:
        subprocess.run([PY, str(ROOT / "code" / "122_mg_plan_notify.py")], cwd=ROOT)
