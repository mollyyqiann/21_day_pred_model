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
EXIT   take profit +12%; or stop loss -15%; or out of top-15 for 2
       consecutive publication days (minimum 2-day hold); or 21 trading days.
       TP and SL are checked once daily at 15:45 against the same-session
       price and are exempt from the minimum hold.
SIZE   8 slots, equal weight, fractional (dollar) orders -- regular hours only.

TP is +12% as of 2026-09-09 (previously +30%). Read the note above TAKE_PROFIT
before changing it: the two levels come from analyses that contradict each
other and the disagreement is NOT settled.

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
import os, sys, json, subprocess, argparse
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
# Take profit. 0.12 is the v1 level from code/121_mg_trade_plan.py, adopted
# 2026-09-09 on the instruction to "hook 121 onto robinhood": once the two
# scripts were diffed, TAKE_PROFIT was the ONLY rule that differed between
# them -- slots, 21-day cap, dropout, min-hold and all three extension
# thresholds are identical -- so running 121's rule set live means this number
# and nothing else. Attaching 121 itself to the broker would instead put two
# schedulers on one account, both writing mg_paper_positions.json and both
# buying the same top-5 names.
#
# ⚠ CONTRADICTION, UNRESOLVED. This file's docstring used to argue for 0.30 on
# the grounds that TP30 > TP20 > TP12 under every realistic fill tested. A
# later analysis says the opposite: +30% fires on only ~11% of picks, while the
# measured MFE of FAVOURABLE picks is a median +12.27%, peaking around day 14
# and giving back ~10.4pp from peak to close. Both cannot be right, and which
# one is has not been settled. It is a single number: override it with
# MG_TAKE_PROFIT=0.30 without editing this file, or change the default here.
TAKE_PROFIT = float(os.environ.get("MG_TAKE_PROFIT") or 0.12)

# Stop loss, added 2026-09-10. Before this the strategy had NO adverse-move
# exit at all: the dropout rule is structurally unreachable (raw_margin is a
# volatility sort with dd_60d corr -0.71, so a position that falls hard ranks
# HIGHER and re-entrenches in the top-15 -- 13 of today's top-15 are >25%
# below their 60-day high), which left "went up 12%" and "21 days passed" as
# the only ways out. A position could halve and simply sit there.
#
# -15% is sized from the book itself, not backtested: the top-15's median
# daily ATR is 5.3% (twice the SP500 median), so -15% = 2.8x ATR -- inside the
# classic 2-4x band; anything under ~2x (-8%, -10%) is daily noise for these
# names. Two honest caveats: (1) UNBACKTESTED -- several MG rules reversed
# under proper controls, this one has not been tested at all; (2) it is
# evaluated once a day at 15:45 against the same-session price, so a gap
# through the level exits at the 15:55 price, not at -15%.
# Override with MG_STOP_LOSS (e.g. 0.12); 0 disables it.
STOP_LOSS = float(os.environ.get("MG_STOP_LOSS") or 0.15)
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

    # Same-session prices for the held names. This is the take-profit input:
    # without it TAKE_PROFIT was a printed footer with no code behind it, and
    # since the model re-picks its own holdings (they stay in the top-15, so
    # the dropout rule never fires either) the only reachable exit was the
    # 21-day cap. Positions could only ever accumulate.
    prices = dict(zip(d.ticker, d.close))
    today = datetime.now().date().isoformat()

    exits, warnings = [], []
    for t, p in openp.items():
        # Day counting must be idempotent. build() used to increment on every
        # invocation, so a manual run -- or the 12:49 executor retry on
        # 2026-09-09 -- silently aged every position by an extra day and would
        # have tripped the 21-day cap at roughly day 10.
        if p.get("last_counted") != today:
            p["miss_base"] = int(p.get("days_out_of_top15", 0))
            p["days_held"] = int(p.get("days_held", 0)) + 1
        held = int(p.get("days_held", 0))
        miss = 0 if t in top15 else int(p.get("miss_base", 0)) + 1

        px, entry = prices.get(t), p.get("entry_price")
        gain = (float(px) / float(entry) - 1.0) if px is not None and entry else None
        if gain is None:
            # A held name absent from the score file cannot be price-checked,
            # so its take-profit silently stops working. Say so out loud.
            warnings.append(f"{t}: no price in today's score file — take-profit NOT evaluated")

        why = None
        if gain is not None and gain >= TAKE_PROFIT:
            # Take profit outranks everything and ignores MIN_HOLD_DAYS: the
            # target is hit, the reason to hold is gone.
            why = f"take profit {gain*100:+.1f}%"
        elif STOP_LOSS and gain is not None and gain <= -STOP_LOSS:
            # Stop loss: same priority logic as the take profit and likewise
            # exempt from MIN_HOLD_DAYS -- a day-1 crash is exactly the case
            # it exists for. The dropout rule cannot do this job (see the
            # STOP_LOSS note above): falling RAISES a name's rank here.
            why = f"stop loss {gain*100:+.1f}%"
        elif held >= MAX_HOLD_DAYS:
            why = "21-day cap"
        elif held > MIN_HOLD_DAYS and miss >= DROPOUT_DAYS:
            why = f"out of top-15 for {miss}d"
        if why:
            exits.append({"ticker": t, "reason": why, "days_held": held,
                          "gain": round(gain, 4) if gain is not None else None})

        p["days_out_of_top15"] = miss
        p["last_counted"] = today

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
            "free": free, "open": len(openp), "state": st,
            "warnings": warnings, "prices": prices}


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
    if p.get("warnings"):
        L += [""] + [f"  !! {w}" for w in p["warnings"]]
    L += ["", f"BUY ({len(p['entries'])}):"]
    if p["entries"]:
        for e in p["entries"]:
            L.append(f"  + {e['ticker']:<6} ~${SLOT_DOLLARS} fractional  (last ${e['close']:.2f}, "
                     f"margin {e['raw_margin']:+.2f})")
    else:
        L.append("  (none — no FAVOURABLE first-time names, or no free slot)")
    L += ["", f"Standing exits: +{TAKE_PROFIT*100:.0f}% take profit | "
              f"-{STOP_LOSS*100:.0f}% stop loss | "
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
                   "days_held": e["days_held"], "gain": e.get("gain")}
                  for e in plan["exits"]],
        "warnings": plan.get("warnings", []),
        "buys": [{"ticker": e["ticker"], "dollar_amount": SLOT_DOLLARS,
                  "last": round(float(e["close"]), 2),
                  "raw_margin": round(float(e["raw_margin"]), 4)}
                 for e in plan["entries"]],
    }
    body = json.dumps(pj, indent=2)
    (PLANS / f"closeplan_{stamp}.json").write_text(body)
    (PLANS / "latest.json").write_text(body)
    STATE.write_text(json.dumps(plan["state"], indent=2, default=str))
    # Shadow ledger: exit-rule B' (sell HALF at +12%, ride the rest) tracked in
    # parallel on the same entry stream -- see 126_mg_shadow_b.py. It places
    # nothing and keeps its own state; the try/except means a shadow bug can
    # never break the live plan. Its one-liner is appended to the plan text
    # BEFORE --notify so the daily Telegram carries the A-vs-B' comparison.
    try:
        _sp2 = _ilu.spec_from_file_location("_shadow", ROOT / "code" / "126_mg_shadow_b.py")
        _shadow = _ilu.module_from_spec(_sp2); _sp2.loader.exec_module(_shadow)
        _line = _shadow.update(plan["watch"], plan["prices"], stamp)
        if _line:
            print(_line)
            txt += "\n" + _line
            (PLANS / f"closeplan_{stamp}.txt").write_text(txt)
            (PLANS / "latest.txt").write_text(txt)
    except Exception as _e:
        print(f"[123] shadow ledger failed (non-fatal): {_e}")
    if a.notify:
        subprocess.run([PY, str(ROOT / "code" / "122_mg_plan_notify.py")], cwd=ROOT)
