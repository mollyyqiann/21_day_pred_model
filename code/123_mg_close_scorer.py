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

CORRECTED 2026-09-10. This block used to claim the 15:45 price runs ~+0.43%
above the close, so that scoring at 15:45 and filling at 15:55 improved the
entry by about that much. Re-measured on 2,933 ticker-days of real 1-minute
bars (data/intraday_store, 2026-04..09, this strategy's own names) that is not
true:

  15:55 vs the 15:59 close   mean -0.039%  sd 0.315%   <- the fill assumption
  15:45 vs the 15:59 close   mean -0.052%  sd 0.655%
  drift 15:45 -> 15:55       mean -0.013%  sd 0.535%

The first line reproduces the fill measurement above almost exactly (-0.04%,
sd 0.39%), which is what makes the second and third credible. There is NO
systematic +0.43% pickup between the decision and the fill -- the drift is a
coin flip with a half-percent standard deviation. That drift is the real,
unmodelled friction in every backtest of this strategy: they all treat the
decision price and the fill price as the same close. It is symmetric, so it
does not bias the mean, but a single position can be filled 0.5-1% away from
the price its exit rule was evaluated against.

RULES ENCODED
-------------
ENTRY  top-5 by raw_margin, extension indicator FAVOURABLE (0 of 3 flags),
       free slot, not already held, and past both re-entry gates: the
       wash-sale window (WASH_DAYS, losing exits) and the re-entry cooldown
       (REENTRY_COOLDOWN, every exit). Until 2026-09-09 this was "first
       appearance ever".
EXIT   NO take profit (2026-09-10; see TAKE_PROFIT). Stop loss -15%, out of
       top-15 for 2 consecutive publication days (minimum 2-day hold), and the
       26-trading-day cap (21 until 2026-09-21; see MAX_HOLD_DAYS). A
       target, if MG_TAKE_PROFIT re-enables one, sells
       half and keeps the slot. TP and SL are checked once daily at 15:45
       against the same-session price and are exempt from the minimum hold.
SIZE   8 slots, equal weight, fractional (dollar) orders -- regular hours only.

There is no take profit as of 2026-09-10. 136 adjudicated the old
+12%-vs-+30% argument (the level was the wrong question -- size was) and 137
then replayed the rules over 31 months instead of 4 and found no cap beats
every cap, on return, volatility and drawdown alike. Read the note above
TAKE_PROFIT before putting one back.

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
# ADJUDICATED 2026-09-09 by code/136_tp_adjudication.py -- and the verdict is
# that BOTH sides were arguing about the wrong thing. The +12% case (135) was
# measured on all 280 top-5 rows, 43% of which are 9th-or-later re-appearances
# of a name (mean 21d -10.60%) that this script can never buy, because entry is
# first-appearance-only. On the 34 first appearances -- the tradeable set --
# hold-21d is +3.58% and every FULL take-profit loses to it, with full+12%
# the worst rule tested by worst-month regret (-9.02pp). The fat-tail claim
# reproduces there (top-3 +71.3%, rest -3.0%). Two claims died in the process:
# "+12% is a same-close-entry artifact" is false (it survives both entry and
# both fill models), and "median MFE = +12.27%, so target +12%" is not a valid
# inference (spearman(peak day, 21d return) = +0.652, n=280 -- winners peak on
# day ~15, losers on day ~6, so a fixed target sells the winners early).
# What the level actually rides on is regime: TP12 - hold is -9.0pp in May,
# +8.5pp in June, 0.0pp in July. Three months, three answers.
# SUPERSEDED 2026-09-10 by code/137_pit_replay.py, which replayed the whole
# rule set over 2024-02..2026-08 -- 193 positions and ~31 independent 21-day
# windows against the 4 the live pick history gives. On that sample the
# take-profit level is monotone in the OTHER direction, and the best setting
# is not to have one (account level, $1000 book, fixed $125 slots):
#
#   rule        per position   account/yr    vol    maxDD
#   half +12%       +4.06%       +31.1%     20.4%   -17.5%
#   half +20%       +4.38%       +33.1%     20.3%   -17.8%
#   half +30%       +4.58%       +34.4%     20.1%   -18.1%
#   NO take profit  +4.63%       +34.6%     19.5%   -15.5%
#
# No cap wins on return AND on volatility AND on drawdown, because a target
# sells the names that trend to the end of the window and leaves the book
# holding the ones that do not. The 2026-06 fat-tail analysis was right and
# the +12% here was an artifact of a 4-month sample -- see 136 for why that
# sample was measuring a pick population the strategy cannot even buy.
#
# So TAKE_PROFIT is now 0 = DISABLED. Exits are the stop, the hold cap
# (26 days since 2026-09-21) and the dropout rule. Set MG_TAKE_PROFIT=0.30 to put a target back; it will sell
# TAKE_PROFIT_PORTION, which is still the one thing both samples agreed on.
TAKE_PROFIT = float(os.environ.get("MG_TAKE_PROFIT") or 0)   # 0 = no target

# How much of the position the target sells. 0.5 = sell half, ride the rest to
# the stop or the hold cap; 1.0 restores the old sell-everything behaviour.
#
# This is the answer to the contradiction above, and it is not a compromise for
# its own sake. Measured on the 34 first appearances (136_tp_adjudication.py),
# mean 21d P&L per position, by month:
#            hold    full+12%  full+30%  half+8%  half+12%  half+30%
#   May    +16.72%     +7.70%   +12.80%  +13.77%   +12.21%   +14.76%
#   Jun     -8.15%     +0.39%    -8.15%   -4.85%    -3.88%    -8.15%
#   Jul     -8.01%     -8.01%    -8.01%   -7.01%    -8.01%    -8.01%
#   worst month vs hold  -9.02pp  -3.92pp  -2.95pp   -4.51pp   -1.96pp
# Half-out beats full-out at the SAME level in every month and in both
# populations, because the two analyses were each half right: most picks do
# pop and fade (so take something), and the mean is made by 3 names that
# trend to the end of the window (so do not take it all). full+12% -- what
# ran here until today -- was the worst rule tested by worst-month regret.
#
# Do not read the pooled numbers as an edge. ~4 independent 21-day windows,
# every confidence interval on the tradeable set crosses zero.
TAKE_PROFIT_PORTION = float(os.environ.get("MG_TAKE_PROFIT_PORTION") or 0.5)

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
# 26, not 21, as of 2026-09-21. 21 was never an exit optimization -- it was
# inherited from the MODEL's target window (21-day +30% touch). A cap sweep on
# 137's 31-month point-in-time replay (live rules, only the cap varied, open
# book marked to the final close, total income on fixed $1000):
#   cap 5 +$555 | 12 +$882 | 21 +$1,071 | 26 +$1,312 | 31 +$1,163 | none +$1,789
# The relationship is "longer is better" because the cap's only real effect is
# amputating the right tail (dropout already does the routine recycling: with
# no cap at all, average hold is still ~21 days). No-cap is the sweep's max
# but is carried by 3 multi-month rides, loses money in the choppy 2024, and
# its edge over 21 is only P=84% by monthly bootstrap. 26 is the step the user
# chose: +$241 over 21 on the replay (P=87%), one extra week of tail, without
# going tail-dependent. The model's TARGET window stays 21 days -- this is a
# holding rule, not a label change.
MAX_HOLD_DAYS = 26
DROPOUT_DAYS = 2
MIN_HOLD_DAYS = 2

# Re-entry. Until 2026-09-09 a name held once was blocked FOREVER
# (`ever_entered`), which 124's docstring called the "first sighting only"
# rule. Nothing in this repo ever tied that to tax, and it is not what the tax
# rule says: a wash sale (IRC 1091) needs a sale at a LOSS plus a repurchase
# within 30 days either side of it. The price you buy back at is irrelevant,
# and a sale at a GAIN is never a wash sale. So the block is now exactly that
# window and nothing wider.
#
# Blocked when the last full exit of a name was at a loss (or at an unknown
# price -- see EXIT LOG below) and was 30 or fewer calendar days ago. Eligible
# again on day 31.
#
# Two things this deliberately does NOT model, because the strategy's own
# shape rules them out: the 30-days-BEFORE leg (it needs a second lot open
# while the first is sold at a loss, and a held name can never be re-bought),
# and cross-account identity (the rule spans all your accounts and a spouse's;
# this book is one account and the tracker only ever sees it).
#
# Measured cost of the change: relaxing the entry rule this way took the
# 4-month pick sample from 34 positions to 41 and the mean 21d P&L per
# position from +3.58% to +2.65% (136_tp_adjudication.py). It buys more,
# slightly worse names. Set WASH_DAYS enormous to restore the old behaviour.
WASH_DAYS = int(os.environ.get("MG_WASH_DAYS") or 30)

# Re-entry cooldown, added 2026-09-09 after measuring what the wash-sale
# relaxation actually let in. It is a SEPARATE constraint from WASH_DAYS and
# applies to every exit, winning or losing.
#
# Sequential simulation of this rule set over the 2026-05..09 pick history
# (scratch: reentry2.py), P&L per position:
#                      first entries        re-entries
#   FAVOURABLE gate    -0.48% (n=23)     -3.11% (n=5)
#   all top-5          +1.00% (n=34)     -8.31% (n=13)   diff 95% CI
#                                                        [-17.97%,-0.89%]
# 12 of those 13 re-entries followed a WINNING exit, and 10 were inside 30
# days of it, so the wash-sale gate -- which only blocks LOSING exits -- never
# touched the group that loses the money. The pattern is its own thing: a name
# you just exited at a gain is still top-5 because it has been running, and
# buying your own recent exit back is buying it extended.
#
# 60 days removes every re-entry in this sample, which is why the sample
# cannot tell 60 apart from "never". 60 is the smaller claim, and it keeps the
# name eligible eventually. Set MG_REENTRY_COOLDOWN=0 to allow re-entry as
# soon as the tax rule permits.
#
# WASH_DAYS is a tax constraint and this is an empirical one. Tune this one.
REENTRY_COOLDOWN = int(os.environ.get("MG_REENTRY_COOLDOWN") or 60)

# EXIT LOG. state["exit_log"][ticker] = {"date": ISO, "gain": float|None}.
# Written here at plan time for every FULL exit (the 15:45 gain is within a
# few basis points of the 15:55 fill), and by 124_mg_reconcile.py with
# gain=None when a position disappears from the broker without this script
# having planned the exit -- a manual sale, or an exit whose price we never
# saw. gain=None is treated as a LOSS: blocking a name we might have been
# allowed to re-buy costs one skipped entry, while re-buying into a real wash
# sale costs a disallowed loss, so the unknown side errs toward waiting.
def entry_block(ticker, exit_log, today):
    """Return None if the name is buyable, else a short reason string.

    Two independent gates, whichever binds longer: the wash-sale window (tax,
    losing exits only) and the re-entry cooldown (empirical, every exit)."""
    e = (exit_log or {}).get(ticker)
    if not e or not e.get("date"):
        return None
    try:
        sold = datetime.fromisoformat(str(e["date"])).date()
    except ValueError:
        return None
    days = (today - sold).days
    gain = e.get("gain")
    # gain is None -> price unknown -> treated as a loss, see EXIT LOG above.
    if (gain is None or gain <= 0) and days <= WASH_DAYS:
        kind = "loss" if gain is not None else "unknown price"
        return (f"wash-sale window ({kind} sale {days}d ago, "
                f"free in {WASH_DAYS - days + 1}d)")
    if REENTRY_COOLDOWN and days <= REENTRY_COOLDOWN:
        return (f"re-entry cooldown (exited {days}d ago, "
                f"free in {REENTRY_COOLDOWN - days + 1}d)")
    return None

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


def rescore(timeout=1800):
    """Refresh the score on the current (near-complete) session."""
    try:
        r = subprocess.run([PY, str(ROOT / "code" / "101_refresh_score_today.py")],
                           cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = -1
        print(f"[123] WARNING: rescore timed out after {timeout}s")
    if rc != 0:
        print(f"[123] WARNING: rescore exited {rc}; score file left as-is")
    return rc == 0


def _score_ok(min_rows, today):
    """Is the score file complete and from today?"""
    try:
        g = pd.read_csv(OUT / "today_score_fresh_sp500.csv")
    except Exception:
        return False, "unreadable"
    if len(g) < min_rows:
        return False, f"{len(g)} rows (< {min_rows}) — partial download"
    asof = str(g["date"].iloc[0]) if "date" in g.columns and len(g) else "?"
    if asof != today:
        return False, f"asof {asof}, not today — stale"
    return True, f"{len(g)} rows, asof {asof}"


def build():
    d = pd.read_csv(OUT / "today_score_fresh_sp500.csv")
    asof = str(d["date"].iloc[0]) if "date" in d.columns else "?"
    ranked = d.nlargest(15, "raw_margin").reset_index(drop=True)
    top15 = set(ranked.ticker)
    st = load_state()
    openp = {p["ticker"]: p for p in st["positions"]}
    exit_log = st.setdefault("exit_log", {})

    # Same-session prices for the held names. This is the take-profit input:
    # without it TAKE_PROFIT was a printed footer with no code behind it, and
    # since the model re-picks its own holdings (they stay in the top-15, so
    # the dropout rule never fires either) the only reachable exit was the
    # 21-day cap. Positions could only ever accumulate.
    prices = dict(zip(d.ticker, d.close))
    today_d = datetime.now().date()
    today = today_d.isoformat()

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
        # A name absent from the score file is UNKNOWN, not out-of-rank: on the
        # 60-row day (2026-09-23) two held names were missing because the
        # download died, and counting that as "out of the top-15" put them one
        # day from a dropout sale caused by an outage rather than a ranking.
        if t not in prices:
            miss = int(p.get("miss_base", 0))
        else:
            miss = 0 if t in top15 else int(p.get("miss_base", 0)) + 1

        px, entry = prices.get(t), p.get("entry_price")
        gain = (float(px) / float(entry) - 1.0) if px is not None and entry else None
        if gain is None:
            # A held name absent from the score file cannot be price-checked,
            # so its take-profit silently stops working. Say so out loud.
            warnings.append(f"{t}: no price in today's score file — gain exits AND dropout not evaluated")

        # FULL exits are evaluated first. The take profit is now partial, so
        # it no longer "outranks everything": on a day that is both +12% and
        # day 21, the cap has to win or the last half would never be sold.
        why, portion = None, 1.0
        if STOP_LOSS and gain is not None and gain <= -STOP_LOSS:
            # Stop loss is exempt from MIN_HOLD_DAYS -- a day-1 crash is
            # exactly the case it exists for. The dropout rule cannot do this
            # job (see the STOP_LOSS note above): falling RAISES a name's rank.
            why = f"stop loss {gain*100:+.1f}%"
        elif held >= MAX_HOLD_DAYS:
            why = f"{MAX_HOLD_DAYS}-day cap"
        elif held > MIN_HOLD_DAYS and miss >= DROPOUT_DAYS:
            why = f"out of top-15 for {miss}d"
        elif TAKE_PROFIT and gain is not None and gain >= TAKE_PROFIT and not p.get("half_done"):
            # Partial take profit, once per position, exempt from
            # MIN_HOLD_DAYS. The position keeps its slot afterwards.
            why, portion = f"take profit {gain*100:+.1f}%", TAKE_PROFIT_PORTION
        elif (TAKE_PROFIT and gain is not None and gain >= TAKE_PROFIT and p.get("half_done")
              and p.get("qty_before_tp") and float(p.get("quantity") or 0)
              >= 0.9 * float(p["qty_before_tp"])):
            # Self-heal: half_done is set when the plan is WRITTEN, but the
            # 15:55 executor can skip a sell (not held, review error, session
            # died). If the reconciled quantity never actually fell and the
            # name is still above the target, re-issue the trim rather than
            # leave the position marked as trimmed forever.
            why, portion = f"take profit {gain*100:+.1f}% (retry)", TAKE_PROFIT_PORTION
            warnings.append(f"{t}: earlier trim never reduced the position — re-issuing")

        if why:
            qty = float(p.get("quantity") or 0) or None
            if portion < 1.0:
                # Mark at plan time, like days_held above: the state file is
                # written before the executor runs, and the retry branch is
                # what covers a sell that never landed.
                p["half_done"] = True
                p["qty_before_tp"] = qty
            else:
                # Full exit: this is the sale the wash-sale window is measured
                # from. Recorded at plan time rather than after the fill,
                # because nothing downstream ever learns the exit price --
                # 124 only sees that the position vanished.
                exit_log[t] = {"date": today,
                               "gain": round(gain, 4) if gain is not None else None,
                               "source": "plan", "reason": why}
            exits.append({"ticker": t, "reason": why, "days_held": held,
                          "portion": "half" if portion < 1.0 else "all",
                          "fraction": round(portion, 4),
                          "quantity": round(qty * portion, 6) if qty else None,
                          "gain": round(gain, 4) if gain is not None else None})

        p["days_out_of_top15"] = miss
        p["last_counted"] = today

    # A trimmed position still occupies its slot -- only a full exit frees one.
    full_exits = [e for e in exits if e["fraction"] >= 1.0]
    free = SLOTS - (len(openp) - len(full_exits))
    entries, watch = [], []
    for r in ranked.head(5).to_dict("records"):
        t = r["ticker"]
        fl = flags(r); b = band(len(fl))
        # A name being sold TODAY cannot be re-bought today: both windows
        # open on the sale, and exit_log is written above before this loop
        # runs, so entry_block already sees it.
        blocked = entry_block(t, exit_log, today_d)
        watch.append({"ticker": t, "band": b, "flags": fl,
                      "held": t in openp, "blocked": blocked})
        if b == "FAVOURABLE" and t not in openp and not blocked:
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
                 + ("  [HELD]" if w["held"] else "")
                 + (f"  [BLOCKED: {w['blocked']}]" if w.get("blocked") else ""))
    L += ["", f"SELL ({len(p['exits'])}):"]
    L += [f"  - {e['ticker']:<6} {'HALF' if e['fraction'] < 1 else 'ALL '} "
          f"{e['reason']} (held {e['days_held']}d)" for e in p["exits"]] or ["  (none)"]
    if p.get("warnings"):
        L += [""] + [f"  !! {w}" for w in p["warnings"]]
    L += ["", f"BUY ({len(p['entries'])}):"]
    if p["entries"]:
        for e in p["entries"]:
            L.append(f"  + {e['ticker']:<6} ~${SLOT_DOLLARS} fractional  (last ${e['close']:.2f}, "
                     f"margin {e['raw_margin']:+.2f})")
    else:
        L.append("  (none — no FAVOURABLE buyable names, or no free slot)")
    tp_txt = (f"+{TAKE_PROFIT*100:.0f}% take profit (sells {TAKE_PROFIT_PORTION:.0%}, "
              f"keeps the slot) | " if TAKE_PROFIT else "no take profit | ")
    L += ["", f"Re-entry: {WASH_DAYS}d wash-sale block after a losing exit, and a "
              f"{REENTRY_COOLDOWN}d cooldown after ANY exit.",
          f"Standing exits: {tp_txt}"
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

    # Fetch with retries. The 2026-09-23 failure was a mid-run yfinance rate
    # limit; the right first response is to back off and try again, not to
    # write off the day. Budget: the plan must exist before the 15:55
    # executor, so retries stop ~7 minutes after start no matter what, and
    # each attempt gets a hard per-attempt timeout (the old 1800s timeout
    # would have blown straight through the execution window).
    _min_rows = int(os.environ.get("MG_MIN_UNIVERSE") or 400)
    _today = datetime.now().date().isoformat()
    if not a.skip_rescore:
        import time as _time
        _t0 = _time.time()
        for _attempt in range(1, 4):
            _remaining = 480 - (_time.time() - _t0)   # chunked fetch is ~3min/attempt
            if _remaining < 60:
                print("[123] retry budget exhausted")
                break
            rescore(timeout=min(260, int(_remaining)))
            _ok, _why = _score_ok(_min_rows, _today)
            if _ok:
                if _attempt > 1:
                    print(f"[123] rescore recovered on attempt {_attempt} ({_why})")
                break
            print(f"[123] attempt {_attempt}: score file not usable ({_why})"
                  + (" — backing off 45s" if _attempt < 3 else ""))
            if _attempt < 3:
                _time.sleep(45)

    # DATA GATE -- added 2026-09-23 after the 60-row day. 101 crashed mid-run
    # (yfinance rate limit) and left a PARTIAL score file: 60 of ~501 names,
    # dated today. The "use the existing file" fallback then fed it straight
    # into build(): the top-15 was ranked inside a 60-name field, and two held
    # names (CRL, WDAY) were counted "out of the top-15" -- one more such day
    # and dropout would have SOLD them over a data outage. A partial file
    # passes the executor's asof guard (its date IS today), so the check has
    # to happen here, before any counter mutates or any plan is written.
    # Bailing out leaves latest.json on yesterday's date, which the 15:55
    # executor already refuses to trade -- same standdown path as a failed run.
    _ok, _why = _score_ok(_min_rows, _today)
    _bad = None if _ok else f"score file: {_why} (after retries)"
    if _bad and not a.force:
        msg = f"[123] DATA GATE: {_bad}. No plan written, no counters touched, nothing will trade."
        print(msg)
        if a.notify:
            subprocess.run([PY, str(ROOT / "code" / "notify.py"),
                            f"MG 15:45 — {_bad}. Standing down today; positions unchanged."],
                           cwd=ROOT)
        sys.exit(0)
    if _bad:
        print(f"[123] WARNING: {_bad} — proceeding anyway (--force)")

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
        # `portion`/`fraction` tell the 15:55 executor how much to sell:
        # "all" = the full live quantity, "half" = live quantity x fraction.
        # `quantity` is this script's own estimate from the last reconciled
        # state and is a cross-check only -- the executor sizes off the LIVE
        # quantity, because that is the one that cannot be stale.
        "sells": [{"ticker": e["ticker"], "reason": e["reason"],
                   "portion": e["portion"], "fraction": e["fraction"],
                   "quantity": e.get("quantity"),
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
    # Shadow ledger: the take-profit counterfactual -- rule A, sell EVERYTHING
    # at +12% -- tracked in parallel on the same entry stream. Until today the
    # shadow held the half-out rule and the live book sold everything; 136's
    # adjudication swapped them, so 126_mg_shadow_b.py now shadows the rule
    # this script just stopped running. It places nothing and keeps its own
    # state; the try/except means a shadow bug can never break the live plan.
    # Its one-liner is appended to the plan text BEFORE --notify so the daily
    # Telegram carries the A-vs-B' comparison.
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
