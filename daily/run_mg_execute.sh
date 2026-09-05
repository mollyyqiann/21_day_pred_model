#!/bin/bash
# MG v3 — 15:55 ET order executor.
#
# This is the second half of the pair the 15:45 job (com.user.mgcloseplan ->
# 123_mg_close_scorer.py) leaves unfinished: that job scores, writes
# trade_plans/latest.{txt,json} and Telegrams the plan, but places no orders.
# This wrapper hands that plan to a headless Claude Code session which submits
# the orders through the Robinhood MCP connector and reconciles the fills.
#
# Why Claude and not a plain script: Robinhood has no scriptable client here.
# The connector is an MCP server reachable from a Claude session, so the
# session IS the order-entry client. The prompt is fixed on disk and the tool
# allowlist below is narrow, so the session's freedom is close to nil.
#
# Why local and not the cloud routine: the plan lives on this Mac. A cloud
# routine runs in an isolated Linux container that cannot see it -- the old
# "Trade execution check" routine failed that way every day it fired.
#
# Usage:
#   run_mg_execute.sh              # live -- submits real orders
#   MG_EXEC_MODE=dry run_mg_execute.sh   # resolves + reviews, places nothing
set -uo pipefail

export PATH="/Users/mollyqian/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONUNBUFFERED=1

ROOT="/Users/mollyqian/stocks"
PY="/Users/mollyqian/anaconda3/bin/python3"
LOG="$ROOT/output/monthly_gainer/execute.log"
MODE="${MG_EXEC_MODE:-live}"
cd "$ROOT"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
alert() { "$PY" "$ROOT/code/notify.py" --channel trade "MG 15:55 EXEC — $1" >> "$LOG" 2>&1 || true; }

# GUARD 1 of 3 -- the NYSE calendar. Weekends, holidays, and 13:00 ET early
# closes all mean a 15:55 market order cannot fill; it would queue to the next
# open, which is the -1.87pp slippage this design exists to avoid.
# Deliberately silent: a Telegram every Saturday and every Christmas is noise,
# and the log records the skip.
CAL=$("$PY" "$ROOT/code/125_trading_day.py") || CAL_BAD=1
if [ "${CAL_BAD:-0}" = "1" ]; then
  # MG_EXEC_IGNORE_CALENDAR exists ONLY to rehearse the pipeline off-hours.
  # It does not make the market open: with MODE=live it would place orders that
  # queue to the next session. Use it with MG_EXEC_MODE=dry.
  if [ "${MG_EXEC_IGNORE_CALENDAR:-0}" = "1" ]; then
    log "NOT A TRADING DAY ($CAL) -- proceeding anyway, MG_EXEC_IGNORE_CALENDAR=1"
  else
    log "not a trading day -> skip ($CAL)"
    exit 0
  fi
else
  log "calendar ok ($CAL)"
fi

PLAN="$ROOT/output/monthly_gainer/trade_plans/latest.json"
TODAY=$(date +%F)

# GUARD 2 of 3 -- plan freshness. Guarded here as well as in the prompt: a
# stale plan is the one failure mode that silently trades the wrong thing, and
# catching it in bash means we never even start a session that could act on it.
if [ ! -f "$PLAN" ]; then
  log "no plan file at $PLAN -> abort"
  alert "no plan file — the 15:45 scorer did not write one. Nothing traded."
  exit 1
fi

PLAN_DATE=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('date',''))" "$PLAN" 2>/dev/null)
if [ "$PLAN_DATE" != "$TODAY" ]; then
  log "plan date '$PLAN_DATE' != today '$TODAY' -> abort"
  alert "plan is dated '$PLAN_DATE', not today ($TODAY). The 15:45 job failed or ran long. Nothing traded."
  exit 1
fi

# GUARD 3 of 3 -- market data freshness, and the only guard that catches an
# UNSCHEDULED closure (day of mourning, weather, exchange outage). No calendar
# knows those in advance, but on such a day yfinance has no bar for today, so
# the plan's `asof` stays on the previous session while `date` is still today.
# This also catches a 15:45 run whose yfinance pull failed and silently fell
# back to the previous score file.
PLAN_ASOF=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('asof',''))" "$PLAN" 2>/dev/null)
if [ "$PLAN_ASOF" != "$TODAY" ]; then
  log "plan asof '$PLAN_ASOF' != today '$TODAY' -> abort"
  alert "plan data is asof '$PLAN_ASOF', not today ($TODAY) — no fresh session bar. Either the market did not open or the 15:45 data pull failed. Nothing traded."
  exit 1
fi

log "=== start (mode=$MODE, plan $PLAN_DATE) ==="

# The account number is injected here rather than written into the prompt file,
# which is public. config/mg_execution.json is not.
ACCOUNT=$("$PY" -c "import sys; sys.path.insert(0,'$ROOT/code'); from mg_account import account; print(account())" 2>>"$LOG")
if [ -z "${ACCOUNT:-}" ]; then
  log "could not resolve the Robinhood account -> abort"
  alert "no Robinhood account configured (config/mg_execution.json). Nothing traded."
  exit 1
fi

PROMPT=$(sed "s/{{ACCOUNT}}/$ACCOUNT/g" "$ROOT/daily/mg_execute_prompt.md")
if [ "$MODE" = "dry" ]; then
  PROMPT="$PROMPT

## MODE: DRY-RUN (overrides section 5 step 3 and section 4)

Do everything above EXCEPT calling place_equity_order — never call it, not once.
Still read the plan, still read positions and buying power, still call
review_equity_order for each buy. Then send the Telegram summary prefixed
'MG 15:55 EXEC [DRY-RUN]' describing exactly the orders you WOULD have placed,
with the reviewed estimated costs. Skip the 124_mg_reconcile.py call entirely --
nothing changed, so there is nothing to reconcile."
else
  PROMPT="$PROMPT

## MODE: LIVE

Place the orders for real."
fi

# The allowlist is the real safety boundary. In -p mode anything not listed is
# denied outright (there is no one to prompt), so the session can reach exactly
# these Robinhood endpoints and exactly these two scripts -- no shell, no
# writes, no other tickers, no other account.
claude -p "$PROMPT" \
  --model claude-sonnet-5 \
  --allowedTools \
    "ToolSearch" \
    "Read" \
    "mcp__claude_ai_robinhood__get_portfolio" \
    "mcp__claude_ai_robinhood__get_equity_positions" \
    "mcp__claude_ai_robinhood__review_equity_order" \
    "mcp__claude_ai_robinhood__place_equity_order" \
    "Bash(date:*)" \
    "Bash($PY $ROOT/code/124_mg_reconcile.py:*)" \
    "Bash($PY $ROOT/code/notify.py:*)" \
  >> "$LOG" 2>&1
RC=$?

log "=== claude exited rc=$RC ==="

# A non-zero exit means the session died before it could send its own Telegram,
# so this is the only notice the user would get.
if [ "$RC" -ne 0 ]; then
  alert "executor session failed (rc=$RC). Check output/monthly_gainer/execute.log. Orders may be PARTIALLY placed — verify in Robinhood."
fi

exit "$RC"
