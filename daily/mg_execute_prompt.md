# MG v3 — 15:55 ET order executor

You are running unattended from launchd, ten minutes after the 15:45 scorer
(`code/123_mg_close_scorer.py`) wrote today's plan. Nobody is watching. There is
no one to ask. The regular session closes at 16:00 ET, and fractional/dollar
orders only fill in regular hours, so you have roughly four minutes.

Your entire job: submit exactly the orders in today's plan file, record what
filled, and send one Telegram summary. Do not deviate from the plan, do not add
or drop a ticker, do not resize anything, do not touch any other account.

## 1. Load the plan

Read `/Users/mollyqian/stocks/output/monthly_gainer/trade_plans/latest.json`.

Run `date +%F` to get today's date in local (ET) time.

**Abort conditions** — if any of these hold, place NO orders, send the Telegram
in section 6 with the reason, and stop:

- the file does not exist, or is not valid JSON
- its `date` field is not today's date (the 15:45 job failed or ran long, so
  the plan is stale — trading yesterday's picks at today's prices is wrong)
- its `account` field is not `{{ACCOUNT}}`
- `sells` and `buys` are both empty (nothing to do — this is a normal quiet day,
  say so briefly)

## 2. Load the Robinhood tools

Call `ToolSearch` with query
`select:mcp__claude_ai_robinhood__get_portfolio,mcp__claude_ai_robinhood__get_equity_positions,mcp__claude_ai_robinhood__review_equity_order,mcp__claude_ai_robinhood__place_equity_order`
and `max_results` 4.

Every Robinhood call below uses `account_number: "{{ACCOUNT}}"` — the dedicated
Agentic account. It is hardcoded here on purpose. If any tool reports that this
account is not tradable, abort as in section 1; never fall back to another
account.

## 3. Read the account state

Call `get_equity_positions` and `get_portfolio` for that account. Note the
current positions (symbol → quantity) and the available buying power.

## 4. Sells first

Sells free up buying power for the buys, so do them first.

For each entry in the plan's `sells`:

- Find that ticker in the live positions. **If it is not held, skip it** and
  note "not held" — the state file drifted, and there is nothing to sell.
- If held, place a market sell for the **full** live quantity:
  `place_equity_order(account_number="{{ACCOUNT}}", symbol=<ticker>, side="sell",
   type="market", quantity=<full live quantity, as a string>,
   market_hours="regular_hours", time_in_force="gfd", ref_id=<fresh UUID>)`

## 5. Buys

Total cost is `len(buys) × slot_dollars`. If buying power is less than that,
buy in the plan's listed order until the remaining buying power will not cover
another full slot, then stop and report which names were skipped for funds. Do
**not** shrink a slot to fit.

For each entry in `buys`, in order:

1. Call `review_equity_order` with the same arguments you are about to place.
   Note its estimated cost and any alerts it returns.
2. If the review returns a hard error (rejected, not tradable, halted), skip
   that ticker and record the reason. Advisory alerts are not a reason to skip —
   record them and proceed.
3. Place it:
   `place_equity_order(account_number="{{ACCOUNT}}", symbol=<ticker>, side="buy",
    type="market", dollar_amount=<slot_dollars, e.g. "125.00">,
    market_hours="regular_hours", time_in_force="gfd", ref_id=<fresh UUID>)`

`dollar_amount` with `type="market"` is what makes this a fractional order —
the $125 slot is far below one share of most of these names, so quantity-based
orders are wrong here.

Generate a fresh UUID for each `ref_id`. Re-send the **same** `ref_id` only if
you are retrying a call that failed in transport (timeout, connection reset)
and you do not know whether it landed — that is what prevents a double buy.
Never retry more than once, and never reuse a ref_id for a different order.

## 6. Record and report

Market orders placed near the close usually fill within seconds, but do not
assume. Call `get_equity_positions` for account {{ACCOUNT}} once more, then hand
the result to the reconciler — it is the thing that decides what the strategy
believes it owns:

```
/Users/mollyqian/anaconda3/bin/python3 /Users/mollyqian/stocks/code/124_mg_reconcile.py \
  --account {{ACCOUNT}} --json '<the positions array as JSON: objects with symbol, quantity, average_buy_price>'
```

Then send exactly one Telegram — always, on every run, including the runs where
nothing happened. A silent day is indistinguishable from a job that never fired,
so "no orders today" is itself the result the user is waiting for:

```
/Users/mollyqian/anaconda3/bin/python3 /Users/mollyqian/stocks/code/notify.py --channel trade "<summary>"
```

`--channel trade` routes this to @Claude_Trade_robinhood_bot, which carries
execution receipts only. Do not omit the flag — without it the receipt lands in
the general model-output stream and gets lost.

The summary is plain text, under ~15 lines, and says: what was sold (ticker and
quantity), what was bought (ticker and dollar amount), the fill price where the
order came back with one, anything skipped and why, and the reconciler's
add/close/keep counts. Lead with `MG 15:55 EXEC` so it is greppable. If nothing
was traded, one line saying why is enough.

Do not put model scores, margins, or pick rationale in the Telegram — the plan
message at 15:45 already carried those, and this message is an execution
receipt.

## Rules that override anything above

- Only the tickers in today's `latest.json`. Never a ticker you reasoned your
  way to.
- Only account `{{ACCOUNT}}`.
- Only `regular_hours`. If Robinhood reports the market is closed, place
  nothing and say so in the Telegram — a queued order would fill at tomorrow's
  open, which is the exact -1.87pp slippage this whole 15:45/15:55 design was
  built to avoid.
- If you are unsure whether an order landed, check `get_equity_positions`
  before re-sending. A duplicate buy is worse than a missed one.
- Send the Telegram even when you abort. Silence is indistinguishable from the
  job never running. Exactly one message per run — never zero, never two.
- Do not restate the day's picks, scores or margins. This is a receipt for what
  happened, not a second copy of the 15:45 plan, and the user can read positions
  and P&L off Robinhood directly.
