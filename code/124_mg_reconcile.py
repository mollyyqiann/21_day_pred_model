"""Reconcile the paper-position state against REAL Robinhood positions.

WHY THIS IS NOT SCHEDULED
-------------------------
The Robinhood connection is an MCP tool available inside a Claude conversation,
not an API a launchd job can call. The only way to make this automatic would be
to store brokerage credentials in a script, which is not something to do. So
reconciliation is a conversational step: Claude reads the live positions, pipes
them in here, and the state file is corrected.

That replaces the manual `121 --confirm TICKER PRICE` / `--close TICKER` dance.
The state is whatever the broker says it is, so it cannot silently drift.

WHAT IT DOES
------------
  in Robinhood, not in state   -> ADD    (uses the real average_buy_price)
  in state, not in Robinhood   -> CLOSE  (you sold, or never filled)
  in both                      -> KEEP   (preserves days_held / dropout count)

`ever_entered` is only ever appended to -- it is what enforces the
"first sighting only" entry rule, so a name you have held and closed must never
become eligible again.

Usage:
    python 124_mg_reconcile.py --json '[{"symbol":"MU","quantity":"0.13",
                                         "average_buy_price":"933.44",
                                         "entry_date":"2026-09-01"}]'
    python 124_mg_reconcile.py --file positions.json
    python 124_mg_reconcile.py --account $MG_RH_ACCOUNT --json '[]' --dry-run

  --account is mandatory and is checked against AGENTIC_ACCOUNT; positions from
  any other Robinhood account are refused outright.
"""
import json, argparse, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "mg_paper_positions.json"

# This strategy runs ONLY in the dedicated Agentic account. The other Robinhood
# accounts hold the real personal portfolio; ingesting those positions here
# would silently adopt them as strategy positions and then "manage" them under
# the exit rules. --account is therefore required and must match.
# The number itself lives in config/mg_execution.json (untracked) -- see
# code/mg_account.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mg_account import account as _account
AGENTIC_ACCOUNT = _account()


def load():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"positions": [], "ever_entered": []}


def reconcile(live, dry=False):
    st = load()
    have = {p["ticker"]: p for p in st.get("positions", [])}
    live_map = {}
    for r in live:
        sym = (r.get("symbol") or r.get("ticker") or "").upper()
        if not sym:
            continue
        try:
            qty = float(r.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:            # zero-quantity rows are not open positions
            continue
        live_map[sym] = r

    added, closed, kept = [], [], []
    for sym, r in live_map.items():
        if sym in have:
            kept.append(sym)
            continue
        px = r.get("average_buy_price")
        added.append({
            "ticker": sym,
            "entry_price": float(px) if px not in (None, "") else None,
            "quantity": float(r.get("quantity", 0) or 0),
            "entry_date": r.get("entry_date") or datetime.now().date().isoformat(),
            "days_held": int(r.get("days_held", 0)),
            "days_out_of_top15": 0,
        })
    for sym in have:
        if sym not in live_map:
            closed.append(sym)

    new_positions = [have[s] for s in kept] + added
    # sync quantity/price for kept names in case of partial fills or adds
    for p in new_positions:
        r = live_map.get(p["ticker"])
        if r:
            try:
                p["quantity"] = float(r.get("quantity", p.get("quantity", 0)) or 0)
            except (TypeError, ValueError):
                pass
            px = r.get("average_buy_price")
            if px not in (None, ""):
                p["entry_price"] = float(px)

    st["positions"] = new_positions
    st["ever_entered"] = sorted(set(st.get("ever_entered", [])) | set(live_map))
    st["last_reconciled"] = datetime.now().isoformat(timespec="seconds")
    st["account"] = AGENTIC_ACCOUNT

    print(f"reconciled against {len(live_map)} live position(s)")
    for p in added:
        print(f"  + ADD   {p['ticker']:<6} qty {p['quantity']:.4f} @ "
              f"{p['entry_price'] if p['entry_price'] is not None else '?'}")
    for s in closed:
        print(f"  - CLOSE {s:<6} (no longer held at the broker)")
    for s in kept:
        p = have[s]
        print(f"    KEEP  {s:<6} day {p.get('days_held', 0)}, "
              f"{p.get('days_out_of_top15', 0)}d out of top-15")
    if not (added or closed or kept):
        print("  (no open positions on either side — nothing to do)")
    print(f"  ever_entered now holds {len(st['ever_entered'])} name(s) "
          f"(blocks re-entry): {', '.join(st['ever_entered']) or '-'}")
    if dry:
        print("\n  --dry-run: state NOT written")
        return st
    STATE.write_text(json.dumps(st, indent=2, default=str))
    print(f"\n  wrote {STATE}")
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--json", help="JSON array of live positions")
    g.add_argument("--file", help="path to a JSON file of live positions")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--account", required=True,
                    help=f"must be the Agentic account ({AGENTIC_ACCOUNT})")
    a = ap.parse_args()
    if a.account != AGENTIC_ACCOUNT:
        print(f"REFUSING: --account {a.account} is not the Agentic account "
              f"({AGENTIC_ACCOUNT}).\nThis strategy must never ingest positions "
              f"from your other accounts."); sys.exit(2)
    raw = Path(a.file).read_text() if a.file else a.json
    try:
        live = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"bad JSON: {e}"); sys.exit(1)
    if isinstance(live, dict):
        live = live.get("positions", [])
    reconcile(live, dry=a.dry_run)
