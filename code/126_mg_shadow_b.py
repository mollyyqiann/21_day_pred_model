"""Shadow ledger — exit-rule B' (sell HALF at +12%, let the rest run).

PLACES NOTHING. This is pure bookkeeping, run from inside 123_mg_close_scorer
right after the real plan is built, on the same 15:45 prices and the same
daily top-5 stream. It answers one question the two conflicting take-profit
analyses could not: on the SAME entries, does "sell all at +12%" (the live
rule) or "sell half at +12%, ride the rest" (this ledger) make more money?

The live book is rule A; this file keeps rule B':
  - entries: identical rule to 123 (top-5, FAVOURABLE, first appearance,
    8 slots x $125) but tracked against its OWN slot/ever-entered state,
    because slot dynamics are part of the strategy: A frees a slot when it
    sells out of a name, B' often does not.
  - exits: first close >= +12%  -> sell HALF, once
            close <= -15%        -> sell the remainder (same stop as live)
            21 publication days  -> close what is left
  - marks: data/mg_shadow_b.json (state) and
           output/monthly_gainer/shadow_b_ledger.csv (one row per day).

Seeded 2026-09-09 from the real book (FTNT / LITE / HUM at their actual fill
prices) so both rules start from the identical portfolio and diverge from the
2026-09-10 run onward. Compare with the real account's P&L, not with a sim.

Idempotent per date: a second run on the same day is a no-op, matching the
day-counting fix in 123.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "mg_shadow_b.json"
LEDGER = ROOT / "output" / "monthly_gainer" / "shadow_b_ledger.csv"

SLOT = 125.0
HALF_AT = 0.12          # first close at/above this -> sell half, once
STOP = 0.15             # close at/below -this -> sell whatever remains
CAP_DAYS = 21
SLOTS = 8


def update(watch, prices, today):
    """watch: plan['watch'] rows ({ticker, band, ...}); prices: ticker->close;
    today: 'YYYY-MM-DD'. Returns a one-line summary, or None if already run."""
    st = (json.loads(STATE.read_text()) if STATE.exists()
          else {"positions": [], "ever": [], "realized": 0.0, "last_date": None})
    if st.get("last_date") == today:
        return None

    events, keep = [], []
    for p in st["positions"]:
        px = prices.get(p["ticker"])
        p["days"] = int(p.get("days", 0)) + 1
        gain = None if px is None else px / p["entry"] - 1.0
        if gain is None:
            events.append(f"{p['ticker']}: no price, checks skipped")
            keep.append(p)
            continue
        if not p.get("half_done") and gain >= HALF_AT:
            half = p["qty"] / 2.0
            st["realized"] += half * (px - p["entry"])
            p["qty"] -= half
            p["half_done"] = True
            events.append(f"half-sell {p['ticker']} {gain:+.1%}")
        if gain <= -STOP:
            st["realized"] += p["qty"] * (px - p["entry"])
            events.append(f"stop {p['ticker']} {gain:+.1%}")
            continue
        if p["days"] >= CAP_DAYS:
            st["realized"] += p["qty"] * (px - p["entry"])
            events.append(f"cap {p['ticker']} {gain:+.1%}")
            continue
        keep.append(p)
    st["positions"] = keep

    held = {p["ticker"] for p in st["positions"]}
    for w in watch:
        t = w["ticker"]
        if (w["band"] == "FAVOURABLE" and t not in st["ever"]
                and t not in held and len(st["positions"]) < SLOTS):
            px = prices.get(t)
            if px:
                st["positions"].append({"ticker": t, "entry": float(px),
                                        "qty": SLOT / float(px), "days": 0,
                                        "half_done": False, "entry_date": today})
                st["ever"] = sorted(set(st["ever"]) | {t})
                held.add(t)
                events.append(f"buy {t} @{px:.2f}")

    unreal = sum(p["qty"] * (prices.get(p["ticker"], p["entry"]) - p["entry"])
                 for p in st["positions"])
    total = st["realized"] + unreal
    st["last_date"] = today
    STATE.write_text(json.dumps(st, indent=2))

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a") as f:
        if new:
            f.write("date,realized,unrealized,total_pnl,positions,events\n")
        f.write(f'{today},{st["realized"]:.2f},{unreal:.2f},{total:.2f},'
                f'{len(st["positions"])},"{"; ".join(events)}"\n')

    return (f'Shadow B′ (½-sell@+12%): total {total:+.2f} '
            f'(realized {st["realized"]:+.2f}) | {len(st["positions"])} pos'
            + (f' | {"; ".join(events)}' if events else ''))
