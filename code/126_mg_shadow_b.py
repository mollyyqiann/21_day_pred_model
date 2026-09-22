"""Shadow ledger — the take-profit counterfactual, tracked against the live book.

PLACES NOTHING. This is pure bookkeeping, run from inside 123_mg_close_scorer
right after the real plan is built, on the same 15:45 prices and the same daily
top-5 stream. It answers one question the two conflicting take-profit analyses
could not: on the SAME entries, which exit size actually makes more money?

FLIPPED TWICE, 2026-09-09 and again 2026-09-10. It first shadowed half-out
while the live book sold everything at +12%; 136 adjudicated that and half
became live, so it shadowed the full-out rule instead. Then 137 replayed the
rules over 31 months rather than 4 and found that NO take profit beats every
level of one -- on return, volatility and drawdown together -- so the live book
now has no target at all and this ledger shadows the rule that was just
switched off: half at +12%.

The rule to beat is always whatever 123 stopped doing. The live book has no
take profit; this file keeps half-at-+12%:
  - entries: identical rule to 123 (top-5, FAVOURABLE, free slot, not held,
    not inside the 30-day wash-sale window, 8 slots x $125) but tracked
    against its OWN slot and exit history, because slot dynamics are part of
    the strategy: A frees a slot when it sells out of a name, the live rule
    often does not -- and A's exits are more often at a gain, which under the
    wash-sale gate also makes it re-buyable sooner.
  - exits: first close >= +12%  -> sell HALF, once (the rule 123 just retired)
            close <= -15%        -> sell the remainder (same stop as live)
            26 publication days  -> close what is left (was 21, matched to 123)
  - marks: data/mg_shadow_b.json (state) and
           output/monthly_gainer/shadow_b_ledger.csv (one row per day).

Seeded 2026-09-09 from the real book (FTNT / LITE / HUM at their actual fill
prices) so both rules start from the identical portfolio and diverge from the
2026-09-10 run onward. No position had reached the target when the direction
was flipped, so the seed is still valid for either side. Compare with the real
account's P&L, not with a sim.

Idempotent per date: a second run on the same day is a no-op, matching the
day-counting fix in 123.
"""
import json
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "mg_shadow_b.json"
LEDGER = ROOT / "output" / "monthly_gainer" / "shadow_b_ledger.csv"

SLOT = 125.0
TP_AT = 0.12            # first close at/above this -> sell HALF, once
TP_PORTION = 0.5
STOP = 0.15             # close at/below -this -> sell whatever remains
CAP_DAYS = 26          # follows 123's MAX_HOLD_DAYS (changed 21->26 2026-09-21)
SLOTS = 8
# Both must match 123's gates or the comparison is not fair.
WASH_DAYS = 30
REENTRY_COOLDOWN = 60


def _blocked(t, log, today):
    e = (log or {}).get(t)
    if not e or not e.get("date"):
        return False
    try:
        sold = datetime.fromisoformat(str(e["date"])).date()
    except ValueError:
        return False
    days = (today - sold).days
    gain = e.get("gain")
    if (gain is None or gain <= 0) and days <= WASH_DAYS:
        return True
    return bool(REENTRY_COOLDOWN) and days <= REENTRY_COOLDOWN


def update(watch, prices, today):
    """watch: plan['watch'] rows ({ticker, band, ...}); prices: ticker->close;
    today: 'YYYY-MM-DD'. Returns a one-line summary, or None if already run."""
    st = (json.loads(STATE.read_text()) if STATE.exists()
          else {"positions": [], "ever": [], "realized": 0.0, "last_date": None})
    if st.get("last_date") == today:
        return None
    log = st.setdefault("exit_log", {})
    today_d = date.fromisoformat(today)

    events, keep = [], []
    for p in st["positions"]:
        px = prices.get(p["ticker"])
        p["days"] = int(p.get("days", 0)) + 1
        gain = None if px is None else px / p["entry"] - 1.0
        if gain is None:
            events.append(f"{p['ticker']}: no price, checks skipped")
            keep.append(p)
            continue
        if not p.get("half_done") and gain >= TP_AT:
            half = p["qty"] * TP_PORTION
            st["realized"] += half * (px - p["entry"])
            p["qty"] -= half
            p["half_done"] = True
            events.append(f"half-sell {p['ticker']} {gain:+.1%}")
        if gain <= -STOP or p["days"] >= CAP_DAYS:
            st["realized"] += p["qty"] * (px - p["entry"])
            why = "stop" if gain <= -STOP else "cap"
            log[p["ticker"]] = {"date": today, "gain": round(gain, 4),
                                "source": "shadow", "reason": why}
            events.append(f"{why} {p['ticker']} {gain:+.1%}")
            continue
        keep.append(p)
    st["positions"] = keep

    held = {p["ticker"] for p in st["positions"]}
    for w in watch:
        t = w["ticker"]
        if (w["band"] == "FAVOURABLE" and t not in held
                and not _blocked(t, log, today_d)
                and len(st["positions"]) < SLOTS):
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

    return (f'Shadow (½-sell@+12%, the retired rule): total {total:+.2f} '
            f'(realized {st["realized"]:+.2f}) | {len(st["positions"])} pos'
            + (f' | {"; ".join(events)}' if events else ''))
