"""Monday open-check (~9:35 ET, after first 5 minutes of trading).

Purpose: if Sunday's verdict said WAIT (because of gap-up risk), this script
checks the actual opening prints and decides if it's now a good time to buy.

Logic:
1. Read Sunday verdict (output/monthly_gainer/sunday_verdict_latest.md).
2. Pull current intraday prices for the holdings via yfinance.
3. For each name:
   - Compute gap %: open vs Friday close
   - Compare current price to open
   - If gap was big AND price has pulled back at least 30% of the gap: BUY now
   - If gap was big AND price keeps running up: STILL WAIT
   - If gap was small/flat: confirm BUY at any time
4. Push updated verdict.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

import json
import warnings
from datetime import datetime
from pathlib import Path

import yfinance as yf
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "monthly_gainer"
DATA = ROOT / "data"

PORTFOLIO = ["INTC", "SMCI", "MRNA"]


def main():
    print(f"[103] Monday open-check at {datetime.now().isoformat()}")

    # Read Sunday verdict
    sunday_path = OUT / "sunday_verdict_latest.md"
    if sunday_path.exists():
        sunday_text = sunday_path.read_text()
    else:
        sunday_text = "(no Sunday verdict found)"

    # Pull intraday for each holding
    rows = []
    for tk in PORTFOLIO:
        try:
            t = yf.Ticker(tk)
            # 5d daily for Friday close
            d = t.history(period="5d", interval="1d", auto_adjust=True)
            if d.empty:
                rows.append({"ticker": tk, "error": "no data"}); continue
            fri_close = float(d["Close"].iloc[-1])
            # Intraday for today
            i = t.history(period="1d", interval="5m", auto_adjust=True)
            if i.empty:
                rows.append({"ticker": tk, "fri_close": fri_close, "error": "market not open"}); continue
            open_price = float(i["Open"].iloc[0])
            cur_price = float(i["Close"].iloc[-1])
            high = float(i["High"].max())
            low = float(i["Low"].min())
            gap_pct = open_price / fri_close - 1.0
            cur_pct = cur_price / fri_close - 1.0
            # How much of the gap has been retraced?
            if abs(gap_pct) > 0.005:
                retrace = (open_price - cur_price) / (open_price - fri_close)  # 0=no retrace, 1=full retrace
            else:
                retrace = 0.0
            rows.append({
                "ticker": tk, "fri_close": fri_close,
                "open": open_price, "cur": cur_price,
                "high": high, "low": low,
                "gap_pct": gap_pct, "cur_pct": cur_pct,
                "retrace_of_gap": retrace,
            })
        except Exception as e:
            rows.append({"ticker": tk, "error": str(e)})

    df = pd.DataFrame(rows)

    # Build verdict
    lines = [
        f"# 🔔 MONDAY OPEN CHECK — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Per-name decision",
    ]
    for _, r in df.iterrows():
        tk = r["ticker"]
        if "error" in r and pd.notna(r.get("error")):
            lines.append(f"- **{tk}**: ⚠️ {r['error']}")
            continue
        gap = r.get("gap_pct", 0)
        cur = r.get("cur_pct", 0)
        retrace = r.get("retrace_of_gap", 0)
        cur_price = r.get("cur", 0)
        open_p = r.get("open", 0)
        fri = r.get("fri_close", 0)

        if gap > 0.015:
            # Big gap up
            if retrace > 0.3:
                action = f"✅ BUY now — gapped up {gap:+.1%} but pulled back {retrace*100:.0f}% of the gap. Good entry."
            elif retrace > 0.0:
                action = f"⏳ WAIT — gap up {gap:+.1%}, currently retracing {retrace*100:.0f}%. Wait for 50%+ retrace or after 10:30am."
            else:
                action = f"🛑 STILL EXTENDED — gap up {gap:+.1%}, kept running. Hold off; check at 11am."
        elif gap < -0.015:
            # Big gap down
            if retrace > 0.5:
                action = f"⏳ WAIT — gap down {gap:+.1%}, already bouncing {retrace*100:.0f}%. Better entry was at open."
            else:
                action = f"✅ BUY — gap down {gap:+.1%} is opportunity (assuming no bad news on the name). Better cost basis."
        else:
            action = f"✅ BUY — gap was flat ({gap:+.1%}). Standard entry, no timing concern."

        lines.append(f"- **{tk}**: ${cur_price:.2f} (open ${open_p:.2f}, Fri close ${fri:.2f}, gap {gap:+.1%}, now {cur:+.1%}). {action}")

    lines += ["", "---", "(Reference: Sunday verdict in sunday_verdict_latest.md)"]

    OUT.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    md_path = OUT / f"monday_open_check_{today_str}.md"
    md_path.write_text("\n".join(lines))
    (OUT / "monday_open_check_latest.md").write_text("\n".join(lines))
    print(f"\n[103] saved {md_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
