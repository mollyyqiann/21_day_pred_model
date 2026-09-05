"""Is today a full NYSE trading day? Guard for the 15:45/15:55 order pair.

WHY THIS EXISTS
---------------
`com.user.mgexecute` fires at 15:55 ET every weekday. Weekends are already
skipped, but a bare weekday check happily fires on Thanksgiving, Good Friday
and July 4th. Two things go wrong then:

  1. On a holiday the market is shut, so a market order placed at 15:55 does
     not fill -- it queues for the next open. That is precisely the -1.87pp
     next-open slippage the whole 15:45/15:55 design was built to avoid.
  2. On an early-close day (13:00 ET) the session is already over by 15:45, so
     the "near-complete session" the scorer thinks it is scoring is actually
     yesterday's, and the order again queues to the next open.

So: trade only on days the NYSE is open through 16:00.

NO EXTERNAL DEPENDENCY ON PURPOSE
---------------------------------
pandas_market_calendars / exchange_calendars are not installed, and a launchd
job that places real orders should not acquire a new pip dependency it can fail
to import at 15:55. The NYSE rule set is small and stable, so it is computed
here from first principles.

WHAT THIS CANNOT KNOW
---------------------
Unscheduled closures -- a national day of mourning, a hurricane, an exchange
outage -- are not on any calendar in advance. Those are caught by the second,
independent guard in daily/run_mg_execute.sh: it requires the plan's `asof`
(the date of the freshest yfinance bar) to be today. On an unscheduled closure
there is no bar for today, `asof` stays on the previous session, and the
executor aborts. Neither guard is sufficient alone; both are cheap.

Usage:
    python 125_trading_day.py            # prints a reason, exit 0 if tradable
    python 125_trading_day.py --json
    python 125_trading_day.py --date 2026-11-27
"""
import sys, json, argparse
from datetime import date, timedelta


def easter(year: int) -> date:
    """Anonymous Gregorian algorithm. Good Friday is an NYSE holiday even
    though it is not a federal one, so this cannot be skipped."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th `weekday` (Mon=0) of the month; n=-1 means the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date, *, shift_saturday: bool = True) -> date:
    """NYSE observation rule: Sunday holidays move to Monday, Saturday holidays
    move back to Friday -- except New Year's Day, which is simply not observed
    when it lands on a Saturday (hence shift_saturday=False for it)."""
    if d.weekday() == 6:
        return d + timedelta(days=1)
    if d.weekday() == 5:
        return d - timedelta(days=1) if shift_saturday else d
    return d


def holidays(year: int) -> dict:
    """Full-day NYSE closures -> name."""
    e = easter(year)
    h = {
        _observed(date(year, 1, 1), shift_saturday=False): "New Year's Day",
        _nth_weekday(year, 1, 0, 3): "Martin Luther King Jr. Day",
        _nth_weekday(year, 2, 0, 3): "Washington's Birthday",
        e - timedelta(days=2): "Good Friday",
        _nth_weekday(year, 5, 0, -1): "Memorial Day",
        _observed(date(year, 6, 19)): "Juneteenth",
        _observed(date(year, 7, 4)): "Independence Day",
        _nth_weekday(year, 9, 0, 1): "Labor Day",
        _nth_weekday(year, 11, 3, 4): "Thanksgiving",
        _observed(date(year, 12, 25)): "Christmas",
    }
    # Juneteenth only became an NYSE holiday in 2022.
    if year < 2022:
        h.pop(_observed(date(year, 6, 19)), None)
    return h


def early_closes(year: int) -> dict:
    """1:00 PM ET closes -> name. The 15:45 scorer and 15:55 executor both run
    after that, so these are as unusable as a full closure."""
    out = {}
    ind = _observed(date(year, 7, 4))
    # July 3 is a half day when the 4th is a normal weekday session.
    if ind == date(year, 7, 4) and ind.weekday() < 5:
        d = date(year, 7, 3)
        if d.weekday() < 5:
            out[d] = "July 3 (half day)"
    out[_nth_weekday(year, 11, 3, 4) + timedelta(days=1)] = "Day after Thanksgiving (half day)"
    xmas = date(year, 12, 24)
    if xmas.weekday() < 5:
        out[xmas] = "Christmas Eve (half day)"
    return out


def check(d: date) -> dict:
    """-> {tradable: bool, reason: str}"""
    if d.weekday() >= 5:
        return {"date": d.isoformat(), "tradable": False,
                "reason": "weekend (" + d.strftime("%A") + ")"}
    name = holidays(d.year).get(d)
    if name:
        return {"date": d.isoformat(), "tradable": False,
                "reason": f"NYSE holiday — {name}"}
    name = early_closes(d.year).get(d)
    if name:
        return {"date": d.isoformat(), "tradable": False,
                "reason": f"NYSE closes 13:00 ET — {name}; the 15:45/15:55 "
                          f"pair would run after the close"}
    return {"date": d.isoformat(), "tradable": True, "reason": "full session"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: today, local time)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    d = date.fromisoformat(a.date) if a.date else date.today()
    r = check(d)
    print(json.dumps(r) if a.json else f"{r['date']}: {r['reason']}")
    sys.exit(0 if r["tradable"] else 1)
