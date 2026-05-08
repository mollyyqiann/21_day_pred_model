"""Extension classifier — distinguishes fresh setups from extended stocks.

Used to prevent the model from re-recommending stocks that have already
made their run. INTC at $99 (+143% in 5 weeks via earnings catalyst) is
DIFFERENT from INTC at $41 (the bottom). The model needs to know.

Categories:

  FRESH        — no significant prior run; classic setup. Primary picks.
  MILD         — modest 20d move (15-30%); still actionable.
  CATALYST     — large 20d move BUT concentrated in 1-2 big single-day spikes
                 (earnings/news rerating). Secondary with caveat — re-rated
                 to new fair price; partial follow-through possible but mean
                 reversion AND continuation are both muted.
  GRADUAL      — large 20d move WITHOUT concentrated big days (slow grind).
                 High mean-reversion risk. Avoid.
  EXTREME      — 60d return > +60% OR 180d return > +100% OR 20d > +50%.
                 Multi-leg run already exhausted. Do not chase regardless of
                 catalyst pattern.
  COOLED       — recent pullback (5d < -8%) on a previously extended name.
                 Could be an entry on a healthy correction OR a top — judge
                 case by case.

The thresholds are tuned for the SP500 universe (Option 1B test fold gave
median 20d return at entry = +9% for picks; >30% is the right warning line;
>60% is danger zone).
"""

from __future__ import annotations
from typing import Optional


def classify_extension(
    ret_5d: Optional[float] = None,
    ret_20d: Optional[float] = None,
    ret_60d: Optional[float] = None,
    ret_180d: Optional[float] = None,
    up_bigdays_20d: Optional[float] = None,
    atr_pct: Optional[float] = None,
) -> dict:
    """Classify a stock's extension level. All returns are decimals (0.30 = 30%).

    Returns dict with:
      - level: one of FRESH | MILD | CATALYST | GRADUAL | EXTREME | COOLED
      - reason: short explanation
      - safe_to_recommend: bool — should this be a primary buy recommendation?
      - sort_priority: int — lower number = higher priority for buy lists
    """
    r5 = ret_5d if ret_5d is not None else 0.0
    r20 = ret_20d if ret_20d is not None else 0.0
    r60 = ret_60d if ret_60d is not None else 0.0
    r180 = ret_180d if ret_180d is not None else 0.0
    big = up_bigdays_20d if up_bigdays_20d is not None else 0.0
    atr = atr_pct if atr_pct is not None else 0.0

    # ATR-based safety override — even if returns look modest, a stock with
    # extreme daily-range volatility (atr > 30%) is too unstable for small-N
    # portfolios. CAR (atr 59%) dropped -15% in 5 days post-classification
    # despite a "safe" extension label. Reject these regardless of return.
    if atr > 0.30:
        return {
            "level": "VOLATILE",
            "reason": f"atr {atr:.0%} too volatile for small-N entry",
            "safe_to_recommend": False,
            "sort_priority": 95,
        }

    # EXTREME — multi-leg run, model history (SNDK at $1,097) shows mean +13%
    # vs +29% on milder partials. Chasing this is the documented mistake.
    if r60 > 0.60 or r180 > 1.00 or r20 > 0.50:
        return {
            "level": "EXTREME",
            "reason": (f"60d {r60:+.0%}, 180d {r180:+.0%}, 20d {r20:+.0%} — "
                       f"multi-leg run already exhausted"),
            "safe_to_recommend": False,
            "sort_priority": 99,
        }

    # CATALYST — large 20d move concentrated in 2+ big-up days (earnings spike).
    # Different stats: re-rated to new fair value; lower vol going forward.
    if r20 > 0.30 and big >= 2:
        return {
            "level": "CATALYST",
            "reason": (f"20d {r20:+.0%} from {big:.0f} big-up days "
                       f"(earnings/news rerating)"),
            "safe_to_recommend": True,  # ok with caveat
            "sort_priority": 30,
        }

    # GRADUAL — large 20d without catalyst (sustained speculative grind).
    # High mean-reversion risk; avoid.
    if r20 > 0.30 and big < 2:
        return {
            "level": "GRADUAL",
            "reason": (f"20d {r20:+.0%} sustained grind ({big:.0f} big days) — "
                       f"speculative extension"),
            "safe_to_recommend": False,
            "sort_priority": 90,
        }

    # COOLED — previously extended name now in 5d pullback. Often a setup.
    if r5 < -0.08 and r20 > 0.10:
        return {
            "level": "COOLED",
            "reason": f"recent pullback (5d {r5:+.0%}) on prior strength (20d {r20:+.0%})",
            "safe_to_recommend": True,
            "sort_priority": 15,
        }

    # MILD — modest run, still actionable
    if r20 > 0.15:
        return {
            "level": "MILD",
            "reason": f"modest run (20d {r20:+.0%})",
            "safe_to_recommend": True,
            "sort_priority": 20,
        }

    # FRESH — no significant prior move. Primary picks.
    return {
        "level": "FRESH",
        "reason": f"no prior run (20d {r20:+.0%})",
        "safe_to_recommend": True,
        "sort_priority": 10,
    }


def attach_extension(df, col_prefix=""):
    """Apply classify_extension to every row of a DataFrame.
    Expects columns: ret_5d_lag, ret_20d_lag, optionally ret_60d_lag,
    ret_180d_lag, up_bigdays_20d, atr_pct.
    Adds: ext_level, ext_reason, ext_safe, ext_priority.
    """
    levels, reasons, safes, prios = [], [], [], []
    for _, r in df.iterrows():
        c = classify_extension(
            ret_5d=r.get("ret_5d_lag"),
            ret_20d=r.get("ret_20d_lag"),
            ret_60d=r.get("ret_60d_lag"),
            ret_180d=r.get("ret_180d_lag"),
            up_bigdays_20d=r.get("up_bigdays_20d"),
            atr_pct=r.get("atr_pct"),
        )
        levels.append(c["level"])
        reasons.append(c["reason"])
        safes.append(c["safe_to_recommend"])
        prios.append(c["sort_priority"])
    df = df.copy()
    df[f"{col_prefix}ext_level"] = levels
    df[f"{col_prefix}ext_reason"] = reasons
    df[f"{col_prefix}ext_safe"] = safes
    df[f"{col_prefix}ext_priority"] = prios
    return df
