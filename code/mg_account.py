"""Which Robinhood account the MG v3 executor is allowed to trade.

Kept out of the source tree on purpose. The account number is not a credential
-- you cannot do anything with it without authenticating -- but this repo is
public, and pairing a real brokerage account number with a full description of
what gets traded in it is a disclosure with no upside.

Resolution order:
  1. $MG_RH_ACCOUNT
  2. config/mg_execution.json -> {"robinhood_account": "..."}
Raises if neither is set, because every caller here places or reconciles real
orders and a silent default would be the wrong kind of convenient.
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "mg_execution.json"


def account() -> str:
    v = os.environ.get("MG_RH_ACCOUNT", "").strip()
    if v:
        return v
    if CONFIG.exists():
        v = str(json.loads(CONFIG.read_text()).get("robinhood_account", "")).strip()
        if v:
            return v
    raise RuntimeError(
        f"No Robinhood account configured. Set MG_RH_ACCOUNT or create {CONFIG} "
        'containing {"robinhood_account": "<the agentic account number>"}.')
