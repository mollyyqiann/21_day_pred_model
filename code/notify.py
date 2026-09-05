"""Pluggable notification backend. One function `send(body)` used everywhere.

Selects backend from config/burst_daily.json:

  {
    "notifier": "telegram",          // "telegram" | "imessage"
    "telegram_token":   "123:ABC...",  // from @BotFather
    "telegram_chat_id": "123456789",   // integer, as string
    "imessage_recipient": "+1..."     // fallback if notifier=="imessage"
  }

No dependencies beyond stdlib — uses urllib for Telegram so the background
launchd job doesn't need anything extra.
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "burst_daily.json"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def send(body: str, *, dry_run: bool = False, channel: str = "default") -> bool:
    """Send `body` using the configured backend. Returns True on success.

    `channel` selects which Telegram bot receives it:
      "default" -- the usual bot, used by every prediction/report job
      "trade"   -- the separate @Claude_Trade_robinhood_bot, used ONLY by the
                   15:55 executor's fill receipts. Kept apart on purpose: those
                   are the messages that report real money moving, and they
                   should not be buried in the daily stream of model output.
    Falls back to the default bot if the trade channel is not configured, so a
    missing key can never silently swallow an execution receipt.
    """
    cfg = _load_config()
    if dry_run:
        print(f"[notify:DRY] {body[:120]}")
        return True
    notifier = cfg.get("notifier", "imessage")
    if notifier == "telegram":
        if channel == "trade":
            token = cfg.get("trade_telegram_token")
            chat = cfg.get("trade_telegram_chat_id")
            if token and chat:
                return _send_telegram({"telegram_token": token,
                                       "telegram_chat_id": chat}, body)
            print("[notify:telegram] trade channel not configured; "
                  "falling back to the default bot")
        return _send_telegram(cfg, body)
    if notifier == "imessage":
        return _send_imessage(cfg, body)
    print(f"[notify] unknown notifier {notifier!r}")
    return False


_TELEGRAM_LIMIT = 4000  # headroom under Telegram's hard 4096-char cap


def _split_for_telegram(body: str, limit: int = _TELEGRAM_LIMIT) -> list[str]:
    """Greedily pack body into <=limit chunks, splitting on blank-line section
    boundaries first (keeps picks/paragraphs intact) and falling back to a
    hard slice for any single section that's still too long on its own."""
    sections = body.split("\n\n")
    chunks: list[str] = []
    cur = ""
    for s in sections:
        candidate = f"{cur}\n\n{s}" if cur else s
        if len(candidate) > limit and cur:
            chunks.append(cur)
            cur = s
        else:
            cur = candidate
        while len(cur) > limit:
            chunks.append(cur[:limit])
            cur = cur[limit:]
    if cur:
        chunks.append(cur)
    return chunks or [""]


def _send_telegram(cfg: dict, body: str) -> bool:
    token = cfg.get("telegram_token") or ""
    chat = cfg.get("telegram_chat_id") or ""
    if not token or not chat:
        print("[notify:telegram] token or chat_id missing in config")
        return False
    chunks = _split_for_telegram(body)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    all_ok = True
    for i, chunk in enumerate(chunks):
        text = f"(part {i+1}/{len(chunks)})\n{chunk}" if len(chunks) > 1 else chunk
        data = urllib.parse.urlencode({
            "chat_id": str(chat),
            "text": text,
            "disable_web_page_preview": "true",
        }).encode()
        try:
            resp = urllib.request.urlopen(url, data=data, timeout=30).read()
            ok = b'"ok":true' in resp
            if not ok:
                print(f"[notify:telegram] API said: {resp.decode(errors='ignore')[:200]}")
            all_ok = all_ok and ok
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors='ignore')[:200] if hasattr(e, 'read') else str(e)
            print(f"[notify:telegram] HTTP {e.code}: {detail}")
            all_ok = False
        except Exception as e:
            print(f"[notify:telegram] {type(e).__name__}: {e}")
            all_ok = False
    return all_ok


def _send_imessage(cfg: dict, body: str) -> bool:
    recipient = cfg.get("imessage_recipient") or ""
    if not recipient:
        print("[notify:imessage] imessage_recipient missing in config")
        return False
    body_esc = body.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        f'tell application "Messages"\n'
        f'  set targetService to 1st service whose service type = iMessage\n'
        f'  set targetBuddy to buddy "{recipient}" of targetService\n'
        f'  send "{body_esc}" to targetBuddy\n'
        f'end tell')
    try:
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            print(f"[notify:imessage] FAILED rc={r.returncode}: {r.stderr.strip()}")
            return False
        return True
    except Exception as e:
        print(f"[notify:imessage] {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    # Quick send test:  python3 code/notify.py "hello"
    # Trade receipt:    python3 code/notify.py --channel trade "MG 15:55 EXEC ..."
    import sys
    args = sys.argv[1:]
    channel = "default"
    if len(args) >= 2 and args[0] == "--channel":
        channel, args = args[1], args[2:]
    msg = args[0] if args else "test from notify.py"
    ok = send(msg, channel=channel)
    print("ok" if ok else "FAILED")
