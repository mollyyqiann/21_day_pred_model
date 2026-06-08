# Daily burst-prediction service — setup guide

## What this does

Every weekday at **08:45 America/New_York**, a job runs locally on your Mac:

1. Refreshes the v4 (>$40) and v5 (upside-asymmetric) panels from yfinance.
2. Pulls pre-market prices (~15 min Yahoo lag) and overnight news for every universe ticker.
3. Scores each ticker with the augmented classifier and the three regression heads.
4. Writes today's log to `output/daily_log/YYYY-MM-DD.json`.
5. Compares yesterday's top-5 predictions against the realized returns that are now observable and computes direction-hit rates and MAE at 1d / 3d / 5d.
6. Formats a compact text message and (if configured) sends it via iMessage.

Example message body:

```
📈 Burst predictions — 2026-04-20

Yesterday's predictions (realized):
  v4 1d: dir 80%  MAE 1.12%  (pred +0.28% vs real +0.45%)
  v4 3d: dir 60%  MAE 2.05%  (pred +1.40% vs real +1.10%)
  v5 1d: dir 60%  MAE 1.30%  (pred +0.10% vs real −0.35%)

Top v4:
  SNDK  p=39%  1d +0.2% / 3d +1.4% / 5d +3.4%  news +3
  APP   p=32%  1d +0.4% / 3d +2.0% / 5d +3.3%  news +0
  …
Top v5:
  WDC   p=34%  1d −0.0% / 3d +0.3% / 5d +0.6%  news +3
  …
```

## Quickstart (local Mac)

### 1. One-time dependencies

Already installed on your box: `yfinance`, `pandas`, `numpy`, `scikit-learn`, `joblib`. Nothing else needed for the daily runner.

### 2. Configure the recipient

On first run, the script writes `config/burst_daily.json` with defaults. Edit it:

```json
{
  "imessage_recipient": "+15551234567",   // your iMessage handle (phone OR iCloud email)
  "send_imessage": true,                   // flip to true after you've tested manually
  "top_n_per_universe": 5,
  "universes": ["v4", "v5"]
}
```

For iMessage, the recipient must already be a contact that Messages.app knows how to iMessage — try sending yourself a manual message first to confirm the handle resolves blue.

### 3. Verify end-to-end — dry run

```bash
cd /Users/mollyqian/Desktop/stocks
python3 code/29_burst_daily_runner.py --as-of-close --dry-run
```

This runs the full pipeline with `overnight_gap=0` and prints the message without sending.

### 4. Verify the iMessage path manually

Set `send_imessage: true` in config, then:

```bash
python3 code/29_burst_daily_runner.py --as-of-close
```

You should receive the message on your iPhone/Mac. The first time you run this, macOS will prompt you to grant Terminal (or your shell) permission to control Messages — **you must approve this** in System Settings → Privacy & Security → Automation.

### 5. Install the scheduler

```bash
cp /Users/mollyqian/Desktop/stocks/scheduler/com.user.burstpredict.plist \
   ~/Library/LaunchAgents/com.user.burstpredict.plist
launchctl load -w ~/Library/LaunchAgents/com.user.burstpredict.plist
```

To verify it's loaded:

```bash
launchctl list | grep burstpredict
```

To run it once on demand:

```bash
launchctl start com.user.burstpredict
```

To remove:

```bash
launchctl unload ~/Library/LaunchAgents/com.user.burstpredict.plist
```

Logs are written to `output/daily_log/launchd.stdout.log` and `launchd.stderr.log`.

### 6. Mac must be awake at 08:45 ET

`WakeToRun` in the plist tells launchd to wake your Mac from sleep if needed. For this to work reliably:

- Your Mac must be plugged in (most reliable) OR have "wake for network access" enabled.
- Run `pmset -g sched` after installing the plist to verify the wake schedule is registered.
- If you have FileVault, pre-boot screen blocks wake — better to leave the Mac logged in.

### 7. Market holidays

The script runs on every weekday; on a market holiday yfinance will return yesterday's close and the message will just repeat the previous day's evaluation. If you want stricter behavior, add a holiday-check using the `pandas_market_calendars` package — not installed by default to keep the dependency set minimal.

---

## Do I need to host this somewhere?

**tl;dr — no, local Mac is the right place for this, because of iMessage.**

| Option | iMessage? | Cost | Reliability | Notes |
|---|---|---|---|---|
| **Local Mac + launchd** (recommended) | ✅ native | free | high if Mac stays awake/online | what this setup uses |
| Cloud VM (AWS/GCP/Fly.io etc.) | ❌ not natively | ~$5–15/mo | very high | must swap iMessage for SMS/Pushover/Telegram |
| Cloud VM + BlueBubbles bridge | ⚠️ indirect | same + bridge complexity | fragile | requires a dedicated Mac acting as iMessage relay — defeats the purpose |
| GitHub Actions (free tier) | ❌ | free | timezone-sloppy, 5-min scheduling granularity | possible for CSV output + email, not iMessage |

**Why local is actually the best fit here:**

- iMessage only works natively on a logged-in Mac. Anything cloud-hosted would require replacing iMessage with SMS/Pushover/Telegram.
- The yfinance calls (~700 tickers) take ~90 s — no resource pressure, doesn't interfere with other work.
- Data and models stay on your machine; no creds to manage.

**If you ever want cloud,** the cheapest clean path is:

1. Rent a $5/mo VPS (DigitalOcean / Hetzner / Fly.io).
2. Replace the `send_imessage` helper with a Pushover (~$5 one-time) or Telegram-bot (free) notification — one function swap.
3. Cron the runner at `45 12 * * 1-5` (UTC → 08:45 ET).

I can build that swap if you decide to go that route. Until then: the local setup above is a one-command install.

---

## Files created by this work

- [`code/28_burst_regression_heads.py`](code/28_burst_regression_heads.py) — trains 1d/3d/5d return regressors (one-time)
- [`code/29_burst_daily_runner.py`](code/29_burst_daily_runner.py) — the daily job
- [`scheduler/com.user.burstpredict.plist`](scheduler/com.user.burstpredict.plist) — launchd schedule (Mon–Fri 08:45 local time)
- [`config/burst_daily.json`](config/burst_daily.json) — recipient + flags (written on first run)
- `output/daily_log/` — per-day JSON log of predictions and realized accuracy

## Model artifacts used by the runner

- `models/burst_gbc_v6_augmented.joblib` — v5 universe classifier
- `models/burst_gbc_v6b_augmented.joblib` — v4 universe classifier
- `models/burst_reg_v6_fwd_{1d,3d,5d}.joblib` — v5 regressors
- `models/burst_reg_v6b_fwd_{1d,3d,5d}.joblib` — v4 regressors
