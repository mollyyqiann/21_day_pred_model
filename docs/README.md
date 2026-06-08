# Recurring Prediction Systems — Index

Each system has its own folder with an authoritative `README.md`. Read the system's README before reasoning about its outputs — multiple systems share words like "burst" and "daily" but have different models, horizons, and universes.

| Folder | Schedule | Model / horizon | Output | Channel |
|---|---|---|---|---|
| [`qqq_daily/`](qqq_daily/README.md) | Mon–Fri **09:00 ET** | RF+ET+HGB ensemble, **5d / 10d / 20d** direction+magnitude on **QQQ only** | `daily/last_message_*.txt`, `daily/predictions_log.csv` | Telegram |
| [`burst_morning/`](burst_morning/README.md) | Mon–Fri **08:45 ET** | v6 augmented GBC + 1d/3d/5d regressors, **1–5d burst** on calm large-caps | `output/daily_log/*.json` | Telegram or iMessage |
| [`drop_alert/`](drop_alert/README.md) | Mon–Fri **09:20 ET** | v1 drop GBC + regressors, **1–5d drop risk** | `output/drop_live_today.csv`, `output/daily_log/*_drop.json` | Telegram or iMessage (separate msg from burst) |
| [`market_briefing/`](market_briefing/README.md) | Mon–Fri **16:30 ET** | **No model** — Claude Haiku 4.5 narrative over indices/sectors/breadth/movers/insiders. Educational, **same-day** read | `output/market_briefing/{date}.md` | Telegram |
| [`sector_daily/`](sector_daily/README.md) | Mon–Fri **18:00 ET** | Composite score over filings/IR/earnings/insider, **multi-week to multi-month** hold windows | `output/sector_daily/{date}.md` | Telegram |

## Quick distinguishing rules

| If the question is about… | The system is… |
|---|---|
| QQQ alone, multi-horizon (5d/10d/20d) | qqq_daily |
| Pre-market top-N picks across SP500 | burst_morning |
| Drop risk, separate post-burst message | drop_alert |
| Post-close sector narrative + ranked candidates with weeks-to-months hold | sector_daily |

## Not in this index

- **Sunday verdict / Monthly Gainer v3** — [`code/102_sunday_check.py`](../code/102_sunday_check.py) — this is **NOT a recurring system** (per user 2026-05-03). It's a one-off check using a separate research model (Monthly Gainer v3, 21-trading-day +30% touch target). Model documented in root [`MONTHLY_GAINER_REPORT.md`](../MONTHLY_GAINER_REPORT.md). Don't conflate with burst or drop — totally different model and horizon.
- **`com.user.aleabitalpha`** (5 PM Mon-Fri) and **`com.user.amznrsi`** (5 PM Mon-Fri) — auxiliary scheduled jobs not on the user's main 5-system list. Not documented here.
- **`com.user.heartbeat`** (7:30 AM Mon-Fri) — system heartbeat / liveness monitor, not a prediction.
- **`com.user.finnhubws`** (4 AM Mon-Fri) — Finnhub websocket data collection, not a prediction.
- **`com.user.burstrefresh`** (10 AM Mon-Fri) — periodic mid-morning refresh of burst features; companion to `burst_morning`.
- **`com.user.intradaycollect`** (9:30 PM Mon-Fri) — intraday data collection.
- **`com.user.v11retrain`** (1st of month, 3 AM) — monthly burst v11 retrain pipeline.

## Why this structure exists

Future-Claude's mistake on 2026-05-03: confused the Monthly Gainer v3 model (21-trading-day, +30% touch target, multi-week shares hold) with the burst model (1–5d horizon, options-friendly). The two share vocabulary ("rank", "top-15", "raw_margin", "prob") but are different models with different targets and different appropriate hold strategies. Each system folder's README spells out its own model, horizon, and distinguishing features — read it cold before answering questions about that system.
