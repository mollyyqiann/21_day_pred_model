# Drop Alert — Morning Drop-Risk Notifications

Cross-sectional drop-risk scoring: which tickers are at elevated risk of a near-term drop? Two outputs: (1) watch-list of recent burst-rec names whose drop probability has now risen, (2) fresh top-5 drop alerts ranked by `p_drop`. Sent ~35 minutes after the morning burst run.

## Schedule
- **Mon–Fri 09:20 ET** (5 min before market open, after the 08:45 burst run has finished).
- Plist: [`scheduler/com.user.dropalert.plist`](../../scheduler/com.user.dropalert.plist).
- Script: [`code/38_drop_daily_runner.py`](../../code/38_drop_daily_runner.py).

## Model (current production = v1)
- **`drop_gbc_v1.joblib`** — drop classifier (probability of drop event). v8 features.
- **`drop_reg_v1_fwd_{1d,3d,5d}.joblib`** — return regression heads at 1d / 3d / 5d.
- Filter at inference: `p_drop ≥ 0.25 ∧ p_burst < 0.50` (avoid double-flagging burst candidates).

## v2 regressors (built but NOT in production)
- v2 regression heads exist at `models/drop_reg_v2_fwd_{1d,3d,5d}.joblib` but are **not loaded** by the runner unless `config/burst_daily.json` adds `"drop_reg_version": "v2"`. See [REPORT_DROP_V2.md](REPORT_DROP_V2.md) for the recipe and gains (−20% MAE on 5d drop subset, +35pp drop-direction at 5d).
- The classifier (`drop_gbc_v1`) is unchanged in v2 work — only regression heads were touched.

## Output
- **Notification** (separate from burst — config key `send_drop_notifications`).
- `output/drop_live_today.csv` — today's drop picks.
- `output/drop_scores_all.csv` — full score panel.
- Daily JSON log: `output/daily_log/{YYYY-MM-DD}_drop.json`.

## Two-part message structure (per [`38_drop_daily_runner.py:1`](../../code/38_drop_daily_runner.py:1))
1. **Watch list from prior recommendations**: scans the last 7 days of burst-prediction logs (`output/daily_log/*.json`) and flags any names whose drop probability is now ≥ `drop_prob_warn_threshold` (default 0.30).
2. **Fresh drop top-5**: today's universe ranked by `p_drop`, surface top 5.

## Hold horizon
- 1–5 day risk window — same horizon as burst but inverse direction.

## Distinguishing features (vs other systems)
- **Inverse-of-burst** signal: ranks by `p_drop`, not `p_burst`.
- **Consumes burst's daily log** for watch-list functionality — only system that reads other systems' outputs.
- **Sends as a SEPARATE message** so the burst and drop signals don't entangle.
- Filter explicitly excludes high `p_burst` rows.

## Related docs in this folder
- [`REPORT_DROP_V2.md`](REPORT_DROP_V2.md) — v2 regressor research. Documents the ablation, why MSE+drop-weighting beat L1, the residual bias at 5d, and the explicit opt-in path to flip production to v2.

## Key files
- Script: [`code/38_drop_daily_runner.py`](../../code/38_drop_daily_runner.py)
- Plist: [`scheduler/com.user.dropalert.plist`](../../scheduler/com.user.dropalert.plist)
- Config (shared): [`config/burst_daily.json`](../../config/burst_daily.json)
- Production models: `models/drop_gbc_v1.joblib`, `models/drop_reg_v1_fwd_{1d,3d,5d}.joblib`
- v2 regressors (dormant): `models/drop_reg_v2_fwd_{1d,3d,5d}.joblib`
- Output: `output/drop_live_today.csv`, `output/drop_scores_all.csv`
- Daily log: `output/daily_log/{YYYY-MM-DD}_drop.json`
