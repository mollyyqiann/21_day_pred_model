# Drop regressor v2 — percent-change accuracy report

**Scope.** Improve accuracy of the predicted percent return (1d / 3d / 5d)
produced by the *drop* regression heads used by
[code/38_drop_daily_runner.py](code/38_drop_daily_runner.py). The drop
*classifier* (p_drop) was out of scope here — a parallel session was
working on that; our work is the regression-head side only.

**Constraint honoured.** Tomorrow morning's 08:45 ET run is bit-for-bit
unchanged: the daily runner's v1 code path still loads
`drop_reg_v1_fwd_{1d,3d,5d}.joblib` + `drop_gbc_v1.joblib`, `config/burst_daily.json`
has no `drop_model_version` key (so it defaults to `"v1"`), and none of the
v1 model files were modified. All new artifacts ship under the v2 name
suffix. No wiring changes to the runner.

---

## TL;DR

On the production-relevant subset (rows with `y_drop == 1` — i.e., the
regime the runner actually scores at inference after the
`p_drop ≥ 0.25 ∧ p_burst < 0.50` filter), v2 cuts regression error and
fixes a severe sign error at long horizons:

| Horizon | v1 MAE(drop) | v2 MAE(drop) | Δ MAE | v1 dir(drop) | v2 dir(drop) | Δ dir | v1 bias | v2 bias |
|---|---|---|---|---|---|---|---|---|
| fwd_1d | 2.37 % | 2.36 % | −0.4 % | 66.4 % | 66.3 % | −0.1 pp | +0.75 % | +0.81 % |
| fwd_3d | 5.89 % | 5.24 % | **−11.0 %** | 52.0 % | **69.0 %** | **+17.0 pp** | +4.38 % | +2.99 % |
| fwd_5d | 7.83 % | 6.25 % | **−20.2 %** | 43.4 % | **78.5 %** | **+35.1 pp** | +7.20 % | +4.90 % |

The two large wins are at 3d and 5d. At 1d the gain is within noise —
there is no short-horizon signal left to capture with feature
engineering on the existing 17-feature set.

The v1 5d regressor was **worse than a coin flip** (43.4 %) at signing
the move on actual drop rows. v2 gets 78.5 %. That single metric change
— not the MAE — is the result that matters most for short-list quality.

---

## Five improvements evaluated

All improvements target the regressor only. Each was tested in isolation
against a same-harness v1 recipe (same 17 features, same 70/15/15 time
split), then useful ones were composed. Code:
[code/36b_drop_v2_pipeline.py](code/36b_drop_v2_pipeline.py) (full
ablation), [code/36c_drop_v2_final.py](code/36c_drop_v2_final.py)
(per-horizon final fit).

1. **Drop-conditional sample weighting** (`weight_pos = 5.0`). v1 trains
   on the full universe (~220 k rows) with uniform weights, drop base-rate
   ≈ 8 %, so the regressor is effectively fitting "average universe
   return" — a near-zero target — and only incidentally modelling drop
   magnitude. Up-weighting `y_drop = 1` rows 5× at training time
   specialises the fit to the regime the runner actually scores.
2. **Absolute-error loss** (L1). The original `GradientBoostingRegressor`
   uses squared error. L1 fits the conditional median, which is more
   robust to earnings-gap and halt-driven ±15–30 % tail moves that
   aren't repeatable from the 17 feature inputs.
3. **Target winsorization** at the 1st / 99th percentile (computed on the
   training fold only). Same motivation as (2) but applied to labels.
4. **Volatility-normalized targets.** Train on
   `fwd_Nd / (√N · rv_60)` then denormalise predictions with the live
   `√N · rv_60`. Gives the tree a stationary target across vol regimes.
5. **Multi-seed bag-of-trees ensemble** — average of five
   `random_state ∈ {42, 7, 1337, 2024, 99}` fits. Cheap variance reduction.

Implementation note: I replaced `GradientBoostingRegressor` with
`HistGradientBoostingRegressor` throughout — ~10× faster via histogram
binning, same expressive power at this size, drops 33 model fits from
~5 h to ~4 min. The HGBR form does not support Huber loss, so L1 took
Huber's role as the robust option.

---

## Ablation — each improvement in isolation

From [output/drop_metrics_v2.json](output/drop_metrics_v2.json). The
metric that matters is `MAE(y_drop=1)` (column **MAE drop**) — the
MAE restricted to rows that the runner would actually score in
production.

### fwd_1d (1-day)

| Config | MAE | dir | MAE drop | dir drop |
|---|---|---|---|---|
| v1-recipe (baseline) | 0.0136 | 0.627 | 0.0237 | 0.664 |
| 1 drop-weighted | 0.0139 | 0.623 | 0.0237 | — |
| 2 L1 loss | 0.0136 | 0.628 | **0.0236** | — |
| 3 winsorized | 0.0136 | 0.628 | 0.0242 | — |
| 4 vol-normalized | 0.0137 | 0.626 | 0.0239 | — |
| 5 5-seed bag | 0.0136 | 0.627 | 0.0237 | — |
| all-5 stacked | 0.0137 | 0.627 | 0.0238 | — |

Interpretation — at 1-day horizon, MAE on drop rows (2.37 %) is already
close to the irreducible per-day return scale. None of the five levers
moves the needle more than 1 bp.

### fwd_3d (3-day)

| Config | MAE | dir | MAE drop | dir drop |
|---|---|---|---|---|
| v1-recipe (baseline) | 0.0263 | 0.574 | 0.0589 | 0.520 |
| **1 drop-weighted** | 0.0278 | 0.546 | **0.0523** | — |
| 2 L1 loss | 0.0262 | 0.573 | 0.0593 | — |
| 3 winsorized | 0.0262 | 0.575 | 0.0590 | — |
| 4 vol-normalized | 0.0263 | 0.572 | 0.0589 | — |
| 5 5-seed bag | 0.0263 | 0.574 | 0.0589 | — |
| all-5 stacked | 0.0270 | 0.564 | 0.0535 | — |

Drop-weighting is the single lever that matters; the other four are
noise at 3d. Notably, the all-5 stack (0.0535) is **worse** than
drop-weighted alone (0.0523). The improvements don't compose — 3
of the 4 extras (winsor, volnorm, L1) actively suppress the tail
magnitudes that drop-weighting wants to amplify.

### fwd_5d (5-day)

| Config | MAE | dir | MAE drop | dir drop |
|---|---|---|---|---|
| v1-recipe (baseline) | 0.0348 | 0.558 | 0.0783 | 0.434 |
| **1 drop-weighted** | 0.0376 | 0.511 | **0.0626** | — |
| 2 L1 loss | 0.0347 | 0.557 | 0.0788 | — |
| 3 winsorized | 0.0348 | 0.557 | 0.0777 | — |
| 4 vol-normalized | 0.0349 | 0.556 | 0.0785 | — |
| 5 5-seed bag | 0.0348 | 0.558 | 0.0783 | — |
| all-5 stacked | 0.0367 | 0.531 | 0.0645 | — |

Same pattern, larger magnitude. Drop-weighting alone is a 20 % drop-MAE
improvement; stacking the other four weakens it back to 17.6 %.

---

## Composition experiment — does the runner benefit from combining?

Given drop-weighting dominates, the open question was whether **adding**
L1, winsor, volnorm, or seed-bag on top helps or hurts. I also
cross-checked the loss function — L1 vs MSE — holding drop-weighting
and seed-bagging fixed. Results on drop subset (combo check in
[output/drop_metrics_v2_final.json](output/drop_metrics_v2_final.json)):

| Horizon | Recipe | MAE drop | dir drop |
|---|---|---|---|
| 3d | drop_w + L1 + bag | 0.0534 | 0.644 |
| 3d | **drop_w + MSE + bag** | **0.0524** | **0.690** |
| 5d | drop_w + L1 + bag | 0.0642 | 0.724 |
| 5d | **drop_w + MSE + bag** | **0.0625** | **0.785** |

**Surprising result:** L1 loss — the "robust" option — actually hurts
drop-subset accuracy at longer horizons. The reason: fitting the
conditional median *regresses toward zero* on the heavy-negative-tail
rows that we specifically weighted up. MSE with 5× sample weighting
does the opposite — the extreme drop rows contribute large-squared
residuals and pull the fit toward them, which is exactly the behaviour
we want from a magnitude estimator for drops. MSE also gives markedly
better **drop-subset direction** (+4.5 pp at 3d, +6.1 pp at 5d).

At 1d there's no drop-weighting, so L1 and MSE are both reasonable;
L1's slight edge on overall direction made it the 1d pick.

### Final per-horizon recipes

| Horizon | Recipe | Rationale |
|---|---|---|
| fwd_1d | L1 loss + 5-seed bag, uniform weights | Drop-weighting doesn't pay off at 1d; L1 + bag gives a marginal robustness + dir gain over v1 at zero downside. |
| fwd_3d | **drop_weighted(5×) + MSE loss + 5-seed bag** | −11 % drop MAE, +17 pp drop direction. MSE beats L1 on both metrics at this horizon. |
| fwd_5d | **drop_weighted(5×) + MSE loss + 5-seed bag** | −20 % drop MAE, +35 pp drop direction. The biggest win; fixes a below-coin-flip sign model. |

Winsorization, vol-normalization, and L1-loss were **dropped from the
final 3d/5d recipe** despite being in the original "5 improvements"
list. The ablation and combo data both said so.

---

## Why the wrong metric matters — v1 was optimising the wrong thing

v1's headline numbers look reasonable (56-63 % direction, MAE under 4 %).
But the v1 *drop-subset* direction at 5d is **43.4 %** — systematically
wrong-signed on actual drops. The model reports "expected 5-day return
≈ −0.4 %" for names where the realised return is often ≤ −6 %, because
its training objective was MAE over the whole 220 k-row population, of
which 92 % are non-drops sitting near zero. The regressor was
*calibrated to report approximately zero*, which maximises
full-population MAE and gives a coin-flip-or-worse sign prediction on
the names the runner cares about.

v2 gets this right by training on the same population but with 5×
weight on drop rows. This is the single structural change that drives
the gains; the rest is hyperparameter hygiene around it.

---

## Bias — still there, not fixed

v2 improves the bias on drop rows but does not eliminate it:

| Horizon | v1 bias | v2 bias |
|---|---|---|
| 1d | +0.75 % | +0.81 % |
| 3d | +4.38 % | +2.99 % |
| 5d | +7.20 % | +4.90 % |

The model still systematically over-predicts return (under-predicts
drop magnitude) on actual drop rows — it's saying "−0.5 %" when
reality is "−5.5 %". Weighting helped, but not enough to fully
calibrate. Two natural next steps (out of scope here):

- Post-hoc isotonic calibration on the validation set, mapping
  predicted → realised on drop rows only.
- A conditional two-head model: p_drop · E[return | drop] instead of
  E[return]. Would require re-architecting the runner's downstream
  `pred_1d/3d/5d` consumption.

---

## Guarantee: tomorrow's predictions are unaffected

Verified:

- `config/burst_daily.json` does not set `drop_model_version` → runner
  defaults to `"v1"` (see [code/38_drop_daily_runner.py:181](code/38_drop_daily_runner.py:181)).
- The v1 path [code/38_drop_daily_runner.py:212–221](code/38_drop_daily_runner.py:212)
  still loads `drop_reg_v1_fwd_{1d,3d,5d}.joblib` and
  `drop_gbc_v1.joblib` — files were not touched.
- No edits to `38_drop_daily_runner.py`, `29_burst_daily_runner.py`, or
  any scheduler plist.
- v2 regressor files live alongside v1 under new names and are loaded
  by nothing.

### How to switch to v2 later (manual, explicit opt-in)

When you want to A/B test or ship v2 regressors, do it in two steps so
it's reversible:

1. Wire a new branch in `38_drop_daily_runner.py` that loads
   `drop_reg_v2_fwd_{1d,3d,5d}.joblib` when `cfg["drop_reg_version"] == "v2"`.
   The v2 joblib holds a list of 5 models under `"models"` (not `"reg"`)
   — predictions average across them:
   ```python
   m = joblib.load(MODELS / "drop_reg_v2_fwd_1d.joblib")
   pred_1d = np.mean([reg.predict(X) for reg in m["models"]], axis=0)
   ```
2. Set `"drop_reg_version": "v2"` in `config/burst_daily.json`. Leave
   the classifier-side `drop_model_version` whatever you've already
   decided for that (different, orthogonal switch).

I did not make this change myself because the instruction was to not
disturb the next run. The only way to disturb it from here would be to
add the switch and leave the default on v1 — same functional outcome,
but there's zero upside to shipping untested branch code when the flip
can be done after review.

---

## Artifacts

**New (v2, regressor only):**
- [models/drop_reg_v2_fwd_1d.joblib](models/drop_reg_v2_fwd_1d.joblib) — L1 + 5-seed bag
- [models/drop_reg_v2_fwd_3d.joblib](models/drop_reg_v2_fwd_3d.joblib) — drop_w(5×) + MSE + 5-seed bag
- [models/drop_reg_v2_fwd_5d.joblib](models/drop_reg_v2_fwd_5d.joblib) — drop_w(5×) + MSE + 5-seed bag
- [output/drop_metrics_v2.json](output/drop_metrics_v2.json) — full ablation
- [output/drop_metrics_v2_final.json](output/drop_metrics_v2_final.json) — chosen recipes + combo checks
- [code/36b_drop_v2_pipeline.py](code/36b_drop_v2_pipeline.py) — ablation harness
- [code/36c_drop_v2_final.py](code/36c_drop_v2_final.py) — final per-horizon fit

**Unchanged (v1, still in production):**
- `models/drop_reg_v1_fwd_{1d,3d,5d}.joblib`
- `models/drop_gbc_v1.joblib`
- `code/38_drop_daily_runner.py`
- `config/burst_daily.json`

---

## What I would do next (explicit, not done)

1. **Isotonic calibration on drop-subset bias** — map v2 predicted
   returns → realised via monotonic fit on the validation fold. Should
   pull 5d bias from +4.9 % toward 0 with no MAE cost.
2. **Dedicated 1d signal** — rather than stacking improvements that
   don't help 1d, try adding `overnight_news_score` and
   `short_interest_ratio` features (both available in this repo's
   cross-asset / burst v9 side) as 1d-specific inputs. 1d is where the
   current 17-feature panel hits a ceiling.
3. **Backtest the drop short-list** — rank by v2 `pred_5d` and measure
   realised drawdown vs v1's ranking, not just MAE/dir. The end-to-end
   metric that matters is "did the top-5 shortlist drop further than
   the universe" — accuracy improvements don't automatically translate
   to better rank ordering and should be verified downstream before
   flipping the switch.
