# SAC fixed discount-horizon comparison

The saved full-state SAC policies beat matched holding on a fitted window but
fail development. Their mostly-long development paths and previous weak TD3/PPO
and regularization outcomes remain visible. All earlier SAC protocols used
`gamma=0.99`. This study tests one different learning horizon, not a promise
of stronger returns or a continuation of interrupted regularization.

## Fixed hypothesis

[Protocol](../configs/experiments/sac_discount.json) compares fresh control0.99
and candidate0.95 discounts. The geometric sum1/(1−gamma) is100 versus20 steps;
normalized mass in the first20 rewards is18.21% versus64.15%. These are discount
weights, not forced holding periods or proof of financial-objective equivalence.
The hypothesis is that shorter discounted targets may reduce sensitivity to
noisy distant estimates. It changes discounted optimization; whether it helps
stocks, ETFs or shorts is unknown until the predefined comparison.

Both arms use exact expanded111-symbol training data cdb93a33…, original26
validation episodes, original immutable normalization,21 full-state inputs,
standard64×64 SAC actor/critic/target initialization, net-log rewardscale1,
auto alpha initial1, target entropy−4, default Adam with no weight decay and
unchanged costs/actions/20-session holding cap. No feature, numerical flush,
reward, temperature, universe or sampler change. Same seeded stock/ETF cycles
and full512 windows; actual symbol/group counts must match, all symbols must
receive transitions. No future outcome labels enter learning or selection.

## Execution and acceptance

Two1024-transition CPU smokes validate actual gamma, initial actor/critic/target
and alpha equality, Adam settings, save/load and frozen predictions. Version21
requires the new current checkpoint-report/hash manifest in every completed
archive. Final reports must be post-update and match saved weights/steps;
selected models remain separately identified. Actual default Colab Python must
match the lock before fitting. All learning stays in Colab; construction tests
perform no optimizer updates locally.

After smoke acceptance, screen seed11 with fresh100k cells/evaluations every10k.
Candidate selected and final gains must each be≥2pp versus control, final group
gains nonnegative and drawdown worsening≤2pp. Selected/final stock AND ETF excess
and median stock excess must each be≥−2pp against matched holding. No early peak
alone qualifies. Failure stops this family with no gamma sweep, extra steps,
confirmation or promotion. Only a passing screen permits seeds11/23/37 under
fixed confirmation gates; prospective evidence remains required. Short success
requires positive total return AND positive net short P&L; future declining-stock
cohorts remain reporting-only. Retired2024–September2026 data stays unopened.

## Budget

Use newly frozen source with account-floor enforcement. Current cumulative
conservative37.787213/150 is retained after explicit reviewed resumption.
Begin project teardown130; independently protect account minimum150 by starting
teardown170, preserving20 for shutdown/metering delay. Maximum2 incremental
units for this study includes smokes/setup/idle/downloads/failures/retries.
Use the shared aggregate ledger, one-minute meter and7200-second allocation
limits. Unknown metering stops every owned session; verify cleanup and durable
retrieval. Never stop unrelated sessions or restart automatically after a stop.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_discount_jobs.py \
 --smoke --output .runtime/discount-smoke-jobs.json
```

Current implementation includes fixed job/payload/worker contracts, actual model
records, mandatory checkpoint identities and restored-weight report validation.
Actual immutable smoke payloads validate. No Colab allocation has occurred for
this study at preparation; actual smoke acceptance and returns are pending.

Preparation acceptance:59 focused model/job/archive/checkpoint/account/fleet/
report tests pass without local fitting; actual immutable payload checks and
script compilation pass. Actual initial/final/selected saved model step counts
are recorded and validated alongside gamma and optimizer state. Source freeze
and official account usage check precede twoCPU smokes.

## Actual Colab smoke acceptance

Two1024 CPU smokes pass on actual default Python3.13.16, frozen workercf41830.
Actual gamma, seeded network/alpha equality, matched sampling and selected/final
saved model steps/optimizers restore correctly.156 exact frozen predictions,
immutable normalization/input rejection and mandatory checkpoint identity pass.
Both sessions stopped. Project37.805591/150; account349.96, zero rate/assignments,
floor150/teardown170 retained. Fixed two100k screen next under original2-unit
study allowance, same worker; no screen outcome or promotion yet.
[Evidence](../results/sac-discount-2026-10-21/README.md).
