# Recurrent policy-update safeguard

Status: fixed protocol, construction/record contracts and forward-only library
acceptance prepared. Dedicated Colab jobs/worker/archive/report integration and
actual smoke acceptance remain. No runtime allocated for this study.

## Evidence and separate hypothesis

The [action audit](../results/cash-actions-2026-10-22/README.md) finds late shifts
toward losing shorts. [Optimizer evidence](../results/cash-optimizer-2026-10-22/README.md)
finds finite moments/weights and expected updates, with higher late KL/clipping.
However, the strongest posthoc control also has high late KL. The [fixed fitted
replay](../results/recurrent-fit-2026-10-22/README.md) finds weak fitted benchmarks
and mixed training/development contrasts. None proves an optimizer cause.

The next hypothesis changes one optimization safeguard on the strongest Gaussian
posthoc baseline. This is a distinct intervention, with fresh runs; the failed
cash-choice family remains closed. No categorical cash/binary variant, sizing
threshold, feature, reward or architecture change is involved.

## Fixed comparison

[Checksummed protocol](../configs/experiments/recurrent_guard.json):

| Arm | Library `target_kl` | Training/evaluation |
| --- | --- | --- |
| control | `None` | Adaptive Gaussian training; unit-sign evaluation/selection |
| kl_guard candidate | `0.01` | Identical action and execution contracts |

SB3-contrib2.7.0 evaluates approximate KL before each optimizer step and stops
remaining minibatches/epochs in that rollout if the estimate exceeds1.5×target.
This is not a guaranteed KL bound, a cap on accumulated policy drift, or a bound
on changes in executed trades after transforms/risk approval. Lower KL alone is
not performance success. Each new rollout uses its own previous-policy reference.

Both fresh100k seed11 runs retain expanded111 training symbols, original26
reused development episodes, original immutable normalization, masked21 inputs,
separate64-unit one-layer LSTMs,64×64 heads, standard initialization, rollout1000,
batch50,ten maximum epochs, learning rate0.0003,gamma0.99,clip0.2,gradient cap0.5,
net-log-equity reward and unchanged costs. Evaluate/save curves every10k. Initial
tensors and actual sampling must match. Actual Adam steps intentionally may
differ; final weight equality is not expected when the guard activates.

Long/short exposure, learned stop/near/zone-width, next-session execution, whole
shares,1× cap and20-session holding remain. Declining-stock cohorts are reporting
only; report positive total return AND positive short net P&L. No retired test,
orders or prospective claim. The old posthoc32.80% is one-seed development evidence.

## Gates and real update counts

Candidate selected AND final means must beat the fresh control by2pp; final
stock/ETF gains nonnegative, mean drawdown worsening≤2pp. Selected AND final
stock/ETF excess and median stock excess must each be≥−2pp against matched
holding. Actual final Adam steps must be positive and strictly fewer than the
control maximum to establish activation. Nonactivation is recorded; never extend
or adjust a threshold to force it. Failed gates close this fixed family, with no
extra steps or KL/learning-rate/epoch grids, confirmation or promotion.

Only a passing screen allows fixed11/23/37 paired confirmation under the
protocol's mean gains, absolute targets, paired-seed and drawdown requirements.
Frozen prospective evidence remains required. Reused development is not an
independent holdout; survivor/revised-history limitations persist.

Record per-parameter Adam steps separately from SB3's nominal epoch counter,
which advances by configured epochs even when KL stopping shortens the update.
Initial state is empty; trained states must be finite, cover all21 parameters and
restore with exact settings. Full control final100k expects20,000 Adam steps;
candidate allows fewer. Saved checkpoint/model/report identities remain mandatory.

## Preflight and budget

A synthetic high-KL unit test calls the installed guard branch with optimizer.step
forbidden. It verifies no weight/optimizer-state changes or actual learning, while the
nominal counter advances. This is forward-only control-flow acceptance, not local
fitting. Two512-transition Colab CPU smokes must then demonstrate actual finite
learning, saved target/Adam evidence, matched initial tensors/sampling, current
checkpoints and156 exact frozen action/state/repeat/reset predictions. Candidate
smoke nonactivation is allowed; full-screen activation is assessed separately.

Match Colab's actual default Python before dependencies/training. Use the existing
CLI and central aggregate one-minute meter, original cumulative ledger,7200-second
session deadlines and2-unit incremental study allowance including setup, idle,
retrieval, failures and retries. Project150/130 and account150/170 protections
remain. Retrieve/checkpoint durably, verify every owned session stop and publish
checksummed artifacts before removing duplicates. Never auto-restart after a
budget stop. Last project38.309006, account349.56 zero assignments/rate; new usage
check required before allocation.

Update-guard construction acceptance:27 focused checks pass, including identical
initial tensors, exact sole-setting difference, nominal/actual update rejection,
activation/nonactivation and a real library high-KL stop before any optimizer
step. No fitting or allocation; dedicated Colab integration remains.
