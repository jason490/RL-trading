# Recurrent policy-update safeguard

Status: version2 actual smokes accepted with exact shared initialization. Both
owned retry sessions stopped; version1 rejection retained. Fixed100k screen next.

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

## Integrated preflight

Version25 binds exact targets, dataset/normalization, sampling and account floor
on controller and worker. Initial/selected/final records preserve actual settings,
finite Adam ownership/moments and per-parameter steps; archived/frozen metadata
must agree. Checkpoint identities and unit exposure/protection traces remain
mandatory. Restored reporting requires return/risk gates AND actual final-step
reduction; nonactivation cannot promote even if performance metrics pass.

75 focused checks pass, including that negative activation case, malformed saved
settings/counts, missing checkpoints, legacy cash artifacts and fleet budget/
cleanup regressions. Two actual immutable smoke payloads validate; scripts
compile. No allocation at this integration milestone.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_guard_jobs.py \
 --smoke --output .runtime/guard-smoke-jobs.json
```

Commit/freeze source, check official usage, then two512 CPU smokes. After durable
retrieval and verified shutdown, `accept_guard_smokes.py` restores saved settings/
Adam states and requires156 exact frozen predictions, matching initial networks
and sampling. Only then run the same-source two100k screen.

## Initial guard smoke comparison rejected

Both512 CPU runs completed/retrieved,156 exact frozen predictions pass individually
and both owned sessions stopped. Combined acceptance rejects different initial
tensor hashes despite matching seed/source/settings/version records. Local
untrained pair hashes agree but differ from both remote hashes; cause/magnitude
not proven, initial arrays unavailable in this attempt. Keep exact equality gate.
Next checksum-pin shared untrained tensor/RNG snapshots per seed, with explicit
revised protocol before repeat smokes; no learning/performance initialization
selection or tolerance relaxation. No100k screen allocation. Both had80 Adam
steps/40 nominal epochs; short-smoke nonactivation was allowed and is separate.
Project38.32607251, original study baseline38.309006 retained, no budget stop.
[Failure evidence](../results/recurrent-guard-2026-10-22/README.md).

## Version2: exact shared initialization

The [version1 failure](../results/recurrent-guard-2026-10-22/README.md) retained
matching seeds/source/versions but different initial fingerprints. Cause and
magnitude are unproven; do not describe a confirmed hardware bug. Version2
realizes the same-initial-tensors requirement with fixed untrained inputs instead
of relying on independent runtime reconstruction. Original version1 protocol,
plans, models and charges remain unchanged; no gate/tolerance is relaxed.

[Artifacts and manifest](../configs/initializations/recurrent_guard_v2/README.md)
contain policy tensors plus CPU RNG for smoke11 and screen/confirmation11/23/37.
They were generated once by the standard seeded constructor without fitting,
datasets or performance selection. Protocol2 pins the manifest, which pins each
file, tensor schema/fingerprint and RNG checksum. These are initialization
inputs, not learned model weights. Both arms load the same seed/size input.

Loading is CPU-only, non-pickle and rejects trained/nonempty-optimizer models,
wrong versions, keys, shapes, dtypes, nonfinite data or invalid RNG. Validation
precedes mutation. Tensor copies preserve parameter identities/optimizer
ownership; exact tensor and RNG restoration is checked. Proof survives actual
model save/load. Initial RNG is a learning-start state, not exact resume support.
Backend numerical differences during training remain possible.

Each run retains `guard-initialization.npz`, copied and checksummed before fitting;
metadata, restored models, plan contract and archive must agree. Model publication
retains this input alongside weights. Controller/worker check assets before fit.
Full initial equality is still mandatory after actual Colab loading.

94 focused checks pass, including restoration from deliberately different weights
and RNG states, all four fixed seed/size inputs, corruption rejection without
model/RNG mutation, saved provenance, archive snapshots, actual Adam steps,
activation and existing budget/cleanup behavior. Two revised actual immutable
payloads validate; scripts compile. No allocation during the fix.

Use new retry paths, preserving version1 state:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_guard_jobs.py \
 --smoke --output .runtime/guard-smoke-v2-jobs.json
```

Commit/freeze version2 source, official usage check, then repeat two512 CPU
smokes with156 frozen predictions and verified shutdown before screen. Shared
study baseline remains38.309006, allowance2 units including the rejected attempt.
Current project38.326073; last official account349.56 zero assignments/rate.

## Version2 actual smokes accepted

Two512 CPU runs from clean workered9d834, Python3.13.16 accepted after durable
retrieval and verified shutdown.156 exact frozen action/hidden/cell/repeat/reset
predictions, immutable normalization/input rejection, restored selected/final
settings/Adam evidence and current checkpoint identities pass. Both load the
same pinned tensors and CPU RNG, with exact initial fingerprint equality and
matching actual sampling. Input snapshots are retained in both runs.

Both smokes record40 nominal epochs and80 actual Adam steps. The short guard
does not activate early stopping, explicitly allowed; actual reduction is still
required at the fixed100k screen. This is wiring acceptance, not performance
evidence or a threshold-selection exercise. Original version1 rejection remains.

All four owned smoke sessions stopped. Project38.34607251/150, remaining
111.65392749; study conservative increment0.03706648 includes rejected attempt.
Official account349.54, zero assignments/rate. Original study baseline38.309006
and2-unit allowance, project150/130 and account150/170 remain. Same-source two100k
seed11 screen next; no tuning from smoke returns.

[Evidence](../results/recurrent-guard-2026-10-22/README.md).


## Completed update-guard screen

Fixed seed11 control selected/final19.62/18.48%, guard28.07/24.44%, holding30.60%.
Guard reduces actual Adam steps20,000→6,165 and passes relative/risk gates, but
both absolute target gates fail. Final stocks27.78% versus32.63% holding;
ETFs6.11% versus19.47%; joint short/total success0/3. No confirmation, extension,
KL sweep or promotion. All six study sessions stopped,156 screen frozen checks
pass, original ledger retained. Project38.39201565/150; official account349.51,
zero rate/assignments. Publication/duplicate cleanup and saved-path diagnosis next.
[Full evidence](../results/recurrent-guard-2026-10-22/README.md).


Update-guard publication accepted:15 server assets match SHA-256/size, four
accepted bundles and interactive curves retained publicly. Only two verified
duplicate archives removed (32,215,676 bytes);167 retained hashes unchanged.
Strict summary reproduces byte for byte after cleanup. All six sessions stopped;
project38.39201565 unchanged. Saved-path diagnosis next, no guard promotion.
[Evidence](../results/recurrent-guard-2026-10-22/README.md).


## Saved update-guard action audit

Four accepted phases/104 episodes audited against exact accepted hashes/panel;
no fit/model/data/provider/Colab access and source hashes unchanged. Guard final
stock/ETF long requests99.51/98.91% versus control93.60/95.77%; cost drag falls
6.07→2.85pp stocks,5.13→2.53pp ETFs. Relative gains combine lower costs and
residual path improvements. Final ETF shortfall13.35pp still includes10.83pp
residual plus2.53pp costs; selected-to-final deterioration occurs despite lower
costs. No causal explanation established.15 focused tests pass. Guard stays
closed; next fixed exit/gap participation audit before another intervention.
Project38.39201565 unchanged, all sessions stopped.
[Evidence](../results/guard-actions-2026-10-22/README.md).
