# Recurrent training and execution sizing alignment

## Why this is a separate hypothesis

[The fixed sizing replay](../results/recurrent-unit-sizing-2026-10-21/README.md)
improves final mean return from24.04% to32.80%, but fails selected-checkpoint and
risk gates. It cannot be promoted or confirmed. Its ordering reverses: adaptive
selected beats final, whereas unit-sized final beats selected. Training and
checkpoint selection used a different execution rule from deployed unit sizing.
This motivates testing alignment; it does not prove a cause or justify a sizing
grid. This study changes the learning execution contract, with fresh controls.

## Frozen comparison

[Protocol](../configs/experiments/recurrent_alignment.json) fixes three fresh
RecurrentPPO arms:

| Arm | Training exposure | Validation, selection and frozen inference |
| --- | --- | --- |
| adaptive | Raw bounded magnitude | Raw bounded magnitude |
| posthoc_unit | Raw bounded magnitude | Sign-preserving unit magnitude |
| aligned_unit candidate | Sign-preserving unit magnitude | Sign-preserving unit magnitude |

All three use identical initial actor/critic tensors, separate one-layer64-unit
LSTMs,64×64 heads,21 masked observations, expanded111 training symbols, original26
development episodes and immutable original normalization. Same seeded stock/ETF
cycles and512-step windows, PPO rollout1000/batch50/ten epochs,100k transitions,
learning rate0.0003, gamma0.99, net-log-equity reward and costs. No reward, feature,
architecture, sampling, holding, numerical or entropy changes. Evaluate every10k.

Adaptive and posthoc training dynamics are identical: deterministic validation
changes reporting/selection, never learning-rate schedules or optimizer updates.
Their final policy tensor hashes must match; a mismatch invalidates the intended
control rather than supporting a causal alignment claim. All actual sampling
counts must match. Candidate trading paths can differ because training executes
unit-sized actions. Initial fingerprints remain identical in all arms.

## Action execution contract

The policy emits four continuous raw components. Unit execution changes only the
first to−1,0,+1 according to sign, before existing risk approval. Stop distance,
near-target distance and zone width remain learned and unchanged. Whole shares,
1× cap, next-session fills, costs, borrowing, protections, corporate actions and
20-session maximum holding remain. Actual exposure need not reach the request.
Unit mode cannot voluntarily request cash except at an exact zero exposure;
that limitation and potentially larger losses are part of this fixed hypothesis,
not addressed by adding an unregistered deadzone or magnitude parameter.

The training environment maps the action before simulation and computes reward
from actual resulting equity. It records the original `policy_action` separately
from executed `raw_action`. Subsequent account observations and recurrent memory
follow that real path. Recurrent state resets at each episode boundary.

Version1 `action_execution` metadata records distinct training/evaluation modes.
Validation callback, best-checkpoint selection, final evaluation and frozen
prediction use the declared evaluation mode. Frozen bundles validate this exact
contract; historical bundles without it retain unrestricted adaptive sizing.
Frozen actions include the transform, so subsequent risk decoding sees the same
execution request used in evaluation. Hidden/cell memory remains unchanged by
the output transform. Shared implementation is in [risk/exposure.py](../src/rl_stocks/risk/exposure.py).

## Gates and operations

Three512-transition CPU smokes precede any three100k seed11 screen. Actual modes,
processed action statistics, equal initialization, adaptive/posthoc final weights,
matched sampling, saved steps/fingerprints, frozen executed-action/state/reset
parity, immutable normalization and checkpoint identities must pass first.
Smokes use fixed32-unit heads/LSTMs,128 rollout and64 batch; returns never tune
settings. Check Colab's actual default Python before syncing locked dependencies.

Candidate selected AND final gains must each exceed BOTH controls by2pp; final
stock/ETF gains nonnegative; final mean drawdown worsening no more than2pp.
Selected AND final stock/ETF excess and median stock excess must each be≥−2pp
against matched holding. Failure closes this fixed family: no extension, sizing
threshold, sequence/architecture sweep, confirmation or promotion. Only a passing
screen permits fixed11/23/37 confirmation under the original paired/absolute
requirements; prospective frozen evidence remains necessary afterward.

Full-period declining stocks are reporting-only; report joint positive total
return AND positive short P&L. No future labels affect observations, sampling,
reward or selection. Reused2021–2023 development is not independent evidence;
retired2024–September2026 stays unopened. Survivor-data limitations remain.
No broker orders or live-data execution are enabled.

Use Colab CPU, up to three concurrent sessions, original centralized ledger,
one-minute aggregate metering and7200-second allocation deadlines. Study limit3
incremental units includes setup, idle, downloads, failures and retries. Project
150/130 ceiling/teardown and account150/170 floor/teardown both remain active.
Never restart after budget stop without reviewed authorization; retrieve/checkpoint
durably, stop/verify every owned session, and retain source/model/failure evidence.

## Current implementation and verification

Protocol and checksummed construction contracts are prepared. Shared transforms,
training environment execution, validation/checkpoint selection routing, frozen
execution metadata and sequence verifier support are implemented.43 focused
checks pass without fitting, including unit-environment Gymnasium/SB3 checkers,
actual equal recurrent initialization, untrained frozen action/state parity,
protective-output preservation, manual execution/reward equivalence, default
bundle compatibility and exact contract rejection. Script compilation passes.

Dedicated job/worker/archive/restored-report integration and actual processed-
action counters are now implemented. Actual Colab smoke acceptance remains required
before the fixed screen. No training, allocation
or smoke acceptance has occurred for this study. Project38.014408/150 is retained;
no owned sessions are active. Existing failed replay/state evidence is unchanged.

## Integrated Colab acceptance

Version23 plans exactly bind training/evaluation modes, dataset, source protocol
and account150/170 contract. Controller and worker validate immutable expanded
data before fitting. Training counters record transitions, nonzero requests,
processed non-unit sizes, changed sizing and protective-component changes. Gym/SB3
checker activity is reset before actual training counts begin. Unit counters must
show transformed non-unit requests and no non-unit processed exposures; protective
changes must stay zero. Adaptive counts must show unchanged raw sizing.

Every checkpoint preserves `training-execution.json`; final metadata and frozen
manifest must match it. Unit selected/final validation traces must preserve all
three protective components and exactly transform the original policy request.
Archives also require saved-model fingerprints, fixed preprocessing/sampling and
current checkpoint/report identities. Model releases retain execution/sampling
counters alongside weights. Legacy default paths remain unchanged.

79 focused checks pass, including malformed modes, corrupted counters, changed
protective outputs, checkpoint identity, budget/floor and owned fleet cleanup.
Actual three immutable smoke payloads validate and scripts compile. No local
fitting was performed. The builder does not allocate:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_alignment_jobs.py \
 --smoke --output .runtime/alignment-smoke-jobs.json
```

Freeze clean pushed source and check official usage before three512 CPU smokes.
After retrieval and verified shutdown, `accept_alignment_smokes.py` requires234
exact frozen predictions, actual mode/counter/step restoration, equal recurrent
initialization/sampling and adaptive/posthoc final tensor equality. Only then
start the fixed three100k screen. `report_alignment.py` independently restores
selected/final weights, validates action traces and matched source/accounting/
holding panels, requires adaptive/posthoc final equality, then applies original
gates. No training or provisioning has occurred at this integration milestone.

## Actual alignment Colab smoke acceptance

Three512 CPU smokes accepted on Python3.13.16, clean workercaddfcb.234 exact
frozen mapped-action/hidden/cell/repeat/reset predictions, immutable normalization
and input rejection pass. Actual counters:adaptive/posthoc512 transitions each,
340 non-unit raw sizes and0 changes; aligned343 changes,0 processed non-unit sizes,
0 protective changes. All initial tensors/sampling match; adaptive/posthoc final
policy tensors match exactly. Actual selected/final records/checkpoint identities
and unit traces pass. All3 sessions stopped; project38.047855/150, account349.78
zero rate/assignments. Fixed same-source three100k screen next within3-unit study
allowance; no tuning from smoke returns. [Evidence](../results/recurrent-alignment-2026-10-21/README.md).
