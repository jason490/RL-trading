# Fixed recurrent-capacity experiment

Status: numeric protocol, pinned untrained starts and construction contracts are
prepared. Dedicated Colab integration, actual environment/model acceptance and
two512 CPU smokes remain before a100k screen. No performance result or allocation.

## Hypothesis and boundaries

Prior recurrent studies kept64-unit recurrent/dense widths. The fitted-period
replay shows weak stock/ETF benchmark performance, but does not prove underfitting
or limited capacity: eligible history is not proof every bar was sampled, and
market periods differ. The session-context feature family failed its fixed gates
and remains closed. This study changes one capacity axis on the unchanged
seven-feature baseline, rather than extending that failed feature experiment.

| Arm | Actor/critic LSTM width | Dense heads | Trainable parameters |
| --- | ---: | --- | ---: |
| Control |64 each|64×64 each|61,513|
| Larger candidate |128 each|128×128 each|221,321|

Both have separate one-layer actor/critic LSTMs and Gaussian signed exposure,
stop, near-target and zone-width outputs. Observation size21 stays fixed: seven
market, five original portfolio fields, and nine masked execution fields. Native
policy, target_klNone, net-log-equity reward, costs, discount, clipping, epochs,
cyclic stock/ETF sampling and next-session simulator are unchanged. Shorting,
100% exposure cap, whole shares and20-session maximum holding remain required.
Both train adaptive exposure and select/evaluate/freeze sign-preserving unit
exposure under the existing declared contract. No size/threshold/protection grid.

## Fixed comparison

[Protocol](../configs/experiments/recurrent_capacity.json) SHA-256:
`ce6ea75e0a0dd286a9b4f8688f52aeb14c5ea3655084f33f52140c55f1801f93`.

Same immutable expanded111 training/original26 development dataset:
`cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b`.
Original seven-field training-only normalization is frozen. No new input/data
family or normalization refit. Train through2020, development2021–2023; retired
2024–September2026 data stays unopened. Survivorship/revised-data limitations
and repeated-development caveats remain.

Fresh seed11 runs each receive100,000 transitions, rollout1000/batch50/10 epochs,
10k evaluation cadence. Compare selected and final post-update weights. Candidate
must gain at least2pp selected AND final versus control, worsen neither final
stock nor ETF mean, and add at most2pp mean symbol drawdown. Selected AND final
stock/ETF excess and median stock excess must each reach−2pp versus matched
holding. Joint positive total return AND short P&L on future-defined declining
stocks is reporting-only, never an input, sampling rule or checkpoint selector.

Only passing permits paired11/23/37 confirmation with the original majority,
absolute/risk gates, followed by a frozen prospective comparison. A single-seed
win cannot establish capacity value or the full objective. Failure closes this
fixed capacity family: no width/depth/steps/reward/feature grid or extension.

## Initialization and interpretation

[Untrained manifest](../configs/initializations/recurrent_capacity_v1/manifest.json)
pins each arm/seed/phase. Control reuses verified untrained64-unit starts
(original21-observation tensor layout); no trained weights are reused. Three
larger128-width snapshots are generated once from fixed seeds with no data,
learning or performance selection. After generating larger weights, the pinned
control CPU RNG is restored before snapshotting, aligning future sampling RNG.

Smoke and screen use the same full architectures and the same per-seed tensors;
smokes differ only in128-step rollout,64 batch,128-session windows and512 total
transitions. Both smokes should record40 nominal PPO epochs/80 actual Adam steps.
This tests the actual larger architecture before research allocation. Different
widths have different tensor shapes and cannot satisfy cross-arm tensor equality.
Require exact pinned per-arm fingerprints and common CPU RNG, and document this
capacity/initialization distinction. Conditional multi-seed checks remain needed.

## Resources and runtime projection

CPU remains preferred; no GPU allocation is implied by a parameter increase.
One shared2-unit incremental allowance starts at cumulative38.514877408036206,
including all setup, idle time, transfer, failures/retries and shutdown. Original
project150/130 and account150/170 limits remain. Use existing central metering,
minute usage checks, owned IDs, durable checkpoints and verified cleanup.

The accepted prior64 control's measured train/evaluate duration is907.9985 seconds,
with its metadata hash pinned in the protocol. After actual full-architecture
smokes, require this conservative screen projection before allocation:

`2 × 907.9985 × max(1, larger_smoke_duration/control_smoke_duration) + 1200 ≤ 7200 seconds`

The1200-second reserve covers setup, artifact retrieval and shutdown. Smoke
wall-time ratios mix training and evaluation; this heuristic is not a runtime
guarantee. Actual per-allocation deadline and shared budget monitor still enforce
teardown. If the projection fails, retain smoke evidence/charges and do not
allocate or automatically restart on different hardware. Hardware changes need
a separate justified, budgeted profile. No initial allocation exceeds two hours.

## Verification before integration

65 focused capacity/session/guard checks pass. All fixed seed/phase fresh models
restore pinned untrained tensors after deliberate perturbation and reproduce
paired CPU random draws. Actual128-width save/load retains settings/provenance.
Wrong shape/guard/step/hash contracts reject. Runtime projection rejects invalid
or over-deadline inputs. Shared recurrent model validation is factored into
`training/recurrent_record.py`; existing session/guard regression behavior passes.
No local learn/optimizer steps or data/provider calls occurred in construction.

Next integrate the fixed plans/worker/runner/retrieval/report gates, run actual
environment checkers, then commit/freeze source and verify official usage before
two Colab smokes. Project38.51487741 and last official account349.41 unchanged;
all owned sessions are stopped.


## Version27 Colab integration and actual construction acceptance

Exact capacity plans bind widths, full-architecture initial snapshots, data and
account floor. Worker/runner restore pinned starts before fitting and retain
`capacity-initialization.npz`, initial/selected/final records and actual Adam
evidence. Study-specific retrieval checks live in `runtime/colab/capacity_artifacts.py`;
common lifecycle, checksum and checkpoint logic is reused.

Long plans require a valid runtime profile. Controller validation checks its
SHA-256 against actual smoke-acceptance bytes and recomputes the projection
before allocating. Profile arithmetic is validated remotely and retained in
the completion plan and final summary. Missing/stale/failed/over-deadline
approval rejects; no automatic hardware restart. Confirmation job preparation
also requires a passing same-protocol screen summary.

96 focused checks pass. Both actual dataset environments/checkers and untrained
recurrent forwards pass; two actual immutable smoke tasks validate. Source
compiles. [Construction evidence](../results/recurrent-capacity-2026-10-22/construction.json)
is model construction/inference only, not local fitting.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_capacity_jobs.py \
 --smoke --output .runtime/capacity-smoke-jobs.json
```

Commit/freeze source and verify official usage before running those tasks through
the existing metered fleet at concurrency2. Preserve the original study baseline
38.514877408036206 and2-unit allowance across attempts. After retrieval/frozen
verification and owned shutdown, run `accept_capacity_smokes.py` with output
`.runtime/capacity-smoke-acceptance.json`. The accepted file is retained unchanged;
its exact bytes bind later profiles. This is operational acceptance, not a
performance or model-selection gate.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_capacity_jobs.py \
 --output .runtime/capacity-screen-jobs.json
```

This writes the exact profile beside the task list as
`.runtime/capacity-screen-jobs.runtime-profile.json`, and refuses missing or
over-deadline approval. After a complete/cleaned screen, pass that same file to
`report_recurrent_capacity.py --runtime-profile`, together with state/output
paths. Strict source/accounting/baseline/optimizer checks and unchanged
selected/final gates still apply. No Colab allocation or new usage yet.
