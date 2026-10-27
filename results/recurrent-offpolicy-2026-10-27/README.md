# Recurrent off-policy feasibility and learned smoke

Status: actual recurrent/native SAC learned-state smoke passed on2026-10-27.
No trading-return experiment or model promotion follows from this result.

## Research and offline acceptance

- [Replay audit](replay-index-audit.json): tagged skrl2.1.1 physical-slot sequence
  counterexample; source/arithmetic review, not installed-library execution.
- [Sampling benchmark](replay-benchmark.json): actual local100k-capacity metadata
  benchmark;93,800 valid starts match reference.256-window median1.873ms excludes
  payload/model/Colab work. [Earlier formatting-only trial](replay-benchmark-before-format.json)
  is retained separately; no selection of trading results.
- [Loss-source audit](loss-source-audit.json): installed SB3 2.7.0 SAC equations
  and squashed-density source hashes.
- [Actual untrained preparation](smoke-preparation.json): exact retained training
  replay, matched batches and bound initializer; no gradient updates at that stage.

## Actual Colab acceptance

[Fixed protocol](../../configs/experiments/recurrent_sac_smoke.json) ran once from
clean sourcefc573595d065e6433767f6a907e704cdd51b3611, package source hash
fbbba6a731f9577e0fc9408b8bbed37faca7f4c8475cfb0be2557d11e8cf2d55.
Actual default Python3.13.16, Torch2.8.0+cu128, CPU AVX2, one intra/inter-op thread,
CUDA unavailable. Sessionrl-stocks-cpu-349d406bc473; fleet87f0178db27e.

[Summary](smoke-summary.json) confirms four gradient updates per recurrent/native
arm, eight total/24 Adam steps, on the exact same128 learning rows per update.
Recurrent windows add16 burn-in rows. No environment transitions were collected;
no data download, validation/test evaluation, checkpoint selection or grid occurred.
Initializers differ by architecture; this is integration acceptance, not an
isolated test of whether memory improves returns.

[Observed updates](smoke-observed-updates.json) records finite native gradients,
nonzero recurrent-memory gradients, entropy→critic→actor order and four exact
Polyak checks. Every actual Adam parameter counter restores to4. Both actor/critic
and raw entropy state changed. Initial/final recurrent and native checkpoints,
four target-equation NPZ snapshots, exact loss-row IDs and prediction records
remain in the retrieved scientific artifact folder.

[Independent physical restoration](smoke-restoration.json) reopens the retained
files without fitting, verifies initial/final tensors/Adam state and all four
saved target equations, and checks local frozen carried/reset prediction parity.
Remote recurrent action recomputation on local AVX512 is not bitwise exact;
this is recorded separately from exact state restoration and within-runtime
learned save/load parity. No cross-host equality guarantee or tolerance relaxation
is inferred. The historical CrossQ stop remains unchanged and blocked.

## Compute, transport and cleanup

[Compute evidence](smoke-compute.json):149.784s inclusive allocation, below900s.
Official rate0.08units/hour while allocated. Integrated charge0.00377519063 units;
rounded account readings report0.00 debit so far, not proof of zero cost. The
ledger conservatively retains the estimate:39.53232284029461/150 used,
110.46767715970539 remaining; last official account348.59/rate0/no assignments.
Late idle reading was settled under the original allocation lock without charging
post-shutdown time or resetting baselines. Future delayed debits must still be
settled against this same stopped attempt before a new allocation.

CLI observation timed out while the existing worker continued; completed artifacts
were retrieved without restarting it. The sole allocation claim remains consumed.
Owned session termination is verified. No session is left running.

[Cleanup inventory](smoke-staging-cleanup.json) removed only the8,531,570-byte
(8.14MiB) owned upload duplicate after every member matched retained copies.
1,851 protected files rehashed unchanged during cleanup. Source export, final
transport, all models/Adam states/replay input/target chains/predictions remain.

Retained artifact root:
`.runtime/recurrent-sac-smoke-source-v1/artifacts/colab/rl-stocks-cpu-349d406bc473/retrieved/artifacts/recurrent-smoke`.
Controller state:`.runtime/recurrent-sac-smoke-state-v1.json`.
Source proof:`.runtime/recurrent-sac-smoke-source-proof-v1.json`.

## Remaining work

Training-only collection and common development evaluation adapters now pass
synthetic/untrained acceptance; see the category guide. Checkpoint/evaluation
curves, preprocessing/frozen bundles, a bounded training driver and registered
actual environment-training smoke remain before a fresh performance run. Exact
interruption continuation is not supported by the current learner snapshot. Register a bounded matched development comparison with
contemporaneous controls, fixed final/multi-seed gates and prospective confirmation.
Retired2024+ data remains unopened. No robust winner has been established.
[Category guide](../../docs/recurrent-offpolicy.md).


## Fresh-environment smoke preparation — not allocated

The separate fixed128-transition/four-update protocol now has an
[accepted cached view](environment-view-preparation.json) and
[actual untrained environment/tensor checks](environment-untrained-acceptance.json).
Six symbols,12 episodes, original preprocessing; no providers or retired data.
[Earlier checker evidence](environment-checker-initial.json) is retained separately.
52 relevant checks pass; guarded worker exists. Fleet budget/claim/routing and
physical curve/report/checkpoint archive acceptance remain before allocation.
The completed replay-only smoke above is not rerun or reinterpreted as returns.
