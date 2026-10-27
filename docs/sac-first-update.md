# SAC first-update numerical diagnostic

## Status and scope

Completed on2026-10-27 in one owned Colab CPU session, clean source7eb2588,
actual default Python3.13.16. Both native SAC conditions produced equal numerical
first-update results. Physical restoration and owned shutdown pass.
[Results and retained evidence](../results/sac-first-update-2026-10-27/README.md).

This diagnosis addresses a possible numerical source of the stopped CrossQ
comparison's historical-SAC mismatch. It does not train CrossQ, evaluate trading
returns, select checkpoints, change the old tolerance, prove the old host's cause
or clear its persistent integrity stop. The allocation claim is consumed; no
automatic retry is allowed.

## Registered comparison

[Fixed protocol](../configs/experiments/sac_first_update_diagnostic.json) compares
original SAC defaults with explicitly serialized defaults on the same CPU.
Seed11,21 observations, four actions, full64×64 networks, batch256, original
unfitted tensors and original Torch/NumPy/Python streams are fixed. Effective
numerical settings agree while native tuple/list and int/float representations
remain visible.

Exactly one native gradient update per condition is allowed: at most two model
updates/six Adam steps total. No environment rollout is collected. The1001-row
training-only replay prefix is fixed. One stochastic actor call at original
row1000 precedes the update; the follow-up observation is original row1001.
No hyperparameter/precision grid or return-based selection is performed.

## Actual findings and limits

| Measurement | Original versus explicit defaults on Colab CPU |
| --- | --- |
| Initial tensors and RNG | Exact |
| Pre-update action/RNG and sampled replay batch | Exact |
| Post-update policy and target tensors | Exact |
| Raw entropy parameter and Adam moment tensors | Exact |
| Post-update action and RNG | Exact |
| Native gradient updates |1 each |
| Completed actor/critic/entropy steps |1 each per condition |

No differing numerical tensor field exists. This rules out a representation
effect for this input/backend only; it is not a universal equivalence proof.
Container metadata is intentionally different and is not part of the numerical
tensor-equality claim.

Local AVX512 replay of the saved AVX2 Colab model has a2-ULP stop-action difference
(5.960464477539063e−8) with identical saved tensors/input/RNG. Inter-op threads,
affinity and environment also differ. No individual backend cause, or cause of
the earlier historical four-ULP near-target divergence, is established. Exact
saved-state restoration and cross-host arithmetic equality are separate facts.
No numerical result automatically authorizes a new performance screen.

## Immutable input

`training/sac_update_input.py` validates protocol/source identities, original
full64 initializer, fixed parameter representations, six finite float32 replay
fields, action bounds, timeout masks and shared follow-up observation.
`scripts/build_sac_update_inputs.py` reads only pinned task-owned original
metadata/replay and the pinned diagnostic checkpoint. Every copied field must
agree bitwise across both retained training replays. Sources are rehashed
unchanged; raw normalized replay input stays in ignored local data storage.

[Input proof](../results/sac-first-update-2026-10-27/inputs.json) binds identity
7f50968c0272bda3d5b5a75861e5ea25d2f4e7d328a1b014af7de219a0d5a8a7.
The builder writes immutable outputs and does not overwrite proof paths.
[Earlier preparation evidence](../results/sac-first-update-2026-10-27/preparation.json)
remains versioned to its original worker hash; it is untrained preparation, not
actual update acceptance.

## Worker and update guards

`training/sac_update_worker.py` copies nested configuration because native SB3
mutates policy kwargs during construction. Shared input immutability is tested
by preparing both conditions from one object. Preparation loads pinned tensors
and streams, fills1001 replay rows and sets original first-update progress
without environment reset/rollout. Both conditions use the historical source's
explicit single intra-op thread setting.

Native updates require default-kernel Python3.13, Colab release context,
`/content/rl-stocks` and CPU hardware, checked before construction/output mutation.
The observer permits one replay sample and one optimizer attempt per actor/
critic/entropy optimizer. Pre/post hooks distinguish attempted and completed
steps. Native one-step replay requires `discounts=None`; only five finite
float32 sample tensors enter NPZ files. Successful output requires actual saved
per-parameter Adam counters1 and native update counter1. Failure paths retain
observer evidence and remove handles, sample overrides and environments.

Native checkpoints are exported between conditions. Each condition retains
`post-update.zip`, `record.json`, `numerical-state.npz`, `sampled-batch.npz` and
`torch-rng.npz`. Four raw Torch states cover initial, pre-update, post-update and
post-action moments; source inputs and summary accompany both conditions.

## Independent physical verification

`training/sac_update_verify.py` reconstructs sampling and two actor forwards
under `no_grad`, without backward or optimization. It loads the native archive
and compares every policy/target/raw-entropy/Adam tensor with NPZ evidence.
RNG snapshots must match recorded hashes and valid native generator geometry.
Action records require finite four-component float32 hashes. Saved stochastic
RNG replay must repeat exactly locally without changing numerical state.
Remote-action recomputation equality is reported separately, with no undeclared
return tolerance or fitting. Caller RNG and source-file hashes are preserved.

The directory verifier recomputes the reported comparison from physical files.
Both valid equal/different outcomes are acceptable; promotion/root-cause claims,
extra transitions, altered counts or CrossQ-stop clearing are rejected. The
version38 archive allowlist is bounded to64MiB and undergoes this verification.
Manual serialization fixtures never call learning/optimizer steps; actual
Colab evidence is recorded separately.

## Budget, metering and cleanup

One dedicated CPU allocation, persistent0.1-unit baseline and immutable
`sac-update-allocation.json` claim are enforced. The900-second inclusive deadline
reserves420 seconds for bounded retrieval/cleanup; other studies retain their
existing concurrency/deadline defaults. Unstarted-worker execution retries are
disabled for this diagnostic. Original150/130 project and150/170 account guards
and the CrossQ integrity marker remain intact.

Measured cost is0.01 units versus integrated0.003961784. The final delayed idle
reading was settled using `Ledger.settle_stopped` for the exact latest stopped
attempt, preserving allocated estimates/timestamps, baselines and stop flags.
Only measured charge increases apply: no unallocated idle-time charge, reduction
after account credits or double-count of repeated readings.

Final project39.52854764966505/150, last official348.59/rate0/no assignments.
Verified duplicate-upload cleanup removed8.08MiB with702 retained files unchanged.
Final model/state/batch/RNG/input/transport evidence remains. See tracked
[cleanup](../results/sac-first-update-2026-10-27/cleanup.json) and
[settlement](../results/sac-first-update-2026-10-27/settlement.json).

## Verification commands

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python -m pytest -q tests/test_sac_update*.py \
  tests/test_budget_settlement.py tests/test_fleet_budget.py
```

For retained actual output, call `verify_directory` with the recorded artifact
root's `sac-update` folder and its `input.json`. This is read-only verification.
The historical builder/controller commands remain in Git history; trying a new
state filename cannot bypass the sole-allocation claim.
