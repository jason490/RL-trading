# Recurrent GAE trace comparison — implementation, not trained

[Fixed protocol](../../configs/experiments/gae_trace_preregistration.json) compares
native lambda0.95 control and one lambda1 candidate. All data/reward/gamma/actions/
architecture/sampling/steps/gates remain fixed. No new GAE training return exists;
shared-start and actual-rollout checking code is implemented below.

## Shared starts and model records

The loader uses original full-size capacity-control archives for seeds11/23/37,
including smoke seed11;61,513 parameters and21 inputs. It checks portable paths,
manifest/Torch/tensor identity, empty owned Adam, seed and actual settings before
loading tensors. Both arms restore the exact same original policy and CPU RNG;
NumPy/Python seeded states agree. Initial records differ only in `gae_lambda`.
No new initializer generation or dataset/provider access is required.

Actual model records include coupled-Adam semantics, optimizer and policy tensor
fingerprints, effective defaults, seed/RNG and shared-source proof. Learned
restoration checks are still pending; manually populated Adam fixtures establish
serialization only. SB3 loading does not promise global RNG restoration.

## Pre-update rollout evidence

`GaeRolloutAudit` reads a completed actual recurrent rollout buffer before any
optimizer update. It preserves raw collector rewards and independently computes
timeout critic values from the same new recurrent state. Stored rewards must
match the timeout adjustment. It checks episode-start/done/timeout masks, original
float32 fields, actual gamma/lambda and nominal pre-update epoch counts.

Separate float64 GAE reconstruction checks actual returns/advantages; independently
segmented discounted dot products check lambda1 returns. Full arrays are retained
as small compressed NPZ files under `gae-rollouts/`, with hashes and compact
`gae-rollouts.jsonl` records. Tolerances are fixed in implementation before any
actual run: absolute2e−6 and relative2e−5. Buffer continuation across rollout
boundaries is allowed; true terminal and timeout boundaries stay distinct.
Callback checks do not modify model/Adam, recurrent caller state or sampling RNG.

## Verification and remaining work

61 focused shared-start, original initialization, parameter, actual-buffer-fixture,
serialization and VIX regression checks pass. Wrong lambda/defaults reject before
tensor mutation; malformed math/masks/dtypes and timeout corrections reject.
Fixtures forbid `.learn` and `Adam.step`; no local optimizer step or policy
training occurred. New modules/tests pass Ruff. Original initializers, dataset,
ledger and `plan.html` remain unchanged.

Strict GAE request dispatch, shared start, rollout callback and
initial/selected/final records are now attached to the runner. Version35 jobs/fleet budget/archive
acceptance and physical learned/frozen verification must precede actual512-step
CPU Colab smokes. Reinspect default kernel Python before dependency sync, preserve
original150/130 project and150/170 account limits, persistent two-unit study
baseline, aggregate minute meter, inclusive deadlines and owned cleanup.

No new session, units, data requests, retired-test access, broker orders or
performance improvement is claimed. Prior failed families remain closed.

## Runner integration accepted; actual training still pending

Strict registered GAE requests reject changes to data/preprocessing, algorithm,
seed/phase, parameters, steps/cadence, CPU device, observation/reward/exposure/
sampling and checkpoint settings before model/environment creation. The runner
restores the shared start, snapshots initializer bytes and records actual
initial/selected/final model/Adam evidence. The per-rollout checker is attached
beside validation; final metadata requires its complete summary. Legacy study
callback/dispatch behavior is preserved.

Malformed target arrays are retained with a hash and pre-update failure marker
before raising, enabling diagnosis without repeating training.78 focused request,
initialization, actual-buffer, failure-preservation, serializer and prior shared/
VIX regressions pass; new code/tests pass Ruff. All fixtures remain non-optimizing.
Version35 jobs/fleet/budget/archive and physical learned smoke acceptance remain
next. No actual GAE policy training, new return or Colab session is claimed.
