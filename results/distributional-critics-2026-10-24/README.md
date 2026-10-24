# Distributional critic study — initialization acceptance

**No SAC/TQC study model has been trained yet.** The registered three-arm method
comparison is being integrated before any Colab allocation or performance score.
The [prior saved-SAC diagnostic](../sac-soft-consistency-2026-10-24/README.md) supplies
a limited critic-consistency rationale, not a TQC return claim.

[Protocol](../../configs/experiments/distributional_critics.json) fixes native SAC,
untruncated TQC and drop2 TQC, original26-symbol source,50k seed11 screens and the
existing relative/absolute selected/final stock/ETF gates. Only a passing screen
can proceed to11/23/37 confirmation and later prospective evidence. The2-unit
study allowance remains inside the original shared project/account limits.

## Accepted untrained tensors and global sampling state

[Evidence](initialization.json) verifies12 checksummed tensor/RNG archives: three
arms for smoke seed11 at32x32 width and full-size seeds11/23/37 at64x64 width.
Native standard SAC tensors, optimizer configuration and post-initialization RNG
are unchanged by the explicit study defaults. Initial actor tensors match across
all arms; TQC0/TQC2 have identical complete policy/critic/target tensors and empty
optimizer states. Both TQC heads remain2×25, with50 or46 retained targets.

| Full-size model | Policy parameters, including target critics |
| --- | ---: |
| Native SAC | 29,644 |
| Either TQC arm | 35,884 |

These counts include target networks and exclude the separate scalar learned
entropy coefficient. They are architecture counts, not performance evidence.

Each arm restores the pinned native SAC **global Torch, NumPy and Python** RNG
states. This includes NumPy replay-sampling state; environment/action-space seeds
must still be verified by the upcoming actual Colab smokes. The initializer
restores complete untrained policy tensors into existing Parameter objects,
preserving optimizer ownership. Fresh-model checks cover actor/critic/entropy
optimizers, replay emptiness, counters, seed, effective parameters/step-frequency
unit, scalar/quantile critic identity and native Adam semantics. Invalid artifacts
or global RNG fail before mutation; repeated application rejects.

61 focused untrained TQC/shared-initialization/regression tests pass, including
perturbed-state restoration, all three RNG draw streams, wrong seed/settings,
nonempty optimizers/replay, malformed global state and actual save/load proof
retention. No optimizer step, learning call or market-data access was used to
construct the assets. New modules/tests pass Ruff; shared initializer introduces
no additional legacy lint findings. Learned optimizer/replay acceptance is still
pending and is not established by these untrained checks.

## Retention and next work

Canonical assets/manifest are in`artifacts/distributional-initialization-v3/`;
[tracked binding](../../configs/experiments/distributional_initialization.json)
pins the manifest hash, and the loader pins that binding. Earlier unbound v1/v2
builds are retained for traceability; all12 tensor/RNG archives are byte-identical
to v3. Differences concern validation/provenance metadata, not chosen weights.
Generation package versions and exact implementation hashes are retained; its
then-dirty source state is recorded explicitly. Current generation files match
those hashes. No temporary initializer files or live builder processes remain.

Next integrate the fixed study with training metadata, task/payload/archive checks,
actual model/optimizer restoration and the existing metered Colab controller.
Inspect the actual default Colab Python before dependency sync; then run three
1024-transition CPU smokes. No screen starts before real tensor/RNG/frozen parity,
artifact and deadline-projection checks pass. Preserve retired-test isolation and
all fixed failure gates. No trained model, new trading returns, normalization fit,
broker orders or Colab units occurred in this initialization milestone.


## Metered study integration ready for Colab smoke validation

Version33 jobs now connect the fixed SAC/TQC study to the training runner,
immutable payload checks, archive validation and aggregate Colab fleet. The12
initializer archives are packaged under `configs/initializations/distributional_v1/`
so uploads include them; their tensor/RNG bytes are unchanged. The original
artifact-generation evidence remains historical provenance.

Training records bind actual initial/final/selected tensors, raw learned entropy,
Adam moment shapes/steps and shared initialization. Owned-artifact restoration
checks actual selected/final/frozen models, replay contents and64 exact frozen
predictions per arm without learning or changing normalization. Local nonempty
optimizer fixtures test serialization only; they do not establish learned smoke
acceptance. Archive validation enforces the original26-symbol development panel,
execution contract, dataset and pinned normalization.

The controller persists one shared two-unit study baseline across smoke/screen
controllers and retries. It also retains the original project150/130 and account
150/170 limits. Full runs require accepted actual smoke artifacts and a conservative
CPU projection within the inclusive7200-second allocation deadline. The projection
is a heuristic with reserve, not a runtime guarantee.

115 focused algorithm, initialization, learned-record serialization, job, archive,
fleet and budget tests pass without local optimization. Actual Colab smoke
validation is next; no new SAC/TQC return or completed screen is claimed.

Prepare the three CPU smoke jobs without allocating:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_critic_jobs.py \
  --smoke --output .runtime/critic-smoke-jobs.json
```

Run from a clean immutable source export through the existing fleet, sharing the
original `.runtime` directory/ledger and immutable dataset. Inspect actual Colab
kernel Python before dependency synchronization; meter aggregate usage each minute,
retain artifacts and verify every owned session stops. After the fleet completes,
`scripts/accept_critic_smokes.py` requires its state, source proof and dataset;
only passing actual restoration/development-report/runtime checks authorize
building full jobs. No failed-family reopening or retired-test access is allowed.
