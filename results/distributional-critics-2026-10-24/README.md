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
