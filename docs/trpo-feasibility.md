# TRPO engineering feasibility

This is a distinct optimizer-feasibility study. It does not reopen failed PPO
safeguards, SAC experiments, CrossQ's integrity stop or forecast families. No
performance comparison or Colab allocation is admitted yet.

## Motivation and limits

Saved RL endpoints repeatedly deteriorate despite valid artifacts, sometimes
without higher costs; supervised residual forecasts also fail their fixed gates.
These observations establish neither a sole optimizer cause nor an information
edge. They justify a limited engineering question: can the maintained TRPO
implementation provide a reproducible, affordable alternative update mechanism
before another matched trading comparison is specified?

TRPO uses conjugate-gradient curvature calculations and KL-constrained backtracking
line search; PPO clipping and an approximate-KL early-stop check use different
update procedures. The [original TRPO paper](https://arxiv.org/abs/1502.05477)
distinguishes theoretical guarantees from the practical approximations and tests
robotics/Atari tasks. It provides no stock-trading guarantee. The
[installed-version API documentation](https://sb3-contrib.readthedocs.io/en/v2.7.0/modules/trpo.html)
supports continuous Box actions, CPU, save/load and deterministic prediction,
with no recurrent policy. Project inference: a different update method is worth
bounded feasibility work; it cannot create missing predictive information.

## Fixed contract

[Engineering protocol](../configs/experiments/trpo_feasibility.json) SHA256:
`ba19cae92e039c0f6d86e01d8309203899835596daefa6abc1ea4245b706c7ab`.

| Contract | Fixed value |
| --- | --- |
| Implementations | Existing SB3PPO2.7.0 / SB3-ContribTRPO2.7.0 |
| Input/output | execution_full21 fields / four float32 actions in−1…1 |
| Actor/value | Separate64×64 tanh MLPs, orthogonal initialization, log_std0 |
| Shared rollout |1000 transitions, critic batch50, gamma0.99, GAE0.95 |
| Shared value learning rate |0.0003 |
| PPO |10 epochs, clip0.2, entropy coefficient0 |
| TRPO |CG15, damping0.1, line-search shrink0.8, maximum10 trials |
| TRPO constraint/value |KL target0.01,10 critic passes, no subsampling |
| Seeds |11/23/37 |
| Smoke |1024 transitions, rollout128/batch32, same64×64 networks |
| Research timing |2000 transitions, original rollout1000, two nominal updates |

Learning-rate interpretation differs: TRPO's setting controls its value optimizer;
its actor uses the constrained step. Equal nominal epochs/critic passes do not
imply equal actor work. Count accepted actor line searches, conjugate-gradient/
Hessian work, nominal updates and actual critic Adam steps separately. A smaller
KL or fewer actor updates is not an economic success criterion.

The existing seven market features, simulator, net-equity reward, costs, whole
shares,100% cap,20-session holding limit and learned stops/target zones remain.
Future performance registration must bind actual data/normalizer/sampler hashes,
fixed training selection and selected/final development gates. Compare matched
PPO plus a strong relevant SAC control with comparable selection; merely beating
a weaker PPO does not satisfy the benchmark/stock/ETF objective. Include all
short-side failures and preserve original multi-seed/prospective requirements.

## Implemented local acceptance

`training/trpo_feasibility.py` validates the fixed protocol and strict21/4 Box
contract. It constructs CPU native TRPO/PPO using existing locked dependencies;
there is no fitting entry point. TRPO remains absent from the generic algorithm
registry so existing commands cannot accidentally start a performance study.

Four focused tests pass without optimization. For all three seeds they prove:
complete common actor/value/log_std initialization,11,465 policy parameters,
empty optimizer state/zero transitions, equal deterministic actions, exact local
prediction repeats, untrained save/load parity, unchanged prediction weights,
and rejection of altered initialization and incompatible geometry/settings.
The fixture refuses any environment step and monkeypatches learning to fail.
These are local engineering checks, not Colab training or cross-machine parity.

## Remaining acceptance before allocation

1. Inspect actual default Colab Python and match it with the current lock; reuse
   the official CLI and bounded bootstrap. No older interpreter or dependency grid.
2. Specify/export frozen inference for the tanh hidden layers and directly clipped
   Gaussian mean. Existing SAC reference actor assumes a different input/activation/
   output contract; it cannot be relabeled as TRPO. Preserve its accepted controls.
3. Verify actual Colab finite updates, critic moments/counters, line-search
   acceptance/rollback, Hessian statistics, save/load/frozen actions and unchanged
   preprocessing. Register one bounded smoke and research-geometry timing study.
4. Require a conservative measured projection covering setup, learning, evaluation,
   transfer and shutdown before any performance screen. Preserve shared150/130
   project and150/170 account protections, prior charges and owned cleanup.
5. Preregister the complete matched experiment and original stock/ETF/median/
   drawdown/selected-final multi-seed gates before observing new development scores.
   Retired2024+ data remains closed; prospective confirmation is still necessary.

No Colab session, model fitting, provider request, retired-test read or new credit
charge accompanied this engineering step. Project usage remains40.213170920/150.


## Frozen actor arithmetic implemented — 2026-11-04

[Reference contract](trpo-reference-inference.md) now implements21-field
tanh-hidden/direct-clipped-mean inference and trusted native PPO/TRPO export.
26 focused checks pass, including existing SAC regressions and isolated
stdlib-only prediction. A retained untrained128-point synthetic probe takes
1.4174s locally; native/reference max difference5.59e−9 is disclosed, not relabeled
exact. Cross-machine/full-path/preprocessing acceptance and actual Colab learning
smoke/profile remain pending. No allocation or optimization occurred.


## Arithmetic probe prepared — 2026-11-04

The [fixed cross-machine probe](trpo-reference-inference.md) binds the retained
untrained actor and128 synthetic vectors/512 reference output coordinates.
It uses one separately capped0.05-unit CPU inference-only allocation, original
shared ledger/account protections and mandatory cleanup; no installation or
optimization.38 focused probe/actor/constructor/lifecycle checks pass. Actual
Colab arithmetic acceptance is next; learning smoke/performance stay unadmitted.


## Actual reference arithmetic accepted — 2026-11-04

The owned Colab probe matches all512 reference coordinates exactly on actual
Python3.13.18/CPU in3.4792 seconds, with no native imports or fitting. Independent
verification and shutdown pass. Project40.213697006/150, measured debit0 versus
0.000526086 conservative charge; fresh account548.04 idle. Next actual Colab
learning-smoke/optimizer/line-search counters, frozen preprocessing/full-path
checks and cost profile. Synthetic arithmetic acceptance is not training success
or permission for a performance allocation.

## Guarded learning smoke worker — 2026-11-04

[Fixed smoke/profile contract](../configs/experiments/trpo_smoke.json) SHA256
`e0b8a26cbfd51eee4046544d49e282abd1409b024ec4a0637d92b774bffc5f7b`
binds the existing111-symbol dataset, original seven-feature normalizer and
accepted arithmetic receipt. It specifies four fresh seed11 cases:

| Case | Transitions | Rollout/batch | Native train calls | Actual Adam invocations |
| --- | ---: | ---: | ---: | ---: |
| Smoke PPO |1024|128/32|8|320|
| Smoke TRPO |1024|128/32|8|320 critic-only|
| Profile PPO |2000|1000/50|2|400|
| Profile TRPO |2000|1000/50|2|400 critic-only|

Total6048 transitions; no smoke-return or checkpoint selection. Network64×64
geometry, reward, features, four actions, costs, training-only normalization and
causal stock/ETF window sampler remain unchanged. Both models in each pair must
share initial actor/value/log_std arrays before either learns. Collection RNGs
reset to seed11 immediately before each case; no warm start.

`training/trpo_audit.py` observes native operations without calling an optimizer
or initiating learning. It records actual Adam invocations, required finite/
nonzero gradient families, complete finite parameter-owned moments and exact
step counters. TRPO actor gradients must be absent during critic Adam; hashes
also prove those steps did not change actor weights. Failed line searches must
restore actor bytes and report zero KL; accepted searches must change actor bytes
and remain strictly below the fixed KL target. At least one accepted search is
required per TRPO smoke/profile case, not assumed from a nominal update count.

Hessian-vector counts are separate: the installed conjugate-gradient solver
uses one initialization product, up to15 iteration products, and one final
scaling product, giving a2…17 bound per native TRPO update. This counts actual
calls, not a nominal15. PPO's native`_n_updates` counts epochs, while TRPO counts
train calls; neither is relabeled as Adam steps. All hooks/wrappers are removed
before checkpoint serialization, including failure paths.

`training/trpo_smoke.py` checks actual default Colab CPU/kernel before any data
construction or fitting. It accepts only the pinned payload, filters environment
episodes to training, and verifies source/preprocessing immutability. It saves
initial/final native ZIPs and reference actors, per-update actor before/after
weights and original observations, actual sampling/backend metadata, reference
probe vectors and durable per-case progress. Native save/load/repeat actions must
match exactly; reference arithmetic is stored separately. No development or
retired-test scoring occurs.

Twenty focused tests pass without local learning, including counter/ownership/
finite-gradient integrity, inconsistent rollback, excessive KL, unauthorized
critic mutation of actor, split/preprocessing rejection and pre-I/O Colab guard.
The observer fixtures populate tensors/state manually and never call backward,
optimizer.step or a real learner. Existing initialization/reference checks pass.

The worker is not yet registered with a Colab controller. Next implement independent
learned-state/KL/rollback/reference restoration and bounded ownership/transport/
budget admission. Actual finite learning, counts, profile timings and full economic
paths are still unverified. This implementation does not admit a performance
screen or change the original project/account protections.

### Independent saved actor restoration

`evaluation/trpo_updates.py` restores the exact seven actor tensors using
read-only affine/tanh operations and a diagonal Gaussian. It independently
recomputes the installed native convention, KL(new || old), averaged over both
rollout samples and action coordinates. It checks saved tensor fingerprints,
update cadence, consecutive actor continuity, finite bounded original observations,
accepted-search KL (the declared1e-6 metadata allowance), and exact rollback.
Reported native accepted KL must still be strictly below0.01. It verifies that
its input tensors remain unchanged and performs no learning or optimizer step.

Eight focused tests cover synthetic accepted updates, corrupted hashes, false
rollback, incorrect reported KL, broken continuity, nonfinite observations,
Gaussian direction, and exact restoration against an untrained native TRPO
policy. These are engineering checks, not evidence that learned updates pass.
The verifier still needs integration with complete retrieved checkpoint/reference
acceptance and a bounded Colab controller before the four-case allocation.

### Retrieved smoke artifact acceptance

`evaluation/trpo_smoke.py` verifies the five bounded regular artifacts per case
against recorded checksums before loading trusted project ZIPs. It restores
initial/final native configurations, transition and epoch/update counters, finite
policy tensors and exact optimizer ownership/moments. PPO requires Adam state
for every policy parameter; TRPO requires critic-only state. Fractional step
counters, missing moments and negative second moments fail acceptance.

Saved actor-update endpoints must match the native checkpoints. Every update
passes the independent Gaussian KL/rollback/continuity verifier. Reference actors
are independently re-exported from each native ZIP and must match exactly;
eight reference predictions use the saved rollout observations and must reproduce
all recorded output bits. Native repeated inference must leave model and optimizer
state unchanged. Sampling, per-update Adam counts, accepted-search/HVP counts,
CPU/thread metadata and positive measured learning time are reconciled. Matrix
acceptance requires equal complete initial PPO/TRPO policies within each pair.
The artifacts are rehashed after inspection.

Seven additional tests cover native initial save/load, counter rejection, both
optimizer ownership patterns, corrupted moments, pre-load checksum rejection,
and complete synthetic PPO artifact restoration plus a tampered reference output.
Synthetic counters/moments are explicitly fixture data, never local optimization
or proof that a real learner ran. Worker observation and controller provenance
are still necessary. Root dataset/preprocessing/source admission and cleanup are
not established by this per-case verifier; the future controller must bind them.
No actual Colab smoke/profile result or performance promotion is claimed.
