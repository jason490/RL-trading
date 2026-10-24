# Distributional critic research review

TQC is a candidate for a distinct algorithm study, not an implemented project
algorithm or a current trading recommendation. The installed `sb3-contrib==2.7.0`
exports TQC without a dependency change. The project factory/frozen runtime do not
yet register it. No TQC policy has been trained, evaluated or promoted here.

TQC estimates distributions of action values with multiple quantile critics and
truncates high target quantiles. Its research motivation is controlling value
overestimation in continuous control. The primary paper provides continuous-control
benchmarks, not stock-trading evidence. See the [original paper](https://proceedings.mlr.press/v119/kuznetsov20a.html)
and [version-matched implementation documentation](https://sb3-contrib.readthedocs.io/en/v2.7.0/modules/tqc.html).
That implementation supports continuous Box actions and does not supply recurrent
policies. Its documented example uses two critics,25 quantiles and two dropped
quantiles per critic; these are possible fixed study settings, not trading optima.

## Evidence needed before choosing the experiment

The existing [SAC learning diagnostic](sac-temperature.md) records large final
soft-value offsets and much smaller one-step TD errors. It explicitly does not
establish overestimation, inaccurate action gradients or the cause of return
failure. Soft critic values include entropy and discounted continuation. Comparing
them directly to raw profit or deterministic episode return would be invalid.
Failed SAC temperature, state, discount and regularization families remain closed.
TQC must not become a pretext for reopening their parameter searches.

The next useful review is an inference-only calibration diagnostic of a fixed saved
SAC policy on training-only windows. Before executing it, fix the saved model/data
identities, starts, stochastic rollout seeds, rollout count and horizon; account
for reward scale, entropy, action-conditioned first-step values, termination and
timeout bootstrapping consistently. Preserve checkpoints, optimizer state,
normalization and RNG; no local fitting. Report sampling uncertainty and signed
calibration differences. Any inability to reconstruct the policy's exact training
MDP or reward/observation contract must stop that diagnostic, not be filled with
an approximation. Calibration evidence could motivate a critic study but would
not prove a causal performance fix.

## Requirements if a matched study is justified

Freeze a new protocol before any performance scoring or Colab allocation. Candidate
arms are native SAC, distributional TQC with zero truncation, and otherwise matched
TQC with one fixed truncation setting. The zero-truncation control distinguishes
critic-distribution effects from truncation. Use the same native market inputs,
execution state, data, episode sampling, reward, costs, actor architecture,
learning rate, replay/gradient budget and temperature behavior. Record the critic
architecture difference rather than calling the complete networks identical.
Verify shared initial actor weights and RNG setup with actual tensors; do not
claim identical training trajectories across different critic objectives.

Implementation would need factory/configuration registration, input and quantile
geometry validation, actual optimizer/critic restoration, frozen deterministic
save/load parity and report/archive identity support. First run bounded Colab CPU
smokes with actual kernel-version inspection and profiling. Any screen must retain
selected and final checkpoints, compact step/return curves, matched stock/ETF
holding, drawdown and short-side diagnostics. Reuse the established relative and
absolute development gates; only passing screens proceed to seeds11/23/37 and
prospective confirmation. No retired-test access, coefficient/quantile grid,
reward changes or extensions after failure. Use the existing shared project and
account ledger/stop thresholds; no new allocation is authorized by this review
alone without the normal preflight and frozen study protocol.

This review runs independently of financial-source feasibility. No SEC feature
is enabled, no financial coverage failure is hidden, and no trading-return
improvement is claimed.

## Fixed saved-SAC soft-consistency diagnostic

[Protocol](../configs/experiments/sac_soft_consistency.json) now fixes the final
50k full-execution SAC seed11 model, original26-symbol train/development-only
source, exact model/metadata/source-state hashes and frozen source revision
`bb0f3e25bcf60765df158401434959458508da7c`. The frozen package identity is verified
before execution. The current environment has changed since training; the worker
must import the retained original package instead of approximating its behavior.

64 distinct reset seeds sample the original `legacy_uniform` training-start
contract; four fixed action-noise seeds repeat each flat-account state. Each
stochastic frozen-policy rollout ends at the original512-session timeout, data
cutoff or true termination. Its action-conditioned soft target is:

```text
r0 + sum(t=1..H-1, gamma^t * (rt - alpha * logpi_t))
   + gamma^H * (min_target_critic_Q - alpha * logpi_tail)
```

The tail is removed only on true termination; timeouts/data cutoffs bootstrap
exactly as training does. Initial-action entropy is excluded from Q(s0,a0), and
the tail uses the saved target critics. Alpha is the saved final value, never
adapted. This is a multistep soft-Bellman consistency diagnostic with a **learned**
bootstrap, not an unbiased true-value estimator. Flat reset states do not establish
held-position/replay-wide calibration or identify wrong action gradients.

Window means are the sampling units. The predeclared descriptive evidence flag
requires mean(minQ−soft-target) minus two conditional sampling standard errors to
be at least0.1 soft-reward units and at least75% of trajectories to reach512 steps.
These are not profit percentages or confidence bounds for future markets. A
passing flag only motivates separately preregistering a matched critic study;
no model is promoted and no trading screen is automatically launched. A failed
flag closes this particular diagnostic rationale without changing seeds/horizon.

Run locally for **inference only**, with the original frozen package:

```bash
PYTHONPATH=.runtime/sac-execution-source/src \
MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
.venv/bin/python scripts/diagnose_soft_consistency.py \
  --output results/sac-soft-consistency-2026-10-24/summary.json \
  --artifacts artifacts/sac-soft-consistency-v1
```

The worker has a900-second execution bound, a durable per-window journal and no
automatic restart. It verifies source hashes, weights, all three optimizer states,
update count, normalization and Torch/NumPy RNG immutability. No replay buffer is
loaded, normalization fitted, optimizer stepped or validation/test policy scored.
11 focused target/termination/timeout/RNG/deadline tests and targeted Ruff pass
before the first actual diagnostic; no outcome is claimed by this preparation.

## Completed diagnostic — descriptive flag passes

[Actual result](../results/sac-soft-consistency-2026-10-24/README.md):64 flat training
reset states/256 frozen stochastic trajectories complete in38.6seconds. Mean
minimumQ4.031877 versus multistep soft target0.234971 gives gap3.796906; descriptive
lower-two-SE3.625062 exceeds0.1.204/256 trajectories(79.69%) reach512 steps, above75%.
Both fixed criteria pass, with all64 window-mean gaps positive. These numbers are
soft-reward units, not profit percentages or new trading returns.

Original source/simulation/reward/normalization are verified; all weights,
optimizer states, updates, normalization, RNG and protected files stayed unchanged.
All durable window/trajectory arithmetic and aggregate/flag reproduce.11 focused
checks passed before execution and199 combined source/financial/diagnostic checks
pass afterward. Worker exited0/environment closed; no task temporary files or
Colab allocation. No training, checkpoint selection or retired-test evaluation.

The next authorized research work is to preregister and implement one bounded
native-SAC versus untruncated-TQC versus fixed-truncation-TQC comparison, then
verify configuration, frozen inference, optimizer restoration and actual Colab
smokes before a screen. No TQC model exists yet. The result supports that study's
rationale without proving true-value bias, wrong gradients, a trading improvement
or permission to tune the truncation setting after failure.

## Three-arm study preregistered; implementation pending

[Fixed protocol](../configs/experiments/distributional_critics.json) uses native
SAC, untruncated TQC and TQC dropping two of25 quantiles per each of two critics.
All arms retain64x64 actor networks, the original26-symbol source and50k-step
seed11 budgets, full execution state, sampling, costs, reward and10k checkpoint
cadence. This preserves the diagnostic/control data rather than confounding the
algorithm test with the separate107-stock expansion. TQC0 versus TQC2 isolates
truncation; SAC versus TQC0 also changes critic objective/output/aggregation.

Initial actor tensors must match across all arms. Both TQC arms must have exact
shared initial actor/critic/target tensors and optimizer states, and all arms must
restore the native SAC standard post-initialization RNG state. Full SAC/TQC
network equality is not claimed. Native SAC historical scores must reproduce
within fixed0.01pp checkpoint-mean and0.05pp selected/final per-symbol tolerances;
final learned tensors across hardware are not assumed bitwise identical. A control
mismatch invalidates interpretation, without widened tolerances or automatic retry.

Selected and final candidate means must beat **both** fresh controls by2pp, avoid
final stock/ETF deterioration and more than2pp extra drawdown. Both phases must
also meet stock/ETF holding excess≥−2pp and median stock excess≥−2pp. Failure closes
the fixed study with no truncation/quantile/reward/step grid or extension. Only a
passing screen proceeds to the fixed three-seed gates and later prospective
evidence. Smokes must verify real tensors, RNG, geometry, optimizer/frozen parity
and artifacts before any screen. No TQC model has been constructed or trained.

The study has a2-unit incremental allowance within the unchanged original shared
ledger and project150/130/account150/170 stops. Prefer CPU, inspect actual Colab
Python, meter every minute and enforce7200-second inclusive owned deadlines with
profiling/projection reserve. Reuse the existing central fleet and cleanup logic.
Factory/configuration/runtime/report support is the next implementation task.

## Base TQC algorithm and frozen contract implemented

The factory now resolves/constructs `tqc` with the existing SAC collection/update
settings, explicit two25-quantile critics and default drop2. Policy overrides keep
explicit critic defaults; noninteger/invalid critic counts, empty retained-target
geometry and gSDE overrides reject before construction. The fixed study can set
drop0 without changing the architecture. Existing native SAC full/smoke parameters
remain exactly equal to their prior frozen contracts.

Saved TQC bundles require an explicit `critic_geometry` declaration matching the
actual actor/critic/target model. The loader verifies algorithm/policy identity,
critic counts/head widths/target agreement, truncation and retained quantiles.
Missing declarations, checksummed false metadata (including bool-as-int fields),
wrong target heads and TQC checkpoints mislabeled as SAC reject. The generic
training runner now records this declaration for TQC; this does not enable the
pending study's Colab tasks or bypass its initialization/validation requirements.

23 TQC checks form part of42 passing algorithm/frozen regression checks. They
verify actual `(batch,2,25)` critic/target outputs,46 retained targets for drop2
and50 for drop0, exact shared actor tensors across SAC/TQC at32/64 widths,
identical initial TQC0/TQC2 networks/optimizer states, native SAC parameter/
weight/RNG preservation and exact untrained frozen prediction parity with
immutable normalization. A nonempty Adam serialization fixture round-trips
without calling any optimizer step; it is **not learned optimizer acceptance**.
One unrelated gradient-check test was deliberately excluded from this run.
New files pass Ruff; touched legacy modules add no findings to their existing
lint baselines. No local optimization, provider access or Colab allocation.

Next implement the shared native-SAC post-initialization RNG payload, strict study
parameter/archive identity checks and metered Colab job integration. Actual
learned optimizer restoration and all smoke/screen performance gates remain
unverified. No TQC model has been trained or promoted.

## Shared untrained initialization accepted

[Acceptance](../results/distributional-critics-2026-10-24/README.md) now pins12
native actor/critic/target artifacts and the native SAC post-construction global
Torch/NumPy/Python RNG for smoke11 and full11/23/37. Canonical files are under
`artifacts/distributional-initialization-v3`; the tracked binding and loader hashes
prevent silent regeneration. No dataset or provider was accessed by the builder.

`training.distributional_critics` fixes parameters and checks actual fresh models;
`training.distributional_assets` validates the binding/manifest/NPZ/global RNG;
`apply_initialization` preserves Parameter ownership and verifies exact state.
The shared tensor loader now supports fresh off-policy actor/critic/entropy
optimizers and rejects nonempty replay. It preserves existing PPO behavior.
All RNG schema validation precedes mutation, using private generators as probes.
Future restoration of trained checkpoints does not imply restoration of process
RNG; these assets bind the start of learning only.

61 focused tests pass without optimizer steps. They include actual perturbed
state restoration at both widths/all seeds, exact cross-arm draws from all three
global streams, save/load proof retention, ownership, and rejection of malformed
state, wrong seeds, changed effective step-unit/Adam/critic settings, used replay
or optimizers and double application. Package/source hashes and all12 archives are
verified. New files pass Ruff and the touched shared loader adds no legacy findings.
SAC/TQC total policy parameters are29,644/35,884 including target critics.

To deliberately build a new unbound artifact set (never overwrite the pinned set):

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
.venv/bin/python scripts/build_distributional_initializations.py \
  --output artifacts/distributional-initialization-NEW
```

A new set is not adopted automatically; a reviewed binding/version change and
checks are required, without any performance-based seed/tensor selection. The
remaining work is training/task/archive/report integration and actual Colab
smokes, including learned optimizer/replay restoration. No trained TQC result,
financial input, checkpoint promotion or new Colab allocation is claimed.


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


Distributional-critic actual Colab smokes accepted:3×1024 CPU transitions, actual
Python3.13.16, initial tensor/RNG and learned Adam/replay restoration,192 exact
frozen predictions and immutable normalization. All3owned sessions stopped.
Conservative project39.039379/150, incremental0.025001, official account349.00
with zero assignments. Fixed screen runtime projection3717.83<7200seconds;
50k seed11 three-arm plans prepared. No completed screen, return improvement,
retired-test access or promotion. Evidence:results/distributional-critics-2026-10-24/smokes.json.


## Fixed screen reporting

`scripts/report_critic_study.py` accepts only the complete three-arm seed11 fleet
with verified owned shutdown, same clean source/settings, exact task plans and
validated retained archives. It rechecks actual learned optimizer/replay/frozen
restoration and development reports, reconciles matched holding paths, and retains
checkpoint returns/change curves and reporting-only declining-stock diagnostics.
The reporter never learns, fits normalization, selects a new checkpoint or opens
retired data.

Historical native SAC metadata is checksum-pinned. Every checkpoint mean must
match the registered history within0.01pp; selected/final per-symbol returns must
match within0.05pp. Any failure invalidates interpretation/promotion without
automatic retry or widened tolerance. The candidate must also pass both fresh
controls and all fixed absolute stock/ETF/median/drawdown gates. A passed screen
still requires three-seed confirmation and later prospective evidence.

After screen retrieval and shutdown, from the repository root:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
.venv/bin/python scripts/report_critic_study.py \
  --state .runtime/critic-screen-state.json \
  --source-proof .runtime/distributional-source-v1/export-proof.json \
  --output results/distributional-critics-2026-10-24/screen.json
```

Standalone interactive HTML accompanies the compact JSON. Preserve raw artifacts
and release-checksum verification before removing redundant transports. The
reporter is implemented during training; no screen outcome is implied by its
unit tests or availability.


Selected/final critic reports also include the existing reconciled action/exit
diagnostics for stocks and ETFs: requested and closing exposure, modeled cost
drag, long/short net contributions, protective/holding exits and drawdown-window
exposure. Same-session partitions describe observed paths, not counterfactual
causal action effects.20 focused action/accounting/gate checks pass, and both
actual completed nativeSAC reports reconcile across26 development episodes.
No model inference, fitting, normalization change or checkpoint reselection is
needed to inspect these retained paths.


Completed fixed critic50k screen failed: nativeSAC selected/final7.9281/7.9281%,
untruncatedTQC1.9991/1.9991%, truncatedTQC9.7018/7.6975%, holding30.6019%.
Candidate fails both-control/absolute/risk gates; no confirmation, grid or
extension. NativeSAC reproduces every checkpoint and selected/final symbol
exactly.384 smoke/full frozen comparisons and actual learned optimizer/replay
checks pass; six owned sessions stopped. Conservative project39.154379/150,
study0.140001/2, official account348.89 with zero rate/assignments. Retired-test
isolation and immutable normalization preserved; objective remains unmet. See
results/distributional-critics-2026-10-24/screen.json and compute.json. Publication
and scoped duplicate cleanup remain next; no promoted model.


Distributional-critic publication/cleanup verified:12 GitHub release asset hashes
and sizes match;18 duplicate transports/upload payloads removed (520,262,328
bytes),191 retained files rehashed at cleanup. All models/replay/checkpoints,
source/data, curves and owned recovery records retained. Subsequent README
publication update is explicitly recorded in cleanup evidence. Six owned sessions
stopped, zero assignments; no additional units or training. Fixed critic study
failed and remains closed; next work is bounded saved-policy/execution sensitivity
diagnosis, not extension or promotion. Release:
https://github.com/jason490/RL-trading/releases/tag/distributional-critics-2026-10-24


## Fixed inference-only final-update diagnosis

`configs/experiments/critic_update_diagnostic.json` pins the accepted failed study
summary, cleanup inventory, frozen source package and numeric helper before
execution. For all three arms, compare the retained50k-transition checkpoints
with48,999 versus49,000 actual updates. Measure actor/critic tensor displacement,
normalized deterministic-action differences and diagonal pre-tanh Gaussian
KL(before||after) on256 evenly spaced observations from the accepted truncated
model's training replay. The exact same observations are used for all arms.
This selected replay is not representative deployment or independent validation
coverage. No new price rows, returns, checkpoints or data preprocessing are fit
or selected; no promotion threshold or rescue sweep is introduced.

The worker requires verified owned shutdown/source hashes and learned-state
contracts; it preserves weights, optimizers, files and all global RNG streams.
It has a120-second computation deadline and durable progress journal; an existing
journal rejects automatic restart. Three numeric tests cover known parameter
norms, action sign/shape/bounds and analytical Gaussian KL/asymmetry. This is
preflight evidence; actual diagnostic outcomes remain pending.

```bash
PYTHONPATH=.runtime/distributional-source-v1/src \
MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
.venv/bin/python scripts/diagnose_critic_update.py \
  --output results/critic-update-2026-10-24/summary.json
```


The first actual diagnostic was rejected before comparisons because the generic
compact callback stores `checkpoint-50000.zip` after the final update. Actual
ZIP counters and the callback implementation confirm `latest-checkpoint.zip`
is the retained on-step48,999-update file for all three arms; latest.zip has
49,000. The failed journal and original protocol are preserved. Reviewed v2
(`configs/experiments/critic_update_diagnostic_v2.json`) changes only checkpoint
file selection, pins the failure journal and uses a separate progress journal.
Models,256 observation indices, measurements, deadlines and interpretation stay
fixed. No returns/comparisons or optimizer steps occurred in the rejected attempt.


[Saved critic final-update sensitivity](../results/critic-update-2026-10-24/README.md) complete: three one-update pairs on
256 fixed candidate training-replay observations; truncatedTQC actor relativeL2
0.03356%, exposure mean absolute delta0.001462, GaussianKL0.001133 and zero
exposure-sign flips. These are normalized action/tensor diagnostics, not returns
or a cause of development deterioration. Independent Torch arithmetic agrees;
all protected files/model/Adam state unchanged, worker exited0, no local
optimization or Colab units. First phase-filename rejection and reviewed v2
correction retained; sample/metrics unchanged. Fixed failed family remains
closed. Next bounded saved-development-path audit before another training
hypothesis; no promotion or retired-test access.
