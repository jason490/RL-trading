# TRPO/PPO reference inference

This module gives the proposed tanh-hidden policy a separate frozen arithmetic
contract. It leaves existing task-SAC actors and accepted results unchanged.
[Local engineering evidence](../results/trpo-feasibility-2026-11-04/README.md)
and the fixed cross-machine probe pass; learned-policy and full-simulator
acceptance remain pending.

## Input, activation and output contract

`inference/canonical_tanh_policy.py` uses schema`canonical-tanh-policy-v1` with
exactly21 already-prepared observation fields, two64-unit tanh layers and four
linear means. The final means are clipped directly to−1…1. There is no final
tanh or SAC Box-unscale transform. The actor emits the existing four-component
action; simulator/risk decoding is unchanged.

Every input/weight/bias is finite binary32. The existing reference math sums
exact products, rounds each affine result once to nearest-even binary32 and
certifies hidden tanh rounding using Decimal intervals. Final clipping uses
exactly representable endpoints. This is a defined arithmetic model, not a
claim that native Torch BLAS/tanh produces identical bits. Existing SAC schemas,
input22 contract, ReLU layers, outputs and arithmetic implementation are intact.

The JSON envelope binds the whole source-checkpoint SHA256, algorithm label,
geometry, output transform and encoded weights with a canonical checksum.
Loading bounds size to512KiB and rejects duplicate/extra fields, bad dimensions,
unknown contracts and nonfinite weights. Internal tuples/Decimal values are
immutable; exported manifest views are copies. Prediction loads no Torch/NumPy,
updates no optimizer and has no access to normalization, data or order execution.
Callers must still supply a separately verified frozen preprocessing/schema/feed
bundle. This actor alone is not a complete deployment bundle.

## Trusted checkpoint export

`inference/canonical_tanh_export.py:export_policy` accepts only bounded task-owned
PPO/TRPO checkpoints, never arbitrary untrusted pickle files. It checks algorithm
metadata, exact native policy/extractor/distribution classes, float32 state,
21/4 Box geometry,64×64 tanh layers and direct Gaussian mean head. Wrong PPO/TRPO
source labels reject export. Host Python/NumPy/Torch RNG states and original
checkpoint bytes are preserved; output is created once rather than overwritten.
No model fitting or preprocessing update occurs.

```python
from rl_stocks.inference.canonical_tanh_export import export_policy
from rl_stocks.inference.canonical_tanh_policy import CanonicalTanhPolicy

export_policy("TRUSTED_OWNED.zip", "NEW_reference.json", algorithm="trpo")
# Bind expected_source to the independently recorded SHA256 of the source ZIP.
actor = CanonicalTanhPolicy.load("NEW_reference.json",
                                expected_source=source_sha256,
                                expected_algorithm="trpo")
action = actor.predict(prepared_observation_21)
```

## Actual local acceptance and remaining work

Twenty-six focused checks pass: known-value two-stage tanh, direct clipping,
geometry/contract/source/checksum rejection, immutable state, trusted export and
RNG preservation, paired PPO/TRPO reference predictions, isolated stdlib-only
prediction, and regression acceptance of old SAC actors. Three seeds' untrained
native PPO/TRPO starts and local save/load/repeated predictions also pass.

The retained untrained128-point synthetic probe takes1.4174s locally, with native/
reference maximum action difference5.58794e−9. Only74/512 coordinates match bits;
this is reported honestly. No approximate match was substituted for exact parity.
No economic path, market return or accuracy claim follows from these observations.

Before performance allocation, reproduce the exact reference vectors on the actual
default Colab kernel, test actual learned state/save-load/frozen predictions,
validate preprocessing and complete economic paths, and measure both learning and
reference evaluation costs. A future matched performance protocol must declare
its numerical execution and selection rules before new results; it must not
silently reuse differently evaluated controls or widen old failed tolerances.
CPU profiling, original cumulative budgets/account floors and verified owned
cleanup remain mandatory. No new Colab charge accompanies this local work.

## Exact cross-machine probe runtime

[Fixed probe protocol](../configs/experiments/trpo_tanh_probe.json) SHA256
`4e50eaa4cf31396172e397ad312a78b78085f4e69b7f29c040779e82a253219b`
binds the untrained actor,128 synthetic21-field inputs and512 expected output
coordinates. It permits one CPU inference-only allocation, maximum0.05 units,
1200-second deadline and360-second teardown reserve. Original project150/130
and account150/170 limits remain; actual rate/balance are read after allocation.
No optimization, market evaluation, normalization, dependency installation,
interpreter replacement or performance allocation is admitted.

```bash
MPLCONFIGDIR=$PWD/.runtime/matplotlib PYTHONPATH=src \
.venv/bin/python scripts/prepare_tanh_probe.py \
  --output .runtime/trpo-tanh-reference-probe-v1.json
MPLCONFIGDIR=$PWD/.runtime/matplotlib PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.tanh_probe \
  .runtime/trpo-tanh-reference-probe-v1.json \
  --cli-python /home/jassno/.local/share/uv/tools/google-colab-cli/bin/python
```

Preparation checks retained artifact hashes, regenerates host results in an
isolated stdlib process and compares all retained output bits. The controller
rejects input/protocol changes before provisioning. It reuses the original shared
ledger/allocation lock, one immutable claim, minute metering, bounded downloads
and mandatory owned cleanup. No GPU/TPU flag is passed; the installed CLI documents
that this creates CPU, and post-allocation status is independently checked.

The worker ships only the minimal Python source and reference JSON to the actual
default kernel. It checks Python3.13 before invoking a fresh`-I` subprocess with
an180-second bound. Exact output bits/checksums,128×4 shape, default major/minor,
finite clipped actions and absence of imported Torch/NumPy are mandatory. This
checks reference arithmetic, not native Torch bit parity or economic-path parity.
The CLI exec timeout is explicitly240 seconds.

Claim: `.runtime/trpo-tanh-reference-probe-v1-claim.json`. Retrieved output and
cleanup summary remain under `artifacts/colab/OWNED_SESSION`. Never delete the
claim to retry. On any failure/interruption the controller attempts retrieval,
stops only its owned session and verifies termination; cleanup failure reports
the owned ID and recovery command. Thirty-eight focused probe/actor/constructor/
shared-lifecycle checks pass, including failure cleanup, bit/runtime mismatch and
repeat-allocation rejection. Actual Colab outcome remains pending.


## Actual cross-machine outcome — 2026-11-04

[Exact acceptance](../results/trpo-feasibility-2026-11-04/cross-machine-acceptance.json):
all128 fixed points/512 output coordinates match host reference bits on actual
ColabPython3.13.18/CPU. Neither native package was imported;3.4792s inference.
Independent post-cleanup verification passes, and initial source hashes remain.
No training, markets or preprocessing updates. Owned shutdown verified, measured
debit0 and conservative charge0.000526086; project40.213697006/150. Fresh account
548.04, rate0, assignments0. One verified staging duplicate removed, retained
weights/source/results/ledger/claim unchanged. This does not admit a performance
study: learned-policy, bundle/full-path acceptance and actual learning profile
remain necessary. The claim prevents repeating this allocation automatically.

## Full frozen preprocessing bundle — 2026-11-08

`inference/tanh_bundle.py` packages the already exported learned actor with the
original frozen seven-feature normalizer, full21 observation schema, four-action
contract, original simulation settings, data provenance and dependency lock.
Content-addressed identity and exact checksummed file sets reject corruption,
extra files, symlinks and schema/feed/execution mismatches. Source actor and
normalizer identity are preserved. Creation never overwrites an existing bundle.
The generic SAC bundle format and accepted numerical contracts are unchanged.

`FrozenTanhPolicy.observation` reproduces the environment's raw-feature
normalization followed by binary32 conversion and14 portfolio-state fields.
`predict_features` checks research symbol, UTC availability/decision/current time
ordering and maximum input age before deterministic reference prediction. It
returns research actions only, never submits orders or updates preprocessing.
Manifest access returns a copy; normalizer fields remain frozen tuples. Callers
must supply state from the matching simulator/account contract; arbitrary state
values are not independently verified broker positions.

Four focused tests cover complete normalization/reference parity, immutable
state, create-only packaging, extra/checksum/schema/feed/bundle-ID rejection,
feature ordering/shape, unavailable/stale inputs and unknown symbols. Actual
local inference acceptance packages all four learned Colab actors and compares
32 original training-row observations against the real environment, then checks
exact frozen/direct-reference actions. Original dataset/actors remain unchanged.
[Receipt](../results/trpo-smoke-2026-11-08/frozen-preprocessing.json) binds retained
bundles and detailed proof under `artifacts/trpo-frozen-preprocessing-v1`.
No local fitting, return evaluation, checkpoint selection, Colab allocation or
retired-test access occurred. Cross-machine full preprocessing and complete
saved economic-path parity remain pending before a performance comparison.

## Fixed complete economic-path parity protocol

[Protocol](../configs/experiments/tanh_full_paths.json), canonical SHA256
`f81d5de49c3206ae92272accf1c35db154cf997ad8a59177b5807ebb68668834`, fixes
all four accepted learned bundles and all26 original symbols, with unchanged
training rows anchored2017-12-29 through the purged final2020-12-02 session.
Each complete non-bankrupt panel has19,136 decisions. Exact episode-view digest,
original dataset/normalizer and accepted frozen receipt prevent panel changes.
This is engineering acceptance, with no profit gate, checkpoint selection,
development scoring or performance promotion; retired2024+ remains closed.

`evaluation/tanh_paths.py` creates two independent simulation accounts per episode.
One uses the direct reference actor on ordinary environment observations; the
other uses the frozen bundle's raw-feature/state normalization and availability
checks. Every observation/action binary32 value must match. Every subsequent
reward, fill/event, cash/quantity/basis, costs/borrow, age, stop/target state,
termination flag and final metric must match exactly. JSONL evidence preserves
binary64 values as hexadecimal strings, including signed zero. There is no
numerical tolerance. Full paths and per-episode summaries remain outside Git;
per-case sparse progress supports observation and failure recovery. Partial
outputs are retained on failure and fresh destinations prevent silent reruns.

Short-side attribution is retained per episode for later reporting; no declining
stock label enters prediction or acceptance. Bankruptcy may legitimately end an
episode early; otherwise every available decision is required. Frozen bundle
files, preprocessing and the complete original dataset are rechecked unchanged.
Existing simulator accounting acceptance remains a separate evidence layer;
two executions of the same simulator do not independently prove its economics.

Four new fixture tests cover complete/repeated path identity, action corruption,
nontraining rejection, a1e-8 equity discrepancy, and signed-zero/nonfinite state.
Together with bundle checks, eight focused tests pass without learning. Actual
full-panel local replay and subsequent cross-machine replay are still pending.
The local command is inference/backtesting only, never local optimization:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_tanh_paths.py \
  data/datasets/cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b/dataset.json \
  --bundles artifacts/trpo-frozen-preprocessing-v1 \
  --output artifacts/trpo-full-path-local-v1
```

No Colab allocation is admitted by this protocol; a separate bounded runtime
contract is required for cross-machine execution. Native Torch actions are not
claimed bit-identical to reference arithmetic; this verifies the explicit frozen
reference execution contract rather than reopening native numerical failures.

## Independent saved-path restoration

`evaluation/tanh_path_acceptance.py` restores canonical binary64 evidence and
checks every recorded raw-feature/state observation, next-observation continuity,
action bits, equity identity, rewards, execution dates and terminal conditions.
It replays recorded actions against the original bars through a fresh simulator,
requiring exact fills/events, cash, basis, borrow/cost totals, protection and
holding state. Final metrics and directional attribution are independently
recomputed. Missing records, untagged/noncanonical/nonfinite floats, oversized
records and even1e-8 accounting/summary changes fail acceptance.

Complete-directory acceptance requires the fixed four-case order, original
training-panel identity, all paths and per-case/root metrics/digests. A comparison
requires both directories to pass restoration before comparing every path hash
and result exactly. This comparison alone does not prove two machine locations;
`runtime_locations_verified=false` explicitly leaves Colab/host provenance and
owned-session cleanup to a future runtime controller.

Eight new tests cover complete file/directory restoration and mutation rejection;
16 focused acceptance/path/bundle checks pass without fitting. The existing local
replay continues under its original process and source3607704; the new checker
has independently restored its first16 completed real episodes. This is partial
progress, not full-panel or cross-machine acceptance. Source provenance was saved
before adding the checker; the replay's calculation modules were not modified.

## Prepared inference-only remote transport

`scripts/remote_tanh_paths.py` prepares an inference-only Colab worker using the
actual default Python3.13 kernel, the existing locked dependency environment and
the accepted bootstrap deadline helper. Notebook arguments do not select worker
mode; only the explicit subprocess `--worker` entry does. The worker requires
owned Colab CPU/default-version guards before dataset access and checks installed
dependency versions. It never calls a learner or changes normalizers/checkpoints.
It reuses the exact full-path calculation function; a wrapper only publishes
durable partial artifacts at300-second intervals between completed episodes.
Final success/failure exports include complete path files and provenance.

The shared metered CPU controller now exposes configuration, input-packaging,
preflight, worker-path and artifact-acceptance hooks. Inference subclasses can
reject admission before claims/allocation and record `training=false` plus
`artifacts_verified` without falsely claiming training. Existing budget guards,
minute metering, single-use claims, same-worker stream-loss handling and verified
cleanup remain shared. Original training/review fixtures continue to pass.

Twenty focused worker/controller/review checks pass without allocation, including
local pre-I/O refusal, notebook argv dispatch, specialized preflight rejection,
correct inference labels and post-cleanup acceptance. This worker is unregistered:
no remote allocation is admitted. Next bind the completed independently restored
local result, original numerical source inventory, four bundle assets and source
receipt into an exact runtime/upload contract. That contract must enforce shared
150/130 project and150/170 account limits and a separate bounded inference cap;
location provenance and exact remote comparison are still unverified.

## Remote input and bounded-container preparation

`runtime/colab/tanh_path_inputs.py` discovers the internal declared import closure
of the exact replay entry, including imports inside functions and relative imports.
All114 numerical import files currently match Git revision3607704, the original
local replay revision. The separately saved local provenance/hash must match that
baseline. This inventory covers declared imports; arbitrary dynamic import strings
are not interpreted as a claim of universal executable-code analysis. The actual
replay source uses the checked entry/import structure.

Remote preflight refuses a missing complete local result before restoration or
allocation. A complete result must independently restore all fixed paths and
metrics before any runtime-specific budget admission. The returned evidence does
not itself authorize allocation. Four content-addressed bundles are checked for
identity, feed/schema, actor/model hashes and exact file sets; their16 files plus
the original accepted public receipt form17 allowlisted source assets. A separate
sidecar records the checked numerical inventory. The remote worker compares that
inventory against its uploaded calculation sources before reading market inputs.

The existing source uploader now accepts explicit bounded inference assets under
controlled archive prefixes, rejecting duplicate/absolute/traversal/hidden paths,
symlinks and oversized files. Default source uploads still omit credentials,
artifacts and generated caches; standalone pyc/pyo files are now excluded too.
`runtime/colab/bounded_archive.py` centralizes pre-extraction count/expanded-size,
path/link/duplicate and failure-marker checks. Original smoke extraction retains
its unchanged128MiB/100-member bounds; a future full-path runtime must explicitly
supply its own bounds rather than weakening the smoke contract.

Twenty-seven focused import/source/asset, worker, controller and transport checks
pass without allocation. Actual checks confirm all114 original numerical sources
unchanged and17 valid bundle/receipt assets. The original local replay remains
active; its calculation files have not changed. A fixed remote runtime contract,
complete local-result binding, location provenance and cross-machine acceptance
remain pending. No additional Colab credits or local fitting occurred.

## Gated full-path Colab runtime

[Runtime contract](../configs/experiments/tanh_paths_runtime.json), canonical SHA256
`0682ddded6921b4bea243f0e952cd8a3cdc0285ce9435a625504cd9336765be8`, fixes
one CPU inference allocation at0.25 units,7200-second deadline,900-second
retrieval/shutdown reserve,6300-second preferred execution window (clipped after setup) and300-second
checkpoint interval. Original cumulative150/130 project and150/170 account
limits remain enforced by the shared controller. Setup, idle, failures and
retrieval count. No automatic worker/allocation retry is introduced.

`runtime/colab/tanh_paths.py` requires complete independent local restoration and
unchanged numerical source/bundle assets before creating a single-use claim.
That claim captures the exact local result/provenance hashes and numerical
inventory. Remote artifacts are accepted only after the owned session is stopped,
with matching source/Git/kernel/dataset/inventory and disabled training/test flags.
The full-path container has explicit512MiB/256-member bounds; original smoke
bounds remain unchanged. Complete remote manifests and both restored path sets
must agree exactly before location verification is marked true. Local result,
provenance and numerical sources are rechecked to detect changes during the job.

Twenty-four focused runtime/input/worker/controller checks pass without allocation,
including stopped-session enforcement, partial-panel preflight rejection,
source/kernel/test-flag/file-manifest corruption and local-provenance mutation.
No runtime has been allocated for this full-path job. The active local replay is
still incomplete; the command below must fail before allocation until all four
local cases pass. Once it is consumed, do not rerun its claim.

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.tanh_paths \
  data/datasets/cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b/dataset.json \
  --cli-python /path/to/existing/colab-cli/python
```

This is inference-only numerical acceptance. Its output never selects actors,
changes failed-family gates or establishes improved trading returns.

Before any full-path allocation, the preferred worker limit was changed from5400
to6300 seconds, with the same7200 allocation deadline/900 reserve and0.25 units.
The controller clips this preference to actual time remaining after setup, with
an additional10-second margin. This removes an arbitrary shorter worker cap for
the long complete-panel replay; it does not extend a deadline, change the
scientific panel or restart a failed remote operation. A test checks clipping
including600 seconds of elapsed setup. Current local replay has completed three
panels; first two independently restore all38,272 decisions. Final panel pending.

### Actual complete local path acceptance

The original process has now finished all104 episodes/76,544 decisions. Independent
saved-path restoration passes the full matrix, and all114 numerical import files
remain identical to original replay revision3607704. [Bound local proof](../results/trpo-smoke-2026-11-08/local-full-paths.json).
Five superseded progress summaries were removed only after reconciliation; all
complete lossless paths, metrics, source provenance, independent acceptance and
protected ledger/plan hashes are retained. No credits or fitting were used.
The gated remote inference comparison can now run after fresh admission/metering;
remote equivalence and trading-return improvement are not yet established.
