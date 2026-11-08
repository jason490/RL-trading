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
