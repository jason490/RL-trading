# TRPO/PPO reference inference

This module gives the proposed tanh-hidden policy a separate frozen arithmetic
contract. It leaves existing task-SAC actors and accepted results unchanged.
[Local engineering evidence](../results/trpo-feasibility-2026-11-04/README.md)
is complete; cross-machine and full-simulator acceptance remain pending.

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
