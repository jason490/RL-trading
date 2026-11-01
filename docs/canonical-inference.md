# Reference inference arithmetic

This category covers a proposed execution path for reproducible saved-model
inference. It does not accept the stopped task-conditioning family, change its
performance gates or establish improved trading returns. The existing native
reports and numerical rejections remain preserved.

## Why this is needed

The masked control's native reports replay exactly on the host with one CPU math
mode, but the tagged control does not. A first-checkpoint diagnostic finds small
floating differences without audited discrete trade differences. Neither that
small size nor backend documentation makes a failed exact contract pass. Repeated
backend guessing also does not provide a reliable contract for future runtimes.
The next path needs explicitly defined arithmetic used by both evaluation and
frozen prediction, with its own version and measurements before results are read.

## Implemented arithmetic core

`inference/canonical_math.py` defines `binary32-exact-dot-certified-tanh-v1`:

- Inputs are finite binary32 values. Dot products sum exact input/weight products
  and bias at512 decimal digits, then round once to nearest-even binary32. Vectors
  are bounded to4096 elements. Finite binary32 products and their bounded sums
  require fewer than400 significant decimal digits, so this accumulation is exact.
- Decimal-to-binary32 conversion checks neighboring candidates by exact distance
  and tie parity. This avoids double-rounding errors from an intermediate binary64
  conversion. Zero has a canonical positive representation; overflow/nonfinite
  values reject rather than propagate into policy actions.
- Tanh obtains a correctly rounded Decimal exponential, constructs an enclosing
  exponential interval, and evaluates monotone endpoint bounds with directed
  division. It returns a binary32 answer only when both endpoints round to the same
  bits. Precision increases from80 through1280 digits; unresolved rounding rejects.
  Magnitudes at least16 safely round to signed one in binary32.
- Private contexts specify precision, rounding, exponent limits and traps. Ambient
  Decimal settings and CPU math environment do not control these calculations.
  There is no BLAS, native tanh, model fitting, thread mutation or RNG operation.

Python documents exact float-to-Decimal conversion, directed rounding and the
correctly rounded nearest-even exponential in its
[Decimal reference](https://docs.python.org/3.13/library/decimal.html#decimal.Decimal.exp).
The enclosing-interval and binary32 rounding logic is this repository's own code;
the documentation is not evidence that the model integration is already verified.

## Verification and remaining work

Fourteen host unit checks cover cancellation, tie parity, double rounding including
a600-digit perturbation, subnormals, odd symmetry, reference tanh examples, overflow,
nonfinite input and isolation from ambient precision/rounding/traps/exponent limits.
These checks do not constitute cross-machine model acceptance.

## Portable task actor export and loading

`inference/canonical_export.py:export_actor(source, output)` loads a trusted,
task-owned SB3 SAC checkpoint on CPU and writes a fresh JSON file. Loading a SAC
ZIP uses pickle: this exporter is not a safe reader for untrusted downloads.
It checks the actual TaskSACPolicy/TaskActor types, flattened22 input, two64-wide
Linear/ReLU layers, four outputs, normalized Box bounds and finite binary32 actor
state. Routed actors export separate stock/ETF mean heads. Stochastic log-standard
deviation heads are not used in deterministic prediction. The exporter preserves
Python, NumPy and CPU Torch RNG state; it performs no optimization or fitting.
Existing output files are never overwritten.

`inference/canonical_actor.py:CanonicalActor.load(path, expected_source=...,
expected_role=...)` requires only the standard library. It checks the envelope
SHA256, exact schema fields, dimensions, finite hexadecimal IEEE binary32 weights,
role and optional caller-supplied source-checkpoint identity. JSON duplicate keys
and files larger than512KiB are rejected. The body identifies schema
`canonical-task-actor-v1` and the arithmetic contract above. SHA256 detects content
changes; it does not authenticate who produced a file. Callers must pin a trusted
expected identity. Internal decoded layers are immutable tuples; manifest views
are independent copies.

Prediction consumes one already-preprocessed22-coordinate vector. Masked actors
require type0; tagged actors require stock−1 or ETF+1. Each hidden layer applies
exact-dot/one-round arithmetic followed by ReLU. The selected mean head uses
certified tanh, then the explicit SB3 Box unscale formula with binary32 rounding
after each add/multiply. This scaling is recorded as
`sb3-box-minus-one-plus-one-binary32-v1`; it is not assumed to be a bitwise identity
even for Box[-1,1]. No RNG, normalization updates, native BLAS or Torch prediction
occurs. This is an actor component, not a complete frozen research bundle: callers
must still supply the correct frozen preprocessing, causal instrument registry,
action-execution rules and episode state.

Tests cover shared/routed heads, exact cancellation through hidden layers, ReLU,
output scaling, checksum/source/role rejection, malformed geometry/nonfinite
weights, duplicate fields and size limits, immutable views, all three trusted
checkpoint exports, preserved RNG and no-local-optimization guards. Fresh Python
subprocesses produce identical output bits with different native backend selector
environments and hostile ambient Decimal settings; those subprocesses assert
neither Torch nor NumPy was imported. These are host tests, not cross-machine
Colab acceptance or proof of native Torch parity.

## Remaining integration and research gates

Full frozen bundle integration, saved-model episode replay, throughput profiling
and actual cross-machine validation remain pending. The execution variant and its
gates must be declared before reading its new evaluation outcomes. Latency and
Colab default Python must be checked before allocation. Original native reports
remain unchanged and must not be relabeled as reference-arithmetic results. No
new performance conclusion or native-study restart is authorized by these unit
checks. Any later training remains Colab-only and shares the unchanged project
ledger and account protection.
