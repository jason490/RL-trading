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

Actor weight export, schema-checked loading, ReLU/output scaling, deterministic
stock/ETF routing, frozen bundle checksums and actual saved-model/episode replay are
not implemented for this arithmetic path yet. Latency and supported Colab default
Python must be checked before any allocation. The execution variant and its gates
must be declared before reading its new evaluation outcomes. Original native reports
must not be relabeled as reference-arithmetic results or used to claim exact parity
with this different numerical definition. Any later training remains Colab-only
and shares the unchanged project ledger and account protection.
