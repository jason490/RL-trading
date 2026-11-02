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

`inference/canonical_bundle.py` now derives a separate checksummed reference bundle
from a trusted native task bundle. It retains the original manifest, original
bundle/checkpoint identities and source file checksums, plus the new actor identity
and arithmetic version. Source files are checked before and after export and stay
unchanged. The new directory identity hashes its own manifest; loading optionally
requires a caller-pinned bundle ID. Normalization, feature/feed/action schemas,
full21 observation specification, coarse-type evidence scope and dated symbol
registry are validated before prediction. `raw_action` retains all four output
components; `predict` applies the frozen evaluation exposure filter. The source
native ZIP remains in its original durable archive, not duplicated in the new
bundle. This derived format is research-only and makes no native-parity claim.

`evaluation/canonical_task.py` adds one chronological cursor per episode and
applies the execution filter once through the existing simulator/evaluator.
`evaluate_canonical_training` requires the exact source dataset, normalizer and
type-manifest identities, then restricts replay to the fixed original26 training
window. It has no development/test argument. The episode adapter rejects non-train
splits and dates after2020. A short synthetic episode reproduces the complete
rows/events/metrics/baselines exactly against an independent direct-actor adapter;
this validates wiring, not performance on real saved market episodes.

The [host timing record](../results/reference-inference-2026-11-02/host-profile.json)
uses the saved tagged selected actor and128 fixed synthetic full21 vectors:
0.3911 seconds total, about3.06ms per prediction on localPython3.13.16. Inputs
are generated as `((i+j)%17-8)/16` for i=0..127 and j=0..20, with the ETF tag
supplied by the frozen SPY registry. Four warmup calls precede timing. This is not
an episode return evaluation or Colab throughput guarantee. Forty-two focused
actor/arithmetic/bundle/native-bundle tests pass, including corruption and source
binding, normalization preservation and no-local-optimization guards.

Actual saved-market episode replay, complete end-to-end throughput profiling and
cross-machine validation remain pending. The execution variant and its
gates must be declared before reading its new evaluation outcomes. Latency and
Colab default Python must be checked before allocation. Original native reports
remain unchanged and must not be relabeled as reference-arithmetic results. No
new performance conclusion or native-study restart is authorized by these unit
checks. Any later training remains Colab-only and shares the unchanged project
ledger and account protection.

## Bounded cross-machine engineering probe

The v1 probe is fixed before its Colab outcomes: masked shared selected50k,
tagged shared selected50k and routed smoke1024 from the source bundle IDs pinned
in `scripts/prepare_reference_probe.py`. Using different-duration sources tests
serialization/routing, not relative model performance. Original native failures
remain unaccepted. Each actor consumes128 fixed synthetic vectors per supported
tag, generated by the host-timing formula above:128 masked plus256 each tagged
role,640 predictions total. Compare every binary32 output bit, input/actor hashes
and output digest, with no tolerance. Neither market data nor return evaluation
is included. Each role must finish below120 seconds; this is an engineering
bound, not a prospective throughput promise.

Preparation exports only trusted retained checkpoints and launches a fresh
isolated standard-library Python subprocess for host expectations. The remote
worker uses Colab's actual default `sys.executable` in a fresh isolated subprocess,
requires Python3.13 and rejects native Torch/NumPy imports. A changed Colab minor
version must be inspected and supported by updating the project before continuing;
no interpreter is installed or replaced. Only the small allowlisted stdlib source
ZIP and actor JSON travel in the cell; there are no dependency downloads or fits.

Run after a clean committed preflight:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_reference_probe.py
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.reference_probe \
  .runtime/canonical-reference-probe-input-v1/input.json \
  --cli-python /home/jassno/.local/share/uv/tools/google-colab-cli/bin/python
```

The controller reuses the existing official CLI and the shared allocation lock,
original cumulative ledger150/130, account floor150/teardown170 and minute metering.
Only one CPU allocation is allowed by its one-time persistent claim. The deadline
is20 minutes including provisioning/retrieval, with360 seconds reserved for
teardown, and incremental allowance0.05 units including failures/idle/downloads.
It checks idle balance before allocation, rate afterward, projects allowance
between readings, retrieves with bounded retries, and stops/verifies its owned
session on success, failure or handled interruption. No automatic retry or native
study restart follows a failed probe. Shared budget stops remain persisted.
The claim and ledger retain the session ID; cleanup failure prints the exact
`colab_safe.py stop -s <owned-session>` recovery command. No unrelated sessions
are stopped; an existing assignment prevents allocation.

An accepted synthetic probe would still leave real saved-market episode replay,
complete simulator cross-machine equality and a declared numerical execution
variant for future performance research pending. No gate can promote a policy
from this synthetic check.

### Actual v1 outcome — 2026-11-02

The [Colab acceptance receipt](../results/reference-inference-2026-11-02/cross-machine-acceptance.json)
passes all640 exact prediction-bit comparisons, including both tagged/routed
branches. Actual defaultPython3.13.18 matches the supported3.13 minor. No native
packages were imported by the worker. CPU latency was6–8ms/prediction; complete
simulator throughput is still unmeasured. Owned session
`rl-stocks-cpu-f04787394fc0` terminated and was independently confirmed idle.
Conservative probe charge0.000553813 units, measured CLI debit0.00 at reported
resolution; project40.121717184/150. No dependency installation, fitting, market
returns, original-native promotion or retired-test access occurred. The one-time
claim remains consumed. This successful engineering check does not authorize
rerunning the stopped native family or change its numerical rejection.

## Fixed saved-market replay protocol

`configs/experiments/canonical_task_replay.json`, canonical SHA
`f5fb168a3e71ee071917f53697965e378296237bcb8dff1fce9719a1aca8d328`,
is declared before its new saved-market results. This is a different numerical
execution definition from original Torch inference; the original rejected tagged
reports stay rejected. It uses the already retained tagged-shared selected50k
checkpoint (source bundle16b5ce4a…, checkpoint SHA066f23b7…) and does not select
another checkpoint or compare it to development outcomes.

Exactly26 original symbols and19,136 decisions cover the2017-12-29 anchor,
2018-01-02 first fill and2020-12-02 final session. The full input-window digest,
source dataset, seven-feature normalization, checkpoint and action execution
contracts are pinned. Existing next-session simulator defaults, full21 causal
state and unit-magnitude evaluation exposure remain unchanged. Source data,
weights and frozen metadata are hashed before/after. Complete chronological
paths, matched holding, per-session long/short attribution and an interactive
report are retained. Falling-stock labels are descriptive only.

Run once from clean committed source:

```bash
MPLCONFIGDIR=$PWD/.runtime/matplotlib PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/replay_canonical_task.py
```

The local process has a hard1,200-second deadline; it performs inference and
simulation only. Guards reject SAC native `predict`, `learn`, `train`, Adam
`step` and tensor `backward`, restoring methods on handled exit. It cannot
allocate Colab, query providers, update preprocessing, open development/test
results or grant performance promotion. The output directory must be fresh;
failures retain evidence and are not automatically retried. Full simulator
cross-machine equality and a separately declared future performance study remain
required regardless of fitted-window return. Three focused diagnostic tests cover
the pinned protocol, chronological/split rejection and guard restoration, alongside
35 actor/arithmetic/bundle tests.

### Actual saved-market outcome

The [fixed market receipt](../results/reference-inference-2026-11-02/market-replay.json)
completes26 training episodes and19,136 decisions in66.58s, total70.36s including
export/reporting. Exact source identities remain unchanged. Directional P&L
reconciles every row with maximum error3.21e-10 dollars. The HTML report and full
JSON paths are retained under `artifacts/canonical-market-replay-v1`; hashes were
independently rechecked after execution. This is one local full replay, not a
cross-machine simulator equality claim.

The fixed selected50k reference actor has mean training-window return50.27%
versus holding56.45%; stocks54.08/58.15%, ETFs29.29/47.15%, mean maximum drawdown
−35.33%. CVX is positive overall and on short P&L, while XOM remains negative
overall despite positive short P&L. No policy promotion, checkpoint reselection,
new learning or development/test evaluation follows these engineering outcomes.
No Colab allocation or credit use occurred in this local replay.

## Full simulator cross-machine probe declaration

`configs/experiments/canonical_market_probe.json`, SHA
`b9bcb419386de24f18201e52cb84533232ba20970727eac975b8b48f59b7d249`,
declares a separate inference-only Colab engineering allocation. The previous
local replay protocol stays unchanged. The target is exact canonical result SHA
`b2c464c47f5d9c0f1520381d2fac72b0f944afdc6693997308dd080da6fb5b4a`
from the completed local26-symbol/19,136-decision replay: all rows, fills,
metrics, matched baselines and per-session direction attribution. No tolerances,
checkpoint selection or native-result acceptance changes are allowed.

`evaluation/canonical_window.py` calls the existing TradingEnv, simulator,
metrics, baselines and attribution directly, using the reference episode adapter.
It omits the generic RL PredictionStream and does not duplicate accounting rules.
A fixture matches the original evaluator's entire canonical result. A fresh-process
check confirms no Torch or SB3 import. The actual full worker will also be checked
locally before allocation, using the unchanged original full replay hash.

Only the fixed training windows, frozen reference actor/metadata, public source
modules and two engineering protocols are uploaded. No native model pickle,
validation/test episode, `.env` or credentials enter the payload. Its SHA and Git
revision are bound before allocation. `configs/reference-inference.lock` copies
five pins and hashes from the existing project lock: NumPy, Gymnasium,
cloudpickle, typing-extensions and Farama notifications. Colab's actual default
Python must be3.13 before setup; a pip-free venv uses that same interpreter. The kernel's existing pip must
support `--python`; it targets that isolated venv and installs only these
hash-checked binary inference dependencies. Torch/SB3 are absent from
the isolated environment; no model fitting is possible through them.

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.canonical_market_probe --prepare
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.canonical_market_probe \
  --cli-python /home/jassno/.local/share/uv/tools/google-colab-cli/bin/python
```

The one-time owned CPU allocation uses the existing CLI and shared ledger/lock.
Maximum30 minutes includes provisioning, dependency setup and retrieval; six
minutes are reserved for teardown. Incremental allowance0.1 units includes all
failures, idle time and downloads. Each projected next operation reserves teardown
before starting; aggregate150/130 and account150/170 limits remain unchanged.
Minute metering, bounded downloads and failure/interruption cleanup reuse the
already tested probe lifecycle. Original640-actor probe claims remain consumed;
this has its own persistent session claim and cannot restart automatically.

The worker retrieves both a compact metadata receipt and full canonical paths
compressed as gzip. Verification checks the compressed-file SHA and independently
decompresses at most128MiB to verify the exact complete-result SHA. Failures keep
retrieved evidence and stop the owned session. A pass proves this one reference
market replay across two machines, not training determinism, unseen-data quality,
policy promotion or permission to restart a failed native family.


### Full simulator probe outcome and setup correction

The [original failed attempt](../results/reference-inference-2026-11-02/market-probe-failure.json)
retains its consumed claim, source d849084 and local complete-path preflight.
Colab failed at plain `python -m venv`, exit1; child stdout/stderr were not
captured, so the exact cause remains unknown. Remote market replay did not start.
The CLI's return did not override missing-artifact rejection. Owned session
`rl-stocks-cpu-3cdf89d21cd1` was stopped/verified; official idle balance548.11,
zero rate/assignments. Conservative charge0.000525920 units, cumulative
40.122243104/150. No automatic retry occurred.

`runtime/colab/inference_setup.py` now creates the same-interpreter venv with
`--without-pip`, avoiding dependence on distribution-provided ensurepip. It checks
the existing kernel pip's `--python` capability before environment creation and
installs the five locked dependencies only into that venv. It does not install
another interpreter, change the base packages or install/upgrade the Colab CLI.
If the kernel pip is absent or incompatible, setup fails with its actual captured
error rather than guessing or installing a replacement automatically.

Child commands now capture bounded stdout/stderr, redact URLs and attach stage,
return code and diagnostics to a small independently retrievable failure receipt,
even if failure occurs before dependencies are available. A failure receipt skips
attempts to download nonexistent market paths and cannot pass acceptance. Local
checks create a real pip-free environment with exactly the current interpreter,
verify package installation targets, capture real child stderr and execute a
whole generated bootstrap failure without network or allocation. This correction
has not been validated on Colab. The v1 failed claim stays consumed; a separate
reviewed execution plan is required before another allocation. Complete simulator
cross-machine equality, new performance research and the broader goal remain open.
