# Numerical performance diagnostics

Diagnose slow research workloads using retained weights and compact timing
records before allocating different hardware. Inference-only local profiling is
allowed; learning, optimizer steps and backward/update benchmarks stay in Colab.
Keep financial performance gates separate from throughput measurements.

## Subnormal audit

`scripts/diagnose_regularization_numerics.py` verifies source model checksums,
restores complete/partial SAC checkpoints and counts zero/nonfinite/subnormal
policy and Adam-moment values using each tensor dtype's own range. It performs
an isolated one-thread actor/critic/target forward comparison with CPU
flush-denormal mode off/on. Fixed synthetic observations do not open any data
partition. No optimizer or normalization updates are performed.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
 .venv/bin/python scripts/diagnose_regularization_numerics.py \
 --output /tmp/regularization-numerics.json
```

Use a new output path. Warmup10, three100-forward blocks per mode, batch256×21;
report medians, output equality/difference, unchanged weights/update counters,
source hashes and exact versions. Run in a separate process so arithmetic mode
changes do not affect other work. Performance timings are host-specific; tests
check numerical boundaries and file/state integrity, not a speed ratio.

## Observed evidence and limits

Retained regularized80k policy contains5942/29644 subnormal tensor elements.
On this local host its forward pass accelerates14.10× with flush mode; the
control100k has no subnormal policy values and changes1.02×. Outputs on the one
fixed batch are bit-identical. This does not guarantee equality over every input,
training gradient or optimizer trajectory. Adam states were counted, not updated.
Remote checkpoint timings include learning/evaluation/export and different
hosts, so they do not isolate a training slowdown mechanism.

No arithmetic setting has been applied to training. A future change would need
an explicitly declared numerical experiment, real Colab acceptance and preserved
source/dependency identities. Faster execution does not establish stronger
returns or cure the incomplete regularization screen. New allocations remain
stopped after its metering failure; retain conservative ledger and require
explicit reviewed authorization before restart.
[Actual results and provenance](../results/regularization-numerics-2026-10-21/README.md).
