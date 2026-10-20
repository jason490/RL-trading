# SAC execution-state research

The [saved SAC audit](../results/sac-diagnostic-2026-10-20/README.md) restores
three existing100k models and checks six selected/final reports without opening
any dataset, fitting or allocating Colab. The actual model input size is12,
confirming missing execution-state fields; metadata absence alone is not the
proof. Source model/report hashes remain unchanged. Daily close fractions do
not establish directional profitability, and missing corporate-action bars
are not fabricated for attribution.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python scripts/diagnose_sac.py \
  --output artifacts/sac-diagnostic/new-summary.json
```

## Fixed next comparison

[Protocol](../configs/experiments/sac_execution_state.json) declares SAC with
legacy12 inputs, masked21 inputs and full21 inputs on the same immutable base
train/validation dataset. Masked/full have identical network capacity; the nine
additional fields are zeros versus actual causal held execution state. These
include held protection geometry, partial-target status, previous risk actions
and share-unit scale. The legacy arm controls the previous information contract;
prior100k results are not a matched50k control.

Use standard native SAC initialization,64x64 networks, learning rate0.0003,
discount0.99, replay100k, batch256, warmup1000, one gradient update per step,
tau0.005 and automatic entropy. No reward shaping, universe expansion or feature
addition is combined with this change. Simulation and original random-start
semantics remain identical across arms. Record10k validation curves, selected
and final models and replay restoration limits.

Three1024-transition Colab CPU smokes precede three50k seed11 screens. Smoke
quality never selects a configuration. Full state must beat **both** controls
by2pp at selected and final checkpoints, avoid final stock/ETF deterioration
against either control and keep relative final drawdown within2pp. Absolute
stock/ETF excess and median stock excess must each be at least−2pp in both
phases. Failed screen stops this configuration; no extra steps or grid.

A passing screen permits fixed seeds23/37 with controls and the declared
multi-seed targets. A selected validation peak cannot complete the objective:
repeated development evidence needs frozen prospective confirmation. New runs
must reconcile directional accounting; positive net short P&L alone is not
positive total profit. Future falling-stock cohorts stay reporting-only.

## Runtime and current readiness

Use existing Colab CPU/fleet tooling with original150-unit shared ceiling,
130-unit teardown threshold and bounded2-hour owned allocations. The complete
study has a2-unit incremental allowance, including setup, idle, retries and
retrieval. Check current default Colab Python before dependency sync. No CLI
reinstallation, local fitting or broker orders. Preserve all unsuccessful
models/evidence and verify publication/shutdown before duplicate cleanup.

Current evidence: diagnostic passed, protocol fixed, ten focused saved-path and
participation tests pass. Job/worker/artifact integration and actual Colab smoke
acceptance remain pending. No new allocation or performance result. Cumulative
project usage remains35.917535/150; the goal is unachieved.

Version16 integration now verifies fixed protocol/data identity before allocation
and again in the worker. Metadata/frozen bundles carry the protocol hash; archive
checks cover SAC parameters, initialization, observation profile and complete
selected/final checkpoint curves. All three immutable smoke task payloads pass
local validation. Actual Colab acceptance is next, not implied by fixtures.
