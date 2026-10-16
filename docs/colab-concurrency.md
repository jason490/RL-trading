# Concurrent Colab sessions and the shared budget

## Policy and ownership

Concurrent sessions are authorized. There is no fixed two-session or CPU-only
limit: `--concurrency` is a positive configurable maximum, and each task declares
its validated runtime plan. The installed project plans currently support CPU,
T4 and A100; backend availability and account limits can reject an allocation.
Hardware must suit the workload and be justified by profiling/cost. The current
small-network algorithm comparison intentionally uses CPU for every arm.

One coordinator owns the complete pool. It holds `.runtime/allocation.lock` to
exclude another coordinator or the legacy serial runner. This lock does not
limit the pool to one VM. Do not launch multiple legacy `ColabJob` processes
against the same ledger: their single-session accounting is not a pool monitor.
The official CLI installation, authentication and per-session names are reused.

## Metering without double-counting

The cumulative `.runtime/usage-ledger.json` is preserved. A pool adds one `fleet`
attempt containing a `children` map of every intended session. Intent and a
unique ID are persisted before allocation. One central loop reads account usage
every minute, after each serially admitted allocation, and at final shutdown;
uploads, installs, executions and downloads can overlap. It also verifies the
backend session list contains no unrelated assignments. Account rate and balance
are never printed through raw authenticated logs.

The account's overlapping consumption is charged once per fleet, rather than
once per child. As with serial attempts, the fleet uses the larger of its
conservative rate estimate and measured account debit. Across observation gaps,
it uses the larger prior/current aggregate rate plus the existing safety margin.
Per-child records report ownership, deadlines, artifacts and cleanup; they do not
pretend to allocate an exact portion of the shared account debit to each GPU.
Historical serial attempts keep their original records and charges.

Every runtime shares the **150-unit cumulative project ceiling** and **130-unit
working stop**. Guards project the larger of the next operation and the 360-second fleet
cleanup window using the whole active account rate, check remaining account balance, and enforce each child's two-hour deadline
including allocation, installation, execution and retrieval. Long execution
leaves 600 seconds for final downloads and shutdown. A campaign can contain
successive waves; this does not extend an individual allocation's deadline.
A budget/deadline/metering stop cancels pending work and stops all owned sessions.
The persistent stop flag prevents automatic restart.

## Run the controlled CPU comparison

Check the current Colab default Python before dependency changes or training. The
reviewed environment is 3.13.16 locally and on inspected CPU kernels; every remote
bootstrap rejects a changed interpreter family. Use a clean committed source tree
for research. Payloads are prepared before allocation under separate
`.runtime/payloads/SESSION/` directories, so job plans and datasets cannot overwrite
each other. All experiment arms receive the same committed implementation.

First build and run the lower-credit concurrent smoke preflight:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_stability_jobs.py --smoke --output .runtime/stability-smoke-jobs.json
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.fleet \
  --jobs .runtime/stability-smoke-jobs.json --concurrency 3 \
  --cli-python /path/to/existing-colab-cli/python \
  --state-output .runtime/stability-smoke-state.json
```

This creates independent CPU smoke sessions for SAC (1,024 steps), PPO (512) and
RecurrentPPO (512). It validates remote Python, runtime identity, complete job
counts and bundle checksums, retrieves artifacts and verifies every shutdown.
Run the fixed research cells only after that preflight passes:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_stability_jobs.py --output .runtime/stability-research-jobs.json
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.fleet \
  --jobs .runtime/stability-research-jobs.json --concurrency 3 \
  --cli-python /path/to/existing-colab-cli/python \
  --state-output .runtime/algorithm-stability-state.json
```

The number 3 is an example pool setting, not a hard cap. Resource rejection
enters owned cleanup; review the failure and completed cells before a bounded
retry with a suitable setting. Never automatically retry a budget/metering stop.
The fixed [algorithm protocol](../configs/experiments/algorithm_stability.json)
controls the nine research cells; concurrency does not expand the search budget.

## Generic task contract

`--jobs` reads a nonempty JSON list. Each entry has exactly `id`, `dataset` and
`plan`. IDs are unique safe names; the dataset directory must contain its immutable
`dataset.json`; plans must pass `validate_plan()`. Construct plans using
`job_plan()`, `study_plan()`, `learning_plan()`, `stability_plan()` or
`single_smoke_plan()` in `runtime/colab/jobs.py`. Mixed hardware is supported by
the coordinator. A100 smoke-only allocation is rejected, and CPU-first preflight
still precedes expensive research. The algorithm stability recipe itself stays
CPU-only to keep that comparison controlled.

Private per-session logs are under `.runtime/SESSION/`. Checkpoints and final
archives go to separate `artifacts/colab/SESSION/` directories. Completed roots
are recorded by task ID in the requested state file. A CLI execution-stream loss
never authorizes duplicate execution of a started worker; the coordinator waits
for an atomic export marker within the existing deadline. Only a confirmed
missing worker directory allows one prestart retry. Final downloads have three
bounded attempts, and periodic checkpoint downloads run independently while the
central meter continues polling.

## Interrupt, failure and recovery

A handled interrupt or any failed task cancels queued allocations, retrieves the
latest checkpoint when budget/metering permit, and sends stop commands to every
owned runtime. Stop/verification attempts are bounded and run concurrently.
Other account assignments are never stopped. Cleanup failure persists unresolved
IDs and prints each session's exact stop command.

The ledger stores a single fleet ID and all child IDs. To recover a stranded pool,
use the fleet ID and its original state path:

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.fleet \
  --recover rl-stocks-fleet-EXACT_ID \
  --cli-python /path/to/existing-colab-cli/python \
  --state-output .runtime/ORIGINAL_STATE.json
```

Recovery acquires exclusive project ownership, validates the recorded fleet,
concurrently stops only unresolved child sessions, verifies absence, reconciles
the shared ledger and preserves any budget stop. It never allocates or trains.
Keep CLI authentication/session state private and retain it until cleanup is
verified. A process already running the legacy runner retains its loaded behavior.

Mocked acceptance covers more than two mixed-hardware sessions, overlapping
account debits counted once, per-child/global stops, immutable payload separation,
stream-independent progress, owned interruption, metering outage, cleanup failure
and recovery, and preservation of unrelated sessions. Actual remote concurrency
must be reported separately after the smoke run; mocked acceptance is not proof
of an account's permitted runtime count.
