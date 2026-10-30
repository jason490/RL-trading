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

Transient completion-marker transfer errors and final-download timeouts get at
most three attempts within the original execution/allocation window. A missing
marker is a normal not-ready response; unauthorized marker access fails closed.
These transfer retries never suppress an unreliable account meter or extend the
deadline. The first real three-CPU smoke run exposed a marker ReadTimeout: PPO
was accepted, SAC/RecurrentPPO checkpoints were salvaged, and all three sessions
were stopped and verified. The two unfinished acceptance tasks may be retried
only after that reviewed ordinary transport failure; no budget stop occurred.

## Optional tighter study allowance

`--study-unit-limit UNITS --study-start-units EXACT_LEDGER_VALUE` applies a
smaller allowance across successive phases while preserving the same project
ledger. Repeat the original exact baseline for smoke/screen/confirmation; do not
reset it at each wave. Guards project the next operation plus at least the
360-second cleanup window using the aggregate rate and unmetered time estimate.
An exhausted study allowance stops owned work, persists a budget-stop flag and
blocks automatic restart. Recovery accepts no budget overrides. The shared
150/130-unit limits remain authoritative. Registered allocation intent is also
cleaned up if a guard rejects a command immediately after admission.

## Delayed proof of an unstarted worker

A CLI connection can fail before bootstrap while the immediate authenticated
absence check is temporarily unavailable. A later confirmed HTTP404 for the
owned project directory now permits the existing single same-session retry
while waiting for export. Unknown/unreachable status or a present project
never permits a retry. A successful CLI execution is not retried just because
export is delayed. Both initial and deferred retries preserve the earlier
execution deadline and the original allocation deadline; budgets/ledger never
reset and no new session is allocated. One used retry prevents another.

The diversity smoke exposed this path: original CLI failed before bootstrap,
later authenticated project/job/log404 checks proved absence, and one manual
same-session retry resumed setup. Existing frozen workers were not modified.
The future-controller fix passes16 lifecycle/budget/deferred-retry checks with
fake processes, including unknown→absence, no second retry, success exclusion
and deadline preservation. It never treats observation timeout alone as proof
that training stopped.

## Opt-in longer allocation foundation

The budget primitive now supports a separately declared ceiling up to21,600
seconds (six hours). `FleetBudget` still defaults to7,200 seconds; existing task
plans1–42 retain that ceiling. No current task plan opts into the longer mode,
and it does not clear a ledger stop, provision hardware or authorize a study.

A future admitted long study must explicitly construct the budget with
`allocation_limit_seconds=21600` and pass one `allocation_plan_seconds` entry per
registered task to `start`. For the costed recurrent comparison, the proposed
entries are21,600/10,800/7,200 seconds. The complete sum bounds a campaign with
successive waves; each child retains its own deadline and teardown reserve.
Missing, oversized, noninteger or incomplete plans reject before ledger mutation.
Longer mode also requires persisted account protection of at least150 units.

Each declared duration slot can be claimed only once, including after its child
stops. Additional sessions cannot silently reuse completed slots as retries.
Legacy plans that do not declare durations retain their existing behavior. This
slot accounting limits registered tasks, not configurable global concurrency.

The controller's CLI execution preference follows a declared long child duration,
while preserving the retrieval reserve and any existing observation deadline.
It is no longer limited to a90-minute preference in that opt-in case. Legacy
execution preferences remain unchanged. Metering/ownership checks, single
aggregate account charging, project150/130 limits, account150/170 protection,
working-deadline teardown and verified cleanup are unchanged.

This is infrastructure support only. A long performance task still needs a
registered execution plan, accepted physical profiling, cost admission and
role/seed claims. The corrected20-minute profile subsequently completed after explicit user
approval under the [review process](recurrent-execution-review.md); it neither
used longer mode nor admitted100k training. No session was allocated while
implementing this foundation.
