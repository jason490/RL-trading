# Measured CPU comparison execution

The corrected Colab profile is accepted; see the [dated evidence](../results/recurrent-offpolicy-2026-10-30/README.md). Version43 prepares the fixed seed11 comparison of GRU-SAC, native SAC and RecurrentPPO. Implementation is available; no100k performance result has been produced by this change.

## Admission and duration

The execution-only [contract](../configs/experiments/recurrent_sac_cpu_screen.json) pins the accepted profile archive and its compute-source checksums. Admission physically rechecks that archive and requires the current learner, simulator and dataset implementation to match the profiled versions. The original two-hour projection remains recorded as failing for GRU; it is not rewritten.

| Role | Projected full allocation | Allocation ceiling |
| --- | --- | --- |
| GRU-SAC | 4.67 hours | 6 hours |
| Native SAC | 1.95 hours | 3 hours |
| RecurrentPPO | 1.21 hours | 2 hours |

These ceilings include setup, execution, validation, retrieval and shutdown. Each allocation reserves360 seconds for teardown. Concurrency remains configurable; all roles share centralized metering and the original two-unit study allowance, including both profile attempts. The cumulative project150/130 and account150/170 protections remain unchanged. No budget baseline is reset. The initial smoke allocation rule remains unchanged; these are subsequent measured performance allocations.

## Preparation

From a clean source checkout with the immutable dataset, portable starts, accepted profile archive and original shared ledger present:

```bash
PYTHONPATH=src python scripts/build_recurrent_screen_jobs.py --output .runtime/recurrent-screen-jobs.json
```

This only writes three tasks; it does not allocate or train. Run through the existing centralized fleet controller after its preflight. Inspect the actual default Colab Python before dependency installation or learning; the accepted profile used3.13.18. Do not install an older interpreter. Preserve the original shared `.runtime` ledger when exporting clean source.

## Results and failure handling

For compact local progress without model imports, cloud calls or dataset parsing:

```bash
PYTHONPATH=src python -m rl_stocks.runtime.colab.recurrent_screen_status \
  --root .runtime/recurrent-screen-source-v1 \
  --state .runtime/recurrent-screen-state-v1.json
```

Run this command from the current checkout; `--root` points to the captured
training source and its artifacts. Add `--charts artifacts/recurrent-screen-progress`
only when you want refreshed copies of the existing partial HTML curves.
The default is read-only. It reports meter age, shared project usage, recorded
phases, selected checkpoints and sparse provisional returns. Missing or incomplete
downloads remain labelled observations. Progress members are bounded and regular;
duplicates, oversized members and chart symlink destinations are rejected.
The command never establishes process liveness or full scientific acceptance.
Poll the existing controller process/job handle independently; a saved `running`
state alone is insufficient. During waits, reuse this summary and the same live
handle instead of repeatedly reading full reports or starting new workers.

The [research protocol](recurrent-sac-comparison.md) remains fixed:100k transitions, seed11, checkpoints every10k, unchanged features/rewards/normalization/data/costs and selection gates. Confirmation seeds23/37 are not exposed by this runner. They require a complete passing seed11 matrix and separate budget admission. The retired2024+ test stays closed.

Each role/seed consumes an immutable allocation claim. An execution timeout probes the same worker's export; it never restarts learning. A failed or interrupted fleet cannot automatically retry a role or continue a partial matrix. Budget stops require explicit review.

Final archives use an exact bounded allowlist before model loading. Full checkpoint and recorded-economic verification runs in a bounded300-second local subprocess, one verifier at a time, so the controller can continue minute metering. This performs inference and artifact checks only; learning runs exclusively in Colab. Accepted cells retain archives, checkpoints, reports, provenance and compact return curves. Every owned session must be stopped and termination verified, including failure paths. Operational failure is not a performance finding.

## Verification recorded with implementation

The focused recurrent comparison, remote dispatch, acceptance, artifact, job-plan and fleet suites passed100 tests. New checks cover immutable role/seed settings, original budget preservation, per-role claims, hostile/incomplete transport rejection before loading, and deferred serialized verification. A separate read-only physical recheck of the actual accepted profile archive passed for all three new deadlines; it still reports the original two-hour projection as failing. New modules pass Ruff; `plan.html` remains unchanged. Full100k saved-cell acceptance awaits actual Colab runs.

The later compact-status feature passed20 status/screen/fleet/budget tests and
Ruff, and was exercised against the running screen's downloaded snapshots.
Its fixtures cover read-only operation, incomplete archives, duplicate members,
exact optional chart copying and refusal to infer liveness or acceptance.
