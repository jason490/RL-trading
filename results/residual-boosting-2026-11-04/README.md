# Residual boosting bootstrap failure — 2026-11-04

The only claimed CPU attempt stopped during bootstrap installation. The model
worker never started: no ridge/tree fitting, prefix parity or predictive scoring
occurred. This supplies no model-performance result.

[Failure receipt](bootstrap-failure.json) binds the original controller state,
allocation claim, budget baseline, ledger and installer traceback. Installed
official CLI0.7.4 `install` calls `run_automation` without an execution timeout;
`runtime.execute_code` consequently leaves the kernel-client10-second default.
The traceback records `TimeoutError: Timeout waiting for reply`. The local
controller's180-second process allowance did not change that inner timeout.

The owned session `rl-stocks-cpu-545b22d6c5eb` is stopped and shutdown verified.
Fresh account548.05, rate0, active assignments0. Conservative project usage is
40.199364701/150; this attempt charged0.002382336 units (measured debit0).
Original0.25-unit study baseline40.196982365 remains unchanged; claims/failures/
charges are retained and no automatic restart occurred. No cleanup recovery is
required. No partial outputs or learned artifacts need cleanup; the source upload
and original bootstrap logs remain as provenance.

The job's bootstrap now uses existing official CLI `exec --timeout180` with an
installer subprocess bounded to150 seconds. A failed bootstrap stops without
repeating it or launching the worker. No CLI installation/upgrade, interpreter
replacement, scientific protocol change or allocation accompanies this fix.
Any further allocation needs a separately recorded execution review preserving
this failed claim and the same original project/study charges; the original
one-allocation/no-automatic-retry contract is not silently reset.
