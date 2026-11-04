# Residual boosting bootstrap failure — 2026-11-04

The initial claimed CPU attempt stopped during bootstrap installation. The model
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


## Reviewed continuation: duration admission stopped

[Reviewed duration evidence](reviewed-duration-stop.json) records the explicitly
reviewed session `rl-stocks-cpu-66edf80ae135`, source7b38242, actual default
Python3.13.18/CPU. Corrected bootstrap and dependency setup completed. Two ridge
baselines and the discarded two-tree residual prefix fitted; producer native/
portable parity passed before duration rejection. No full model, probe metric or
predictive gate completed. Original no-retry claims remain preserved.

The projection was9981.881 seconds versus7058.859 remaining. The original timer
combined cold construction/imports with fit/export/parity and scaled the entire
interval by tree count. Components were not separately recorded, so this receipt
does not assert which component dominated. The stop honored the original
allocation deadline; no duration or budget threshold was relaxed.

Future timing instrumentation separately records cold startup, prefix fitting,
prefix acceptance and warm construction. It scales fit/export/parity from two
to256 trees for six models, charges warm construction per model, preserves2x
margin plus900 seconds, and treats already-spent cold startup as elapsed time.
Scientific fits, targets, folds, seeds and gates are unchanged. Actual component
measurements/admission remain pending; this fix does not establish that the fits
will fit the deadline. Original reviewed admission now rejects the changed worker
bytes. Any additional allocation requires another explicit execution review.

Failure archive, portable prefix and timing record are retrieved; owned shutdown
is verified. This reviewed attempt charged0.003806218 units, measured debit0.
Combined original study usage is0.006188555/0.25, project40.203170920/150.
No automatic restart or trading-policy allocation occurred.
