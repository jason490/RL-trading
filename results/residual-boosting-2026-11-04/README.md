# Residual boosting feasibility completed — 2026-11-04

**Both fixed feature arms fail the original predictive gate, with zero passing
seeds. This candidate is closed. No trading-policy integration or promotion.**
The broader robust-return objective remains unachieved.

| Predictor | Mean internal-probe MSE | Worse than constant | Worse than own ridge | Mean daily rank IC |
| --- | ---: | ---: | ---: | ---: |
| Constant |0.005942941|—|—|—|
| Ridge masked |0.005785288|—|—|—|
| Ridge full |0.005782101|—|—|—|
| Residual boost masked |0.006890324|15.94%|19.10%|0.02159|
| Residual boost full |0.006838151|15.06%|18.26%|0.04613|

These are **forecast errors on2018–2020 internal training-era probe labels**,
not trading returns, an untouched test, or short-profitability evidence. Both
arms miss the required2% improvement over both constant and own ridge;0/3 seeds
pass. Both meet the rank-IC floor, but yearly noninferiority also fails, including
2020 for both arms. Positive ranking does not override the predefined failed MSE
gates or permit a ranking/target/parameter rescue of this closed candidate.

## Completed fits and independent acceptance

[Actual results](predictive-results.json), [independent restoration](acceptance.json)
and [family closure](family-closure.json) bind the exact completed-label payload,
normalizers, masked/full arms and seeds11/23/37. All six256-tree models completed
on owned Colab CPU/defaultPython3.13.18, clean sourceab24314. The completed run
fitted1,538 trees including its two discarded prefix trees; earlier failed
probe/setup evidence is retained separately below. Both ridge baselines use fit
rows only. No project-development or retired-test data was opened.

All111,696 saved boosted forecasts restore, including exact local repeated
prediction, immutable preprocessing, normal-equation checks for ridge, complete
finite stage-loss histories and independently recomputed metrics/gates. Recorded
maximum native/portable forecast error is0.0 for every full model. After duplicate
cleanup, the complete restoration passes again. Forty focused tests passed before
the completed allocation; actual artifact acceptance supplies additional evidence.

Component timing passed1310.720 projected remaining seconds versus7021.202
available, retaining2x margin/900-second reserve. Cold construction6.0954s was
already elapsed; prefix fit0.2528s, parity/export0.0145s, warm construction0.0201s.
The original deadline, full-fit budget and scientific gates were never relaxed.

Standalone interactive report:
`artifacts/residual-boosting-comparison-v1.html`, with probe MSE and all fit-stage
curves. [Report identity](report.json) binds both inputs and HTML checksum. Full
models, original transport, prefix/timing/progress/forecasts and source remain
under `artifacts/colab/rl-stocks-cpu-c621458a688c`.

## Final credits and cleanup

[Settled usage](usage-final.json): final attempt measured0.010000 units, estimated
0.009519861, charged0.010000 after idle settlement. Total study including both
prior failures0.016188555/0.25; original baseline40.196982365 unchanged. Project
**40.213170920/150**, remaining109.786829080; original130 teardown limit retained.
Fresh account**548.04**, rate0, assignments0. All three owned session shutdowns
verified; no recovery command is needed. Original claims and stop records remain.

[Cleanup audit](cleanup.json): six stage-model copies matched retained frozen
files byte-for-byte and were removed,1,708,566 bytes. Sixteen protected files
rehash unchanged; complete post-cleanup forecast restoration passes. No model,
source, forecast, failure, budget history or original transport evidence was lost.

Next review training-only forecasting/decision evidence for a distinct hypothesis.
Do not extend/reseed this candidate, transfer it into RL, or reinterpret its
positive rank IC as a passed gate.

## Historical bootstrap and timing attempts

# Initial bootstrap failure — 2026-11-04

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


## Saved forecast-error diagnosis

[Fixed audit](saved-error-diagnosis.json) covers all six accepted boosted models
on the pooled probe and each2018/2019/2020 year:24 cells, with source/ledger/
closure hashes unchanged. No model load, fitting, calibration, new dataset,
Colab allocation, gate change or family reopening. Fourteen focused identity/
validation checks pass; all24 actual residual-MSE identities reconcile.

Across seeds, masked/full MSE increases versus ridge are0.00110504/0.00105605.
The mean product of ridge's remaining error with the added correction is
negative (−0.00021680/−0.00021886). Thus the correction's added squared magnitude
and its adverse alignment both worsen pooled MSE; large magnitude is not the
only descriptive issue. These moments do not prove an optimizer cause.

Weighted session-common components account for91.92%/87.15% of that pooled
increase, while within-session components also worsen by0.00008928/0.00013568.
2019 improves for both arms, but2020 increases MSE0.00367276/0.00359821 versus
ridge and accounts for nearly all overall deterioration. No year is excluded
from gates or selected for a new calibration. Positive rank correlation therefore
does not establish robust absolute signal quality or profitable trading.

The failed candidate remains closed. A future hypothesis must be distinct and
preregistered; this audit does not authorize centering, shrinking, new targets,
extra stages or transfer of these models into RL.
