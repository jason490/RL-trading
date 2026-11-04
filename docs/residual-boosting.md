# Ridge-residual boosting feasibility

This is a separate forecast-feasibility hypothesis, not a continuation of the
closed routed SAC, random-forest or supervised-MLP families. No trading model is
promoted and no allocation is admitted by the current implementation.

## Reason for the experiment

The completed signal study found that ridge reduced probe MSE about2.7% relative
to the constant, while its MLPs and later bagged forests failed their combined
predictive gates. The later routed SAC diagnosis also shows direction/exit P&L
weakness beyond increased transaction costs. These observations do not establish
that another forecaster will help trading. They motivate one limited question:
can a small nonlinear residual correction improve the existing linear forecast
on the same purged training-era task?

Gradient boosting builds an additive sequence of residual trees; bagged forests
average separately fitted trees. This is a concrete different fitting mechanism,
not evidence of stock-market alpha. The
[official GradientBoostingRegressor API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
describes shrinkage, estimator count, initialization and public tree estimators.
The [original Friedman paper](https://doi.org/10.1214/aos/1013203451) establishes
the stagewise function-approximation method, not success on these trading assets.

## Fixed numerical contract

[Protocol](../configs/experiments/residual_boosting.json) SHA256:
`7c9fa2818b68da79ffd0528be8496f4e392a5131caa0fd1fde209cf2cd169d79`.

- Reuse immutable completed-label payload
  `84b1e38843f8e623c3abbee3d1373c4e20556b39b85891ba8201c1ff54723235`.
- Ten existing features; original masked/full arms, seeds11/23/37. No extra
  indicators, targets, labels or universe search.
- Fit labels complete through2017; probe2018–2020, purged at boundaries. Fit
  preprocessing on fit rows only. Never read project development or retired test.
- Fit the existing ridge with penalty0.01 and unpenalized intercept. Fit trees to
  fit-row target minus fit-row ridge prediction. Predict ridge plus residual.
- GradientBoostingRegressor: squared error,256 stages, learning rate0.03,
  depth2, minimum leaf64, subsample0.8, all features, zero residual initialization.
  No early stopping, tree-count selection, warm start or parameter grid.
- Reuse the original predictive gates:2% mean MSE improvement over both constant
  and own ridge, at least two passing seeds, mean daily rank IC at least0.02,
  and yearly mean noninferiority for2018/2019/2020. Failure closes this candidate.

Forecast targets remain gross20-session unit log returns. They omit trading costs,
borrow, protective exits and whole shares. Passing predictive gates would not
establish profitable trading, short profitability or satisfy stock/ETF gates.
Any later integration needs a separate matched trading protocol retaining learned
protection and the original multi-seed/prospective requirements.

## Implemented components and verification

`training/supervised/boosting_spec.py` validates the fixed protocol and constructs
untrained estimators using the existing locked tree dependencies. It has no fit
entry point and does not change dependencies or install an interpreter.

`inference/boosting_arrays.py` stores ridge coefficients and tree arrays in a
bounded NPZ format without pickle/sklearn. Full models require256 depth-two trees;
the discarded-prefix format requires exactly two. Prediction preserves float64
ridge arithmetic and float32 tree comparisons, sums shrunk residual trees from
zero, then adds ridge. Arrays have immutable byte backing and tree mappings are
read-only. Loading rejects extra/duplicate/oversized members, invalid topology,
nonfinite parameters and wrong geometry. Existing forest interfaces are unchanged.

Six focused boosting/forest-array checks pass without local fitting. They cover
hand-computed additive predictions, float32 threshold rounding, exact repeats,
normalization-independent predictor inputs, save/load, model immutability,
malformed topology/archives and protocol/constructor constraints. This is local
engineering evidence. Native sklearn prediction parity on actual fitted trees
remains required in Colab and after retrieval.

## Pending runtime and budget admission

Implement the Colab-only fit worker, fit-only preprocessing, immutable bundle,
complete forecast export and independent metric/gate recomputation next. Require
finite prefix loss, actual native/portable parity and a conservative timing
projection before the six fixed fits. Bind all source/data/dependency identities.
Inspect the actual default Colab kernel before dependency setup; never install an
older interpreter. Reuse the official CLI and existing hash-locked dependencies.

The proposed incremental allowance is0.25 units, one CPU allocation at most7200s,
no automatic retry. This is a new bounded study allowance, never a reset of the
original project ledger or previous charges. Project150/teardown130 and account
floor150/teardown170 remain binding; meter each minute, count setup/idle/downloads/
failures, retrieve durable artifacts and verify owned shutdown on every exit.
At0.08/hour plus0.01 margin, two hours project0.18 units; admission must use the
actual reported rate and stop if the complete operation will not fit. Existing
project usage remains40.196982365 units; no new allocation or optimization occurred.

## Worker and restoration implemented — 2026-11-04

`training/supervised/boosting_runner.py:run` now implements the fixed fit. It
requires the actual Colab CPU/kernel guard and owned-worker marker before reading
inputs, creating output or fitting. The caller must provide the bounded remaining
allocation time. It validates the immutable labeled payload and locked tree
versions; the existing fit-fold normalizers remain unchanged.

The discarded two-tree prefix uses fit rows and checks finite training loss,
export and native/portable forecast agreement. Its measured fit/export/prediction
time projects six256-stage fits with a2x margin plus900 seconds for remaining
work/retrieval. Full fitting is refused if that projection exceeds the remaining
allocation; a900-second reserve is checked between fixed fits. These worker
checks supplement the future controller deadline and metering, not replace them.
Fit-only loss curves and completed-model snapshots are persisted after each fit.
The total is1,538 fitted trees, including the discarded prefix.

`inference/boosting_signal.py` restores a checksummed forecast-only bundle with
all six fixed models and fit-only preprocessing. It supplies constant/ridge
baselines and boosted forecasts; it has no trading actions. Native/portable and
frozen repeated predictions are checked for every probe row at absolute1e-10
agreement, with exact local repeated predictions.

`training/supervised/boosting_artifacts.py:verify` independently restores every
forecast, binds labels/dates/symbols to the immutable payload, checks model ridge
coefficients against the fit-fold normal equations, and recomputes all original
metrics and gates. Normal-equation checking does not fit a local estimator.
The verifier rejects missing models, changed normalization/dependencies, altered
labels/predictions, false scope flags and fabricated success. It verifies retained
artifact hashes remain unchanged during restoration.

Ten focused worker, bundle, boosting-array and shared forest-array checks pass
without fitting locally. Tests include fixed-model export contracts, pre-I/O
local-fitting rejection, forecast restoration and tampered gate/label rejection.
Actual native sklearn parity on fitted trees remains pending Colab acceptance.
The runtime is not yet registered: next implement one-time ownership, exact job
and payload/source binding, actual default-kernel preflight, shared-ledger budget
admission, bounded transport and guaranteed owned cleanup. No session has been
allocated and no new predictive or trading result exists.

## Metered runtime integration — 2026-11-04

The dedicated version49 CPU job is now registered. Its exact plan binds the
original completed-label payload and fixed protocol. One-time allocation
ownership and a persistent0.25-unit study baseline live under `.runtime`;
the original cumulative project ledger is preserved. An existing claim rejects
all automatic repeats, including attempts to reuse the same owned session ID.

The controller inspects the default kernel before installing dependencies,
passes the original allocation deadline into the worker, enforces a900-second
reserve and samples shared consumption each minute. Full execution is never
retried after a lost CLI reply. Five-minute uniquely named checkpoint retrieval
preserves the last valid archive. Complete final transport is allowlisted and
bounded before extraction; it binds the plan, clean captured source and actual
CPU/kernel metadata. The owned runtime is stopped and shutdown verified before
local forecast restoration. Post-stop verification is bounded to1800 seconds;
failure retains source artifacts and cannot restart training. A failed predictive
gate writes a persistent family closure and permits no automatic policy training.

From a clean committed repository, reuse the installed official CLI:

```bash
MPLCONFIGDIR=$PWD/.runtime/matplotlib PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/run_residual_boosting.py \
  --cli-python /home/jassno/.local/share/uv/tools/google-colab-cli/bin/python
```

State: `.runtime/residual-boosting-fleet-v1.json`; baseline/claim:
`.runtime/residual-boosting-budget-v1.json` and
`.runtime/residual-boosting-allocation-v1.json`. Do not delete them to rerun.
Source/model/forecast artifacts stay under `artifacts/colab/OWNED_SESSION`;
independent acceptance is `artifacts/residual-boosting-restoration-v1.json`.
Use the existing fleet recovery command with the exact recorded fleet/state if
cleanup fails; it stops only owned sessions and performs no learning.

Twenty-seven focused controller/model tests pass, including unsupported-kernel
shutdown before installation, metering-failure cleanup, nonrepeating claims,
exact transport, immutable saved forecasts, false-success/label rejection and
finite fixed loss histories. Existing task-reference lifecycle tests still pass.
Actual Colab prefix timing/parity and the six forecasts remain pending; this
integration is not evidence of improved predictive or trading performance.


## Initial bootstrap failure and timeout fix — 2026-11-04

[Actual failure](../results/residual-boosting-2026-11-04/README.md): the first
claimed session stopped before any fitting because official CLI `install`
inherited a10-second kernel-client timeout. The controller verified shutdown;
fresh account is idle. Original claim/baseline/failure remain retained; no
automatic retry. Project40.199364701/150, original study charge0.002382336/0.25.

For this job, bootstrap now uses existing official CLI `exec --timeout180` to
run `scripts/install_colab_bootstrap.py`, whose default-kernel pip subprocess
is bounded to150 seconds. The actual kernel is inspected first; no local CLI
upgrade or interpreter replacement occurs. Bootstrap failure immediately stops
without launching training. Further allocation requires a separately recorded
execution review; deleting the original claim/state is forbidden.

## Reviewed setup-only continuation — 2026-11-04

[Execution review](../configs/experiments/residual_boosting_execution_review.json)
SHA `d5c2985a8107f835a7d7d5f28d3c3b06251c6396dc5a2d2ae38b7d6d6ad88454`
permits exactly one additional allocation after the failed pre-worker setup.
This explicitly changes the earlier one-allocation execution limit to at most
two total allocations including that failure; the scientific protocol stays
unchanged. The review binds the original failure/claim/state, stopped owned
session, preserved failure charges, exact original0.25-unit baseline, bounded
bootstrap script and unchanged scientific source bytes from the failed attempt.
No prior forecast result exists and no model family was reopened.

Version50 uses a separate reviewed claim/state; version49 remains blocked by its
original claim. The new claim is also single-use, with no automatic repeat.
Admission rejects source drift, reset budget baselines, missing failure charges,
unresolved ownership, family closure or an already claimed continuation. All
project150/130 and account150/170 limits, deadlines, kernel checks, metering,
retrieval, shutdown and post-stop acceptance remain binding.

Run the explicitly reviewed route from committed source:

```bash
MPLCONFIGDIR=$PWD/.runtime/matplotlib PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/run_residual_boosting.py --reviewed-bootstrap \
  --cli-python /home/jassno/.local/share/uv/tools/google-colab-cli/bin/python
```

Reviewed state: `.runtime/residual-boosting-reviewed-fleet-v1.json`; reviewed
claim: `.runtime/residual-boosting-reviewed-allocation-v1.json`. The original
baseline, claim, state and failure evidence remain. Thirty-one focused checks
pass, including both version49/50 transport and lifecycle, rejection of reset
baselines/source drift/reused review claims, and existing cleanup protections.
Actual read-only admission passes against the retained original failure and
current ledger. This is runtime admission evidence, not a predictive result.

## Reviewed duration stop and component timing — 2026-11-04

The reviewed bootstrap succeeded on actualPython3.13.18/CPU. Both ridge baselines
and the discarded two-tree prefix fitted and producer native/portable parity
passed. Full fitting stopped before any six-model forecast/probe metrics because
9981.881 projected seconds exceeded7058.859 remaining. The controller retrieved
the complete failure archive and verified owned shutdown.
[Evidence](../results/residual-boosting-2026-11-04/reviewed-duration-stop.json).

The original aggregate timer included constructor/import startup and scaled it
with tree count. Because separate components were not saved, the remote dominant
cost cannot be identified after the fact. Future instrumentation records cold
construction, fitting, acceptance/export and warm construction individually.
Only already-spent cold construction is excluded from remaining-work projection;
fit/acceptance still scale2→256 trees across six models, warm construction is
charged per model, and2x margin plus900 seconds remain. Invalid/zero/nonfinite
measurements and overflow reject admission. This corrects measurement granularity
without changing a deadline, model, fit count, target, seed or predictive gate.

Twenty-two timing/worker/predictor/runtime checks pass. Retrieved prefix arrays
restore with valid topology and finite predictions; this is not independent
reproduction of the producer's native sklearn prefix. Original archives, prefix,
state, claims and charges remain retained. Combined study usage0.006188555/0.25,
project40.203170920/150. No further allocation or automatic repeat occurred.
The previous execution review rejects the changed worker; another explicit
execution review with actual component timing is required before new fitting.
