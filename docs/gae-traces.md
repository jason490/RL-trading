# Fixed recurrent GAE trace comparison

The [preregistration](../configs/experiments/gae_trace_preregistration.json) fixes
one candidate: native RecurrentPPO with GAE lambda1 versus the original lambda0.95
control. Runner request contracts, shared starts and pre-update rollout checks are
implemented, including version35 Colab tasks and archive/budget checks. Physical
learned smoke acceptance remains pending; no study training result exists.

## Rationale and limits

The [saved value audit](../results/recurrent-values-2026-10-25/README.md) found
selected/final agreement differs between observed discounted rewards and
self-bootstrapped lambda targets. It does not establish value bias or a cause of
development deterioration. The [GAE paper](https://arxiv.org/abs/1506.02438) motivates
an estimator bias/variance trade-off; its robotics results are not trading evidence.
The maintained [installed-release implementation](https://sb3-contrib.readthedocs.io/en/v2.7.0/modules/ppo_recurrent.html)
already exposes `gae_lambda`; no custom optimizer is proposed.

At lambda1, intermediate critic terms telescope within each uninterrupted segment.
The current-state baseline and endpoint value bootstrap remain. This can increase
advantage/target variance. Finite horizons, clipping, repeated PPO optimization
and stochastic policy paths still limit theoretical interpretation. No claim of
unbiased expected-return estimates, exact terminal-wealth optimization or improved
returns follows from this identity. Gamma remains0.99 and maximum holding20.

Benchmark coefficient-one subtraction failed its separate feasibility gate and
remains closed. This candidate retains raw net-log-equity rewards and introduces
no benchmark term. Every previously failed family remains closed.

## Matched experiment

Both arms retain the same immutable111-symbol training history and original26
repeated-development symbols, seven market fields/21 total inputs, training-only
preprocessing, full64 MLP/LSTM, separate actor/critic memory, native Gaussian
four-component actions, costs, stops/zones, whole shares, exposure cap and capital.
Training exposure remains unrestricted; evaluation uses the existing unit-magnitude
contract. Original group/deck/window sampling,512-session episodes, rollout1000,
batch50,10 epochs, learning rate0.0003, Adam epsilon1e−5 and zero decay remain.
Only `gae_lambda` differs. Selection stays equal-weight raw development net return.

Use100k seed11 steps,10k curves and independent actual-final post-update evaluation.
Smokes use512 transitions, rollout128/batch64, episode128 and256 cadence, while
retaining the full64 network and61,513 parameters. Both phases must use the original
full-size per-seed untrained tensors/CPU RNG, including smokes. Do not substitute
the older capacity-study32-wide smoke initializer. Verify exact both-arm policy,
empty Adam and all Torch/NumPy/Python sampling starts before learning.

The fresh control must reproduce the pinned historical checkpoint means within
0.01 percentage points and selected/final per-symbol returns within0.05 points.
Mismatch invalidates interpretation/promotion without widened tolerances or retry.
Candidate selected/final gain must each reach2pp; final stock/ETF returns cannot
worsen, additional mean drawdown cannot exceed2pp, and stock/ETF holding excess
and median-stock excess must reach−2pp in both phases. Only a passing screen
permits fixed11/23/37 confirmation, followed by prospective evidence. Declining
cohorts stay reporting-only with positive total/short P&L required for short claims.

## Required acceptance and operation

Before optimization in Colab, reconstruct actual buffer returns/advantages for
every rollout, respecting episode boundaries, already-adjusted timeout rewards
and last-value bootstrap. The candidate's lambda1 returns must match independent
segmented discounted sums. Retain finite-error summaries in a compact per-rollout
sidecar. Validate actual parameters/defaults, model/Adam/checkpoint state, source,
normalization, sampling/execution and at least78 frozen recurrent predictions
per arm after smokes and screens. Serialization fixtures are not learned acceptance.

Use the existing central CPU fleet, original cumulative150-unit ledger/130-unit
teardown, account floor150/teardown170 and a persistent two-unit GAE study baseline
across controllers/retries. Inspect each actual default Colab Python before
synchronizing dependencies. Minute aggregate metering, inclusive7200-second
allocations, smoke-backed projection with1200-second reserve, durable retrieval
and verified owned shutdown remain mandatory. No automatic restart after budget
or integrity stops; never stop unrelated sessions.

Failure closes this fixed comparison without lambda/gamma/rollout/minibatch/epoch/
learning-rate/model/reward/feature/step grids, schedules, extension or confirmation.
Untrained model construction and serialization fixtures are covered; no optimizer
step, study training, reward change or Colab allocation has occurred. Physical learned restoration and actual Colab smoke acceptance are next.

## Shared-start and rollout-check implementation

[Implementation evidence](../results/gae-traces-2026-10-25/README.md) records61
focused passing checks. Original full64 shared tensors/RNG, empty coupled Adam
and effective defaults are validated; both arms differ only in lambda. Invalid
fresh settings reject before tensor replacement. Model/optimizer/RNG records and
real save/load fixtures are covered without optimization.

The standalone `GaeRolloutAudit` callback retains actual float32 buffers plus
raw rewards/timeout values/done masks as compressed per-rollout NPZ files. It
checks pre-update epoch/transition order, timeout correction, GAE reconstruction
and independently segmented lambda1 returns, with fixed absolute2e−6/relative2e−5
tolerances. It accepts continuation at rollout boundaries and rejects corrupt
math/masks/dtypes. Fixture calls preserve model/Adam/state/RNG and forbid learning.
The runner now attaches the callback for registered GAE requests; these are not
learned Colab results. Job/fleet/archive and physical learned acceptance remain next.

### Runner integration

Registered requests now validate algorithm, exact dataset/preprocessing, all fixed
parameters, typed seed/phase/steps/cadence, CPU device, reward/exposure/observation/
sampling and compact single-milestone contracts before environment/model creation.
The runner restores the common initialization, retains its checksummed NPZ, records
actual initial/selected/final model and Adam evidence, and attaches the rollout
checker beside the unchanged validation callback. Final metadata requires complete
per-rollout evidence. Other study dispatch remains unchanged.

On target arithmetic failure, the offending arrays are durably retained with a
hash and pre-update failure marker before the exception propagates. This is failure
evidence, not an accepted training artifact.78 focused initialization, request,
actual-buffer, serialization and prior shared/VIX regressions pass without
optimization; task/budget/archive and actual learned acceptance remain pending.

### Version35 task, budget and archive integration

Prepare two CPU smoke tasks without allocation:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_gae_jobs.py \
  --smoke --output .runtime/gae-smoke-jobs.json
```

The fleet and remote worker validate exact payload/initializer contracts. A
dedicated GAE fleet automatically preserves `.runtime/gae-study-budget.json`
beside the resolved original usage ledger, retaining the same two-unit baseline
across smokes/screens/retries. Original project150/130 and account150/170 limits
still apply. Mixed-study fleets cannot override this baseline.

Full plans require a checksummed `.runtime/gae-smoke-acceptance.json` with actual
saved-model restoration, rollout-array verification, at least78 frozen predictions
per arm, exact common starts and verified owned cleanup. Deadline projection uses
twice the historical control duration times the slowest relative smoke duration,
plus1200 seconds reserve; above7200 seconds rejects. Root preflight rechecks the
actual acceptance file, not only its profile declaration.

Retrieved archives reconstruct every actual rollout array, verify timeout and
GAE/MC arithmetic, checksums, file set, counters, phase and complete summary.
Missing/extra arrays, failure markers, altered summaries, changed typed lambda or
corruption with updated checksums reject. Initial/selected/final model/Adam,
initializer bytes, actual execution/training-year sampling,26-symbol development
panel and frozen weights/preprocessing also validate. Physical model restoration
and full report accounting remain additional acceptance requirements.

113 focused task/archive/model/runner/VIX/fleet checks and five further smoke-gate
rejection cases pass without training; new modules/scripts/tests pass Ruff.
Static/archive fixtures are not learned Colab evidence. Physical learned/frozen
restoration and smoke acceptance tooling remain next; no session allocated.
