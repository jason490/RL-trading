# Training-only supervised signal feasibility

The saved-path diagnosis found material signal and participation losses after
cost add-back. A supervised forecast may expose a learnable signal more directly
than simultaneous PPO exposure/protection learning. This is a hypothesis, not a
profitable strategy or a decision to warm-start PPO.

## Target and information timing

The target is a gross log total return over 20 sessions. A decision uses only
its completed bar's causal market features. Entry is at the next open; exit is
at the twentieth session's close. The label becomes available only at that exit
bar's declared availability time.

A unit share bought at entry is not entitled to that morning's corporate actions,
because the simulator processes them before opening a position. Subsequent
splits adjust unit inventory and dividends add cash; dividends are not reinvested.
The unit convention allows fractional inventory and excludes fees, impact,
borrow, protective stops/targets and whole-share rounding. It measures an
underlying total-return forecast, not a tradable fill path, risk action, or oracle
exposure label. Learned policy protection remains a separate later requirement.

Targets may use later **training-era** outcomes as supervision once completed.
Those outcomes never become model inputs. Full-period declining-stock labels
from development validation or prospective reporting are never supervision,
inputs, rewards, sampling weights or selection criteria.

## Chronological folds and preprocessing

| Fold | Eligibility |
| --- | --- |
| Fit | Target completed and available by 2017-12-31 |
| Internal probe | Decision on/after 2018-01-01; target completed and available by 2020-12-31 |
| Project validation | 2021–2023 is not used by this feasibility study |
| Retired test | 2024–September 2026 remains isolated |

Windows crossing a fold boundary or episode end are discarded. Source episodes
already stop at known gaps and split boundaries. Earlier bars can provide causal
feature context, but future outcomes cannot cross the fold. The existing source
training purge remains, so the last probe label completes on 2020-12-02.

The internal probe is out of fit for these predictors. It has been part of older
RL training and is not new, independent market evidence. Overlapping targets and
correlated symbols also prevent treating rows as independent observations.

The source's pooled 2010–2020 normalizer is ignored. Means/scales are fitted on
fit-fold feature rows only and frozen for probe prediction. Both arms have ten
inputs: seven baseline indicators plus three context fields. `masked` replaces
the latter with zero; `full` retains them. Their first-seven fit statistics and
initial model tensors must match, with the same architecture and seed.

## Frozen feasibility protocol

[The protocol](../configs/experiments/supervised_signal.json) fixes:

- A fit-mean constant predictor and ridge regression with penalty 0.01.
- Two-layer 64-unit Tanh MLPs, fixed final 2,000 Adam updates, batch 256, learning
  rate 0.001, weight decay 0.0001, seeds 11/23/37 for both feature arms.
- No checkpoint selection or hyperparameter sweep on the probe.
- At least 2% mean MSE reduction against the constant and the same-arm ridge, at least two passing seeds, mean daily
  cross-sectional Spearman IC at least 0.02, and no annual MSE deterioration
  against the declared baselines in 2018, 2019 or 2020. Each passing seed must
  independently clear both overall MSE thresholds; yearly and IC gates use
  equal-weight seed means. Both feature arms are declared candidates. If more
  than one passes, lowest mean probe MSE ranks first, with protocol order for
  exact ties. Report the paired feature comparison regardless of the verdict.

These gates are predictive feasibility gates. They cannot meet the project's
stock/ETF return goal or establish profitable shorting. Failure stops this
family. Any later policy warm-start trial needs a separate matched protocol,
actual simulated returns, the four-component learned-action contract, stock/ETF
and drawdown gates, short-side reporting and prospective confirmation.

## Data preparation and status

```bash
PYTHONPATH=src .venv/bin/python scripts/preflight_supervised_signal.py \
  --output artifacts/supervised-signal/preflight.json
```

Preparation creates a checksummed, write-once `data/signals/ID/dataset.json`
containing only fit/probe feature rows, completed targets and fit preprocessing.
It has no raw OHLC or project-validation episodes. The preflight binds its
identity to the protocol, records fold hashes/counts and verifies immutable
readback. This labeled dataset can be uploaded to Colab without sending the
project validation partition or recomputing targets on another backend.

Tests cover target completion, fold purging, future perturbations outside the
horizon, separate episodes, entry-day dividend/split exclusion, subsequent actions,
delayed availability, fit-only preprocessing, and unused project-validation data.

The prepared data has 45,266 fit and 18,616 probe examples across 26 instruments.
No supervised model has been fitted yet. The planned one-CPU feasibility
allocation will use the existing metered fleet, original ledger, a one-unit
incremental study allowance and two-hour deadline. Actual Colab Python and tiny
finite-loss/save/load checks must pass before the fixed full fits. No dependency
or interpreter changes, orders, or new allocations occurred during preparation.
The retrospective universe and survivorship limitations remain unchanged.

## Implemented worker and frozen predictor

Version13 Colab plans bind the immutable signal payload and complete protocol
hash. The CPU worker runs two discarded prefix optimizer updates with finite
loss, input rejection and save/load checks, then reseeds all six fixed full fits.
The12002-step plan counts optimizer updates, not environment transitions. The
fit-only loss curve is recorded every200 updates; probe scores are computed only
at the fixed final model. Completed fits are checkpointed durably.

`inference.signal.SignalBundle` restores all six MLPs plus ridge/constant forecasts
with exact names/source contracts and frozen fit preprocessing. It has no trading
actions. Returned archives require complete seed sets, paired initial tensor
fingerprints, matched date/symbol panels, checksum integrity and independently
recomputed metrics/gates. Local post-download verification checks every forecast
against the immutable probe inputs: repeated CPU prediction is exact, cloud/CPU
predictions allow at most1e-6 absolute error, and metric tolerance is1e-10. The
boolean gate must agree with recomputation. No future labels are model inputs.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_signal_jobs.py \
  --output .runtime/signal-feasibility-jobs.json
# Execute with the existing centrally metered fleet, frozen source and original
# ledger; concurrency1, study-unit-limit1, study-start-units from that ledger.
PYTHONPATH=src .venv/bin/python scripts/verify_signal_runs.py \
  artifacts/colab/OWNED_SESSION/retrieved/artifacts \
  --output artifacts/supervised-signal/restoration.json
```

Thirty-six focused model, target, fleet and artifact tests pass without local
model fitting. They include predictor restoration, nonfinite/schema/checksum
rejection, fixed-plan checks and tampered-metric detection. The real Colab fits
remain pending; these tests do not establish forecast accuracy.
