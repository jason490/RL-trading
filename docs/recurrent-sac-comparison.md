# Recurrent SAC development comparison

## Status and purpose

The replay-only and fresh-environment Colab smokes passed. They validate learning,
restoration and reporting mechanics; they do not establish trading quality.
The next performance hypothesis is that independent observation memory plus
SAC replay improves both checkpoint selection and end-of-run performance.

The fixed source of truth is
[recurrent_sac_comparison.json](../configs/experiments/recurrent_sac_comparison.json).
Its canonical digest is
`58351c3c62f4e65fb59c49e29ab87edb3226ca17445b155d8268a885a6b9c3eb`.
The protocol validator, timing projections, post-update checkpoint helper and
synthetic gate tests are implemented. Portable initialization assets, an actual
Colab profiling worker, fleet integration and physical artifact acceptance remain
required before performance allocation. No full comparison has run yet.

## Fixed data and controls

Use immutable dataset
`cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b`:
111 training symbols during 2010–2020 and the original 26-symbol development
panel during 2021–2023. Keep the original training-only normalizer frozen.
Retired 2024 onward data remains closed. Repeated development use, survivor-only
coverage, correlated symbols and revised free history limit generalization claims.

| Role | Model | Observation information | Update recipe |
| --- | --- | --- | --- |
| Native control | SB3 SAC, MLP64×64 | Full21 | Batch256, 1,000 warmup, one update per subsequent transition |
| Strong package reference | SB3 RecurrentPPO, separate LSTM64 | 21 dimensions with nine execution coordinates masked | Rollout1,000, batch50, ten epochs |
| Candidate | GRU-SAC, independent actor/twin-critic memory64 | Full21 | Burn-in16, unroll16, eight sampled windows, 1,000 warmup, one update per subsequent transition |

This compares complete algorithm packages. Input information, model capacity,
batch sizes and on/off-policy updates differ. Native SAC is the input-matched
control; the PPO reference is an additional strong package comparison. A gain
cannot establish that memory alone caused it.

Preserve the four actions, net-log-equity reward, costs, whole shares, $100,000
starting equity, exposure cap and 20-session holding limit. Training uses the
registered stock/ETF cycle-window sampler and adaptive exposure; development uses
the fixed unit-magnitude execution setting. Do not silently change these contracts.
Generate fresh portable unfitted starts for seeds11/23/37 once per architecture,
including RNG state and exact empty-optimizer acceptance. No fitted transfer.

## Checkpoints and curves

Each cell collects 100,000 environment transitions. This is a transition budget,
not 100,000 compute units. Seed11 initially has three cells, or 300,000 transitions.
Evaluate every 10,000 transitions after optimizer work has completed. Select the
highest mean development net return, keeping the earliest checkpoint on ties.
Keep selected and final results distinct; a selected-only gain cannot pass.

`training.native_post_updates` wraps native training calls without changing their
optimizer implementation. Its checkpoint callback runs after the native call;
it temporarily removes the instance hook during serialization to prevent closure
capture. The wrapper itself neither starts training nor allocates Colab.

Retain compact JSONL/CSV/HTML checkpoint curves with returns, holding comparison,
change in percentage points, drawdown and costs. Preserve model, normalization,
source/data identities, counts and selected/final full reports. An external
producer/verifier must authenticate physical results before calling the pure
report reducers; a passing reducer over synthetic dictionaries is not evidence
of learned performance.

## Screen and confirmation

Seed11 must improve selected and final mean return by at least two percentage
points against each control. Final stock and ETF means must not worsen against
either control; final mean drawdown may worsen by at most two points. Selected
and final stock/ETF excess over holding and median-stock excess must each be at
least minus two points.

Only after all gates pass, add seeds23 and37 for all roles. Do not repeat seed11.
Require at least two individually passing seeds, aggregate selected/final gains
of two points against both controls, the same aggregate risk/absolute gates and
at least two positive paired final seeds against each control. Average each
seed's stock median; do not pool stocks to manufacture a different median.
A development outperformance label additionally requires nonnegative group excess
in both phases on average and in at least two individual seeds.

Failure closes this fixed candidate family without extra steps or rescue grids.
Operational/profile failures are not performance findings. Existing failed
families and the persistent CrossQ integrity stop remain intact. Declining-stock
labels are reporting-only; profitable shorting requires both positive total
return and positive net short P&L. No orders are authorized.

## Colab admission and budget

Profile on one Colab CPU session first, using its actual default Python kernel.
Measure 512 candidate collection steps, 64 candidate updates, full512-observation
actor refresh, 64 native SAC updates, three PPO1,000-transition rollouts, one
full26-symbol development evaluation per arm, and checkpoint/report/archive I/O.
Retain physical counts, models and timing samples. Profile scores cannot select
models or admit a candidate on performance.

The registered conservative p95 projection adds evaluation/I/O, a1.5 multiplier,
fixed work reserve, 600 seconds setup and 360 seconds teardown. Every100k arm
must fit7,200 seconds. A failed projection blocks performance allocation without
reducing100k or altering the model. Profiling gets900 seconds with420 seconds
reserved for retrieval/teardown. A pure projection pass is insufficient without
physical acceptance and current aggregate credit admission.

The entire study, including profiling/screen/confirmation, has a two-unit
incremental allowance; profiling has a0.1-unit allowance. Both sit inside the
unchanged original150-unit project ceiling/130-unit teardown threshold and
account150-unit minimum/170-unit teardown threshold. Preserve one study baseline
and the original ledger, historical charges and stops. Initial performance
concurrency is three, subject to aggregate admission, not a global concurrency
limit. Meter every minute, estimate between readings, retain every owned session
ID, retrieve bounded artifacts and verify owned shutdown. No automatic cell
retry or restart after a stop.

## Acceptance boundary

No development pass completes the project goal. Before promotion, require a
checksummed portable preprocessing/model bundle, deterministic inference acceptance
and an independently frozen prospective comparison. Continue documenting failed
attempts and uncertainty; do not reopen the retired interval for selection.
