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
synthetic gate tests are implemented. Portable initialization assets now pass actual local construction/restoration
acceptance; a guarded Colab profiling worker is implemented, with dedicated fleet admission and downloaded-artifact verification implemented;
physical Colab acceptance remains
required before performance allocation. No full comparison has run yet. The first actual CPU profile was deadline-stopped; see the final section for evidence and current allocation block.

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

## Portable-start preparation acceptance

`training.recurrent_comparison_starts` constructs the registered native models and
GRU learner without collection or fitting. The builder publishes all nine cells
atomically with a checksummed file inventory, native tensor archives/GRU
checkpoints and full Python/NumPy/Torch RNG records. The acceptor independently
reconstructs each seed, restores exact saved tensors, checks empty optimizer states and
constructor RNG streams, and preserves caller streams even on rejection.

Actual assets are retained at `.runtime/recurrent-comparison-starts-v1`; they are
not committed model binaries. Local Python3.13.16/Torch2.8.0 preparation and a
second separate-process acceptance both passed. This does not replace actual
Colab default-kernel and portable-start checks before gradients.

```bash
MPLCONFIGDIR="$PWD/.runtime/matplotlib" PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/build_recurrent_comparison_starts.py \
  .runtime/recurrent-comparison-starts-v1 --accept-only
```

Omit `--accept-only` only for a fresh immutable destination. Existing outputs
reject; never regenerate the registered assets to select more favorable weights.
The profile worker must use these exact accepted assets, not fresh runtime draws.

## Profiling worker implementation

`training.recurrent_comparison_profile.execute_colab` rejects local execution
before dataset access, output creation or model construction. It loads accepted
portable starts, checks the exact111/26 panel and immutable dataset identity,
and enforces CPU/default Python before updates. It performs the fixed512-step
candidate schedule with64 updates,64 native SAC updates on the same collected
training replay, and3,000 PPO transitions. Native optimizer post-hooks record
completed steps rather than infer them from nominal epochs.

It retains collected replay arrays, candidate update diagnostics, learned
checkpoints, full development reports, timing arrays and backend provenance.
Native PPO rollout timings include both collection and optimization. Full actor
refresh timing includes a512-observation tensor conversion and no-grad GRU pass.
Evaluation preserves RNG streams. Partial timing evidence and bounded owned
artifact export are attempted on failure, and environment/hook cleanup runs in
`finally`. The external fleet controller remains responsible for credit/deadline
monitoring, retrieval and verified session termination.

This implementation has offline guard/restoration acceptance only. No actual
profile has executed. Dedicated runtime admission, archive verification and
physical Colab acceptance remain before any100k performance task.

## Dedicated runtime integration

Plan version41 admits exactly one CPU profiling task and packages the accepted
initialization matrix explicitly. It dispatches directly to the profile worker,
never the generic trainer. The shared original-ledger baseline records the full
two-unit study allowance; profile admission uses its0.1-unit suballowance and a
persistent single-allocation claim. Failure or budget stop cannot trigger a
worker retry. The standard fleet enforces aggregate metering, the account floor,
900-second deadline,420-second teardown reserve, bounded downloads and verified
owned-session termination. The CrossQ stop is preserved.

The downloaded archive uses a strict256MiB file allowlist and rejects duplicate
paths, links, traversal and incomplete completion evidence before loading models.
Physical acceptance restores all learned models, validates actual Adam moments
and64/64/600 optimizer counts, native hyperparameters, replay shapes, update
schedule, source/start identities and full-panel report/baseline accounting.
A projection failure can still be a valid completed profile; it blocks subsequent
performance allocation. No performance tasks are implemented by this profile plan.

Prepare the task without allocating:

```bash
MPLCONFIGDIR="$PWD/.runtime/matplotlib" PYTHONPATH=src \
  .venv/bin/python scripts/build_recurrent_comparison_profile_jobs.py \
  --output .runtime/recurrent-comparison-profile-jobs-v1.json
```

The first actual execution is recorded below; full physical acceptance remains incomplete.

Portable acceptance compares loaded weights with the saved checksummed assets.
It does not require freshly regenerated orthogonal weights to be bitwise equal
across CPU backends. Constructor stream consumption remains checked, and native
tensor loading checks exact archive-to-model restoration. No learned prediction
tolerance or historical integrity-stop rule is changed.

## First actual profile: stopped, no automatic retry

CPU session `rl-stocks-cpu-c5d2ec2da580` ran from clean `c19f6d7` on Colab
Python3.13.18. The900-second allocation began teardown at its480-second working
threshold. Owned shutdown is verified; the persistent claim and budget-stop flag
remain. Installation alone prepared84 packages in133 seconds.

All learned checkpoints, three full development reports and512 replay rows were
salvaged. Independent restoration validates GRU64 updates, native SAC64 updates
per optimizer, and PPO3,000 transitions/600 Adam steps. Timing JSON was retrieved
separately before teardown. Completion/provenance/final partial-profile evidence
is missing, so the full physical profile is **not accepted**. No missing evidence
is fabricated or inferred from checkpoints. These short policies are not a
performance comparison.

The separately retrieved timings imply required allocation seconds of18,744.80
for GRU-SAC,7,643.47 for SAC and4,103.10 for PPO, including fixed reserves. Both
SAC arms fail7,200 seconds;100k training remains blocked. The candidate p95
sample/update is55.81ms and full512 refresh36.29ms. These are preliminary timing
findings with the incomplete-acceptance limitation above. No automatic restart,
shortened100k run or model/reward rescue follows. Next review implementation/I/O
costs and a separately justified execution proposal without changing performance
gates or erasing this attempt.

The retained snapshot expands to184,189,350 bytes, exceeding the initial128MiB
validator cap. The infrastructure cap is corrected to256MiB while preserving the
exact allowlist and incomplete-run rejection. Repeated full-report archive work
also consumed meaningful time (16.22/20.06/32.74 seconds); durable checkpoint-first
publication and cheaper archive handling need review before another proposal.

At verified cleanup, project charge is39.54822735/150 (110.45177265 remaining).
This attempt charged0.01209140 estimated units against a rounded measured debit
of0.01. Account348.58, rate0, assignments0. All scientific artifacts remain; only
verified duplicate staging data is removed. See the compact evidence in
[results](../results/recurrent-offpolicy-2026-10-27/comparison-profile-run.json).
