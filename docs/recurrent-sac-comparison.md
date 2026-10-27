# Recurrent SAC development comparison

## Status and purpose

The replay-only and fresh-environment Colab smokes passed. They validate learning,
restoration and reporting mechanics; they do not establish trading quality.
The next performance hypothesis is that independent observation memory plus
SAC replay improves both checkpoint selection and end-of-run performance.

The fixed100k training worker and full-cell verifier are implemented with offline
untrained/contract checks. No actual100k result exists. The first Colab profile
was deadline-stopped and remains incomplete; allocation is still gated on a
reviewed execution plan and physical acceptance.

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

## Cost review and recovery publication improvements

A read-only four-sample local benchmark of the actual retained184,189,350-byte
tree compares gzip6/1/1/6. Mean level6 takes2.894 seconds; level1 takes1.199
seconds (2.41× faster) and produces37.41MB versus29.86MB (25.3% larger). All13
member file hashes match at both levels, original files remain unchanged, and
temporary archives are removed. This is local CPU I/O evidence, not measured
Colab speed or a new training result.

The shared checkpoint exporter and final Colab bootstrap archive now default to
level1; the exporter permits an explicit level1–9 override. Atomic replacement
preserves a previous completed archive on failure and removes incomplete temporary
archives. Scientific file content, model weights and checkpoint checksums are
unchanged by the transport compression choice.

The profiling worker now saves and publishes each learned checkpoint immediately
after its bounded optimizer work, before the next model, prefix benchmark or full
evaluation. It publishes partial timings alongside the model. Later checkpoint
I/O timing includes both this early save/export and report/archive work, so the
extra recovery work is counted. On failure a previous published snapshot remains
available. These changes have offline checks; they do not authorize replaying the
stopped allocation or accept missing completion evidence.

The GRU timing projection still needs about4.89 hours even if evaluation and
archive costs were zero. Faster compression alone cannot justify a two-hour
allocation. A separate longer-CPU execution option would budget6/3/2 hours for
GRU-SAC/SAC/PPO, keeping100k transitions and all performance gates. At the observed
0.08 units/hour this is0.88 units for the three-cell screen or2.64 units for all
three seeds' allocation ceilings. A possible four-unit study envelope would need
a separate registered review; the current two-unit envelope, stop and claim remain
unchanged. No such option is allocatable yet, and rates must be checked afresh.

[Colab's FAQ](https://research.google.com/colaboratory/faq.html) describes variable
resource availability and typically at most12-hour runtimes; a six-hour deadline
is therefore an execution proposal, not a guaranteed session lifetime. Durable
retrieval and verified cleanup remain required. No GPU allocation is justified
by the current evidence.

Evidence: [compression benchmark](../results/recurrent-offpolicy-2026-10-27/archive-compression-benchmark.json)
and [cost review](../results/recurrent-offpolicy-2026-10-27/execution-cost-review.json).

## Full-development worker implementation

`training.recurrent_comparison_worker.execute_colab` implements the fixed100k
cells without provisioning sessions. Actual Colab CPU/default-Python verification
runs before data, model, RNG or output work. Exact panels, immutable data identity,
registered architecture/seed and fresh output checks precede learning. Dedicated
fleet admission, confirmation-seed gate evidence and full physical result
verification remain required before this worker can be dispatched. No runtime
plan currently exposes it, so this implementation does not unblock allocation.

Native SAC and RecurrentPPO load the portable unfitted weights and RNG streams
into the fixed training environment. The GRU loop now accepts only a fresh
matching learner with zero updates, empty optimizer state, fixed learning
rate/gamma/tau/alpha and matching maximum updates/geometry. This path preserves
the restored initialization streams rather than reseeding after loading. Existing
smoke calls retain the original seeded-construction path.

Native optimizer hooks bound actual steps and reject missing/nonfinite gradients
or nonfinite updated weights. At post-update checkpoints, physical Adam counters
and finite moments must match the actual environment transitions and library
update counters. At100k, SAC requires99,000 actor/critic/entropy steps each; PPO
requires20,000 policy Adam steps, with the separate library epoch counter1,000.
PPO's epoch counter is explicitly labelled in curve diagnostics, never described
as an Adam-step count.

Initial and every10k learned snapshots are exported before full development
evaluation. Native save closures are removed by the post-update wrapper during
serialization. Both paths retain checksummed model states, strict earliest-tie
selected pointers, final pointers, separate selected/final JSON/HTML reports,
training sampling/execution diagnostics and compact JSONL/CSV/HTML return-change
curves. Evaluation uses the exact development panel and unit-magnitude execution;
training retains adaptive exposure and the unchanged net-equity objective.

The native worker preserves RNG across evaluation; each checkpoint's completed
state exists before curve/pointer publication. Failures retain a failure record,
preserve earlier checkpoints, remove optimizer hooks, close the training environment
and attempt owned artifact export. Snapshots do not promise exact interrupted
replay/environment continuation; automatic retry/resume remains unsupported.

Offline tests use actual untrained native constructors and save/load parity with
learning/update/Adam methods forbidden. They verify local guard ordering, counter
contracts, mismatched-initializer rejection and zero-update snapshot restoration.
They do not establish a successful100k execution, portfolio performance, prospective
acceptance or a portable deployment bundle.

## Full-cell result verification implementation

`training.recurrent_comparison_verify.verify_cell` requires externally supplied
role, seed, source-package hash and Git revision. It validates the full immutable
panel and portable initializer matrix, exact metadata and complete bounded file
inventory. It restores all eleven checkpoints, checks initial weights against
portable starts, actual model counters/Adam moments and fixed native or GRU
settings. Native ZIP loading uses the exact bytes whose checksum was checked;
caller RNG is preserved during restoration.

The verifier requires ten10k post-update curve points, strict earliest-best
selection and an independent final pointer. It checks completed-run identities,
selected/final reports and100k sampling/execution counts. Curve aggregates and
changes recompute from per-symbol compact records; CSV must match the full
required column set and JSONL values. These checks do not rerun all intermediate
checkpoint policies or claim cross-CPU bitwise learned action reproduction.

A shared `evaluation.recorded_economics.verify_report` checks each reported
selected/final path against recorded-action simulator replay, including cash,
positions, equity, costs, turnover, age, borrow and actual fills. Unit-exposure
and unchanged protective components are required. GRU results additionally check
local direct-versus-frozen carried/reset prediction parity. This separates
recorded economics from cross-backend actor arithmetic without changing an
existing historical tolerance. The prior environment-smoke verifier reuses this
same path; its original frozen checks remain enabled.

Compact summaries enforce one episode per each original26 symbols, with separate
22-stock/four-ETF matched means, median stock excess and drawdown. Declining-stock
short P&L is a separate reporting field and requires positive total policy return
and positive marked short P&L before counting success. It cannot change scores,
groups, checkpoint selection or the predefined reducer gates.

Untrained snapshot tests exposed an effective-parameter representation bug in
the earlier profile verifier: SAC adds `use_sde=False` to policy kwargs and
converts target entropy−4 to−4.0; PPO wraps fixed clipping in a native
`FloatSchedule(ConstantSchedule(...))`. Shared validation now accepts those exact
installed-library defaults and numeric values, while rejecting changed entropy,
SDE, optimizer learning rates and nonconstant schedules. Actual saved SAC64/PPO600
Adam-state snapshots pass this corrected read-only check; original files remain
unchanged. This does not fill the stopped profile's missing completion/provenance.

The full-cell verifier is implemented but has no actual100k result to accept yet.
Tests cover untrained restoration and independent synthetic corruption cases;
they are not learned full-run acceptance or performance evidence. Dedicated
archive/fleet integration, reviewed execution admission and actual physical
full-run verification remain before any research promotion.

## Corrected engineering-profile review prepared

A separate version42 task and explicit-review path are implemented. The sole
corrected CPU profile gets20 minutes including420 seconds teardown reserve,
sharing the original0.1-unit profiling allowance. It preserves the original
baseline, claim, stop and two-unit study cap; it admits no100k task. Actual input
preflight passes while the original ledger remains byte-for-byte unchanged and
stopped. User authorization is pending; no new session has been provisioned.
See [execution review](recurrent-execution-review.md) for scope and commands.

## Offline matrix acceptance command

The complete-cell acceptance command and standalone comparison dashboard are
implemented. They require physical cell verification, unchanged passing seed11
receipt/artifacts before confirmation, fixed gates and prospective requirements.
Selected/final stock/ETF means, incremental curves and individual-stock excess
are retained without duplicating full paths. See [results guide](recurrent-comparison-results.md).
No actual100k matrix exists; the corrected profile still awaits explicit approval.
