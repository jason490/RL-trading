# Recurrent off-policy feasibility

## Decision and scientific scope

Recurrent SAC remains an independent candidate, not an accepted replacement for
RecurrentPPO. A separate four-update-per-arm Colab learned smoke has now passed;
no trading-return comparison or generic algorithm registration follows. The
stopped CrossQ study remains stopped.
Existing failed indicator/reward families and retired-test isolation remain intact.

Memory with replay could improve sample reuse when the current observation hides
relevant history. That hypothesis needs a matched experiment: a paper's control
benchmarks cannot establish stock returns. See [literature review](model-research.md).

## Tagged implementation audit — 2026-10-27

skrl2.1.1 provides SAC_RNN. Its learner stores policy recurrent states and passes
them to current/next policy and critic calls. Collection resets memory at either
termination or truncation; Bellman targets mask termination only. Separate critic
history, next-state alignment and burn-in are not established by those calls.
This is an integration concern, not proof that every supplied model is incorrect.
[Tagged learner](https://raw.githubusercontent.com/Toni-SM/skrl/2.1.1/skrl/agents/torch/sac/sac_rnn.py).

RandomMemory builds sequences by adding environment-strided offsets to randomly
selected physical slots. Its sampler does not consult episode flags or the ring
write cursor. After overwrite, a physically adjacent sequence can cross the
newest-to-oldest discontinuity. Reset behavior within episodes also needs model
inspection rather than assuming the sampler prevents crossings.
[Tagged sampler](https://raw.githubusercontent.com/Toni-SM/skrl/2.1.1/skrl/memories/torch/random.py).

The base memory tracks its write cursor and full status, so chronological validity
can be checked explicitly by an integration. Storage contiguity alone does not
provide that check.
[Tagged memory](https://raw.githubusercontent.com/Toni-SM/skrl/2.1.1/skrl/memories/torch/base.py).

## Concrete counterexample and evidence limits

[Arithmetic evidence](../results/recurrent-offpolicy-2026-10-27/replay-index-audit.json)
records source hashes and independently enumerates the published index formula.
Capacity8, one environment, sequence length3, after insertion times0 through10:
physical slots contain `[8,9,10,3,4,5,6,7]`, with the next write at slot3.
Possible indices `[1,2,3]` produce times `[9,10,3]`; `[2,3,4]` produce
`[10,3,4]`. Two of six possible starts are temporally invalid.

This is a source/arithmetic audit, not an installed-library runtime test or
performance measurement. No package was installed, model constructed, backward
pass run, provider contacted, Colab allocated or retired-test data opened.

## Required replay contract before training

1. Assign every transition an immutable insertion counter, environment ID,
   episode ID and within-episode step. Sample chronological sequences from valid
   transitions, rejecting overwritten/incomplete starts and write-cursor crossings.
2. Keep an episode boundary explicit. Prefer sequences wholly inside one episode
   initially. Any padded suffix must have a loss mask and cannot contribute to
   entropy, critic, actor or normalization statistics.
3. Preserve the true terminal observation before auto-reset. Stop bootstrapping
   on termination; bootstrap a time limit from its final observation, never the
   next episode's reset observation. Collection memory resets at both boundaries.
4. Reconstruct current-network recurrent state from a causal prefix. Specify
   burn-in separately from loss-bearing unroll length; detach after burn-in.
   Stored collection states may be stale after updates and cannot silently stand
   in for current actor, critic and target-critic histories.
5. Give each recurrent network its own state lineage. Next-observation targets
   must follow the same history advanced by one transition, with the correct
   prior action/reward if the chosen architecture consumes them.
6. Check uninterrupted versus chunked forward parity, symbol resets, vector-env
   isolation, partial buffers, ring overwrite, terminal/truncation boundaries,
   padded-loss exclusion and frozen recurrent inference/save-load parity.

## Next bounded implementation

The sequence-index contract below is implemented without optimization or market-data
access. Next audit recurrent model and wrapper state/observation handling against it.
Only adopt a library after actual default-Colab-Python dependency acceptance;
metadata compatibility alone is insufficient. Keep preprocessing frozen and the
existing four-action execution/reward/cost contracts unchanged.

Before any learner smoke, register architecture, sequence/burn-in lengths,
replay capacity, update ratio, seeds, transition/update counts, checkpoint gates,
maximum units and allocation deadline. A library change also needs a native
feedforward SAC control to distinguish implementation effects from memory.
Compare against the retained RecurrentPPO control on matched development rows.
Screen once, confirm promising results across11/23/37, and retain negative runs.
Neither this review nor a passing smoke clears the historical CrossQ stop or
establishes profitable trading.

## Implemented sequence indexing

`training/recurrent_replay.py` adds a dependency-light metadata ring. The owning
replay store must associate each returned insertion ID with an immutable payload.
Each append validates environment, increasing episode identity after an end,
contiguous within-episode steps and unchanged symbol before mutation. Capacity is
shared across environments. A valid sequence may wrap physical slot zero, but
its insertion IDs and episode steps must remain chronological.

`windows(burn_in=..., unroll=...)` enumerates complete, unpadded windows confined
to one environment/episode. `sample` selects uniformly with replacement using a
caller-owned NumPy generator. `resolve` revalidates retained metadata immediately
before payload gathering, rejecting snapshots invalidated by overwrite. Concurrent
writes require owner synchronization; this object does not provide thread safety.

The loss mask excludes burn-in; bootstrap masks exclude termination only, while
reset-after flags include termination and truncation. These masks specify a
contract: this module does not calculate a loss, enforce learner masking, store
observations, recover final observations or reconstruct neural hidden state.
A window starting after episode step zero has a finite burn-in prefix, not a
proof of exact full-history state reconstruction. That approximation must be
specified and tested in the eventual learner protocol.

`windows` retains the initial full enumerator as a correctness reference.
`sample` now uses the incremental pool described below. No learner currently
calls either path; learner/Colab throughput remains unmeasured.

19 offline checks cover an independent history-based oracle over120 interleaved
writes/overwrites, the audited ring counterexample, valid physical wrap, symbol/
episode isolation, stale/altered references, terminal/truncated masks, burn-in,
incomplete windows, atomic bad-append rejection and private-RNG repeatability.
Ruff passes. Run:

```bash
.venv/bin/python -m pytest -q tests/test_recurrent_replay.py
.venv/bin/ruff check src/rl_stocks/training/recurrent_replay.py tests/test_recurrent_replay.py
```


## Implemented payload and final-observation path

`training/recurrent_payload.py` owns copied finite float32 observations, four
bounded action components, rewards and next observations. `append_raw` requires
the unreset Gymnasium next observation. `append_sb3` accepts one explicitly
identified lane of SB3 VecEnv output: an ended transition requires its
`terminal_observation`; `TimeLimit.truncated` distinguishes bootstrap from true
termination. It rejects missing terminal observations instead of substituting the
returned reset observation. Other autoreset APIs need their own explicit adapter.

Ongoing transitions must continue the previous stored next observation exactly
after float32 conversion. Symbols, environment/episode IDs and step continuity
are checked by the metadata index. Validation failures occur before insertion.
The collector still owns reset/episode counters; IDs must not be inferred from
symbols because repeated episodes can use the same symbol.

`gather` revalidates every sequence and produces copied, initially read-only
arrays: observations `[B,T+1,D]`, actions `[B,T,4]`, rewards/masks/IDs `[B,T]`.
For transition t the current input is observation t and its next input is t+1;
the last observation is the true final observation when the window ends at a
boundary. Batch geometry and burn-in must agree. Re-enabling a batch copy's
write flag cannot change the stored payload. Writes/gathers require owner
synchronization; the exposed index must not be independently mutated by callers.

12 new payload checks plus19 index checks pass. One executes the actual project
TradingEnv through installed SB3 DummyVecEnv with a two-step synthetic episode:
returned reset feature0 differs from terminal feature2; replay retains feature2
and allows truncation bootstrapping. Other fixtures verify true-termination
masking, missing-terminal rejection, float32 finiteness, observation continuity,
copy isolation, mixed geometry and overwritten snapshots. These tests construct
no policy and run no optimizer, provider request or Colab allocation.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_recurrent_payload.py tests/test_recurrent_replay.py
```

Next: verify independent actor/critic/target recurrent histories and causal
burn-in/one-step target alignment using forward-only fixtures, then replace or
profile the reference enumerator before any registered learner smoke. The
payload store alone does not implement SAC, sequence loss masking or bootstrapping.


## Implemented causal GRU history alignment

`training/recurrent_history.py` provides forward-only observation-history helpers
for the prospective adapter. It is not a SAC learner or an adoption of skrl's
recurrent implementation. The current candidate design uses separate
unidirectional, batch-first, zero-dropout PyTorch GRUs for actor, critic1/2 and
target critic1/2. Actions would enter downstream Q heads; previous actions/rewards
are not separate recurrent inputs in this design. Those architecture decisions
must be fixed in the eventual experiment protocol, not presented as a paper
replication or an empirically chosen improvement.

`encode_history` starts from fresh zero state on each sequence call, runs the
burn-in prefix under no-grad, detaches that state and the replay inputs, then
processes the remaining observations once. Current features are suffix outputs
excluding the final element; next features are the same outputs shifted by one.
This includes the true final observation already retained by the payload store.
Starting a separate next-observation pass from the unadvanced prefix state would
misalign the history and is covered by a negative test.

`sac_histories` requires five named encoders and rejects shared parameter storage.
Every network reconstructs its own prefix with its own parameters; no actor state
is reused as a critic/target state. Targets always disable gradient tracking.
Optional online suffix graph construction is explicit and respects an enclosing
no-grad context. The helpers never call backward or an optimizer, change model
mode, update normalization or retain a collection-state cache.

22 native untrained-forward checks pass alongside31 replay/payload checks.
Evidence covers full versus chunked forward agreement, current/next alignment,
future-input causality, independent batch lanes and networks, fresh episode
starts, unchanged weights/RNG/parameter gradients, detached burn-in, target graph
exclusion and invalid geometry/dtype/recurrent semantics. An integration fixture
passes actual copied two-episode payload batches through all five encoders and
compares against separate full-history forwards. No local differentiation or
training occurs; graph flags are inspected without running backward.

Exact full-history equivalence is tested when the supplied sequence includes the
episode start. At later sampled starts, finite burn-in remains an explicit
approximation. Untrained forward acceptance does not prove correct SAC losses,
update schedules, target updates, learned serialization or profitable returns.
Before learner integration, address scalable valid-window sampling and select a
maintained implementation/adapter with a pinned, audited loss path. Actual
optimizer/save-load smoke acceptance belongs in Colab under a registered budget.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_recurrent_history.py tests/test_recurrent_payload.py tests/test_recurrent_replay.py
```


## Incremental valid-window sampling

`sample` now caches one complete-window start pool for the requested total
burn-in plus unroll length. A retained `(environment,episode,step)` lookup and a
dense start list support uniform sampling with replacement. Pool initialization
is linear in retained transitions; changing total length rebuilds once. Changing
only the burn-in/unroll split with the same total length reuses the pool.

On append, the fixed global FIFO ring evicts its oldest reference. Any containing
window that started earlier was already invalidated when its own start was
evicted. Therefore only a window starting at this evicted ID remains to remove;
swap-removal updates its dense position in constant time. A new valid start is
added when a complete same-episode window ends at the inserted transition.
Start/end retention implies intermediate retention under FIFO ordering. Sampling
materializes only the selected windows, proportional to batch size times length.
The owner still synchronizes writes and resolves snapshots before gathering.

Six additional tests compare the incremental pool to the independent reference
after every interleaved append/overwrite, including length1, multiple episodes,
geometry changes, failed writes and an assertion that repeated sampling does not
call full enumeration.59 combined replay/payload/history checks and Ruff pass.
Pool maintenance changes sampling order, so this is a versioned prospective path,
not a way to reproduce or restart an earlier stopped learner study.

[Benchmark evidence](../results/recurrent-offpolicy-2026-10-27/replay-benchmark.json)
uses local Python3.13.16,100,000 capacity,110,000 initial synthetic writes,
eight environments,512-step episodes,16 burn-in plus16 unroll,256-window batches
and30 draws. All93,800 valid starts equal the reference; every sampled sequence
resolves. Median cached draw1.873ms, maximum2.001ms; one reference enumeration
1.468s and initial pool build25.481ms. An additional10,000 writes with active
pool maintenance took22.052ms. Timings exclude payload gathering, model forwards,
backward, final reference resolution and Colab setup. This is neither an
end-to-end speedup claim nor a hardware allocation projection.

Reproduce with a new output path (existing evidence is never overwritten):

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_recurrent_replay.py \
  --output /tmp/recurrent-replay-benchmark-new.json
```

A [prior trial](../results/recurrent-offpolicy-2026-10-27/replay-benchmark-before-format.json)
is retained from before removal of one trailing blank line. Its source hash equals
the current module plus one newline; both trials agree on all validity counts.
No model/data-provider/Colab activity occurs. The benchmark uses fixed synthetic
symbol names and stores only compact timing/provenance JSON; no bulk output needs
cleanup. Next audit and integrate the pinned learner loss, then register the
matched candidate/control smoke and performance protocol before optimization.


## Recurrent networks and SAC objectives

`rl/recurrent_sac/networks.py` implements a prospective, unregistered network
package: independent observation-only GRUs, actor hidden/ReLU plus four mean and
log-standard-deviation outputs, and twin critic heads consuming recurrent features
plus four actions. Target critics are independent deep copies initialized to
online weights with gradients disabled. Hidden width defaults64 but has not been
selected through a performance experiment. No generic algorithm factory route
or optimizer is added by this feature.

The actor clamps log standard deviation to[-20,2] and uses installed SB3's
SquashedDiagGaussianDistribution for its tanh action density, flattening batch/time
before that API's action-dimension reduction. Optional explicit standard-normal
noise supports deterministic equation fixtures. The distribution is local to the
call, so no mutable action cache is retained in the network.

`training/recurrent_sac_losses.py` validates unpadded copied replay tensors and
computes separate critic and actor objectives. Only transitions after burn-in
enter reductions. Targets use next actor actions and minimum twin target Q minus
entropy, discounted with a terminal-only bootstrap mask. Critic loss is half the
sum of two mean squared errors. Actor loss is mean(alpha*log_probability-minQ).
Temperature loss uses detached log probabilities and target entropy-4 by default.
Alpha is detached from actor/critic objectives; targets are no-grad. During actor
Q computation critic parameters are temporarily frozen, preserving derivatives
through sampled actions, and original requires-grad flags restore on error.

[Source audit](../results/recurrent-offpolicy-2026-10-27/loss-source-audit.json)
pins installed SB3 2.7.0 train-method and squashed-distribution hashes. Its native
update order is temperature, critic, actor Q recomputation, actor, target update.
The objectives are deliberately separate so a future Colab update driver can
recompute actor Q after the critic step instead of reusing a stale pre-update
loss graph. This is custom recurrent sequencing around the audited SAC equations;
it is not an unchanged native SB3 learner or a skrl learner acceptance claim.
[Native source](https://github.com/DLR-RM/stable-baselines3/blob/v2.7.0/stable_baselines3/sac/sac.py).

10 additional forward/analytic checks bring focused acceptance to69. Independent
NumPy targets distinguish true termination from time-limit truncation; native
SB3 density comparison verifies batch/time flattening. Separate full-episode GRU
forwards verify target, critic and actor reductions against the chunked replay
path. Tests check unchanged weights/RNG/gradients, graph flags without local
backward, restored critic flags on exceptions and corrupted replay rejection.
These checks do not establish actual gradients, optimizer steps, Polyak behavior,
learned restoration or profitability. Local fixtures perform no differentiation.

Next implement a bounded Colab-only update driver, durable learned serialization
and frozen recurrent prediction. Register a candidate/control smoke with fixed
counts, source/data/initialization identities and shared budget guards before
any optimizer call. Only after actual learned smoke acceptance should a matched
performance comparison be scheduled. The historical CrossQ stop stays intact.


## Bounded learner driver, snapshots and frozen actor

`training/recurrent_sac_learner.py` adds an explicit lifetime update allowance.
`update` rejects outside default Python3.13, Colab release context, the owned
`/content/rl-stocks` workspace and CPU hardware before conversion, RNG, counters or
backward. The existing runtime controller must still establish ownership, actual
Python version, shared metering and cleanup; environment markers alone are not
an ownership or budget controller. No generic training route is enabled.

Each allowed update validates replay, increments attempts and draws one current
actor noise tensor. It caches pre-temperature alpha, applies temperature then
critic Adam, recomputes actor/online-critic histories after that critic step with
the same current-action noise, applies actor Adam, then Polyak-updates targets.
All optimized parameters require present finite gradients and finite post-step
values. Per-role completed counters and phase order are retained. A partial
failure marks the learner failed, clears gradients and prevents update retry or
normal checkpointing; the worker must preserve prior completed checkpoints and
failure diagnostics. Max updates survives checkpoint restoration.

`training/recurrent_sac_checkpoint.py` stages immutable checkpoints containing
network/target tensors, raw log-alpha, three native Adam states and raw Torch RNG.
JSON manifests bind geometry, optimizer settings, counters, source/data/observation
schema lineage and SHA256. Loading validates checksum/size, strict model shapes,
optimizer configuration and every native Adam counter/moment geometry; partial
or failed state rejects. Loading uses tensor-only PyTorch deserialization and
preserves caller RNG unless explicit Torch RNG restoration is requested. These
checks provide integrity, not cryptographic publisher authentication.

The checkpoint is a learner snapshot, not an exact environment/replay resume or
a deployable inference bundle. Replay payloads, sampler NumPy/Python streams,
current observation and episode state are not included yet. A worker must not
claim exact continuation from this file alone. Preprocessing/normalizer and
full feature/action/provider manifests still belong in a future frozen bundle.

`rl/recurrent_sac/prediction.py` copies a private eval/no-grad actor and accepts
already-normalized observations with a required matching schema. Deterministic
actions are tanh(mean); returned single-lane state is copied and reset explicitly
at episode starts. It does not fetch data, transform/update normalization or send
orders. Modifying the original learner after freezing cannot change its actor.

Eight additional offline checks bring acceptance to77. Tests verify local update
rejection before mutation; bitwise tensor/raw-entropy/Adam restoration; carried
prediction parity; private-copy/reset/schema behavior; corruption/native-counter
rejection and immutable save destinations. Adam fixtures are manually populated
without gradients or optimizer calls. No test bypasses the local update guard.
Actual gradients, phase ordering, Polyak math during learning and learned-model
restoration remain unproven until a registered Colab smoke observes them.

Next register a bounded smoke on training-only data with explicit transition and
update counts, native feedforward control, shared ledger/account floors, durable
failure artifacts and owned cleanup. Integrate its runner/archive verification
before allocation. No performance run, paper eligibility or old CrossQ retry is
authorized by these offline fixture results.


## Registered learned smoke and worker preparation

[Fixed protocol](../configs/experiments/recurrent_sac_smoke.json), canonical hash
`a530831e6f8a1c28b6e629ef8143fce43d91e93aaa8826ed39ea6e4ba38fea12`,
registers four native gradient updates each for recurrent and feedforward SAC:
eight model updates/24 Adam steps total. No environment rollout, evaluation,
selection or parameter grid is allowed. One CPU allocation,0.1 units,900 seconds
including420 seconds teardown reserve, original150/130 project and150/170 account
guards, persistent baseline and no automatic retry are fixed. This smoke does
not clear the CrossQ stop or establish performance comparability.

Both arms use the previously verified1001-row training replay prefix, immutable
identity7f50968c0272bda3d5b5a75861e5ea25d2f4e7d328a1b014af7de219a0d5a8a7.
A private NumPy11 generator precomputes four batches of eight32-row windows:
16 burn-in plus16 learning rows. The native control consumes exactly the same
128 loss-bearing rows per update, flattened. Their initial weights differ by
architecture; this is an operational control, not an isolated memory ablation.
Recurrent initialization is fresh seeded PyTorch; native initialization reuses
the original pinned full64 untrained asset. No weights are fitted during preparation.

Replay dones identify a timeout at row511, yielding two contiguous segments and
939 complete windows. Native replay lacks source-symbol metadata, so labels are
explicit replay-segment ordinals used only for indexing; no real-symbol attribution
is inferred. Ongoing next/current observations agree exactly. The original input
and its old numerical-diagnostic protocol remain immutable and referenced as
provenance, not rewritten for the new study.

[Actual preparation](../results/recurrent-offpolicy-2026-10-27/smoke-preparation.json)
verifies all matched fields, empty native/recurrent optimizers, exact initial
targets and initial recurrent tensor hash6832b68b7141e304ecf582cf80dae567cb5318dee8df651de535b36dae438649.
Parameter counts: actor21,384, online twin critics42,370, target critics42,370
(nontrainable), plus one learned entropy scalar. Protected input/ledger/CrossQ
stop/initializer files rehash unchanged; caller Torch RNG restores. No backward,
optimizer step, allocation, provider call or retired-test access occurred.

`training/recurrent_sac_smoke_worker.py` rejects local execution before output or
construction. In Colab it observes native optimizer pre/post hooks, finite actual
gradients and recurrent-memory gradients, exact phase counts and actor/critic
ownership. It checks each actual recurrent target update against retained
pre-update targets and post-update critics. Initial/final snapshots, compact
progress, actual hook evidence, matched row IDs and carried/reset predictions
are retained. Both restored recurrent tensors/raw entropy/Adam and native
checkpoint tensors/moments are compared physically; prediction parity is checked
within the runtime. Failure records preserve attempted/completed counts. Bounded
checkpoint archives export after initial and completed stages and on handled
failure; the controller still owns transport bounds and verified teardown.

82 focused checks/Ruff pass (local forward/manual-state fixtures only). New checks
cover the fixed protocol, replay segmentation/matched native rows, timeout versus
terminal masks, unflagged reset rejection, pre-output local-worker rejection and
native hook rejection without optimizer calls. At that preparation stage actual learned acceptance remained
unexecuted. Subsequently integrate a dedicated persistent budget/allocation claim, exact
job validation, worker dispatch and independent bounded archive verifier into the
existing metered fleet, then run the registered smoke from a clean source export.

Preparation command (use a new evidence path; does not allocate):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  PYTHONPATH=src .venv/bin/python scripts/prepare_recurrent_sac_smoke.py \
  --output /tmp/recurrent-sac-smoke-preparation-new.json
```


## Metered runtime and physical archive integration

Version39 job validation now fixes the sole CPU probe and exact training input.
The existing fleet uses a dedicated `recurrent-sac-smoke-budget.json` baseline
and immutable `recurrent-sac-smoke-allocation.json` claim in the original shared
runtime directory. Fresh filenames cannot bypass a consumed claim. It inherits
original charges, floor150/teardown170,0.1 incremental allowance and900-second
allocation with420 seconds reserved for retrieval/cleanup. Worker execution
retries are disabled. Other study versions and their historical stops stay intact.

The worker saves four compact NPZ target-equation snapshots (old targets,
post-critic source tensors and actual targets). Independent forward-only
verification chains every update from initial through final target tensors,
recomputes Polyak arithmetic, reloads native/recurrent Adam states and checks
loss-row identities, phase/gradient records, counts and schema claims. Local
frozen action parity is checked; remote cross-host action recomputation is
reported separately because backend arithmetic need not be bitwise identical.
No return tolerance or performance gate is changed.

The archive is limited to64MiB and an exact completed-artifact allowlist, rejecting
extra files, duplicate members and nonregular entries before physical extraction.
Remote completion must identify clean source, CPU and actual default Python3.13.
Manual physical fixtures and corruption tests never run backward/optimizer steps.
160 focused recurrent/numerical/budget checks pass. Broader fleet regressions are
recorded separately; this is implementation acceptance, not actual learned smoke
acceptance. Account preflight348.59/rate0/no active assignments on2026-10-27.

Build the fixed task without allocation:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_recurrent_sac_smoke_jobs.py \
  --output .runtime/recurrent-sac-smoke-jobs-v1.json
```

Actual allocation requires a clean immutable source export, original whole-runtime
ledger symlink and bound input, then the existing fleet controller. Preserve that
export and every owned ID for recovery. The original CrossQ stop remains blocked.


## Actual learned smoke accepted — 2026-10-27

[Results and artifact index](../results/recurrent-offpolicy-2026-10-27/README.md)
records the single Colab CPU allocation from cleanfc57359/default Python3.13.16.
Both arms completed four actual updates,24 Adam steps total; finite observed
parameter/memory gradients, optimizer order, all four physical Polyak chains,
matched rows, actual counters and learned snapshot restoration pass. Independent
local frozen carried/reset parity passes. Remote AVX2 versus local AVX512 actions
are not bitwise equal; exact saved-state restoration is verified separately.
No return evaluation, promotion or historical-stop clearing occurred.

Inclusive149.784s; estimated charge0.003775191, measured rounded debit0 so far.
Original ledger conservatively39.53232284029461/150, account348.59/idle; sole
session stopped/verified. Removed8.14MiB duplicate upload with1,851 retained files
unchanged. Scientific model/input/moment/target/prediction evidence retained.
Next environment collection/evaluation/resume integration and registered matched
performance gates; do not rerun the consumed smoke or reopen retired test data.
