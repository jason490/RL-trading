# Recurrent off-policy feasibility

## Decision and scientific scope

Recurrent SAC remains an independent candidate, not an accepted replacement for
RecurrentPPO. No dependency, algorithm registration, training allocation or return
comparison is added by this review. The stopped CrossQ study remains stopped.
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
