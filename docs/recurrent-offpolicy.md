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

The initial enumerator sorts retained metadata and materializes all complete
windows. Profile or replace that path with incremental valid-start indexing
before production-size per-update sampling; it is a correctness reference, not
an accepted training throughput result. No learner currently calls it.

19 offline checks cover an independent history-based oracle over120 interleaved
writes/overwrites, the audited ring counterexample, valid physical wrap, symbol/
episode isolation, stale/altered references, terminal/truncated masks, burn-in,
incomplete windows, atomic bad-append rejection and private-RNG repeatability.
Ruff passes. Run:

```bash
.venv/bin/python -m pytest -q tests/test_recurrent_replay.py
.venv/bin/ruff check src/rl_stocks/training/recurrent_replay.py tests/test_recurrent_replay.py
```
