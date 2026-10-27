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

Build and test the sequence-index contract without optimization or market-data
access first. Then audit the supplied recurrent model and wrapper against it.
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
