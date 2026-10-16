# Algorithm and recurrent-memory stability study

The [protocol](../../configs/experiments/algorithm_stability.json) fixes SAC, PPO
and RecurrentPPO on basic inputs at 100k steps and seeds 11/23/37. The concurrent experiment has started; the first completed cell is PPO seed 11
(selected/final 19.93% at 100k). This is a single-seed validation result; no family
gate is available yet. The three algorithms must share fresh clean source/dependency
proofs, data, reward and simulator assumptions. The inspected test interval is
retired and no orders are enabled.

[Saved-path diagnosis](diagnosis-before-study.json) covers all 12 completed 250k
indicator/control runs, using existing selected and final validation reports.
It records source-file hashes, action saturation, exposure and annual return
changes, trading costs and available optimizer diagnostics. This does not run
a new model evaluation or establish a causal optimization mechanism.

The completed indicator experiment is published separately. Its best checkpoints
remain retained; later training did not overwrite those frozen policies.

The [real concurrent preflight](concurrent-preflight.json) accepts all three
models and confirms a peak of three CPU assignments at 0.24 units/hour, shared
metering, exact frozen sequence/state parity and verified shutdown.
[Current comparison](comparison.json), [summary CSV](comparison.csv) and per-seed
CSV curves are updated as cells finish. Partial groups have no mean/gate.
