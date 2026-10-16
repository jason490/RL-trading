# Algorithm and recurrent-memory stability study

The [protocol](../../configs/experiments/algorithm_stability.json) fixes SAC, PPO
and RecurrentPPO on basic inputs at 100k steps and seeds 11/23/37. No cells have
started yet. The three algorithms must share fresh clean source/dependency
proofs, data, reward and simulator assumptions. The inspected test interval is
retired and no orders are enabled.

[Saved-path diagnosis](diagnosis-before-study.json) covers all 12 completed 250k
indicator/control runs, using existing selected and final validation reports.
It records source-file hashes, action saturation, exposure and annual return
changes, trading costs and available optimizer diagnostics. This does not run
a new model evaluation or establish a causal optimization mechanism.

The completed indicator experiment is published separately. Its best checkpoints
remain retained; later training did not overwrite those frozen policies.
