# SAC entropy and saved-target diagnosis

Three final50k SAC models and their own locally retained training replay buffers
are inspected without fitting, dataset access or Colab allocation. Each buffer
contains50,000 rows and the saved model49,000 updates. A fixed seed11 selects
4,096 unique replay rows per model; next-action randomness is isolated in a
restored RNG scope. Every policy/critic/target/temperature tensor and all nine
source file hashes remain unchanged. [Evidence](summary.json).

| Source arm | Mean absolute replay reward | Final alpha | Mean replay-action Q, critic1 |
| --- | ---: | ---: | ---: |
| Legacy | 0.005856 | 0.00000989 | 3.9670 |
| Full execution | 0.005863 | 0.00001506 | 4.0332 |
| Masked execution | 0.005498 | 0.00001203 | 4.0148 |

The first10k reward magnitudes average0.00488–0.00509. Logged entropy coefficients
are about0.067 at10k,0.0034 at20k and0.00018 at30k, before declining near1e-5.
Default automatic-temperature initialization was1. Final sampled log densities
are positive (3.09–4.10), and the discounted entropy terms are negative, about
−3.5e-5 to−6.1e-5. It would be wrong to call final entropy a large positive bonus.

Final sampled one-step critic RMSE is0.0082–0.0099 despite Q means near4 and
small financial rewards. The target decomposition separately records immediate
reward, discounted target-critic value and entropy. Time-limit truncations
bootstrap; genuine terminations do not. These are final-policy, final-replay
samples, not reconstruction of early critics or historical optimizer minibatches.

Large Q offsets do not by themselves establish wrong action gradients; an
action-independent offset cancels from the actor's action derivative. Soft Q
values also include entropy and historical learning dynamics, so comparison with
financial reward alone is not a calibration proof. Small TD error is not evidence
of benchmark performance. This diagnosis establishes neither causal optimizer
failure nor profitable trades and changes none of the failed state-study gates.

Six focused numeric tests pass: timeout versus terminal behavior, entropy sign,
component identity, finite-vector and malformed-input rejection. Actual replay
and model inspection passes, with zero optimizer steps. No source artifacts were
changed or removed; project remains36.009692/150, no owned sessions allocated.

Next test one preregistered lower initial temperature against default, with the
full execution observation contract fixed. Keep automatic entropy adaptation,
target entropy, reward/costs, data, steps and networks unchanged. This changes
the exploration/soft-target startup trajectory, not merely cosmetic numerical
units; it is not assumed equivalent to reward scaling. Stop at fixed gates,
retain unsuccessful results, and do not search a temperature grid.

Implementation checked against installed SB3 2.7.0 and its
[official SAC source](https://stable-baselines3.readthedocs.io/en/v2.7.0/_modules/stable_baselines3/sac/sac.html).
