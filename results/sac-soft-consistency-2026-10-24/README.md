# Saved SAC soft-value consistency — 2026-10-24

**The descriptive inconsistency flag passes. This is not a trading-return result.**
The final full-execution SAC seed11 policy from the completed50k state comparison
was evaluated without fitting on64 fixed flat-account training reset states, with
four fixed stochastic repetitions each. No development/test policy was scored.

| Diagnostic quantity | Result |
| --- | ---: |
| Mean initial minimum critic Q | 4.031877 |
| Mean multistep soft target | 0.234971 |
| Mean Q minus soft target | 3.796906 |
| Conditional window sampling standard error | 0.085922 |
| Descriptive mean minus two SE | 3.625062 |
| Positive window-mean gaps | 64/64 |
| Full512-step trajectories | 204/256 (79.69%) |
| Fixed flag: lower band≥0.1 and full horizon≥75% | Pass |

Values are **soft-reward units, not profit percentages**. The target excludes
initial-action entropy, includes future-policy entropy, and uses the saved target
critics for timeout/data-cutoff bootstrapping. True termination removes the tail.
The final saved alpha is0.00001506184435. Shorter training tails remain included;
no states or seeds were replaced. Conditional sampling uncertainty describes the
fixed historical reset distribution, not future markets.

The learned bootstrap prevents an unbiased true-Q calibration claim. Flat reset
states do not establish replay-wide or held-position calibration. This result
does not establish wrong action gradients, a cause of failed returns, a profitable
TQC model, or robust benchmark performance. It supports preregistering the distinct
critic hypothesis described in the [research review](../../docs/distributional-critics.md),
with native SAC and untruncated distributional controls. Closed SAC parameter
families remain closed. No financial feature, reward change or model promotion.

## Source and verification

The [protocol](../../configs/experiments/sac_soft_consistency.json) was committed
before execution in`aa1e7d1`. Exact model/metadata/source-state hashes identify the
source run; its original source revision is`bb0f3e25bcf60765df158401434959458508da7c`
and original immutable dataset is`672a21621091a56c1c7beb30557464e7edd8df38874494659bab2c83738e1bf9`.
The worker verified and imported the retained original package rather than using
the changed current environment. Original simulator costs, normalization, action
contract,512-step truncation and legacy-uniform training starts were preserved.
Only training episodes through2020 were rolled out; the source dataset contains
train/development only and no retired-test rows.

[Summary](summary.json) records inputs, conditional aggregate and fingerprints.
[Verification](verification.json) checks all64 durable window artifacts/256
trajectories, target/gap arithmetic, source hashes, aggregates and the fixed flag.
All weights, three optimizer states, update count, normalization and Torch/NumPy
RNG remained unchanged. Worker handle56325 exited0 after38.6seconds; the environment
closed and no task temporary files remain. Durable window records are in
`artifacts/sac-soft-consistency-v1/`; journal is
`.runtime/sac-soft-consistency-v1.json`. Existing paths reject automatic reruns.
11 focused tests/Ruff passed before execution;199 combined source/financial/
soft-consistency checks pass afterward. No optimizer steps, training, Colab
allocation, checkpoint selection or retired-test evaluation occurred. Account
usage remains unchanged from the last official349.02unit/zero-rate reading;
cumulative project consumption remains39.01437789/150units.
