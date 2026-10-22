# Recurrent cash availability research

Status: protocol and hybrid distribution/policy prepared; integration and actual
Colab training remain. No runtime has been allocated for this study.

## Evidence and hypothesis

The [saved alignment risk/action audit](../results/alignment-risk-actions-2026-10-22/README.md)
finds nonzero policy exposure requests throughout the accepted development paths.
Gaussian sign mapping can only request cash at exact zero. Aligned unit training
failed both return and drawdown gates, with losses beyond its cost drag. This
motivates a distinct action-distribution hypothesis: an explicit learned cash
category may reduce harmful compulsory exposure. It does not establish that cash
caused the failure or guarantee improvement. Prior failed families remain closed.

## Fixed comparison

The checksummed [protocol](../configs/experiments/recurrent_cash.json) fixes:

| Arm | Direction distribution | Training exposure | Evaluation exposure |
| --- | --- | --- | --- |
| posthoc_unit | Gaussian | Adaptive | Unit sign |
| gaussian_unit | Gaussian | Unit sign | Unit sign |
| categorical_binary | Categorical short/long | Exact −1/+1 | Exact −1/+1 |
| categorical_cash candidate | Categorical short/cash/long | Exact −1/0/+1 | Exact −1/0/+1 |

The candidate versus binary estimates the cash-availability contrast. Binary
versus aligned Gaussian describes the distribution contrast. The strongest
posthoc control prevents promotion against a weak baseline. Each hybrid head
has six outputs: three direction logits and three Gaussian protection means,
plus three learned log deviations. Binary omits the middle logit from its actual
categorical support, likelihood and entropy. It does not replace sampled cash
with another action. Binary/cash initial tensors match exactly; Gaussian initial
tensors match within their pair. Cross-family initial tensor equality is neither
possible nor claimed. Record parameter counts and fingerprints explicitly.

The environment still receives four components: signed target exposure, stop,
near target and zone width. Hybrid direction is sampled categorically; independent
Gaussian protections retain standard SB3 clipping at the environment boundary.
PPO scores the stored original protection samples and exact category with the
joint log likelihood. Entropy sums both distributions. Cash follows the same
next-session execution, protective preemption, costs and constraints as every
other action. A cash request is not a promise of an immediate flat position.

## Unchanged conditions and gates

All four fresh runs use expanded111 training symbols, original26 development
symbols, frozen original normalization, masked21 observations, separate64-unit
one-layer actor/critic LSTMs,64×64 heads, seed11,100k transitions,1000-step rollouts,
batch50,ten epochs, gamma0.99 and net-log-equity rewards. Evaluate/save curves at
10k intervals. Actual sampling counts must match. No future losing-stock labels
enter training, rewards, sampling or selection; report joint positive return and
positive net short P&L separately. Reused2021–2023 is development. Retired
2024–September2026 remains unopened; survivor-data limitations remain.

Candidate selected AND final mean returns must each beat EVERY control by2pp;
final stock/ETF gains must be nonnegative and mean drawdown worsening≤2pp versus
each. Selected AND final stock excess, ETF excess and median stock excess must
each be≥−2pp versus matched holding. Failure closes this fixed family: no extra
steps, cash thresholds or entropy/architecture grids. Only a passing screen
permits fixed11/23/37 paired confirmation with the protocol's absolute targets,
paired gains and drawdown limits. Frozen prospective evidence remains required.
Neither a favorable checkpoint nor these repeatedly reused bars prove success.

## Acceptance and operation

Construction checks run locally without fitting. Before the screen, four512-step
Colab CPU smokes must prove actual hybrid sampled support, sampled voluntary cash,
finite losses and changed weights; matched within-family initialization and
all-arm sampling; exact selected/final identities; frozen action/hidden/cell/reset
parity and immutable normalization. Actual Colab Python must match before syncing
dependencies. Screen integration must bind distribution and cash availability in
plans, workers, archive verification, metadata and frozen loading. It is not yet
implemented; the existing research runner does not select this policy.

Use the original centralized fleet ledger, one-minute metering,7200-second session
deadlines and a4-unit incremental study allowance including setup/idle/retrieval/
failures. Project150/130 and account150/170 protections both apply. Retrieve and
verify artifacts, stop/verify all owned sessions, publish checksummed evidence
before removing duplicates. Never auto-restart after a budget stop. No orders.

## Current evidence

`rl/recurrent_ppo/hybrid.py` provides the mixed distribution and recurrent policy.
Tests check analytical joint likelihood/entropy, sampled action support, invalid
categories, zero binary cash-logit gradient, matching initialization, optimizer
parameter ownership, actual recurrent forward/evaluate likelihood agreement and
save/load action/state parity. Backpropagation tests inspect derivatives only;
no local optimizer updates or model fitting occur. Actual Colab smoke updates
remain mandatory before any performance claim.
