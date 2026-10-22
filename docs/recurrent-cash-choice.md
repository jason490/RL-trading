# Recurrent cash availability research

Status: corrected four512 Colab smokes accepted; fixed four100k seed11 screen is next.
Initial failed smoke evidence and all eight verified session stops are retained.

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
plans, workers, archive verification, metadata and frozen loading. Version24
implements these bindings with actual saved optimizer/parameter-count records,
execution counters and mandatory checkpoint identities. The shared training
factory now supports `hybrid_recurrent_ppo` with explicit boolean `cash_enabled`
in policy kwargs; missing/ambiguous settings reject. Training metadata records
`action_distribution`. Frozen loading verifies this against the actual policy,
algorithm, categorical support and head/deviation shapes. Missing hybrid metadata,
wrong cash availability and malformed schemas reject even with valid checksums.
Historical native bundles retain their established compatibility. The saved-run
frozen verifier recognizes the hybrid recurrent algorithm and carries its state.

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

Factory/frozen integration:37 focused checks pass, including existing native/branch
architecture, checkpoint identity and sequence-verifier regressions. Both hybrid
variants have exact untrained frozen recurrent action/state parity. No optimizer
updates occurred; actual Colab smoke acceptance remains outstanding.

## Integrated preflight

Version24 builders/worker/fleet/archive integration, smoke acceptance and restored
screen reporting are prepared.96 focused checks pass, including malformed cash
counters/support, optimizer update evidence, paired initialization, mandatory
checkpoints, account floor, shared metering and cleanup regressions. All four
actual immutable smoke payloads pass validation; scripts compile. No local fit
or runtime allocation occurred. `plan.html` is unchanged.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_cash_jobs.py \
 --smoke --output .runtime/cash-smoke-jobs.json
```

Freeze clean pushed source and inspect official account usage before allocation.
Four512 CPU smokes require312 exact frozen sequence predictions, actual updates
and sampled cash/binary support, paired initial tensors, identical sampling and
verified shutdown. `accept_cash_smokes.py` restores selected/final models;
`report_cash.py` also checks full source-bar/accounting/matched-holding paths and
requires gates against all three controls. No Gaussian/hybrid final-weight
equality is expected because their training execution differs.

Last verified project charge remains38.136658/150 and last account349.71 with
zero assignments/rate. New usage must be checked before provisioning; the original
shared ledger, account150/170 and project150/130 rules remain authoritative.

## Initial smoke failure and correction

The first four512 CPU attempt failed before acceptance: the binary policy
rejected all-zero recurrent padding when sb3-contrib evaluated minibatches before
masking their losses. This is a compatibility bug, not a failed performance gate.
Only all-zero binary evaluation placeholders now receive a valid dummy direction;
valid actions and strict binary distribution/execution support remain unchanged.
A real recurrent-buffer regression exercises unequal sequence lengths, padded
likelihoods, masks and finite gradients without local fitting or optimizer steps.

All four owned sessions stopped and verified; one failure archive retained with
checksums, others had no salvage snapshot. No accepted model/performance claim.
Project38.14814485, attempt conservatively0.01148649; rounded observed debit0,
official account349.70 zero rate/assignments. Original failed state, ledger, source
and four-unit study baseline retained. Corrected source needs a new export/retry
state and repeated four512 smokes before screen.
[Failure evidence](../results/recurrent-cash-2026-10-22/README.md).

## Corrected Colab smokes accepted

Four512 CPU smokes from clean worker70aa505 on Python3.13.16 accepted:312 exact
frozen action/hidden/cell/repeat/reset predictions, immutable normalization and
input rejection, actual saved selected/final optimizer/configuration restoration,
current checkpoint identities and equal sampling. Initial tensors match within
Gaussian and categorical pairs; cross-family equality is not claimed. Cash arm
requests205 cash/307 directional samples, binary512 directional/zero cash; both
perform40 optimizer updates, have21 optimizer parameter states and changed weights.
No sizing/protection transforms occur in categorical arms. These are wiring
checks, not performance evidence or settings selection.

All four retry sessions stopped/verified (eight including initial failure).
Project38.18725015/150; study increment0.05059178 including failure, remaining
111.81274985. Account349.67 zero rate/assignments. Source, original ledger and
failure evidence retained. Fixed same-source four100k seed11 screen next within
original4-unit study allowance; no tuning from smoke returns.

[Evidence](../results/recurrent-cash-2026-10-22/README.md).
