# Bounded autonomous research

The user authorized continuing research toward returns comparable to or better
than holding the same stocks and ETFs, including causal indicators, different
models, more data, training and carefully tested reward changes. This is a
research objective, not a profitability guarantee. The total project ceiling
remains 150 Colab units, with teardown beginning by 130; authorization does not
reset the ledger or permit broker orders.

## Efficient order of work

1. Diagnose existing saved paths before training again. Reuse immutable raw
   snapshots and selected/final reports. Include short-side attribution and
   causal constant/trend baselines under the same execution rules to distinguish
   model failures from execution constraints.
2. Predeclare each bounded experiment with one hypothesis, dataset identities,
   costs, action/reward contracts, seeds, steps, checkpoint cadence, promotion
   gate and maximum cost/duration. Keep negative results in the registry.
3. Screen a small fixed budget, then confirm promising changes on seeds
   11/23/37 with a matched control. More training follows evidence of stable
   final performance; longer runs are not the default response to deterioration.
4. Expand the training universe only with explicit provider/action/coverage
   acceptance. The 107-stock complete Yahoo panel can be studied as retrospective
   survivor research. EA/AVB/EQR's separate partial Alpaca history must not be
   silently merged. Historical membership and delisting outcomes remain gated.
5. Confirm a frozen candidate prospectively or on a separately preregistered
   untouched interval. Preserve retired-test isolation. Publish verified
   artifacts and stop/verify every owned runtime before cleaning duplicates.

## Development targets

Before starting a new search, freeze its numeric protocol. Default screening
targets are mean excess return of at least -2 percentage points for **both**
stocks and broad ETFs separately, on the same symbols and execution dates.
Outperformance requires nonnegative excess in both groups. Confirmation should
meet the declared threshold in at least two of three seeds, with median stock
excess at least -2 points and no more than 2 points of additional mean maximum
drawdown. Evaluate both selected and final weights; a temporary best checkpoint
alone is insufficient. These are development gates, not statistical proof.

Report the [short-side objective](short-side.md) separately: a negative holding
return must not be described as profitable shorting unless policy return and
short net P&L are both positive. Full-period declining-stock labels are for
reporting only, never checkpoint selection or training. Do not claim that every
falling stock will be profitable.

The 2021–2023 interval has already guided repeated research and is development
validation. The previously inspected 2024–September 2026 test is retired and
must not select new checkpoints. New indicators and more optimization over the
same bars do not create fresh evidence. A goal cannot be marked achieved solely
because one selected validation result clears a gate.

## Token-efficient operation

- Read `AGENTS.md`, this guide and the current experiment's compact protocol,
  status and summary first. Search targeted functions instead of repeatedly
  dumping whole files, historical conversations or large manifests.
- Persist exact hypotheses, source revision, dataset hashes, run IDs, owned
  session IDs, deadlines, usage and next action in machine-readable local state.
  Keep credentials and authenticated logs out of tracked results.
- Reuse caches and established runners; avoid reimplementing budget monitoring,
  polling loops and artifact verification per experiment. Aggregate status
  changes and metrics rather than streaming full logs. Meter every minute even
  when nothing useful changed; quiet status output must not disable monitoring.
- Retain compact learning curves at the declared checkpoint cadence (normally
  every 10k steps), selected/final weights and failure evidence. Retrieve durable
  artifacts before teardown, verify release checksums before duplicate cleanup,
  and preserve the ledger across retries and context compaction.
- Batch independent inspections, bound searches/retries/experiments, and stop an
  unsuccessful family at its declared gate. Do not spend tokens or credits
  repeating a completed comparison without a new testable hypothesis.

If the working budget, reliable metering, data integrity or prospective evidence
blocks further work, record the exact limitation and next feasible action.
Do not restart automatically after a budget stop or imply the objective passed.

## Initial diagnostic results

The [short-side audit](../results/short-side-2026-10-19/README.md) confirms
inconsistent profits from shorts in existing models. The subsequent
[causal execution diagnostic](../results/execution-rules-2026-10-19/README.md)
finds fixed 95% long exposure returns 27.61% on stocks and 11.64% on broad ETFs,
versus holding 32.63% and 19.47%. A simple EMA trend/flat rule loses 16.26% and
29.59%; do not assume it is a useful imitation teacher. These local simulations
use the original 22-stock/four-ETF panel, no new training or retired-test data.
Next research should preregister a matched exploration/action-initialization
diagnosis, retaining short capability and the original net-equity objective.

## Confirmed initialization result and next hypothesis

The completed matched 50k PPO study improves selected/final means from
11.77%/10.49% to 14.23%/13.89%, with final gains in all three seeds. Absolute
stock/ETF targets fail, and declining-stock final returns worsen from -12.20%
to -16.50%, with zero of nine wide observations earning positive total returns
and positive short P&L together. Retain this as an overall-return research
baseline, not proof of profitable shorts or a reason for blind longer training.

Next predeclare a causal down-regime sampling experiment, potentially paired
with the verified 107-stock training universe. The expanded training-only audit
has 5.15% persistent trailing-decline observations versus 2.31% in the original
panel. Sampling labels must use only already-observed prices, with matched
eligibility and episode lengths across controls/candidates. Full-period declining
validation-stock labels remain reporting-only. Freeze the next protocol before
allocation, test sampling isolation/RNG behavior, and retain the unchanged shared
project ledger. [Current result](../results/action-initialization-2026-10-19/README.md)
records verified publication, cleanup and remaining budget.

## Sampling outcome and execution observability

The causal start/universe screen failed its ETF constraint; no confirmation
was run. Oversampling severe past declines increased downtrend transitions but
did not establish a profitable short policy. Training-only 20-session outcomes
were positive on average after those starts, consistent with rebound buying.
Do not turn future recovery outcomes into sampling labels.

A concrete synthetic witness finds identical visible market/account vectors
with different active protection and different next-step outcomes. Next test
explicit, versioned execution state: actual stop/targets, partial-exit status,
last protective outputs and share-unit scale. Include masked same-dimension and
legacy controls, preserve existing frozen bundles, and keep actions/reward/costs
fixed. Diagnose this known observability limitation before another blind training
extension. [Sampling evidence](../results/causal-sampling-2026-10-19/README.md)
contains the witness and complete negative results.

## Current direction diagnostic and next hypothesis

Universal restrictions on fixed wide PPO models confirm some short profits
(6/9 declining-stock/seed episodes) but short-only loses 1.19% on the full
panel. Long-only improves to 15.49%, still far below 30.60% holding. A universal
filter is not an all-market solution. Full-state policy requests change direction
on 14.70% of common unrestricted/short-only decisions before mechanical clipping.
Next test a market-only exposure head with account-aware learned protection and
critic, against same-capacity unmasked and native controls. This is a proposed
architecture ablation, not an established cure. Freeze its numeric protocol and
verify masks/legacy inference before Colab allocation. The completed diagnostic
adds no Colab usage; project remains 35.687823/150.

## Reward objective outcome

The2026-10-20 matched seed11 screen rejects initial-capital equity changes:
selected23.38%/final20.43% versus unchanged log control22.59%, holding30.60%.
Both final groups worsen; no declining stock achieves joint positive total and
short P&L. No confirmation or further extension. All4 CPU allocations stopped,
312 exact frozen predictions pass; project35.795378/150. The control reproduces
prior tensors/checkpoints exactly. Finite losses and modest finalKL do not prove
an optimizer failure. Next diagnose longer-horizon causal context before any
new preregistered feature/control experiment.

## Longer-horizon context outcome and next diagnosis

Explicit63/252-session price returns and252-close drawdown do not pass the
matched50k screen. Full final13.89% versus same-dimension masked13.93% and
matched-warmup baseline0.83%; final ETF3.07% versus8.91% masked. Both control
gates fail, full joint short/total success0/3, no confirmation. Retain default
features. All6 CPU sessions stopped;468 frozen predictions and14 release asset
checks pass, project35.856639/150. Next inspect saved exposure/turnover/cost and
protective-exit behavior to identify missed market participation versus signal
errors before preregistering another change. Do not infer that a gain against
the smaller baseline alone establishes feature value.
