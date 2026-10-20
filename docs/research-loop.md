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

## Participation/cost diagnosis and next hypothesis

The saved-path audit covers5 recent50k PPO models and both phases, with no
dataset access, training or Colab. Native stock/ETF shortfalls7.49/10.83pp split
into4.85/3.88pp incremental cost drag and2.65/6.95pp non-cost gaps. Full-context
ETF short contribution−6.79pp versus−0.15pp masked accompanies lower costs.
Sub5%-equity resize costs are only0.05–0.12pp, so a small deadband is low priority.
Holding-limit exits remain required. Next investigate strictly training-only
supervised signal/warm-start feasibility, with chronological target completion,
purging and independent preprocessing before allocation. This proposes no new
training yet; future validation decline labels remain reporting-only.
[Evidence](../results/participation-diagnostic-2026-10-20/README.md).

## Supervised feasibility outcome

Neither fixed MLP arm passes predictive gates; errors exceed the constant by
5.52% masked and12.56% full. Positive daily rank correlation around0.04 does not
establish accurate level forecasts or profitable trades. Ridge error is about
2.7% better but its rank correlation is below0.02. No PPO transfer or more MLP
updates follow. All6 predictors restore, the owned session stopped, and12 assets
passed publication checks. Project35.861240/150. Fixed forecast moments find
negative pooled prediction/target covariance for every MLP despite positive
within-day ranking. Next preregister a causal same-session common-component
diagnostic on fixed forecasts before any new fitting; retain the original failed
gate and the internal probe's development-only status.

## Fixed common-component outcome

The preregistered no-fit transformation improves raw MLP MSE7.39%/12.96% but
still loses to ridge by0.38%/0.70%, with zero passing seeds and yearly failures.
All availability/source checks pass and ranks remain unchanged. No promotion,
new training or Colab units. Stop this centering family. Next investigate a
distinct tabular tree ensemble, checking compatible dependencies and CPU
acceptance before a new committed fixed comparison. Preserve the same purged
targets and do not reinterpret forecast error as stock-trading performance.

## Forest outcome and next data-coverage diagnosis

The fixed masked/full forests fail (16.23%/17.47% worse than own ridge), with
zero passing seeds. No PPO transfer. A saved-forecast decomposition finds
negative2020 forecast/target covariance in both arms, dominated by common-day
error, with small pooled bias. Negative forecast precision is22.94%/25.36% in
2020 versus39.20% negative gross targets. Neither forecast signs nor this ex-post
cohort establishes profitable shorting. No forest centering or extra trees by
default. [Complete diagnosis](../results/forest-signal-2026-10-20/README.md).

Next inspect chronological training-regime coverage and feasibility of a fixed
pre-2010 ETF history extension, with explicit provider/corporate-action/coverage
acceptance before merging or fitting. Use chronology, never future losing-stock
labels, to define inclusion. A later matched study would need to distinguish
extra market history from extra ETF sampling and transition budget. Keep retired
isolation, predefined stock/ETF gates and prospective confirmation. Project usage
remains35.871240/150; this saved-path diagnosis adds no allocation or training.

## Historical augmentation preparation

Older source/overlap checks pass. Three immutable original/recent-ETF/earlier-ETF
arms preserve existing observations/normalization/validation; extraETF mass is
matched in augmented arms. Full training windows avoid near-end truncation
confounds; actual group transitions are logged. The fixed protocol requires
beating both controls and passing absolute stock/ETF gates before confirmation,
not merely improving an already weak control. Complete metered job/artifact
integration and source freeze before threeCPU smokes. No fitting or credits
used at preparation; [protocol](../configs/experiments/historical_augmentation.json).

## Historical augmentation outcome

Chronological oldETF augmentation improves final17.06% versus original15.01%
but loses selected−2.17pp and misses the recent-control2pp gain. Stock/ETF
benchmarks and median targets fail; joint falling-stock total/short success1/3.
Exact augmented symbol transition counts remove one sampling confound, not
market uncertainty. Stop this family; no longer runs or confirmation. All6
owned sessions stopped,468 parity checks and15 release assets pass; project
35.917535/150. [Evidence](../results/historical-augmentation-2026-10-20/README.md).

Before another allocation, inspect saved off-policy SAC behavior and whether
its previous experiments included complete execution state. A separately fixed
full-state off-policy comparison may test a different optimization mechanism;
a past selected peak alone is not evidence for promotion. Keep the existing
base data, causal inputs and short-side accounting rather than adding another
feature/data family without a diagnosis. Any new comparison must retain
selected/final stock/ETF and drawdown gates, multiple seeds when justified, and
frozen prospective confirmation.

## SAC diagnostic and next fixed hypothesis

All three saved100k SAC models actually have12 legacy inputs, omitting active
protection/partial-exit/previous-risk state. Six saved phases reconcile without
opening datasets; final mostly-long behavior and costs do not explain the
entire holding gap. Missing state is verified, its performance effect is not.
The fixed SAC legacy/masked/full protocol tests that effect with unchanged base
data/reward/costs and equal50k cells. Require both controls and absolute selected/
final stock/ETF gates before confirmation. No assumption that failed PPO state
ablations imply SAC success. Ten checks pass; project35.917535/150, no new units.
[Protocol](../configs/experiments/sac_execution_state.json).

## SAC execution-state outcome

The fixed full-state SAC screen fails: selected/final7.93% versus
legacy7.44%/7.39%, masked17.84%/17.69%, holding30.60%. Full final stock8.68%/ETF3.77%
and median stock excess−17.40pp fail. Extra execution state did not repair this
configuration; avoid universal claims about information value. Drawdown improves
but cannot override return gates. Joint falling-stock total/short positives1/3.
No confirmation or more steps.468 exact restored predictions,6 owned stops,13
verified assets; project36.009692/150.
[Complete evidence](../results/sac-execution-state-2026-10-20/README.md).

Next inspect saved SAC entropy coefficients, losses and net-reward magnitudes
before another protocol. Do not assume feature insufficiency, optimizer failure
or that reducing costs alone fixes the non-cost gap. If considering numerical
reward scaling, distinguish scalar conditioning from changes in SAC's effective
entropy/reward balance; it is not automatically an unchanged learning objective.
Require a new fixed matched hypothesis and the same benchmark/short-side gates
before spending more units. No new allocation is implied by this diagnostic.

## SAC learning-scale diagnostic and next hypothesis

Saved50k replay rewards average0.0055–0.0059 absolute; alpha starts1, logs~0.067
at10k and ends~1e-5. Final critic values~4 with small one-step residuals do not
prove wrong gradients or financial calibration. Final entropy contribution is
small/negative, so do not call final exploration excessively rewarded. No early
critic reconstruction, optimizer steps or source mutation occurred.

Test one preregistered initial learned-alpha0.001 versus default1, keeping full
state, target entropy, rewardscale1 and all other data/parameters/costs fixed.
This changes startup exploration/soft targets, not merely units. No temperature
sweep. Require fresh default reproduction and selected/final stock/ETF gates
before further seeds. Original failed state study remains failed. No allocation
yet; project36.009692/150.
[Protocol](../configs/experiments/sac_temperature.json).

## Completed initial-temperature comparison

The two 50k SAC seed-11 screens are complete and fail the preregistered gates.
Initial alpha 0.001 improves selected mean return to 19.67% versus default
7.93%, but final falls to 9.84%; the final gain of 1.92pp misses 2pp and
drawdown worsens 4.27pp. Final stocks/ETFs 10.78%/4.72% trail matched holding
32.63%/19.47%; joint falling-stock total/short positives are 1/3 selected,
0/3 final. No extension, confirmation seeds, temperature sweep or promotion.

Fresh default configuration, all checkpoint scores and selected/final network
tensors reproduce the prior full-state SAC exactly. Initial network hashes
match across arms; 312 smoke/screen selected-bundle prefix predictions are exact.
All four owned sessions stopped. Thirteen release assets have verified server
digests; duplicate-only cleanup removes 15 files (167,684,918 bytes), preserving
354 file hashes including models, replay, data, source, ledger and plan.html.
The post-cleanup report reproduces byte for byte. Eighteen focused temperature
and report-contract checks pass. Cumulative project 36.069198/150; account
350.48 with zero assignments/rate after shutdown. Retired test stays isolated.
[Complete evidence](../results/sac-temperature-2026-10-20/README.md).

Next inspect saved selected/final action paths, directional contributions,
exposure and cost differences to distinguish deterioration mechanisms before
preregistering another bounded hypothesis. This diagnostic needs no fitting,
provider calls or Colab allocation. Development results remain repeatedly used;
prospective confirmation and the overall benchmark/short objective remain unmet.
