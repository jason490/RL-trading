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

## Saved temperature-path outcome

Candidate selected-to-final stock/ETF net return changes -10.47/-6.25pp split
into long -7.34/-3.45pp and short -3.13/-2.80pp. Cost drag rises only0.91pp
for stocks and falls0.75pp for ETFs. Saved-path non-cost changes -9.56/-7.00pp
show costs alone cannot account for deterioration; exposure rises while both
side contributions worsen. This is descriptive accounting, not a causal
cost-free replay or proof of an optimization mechanism. No fitting/data access/
Colab and source hashes unchanged;11 existing numeric/contract checks pass.
[Evidence](../results/sac-temperature-paths-2026-10-20/README.md).

Before new allocation, audit existing TD3 input/exploration contracts and whether
a fixed full-state delayed-update comparison provides a distinct hypothesis.
Prior weak TD3 results stay visible. No promise of improvement, automatic
allocation, temperature rescue or relaxation of stock/ETF/short/prospective
requirements. Project remains36.069198/150.

## TD3 audit and fixed protocol

Three owned 50k seeds verify legacy12 inputs, Gaussian sigma0.1 and delayed
updates; no training/data access/Colab. Final stocks/ETFs -0.08%/-0.37% remain
weak. Fixed legacy/masked/full TD3 comparison preregistered with unchanged data,
reward, costs, parameters and strict absolute/relative gates;12 protocol/off-policy
checks pass. No allocation before runtime/initial-noise preflight.
[Category guide](td3-execution-state.md),
[audit](../results/td3-diagnostic-2026-10-20/README.md).

## TD3 state screen outcome

Full TD3 selected/final6.19% fails fixed controls and absolute stock/ETF gates;
all arms achieve0/3 joint profitable-short cases. No more steps/confirmation.
468 exact restored predictions,6 owned stops,13 verified release assets and
post-cleanup byte-identical report pass; project36.149107/150.
[Outcome and next fit-versus-generalization diagnostic](td3-execution-state.md#completed-td3-state-comparison).

## Fixed fitted-period diagnosis

Before new fitting, replay only final50k full-state PPO/SAC/TD3 policies on the
fixed2018–2020 training window, preserving original purge cutoff and train
labels. Compare per-period matched excess, not raw returns across regimes.
No checkpoint selection or algorithm ranking; PPO initialization differs.
Twenty slice/isolation/accounting checks pass; actual replay not yet executed.
[Protocol and commands](training-fit-diagnostic.md).

Training-fit replay completed without fitting or Colab: SAC passes fitted-window
comparability but fails development; PPO/TD3 fail fitted flags. All source,
weights, update counts and normalization remain unchanged. Three release assets
verified, complete derived paths retained. Project36.149107/150.
[Evidence and training-diversity audit next](training-fit-diagnostic.md#actual-inference-only-outcome).

## SAC training-diversity preparation

Cached107-stock plus4-ETF data preserves every original episode and all26
validation episodes exactly. New immutable cdb93a33… fixes original preprocessing
without fitting or modifying sources.18 data/slice tests pass. Fixed protocol
retains22/26 stock and4/26 ETF sampling mass, explicit full-window group cycles
and actual symbol coverage. Fresh equal100k arms allow traversal of107 stocks;
50k cannot. No sampler rescue, return-label weighting or relaxed development
gates. Prior PPO expansion failures remain negative evidence. Runtime/sampler
integration and Colab smokes pending, no allocation; project36.149107/150.
[Guide](sac-diversity.md),
[audit](../results/sac-diversity-data-2026-10-20/README.md).

## Completed training-diversity comparison

Both100k seed11 Colab CPU screens are complete. Original selected/final mean
returns24.35%/10.38%; expanded24.19%/24.19%; matched holding30.60%.
Expanded final stocks26.06%/ETFs13.88% trail holding32.63%/19.47%; median
stock excess−5.49pp. Selected gain−0.16pp fails the required2pp. All absolute
gates fail and expanded joint profitable-total/short cases are0/3. No extension,
confirmation seeds or promotion under this protocol.

All training symbols receive transitions. Intended ETF probability matches,
but actual ETF transitions differ15.36% versus19.97%; added stocks alone cannot
explain the observed return difference. Both survivor-data and repeatedly-used
development limitations remain. All312 frozen prefix predictions are exact,
all4 owned sessions stopped,14 release asset digests verified. Duplicate cleanup
removes16 files/248,893,520 bytes with643 retained hashes unchanged; the complete
report reproduces byte for byte afterward. Thirty-six focused sampling/job/
archive/gate/report tests pass. Project usage36.263895/150, remaining113.736105;
post-shutdown account350.31, zero rate/assignments. Retired test stays isolated.
[Evidence and interactive artifacts](../results/sac-diversity-2026-10-20/README.md).

Next inspect saved selected/final cost, exposure and directional contributions
before a distinct hypothesis. This requires no fitting/provider/Colab; do not
extend the failed family solely because its late checkpoint improved.

Saved diversity-path diagnosis finds expanded stock/ETF non-cost gaps3.05/2.93pp
plus incremental costs3.51/2.66pp; small resize costs only0.23/0.21pp. Mostly-long
behavior with very little short exposure persists. No training extension or
cost-only cure is established. Next inspect strictly training-only decline/
rebound decision coverage before freezing another hypothesis; no future decline
labels may enter learning or checkpoint selection.
[Full evidence](../results/sac-diversity-paths-2026-10-20/README.md).

Final100k diversity fitted-window replay finds original91.72%/expanded62.43%
versus holding56.45%, both training-comparable while development fails.
Expanded remains more long-biased in fitted persistent-decline cohorts. This
supports a fixed generalization/regularization hypothesis over blind extra
steps, without proving the mechanism. Next preregister one regularization arm
against fresh identical expanded-data control, preserving seeded actual sampling,
initialization, reward/entropy/costs and absolute/relative gates. No sweep or
extension of the failed diversity family.
[Inference-only evidence](../results/sac-diversity-fit-2026-10-20/README.md).

Fixed actor/critic Adam L2 study preregistered:control0 versus1e-4, identical
expanded111 dataset/sampling, fresh100k cells and unchanged entropy optimizer.
Six actual-construction/protocol checks pass without fitting. Runtime/archive/
report integration must precede Colab smokes; no allocation yet.
[Protocol and gates](sac-regularization.md).

## Regularization screen interruption

Fresh control finishes21.63%; candidate latest80k curve5.10% lacks100k final.
Metering outage caused protective budget stop; do not substitute partial results
for fixed endpoints, promote, extend or restart. Exact-fleet recovery now verifies
all6 attempt sessions stopped;13 asset digests and656 retained hashes pass.
Project conservative37.787213/150 retains gap estimate (screen measured0.32,
estimated1.489616). Budget-stop flag stays set; additional allocation requires
explicit reviewed authorization. Offline evidence diagnostics remain possible.
[Full incomplete evidence](../results/sac-regularization-2026-10-20/README.md).

Interruption evidence fix completed offline: new snapshots bind latest/selected
weights and validation paths to exact steps/phases/checksums, with a separate
current JSON report on every checkpoint.33 checks cover worsening snapshots,
corruption and final markers without fitting. Do not relabel old partial reports;
existing interruption outcome remains unchanged. No allocation/stop-flag reset.
Next offline work may diagnose retained partial policy behavior; new Colab runs
still require explicit review after the metering stop.

Offline numerical audit finds5942 subnormal regularized-policy elements versus0
control. Same-host fixed synthetic forward comparison changes14.10× versus1.02×
with flush arithmetic, outputs exact on that batch and source/weights/updates
unchanged. This supports a possible speed mechanism, not proof of training
causality, return improvement or all-input parity. No local backward/optimizer
steps or Colab. Training numerical settings remain unchanged; explicit reviewed
authorization is still needed before allocation after the metering stop.
[Evidence and profiling limits](numerical-performance.md).

Fixed retained80k diagnostic replay reveals long/flat behavior,0/3 joint short
successes and stock/ETF returns6.92%/−4.87%. Cost and non-cost gaps both matter.
Replay misses the declared1e-10 score tolerance by2.21088e-8; preserve that failure
and no parity claim despite only$0.00221 mean equity difference. Original study
remains incomplete.20 accounting checks pass, no fitting/Colab or stop reset.
[Evidence](../results/regularization-80k-replay-2026-10-21/README.md).

## Reviewed resumption and additional account protection

User reactivated the blocked goal and explicitly permits further research while
account credits stay≥150. Reviewed resumption records that authorization,
healthy idle349.97 balance/zero assignments and verified cleanup; preserves all
37.787213 conservative units and prior failure records. Account teardown170 with
20 reserve complements original project150/130 ceilings.20 budget/fleet tests
pass. No session allocated at review. New fixed studies require new frozen source
with floor enforcement; old interrupted family remains incomplete, no extension.
[Audit](../results/budget-review-2026-10-21/README.md).

## Fixed discount-horizon hypothesis

After reviewed resumption, preregister a distinct SAC gamma0.99/0.95 comparison.
Both fresh expanded-data arms keep actor/critic/alpha initialization, reward,
costs and actual sampling matched; no weight decay or numerical flush change.
Shorter discounted targets are a hypothesis, not equivalence to20-day holding or
proof of a cure. Mandatory checkpoint identities and floor-aware controllers
precede two1024 CPU smokes, then fixed100k screen only after acceptance. Keep
absolute/relative selected/final, seed and prospective gates. Project37.787213
retained, study allowance2 includes all attempts; account floor150/teardown170.
[Protocol and controls](sac-discount.md).

## Completed discount screen — gates failed

Fresh100k CPU candidate gamma0.95 selected/final22.79%/22.16% trails gamma0.99
control24.19%/24.19% and matched holding30.60%. Final stocks23.88% and ETFs12.64%
trail holding32.63%/19.47%; median stock excess−9.78pp. Candidate joint short
success1/3 versus control0/3 does not rescue failed return gates. No extension,
gamma sweep, confirmation seeds or promotion. Reused development is not independent
evidence; retired test remains unopened and the overall objective is unmet.

All4 owned sessions stopped;312 exact frozen predictions and35 focused checks
pass.13 server asset digests verified; duplicate publication archives totaling
31,529,571 bytes removed with398 retained file hashes unchanged. Strict report
reproduces byte-for-byte. Project37.897158/150, remaining112.102842; last account
349.88 with zero assignments/rate. Account150/170 and project150/130 protections
remain. Next inspect saved selected/final cost and directional contributions,
without fitting or allocation, before any distinct hypothesis.
[Detailed evidence and release](../results/sac-discount-2026-10-21/README.md).
