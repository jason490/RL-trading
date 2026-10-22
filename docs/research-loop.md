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

Saved discount-path diagnosis finds candidate final stock/ETF excess−8.74/−6.82pp
equals non-cost excess−4.73/−3.94 minus cost drag4.01/2.89pp. Whole-panel shorts
contribute−1.42%/−1.30% despite positive shorts on reporting-only decliners.
Both costs and participation matter; no optimizer-cause claim or more-step/gamma
sweep follows.12 accounting/behavior/report checks pass, source hashes unchanged,
no fitting/data access/Colab. Next audit existing recurrent input/memory contracts
before any distinct protocol. [Evidence](../results/sac-discount-paths-2026-10-21/README.md).

## Recurrent input and memory audit

Six retained100k recurrent selected/final models use legacy12 inputs, omitting9
explicit execution fields.96 fixed synthetic sequence observations show exact
direct-call/stream parity and reset replay; resetting each observation changes
actions, so memory is active. One regression passes, weights/updates/source
hashes unchanged; no fitting/dataset/test/Colab access. Missing explicit state
is a distinct testable hypothesis, not proof of the previous performance cause.
Next freeze a matched recurrent masked21/full21 comparison with feed-forward
control and unchanged absolute/relative gates before smoke acceptance.
[Audit](../results/recurrent-contract-2026-10-21/README.md).

## Recurrent full-state protocol prepared

Distinct fixed three-arm comparison: recurrent_full candidate, recurrent_masked
and feedforward_full controls; expanded111 data, unchanged normalization/sampler/
reward/costs, fresh100k seed11 screen after three512 CPU smokes. Same recurrent
initial tensors; feed-forward is not parameter-count matched. Both relative and
absolute selected/final gates required before three-seed confirmation/prospective
evidence. Seven actual-construction/save-load/sequence/reset/gate checks pass
without fitting. No allocation: job/worker/archive/restored-report integration
remains required. Project37.897158 retained; incremental study allowance3 includes
all attempts, account150/170 and project150/130 enforced at future allocation.
[Protocol and status](recurrent-execution-state.md).

Recurrent version22 remote integration complete: fixed plans, actual dataset
preflight, floor-aware controller, saved-model fingerprints, mandatory checkpoint
identity, strict archives and restored-weight matched report.69 focused checks
pass; actual three smoke payloads validate, scripts compile. No allocation or
training yet. Freeze/push source and official usage check precede three512 CPU
smokes; frozen sequence/reset acceptance precedes any100k screen.
[Guide](recurrent-execution-state.md).

Actual three512 Colab CPU smokes accepted on Python3.13.16, clean workerdd3b256.234 exact frozen prefix predictions include carried
hidden/cell state, repeats, reset and caller-state immutability; actual saved
steps/settings/tensor fingerprints, equal recurrent initialization and all-arm
sampling pass. All three owned sessions stopped; project37.927940/150, account
349.86 zero rate/assignments. Next fixed three100k seed11 screen on same worker,
no tuning from smoke returns. [Evidence](../results/recurrent-state-2026-10-21/README.md).

## Completed recurrent screen — gates failed

Fresh100k seed11 selected/final means:feed-forward4.46%/3.34%, recurrent masked
24.99%/24.04%, full-state recurrent candidate3.95%/−3.11%, holding30.60%. Candidate
loses0.51/6.45pp to feed-forward and21.05/27.16pp to masked. Candidate final
stock−2.25% versus holding32.63%, ETF−7.84% versus19.47%, median stock excess
−29.25pp. Candidate selected/final joint profitable-short outcomes0/3; masked
1/3 selected,0/3 final. Relative AND absolute gates fail. No extension, sweep,
confirmation or promotion; masked control cannot be retrospectively promoted.

All6 owned smoke/screen sessions stopped and cleanup verified.468 exact frozen
predictions include recurrent action/hidden/cell/reset parity and immutable
normalization. Actual selected/final records/checkpoint identities, equal initial
recurrent tensors, all-arm sampling counts and matched holding paths pass. Project
38.014408/150, remaining111.985592; study increment0.11725006. Official account
349.80, zero rate/assignments; floor150/teardown170 retained. No retired-test use.
Next saved-path cost/directional diagnosis without fitting before a distinct
hypothesis. Overall multi-seed/prospective objective remains unmet.
[Evidence and release](../results/recurrent-state-2026-10-21/README.md).

Recurrent outcome publication accepted:13 server sizes/digests match; duplicate
publication archives47,914,914 bytes removed,439 retained hashes unchanged. Strict
report reproduces byte for byte after cleanup. All6 sessions stopped; all models,
frozen bundles, current checkpoint identities, source, data and ledger retained.
[Audit](../results/recurrent-state-2026-10-21/cleanup.json).

## Saved recurrent path diagnosis

Three100k runs/six phases audited without fitting/data/test/Colab access. Full
recurrent final stock/ETF gaps34.88/27.31pp include5.81/5.20pp costs; weak non-cost
participation dominates. Masked control gaps6.81/5.19pp include6.11/4.73pp costs,
non-cost excess−0.69/−0.47pp. Small-resize costs0.067/0.081pp cannot close gaps.
Whole-panel shorts contribute+0.48%/−1.52% masked,−3.78%/−6.13% full. Cost addback
is descriptive, not a cost-free replay; no cause/profitability claim.15 checks
include explicit mixed algorithm rejection, accounting and source immutability.
Next audit cost by fill reason before any distinct matched cost intervention;
failed state family stays closed. [Evidence](../results/recurrent-state-paths-2026-10-21/README.md).

## Recurrent fill-cost attribution

Observed masked final stock/ETF same-side resizing costs2.13/1.51pp, entries
1.22/0.97, protective+holding exits1.69/1.26, borrow0.64/0.77. Categories reconcile
all fees/impact/borrow and executed quantities; unknown reasons/corruption reject.
Even arithmetic addback of every resize leaves4.68/3.68pp gaps; adding all entries,
resizes, closes and reversals still leaves~2.99/2.47pp. These are saved-path
bookkeeping bounds, not counterfactual profits. Resize-only savings insufficient;
required protections/20-session exits/borrow remain.18 checks pass, source hashes
unchanged, no fitting/data/Colab. Next separate direction from protective-exit
participation before a distinct protocol. [Audit](../results/recurrent-fill-costs-2026-10-21/README.md).

## Saved action/execution relationship

Masked final stock/ETF requested long magnitude74.06%/73.15% versus end-long
67.04%/67.44%; sign agrees93.93%/94.77% of sessions. Flat-after-protection
2.09%/0.92%, flat-after-holding2.84%/3.65%; rejected rows separate. These observed
relationships do not prove a causal protection bottleneck. Nine checks reconcile
action categories and net P&L with unchanged sources; no fit/data/Colab. Next
consider preregistered fixed unit-magnitude sign-preserving replay, with original
protective outputs and exact unrestricted parity gates before interpretation;
no replay yet. [Evidence](../results/recurrent-actions-2026-10-21/README.md).

## Fixed recurrent sizing replay prepared

Preregister four saved masked100k seed11 cells: selected/final unrestricted versus
unit-magnitude sign-preserving exposure. Other three protective outputs, costs,
holding cap and causal recurrent/account updates stay intact. Both unrestricted
paths must pass exact-quantity/bounded-numeric parity before any candidate replay.
Fixed relative+absolute selected/final gates, no threshold sweep or promotion.
12 filter/state/reference checks pass; no actual replay/fitting/Colab yet. Freeze
clean source before execution. [Protocol](recurrent-sizing.md).

## Completed fixed sizing replay — full gates failed

Both unrestricted phases pass preregistered tolerances with exact quantities;
reports are not byte-identical (maximum equity differences$0.013815/$0.005020).
Four cells complete with unchanged weights/updates/normalization. Unit selected/
final means28.53%/32.80% versus control24.99%/24.04%, holding30.60%. Final stocks
35.40% (+2.77pp excess), ETFs18.55% (−0.92pp), median stock excess+0.41pp meet
final comparability targets. Selected ETF excess−3.36pp and median stock−7.56pp
fail; final mean drawdown worsens3.21pp versus allowed2. Unit joint short success
0/3 in both phases. Preserve the full failure: no threshold sweep, phase reselection,
extra training or promotion under this replay protocol. No new Colab units.

Eight published asset sizes/digests match; verified duplicates removed, retained
paths reproduce all four group/short metrics exactly.12 filter/state/reference
checks passed. Overall goal remains unmet. A separately preregistered training/
execution-alignment hypothesis may examine why deploying unit size reverses
selected/final ranking; it is not confirmation of this failed replay or a proven
cause. [Outcome and audit](../results/recurrent-unit-sizing-2026-10-21/README.md).

## Training/execution sizing alignment prepared

New fixed hypothesis distinguishes adaptive training/evaluation, adaptive training
with unit evaluation, and aligned unit training/evaluation. Same fresh masked21
RecurrentPPO/data/initialization/sampling/100k budget; both controls required.
Adaptive/posthoc final tensors must match. Shared explicit execution contract now
routes training action mapping, actual net-equity reward, validation selection
and frozen predictions consistently; default historical bundles stay adaptive.
43 focused checks including unit Gymnasium/SB3 checkers pass without fitting.
Dedicated job/archive/report/counter integration remains before three512 Colab
smokes. No allocation, project38.014408 retained, study allowance3 with account
150/170 and project150/130. Prior failed replay remains closed; this is a new
learning-contract intervention. [Protocol and status](recurrent-sizing-alignment.md).

Alignment version23 integration complete: actual processed-action counters,
strict training/evaluation modes, saved records, unit action/protection traces,
checkpoint identities and restored matched reporting.79 focused checks and actual
three immutable smoke payloads pass without fitting; scripts compile. Smoke
acceptance requires234 exact frozen action/state predictions plus adaptive/posthoc
final tensor equality. No allocation at integration, project38.014408 retained.
Freeze clean pushed source and official usage check before three512 CPU smokes.
[Guide](recurrent-sizing-alignment.md).

## Actual alignment Colab smoke acceptance

Three512 CPU smokes accepted on Python3.13.16, clean workercaddfcb.234 exact
frozen mapped-action/hidden/cell/repeat/reset predictions, immutable normalization
and input rejection pass. Actual counters:adaptive/posthoc512 transitions each,
340 non-unit raw sizes and0 changes; aligned343 changes,0 processed non-unit sizes,
0 protective changes. All initial tensors/sampling match; adaptive/posthoc final
policy tensors match exactly. Actual selected/final records/checkpoint identities
and unit traces pass. All3 sessions stopped; project38.047855/150, account349.78
zero rate/assignments. Fixed same-source three100k screen next within3-unit study
allowance; no tuning from smoke returns. [Evidence](../results/recurrent-alignment-2026-10-21/README.md).

## Completed alignment screen — candidate gates failed

Fresh100k seed11 selected/final means:adaptive24.99%/24.04%, posthoc unit32.80%/
32.80%, aligned unit20.04%/13.93%, holding30.60%. Adaptive/posthoc final tensors
match exactly. Candidate selected/final loses4.95/10.12pp to adaptive and12.76/
18.88pp to posthoc. Candidate final stocks14.86% versus holding32.63%, ETFs8.77%
versus19.47%, median stock excess−24.77pp. Final mean drawdown−37.41% worsens
7.42pp versus adaptive and4.21pp versus posthoc. Selected joint short outcomes
2/3 deteriorate to0/3 final. Relative/absolute/risk gates fail: no extension,
threshold/architecture sweep, confirmation or promotion of this family.

All6 owned sessions stopped;468 exact frozen mapped-action/state predictions,
actual execution counters/traces, restored models and checkpoint identities pass.
All initial tensors and actual sampling counts match; all111 symbols receive
training transitions. Unit training maps62,293 sizes, processes0 non-unit sizes,
changes0 protection components; all100,000 requests nonzero. These facts establish
execution, not a proven failure cause. Posthoc remains a one-seed development
control, not robust/prospective success; original failed replay is unchanged.

Project38.136658/150, remaining111.863342; study conservative0.12225039 (ledger
observed debits0.08; rounded account decrease0.09 includes meter delay). Account
349.71 zero rate/assignments, account150/170 and project150/130 retained. No retired
test or orders. Overall goal unmet. [Full evidence](../results/recurrent-alignment-2026-10-21/README.md).

Alignment publication/cleanup accepted:13 server sizes/digests match;48,871,879
duplicate bytes removed,456 retained hashes unchanged. Post-cleanup strict report
reproduces byte for byte. All6 sessions stopped and required source/data/model/
counter/curve evidence retained. [Audit](../results/recurrent-alignment-2026-10-21/cleanup.json).

## Alignment saved risk/action audit — 2026-10-22

All six accepted phases audited without fitting/data/Colab. Aligned final stock/ETF
cost drag11.08/8.37pp and noncost excess−6.68/−2.33pp both contribute to failure.
Mean symbol drawdowns−37.89%/−34.78% contain only2.87%/2.63% of peak equity in
window costs; this is descriptive bookkeeping, not a cost-free replay. All saved
policy exposure requests are nonzero; original Gaussian output and issued unit
action are now distinguished. Consider explicit categorical cash as a separate
preregistered hypothesis; no causal fix established, no allocation or promotion.
19 focused tests pass; accepted source hashes unchanged. Project38.136658 retained,
all owned sessions stopped. [Evidence](../results/alignment-risk-actions-2026-10-22/README.md).

## Recurrent cash-choice protocol and policy construction — 2026-10-22

Separate fixed four-arm hypothesis: explicit categorical short/cash/long versus
matched binary short/long, aligned Gaussian unit and strongest posthoc Gaussian
unit control. Candidate must pass relative gates against every control plus
selected/final stock/ETF/median holding and drawdown targets before11/23/37
confirmation and prospective evidence. Same data/reward/memory/sampling,100kseed11;
no rescue of failed alignment family. Hybrid policy/likelihood construction and
local no-fit checks prepared; dedicated plan/worker/archive/frozen integration and
actual four512 Colab smokes remain. No allocation; project38.136658 retained.
[Protocol, implementation and acceptance](recurrent-cash-choice.md).

Cash-choice factory/frozen integration: explicit hybrid algorithm and strict
action-distribution metadata now bind actual policy/cash support to frozen
loading; semantic corruption rejects despite valid bundle checksums. Native
legacy compatibility retained.37 focused no-fit checks pass, including branch,
checkpoint and sequence regressions. Dedicated Colab plans/worker/archive/report
integration and actual smokes remain; no new credits used.

Cash-choice version24 integration complete: fixed four-arm plans/worker/archive,
actual optimizer/parameter-count records and cash execution counters, mandatory
checkpoint identities, restored smoke/screen reporters and all-control gates.
96 focused checks and four actual immutable payload validations pass; scripts
compile. No fitting/allocation. Freeze clean pushed source, official usage check,
then four512 CPU smokes requiring312 exact sequence predictions and verified
cleanup before100k screen. Project38.136658 retained; account150/170 and
project150/130 remain. [Guide](recurrent-cash-choice.md).

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
record40 PPO epochs and80 actual Adam steps, have21 optimizer parameter states
and changed weights. The saved `optimizer_updates` field is SB3’s `_n_updates`
epoch counter; it is not an Adam minibatch-step count. Restoring all four final
smoke optimizers confirms every parameter state has step80.
No sizing/protection transforms occur in categorical arms. These are wiring
checks, not performance evidence or settings selection.

All four retry sessions stopped/verified (eight including initial failure).
Project38.18725015/150; study increment0.05059178 including failure, remaining
111.81274985. Account349.67 zero rate/assignments. Source, original ledger and
failure evidence retained. Fixed same-source four100k seed11 screen next within
original4-unit study allowance; no tuning from smoke returns.

[Evidence](../results/recurrent-cash-2026-10-22/README.md).

## Completed fixed screen — cash candidate failed

Strict matched source-bar/accounting/holding validation, actual selected/final
model restoration, mandatory checkpoint identities, within-family initial tensor
equality and identical actual sampling pass. All111 training symbols receive
transitions. Mean net returns on the original26 development episodes:

| Arm | Selected | Final100k |
| --- | ---: | ---: |
| Posthoc Gaussian unit | 32.80% | 32.80% |
| Aligned Gaussian unit | 20.04% | 13.93% |
| Binary categorical | 34.38% | −1.25% |
| Cash categorical candidate | 29.24% | 6.63% |
| Matched holding | 30.60% | 30.60% |

Cash final stocks7.28% versus holding32.63%, ETFs3.03% versus19.47%; median stock
excess−25.65pp. Cash selected stocks32.02%, ETFs13.97%, median stock excess−3.25pp;
selected ETF/median targets fail. Selected/final gains versus strongest posthoc
control are−3.56/−26.18pp. Cash final mean symbol drawdown−31.03% improves2.16pp
over posthoc's−33.20%, but lower drawdown alone cannot pass return gates. Cash
loses selected5.14pp to binary, while final beats binary7.87pp; final ETF gain
versus binary is−0.45pp. All three relative control gates and both absolute
selected/final gates fail.

Cash joint positive total return AND positive net short P&L improves from1/3
selected declining stocks to2/3 final: AMZN+9.49% with+$4,299.85 short P&L,
NEE+20.35% with+$10,635.29. DIS remains−31.09%, short P&L−$6,968.94. These
future-defined cohorts are reporting-only. Partial short success does not offset
weak overall stock/ETF results. Binary joint outcomes fall2/3 selected to0/3 final;
its temporary34.38% checkpoint is not a stable winner. Posthoc remains one-seed
reused-development evidence, not robust/prospective success.

The fixed cash-choice family is closed: no extra steps, cash thresholds, entropy
or architecture grids, confirmation or promotion. No retired-test evaluation,
orders or prospective success. Next inspect saved action/participation and
optimizer diagnostics before any distinct preregistered intervention.

All12 owned sessions stopped/verified, including initial failure.624 exact frozen
predictions across accepted smokes and screens pass. Models, full reports, source,
execution/sampling counters, checkpoints and compact10k curves retained.
Publication/checksummed duplicate cleanup remain.

Project38.30900603/150, remaining111.69099397; conservative study increment
0.17234766 includes failure/setup/idle/retrieval. Ledger observed debits0.14.
Account349.56 zero rate/assignments; original account150/170 and project150/130
limits retained. Original ledger and all failure evidence preserved.

[Full evidence](../results/recurrent-cash-2026-10-22/README.md).

## Publication and cleanup accepted

[Model/report release](https://github.com/jason490/RL-trading/releases/tag/recurrent-cash-2026-10-22) contains15 assets, all server sizes and
SHA-256 digests verified. Eight accepted model bundles retain selected/latest
weights, metadata, checkpoint identities and execution/sampling counters.
`comparison.html` shows interactive10k curves; `learning-curves.csv` retains
percentage-point increments. Failure evidence remains visible.

Cleanup removed only two verified duplicate publication archives,58,832,238 bytes
(about56.1MiB). All249 inventoried retained hashes are unchanged, including
original transports, complete model/frozen/report/checkpoint evidence, source
archives, failure archive, immutable dataset and original ledger. The strict
summary reproduces byte for byte after cleanup, restoring all four screen models
and rechecking source/accounting/gates. All12 owned sessions remain stopped.
Project38.30900603 and last official account349.56 remain; no further compute
allocated. Next audit saved behavior/optimizer evidence before a distinct
preregistered intervention. No cash-family extension or promotion.

[Verification](../results/recurrent-cash-2026-10-22/release-verification.json), [cleanup](../results/recurrent-cash-2026-10-22/cleanup.json), [reproduction](../results/recurrent-cash-2026-10-22/reproduction.json).

## Cash-choice saved action audit — 2026-10-22

All eight accepted phases/208 symbol episodes audited with unchanged source hashes
and no fit/model/data/provider/Colab access. Cash stock long/short/cash requests
shift98.01/0.35/1.64%→68.78/16.30/14.92%; stock short contribution+0.83→−9.21pp.
Binary stock shorts6.06%→31.04%, contribution+3.45→−15.66pp. Cash final stock/ETF
excess−25.35/−16.43pp reconciles noncost−17.83/−8.95 minus cost7.51/7.49pp;
costs alone insufficient. Observed direction drift is not a proven optimizer
cause.28 focused tests pass, including exact registered arms/source identities,
cash requests and legacy compatibility. No family rescue or new training; saved
optimizer diagnosis next. Project38.309006 retained, all sessions stopped.
[Full audit](../results/cash-actions-2026-10-22/README.md).

## Saved cash-study optimizer audit — 2026-10-22

All eight selected/final models restored read-only against retained/publication
hashes. All weights/Adam moments finite; all21 parameter states present, expected
shapes and counts. Final100k has20,000 Adam steps, distinct from1,000 nominal PPO
epochs; selected cash/binary have1,800/7,800 actual steps. Late logged KL/clipping
rise in failed categorical models and strongest posthoc control alike, so no
causal optimizer explanation established.12 helper checks pass; no fit/dataset/
provider/Colab access, sources unchanged. Fitted-period/generalization diagnosis
next before any distinct safeguard protocol. Project38.309006 retained, all
sessions stopped. [Audit](../results/cash-optimizer-2026-10-22/README.md).

Recurrent fitted-period protocol prepared: all four accepted arms × selected/final,
original26 symbols, fixed2018–2020 common window, frozen normalization/memory and
execution rules. Explicit original-panel slicing preserves legacy rejection;
full optimizer fingerprint guards read-only replay. Commit before actual inference,
no fitting/provider/Colab; descriptive flags never promote failed cash family.
[Protocol/method](training-fit-diagnostic.md). Actual replay pending.

## Actual recurrent fitted-period replay

Eight saved selected/final models complete26 episodes/19,136 rows each, with
unchanged tensors, full Adam fingerprints, counters, normalization and source
hashes; all matched holding paths equal. No fitting/provider/Colab/test access.
All eight fitted stock/ETF/median flags fail. Fitted holding58.15% stocks,47.15%
ETFs. Posthoc final stocks27.22%, ETFs68.43%; cash final19.92%/27.07%. Cash
fitted mean improves3.22→21.02% while development29.24→6.63%; binary fitted
mean7.79→15.28% while development34.38→−1.25%. This is consistent with a
generalization tradeoff, not pure deterioration on every period; fitted benchmark
weakness remains and no cause/promotion proven.32 focused checks passed before
committed protocol6f2e10c and actual replay. All sessions stopped, project38.309006
retained. [Evidence](../results/recurrent-fit-2026-10-22/README.md).

## Separate Gaussian policy-update guard protocol — 2026-10-22

Fixed fresh control target_kl=None versus candidate0.01, same strongest Gaussian
posthoc training/evaluation contract, data/reward/memory/sampling and100kseed11.
No categorical/cash rescue. Same selected/final relative/absolute/risk gates;
actual candidate Adam-step reduction required for activation. No grids/extension.
Construction records distinguish nominal epochs from actual Adam steps, and a
forward-only high-KL library test forbids optimizer.step and proves no learning.
Dedicated Colab integration and two512 smokes remain; no allocation.
[Protocol and constraints](recurrent-update-guard.md).

Update-guard version25 integration complete: exact target/floor plans and worker,
actual Adam metadata and strict archives, restored smoke/screen reporting with
performance AND activation gates.75 focused checks and two actual immutable smoke
payload validations pass; scripts compile. No allocation at this milestone.
Commit/freeze source and official usage check before two512 CPU smokes requiring
156 exact frozen predictions and verified cleanup. Project38.309006 retained.
[Guide](recurrent-update-guard.md).
