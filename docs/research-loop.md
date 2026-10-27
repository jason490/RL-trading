# Bounded autonomous research

Current frontier (2026-10-27): no robust/prospective winner. The recent matched
RecurrentPPO baseline returns19.62% selected/18.48% final, policy-update guard
28.07%/24.44%, versus holding30.60% on reused2021–2023 development data.
Treasury26.51%/−4.56%, VIX, GAE and prior failed families remain closed; retired
2024+ test data stays unopened. See the compact result directories for evidence.

CrossQ [fixed protocol](../configs/experiments/crossq_comparison_preregistration.json)
compares native small64×64 SAC/CrossQ packages on original26-symbol data. Delay,
Adam beta and1000-update renorm warmup differences are fixed, not an isolated
normalization ablation or paper-size replication. Shared untrained tensors/RNG
and portable provenance are pinned. Native geometry, phase-sensitive actual
Adam/normalization/replay, observed forwards and immutable frozen inference pass.

[Two actual4096 CPU smokes](../results/crossq-feasibility-2026-10-26/README.md)
from clean1a359c7/Python3.13.16 are accepted:256 exact predictions, actor/critic
renorm, replay/timeout restoration and both owned sessions stopped. Project
39.497818/150 at smoke cleanup, official account348.62/idle then. Raw log-alpha,
weights, Adam and normalization restore bitwise; only derived float32 exp permits
one-ULP CPU rounding. Canonical JSON preserves tuple/list parameter semantics.
Projection3732.56s fits7200; original study baseline39.47256169564856/2 persists.

The fixed50k seed11 screen from clean36e4dd4 was deliberately stopped after
[historical SAC integrity failure](../results/crossq-integrity-2026-10-27/README.md):
10k discrepancy0.121578pp>0.01pp. Initial tensors/RNG/contracts match, first1001
actions bitwise equal; first divergence after learning is5.96e−8/four ULPs.
Root cause unproven. No candidate interpretation/promotion, tolerance change or
automatic retry. SAC20k snapshot retained; no available CrossQ screen checkpoint
before interruption (salvage404, logs retained). Both owned sessions stopped.
Project39.518548/150, official account348.60/rate0/no assignments. Original
CrossQ baseline39.47256169564856/2 unchanged. Next read-only backend/implementation
diagnosis; any new protocol must preserve this failed attempt and require review.
Saved actor/critic/entropy optimizer groups now independently agree in effective
numerical values; container/default representations differ. Scoped source audit finds65 unchanged files and matching environment/seed/create/
on-step calls; both sources explicitly set intra-op threads1. Runner/registry
changes remain, and actual backend execution is not proven equal. Original backend
metadata is unavailable, so cause remains unproven. Future runner metadata now
captures actual CPU/thread/precision/determinism settings read-only; no retry
or tolerance change follows. [Guide](execution-provenance.md).
Persistent `.runtime/crossq-integrity-stop.json` blocks new study controllers and
root-validated profiles even with prior smoke acceptance; no ledger reset.

The separate [SAC first-update diagnostic](sac-first-update.md) completed on one
Colab AVX2 CPU from clean7eb2588/Python3.13.16: two actual native updates/six
Adam steps; original/explicit defaults give identical tensors/actions/batch/RNG.
Both saved models, moments and raw RNG restore exactly. Local AVX512 action
replay differs2 ULPs with identical state/input/RNG; context also differs in
interop/affinity, so the old historical cause remains unproven. No performance
conclusion or CrossQ retry follows. Sole diagnostic session stopped/claim kept.
Delayed idle metering settled measured0.01 units conservatively: project
39.528548/150, official348.59/rate0/no assignments. Original baselines intact.
Verified duplicate upload cleanup removed8.08MiB,702 retained files unchanged.
Independent [recurrent-off-policy review](recurrent-offpolicy.md) found a concrete
ring-buffer sequence risk in tagged skrl2.1.1 using source/index arithmetic.
Chronological metadata indexing now passes19 offline checks, including independent
interleaved-history/overwrite oracles. Copied payload gathering and an explicit
SB3 final-observation adapter now pass12 additional checks, including actual
TradingEnv/DummyVecEnv time-limit auto-reset. Independent five-role observation-only
GRU history helpers now pass22 native untrained-forward checks: detached burn-in,
one-pass t/t+1 alignment, target graph exclusion and fresh episode states.
Incremental sampling now matches the reference after interleaved overwrite and
all93,800 valid starts at100k capacity; local256-window draw median1.873ms
(metadata only). Independent recurrent networks and separate SAC objectives now
pass10 further analytic/forward checks (69 total), with installed SB3 train/density
source hashes pinned. Colab-only bounded update driver, checksummed learner
snapshots and private deterministic carried-state actor now pass8 additional
offline/manual-state checks (77 total), with no local backward or optimizer call.
A separate fixed training-prefix smoke now registers4 updates per recurrent/native
arm (24 Adam steps),0.1units/900s/no retry. Actual untrained matched-batch and
initializer preparation passes with source/ledger/stop unchanged. Guarded worker
implemented. Version39 dedicated budget/claim/fleet dispatch and64MiB allowlisted
physical archive restoration now pass160 focused checks, including four-step
physical target-equation chains/manual Adam fixtures. Account348.59/idle preflight.
Actual single Colab CPU smoke from cleanfc57359/Python3.13.16 passed:4 updates
per arm/24 Adam steps, observed gradients/order, all physical target equations,
matched rows and learned restoration. Local frozen parity passes; cross-host
remote actions differ and are reported separately.149.784s inclusive, estimated
0.003775191/rounded measured debit0 so far; project39.53232284029461/150,
account348.59/idle. Session stopped/verified,8.14MiB duplicate upload removed,
1,851 retained files unchanged. No performance result/promotion. Raw training-only
collector with full-prefix actor-history refresh after updates now passes13 new
untrained checks (90 focused collection/replay/learner fixtures total). Development evaluation now reuses the common simulator/baselines/short attribution,
with private per-episode frozen GRU state and explicit exposure mode.11 new
synthetic checks/44 shared regressions pass; public test-route guard remains.
Bounded fresh-run loop now integrates scheduled collection/replay updates, immutable
checkpoint selection/final pointers, compact percentage-change curves and shared
selected/final visual reports.17 new scheduling/spy/export checks pass (51 related
checks); no local optimizer or new allocation. Fresh environment smoke now registers
128 transitions/4 updates,0.1unit/900s/no retry on a fixed six-symbol cached prefix.
Actual12-episode/1,164-row view, unchanged original normalization, Gymnasium/SB3
checkers and initial tensors pass;52 related tests pass. Worker guarded, no
allocation. Version40 dedicated budget/claim/fleet and64MiB physical verifier now
pass70 focused tests, including recorded-action economic replay, checkpoint/curve/
selection identity and local frozen parity. Prior stopped attempt settled;
account348.59/idle and project39.53232284029461 unchanged. Next clean-source
environment smoke completed from clean2fe079a on actual Colab Python3.13.18 CPU:
128 transitions/4 updates/12 Adam steps, checkpoint/report/economic replay/curve/
frozen parity verified. Selected warmup0-update prefix6.8173%, final4-update2.3990%
vs9.7706% holding: integration only, no promotion.151.151s inclusive, estimate
0.003813113/measured rounded debit0; project39.53613595309791/150, account348.59
idle. Session stopped/verified;7.97MiB duplicate removed,1,908 files unchanged.
Next bounded matched111-panel development protocol with contemporaneous controls
and cost profiling; portable bundles before promotion. Exact
interruption continuation is unsupported until full replay/environment state
is saved; no new market-performance result or allocation;
no candidate protocol, dependency change or further allocation is registered.

CrossQ packaging now requires exact initializer, replay-hash and observed-renorm
sidecars, includes arm labels, excludes raw replay/credentials.22 publication
regressions and actual retained smoke evidence pass. Nothing new published yet.

Read this frontier and its compact protocol before the historical chronology below.

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

## Initial guard smoke comparison rejected

Both512 CPU runs completed/retrieved,156 exact frozen predictions pass individually
and both owned sessions stopped. Combined acceptance rejects different initial
tensor hashes despite matching seed/source/settings/version records. Local
untrained pair hashes agree but differ from both remote hashes; cause/magnitude
not proven, initial arrays unavailable in this attempt. Keep exact equality gate.
Next checksum-pin shared untrained tensor/RNG snapshots per seed, with explicit
revised protocol before repeat smokes; no learning/performance initialization
selection or tolerance relaxation. No100k screen allocation. Both had80 Adam
steps/40 nominal epochs; short-smoke nonactivation was allowed and is separate.
Project38.32607251, original study baseline38.309006 retained, no budget stop.
[Failure evidence](../results/recurrent-guard-2026-10-22/README.md).

## Update-guard shared initialization version2

Fixed untrained tensor/CPU-RNG artifacts enforce exact paired initialization
after version1 hash mismatch; original failed source/plans/charges retained. No
learning/data/performance selection, no tolerance/gate relaxation. Manifest and
per-file/schema/RNG checksums bind controller, worker, metadata, saved model and
retained run snapshot.94 focused checks and two revised actual immutable payloads
pass; source compilation passes. No allocation during fix. Commit/freeze new
source and repeat two512 CPU smokes using new state; original2-unit study baseline
38.309006 and project/account safeguards retained. Project38.326073 currently.
[Method](recurrent-update-guard.md).

## Version2 actual smokes accepted

Two512 CPU runs from clean workered9d834, Python3.13.16 accepted after durable
retrieval and verified shutdown.156 exact frozen action/hidden/cell/repeat/reset
predictions, immutable normalization/input rejection, restored selected/final
settings/Adam evidence and current checkpoint identities pass. Both load the
same pinned tensors and CPU RNG, with exact initial fingerprint equality and
matching actual sampling. Input snapshots are retained in both runs.

Both smokes record40 nominal epochs and80 actual Adam steps. The short guard
does not activate early stopping, explicitly allowed; actual reduction is still
required at the fixed100k screen. This is wiring acceptance, not performance
evidence or a threshold-selection exercise. Original version1 rejection remains.

All four owned smoke sessions stopped. Project38.34607251/150, remaining
111.65392749; study conservative increment0.03706648 includes rejected attempt.
Official account349.54, zero assignments/rate. Original study baseline38.309006
and2-unit allowance, project150/130 and account150/170 remain. Same-source two100k
seed11 screen next; no tuning from smoke returns.

[Evidence](../results/recurrent-guard-2026-10-22/README.md).


## Completed update-guard screen

Fixed seed11 control selected/final19.62/18.48%, guard28.07/24.44%, holding30.60%.
Guard reduces actual Adam steps20,000→6,165 and passes relative/risk gates, but
both absolute target gates fail. Final stocks27.78% versus32.63% holding;
ETFs6.11% versus19.47%; joint short/total success0/3. No confirmation, extension,
KL sweep or promotion. All six study sessions stopped,156 screen frozen checks
pass, original ledger retained. Project38.39201565/150; official account349.51,
zero rate/assignments. Publication/duplicate cleanup and saved-path diagnosis next.
[Full evidence](../results/recurrent-guard-2026-10-22/README.md).


Update-guard publication accepted:15 server assets match SHA-256/size, four
accepted bundles and interactive curves retained publicly. Only two verified
duplicate archives removed (32,215,676 bytes);167 retained hashes unchanged.
Strict summary reproduces byte for byte after cleanup. All six sessions stopped;
project38.39201565 unchanged. Saved-path diagnosis next, no guard promotion.
[Evidence](../results/recurrent-guard-2026-10-22/README.md).


## Saved update-guard action audit

Four accepted phases/104 episodes audited against exact accepted hashes/panel;
no fit/model/data/provider/Colab access and source hashes unchanged. Guard final
stock/ETF long requests99.51/98.91% versus control93.60/95.77%; cost drag falls
6.07→2.85pp stocks,5.13→2.53pp ETFs. Relative gains combine lower costs and
residual path improvements. Final ETF shortfall13.35pp still includes10.83pp
residual plus2.53pp costs; selected-to-final deterioration occurs despite lower
costs. No causal explanation established.15 focused tests pass. Guard stays
closed; next fixed exit/gap participation audit before another intervention.
Project38.39201565 unchanged, all sessions stopped.
[Evidence](../results/guard-actions-2026-10-22/README.md).


## Saved exit/inherited-position diagnosis

Four phases/104 episodes partitioned with reconciled fill/quantity/equity/holding
chains; no fit/data/Colab. Guard final ETF excess−13.35pp includes flat/no-exit
−5.13pp and inherited-short/no-exit−4.74pp, versus protection−0.03pp. Daily
observed partitions are not isolated gap P&L or causal effects. Reports lack
all opens; next preregister exact bar-backed overnight/intraday cash-flow
reconciliation before another intervention.26 focused tests pass. Guard closed;
project38.39201565 unchanged, all sessions stopped.
[Evidence](../results/guard-exits-2026-10-22/README.md).

Fixed bar-backed P&L bridge implemented with40 focused checks; commit method
before actual four-phase development-bar diagnostic. No training or allocation.
[Method](pnl-bridge.md).


## Actual bar-backed cash-flow bridge

Committed method96dddd9 precedes four accepted phases/104 episodes. Every
policy/holding session reconciles (worst error1.28e-10 dollars); source hashes
unchanged, no fitting/new prediction/provider/Colab/test access. Guard final
ETF excess−13.3541pp comprises overnight−7.6351, inherited intraday−1.2040,
fill-to-close−1.5044, dividends−0.4841 and fees/impact/borrow−2.5265pp.
Overnight differences are substantial, not proof of a holding-rule cause. Next
fixed partition by inherited side and preceding actual exit reason before any
intervention. Project38.39201565 unchanged, all sessions stopped.
[Evidence](../results/guard-bridge-2026-10-22/README.md).


## Actual overnight partition complete

Fixed methoda19f350, four phases/104 episodes;41 focused checks and exact
bar-backed bridge pass, sources unchanged. Guard final ETF overnight excess
−7.6351pp splits flat-after-holding−3.1580, inherited short−4.3602,
flat-after-protection−0.6304 and long+0.5135pp. Guard selected same holding
rule gives flat-after-holding−0.1277pp; no fixed performance ceiling or causal
rule-change benefit established. Both short timing and missed exposure matter.
Diagnostic sequence complete; choose a distinct preregistered learning
hypothesis before new allocation, preserving short/holding/absolute gates.
Guard closed, no promotion. Project38.39201565 unchanged, all sessions stopped.
[Evidence](../results/guard-overnight-2026-10-22/README.md).


## Distinct session-decomposition information hypothesis

Optional completed overnight/intraday log returns and EMA20 components implemented
with none/masked/full modes; defaults unchanged. Synthetic witness proves baseline
seven features omit this open/close decomposition, not predictive value.53 focused
checks pass (formulas, causality, splits, masks, training-only normalization, gaps).
No real-data preflight/training or allocation yet. Next bind expanded111/original26
datasets, exact shared controls/initialization and numeric gates before Colab
smokes. Guard family remains closed; project38.39201565 unchanged.
[Method and pending requirements](session-context.md).


## Actual session-context data preflight

Retained development-only raw source verifies; default generated identity
unchanged. New masked/full datasets preserve all bars/ATR/original features and
original first7 normalization exactly. Each has298,946 training and19,058
development rows,111 training/26 development symbols. Added-column normalization
fits training rows only; future/test data unopened.53 focused checks pass.
No training/allocation; next fixed three-arm protocol and shared untrained
initialization, then Colab integration/checkers/smokes. Project38.39201565 unchanged.
[Evidence](../results/session-context-2026-10-22/README.md).


Session-context numeric protocol and pinned untrained starts prepared: actual
legacy/masked/full dataset IDs, fixed100kseed11 screen, unchanged selected/final
relative+absolute/risk gates, conditional11/23/37 confirmation,2-unit incremental
allowance from38.39201564833781.31 focused protocol/data/shared-start checks pass,
including all fixed seed/size constructions and actual save/load without learning.
Dedicated Colab integration/checkers/smokes remain; no allocation.
[Protocol and initialization](session-context.md).


Session-context version26 Colab integration implemented;76 focused checks pass.
Actual three immutable smoke tasks validate; real Gymnasium/SB3 checkers and
untrained recurrent forwards pass at21/25/25 fields. Shared model evidence and
study-specific artifact checks are separated by ownership. Source compilation
passes. Actual remote smokes remain before screen; no allocation/usage yet.
[Details](session-context.md), [current results index](../results/README.md).


## Actual session-context smokes accepted

Three512 Colab CPU runs from cleanb1cfa00/defaultPython3.13.16 completed and
retrieved; all owned sessions verified stopped.234 exact frozen recurrent
predictions, normalization/rejection checks, actual selected/final optimizer
restoration and checkpoint identities pass. Each has80 Adam steps/40 nominal
epochs. Masked/full initial tensors/RNG/settings match exactly; actual sampling
agrees across all arms. One CLI reply timeout recovered from completed export,
without training restart. Fixed same-source three100kseed11 screen next; no
tuning from smoke returns. Project38.42515390/150, study increment0.03313825;
official account349.48 zero rate/assignments. Original2-unit study baseline
38.39201564833781 and all project/account protections retained.
[Evidence](../results/session-context-2026-10-22/README.md).


## Completed session-context screen — gates failed

Legacy selected/final19.62/18.48%, masked27.66/10.76%, full26.42/16.08%,
holding30.60%. Full fails both relative control gates and both absolute phases;
final stock/ETF16.56/13.45% versus holding32.63/19.47%. Joint short/total
reporting-only cohort2/3 selected→1/3 final. No feature/window/step/reward sweep,
confirmation or promotion; fixed family closed.

Strict immutable source/accounting/holding and actual optimizer restoration pass;
full/masked exact initialization and sampling match. Final20,000 Adam steps,
61,513 legacy/63,561 augmented trainable parameters.234 screen frozen predictions,
all six owned sessions stopped; legacy tensors/sampling reproduce prior control.
Project38.51487741/150, study increment0.12286176; official account349.41 zero
rate/assignments. Source/ledger/data retained. Publication/verified duplicate
cleanup next. [Full evidence](../results/session-context-2026-10-22/README.md).


Session-context publication complete:17 server assets verified, six accepted
model bundles and interactive reports public. Only two duplicate publication
archives removed (49,579,965 bytes);178 retained hashes unchanged. Strict summary
reproduces byte for byte, all three screen models/accounting/gates rechecked.
All six sessions stopped; project38.51487741 unchanged. Fixed family closed;
next distinct preregistered learning hypothesis, no extension/feature grid.
[Evidence](../results/session-context-2026-10-22/README.md).


## Distinct fixed recurrent-capacity hypothesis

Prior recent recurrent protocols keep64-unit widths; no direct64/128 ablation
found. Weak fitted-period benchmarks motivate a capacity hypothesis but do not
prove underfitting. Freeze control64 versus candidate128 recurrent/dense widths,
61,513/221,321 parameters, unchanged baseline seven features/data/reward/actions/
sampling and100kseed11 budget. No session-feature rescue or width/depth grid.
Original selected/final relative/absolute/risk and conditional11/23/37/prospective
gates retained. Fixed2-unit allowance from38.514877408036206.

Pinned untrained per-arm starts/common CPU RNG prepared; both smokes use full
architectures.65 focused construction/regression checks pass, including actual
save/load and invalid/runtime-deadline rejection. Conservative smoke-duration
projection must pass before long CPU allocation. Dedicated integration/checkers/
Colab smokes remain; no allocation or new usage.
[Protocol and constraints](recurrent-capacity.md).


Capacity version27 integration prepared:96 focused tests pass, actual full64/128
environment checkers/untrained forwards and two immutable smoke tasks validate.
Required runtime profile is bound to actual accepted smoke bytes and recomputed
before controller allocation; missing/stale/over-deadline profiles reject.
Worker/runner/retrieval/model/optimizer evidence and confirmation/report routing
are integrated using existing lifecycle/budget code. Source compilation passes.
Actual two512 CPU smokes still required before100k; no allocation/usage yet.
[Evidence](../results/recurrent-capacity-2026-10-22/README.md).


## Actual capacity smokes accepted and runtime projection passes

Two512 full64/128 CPU runs from39f94a0/defaultPython3.13.16 completed/retrieved
and verified stopped.156 frozen recurrent predictions, immutable normalization
and rejection checks, restored models/optimizers/checkpoints, same sampling and
common CPU RNG pass. Full61,513/221,321-parameter architectures each perform
80 Adam steps/40 nominal epochs. Durations140.96/136.94s are mixed train/evaluate,
not a reliable speed ranking. Fixed projection3016s/50.3min passes7200s; exact
acceptance bytes bind approved plans. Same-source two100kseed11 screen next.
Project38.53588481, study increment0.02100741; official account349.40 zero rate/
assignments. Original baseline38.514877408036206/2-unit allowance retained.
[Evidence](../results/recurrent-capacity-2026-10-22/README.md).


## Completed capacity screen — 2026-10-23

Both fixed seed11 runs completed100,000 transitions in Colab CPU, defaultPython
3.13.16, from frozen clean source39f94a001044d9a0f24e4f2b1384014495f835e1.

| Arm | Parameters | Selected return | Final return |
| --- | ---: | ---: | ---: |
| Control64 |61,513|19.6245%|18.4844%|
| Larger128 |221,321|28.1508%|13.4278%|
| Matched holding |—|30.6019%|30.6019%|

Separate26-symbol mean net total returns on repeatedly reused2021–2023
 development history; neither annualized nor a combined portfolio or independent
evidence. Larger selected at60k versus control80k. Selected improvement8.5263pp,
final change−5.0565pp; final stock gain−8.9396pp and ETF gain+16.3002pp versus
control. Mean drawdown worsens0.5327pp. Larger selected stock excess−3.2133pp
misses−2pp; final stock/ETF excess−19.7987/−2.7383pp and median stock
excess−13.4832pp fail. Fixed capacity family closes: no width/depth/step grid,
extension, confirmation or promotion. No robust accepted winner established.

Strict source/dataset/accounting/holding checks pass. Both selected/final actual
weights, settings and Adam states restore; each final has20,000 actual Adam
steps/1000 nominal epochs. Same actual111-symbol training sampling and common
initial CPU RNG verified, with different pinned per-width tensors as declared.
156 screen frozen action/hidden/cell/repeat/reset predictions pass;312 including
smokes. Normalization immutable, schema/feed/nonfinite rejection passes. Retired
2024+ data unopened; no fitting by local reports or broker orders.

All four owned smoke/screen sessions stopped and cleanup verified. Official
account349.34, zero rate/assignments. Cumulative conservative project38.60166524,
remaining111.39833476; study increment0.08678784 includes setup/idle/downloads/
shutdown. Observed account debit0.07 versus conservative charge0.08678784.
Project150/130 and account150/170 safeguards unchanged; original ledger retained.


[Evidence](../results/recurrent-capacity-2026-10-22/README.md).


Capacity publication verified:15 GitHub assets match server sizes/SHA-256,
four accepted model bundles and interactive reports retained. Only two duplicate
publication archives removed (45,880,597 bytes),142 retained hashes unchanged.
Original transports/source/data/models/checkpoints/ledger/plan.html preserved.
All four sessions stopped; project38.60166524 unchanged. Fixed family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/recurrent-capacity-2026-10-22).


## Capacity saved-path diagnosis — 2026-10-23

Four accepted phases/104 episodes audited without data/model fitting/Colab.
Larger stock selected→final29.41→12.83%, short contribution−0.89→−7.41pp,
cost drag3.78→5.14pp and noncost excess+0.57→−14.66pp. ETF21.21→16.73%,
cost3.01→5.13pp; neither causal explanation nor a short-ban justification.
21 focused audit/action checks pass; source hashes unchanged. Capacity family
closed. Next audit training-history coverage/sampling before preregistering a
distinct matched full-history/recent-window study; no allocation or window
selection from returns. Project38.60166524 unchanged.
[Evidence](../results/capacity-actions-2026-10-23/README.md).

## Date-only history audit — 2026-10-23

All111 symbols support512-transition windows in full history and fixed2016-onward
subset. Uniform starts underweight edge years: expected2020 execution share
about2.4% full versus7.3% subset. Existing logs cannot establish actual year
visitation.11 exact enumeration/boundary/isolation checks pass, no fitting/provider/
Colab usage. Next add measured decision/execution year counters, then preregister
one immutable composition comparison with unchanged evaluation/model/reward/
normalization/budget. No date choice from profits or capacity-family reopening.
[Evidence](../results/training-history-2026-10-23/README.md).

Measured sampling-year instrumentation is implemented: episode-start, decision,
next-bar execution, cross-year pair and per-symbol counters. Existing runner
persists them at checkpoints/final metadata. Global/pair/symbol margins reconcile
and reject corrupt year evidence; legacy artifacts remain readable but cannot
claim measured year visitation. No model fitting or Colab allocation. Next build
an immutable recent subset and freeze the numeric matched-history protocol before
smoke acceptance/training. [Contract](training-history.md).

Immutable history suffix preflight passes:111 symbols,137,529 recent training
rows and19,058 exact common development rows; retained features/ATR/bars and
full-training normalization unchanged. Fixed numeric `training_history.json`
pins full/recent64-unit100k seed11,2pp selected+final improvement, stock/ETF/
median−2pp targets and shared2-unit allowance from38.60166524433091. Failure closes
that single window comparison, no year grid.23 subset/coverage checks pass.
Dedicated Colab contracts/checkers/smokes still required, no allocation.
[Details](training-history.md).

Training-history version28 integration and actual construction acceptance pass:
111 focused regressions, both actual datasets/checkers/untrained forwards and
immutable smoke task preflight. Distinct recency runtime names preserve older
historical study. Missing year logs reject new artifacts; exact actual smoke
acceptance bytes and runtime projection required before100k. Commit/freeze and
official usage/Python check before two512 CPU smokes; no allocation/new units.
[Evidence](../results/training-history-2026-10-23/README.md).

Training-history actual smokes accepted: both512 CPU/defaultPython3.13.16 runs,
156 exact frozen predictions, equal initial tensors/RNG/symbol counts, measured
year reconciliation and restored models/optimizers/checkpoints pass. Both stopped.
Projection3504s/58.4min passes7200s; acceptance bytes bind prepared100k plans.
Same-source full/recent seed11 screen next. Project38.62221390; original baseline
38.60166524433091/shared2-unit allowance retained, official account349.32 zero
rate/assignments. [Evidence](../results/training-history-2026-10-23/README.md).


## Completed history-composition screen — 2026-10-23

Status: fixed seed11 comparison failed; family closed. No confirmation,
extension, start-year/window grid, promotion or performance goal achieved.
Both fresh100k runs completed in Google Colab CPU/defaultPython3.13.16 from
frozen clean2dfa1fbd45d6a3c120fb11311c9d285781556b57. Earlier sections retain
preparation/smoke history; this is the latest outcome.

| Arm | Selected return | Selected step | Final return |
| --- | ---: | ---: | ---: |
| Full2010–2020 |19.6245%|80,000|18.4844%|
| Recent2016–2020 |20.9523%|50,000|5.5976%|
| Matched holding |30.6019%|—|30.6019%|

Mean net total returns across separate26-symbol episodes on repeatedly reused
2021–2023 development history; neither annualized nor a combined portfolio or
independent test. Candidate selected improvement1.3278pp misses the2pp gate;
final change−12.8868pp fails. Candidate final stock/ETF4.4602/11.8533% versus
holding32.6265/19.4660%. Its final stock change−17.3073pp versus control fails
the nonworsening group gate; ETF improves11.4258pp. Mean drawdown worsening
0.0127pp is within risk tolerance but cannot compensate for failed return gates.
Both phases fail stock/ETF/median−2pp absolute targets. Selected stock/ETF excess
−10.3746/−5.6617pp, median−8.5500pp; final−28.1664/−7.6127pp, median−15.6457pp.
Reporting-only declining-stock joint positive total AND net short success1/3
selected→0/3 final; no robust profitable-short capability established.

## Actual training years and verification

Measured2020 execution share rises as intended: full stocks1.978%/ETFs3.005%,
recent stocks6.637%/ETFs8.263%. These realized finite samples differ from exact
complete-window expectations. Per-symbol/global/decision/execution/pair margins
reconcile; actual symbol draw counts match. Recent history changed composition
without providing stable development improvement. It does not prove all recency
methods fail, or establish a sole cause of deterioration. This fixed family is
closed; a future study needs a distinct preregistered hypothesis, not a year grid.

Full control reproduces the prior capacity control's exact selected/final tensor
hashes and symbol sampling, supporting instrumentation invariance. It was trained
fresh, not resumed or selected from old weights. Both arms restore identical
pinned untrained tensors/RNG. Strict source/data/accounting/holding checks and
actual selected/final models/Adam records restore. Final20,000 actual Adam steps,
1000 nominal epochs,61,513 parameters.156 screen frozen recurrent action/state/
repeat/reset checks pass;312 including smokes. Normalization remains immutable;
schema/feed/nonfinite rejection passes. Retired2024+ unopened; no local fitting,
new provider call or orders in reporting.

## Compute, artifacts and next work

All four smoke/screen sessions verified stopped. Official account349.27, zero
rate/assignments. Original project ledger38.68329334/150, remaining111.31670666;
study conservative increment0.08162810 versus observed account debit0.07,
including setup/idle/transfers/shutdown. Original baseline38.60166524433091/shared
2-unit allowance and project150/130/account150/170 safeguards retained.

Strict summary, compact10k learning curves with incremental changes, measured
year CSV, restored models, checkpoints, complete reports/source/datasets/ledger
retained. Interactive comparison is `artifacts/training-history/summary.html`.
Verified GitHub publication and duplicate-only cleanup follow this outcome commit.
Next diagnose accepted saved paths before choosing a distinct experiment; no
automatic training restart or retired-test evaluation.

[Evidence](../results/training-history-2026-10-23/README.md).


Training-history publication verified:17 GitHub asset sizes/digests match, four
frozen bundles and interactive reports public. Only two duplicate publication
archives removed (35,931,249 bytes),148 retained hashes unchanged. Original
transports/models/source/both datasets/ledger/plan.html retained. All four sessions
stopped, project38.68329334 unchanged. Fixed family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/training-history-2026-10-23).


## Saved history-study actions — 2026-10-23

Four accepted phases/104 episodes audited without dataset/model fitting/provider/
Colab access. Recent stock return22.25→4.46%, short contribution−3.58→−10.54pp,
closing short exposure11.82→19.56%; cost drag declines7.89→7.26pp. Observed
accounting, not a causal explanation or short-ban justification.22 focused
audit/action tests pass; source hashes unchanged. Fixed history family closed.
Next preregister read-only saved-policy replay on a common eligible2018–2020
training window before new training; not yet executed. No prospective success
or assumption that every bar was sampled. Project38.68329334 unchanged.
[Evidence](../results/history-actions-2026-10-23/README.md).

History-model read-only fitted-period protocol prepared: identical common
2018–2020 bars/features/normalizer across full/recent saved selected/final models,
original26 panel, unchanged simulator/unit exposure.39 focused checks pass,
including exact paired-view and commit-before-replay rejection. Commit before
actual replay. No fitting/provider/Colab usage or checkpoint selection; eligibility
is not proof every bar sampled. Closed history family stays closed.
[Method](../results/history-fit-2026-10-23/README.md).


## Actual saved-policy replay completed

All four original26-symbol replays complete:19,136 execution rows per model,
76,544 total. Both immutable datasets produce identical common eligible
2018-01-02–2020-12-02 execution bars/features/ATR and original normalization,
anchored2017-12-29. All four descriptive training stock/ETF/median−2pp flags fail.

| Saved model | Training-window stocks | Training-window ETFs | Development mean |
| --- | ---: | ---: | ---: |
| Full selected |40.53%|50.22%|19.62%|
| Full final |33.32%|36.13%|18.48%|
| Recent selected |32.36%|28.54%|20.95%|
| Recent final |46.66%|31.18%|5.60%|
| Matched training-window holding |58.15%|47.15%|—|

Separate-symbol mean net total returns, not annualized or a combined portfolio.
Training and development have different dates/regimes/durations; raw cross-period
returns cannot identify a causal generalization effect. Recent training-window
mean improves31.77→44.28% while development20.95→5.60% declines, consistent with
a generalization tradeoff but not proof of its sole cause. Full training mean
42.02→33.76% and development19.62→18.48% both decline. Final recent median stock
excess−4.88pp still misses−2pp; all fitted stock means underperform holding.

This does not establish an accepted winner, justify more width/steps, reopen
history/capacity families or select another checkpoint. Actual visitation year
counts cannot prove every replay row was sampled. All phases are preexisting
saved weights and flags are reporting-only. No fitting, optimizer step,
normalization change, provider request, Colab allocation or retired-test access.
Complete weights/Adam moments and source hashes remain unchanged. Original
project ledger38.68329334/150 and last official account349.27 zero usage remain;
all prior owned sessions stopped.


[Evidence](../results/history-fit-2026-10-23/README.md).


History-model fitted-period charts published:7 server assets verified,104 equity
plots on four pages, no model duplication or raw bars. Duplicate chart archive
removed (3,373,270 bytes),19 retained hashes unchanged; full paths retained. No
additional Colab units; project38.68329334 unchanged.
[Release](https://github.com/jason490/RL-trading/releases/tag/history-fit-2026-10-23).


Saved fitted/development action comparison:8 accepted paths/208 episodes, source/
model/report hashes verified with no fitting/dataset/provider/Colab access. Final
recent stock closing shorts19.36% training versus19.56% development, but short
contribution−4.07 versus−10.54pp. Training cost drag8.95pp exceeds development
7.26pp despite better return; aggregate frequency/cost does not establish cause.
Explicit fixed training-window reporting scope added; default development-only
checks preserved.29 focused regressions pass. Next saved optimizer/parameter
audit before a distinct learning hypothesis; closed families remain closed.
[Evidence](../results/history-fit-actions-2026-10-23/README.md).


History saved optimizer/parameter audit: four accepted models finite/exact; final
20,000 actual Adam updates. Recent selected/final norms29.00→29.89, displacement
5.88 over10,200 updates; full29.28→29.71, displacement3.74 over4,200. Intervals
differ, and KL growth in both arms does not prove cause or a useful remedy.
37 focused checks pass, no fitting/dataset/provider/Colab. Direct recurrent PPO
L2 study not found in configs; any fixed experiment must be distinct, preregistered
and gated, with no coefficient or closed-family grid. Project38.68329334 unchanged.
[Evidence](../results/history-optimizer-2026-10-23/README.md).

## Distinct fixed recurrent PPO L2 hypothesis prepared

Current config search finds no direct recurrent PPO weight decay comparison.
Prepare control0 versus coupled Adam0.0001 on original full-history111/26 source,
same64-unit model/actions/reward/100k. Explicit eps1e-5 retained; actual untrained
models confirm coupled semantics and identical pinned tensors with no optimizer
steps or data access. This is unproven; KL/norm observations do not establish
cause, and failed SAC/MLP decay does not prove PPO outcome. Fixed2pp selected+final
relative/−2pp group/median/risk gates and shared2 units from38.68329334472413.
Dedicated integration/checkers/Colab smokes remain, no allocation or performance.
Closed history/capacity/guard families remain closed; no coefficient grid.
[Protocol and next work](recurrent-regularization.md).

Recurrent L2 version29 wiring/actual construction prepared:138 focused regressions,
actual data/checkers/untrained models and immutable smoke tasks validate. Strict
coefficient/epsilon/coupling plus required years/checkpoint/initialization restored;
legacy formats unchanged. Commit/freeze and usage/Python check before two512 CPU
smokes, actual acceptance/runtime bound before100k. No allocation/new units.
[Evidence](../results/recurrent-regularization-2026-10-23/README.md).

Recurrent L2 actual two512 CPU smokes accepted:156 exact frozen checks, restored
coefficient/epsilon/coupled Adam/model/moments, same initial tensors/RNG and
sampling/year/checkpoint proof. Both stopped. Actual-bound runtime projection
3426.6427s/57.1min passes7200s; same-source two100k seed11 screen next. Project
38.70365471, original study baseline38.68329334472413/shared2-unit allowance
retained; official account349.26 zero rate/assignments. No smoke-return tuning.
[Evidence](../results/recurrent-regularization-2026-10-23/README.md).


## Completed coupled-L2 screen — 2026-10-23

Status: fixed recurrent PPO L2 comparison failed; family closed. No coefficient,
optimizer/model/reward/step grid, extension, confirmation or promotion. Earlier
sections retain preparation/smoke history. Both100k seed11 CPU runs completed in
Google Colab/defaultPython3.13.16 from clean frozen
1d640eebf0ad604461d0d510f584477efee75844.

| Arm | Selected return | Selected step | Final return |
| --- | ---: | ---: | ---: |
| Control decay0 |19.6245%|80,000|18.4844%|
| Coupled L2 decay0.0001 |30.4487%|50,000|18.4155%|
| Matched holding |30.6019%|—|30.6019%|

Separate26-symbol mean net total returns on repeatedly reused2021–2023
development history; neither annualized, combined portfolio nor independent test.
Selected relative gain+10.8241pp passes2pp, final−0.0689pp fails. Final stock
change−2.4975pp fails nonworsening; ETF+13.2886pp improves. Mean symbol drawdown
improves1.3741pp but cannot compensate for failed return gates.

Candidate selected stocks33.9981% versus holding32.6265%, but ETFs10.9265%
versus19.4660%; selected median stock excess−3.1679pp. Thus the near-matched
selected overall mean does not pass separate stock/ETF/median−2pp gates. Final
stocks19.2699%/ETFs13.7160%, excess−13.3566/−5.7500pp and median−6.7186pp,
all fail. Reporting-only declining-stock joint positive total AND net short P&L
1/3 selected→0/3 final. No robust profitable-short capability established.

## Actual optimizer/data/frozen evidence

Both actual selected/final models, moments, checkpoint IDs and default accounting/
holding paths restore and reconcile. Actual coupled Adam defaults and group
flagsFalse, coefficient0/0.0001 and explicit eps1e-5 verified. Final20,000 actual
Adam updates/1000 nominal epochs,61,513 parameters each. Exact same pinned
untrained tensors/CPU RNG, actual symbol sampling and measured year margins pass.
Fresh control reproduces prior full-history control's exact selected/final
tensor hashes and sampling, supporting preserved epsilon/default behavior; model
ZIP bytes need not match across saved provenance. No trained weights reused.

156 exact screen frozen action/hidden/cell/repeat/reset predictions,312 including
smokes; immutable normalization/schema/feed/nonfinite rejection pass. Source/data/
reward/action/model architecture/steps remain fixed. No retired2024+ access,
local fitting, new provider calls or broker orders in report generation.

## Compute, retained artifacts and next work

All four smoke/screen sessions verified stopped; official account349.23, zero
rate/assignments. Original project ledger38.74617914/150, remaining111.25382086;
study conservative increment0.06288579 versus observed rounded account debit0.04,
including setup/idle/transfers/shutdown. Original baseline38.68329334472413/shared
2-unit allowance and project150/130/account150/170 safeguards retained.

Strict summary, compact10k curves with incremental changes, actual optimizer/
initialization/year/checkpoint records, full models/reports/source/dataset/ledger
retained. Interactive comparison: `artifacts/recurrent-l2/summary.html`. Verified
GitHub publication and duplicate-only cleanup follow this outcome commit, then
saved-path diagnosis before any distinct hypothesis. No automatic restart or
relaxed gates. The full research objective remains unachieved.

[Evidence](../results/recurrent-regularization-2026-10-23/README.md).


Recurrent L2 publication complete:16 server assets verified, four model bundles
and interactive reports public. Only two duplicate archives removed (35,744,305
bytes),145 retained hashes unchanged; original source/data/models/checkpoints/
ledger/plan.html preserved. Strict summary reproduces byte for byte from retained
transports with actual models/Adam/coupling restored. All four sessions stopped;
project38.74617914 unchanged. Fixed L2 family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/recurrent-regularization-2026-10-23).


L2 saved-action diagnosis:4 phases/104 episodes, accepted hashes/panel unchanged,
no model fitting/dataset/provider/Colab access. Candidate stock return34.00→19.27%,
closing longs91.36→91.31%, cost drag4.33→3.74pp; aggregate participation/cost
does not explain deterioration.30 focused regressions pass. Next preregister a
read-only2×2 selected/final direction/protection crossover with independent
recurrent states, native-control reproduction and unchanged execution. Not yet
implemented/executed, no hybrid selection/promotion or failed-family rescue.
Project38.74617914 unchanged. [Evidence](../results/l2-actions-2026-10-23/README.md).

Fixed L2 component crossover prepared: all four selected/final direction/protection
combinations, same original26 development episodes, source-specific recurrent
memory on shared account observations and unchanged execution. Native diagonals
must reproduce full accepted paths before mixed cases; predeclared exact counts/
labels and numeric tolerance.40 focused checks pass. Commit before actual replay;
no fitting/Colab or hybrid selection/promotion. Closed L2 family unchanged.
[Method](action-crossover.md).


L2 crossover native control rejected fixed tolerance before any mixed case.
Bounded diagnostic retry retained exact failure; direct unwrapped replay equals
wrapper result exactly. Both differ from original Colab by at most$0.00467059
daily equity, with no structural/count/quantity/date mismatch. Runtime rounding
is plausible, not established cause. No tolerance widening, mixed-policy returns,
hybrid selection/promotion or family rescue.41 tests pass; source/model/optimizer/
normalizer unchanged, no training/provider/Colab/test access or new credits.
[Evidence](../results/l2-crossover-2026-10-23/README.md).

## Outcome-free frozen feature-scale audit

Expanded training stocks reach50.55σ return,51.13σ ATR ratio,80.73σ relative
volume under the original frozen normalizer. Fixed per-feature>5σ fractions
roughly0.57–1.20% except RSI0; distributions are available-row, not sampled mass.
Nine source arithmetic examples match; independent provider truth not established.
Seven focused checks pass, source/normalizer unchanged, no model/provider/fitting/
Colab/test access. Consider one preregistered explicit input cap, preserving raw
prices/execution/rewards and legacy defaults/bundles; no clipping implemented or
performance inference. Project38.74617914 unchanged.
[Evidence](../results/feature-scale-2026-10-23/README.md).

Cap5 preprocessing and immutable payload prepared after fixed preregistration.
Original raw full-history111/26 episodes and means/scales unchanged; candidate
c895d6af… uses explicit clipped-normalizer field/new schema, no candidate refit.
Legacy default generator identityc2e6c58c… reproduced exactly; old default class/
serialization unchanged.83 focused contracts/core/frozen checks pass, including
actual untrained recurrent bundle parity and fixed-action accounting equivalence.
Final protocolad11f4be… pins deterministic payload without changing cap/gates/
resource limits. Dedicated Colab integration/construction/smokes remain, no
allocation or performance claim. [Evidence](../results/input-clipping-2026-10-23/README.md).


### 2026-10-23 — fixed market-input clipping screen failed

Two100k Colab CPU runs completed; control selected/final19.6245/18.4844%,
clip5 selected/final24.7028/18.5720%, matched holding30.6019%. Selectedrelative
+5.0783pp but final+0.0877pp fails2pp; stock nonworsening and all absolute
stock/ETF/median gates fail. Family closed, no cap grid/extension/confirmation.
312 frozen checks including smokes pass; actual models/Adam/initialization/year
sampling restore; fresh control exactly reproduces prior tensors/sampling.
Control CLI disconnect recovered via advancing checkpoints/final export without
restarting training. All four owned sessions retrieved/stopped. Project38.81646887,
account349.17 zero rate/assignments. Full artifacts retained; publication and
saved-path diagnosis next, robust-return objective unachieved.

[Detailed evidence](../results/input-clipping-2026-10-23/README.md).


Clipping saved-action audit: four accepted phases/104 episodes,31 focused tests.
Selected→final closing short exposure stock3.47→6.87%/ETF1.15→10.59%; net short
contribution−0.99→−4.04pp/+1.73→−5.98pp, incremental cost drag5.11→5.92pp/
4.90→6.50pp. Observational relationships, not causal attribution or justification
for disabling shorts/reward edits. Exact source hashes preserved; no fitting/data
access/Colab allocation. Family remains closed; distinct hypothesis review before
new training. [Evidence](../results/clipping-actions-2026-10-23/README.md).

### Next diagnostic: native Gaussian action-boundary mass

Preregistered a read-only conditional-distribution diagnostic on32 fixed training
prefix observations for all111 symbols, comparing pinned untrained and existing
control selected/final models. Flat fixed actions, original normalization, no
policy-action execution or validation return evaluation. Exact Gaussian boundary
mass may identify an exploration/execution mechanism but cannot establish poor
performance causality or a better policy. Separate fixed protocol/compatibility
checks required before any support-aware training. [Method](action-boundaries.md).


### Completed action-boundary diagnostic

Three fixed Gaussian policies ×111 training prefixes ×32 observations audited
without fitting/policy-action execution/validation-return scoring. Conditional
any-component mass beyond[-1,1]:untrained78.28%, selected79.65%, final79.38%.
Final marginals exposure36.23%, stop32.44%, near37.67%, width24.06%. Full model/
Adam/RNG/normalizer/source invariants pass;54 focused tests. These are conditional
probabilities, not historical rollout frequencies or a causal return diagnosis.
Next separately preregister a same-network squashed-diagonal-Gaussian comparison;
do not conflate with failed market-input clipping or add gSDE simultaneously.
No allocation or extra units. [Evidence](../results/action-boundaries-2026-10-23/README.md).

Fixed support-aware recurrent policy protocol prepared in
[Squashed actions](../configs/experiments/squashed_actions.json): one tanh Gaussian
candidate versus matched native control, same full64 tensors/model/data/reward/
optimizer/100k, no gSDE or input preprocessing change. Existing2pp relative,
stock/ETF/median/risk gates and conditional11/23/37/prospective requirements
retained. Shared2-unit baseline38.81646887344713; no implementation/allocation yet.


### 2026-10-23 — fixed squashed-action screen failed

Two100k Colab CPU runs completed; native control selected/final19.6245/18.4844%,
squashed15.4740/5.1381%, matched holding30.6019%. Relative and absolute stock/ETF/
median gates fail; final stock6.4716%/ETF−2.1964%. Fixed family closed with no
transform/temperature/log_std grid/extension/confirmation.312 frozen checks
including smokes, actual model/Adam/distribution/initialization/year proofs pass;
fresh control exactly reproduces prior tensors/sampling. All four owned sessions
retrieved/stopped. Project38.90354117/account349.10 zero assignments. Large
conditional boundary mass did not demonstrate a causal cure or policy improvement.
Full artifacts retained; publication and saved-path diagnosis next. Robust-return
objective unachieved. [Evidence](../results/squashed-actions-2026-10-23/README.md).


Squashed saved-path diagnosis: four accepted phases/104 episodes,32 focused tests.
Selected→final closing short exposure stock4.59→10.14%/ETF4.34→9.26%; net short
contribution−5.38→−11.18pp/−1.24→−5.41pp; long contributions also deteriorate.
Incremental cost drag3.88→5.05pp/3.54→4.56pp is not the whole gap. Observational
relationships, not a cause or justification for disabling shorts/reward edits.
Exact source hashes preserved; no fitting/data access/Colab allocation. Fixed
family closed; distinct hypothesis review before new training.
[Evidence](../results/squashed-actions-diagnosis-2026-10-23/README.md).

### Next distinct information-family preflight

Preregistered a no-fit/no-performance coverage audit for compact SPY20-session
return/volatility and own-minus-SPY20-session return, using only cached2010–2023
bars and original111/26 rows. Require exact matched sessions/availability/known-gap
and raw-source identity before any new feature payload or training. Distinct from
old high-dimensional sector proxies; benefit unproven. [Method](market-context.md).

Compact market-context coverage passed: all298,946 training/19,058 development
rows available with exact raw/source identity and synchronized causal windows.
60 focused checks pass; original preprocessing/data unchanged, no fitting,
inference or extra Colab units. Next separately fix full/masked/legacy controls,
extra-field training-only normalization and immutable payload identities before
model/construction/Colab acceptance. Coverage is not predictive evidence.
[Audit](../results/market-context-2026-10-23/README.md).

Compact-market-context numeric preregistration now fixes legacy/masked/full arms,
100k seed11/10k cadence, exact raw/source/first-seven normalization, training-only
full extra statistics and exact masked-zero inference. Matched24-input initial
tensors plus original21-input control/common-column proofs separate information
from input dimension. Existing all-control/absolute/risk gates unchanged, no grids
or extension on failure. Three512 Colab CPU smokes precede100k;2-unit allowance
starts38.90354116934003. [Protocol](../configs/experiments/market_context_preregistration.json).

Market-context payload stage passes: immutable maskeda392ba93…/fullf439b8f4…,
original318,004 rows retained (298,946 train/19,058 development), first-seven stats
exact, full three-field fit training-only and masked zero/identity.48 focused
checks plus156 actual legacy frozen predictions pass; no policy fitting or new
units. Next shared untrained initialization assets and final identity-bound
protocol before dedicated Colab integration/checkers/smokes. [Evidence](../results/market-context-2026-10-23/payload-preflight.json).

Market-context shared initialization assets and final protocolf9045fe8… frozen:
all three seeds, full/masked identical mapped24-input tensors, original common
columns/parameters and per-seed RNG exact, native new columns preserved.54 checks
pass; no observed data/fitting in generation, no extra Colab units. Fixed gates
unchanged. Next dedicated training/worker/artifact/report wiring and actual source/
model construction/checkers before three512 CPU smokes. [Evidence](../results/market-context-2026-10-23/initialization.json).


### Compact market-context screen failed — 2026-10-24 UTC

Three100k Colab CPU runs complete: legacy selected/final19.6245/18.4844%, masked
18.7314/18.7314%, full7.6978/−3.7731%, holding30.6019%. Full fails both-control
relative/stock/ETF/risk and all absolute/median gates. Final stock−3.4809%/ETF−5.3803%;
drawdown worsens4.7250/4.9084pp versus controls. Family closed, no benchmark/window/
subset/model/reward/step grid or extension/confirmation.468 frozen checks including
smokes, actual models/Adam/mapped tensors/RNG/year sampling restore; legacy control
reproduces prior native tensors/sampling. All six sessions retrieved/stopped.
Project39.01437789/account349.02 zero assignments. Coverage did not imply alpha;
full artifacts retained, publication/diagnosis next, objective unachieved.
[Evidence](../results/market-context-2026-10-23/README.md).


Market-context saved-action audit: six accepted phases/156 episodes,33 focused
checks. Full selected→final short exposure decreases stock15.12→8.55%/ETF4.65→1.53%,
but net short contribution remains negative/slightly worsens; net long contribution
falls19.74→8.53pp/5.69→−1.64pp. Cost drag4.96→5.96pp/3.20→4.63pp is not the whole
gap. Observational accounting, not a sole cause or reason to ban shorts/change
rewards. Source hashes unchanged; no fitting/data/provider access or new units.
Family closed; distinct hypothesis review before new allocation.
[Evidence](../results/market-actions-2026-10-24/README.md).

### Financial filing source audit registered — 2026-10-24 UTC

Next distinct information-family review: outcome-free SEC filing coverage audit,
2009q2–2023q4, no financial inputs or training yet. Primary table documentation
identifies a future-information trap: `prevrpt` describes subsequent amendment;
exclude it from historical inputs/eligibility. Acceptance versus filing date,
public dissemination, historical CIK mapping, NUM durations/units and archive
reprocessing need verification. No configured SEC identity, no bulk downloads or
new allocation/units. Freeze feature/quantitative model protocol only after
source feasibility; retired test remains isolated. [Guide](filing-data.md) and
[fixed scope](../configs/experiments/filing_coverage_audit.json).

SEC offline structural reader implemented:18 focused synthetic checks cover
retired-record rejection, amendment retention without retrospective `prevrpt`,
orphan accessions, nonfinite values and archive-member/resource rejection.
Streaming local inspection only; timestamps remain source-local/unverified and
causal readiness is explicitly false. Real coverage, source provenance, concept
and vintage verification remain pending; no model inputs or new Colab units.

SEC bounded acquisition implemented and checked with32 offline reader/downloader
tests: fixed pre2024 endpoints, private configured identity, local serialized
requests, bounded retries/timeouts/bytes, immediate access/rate-limit stop,
checksummed archive/receipts and verified cache reuse. Real requests remain
pending identity configuration; no downloaded-data or financial readiness claim,
no allocation or new compute units. [Operation](filing-data.md).

### SEC schema/publication review — 2026-10-24 UTC

Primary-source review found December2024 archive reprocessing and a `segments`
NUM key absent from the older linked PDF. New structural reportv2 distinguishes
legacy/segment-aware schemas and audits duplicate/conflicting numeric keys using
temporary SQLite storage; no totals inferred from segment values. SEC FAQ also
confirms acceptance is not a guaranteed first-publication timestamp.37 offline
checks pass. Original audit scope unchanged, no observed financial data, returns,
training, allocation or additional units. Real acquisition still awaits private
request identity; historical fidelity remains unverified.
[Evidence](../results/filing-source-review-2026-10-24/README.md).

### Actual first SEC archive inspection — 2026-10-24 UTC

Request identity configured privately; official2020q1 archive acquired once,
96,584,608 bytes/SHA48ed9834…. Real-format regressions fixed: exact optional
`readme.htm` allowed without extraction; `.0`/fractional acceptance strings
strictly parsed/preserved.44 focused tests and lint pass. Full structural audit
failed on the retained pre2024 NUM-period rule. Separate metadata-only complete
scan:3,099,352 numeric records, one2024-12-31`AmortizationOfIntangibleAssets`label.
Its semantics need original-filing review; no amount copied into diagnostic, no
model inputs, retired bars/returns, normalization, training or Colab allocation.
Source/receipt and failed outcome retained. No new compute units.
[Evidence](../results/filing-source-review-2026-10-24/README.md).

### First forward SEC period resolved — 2026-10-24 UTC

Original2020 filing's inline XBRL context explicitly labels the2024 amortization
record`ScenarioForecastMember`: known forward disclosure, not realized retired
period data. Original HTML/receipt retained SHA dede7989…; metadata-only resolution
published. Preserve strict failure/default reader; separate explicit addendum
quarantines every NUM period later than its accession SUB period, including
pre2024 forecasts. No excluded amount enters numeric comparisons/model inputs.
48 focused checks pass; optional actual archive inspection running. Publication,
identity, concept/duration and vintage fidelity remain unverified. No training,
normalization, new allocation or compute units.

Explicit-quarantine2020q1 inspection completed:5,816 filings/5,404CIKs,
3,099,352NUM total,3,103 excluded forward-period records,3,096,249 inspected unique
keys.108,017 missing values,1,538,091 segmented rows;3 acceptance-date anomalies.
No duplicate-key conflict among inspected records. Full immutable report retained
SHAbc452aa8…; compact summary in results. Not all excluded records are forecasts,
and structural completion leaves all causal-readiness flags false. Original
strict failure preserved. No financial features, fitting, test bars or new units.

### SEC identity discovery/coverage distinction — 2026-10-24 UTC

Current SEC ticker snapshotSHA93050f93… matches107/107 research stocks; discovery
only. Read-only join to2020q1 finds102 current CIK candidates with qualifying
filings; APA/BLK/IBM/KR/XOM absent. Original IBM index proves filing existence
independent of ZIP coverage; APA2021 reorganization proves current-parent
backdating unsafe. Reproducible report retains all historical/model-input flags
false.51 focused checks/lint pass; raw metadata/receipts retained privately,
derived evidence documented. No training, normalization or new compute units.
Next dated source identities and actual missing-accession/quarter coverage;
do not infer full-history readiness from102 positive current-ID matches.

### Dated original issuer/ticker evidence — 2026-10-24 UTC

Three original2020 filings verify document-level APA/6769,BLK/1364742,XOM/34088
pairings, distinct from current CIK candidates. HTML/checksummed receipts retained;
inline extractor preserves security classes/context and rejects malformed/shared
issuer ambiguity. No whole-history validity interval or feature eligibility
inferred.56 focused SEC checks/lint pass. Derived dated evidence published;
IBM/KR source gaps and chronological identity coverage remain pending. No model
training, normalization, retired prices/returns or additional compute units.

### IBM/Kroger adjacent-quarter/source inspection — 2026-10-24 UTC

Kroger annual published2020-04-01: exact accession appears in newly retained Q2
ZIP; IBM April/Kroger June quarterly filings also present in SUB. Targeted
metadata inspection only, no full Q2 NUM audit. IBM February annual main has no
selected financial concepts; Exhibit13 has them. Actual extraction cause unknown;
original supplementation needs its own validated contract. Main IBM issuer/ticker
match confirmed. Kroger original periodFeb1 versus SUBJan31 and acceptedseconds
versus minute label require preserving both; no exact-period/time parity claim.
Original sources/receipts and derived evidence retained.56 focused tests/lint
pass. No model fitting, retired price/return access or new Colab units. Large
master-index request stopped at byte cap; no data used. [Evidence](../results/filing-source-review-2026-10-24/README.md).
Next chronological source coverage and original-filing financial extraction/
period validation; current candidate matches remain insufficient for model inputs.

### Original XBRL money-source extraction — 2026-10-24 UTC

Restricted source reader now preserves exact original periods/units/decimal
values and all competing selected US-GAAP facts; no feature formulas or readiness
promotion. Actual IBM237/211excluded/26unqualifiedUSD and Kroger54/21/33 records.
Kroger exactannual2019-02-03–2020-02-01 retained; rounded ZIPperiod mismatch rejects.
71 focused SEC checks pass, including excluded amount suppression, duplicates,
source corruption and parser/reference failures. Original instances/receipts and
private amount-bearing extraction retained; public summary has metadata only.
No fitting, retired test prices/returns or new Colab units. Next chronological
source coverage and explicit financial-concept/vintage gates before any training.

### Full SEC source collection and chronological inventory — 2026-10-24 UTC

Bounded sequential collector completed59/59 registered quarters in601.7sec,
4,613,634,809 compressed bytes; all raw ZIPs/receipts retained, no temporaryfiles.
Hash-verified SUB inventory:366,856 unique submissions,342,981 qualifying filings,
14,100 issuers, no cross-archive duplicates/conflicts. First strict failure on
three blank periods retained; all are non-periodic/outside financial-form family.
Updated observer reports blanks, keeps period dates mandatory for10-K/10-Q.
81 focused checks/lint pass. Candidate/year heatmap and JSON reproduce exactly;
current CIKs fail to represent older BLK/XOM/APA/GOOGL histories. No lineage,
financial-feature or full NUM readiness claim; no amount read by inventory.
Last official account349.02/zero rate/assignments; no new Colab units or training.
Next dated lineage, concept/period equivalence and causal financial acceptance;
retired test remains isolated. [Evidence](../results/filing-source-review-2026-10-24/README.md).

### Source-backed issuer events and conservative daily availability — 2026-10-24 UTC

Original event/index sources verify Google→Alphabet2015-10-02 and
Praxair/LindeAG→Lindeplc2018-10-31. Registry preserves event type/source hashes;
no ratio comparability or full price-history identity inferred. Causal query
requires both effective session and available timestamp, keeps prehistory unknown,
and rejects an undocumented CIK switch. Fixed source-date+two-XNYS-open policy
reconstructs availability without guessing naive-hour UTC/DST. Explicit assumption,
not guaranteed first-publication evidence; exact-publication flag remains false.
90 focused SEC checks pass, including future isolation and late2023 availability
outside2023. Actual two source-backed queries verified. No financial inputs,
normalization, training, retired price/return access or additional units.
Next issuer/price-history completion and financial concept/vintage acceptance.

### Coherent-accession financial arithmetic — 2026-10-24 UTC

Fixed three source ratios before price joins/performance scoring: revenue growth,
net margin, liabilities/assets. Exact interval/form bounds, same-concept unique
prior comparables, duplicate equality and coherent accession vintage required;
conflicts/missing denominators remain unready. Actual IBM/Kroger original snapshots
match independent rational arithmetic exactly.108 focused SEC/financial checks
pass. Pure optional module remains outside market registry; no normalization,
price join, financial dataset acceptance, training or additional Colab units.
Reported ratios are company financials, not model return results. Chronological
issuer/price/availability coverage and bounded experiment gates remain pending.

### Publication-gated financial source history — 2026-10-24 UTC

Fixed as-of/staleness rules: source CIK and available timestamp first, latest
reporting period then available accession; ambiguous equal-rank sources reject.
Unready latest records block fallback. Annual450/quarterly180calendar-day age from
report end is not reset by amendments. Source-backed IBM/Kroger before/at/expiry
checks pass;116 focused tests total. Future amendments do not rewrite earlier
queries and records/results remain independent copies. Complete supplied ledger
still mandatory: two prototypes do not justify a cohort coverage percentage.
No financial-input or price-link approval, no price join/fit/training/new units.
Next original-filing cohort ledger and verified chronological identity/coverage.

### Original-candidate cohort and earliest-source compatibility — 2026-10-24 UTC

Hash-verified59 financial SUB archives yield6,161 original candidates/112CIKs
for107 stocks, including explicit older IDs; no fuzzy names or historical-price
approval. Ledger completeness remains false because table ZIPs omit real filings.
Compressed Q1master3.81MB/expanded29.47MB recovers known IBM missing accession;
full master-index comparison is the next completeness step. Early IBM/Kroger2010
originals use`xbrl.us` namespaces; supported explicitly now. Extraction succeeds,
but fixed arithmetic remains unready (ProfitLoss/goods-sales labels unsupported).
No alias/gate relaxation to force a pass; mixed taxonomy versions unready.
121 focused checks/lint pass. No financial inputs, normalization, training,
retired price/return evaluation or additional Colab units. Retain negative
source outcomes and audit concept coverage before large original downloads.

### Complete master-index candidate comparison — 2026-10-24 UTC

Separate bounded master collector complete:59ZIPs/186.76MB/157.9sec;preflight
cache reused.14,876,676 metadata rows scanned:6,412 issuer/accession pairs,
6,410 unique accessions. All6,161 financial pairs found,251 additional pairs
represent249 additional documents plus2 joint Google/Alphabet links. Six date
differences retained; forms agree, no duplicate conflict. No silent current-ID
backdating, co-registrant collapse or earlier-date selection.131 focused tests/
lint pass; full output reproducible. Current-master metadata completeness is not
original-vintage/price-link/financial acceptance. All sources retained, collectors
terminal/no temporaryfiles; official account349.02/rate0/assignments0, no new units.
Next concept-coverage audit and validated original-statement collection rather
than assuming financial-table completeness. [Evidence](../results/filing-source-review-2026-10-24/README.md).

### Fixed financial-concept presence audit completed — 2026-10-24 UTC

Two local metadata workers scanned148,956,855NUM rows/59quarters in212.4sec,
then stopped.6,161 financial pairs assessed/251 master-only unassessed. Supported
revenue85.4%,NetIncomeLoss85.7%,assets99.4%,liabilities62.9%;all four/commonflow
2,695/6,161=43.7%. Presence is not original arithmetic or stock-price readiness;
flat-table gaps can be recoverable from originals and unsupported counts overlap.
No labels broadened, imputation, ratios, normalization, price joins or training.
144 focused tests/lint pass, metadata/HTML reproducible, dependencies checked
against source commit, evidence retained/no temporaryfiles. No new Colab units.
Next balance-sheet identities/original concept equivalence before bulk extraction
or allocating model runs; avoid spending training credits on unresolved inputs.
[Evidence](../results/filing-source-review-2026-10-24/README.md).

## Fixed saved-SAC soft-consistency evidence — 2026-10-24

Original-source inference-only diagnostic passes its descriptive flag:64 flat
training-reset states×4 stochastic rollouts, mean minQ−soft-target3.796906,
lower-two-SE3.625062≥0.1 and204/256 full512 trajectories≥75%. Learned target-critic
bootstrap and conditional sampling limit interpretation; no unbiased true-Q,
wrong-gradient or trading-performance claim. Source weights/optimizers/updates/
normalization/RNG unchanged; environment closed, zero optimizer/Colab steps.
199 combined focused checks pass. Supports one separately preregistered critic
algorithm comparison, not reopening closed SAC grids. No TQC screen or model
promotion yet. [Evidence](../results/sac-soft-consistency-2026-10-24/README.md).

Critic-study initialization milestone:12 untrained actor/critic/target archives
and the native SAC post-initialization globalTorch/NumPy/Python RNG are pinned.
All three actors match; TQC0/TQC2 full tensors/empty optimizers match; nativeSAC
standard initialization remains unchanged.61 fresh-model/state/seed/ownership/
RNG/serialization checks pass, no optimizer steps, data access or Colab units.
Local assets are retained and package/source hashes recorded; actual learned
optimizer, task/archive/controller and Colabsmoke acceptance remain pending.
[Evidence](../results/distributional-critics-2026-10-24/README.md).


Distributional-critic version33 integration ready: training/task/archive/fleet
contracts, actual saved-model/replay restoration and persistent shared two-unit
study baseline implemented.12 checksum-identical initializers now ship in tracked
configs.115 focused checks pass without local optimization; actual Colab smokes
and all new performance gates remain pending. No new return or promoted model.
See[study guide](distributional-critics.md).


Distributional-critic actual Colab smokes accepted:3×1024 CPU transitions, actual
Python3.13.16, initial tensor/RNG and learned Adam/replay restoration,192 exact
frozen predictions and immutable normalization. All3owned sessions stopped.
Conservative project39.039379/150, incremental0.025001, official account349.00
with zero assignments. Fixed screen runtime projection3717.83<7200seconds;
50k seed11 three-arm plans prepared. No completed screen, return improvement,
retired-test access or promotion. Evidence:results/distributional-critics-2026-10-24/smokes.json.


Fixed critic50k seed11 screens launched in three centrally metered Colab CPU
sessions from immutable8c27af4 source after actual smoke acceptance; all actual
kernels report3.13.16. No completed performance result yet. Reporting now verifies
owned-cleaned archives, actual replay/optimizer/frozen state, matched baseline
paths, selected/final curves and declining-stock diagnostics, checksum-pinned
historical SAC metadata and unchanged0.01pp/0.05pp reproduction tolerances.
118 focused checks/Ruff pass; report-time checks perform no local learning.
Confirmation remains blocked on all fixed gates, not elapsed training steps.


Completed fixed critic50k screen failed: nativeSAC selected/final7.9281/7.9281%,
untruncatedTQC1.9991/1.9991%, truncatedTQC9.7018/7.6975%, holding30.6019%.
Candidate fails both-control/absolute/risk gates; no confirmation, grid or
extension. NativeSAC reproduces every checkpoint and selected/final symbol
exactly.384 smoke/full frozen comparisons and actual learned optimizer/replay
checks pass; six owned sessions stopped. Conservative project39.154379/150,
study0.140001/2, official account348.89 with zero rate/assignments. Retired-test
isolation and immutable normalization preserved; objective remains unmet. See
results/distributional-critics-2026-10-24/screen.json and compute.json. Publication
and scoped duplicate cleanup remain next; no promoted model.


Distributional-critic publication/cleanup verified:12 GitHub release asset hashes
and sizes match;18 duplicate transports/upload payloads removed (520,262,328
bytes),191 retained files rehashed at cleanup. All models/replay/checkpoints,
source/data, curves and owned recovery records retained. Subsequent README
publication update is explicitly recorded in cleanup evidence. Six owned sessions
stopped, zero assignments; no additional units or training. Fixed critic study
failed and remains closed; next work is bounded saved-policy/execution sensitivity
diagnosis, not extension or promotion. Release:
https://github.com/jason490/RL-trading/releases/tag/distributional-critics-2026-10-24


[Saved critic final-update sensitivity](../results/critic-update-2026-10-24/README.md) complete: three one-update pairs on
256 fixed candidate training-replay observations; truncatedTQC actor relativeL2
0.03356%, exposure mean absolute delta0.001462, GaussianKL0.001133 and zero
exposure-sign flips. These are normalized action/tensor diagnostics, not returns
or a cause of development deterioration. Independent Torch arithmetic agrees;
all protected files/model/Adam state unchanged, worker exited0, no local
optimization or Colab units. First phase-filename rejection and reviewed v2
correction retained; sample/metrics unchanged. Fixed failed family remains
closed. Next bounded saved-development-path audit before another training
hypothesis; no promotion or retired-test access.


[Saved TQC path audit](../results/critic-paths-2026-10-24/README.md) complete: all26 selected/final development paths
reconcile−2.004248pp. Stocks closing quantities differ72.94% of sessions, sides
2.97% and protective reasons2.24%; stock long contribution change−2.5417pp, short
−0.1624pp. Same-day partitions are observations, not counterfactual causes or
proof that earlier protection effects are absent. Initial legitimate-null
rejection and reviewed schema correction retained.27 tests and independent
NumPy verification pass, worker exited0/source unchanged, no models/training/
new Colab units or retired data. Current account348.89,0rate/assignments; project
39.154379/150. Failed critic family stays closed. Next candidate is a bounded
source/availability audit of historical option-implied volatility, not another
architecture/steps rescue; no new inputs or training authorized by feasibility
evidence alone.


[Implied-volatility source feasibility](../results/vix-source-2026-10-25/README.md) passed: single date-bounded Yahoo
request retained3,774 observations2009–2023; lagged level/change20 context covers
all3,433 original-dataset decisions (100% overall/eras/years), with exact prior
sessions and assumed availability no later than decisions. Four tests and
independent pandas/source/dataset/normalizer verification pass; worker exited0.
Historical publication vintages unverified; source values remain local and model
inputs disabled. No training, new return or Colab units. Next audit the accepted
111-symbol consumer and preregister a separate matched feature comparison only
after coverage; failed families and retired-test isolation remain unchanged.


[VIX information comparison preregistered](../results/vix-context-2026-10-25/README.md) after expanded-consumer coverage
passes all298,946train/19,058development rows (111/26symbols,100% unique-date/
era/year readiness). Independent pandas/source/normalizer verification passes;
consumer worker exited0, no requests or new units. Fixed legacy/masked/full
21/23-input recurrent controls,100k seed11 steps, original common tensors/RNG
and zero new LSTM columns, immutable seven-field preprocessing, fixed both-
control/absolute/risk gates and historical reproduction. Only passing512smokes
then screens permit11/23/37/prospective; study2units inside original budget.
Nine tests pass; no new policy construction, training, performance score or
Colab allocation. Payload/initializer/frozen/runtime implementation is next;
historical vintage and retrospective universe limitations remain unverified.


[VIX immutable payload/initialization acceptance](../results/vix-context-2026-10-25/README.md): full/masked payloads
reloaded and exactly reconstructed; original bars/ATR/splits/seven fields and
preprocessing unchanged. Only full's2 new fields fit298,946 training rows;
independent pandas/NumPy verification passes.12 initializer entries/3new23-input
archives retain exact original common tensors/CPU RNG and zero new actor/critic
columns; seeded NumPy/Python fingerprints verified. Binding/portable artifacts
are pinned,70 focused tests pass, source workers exited0. No policy optimization
or Colab units. Learned/runtime acceptance remains pending. Next explicit family
dispatch plus task/archive/report/runtime integration and actual Colabsmokes;
legacy-overlap must not change previous controls. No new performance result.

VIX fixed two-coordinate screen is closed failed (2026-10-25): full selected/final
24.09/14.44%, legacy19.62/18.48%, masked24.36/9.20%, holding30.60%. Exact historical
control reproduction,468 frozen sequence checks and independent group/accounting
acceptance pass; return/stock/ETF/median gates fail. No VIX level/change/window/
feed/normalization/model/reward/steps grid, extension, confirmation or promotion.
All six owned sessions stopped; project39.272458, account348.80 idle. See
[formal evidence](../results/vix-context-2026-10-25/README.md). A new experiment
requires a distinct saved-evidence rationale; no robust winner is established.

[Saved recurrent value-target audit](../results/recurrent-values-2026-10-25/README.md)
completed52 fixed training-only stochastic windows/26,624 transitions. Primary
zero-tail EV0.029 selected/−0.084 final, self-bootstrapped GAE EV0.491/0.549.
Independent NumPy/SB3 verification passes; all source models/Adam/data/ledger
unchanged, worker terminal0, no new optimization/Colab units/retired data. These
censored, correlated and partly self-predicted targets do not prove calibration
or a cause of deterioration. Next separately fix a training-only benchmark/noise
decomposition; no immediate reward/GAE grid or rescue of closed families.

Fixed benchmark/noise decomposition completed and failed feasibility: only3/8
conditions pass for coefficient exactly1. Both ETF phases have higher reward and
discounted-target variation; no coefficient/benchmark/window/discount search or
training follows. Independent cash/share accounting exact, covariance/discount
ratio error<=1.12e−15, worker terminal0 in5.22s, no model replay/new units/source
changes. See results/benchmark-variance-2026-10-25/README.md. Prior value-target
consistency still does not establish a causal optimizer diagnosis or reward fix.

A distinct GAE-trace comparison is preregistered before new model construction:
lambda1 versus0.95 native RecurrentPPO, with all data/rewards/gamma/actions/
initialization/steps unchanged. The saved value audit motivates a trace-estimator
hypothesis, not a proven critic bias; benchmark subtraction stays closed. Only one
candidate, fixed both-phase/group/risk gates and exact historical control tolerance;
no trace/discount/optimizer/step grid after failure. Twelve focused parameter,
SB3 segmented-return and prior diagnostic checks pass. No new model or training.
See docs/gae-traces.md and configs/experiments/gae_trace_preregistration.json.
Runtime/shared-start/actual-buffer/archive integration precedes Colab smokes.

GAE shared-start/model records and standalone actual pre-update rollout auditing
are implemented. Full64 original tensors/RNG for11/23/37, source proofs, empty
coupled Adam and one-field lambda difference validate. Actual-buffer fixtures
check raw/timeout-adjusted rewards, boundaries, float32 fields, GAE and independent
segmented MC targets; compact hashed NPZ/JSONL evidence is retained.61 focused
non-optimizing regressions/Ruff pass. Runner dispatch/callback attachment and
version35 fleet/archive/learned acceptance remain before Colab smokes; no new
session, model return, reward or data. See results/gae-traces-2026-10-25/README.md.

GAE runner wiring is complete: strict requests/data/defaults validate before
model creation, common start/snapshot and initial/selected/final learned-state
records attach, and actual pre-update rollout auditing runs beside validation.
Failed target arrays are retained before raising; no retry is implied.78 focused
non-optimizing request/buffer/initialization/serializer/shared/VIX checks pass.
Version35 job/budget/archive and learned acceptance remain before Colab smokes;
no new training, return or session. Previous failed families remain closed.

GAE version35 task/fleet/budget/archive integration implemented: persistent shared
two-unit baseline, actual-smoke/profile gate, original account/project safeguards,
exact retained rollout arrays/hash/file-set/timeout/GAE/MC reconstruction and
model/Adam/frozen/panel contracts.113 focused GAE/prior VIX/fleet regressions plus
5 acceptance-boundary cases pass; no new optimization/session/data/return.
Physical learned restoration and smoke-acceptance tooling remain before actual
512-transition CPU runs. Previous families stay closed and retired test isolated.

GAE physical latest/selected/frozen model/Adam/rollout verification and strict
owned-cleaned smoke acceptance implemented;86 focused physical/archive/rollout/
job/VIX regressions pass without optimization. Only diagnostic max-error values
have tiny inter-host rounding tolerance; hashes/actual target/performance gates
unchanged. No learned acceptance or return yet. Next official usage and clean
source/kernel preflight, then two512-transition CPU smokes before100k screens.

Actual GAE CPU smokes accepted:2×512 transitions from clean315614e, actual default
Python3.13.16, common starts/RNG, learned Adam,156 frozen sequence matches, all8
actual rollout arrays and development accounting verify. Both owned sessions
stopped, retrieved files/preprocessing unchanged. Project39.292458, study0.020000/2,
official348.78 idle/zero assignments. Projection3311.21s allows preparing fixed100k
seed11 jobs; they are not allocated yet. Only unchanged screen/reproduction gates
permit confirmation; no short-side or performance success is established.


### Completed GAE trace screen — 2026-10-25

Both fixed100k CPU seed11 runs completed with verified retrieval and owned shutdown.
Control selected/final19.624533%/18.484355%; lambda1 candidate18.442942%/12.675114%,
matched holding30.601851%. Relative/absolute/median return gates fail; no promotion,
confirmation, extension or trace/gamma/optimizer grid. Historical checkpoint and
per-symbol reproduction is exact.156 full frozen predictions,200 actual rollout
arrays and104 report episodes pass;218 protected files remain unchanged. Smokes
add156 predictions/eight arrays. No retired-test access or broker orders.

Conservative project39.351004 units; total GAE increment0.078547/2. Official account
348.74, zero rate/assignments; all four owned smoke/screen sessions stopped.
[Detailed results](../results/gae-traces-2026-10-25/README.md) include short-side
limitations, fixed gates and evidence. Public release/duplicate cleanup remain
next; the broader performance objective remains unmet.


### GAE trace publication and cleanup

[Research release](https://github.com/jason490/RL-trading/releases/tag/gae-traces-2026-10-25):
14 server asset hashes/sizes verified; four frozen bundles,208 actual rollout
arrays and interactive curves retained. Exact-member cleanup removed nine
confirmed duplicates (290,074,611 bytes/276.64MiB). All1,002 protected files remain
unchanged; all four differing checkpoint snapshots, intermediate weights, source,
data, arrays and original ledger remain. All four owned sessions stopped; no
recovery needed. [Inventory](../results/gae-traces-2026-10-25/cleanup.json).
Fixed trace family failed and remains closed; broader objective remains unmet.


### Saved GAE target variability audit preregistered

Before any new training, compare the two original trace estimators on each of
200 accepted actual training buffers, preserving identical rewards/values/masks
within each pair. Fixed100/first50/last50 summaries, population variance ratios,
explicit zero denominators and raw-reward/endpoint/float32-rounding decomposition.
This does not reconstruct recurrent minibatch normalization or gradients and
cannot diagnose return causally or reopen the failed trace family. No models,
market replay, data requests, optimizer steps, Colab units or retired prices.
[Specification](gae-variance.md) and pinned protocol precede actual audit.


### Saved GAE variance audit complete

All200 actual buffers revalidated read-only. Within-path lambda1/lambda0.95 median
advantage variance ratios4.408252 control/4.018140 candidate; return variance
ratios2.779849/2.422366. Endpoint critic dependence persists (195 timeouts, no true
terminations per arm). Independent discount matrices agree within1.05e−17 variance
and1.60e−14 ratio error. All221 protected source files/ledger unchanged; workers
completed, no model/market replay, optimizer steps, new Colab units or retired data.
Raw target variance does not establish normalized recurrent-minibatch gradient
noise or causally explain returns. No trace/gamma/optimizer/model/reward grid or
extension follows; the fixed family stays closed. [Evidence](../results/gae-variance-2026-10-25/README.md).


### Distinct interest-rate source proposal

After closing GAE, preregister a free official Treasury source audit: two-year
par yield and ten-minus-two-year spread,2009–2023 annual XML only, explicit
second-following-XNYS-session availability assumption, bounded observation age,
methodology-break and vintage limitations. Coverage99.5% overall/era and99% year
on original111/26 episodes precedes any input or training. Offline parser checks
units, missingness, duplicate/schema/date/retired scope and bounded UTF8 XML;
no actual yield history, model replay, returns, preprocessing or Colab allocation.
[Guide](treasury-context.md). Acquisition and causal coverage remain next.


### Treasury acquisition/alignment preflight ready

Bounded annual collector, immutable response/receipt/index cache, global operation
lock/pacing/deadline and secret-free failure evidence are implemented. Lag2 XNYS
alignment handles Treasury/NYSE holiday differences, latest-null rejection and
bounded observation age. Both unique-date and all-consumer-row coverage thresholds
are explicit before observed data; per-symbol/year/era/methodology reports remain
outcome-free.48 offline checks and targeted Ruff pass, including source failure
cleanup/no-restart. No actual yield download, model input, fitted normalization,
training or Colab session yet. [Guide](treasury-context.md).


### Treasury source audit passed

Actual15 annual requests retained3753 observations/5194901 raw bytes in25.56s.
Coverage3432/3433 dates (99.970871%) and317894/318004 rows (99.965409%) passes all
fixed overall/era/year gates; all19058 development rows ready.2010-10-13 remains
unavailable for110 training rows because latest eligible source yields are null.
No fallback/imputation or denominator change. Independent original XML and pandas/
calendar verification matches all values/timestamps/counts exactly; all15 receipts
and source/data/normalizer/ledger hashes verified. Worker exited0, lock released,
no temporary files or new Colab units. Current-vintage/publication assumptions
remain explicit; model inputs stay disabled. Next matched readiness/segmentation
acceptance before a separate training protocol. [Evidence](../results/treasury-source-2026-10-25/README.md).


### Model/technique literature review — 2026-10-26

[Primary-source review](model-research.md) covers CrossQ, recurrent SAC/TD3,
boosted signal baselines, direct momentum objectives and evaluation uncertainty.
CrossQ is present in installedSB3-Contrib2.7 but unregistered; default1024-wide
critics require capacity/cost profiling, and frozen batch statistics need explicit
acceptance. Recurrent off-policy replay is distinct from prior recurrent PPO but
requires sequence/burn-in/reset tests. Prior failed MLP/forest/TQC/GAE gates stay
closed; no paper establishes US-stock gains here. Finish current Treasury payload
work separately before any new algorithm protocol. No installation, model
construction, optimization, Colab allocation, new units or retired-test access.


### Treasury readiness payload implementation

Fixed missing-row/preceding20-row purge, contiguous513-bar eligibility and one
segment per symbol are implemented for matched masked/full datasets. Original
seven-field preprocessing/retained prices/ATR and all validation rows stay exact;
only new full fields fit retained training rows. Unregistered Treasury data rejects
before generic trainer model/environment construction. Synthetic causality, mask,
segmentation, sampler and prior GAE dispatch regressions pass; actual preparation
and independent acceptance remain next. No model initialized, optimizer step,
provider request, new Colab unit or retired data. [Guide](treasury-context.md).


### Treasury matched payload acceptance complete — 2026-10-26

Actual masked/full immutable payloads reload and reconstruct exactly. All111
training symbols,282666 retained training rows,19058 unchanged development rows
and225834 eligible512-step starts agree.16280 training rows excluded entirely in
2010 under fixed missing/purge/short-segment rules;2011–2020 unchanged. Independent
NumPy segmentation/decimal inputs exact, training-only new statistics agree within
1.96e−13; original seven statistics/bars/ATR and nine protected source/ledger files
unchanged.46 focused checks/Ruff pass, worker exited0 in68.11s, no temp progress
file, provider requests, model/optimizer/Colab activity or retired data. Generic
Treasury training remains blocked until a separate registered comparison. Next
fix legacy/masked/full model controls/shared starts and acceptance before any
allocation. [Evidence](../results/treasury-context-2026-10-26/README.md).


### Treasury information model comparison preregistered

After matched payload acceptance, fix three arms legacy/masked/full, native64-wide
RecurrentPPO and unchanged optimizer/reward/actions/costs. Seed11 screens100k/10k,
512-step episodes, actual512 smokes/restoration before allocation. Original common
weights/RNG plus reusable unfitted23-input zero-column archives, no learned VIX
signal or model initialization yet. Same both-control/absolute/median/risk gates,
historical legacy reproduction and fixed11/23/37/prospective requirement; failure
closes the family without grids. Dedicated initializer/dispatch/runtime/archive
integration remains; generic Treasury training stays disabled. No new units or
retired-test access. [Protocol guide](treasury-context.md).


### Treasury shared-start and frozen/request contracts

Pinned original/reusable unfitted21/23-input tensors, zero-column mapping, actual
61,513/62,537 geometry, original RNG, coupled Adam/defaults and seed checks are
implemented. Pre-mutation validation and exact empty ownership guard source loads.
Portable frozen lineage/lag/age/readiness/normalizer checks and masked-channel
rejection preserve recurrent carried/reset state and caller inputs in untrained
fixtures. Manual Adam serialization is not learned acceptance. Dedicated request/
data validators are standalone; generic trainer stays blocked pending runner/
runtime/archive integration and actual Colab smokes. No local optimization, source
requests, new units or retired data. [Guide](treasury-context.md).


### Treasury actual untrained asset/data acceptance

Read-only verification validates six tracked21/23-input initializer archives and
all three actual datasets. Independent NumPy blocks verify copied columns, zero
new columns,61,513/62,537 counts and identical original RNG bytes for11/23/37.
Source hashes/caller Torch/NumPy/Python streams and original ledger unchanged;
verification worker exited0, no models constructed, gradients/optimizer steps,
provider requests or new Colab units.108 focused prior/new regressions pass.
Actual learned smoke acceptance remains; runner/job/archive/budget integration is
next and generic training stays blocked. [Evidence](../results/treasury-context-2026-10-26/contracts.json).

### Full-panel recurrent SAC comparison registered — 2026-10-27

Fixed111-training/26-development GRU-SAC versus full-input native SAC and masked
RecurrentPPO package reference:100k per cell, post-update10k checkpoints, seed11
screen and conditional23/37 confirmation. Both selected/final paired improvements,
absolute stock/ETF/median gates and risk limits are fixed; no rescue grids.
Pure timing admission and synthetic multi-seed gate checks are implemented;
portable starts, actual Colab profiling and worker/fleet/physical acceptance remain
before performance allocation. Different information/capacity/batches are explicit.
Original budget, CrossQ stop and retired isolation persist; no new allocation.
[Protocol and acceptance](recurrent-sac-comparison.md).

### Portable recurrent comparison starts accepted — 2026-10-27

All nine fixed architecture/seed starts generated without fitting and independently
restored in a second process. Exact tensors, empty optimizer state and full RNG
identities pass; actual assets retained under `.runtime/recurrent-comparison-starts-v1`.
Tests forbid local native training, learner updates and Adam steps. No new Colab
allocation or performance finding. Next actual profiling worker/runtime acceptance;
[guide](../docs/recurrent-sac-comparison.md).

### Fixed recurrent comparison profiling worker implemented — 2026-10-27

Guarded actual-Colab-only profile loads accepted starts and fixed full panels;
512 candidate transitions/64 updates,64 native SAC updates and3,000 PPO
transitions with observed optimizer counts. Saves replay, checkpoints, reports,
timings and provenance; failure preserves partial evidence and closes environments
and hooks.32 focused offline tests/Ruff pass. No local fitting or new allocation.
Dedicated fleet/archive physical acceptance remains before actual profile/performance.

### Dedicated recurrent comparison profile runtime — 2026-10-27

Fixed version41 CPU task, explicit portable-start upload, direct remote dispatch,
original-ledger two-unit study baseline/0.1-unit profile admission and single
claim implemented.900-second allocation/420-second teardown reserve, no retries,
account floor and aggregate metering retained. Strict128MiB archive allowlist plus
physical checkpoint/Adam/count/config/replay/report/source verification implemented.
Actual Colab profile remains pending; no performance task or new allocation yet.

### Actual CPU comparison profile deadline stop — 2026-10-27

Clean c19f6d7/defaultPython3.13.18; original CPU allocation stopped by working
deadline, cleanup verified and claim/budget-stop preserved. Salvaged GRU64/SAC64
updates and PPO3,000 transitions/600 Adam steps restore; full completion/provenance
missing, so profile acceptance remains incomplete. Retrieved preliminary projection
requires18,744.80s GRU/7,643.47s SAC/4,103.10s PPO; both SAC arms fail7,200s.
No100k allocation/retry/performance interpretation. Project39.54822735/150,
account348.58 idle;0.01209140 estimated versus0.01 rounded measured units.
43.8MiB duplicate staging removed, scientific artifacts retained. Observed184MB
archive corrects validator bound to256MiB without accepting the incomplete run.
Next review costs/durable checkpoint publication and separate execution proposal.

### Lossless I/O and cost review after profile stop — 2026-10-27

Actual retained184MB tree benchmark: gzip1 mean1.199s vs gzip6 mean2.894s locally,
all13 member hashes exact, temporary archives removed. Shared/final transport
archives use level1; failure preserves previous complete publication. Profiling
now publishes learned model/partial evidence before subsequent work/evaluation,
counting early export in I/O totals. No new allocation or optimizer work.
GRU still projects4.89h with evaluation/I/O entirely removed; I/O cannot rescue
the two-hour plan. Longer CPU6/3/2h option is a costed proposal only: observed
rate0.08 implies0.88 units for seed11/2.64 for all three seeds' ceilings.
Original two-unit study cap, ledger stop, claim and performance gates unchanged.

### Full100k recurrent comparison worker implemented — 2026-10-27

Actual-Colab-only native SAC/RecurrentPPO and portable-initialized GRU paths
implemented, retaining100k transitions/10k post-update checkpoints and all fixed
research gates. Native actual gradient/weight/Adam counters are bounded and
checked; epoch-versus-Adam distinction explicit. Checkpoints export before
evaluation; selected/final reports, compact return-change curves and failure
cleanup retained. Offline untrained save/load and guard/counter checks pass with
optimization forbidden. No allocation or actual100k result; reviewed execution
admission, confirmation gates and physical performance acceptance remain.
