# Research results index

Returns below are mean net total returns from separate matched symbol episodes
on repeatedly reused2021–2023 development data. They are not annualized returns,
a combined portfolio, or independent prospective evidence. No model has passed
the full required multi-seed and prospective acceptance process.

## Current research status

[Fixed distributional critics](distributional-critics-2026-10-24/README.md) failed:
nativeSAC selected/final7.93/7.93%, untruncatedTQC2.00/2.00%, truncatedTQC9.70/7.70%,
matched holding30.60%. Both-control, absolute and risk gates fail; fixed family
closed without confirmation or tuning extension. Historical nativeSAC reproduces
exactly.384 smoke/full frozen checks pass; all six owned sessions are stopped.
Project39.154379/150, official account348.89 with zero assignments. All12 release asset hashes/sizes verified;18 duplicates/496.16MiB removed with
191 retained-file checks. No robust winner or prospective acceptance exists.

## Latest completed history comparison

[Full/recent history](training-history-2026-10-23/README.md): full selected/final
19.62/18.48%, recent20.95/5.60%, holding30.60%. Selected improvement below2pp;
final and absolute gates fail. Fixed family closed, no confirmation/promotion.
Measured2020 stock exposure1.98%→6.64% confirms intended composition change.
312 frozen checks including smokes; all four sessions stopped. Project38.683293,
account349.27 zero rate/assignments. 17 release assets verified;35.93MB duplicate archives removed with148 retained
hashes unchanged.

## Latest completed optimizer comparison

[Recurrent coupled L2](recurrent-regularization-2026-10-23/README.md): control
selected/final19.62/18.48%, candidate30.45/18.42%, holding30.60%. Selected overall
near holding but ETF/median targets fail; final and relative gates fail. Fixed
family closed, no confirmation or coefficient sweep.312 frozen checks including
smokes, all four sessions stopped; project38.746179/150, account349.23 zero
rate/assignments. 16 release assets verified;35.74MB duplicate archives removed with145 retained
hashes unchanged.

## Recent completed comparisons

| Study / arm | Selected | Final | Interpretation |
| --- | ---: | ---: | --- |
| [Cash-study posthoc Gaussian control](recurrent-cash-2026-10-22/README.md) |32.80%|32.80%|Strong prior one-seed control; absolute ETF/other gates remain unmet |
| [Update-guard fresh control](recurrent-guard-2026-10-22/README.md) |19.62%|18.48%|Fresh pinned initialization; matched guard control |
| [Update-guard candidate](recurrent-guard-2026-10-22/README.md) |28.07%|24.44%|Relative improvement, absolute gates fail; family closed |
| Matched buy-and-hold |30.60%|30.60%|Same26 symbol episodes and costs |

Different studies can have different untrained tensor starts despite the same
seed label. Compare candidates with their own matched controls, not as if every
row were an identical initialization. Best interim checkpoint results do not
establish stable final or multi-seed performance.

## Latest completed feature experiment

[Session-context study](session-context-2026-10-22/README.md) completed three
matched100k recurrent PPO runs. Full selected/final26.42/16.08%, masked27.66/10.76%,
legacy19.62/18.48%, holding30.60%. Both relative comparisons and absolute targets
fail; no confirmation/promotion. All six smoke/screen sessions stopped,468 frozen
predictions pass. Publication and checksummed duplicate cleanup are verified; models/reports are linked by the study README.

## Latest completed capacity experiment

[Recurrent capacity](recurrent-capacity-2026-10-22/README.md): both100k seed11
runs complete. Control64 selected/final19.62/18.48%; larger12828.15/13.43%;
matched holding30.60%. Larger selected improvement fails to persist; final and
absolute gates fail. Fixed family closed, no extension/confirmation/promotion.
312 total frozen checks pass; all four sessions stopped. Project38.601665/150,
account349.34 zero rate/assignments. Strict evidence/curves retained;15 release assets verified and45.88MB duplicate
publication archives removed with142 retained hashes unchanged.

## Recent diagnostics

- [Rejected native crossover control](l2-crossover-2026-10-23/README.md): direct and wrapper match, source Colab sub-cent differences exceed fixed tolerance; no mixed outcomes.

- [Saved recurrent L2 actions and costs](l2-actions-2026-10-23/README.md): stock deterioration with similar exposure and lower costs; crossover diagnostic pending.

- [Saved history optimizer/parameter audit](history-optimizer-2026-10-23/README.md): finite restored Adam states; displacement and KL are descriptive, not causal proof.

- [Training/development saved action comparison](history-fit-actions-2026-10-23/README.md): similar final short exposure, different payoffs; descriptive only.

- [Full/recent saved-model common training-window replay](history-fit-2026-10-23/README.md): all four benchmark flags fail; no fitting or new units.

- [History-study saved actions and costs](history-actions-2026-10-23/README.md)

- [Capacity selected-to-final actions and costs](capacity-actions-2026-10-23/README.md)

- [Saved actions and costs](guard-actions-2026-10-22/README.md)
- [Exit and inherited-position partitions](guard-exits-2026-10-22/README.md)
- [Exact overnight/intraday P&L bridge](guard-bridge-2026-10-22/README.md)
- [Overnight gaps by preceding exits](guard-overnight-2026-10-22/README.md)

## File organization

Each dated study folder contains a README, compact machine-readable results,
verification and limitations. Category methods live in [docs](../docs/README.md),
fixed numeric definitions in [experiment configs](../configs/experiments/), and
reusable code in [the package](../src/rl_stocks/). Large models, raw data,
transports and derived HTML stay in ignored artifact/data directories and, where
published, checksummed GitHub releases linked by the study README. Historical
studies remain in their dated directories to preserve source and artifact links.

The detailed chronological decision record is [the research log](../docs/research-loop.md).

[Training-history coverage](training-history-2026-10-23/README.md) audits date-only
availability and expected window year mass. Both full/recent sources support111
symbols; expected2020 mass2.4% versus7.3%. No training or performance claim;
actual year logging and a frozen matched protocol remain prerequisites.

[Completed recurrent PPO regularization](recurrent-regularization-2026-10-23/README.md):
fixed Adam decay0/0.0001, same full-history model/data/steps. Candidate selected/
final30.45/18.42% versus control19.62/18.48%; absolute and final gates fail.
Family closed. Publication/verified duplicate cleanup and strict reproduction
complete; no confirmation, sweep or promotion.

[Feature-scale audit](feature-scale-2026-10-23/README.md): outcome-free frozen
training/development z distributions and source arithmetic. Large training tails,
no model fitting or new units. The fixed cap5 preprocessing protocol and immutable
payload are now prepared in [input-clipping](input-clipping-2026-10-23/README.md);
The completed fixed clipping comparison failed its gates; see its dated result below.


## Fixed input clipping completed

[Clipping](input-clipping-2026-10-23/README.md) failed: selected/final24.70/18.57%
versus control19.62/18.48% and holding30.60%. No cap sweep or extension.312 frozen
checks including smokes, four sessions stopped, project38.816469/account349.17.

[Clipping saved-action diagnosis](clipping-actions-2026-10-23/README.md):
increased short exposure/costs and deteriorating net short contribution at final;
104 saved episodes, no causal claim, fitting or additional Colab units.


[Gaussian action-boundary audit](action-boundaries-2026-10-23/README.md): exact
conditional any-component out-of-bounds mass~79% on fixed flat-account training
prefixes; model/Adam/RNG/normalization unchanged, no fitting or new Colab units.
Supports a distinct preregistered bounded-policy hypothesis, not causal proof.

[Completed squashed-action comparison](squashed-actions-2026-10-23/README.md):
explicit transformed Gaussian sampling/likelihood, unchanged full64 tensors/data/
optimizer. Completed100k screen failed; details and verified cleanup in the dated result.


## Fixed squashed-action comparison completed

[Squashed Gaussian](squashed-actions-2026-10-23/README.md) selected/final15.47/5.14%
versus native19.62/18.48% and holding30.60%. Fixed gates fail; family closed with
no extension or transform/noise grid.312 frozen checks, four sessions stopped;
project38.903541/account349.10. Publication and saved-path diagnosis next.


[Squashed saved-action diagnosis](squashed-actions-diagnosis-2026-10-23/README.md):
both long/short contributions deteriorate at final while short exposure and costs
increase;104 saved episodes, no causal claim, fitting or additional Colab units.

[Compact market-context coverage](market-context-2026-10-23/README.md): SPY return/
volatility and own-relative20-session features are causally available on all
298,946 train/19,058 development rows.60 checks pass; no fitting, new model,
performance score or Colab allocation. Matched dataset/model study remains pending.


[Market-context smokes accepted](market-context-2026-10-23/README.md): three512
CPU Colab runs,234 exact frozen predictions and actual mask rejection, models/Adam/
mapped initialization/sampling restored, three owned sessions stopped. Project
38.934668/account349.08. Fixed100k three-arm plans prepared, no return result yet.


[Market-context saved-action diagnosis](market-actions-2026-10-24/README.md):
full model deterioration mainly reflects reduced long contribution even as short
exposure decreases.156 saved episodes, no causal claim, fitting or extra units.

## Pending distinct information-family audit

[SEC source-schema/publication review](filing-source-review-2026-10-24/README.md)
identified archive reprocessing, segment-aware numeric identity and publication
uncertainty.48 focused checks pass. First2020q1 strict failure retained; original
filing identifies the2024 label as a forecast. Explicit-quarantine inspection
completed3,096,249 numeric keys with3,103 forward-period exclusions. Causal
readiness remains unverified; no financial model inputs or training. This is
feasibility work, not a new return result or a promoted model.

Full SEC source collection now complete:59 quarters/4.61GB,366,856 unique
submission records. Current-CIK candidate coverage heatmap and issuer/year counts
are available;81 focused checks pass. These are source-audit results only;
financial features and training remain disabled pending historical identity,
publication and concept/period gates. No new model returns or Colab units.

Master-index comparison now recovers251 additional issuer/accession pairs
(249 unique documents) beyond financial-table candidates.131 focused checks pass;
all59 master indexes retained, collectors terminal. Original statement coverage,
financial concepts and issuer/price eligibility remain pending; no new model
returns, financial-input activation or Colab consumption.

Financial concept audit:148.96million NUM rows scanned;2,695/6,161 assessed
filing pairs (43.7%) have all required current tags in compatible groups. Direct
liabilities presence is62.9%. This is source metadata, not actual arithmetic,
stock coverage or model performance.251 master-only pairs remain unassessed.
144 focused checks pass; local workers stopped, evidence retained, no new units.


Original balance-identity audit: four retained IBM/Kroger annual instances reproduce
exactly; three Assets−Liabilities−equity-including-NCI residuals are zero, IBM2019
is −$1million and remains nonzero. Parent-equity residuals differ in all four.
Missing temporary equity is unknown, and missing liabilities are not inferred.
Default money extraction/financial rules unchanged; 152 focused checks and targeted
Ruff pass. No training/new Colab units. Financial dataset acceptance and a robust
model winner remain unachieved. See [accounting evidence](filing-source-review-2026-10-24/balance-identity-inspection.json).


Fixed original-source missing-liabilities probe: earliest of33 eligible2020 annual
filings is Intel0000050863-20-000011. Original end2019-12-28 differs from archive
20191231; original also lacks qualifying current direct liabilities, so unchanged
snapshot remains unready. One case only; no general coverage estimate, imputation,
financial-input activation, model return or Colab units. Offline source/selection/
report reproduction verified. [Evidence](filing-source-review-2026-10-24/missing-liabilities-probe.json).


Distinct flow-only financial family fixed/implemented: same-accession comparable
revenue growth and parent margin, no liabilities substitution or income aliases.
Five retained originals yield3 arithmetic-ready/2 unready; legacy IBM/Kroger
snapshots unchanged, ready ratios independently checked by Fraction arithmetic.
Current common-flow tag presence4,512/6,161 (73.2%) is metadata only, not causal
stock coverage.162 focused checks/Ruff pass; family remains disabled, no training
or Colab units. Original-source/issuer-price-day coverage gates remain before any
matched model experiment. [Evidence](filing-source-review-2026-10-24/flow-feasibility-inspection.json).


Original-flow pilot first collection stopped at the20MB source cap on eighth
selected document(AEP0000004904-15-000008). Seven complete originals/26,924,787
bytes retained and reverified;6 arithmetic-ready/1 unready are not a coverage
estimate.312 cases unfinished, gate not evaluated. Terminal failure preserved,
collector exited, no retry/replacement/restart or Colab units.183 focused checks
pass; source hashes/extraction/snapshots reproduce. A reviewed continuation must
retain the failure and count oversized unsupported sources in the fixed sample.
[Evidence](filing-source-review-2026-10-24/original-flow-pilot-summary.json).


Cross-era original-flow pilot complete, **failed**:210/319 ready(65.83%) versus80%
overall; eras69/104(66.35%),66/107(61.68%),75/108(69.44%) all miss70%.297 inspected
include67missingflow/19conflict/1prior failure;19oversized+3unsupported stay in
fixed denominator.300 complete originals/1.44GB retained;210 ratios independently
verified, all297 snapshots reproduce. Initial stop preserved; collector exited0,
locks released/no task temporaries.188 checks/Ruff pass. No bulk financial data
acceptance, training, threshold loosening or new Colab units. A distinct critic
calibration review is the next candidate, not a promoted model. [Evidence](filing-source-review-2026-10-24/original-flow-pilot-reviewed-summary.json).

[Saved SAC soft-consistency](sac-soft-consistency-2026-10-24/README.md):64 fixed flat
training-reset states/256 frozen stochastic trajectories, meanQ−soft-target3.796906,
lower-two-SE3.625062 and79.69% full512 trajectories pass the fixed descriptive flag.
Not trading returns or unbiased true-Q calibration; learned bootstrap and reset-only
scope retained. All states/sources unchanged,199 combined focused checks pass,
worker exited/environment closed. Supports preregistering a distinct critic study;
no TQC training, promotion, retired-test evaluation or new Colab units.

[Distributional-critic initialization](distributional-critics-2026-10-24/README.md):
12 pinned untrained SAC/TQC artifacts, nativeSAC globalTorch/NumPy/Python sampling
RNG, exact actor/TQC-control tensor checks and61 passing focused tests. No local
optimization, trained TQC return or Colab units. Runtime/archive/controller
integration and actual smokes remain next; existing failed families stay closed.


Distributional-critic version33 integration ready: training/task/archive/fleet
contracts, actual saved-model/replay restoration and persistent shared two-unit
study baseline implemented.12 checksum-identical initializers now ship in tracked
configs.115 focused checks pass without local optimization; actual Colab smokes
and all new performance gates remain pending. No new return or promoted model.
See[study guide](../docs/distributional-critics.md).


Distributional-critic actual Colab smokes accepted:3×1024 CPU transitions, actual
Python3.13.16, initial tensor/RNG and learned Adam/replay restoration,192 exact
frozen predictions and immutable normalization. All3owned sessions stopped.
Conservative project39.039379/150, incremental0.025001, official account349.00
with zero assignments. Fixed screen runtime projection3717.83<7200seconds;
50k seed11 three-arm plans prepared. No completed screen, return improvement,
retired-test access or promotion. Evidence:results/distributional-critics-2026-10-24/smokes.json.


[Saved critic final-update sensitivity](critic-update-2026-10-24/README.md) complete: three one-update pairs on
256 fixed candidate training-replay observations; truncatedTQC actor relativeL2
0.03356%, exposure mean absolute delta0.001462, GaussianKL0.001133 and zero
exposure-sign flips. These are normalized action/tensor diagnostics, not returns
or a cause of development deterioration. Independent Torch arithmetic agrees;
all protected files/model/Adam state unchanged, worker exited0, no local
optimization or Colab units. First phase-filename rejection and reviewed v2
correction retained; sample/metrics unchanged. Fixed failed family remains
closed. Next bounded saved-development-path audit before another training
hypothesis; no promotion or retired-test access.


[Saved TQC path audit](critic-paths-2026-10-24/README.md) complete: all26 selected/final development paths
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


[Implied-volatility source feasibility](vix-source-2026-10-25/README.md) passed: single date-bounded Yahoo
request retained3,774 observations2009–2023; lagged level/change20 context covers
all3,433 original-dataset decisions (100% overall/eras/years), with exact prior
sessions and assumed availability no later than decisions. Four tests and
independent pandas/source/dataset/normalizer verification pass; worker exited0.
Historical publication vintages unverified; source values remain local and model
inputs disabled. No training, new return or Colab units. Next audit the accepted
111-symbol consumer and preregister a separate matched feature comparison only
after coverage; failed families and retired-test isolation remain unchanged.


[VIX information comparison preregistered](vix-context-2026-10-25/README.md) after expanded-consumer coverage
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
