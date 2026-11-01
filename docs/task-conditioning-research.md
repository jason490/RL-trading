# Stock/ETF task conditioning: research decision

Reviewed 2026-10-31. This is a design decision and implementation brief, not an
accepted runtime profile or permission to reopen the completed recurrent screen.
No new training, allocation, predictive result or performance claim accompanies it.

## Evidence and hypothesis

The [immutable fitted replay](../results/recurrent-offpolicy-fit-2026-10-31/README.md)
finds final SAC stock return62.78% versus58.15% holding, but ETF return29.35%
versus47.15%. GRU-SAC shows the opposite fitted grouping:41.88% stocks and61.28%
ETFs. Both fail the complete group/median requirements. Different architectures
and fitted paths do not isolate task interference, and the repeatedly inspected
2021–2023 development panel cannot provide fresh independent justification.

A distinct hypothesis is that the shared feed-forward SAC mapping benefits from
explicit instrument-type context, and that group-specific output heads provide
additional benefit beyond that information. A stock/ETF tag denotes the legal
instrument type known before the historical decision, not a retrospective return
class. It supplies no ticker embedding, sector snapshot, future loss label or
new market indicator. Classification provenance must still be verified and
unknown instruments rejected rather than silently called stocks.

## Primary sources and limits

[Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS2020](https://arxiv.org/abs/2001.06782v4)
identifies conflicting task gradients and proposes projecting conflicting
components. It motivates measuring interference, but our return gaps do not
establish its conditions. We defer PCGrad: it adds optimizer changes and would
require group-tagged replay, per-task losses and a separate controlled hypothesis.
Its robotics/supervised results do not establish financial predictability.

[Hessel et al., Multi-task Deep Reinforcement Learning with PopArt](https://arxiv.org/abs/1809.04474)
adapts task contributions to address differing reward scales in game domains.
Our common net-log-equity reward does not prove equal empirical scale, but we
have not measured a scale imbalance warranting target normalization. Defer
PopArt rather than changing reward/value normalization at the same time as routing.

[Yu et al., Meta-World](https://arxiv.org/abs/1910.10897v2)
reports difficulty learning multiple robotic tasks together despite single-task
learnability. Its task definitions differ from securities sharing one execution
MDP. Stock and ETF grouping is an engineering hypothesis, not a replication or
proof that each asset is a separate mathematical task.

## Existing work and distinction

Repository review covered `sac-diversity.md`, `policy-architecture.md`, the
supervised signal protocol and the completed recurrent comparison. Diversity
changed the training pool while preserving stock/ETF sampling mass; it did not
condition policy/critic outputs on type. The PPO branch experiment separated
market versus account information, rather than instrument groups. Earlier MLP
supervision used gross20-session returns without actual protective execution.
Those closed families stay closed. Group routing changes the SAC function class;
it must use fresh controls and cannot inherit a passing historical result.

## Fixed comparison to implement next

Use three fresh SAC roles, unchanged64x64 trunks, four continuous actions,
standard SB3 update and entropy recipe, same replay/start schedule and equal
100k environment transitions. The proposed roles are:

| Role | Type input | Actor and twin critic outputs |
| --- | --- | --- |
| masked shared | Fixed zero in appended slot | Shared heads |
| tagged shared | Stock=-1, ETF=+1 | Shared heads |
| tagged routed | Same verified tag | Two group-specific heads |

All observations have22 slots: existing full21 plus the explicit type field.
Keep the original seven-feature normalizer byte-identical; append the tag after
normalization. Preserve all14 account fields and feature order. Replay retains
the tag on current/next observations, targets use the matching group, and frozen
inference validates both schema and instrument type. A type change within an
episode is invalid. Never infer type from price, return or symbol string shape.

Match tagged and masked shared initial tensors exactly. For the routed role,
clone each shared output head into both groups; shared trunk and targets match.
Initial outputs must agree for both tags when the shared input slot is masked.
The tagged-versus-routed initial parity uses identical tag inputs. Parameter
count and active capacity differ and must be reported: this is a routing package
comparison, not an equal-capacity causal estimate. Do not compensate by adding
width or optimizer changes. Keep a single automatic entropy coefficient to avoid
silently adding task-specific reward/entropy adaptation.

Before implementation freeze, resolve the precise actor/critic layer boundary
and publish effective settings, initialization/RNG checks, parameter counts and
artifact schema in a machine-readable preregistration. This document alone is
not an executable protocol. Do not allocate before those checks exist.

## Admission and stop rules

First require offline no-gradient tests for routing, replay tags, mixed batches,
unknown/change rejection, target copies, serialization and deterministic frozen
parity. Local optimization remains forbidden. Use an actual Colab CPU smoke to
verify nonzero updates in both groups, gradients, optimizer restoration and
measured deadline including publication/download. No automatic retry after stop.

A training-only fixed-window comparison precedes any new development replay.
Require routed stock/ETF mean and median-stock excess versus holding each>=-2pp,
and routed mean gain>=2pp versus each fresh control. A failure closes this
hypothesis without viewing development scores. These training-fit thresholds
are feasibility criteria, not evidence of generalization. Select no checkpoint
on development during this admission phase; use fixed final100k states.

If admitted, freeze the development selection/evaluation schedule before reading
2021–2023 results. Require selected AND final gain>=2pp versus each control,
stock/ETF/median excess>=-2pp, and final drawdown worsening<=2pp. Confirmation
requires all three roles at seeds23/37, at least two of three paired seed passes,
mean paired gain>=2pp in both phases, and the same absolute group/median limits.
Only nonnegative group excess permits an outperformance description. No sweep,
extra steps, retrospective checkpoint choice or relabeling after failure.
A later frozen prospective comparison remains necessary; retired2024+ stays shut.

Use cached cdb93a33 data and b3c8a298 normalizer,111 training symbols through2020
and original26 development symbols, unchanged sampling and simulator costs,
net-equity reward, whole shares,100% cap,20-session holding and next-session
execution. Declining-stock labels remain reporting-only; reconcile short P&L.

All stages share a proposed2-unit incremental allowance, recorded once against
the original ledger before any allocation. Preserve original150/130 project and
150/170 account safeguards. Smokes/profile must establish whether each CPU job
fits a two-hour allocation; do not infer admission from previous SAC timing or
silently opt into longer allocations. Retrieve/checksum artifacts before verified
owned shutdown, preserve failures and remove only verified duplicate staging.

## Next action and status

Implement and test the versioned type/routing contract and machine-readable
protocol, then integrate it with existing metered Colab workers. No dependency
change, new source data, profile approval, ledger reset or runtime is required
for that engineering work. The overall performance goal remains unachieved.

## Implemented policy and observation foundation

`rl/sac/task_policy.py` now provides `TaskSACPolicy(task_role=...)` for the three
roles. Routing occurs only in the actor mean/log-standard-deviation linear heads
and the final scalar head of each critic; both64-layer trunks still see the tag.
Each second head is a deepcopy without RNG consumption. SB3 creates the actor
and critic optimizers after those heads exist and copies both critic groups into
the target. Distribution/log-probability, losses, target updates and automatic
entropy remain SB3 responsibilities. This class is opt-in; the ordinary factory
and previous saved policies are unchanged.

`environment/observations/task.py` provides `InstrumentTypeObservation`. It
appends the tag after the underlying21-value observation, using copied immutable
records with symbol, stock/ETF kind, source and known-from date. Every episode
must have classification available by its first session. Unknown/duplicate
symbols, missing provenance, malformed input and within-episode symbol changes
reject. A provided source string is an evidence reference, not independent proof:
production provenance still requires an audited classification manifest. The
wrapper never fetches classifications or silently infers missing types.

No-grad tests cover shared/routed initial outputs, identical RNG continuation,
masked/tagged initial tensors, both target critics, mixed-batch actor/critic
routing, bad replay tags, optimizer parameter ownership, standalone policy
save/load and wrapper provenance/immutability. These are engineering checks,
not actual replay-collection, gradient-update or frozen-bundle acceptance.
Production manifest, machine-readable protocol, factory/frozen-bundle integration,
Colab workers and actual optimizer smoke remain pending. Do not train this
foundation by bypassing those gates.

## Fixed protocol, factory and cached audit

The [machine-readable protocol](../configs/experiments/task_conditioning.json)
is now pinned by canonical SHA256
`4feb50d28de5a07d851573ba2bf4649cf2941d65d41db2ed3afc06ca5be163d8`.
It fixes checkpoint selection to the earliest strict maximum on the original26
training-window return curve, evaluated every10k; selected/final development
scores are read only after the final training gate passes. This avoids introducing
another checkpoint search on repeatedly inspected development years. Smokes retain
64x64 geometry and use batch64/warmup64; screens use batch256/warmup1000.

`rl/sac/task_factory.py:create_task_sac` is opt-in and validates wrapper/policy
role agreement. Ordinary algorithm construction remains unchanged. Full untrained
SAC model save/load and explicit replay-tag retention pass without optimizer calls.
`training/task_conditioning.py` pins protocol bytes and rejects unreviewed,
wrong-universe, snapshot-only or late historical type evidence at admission.
A source reference and reviewed flag require actual review; structural validation
cannot independently establish the truth of an external historical document.

The [cached audit](../results/task-conditioning-2026-10-31/README.md) confirms
107 stocks/four ETFs and immutable data/normalizer/ledger bytes, but historical
metadata is **not admitted**. Do not turn first observed training dates into
classification availability dates. Complete primary-source historical evidence,
frozen-bundle/worker integration and actual Colab optimizer/profile acceptance
before any performance screen. No new credits were used by this audit.

## Frozen task inference foundation

`inference/task_bundle.py` adds opt-in `freeze_task` and `FrozenTaskPolicy`.
The standard generic bundle format retains model/dependency checksums; task
metadata additionally binds the full reviewed type manifest, its digest, dataset
identity, role and type encoding. Freezing requires historical metadata admission.
Loading checks file/manifest/model identities, feed, feature/action schemas,
seven-feature normalization, full execution state, policy class and role agreement
across actor/twin/target critics. It uses the same SAC model and execution filter.

Prediction takes the original finite21-vector plus explicit symbol/session. The
loader generates its own tag from frozen reviewed records and rejects unknown
symbols or dates before availability. Supplying a22-vector cannot override the
tag. Calls are deterministic/no-grad, manifest views are copies, and the registry
and normalizer are immutable. No broker decision path is introduced. The ordinary
`FrozenPolicy` API remains unchanged; task bundles use this dedicated API.

All three roles have actual untrained bundle prediction parity and immutable
state tests. Corruption, bad feed, forged input tags and relabeled metadata even
with recomputed checksums reject. Existing generic frozen, observation and
execution checks remain compatible. Actual learned checkpoints/Colab optimizer
acceptance and worker integration remain pending.

The [source investigation](../results/task-conditioning-2026-10-31/README.md)
finds earlier cached candidate filings for97/107 stocks. Those current-CIK links
do not establish historical security continuity/type, and the ten absent matches
are not proof of absent filings. Historical metadata remains unadmitted; no
classification dates or causal readiness are inferred from archive coverage.

## Environment and fitted evaluation integration

`training/task_environment.py:build_task_environment` now validates the reviewed
manifest, preserves the frozen normalizer and constructs training-only full-state
episodes with the existing stock/ETF cycle sampler. Research windows remain512
transitions; smokes use128. Reward, execution and simulation defaults stay fixed.
The explicit wrapper adds type after preprocessing. Callers cannot enable a
retrospective metadata fallback, change the sampling rule or supply a test split.
The eventual worker must additionally enforce pinned dataset/protocol identities.

`evaluation/task_conditioning.py:evaluate_task_training` revalidates evidence,
crops the unchanged fixed original-panel training window and delegates each
episode to the existing accounting evaluator. A separate episode predictor adds
the known type at each decision, validates finite full21 observations and rejects
unavailable types or exhausted streams. Raw four-action requests remain in the
report before the existing unit-magnitude evaluation filter. There is no new
holding baseline, cost formula or simulator. Public validation/test evaluation
and generic prediction streams are unchanged; this task API opens only training.

Synthetic no-optimization integration tests exercise mixed stock/ETF tags,
identical seeded group/start sampling across all roles, full episode replay,
immutable source data and Gymnasium/SB3 checkers. These fixtures include deliberate
small accounting windows and are not market coverage or performance evidence.
Actual Colab optimization, checkpoint selection/gates and job/controller/archive
integration remain pending, along with historical metadata admission.

## Admission gate and Colab-only smoke worker

`evaluation/task_gates.py:training_gate` implements the fixed final100k seed11
training-fit comparison. It requires all three roles, the original22-stock/four-
ETF panel, finite metrics, matching holding-path digests and the pinned protocol.
Routed SAC must gain2pp versus each control and satisfy stock/ETF/median excess
floors of-2pp. Incomplete, selected-phase, development, wrong-seed or unmatched
inputs reject. A passing numeric gate permits the later development stage, not
performance promotion or objective completion. These summaries still require
independent physical artifact acceptance; a metric dictionary is not that proof.

`training/task_smoke_worker.py:execute_smoke_colab` implements the fixed1024-step,
seed11 CPU smoke for one role. It rejects outside default-Python3.13 Colab CPU
before outputs/models, then requires clean committed source, exact dataset/
normalization and reviewed historical evidence. It uses64x64 networks,128-session
sampled windows, batch64/warmup64 and post-update512/1024 checkpoints. The worker
never opens development or retired tests. It cannot run a100k screen.

`TaskOptimizerAudit` observes actual actor/critic/entropy Adam steps, finite
parameters/gradients, per-parameter moments and nonzero routed-head activity.
The expected smoke count is960 updates per optimizer; synthetic counters alone
cannot establish a real smoke. Checkpoints publish before fitted replay and
compact return/change curves, with final training HTML/JSON. Original source,
seed, role, type manifest, sampling and counters are retained. Learned state is
fingerprinted across inference. Replay/model save/load checks compare all
weights, optimizer states, raw alpha, counters and replay arrays/cursors; learned
frozen predictions must match for every declared symbol. This does not claim
exact continued-learning restoration of environment/RNG/sampler state.

Hooks/environments close in `finally`; failure evidence and the latest complete
artifact tree are published through the existing exporter. Tests verify local
rejection before construction, failure cleanup and real untrained save/load
corruption detection without optimizer execution. Actual smoke/profile results,
archive/controller acceptance and remote session cleanup remain unverified until
run through the existing centrally metered owned fleet. Historical evidence is
still unadmitted. No runtime was allocated for these engineering checks.

## Strict policy and loader boundary checks

Task policies now verify the effective native ReLU/flattened-input contract after
SB3 construction, including positional constructor arguments. A custom extractor,
shared feature extractor or alternate activation cannot silently pass the fixed
64x64 task study geometry. The ordinary `FrozenPolicy` loader explicitly rejects
`TaskSACPolicy`, even when an unrelated feature schema happens to produce22
inputs; task inference must use the symbol/date-aware loader. Untrained regression
tests cover both bypasses and preserve normal SAC/CrossQ frozen compatibility.
No optimizer execution or performance result is implied by these checks.

## Source coverage and nominal smoke sampling — 2026-11-01

Dated candidate sources now cover all111 instruments after the ABBV/DIS original
primary collection and review. Coarse-type/issuer-association sufficiency remains
unadmitted; complete coverage alone does not authorize training. The full original
price identity and survivorship limitations stay visible.

`audit_task_smoke_sampling.py` replays only the fixed sampler's eight nominal
128-transition starts from the original111-symbol payload, seed11. The
[actual read-only audit](../results/task-conditioning-2026-10-31/smoke-nominal-sampling.json)
starts with DIA followed by seven stock starts, so both groups appear without a
seed sweep or changed sampling. Data/preprocessing/ledger bytes remain unchanged.
No environment or learner is constructed. Nominal schedules do not verify actual
transitions, early terminations, replay batches or routed gradients; those remain
requirements of the real Colab smoke. No allocation or optimization occurred.


## Scoped initial type admission — 2026-11-01

The reviewed manifest `configs/universes/task_types_research_initial_v1.json`
covers111 instruments (107 stocks, four ETFs). Admission is conditional on the
existing dataset provider instrument association and a fixed coarse-type research
contract. It verifies initial stock/ETF facts; exact historical ticker intervals,
share classes, price identity, financial comparability and historical membership
remain unverified. This does not enable financial inputs or paper execution.

Initial type values come from sources available before the first training decision;
validation compares UTC availability timestamps, including same-session ordering.
Later Google/Linde corporate-event documents only corroborate stock-to-stock
continuity and supply no earlier model value. The additional2012 QQQ registration
still says QQQQ: it does not establish a ticker-change date. Same named fund and
CUSIP corroborate only the documented provider-association assumption. The IWM
bank contract names an ETF underlying and does not identify the bank as fund issuer.

The retained `coarse-type-admission-review.json` records source checksums and limits.
No training or Colab allocation accompanied admission. Next are real-data untrained
parity checks and guarded Colab smoke transport/archive acceptance, followed by
actual profiling. The fixed performance gates and closed prior families remain
unchanged.

## Real-data inference acceptance — 2026-11-01

`scripts/check_task_real_data.py --output artifacts/<fresh-name>` loads the pinned
111-instrument dataset and reviewed type manifest with optimization APIs disabled.
The retained `real-data-untrained-acceptance.json` records exact save/load frozen
prediction parity for all111 classifications in each of the three roles, using a
real normalized reset observation as a common input. It does not replay all symbols'
price histories or demonstrate learned performance. Policy state, preprocessing and
the dataset remain unchanged, and learned transitions are zero. Parameter counts
including target critics are29,964 for either shared role and30,744 for routed.

Actual execution passed on November1. Frozen bundles remain under
`artifacts/task-real-data-acceptance-v1`; this does not replace learned Colab smoke
acceptance. The guard test also verifies that a handled interruption restores the
original local APIs. Remote transport, archive verification and measured profiling
remain pending before the fixed100k comparison.
