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

## Remote smoke job registration — 2026-11-01

Job version44 binds each role to the reviewed type-manifest digest, original
111-instrument dataset, CPU, seed11 and1024 transitions. Remote dispatch verifies
both payload identities before calling the learned smoke worker, and records the
completion/source/kernel receipt. Canonical plan validation rejects changed
hardware, steps, seed, smoke mode or metadata identity. It has no full-study route.

Allocation remains explicitly rejected by fleet task validation until the shared
incremental-budget claim and physical artifact verifier are integrated. This is an
implementation staging guard, not a request for user permission. No task smoke has
been allocated or run. Existing fleet budget and cleanup behavior remains in force.

## Shared task-study budget controller — 2026-11-01

`training/task_runtime.py` retains one protocol-bound baseline against cumulative
project usage, including smoke/setup/retrieval and later study attempts. The entire
family allowance is at most two additional units; callers cannot reset the baseline
or clear an existing budget stop. Account floor150/teardown170 remains required.
Each role has a durable, non-repeating CPU allocation claim with a7200-second
allocation deadline and360-second retrieval/shutdown reserve. No automatic role
retry follows a failed allocation or worker.

Fleet wiring requires the complete three-role set, one aggregate study allowance,
clean committed source and projected allocation deadlines; existing aggregate
metering and owned-session teardown apply. Tests cover persistent charges, allowance
exhaustion, attempted reset, invalid allowance, role claims and existing protections.
The pre-allocation staging guard remains until physical archive acceptance is
implemented. These tests use temporary ledgers; the project ledger was not changed.

## Downloaded smoke state verification — 2026-11-01

The version44 archive verifier allowlists checkpoint/replay/report and frozen
bundle files, bounds expanded size to256MiB and member count to40, and rejects
missing, duplicate, linked or unsafe entries before extraction. It binds completion
to the CPU/default-kernel plan, dataset and clean source receipt. Physical inspection
loads the initial/512/1024 SAC states, checks actual Adam ownership/moments/counters,
finite parameters and changed final weights, verifies1024 replay transitions and
type encoding, and compares the final policy to the checksummed frozen model and
its predictions across the reviewed registry. A remote success flag cannot replace
these checks. Only trusted task-owned downloads are candidates for SB3 loading.

Rejection tests include an untrained model paired with a fabricated successful
run record, incomplete downloads, incorrect dataset, links, traversal and duplicate
members. Learned positive-path acceptance still requires actual Colab output;
local training is prohibited. Allocation remains staged off pending final controller/transport review; the first
owned smoke must supply learned positive-path acceptance before research training.

## Smoke launch readiness — 2026-11-01

Version44 fleet admission now verifies the immutable dataset and reviewed type
manifest before allocation. The complete three-role fleet uses the shared two-unit
baseline, persistent per-role claims, CPU defaults, bounded downloads and verified
owned-session shutdown. The temporary allocation staging rejection is removed.
First learned artifacts must pass physical inspection before any100k experiment.
Actual kernel Python is checked by bootstrap before dependencies or optimization;
a default-version change stops the worker rather than installing an older Python.

## Training-only checkpoint selector

`training/task_selection.py` requires the complete ordered10k-through100k training
curve and identical holding-path identities. It selects the earliest strict maximum
mean net return and keeps the final100k point separately. Missing/duplicate steps,
development scores, nonfinite scores or changed holding paths reject selection.
It returns independent copies and never grants development access: the complete
three-role final training gate remains mandatory. This helper does not implement
or authorize the full100k worker; learned smoke acceptance and profiling come first.

## First learned smoke and retained-archive correction — 2026-11-01

The masked shared role completed1024 Colab CPU transitions with960 actor/critic/
entropy updates. Its model/replay restoration and learned frozen predictions pass
independent inspection of the downloaded archive. The original fleet stopped after
archive acceptance rejected the report writer's `final-training.json` sidecar: the
allowlist omitted this expected file. The verifier now requires and allows that
specific sidecar; a regression covers its presence and omission. No model settings,
performance thresholds or numerical tolerances changed, and no training was retried.

`masked-smoke-acceptance.json` retains archive SHA, original failed fleet identity,
source revision, compact training curve and corrected physical acceptance. Original
failure state, per-role allocation claim and shared budget baseline stay intact.
Owned-session cleanup was verified. Tagged shared/routed roles were never allocated;
a reviewed continuation must account for the accepted original role and retain its
charges rather than restarting the complete fleet. These smoke returns do not
admit performance or the full100k screen.

## Reviewed unallocated suffix

The continuation admits exactly tagged shared and tagged routed after rechecking
the pinned masked-role acceptance receipt and archive, physical saved state,
original stopped ownership, original failed fleet state and allocation claim.
It rejects any already-claimed suffix role and keeps the original two-unit study
baseline. It does not retry masked training or overwrite the failed fleet record.
A separate fleet state records the two never-allocated roles; each still receives
one allocation with ordinary shared metering, deadlines and verified cleanup.

Training-window learning charts now derive their title from their recorded phase;
they no longer inherit the generic validation label. Ordinary validation curves
keep their existing labels; unknown or mixed phases say Evaluation. Previously
retrieved immutable smoke reports retain the old display label, while their JSON
phase remains the authoritative training-only scope. No returns or checkpoint
selection changed. Future worker payloads include the corrected chart renderer.

## Tagged shared smoke and controller result contract

The tagged shared role completed1024 transitions and960 updates per optimizer;
training-window return changed from−31.61% at512 to−7.30% at1024. Physical
saved-state verification passed, but fleet acceptance recording raised KeyError
because the verifier summary omitted the controller-required `bundle_id`. The
summary now returns the identity of the physically inspected frozen bundle; a
regression checks that field. The existing masked review remains valid with this
additional descriptive field, while every original acceptance field is rechecked.

`tagged-shared-smoke-acceptance.json` records corrected physical acceptance of the
same retained archive. No training was retried or numerical/performance checks
relaxed. Original failed fleet state and claims stay intact; cleanup was verified.
The routed role remains unallocated and requires its own reviewed continuation.

The final routed-only continuation rechecks both pinned accepted-prefix receipts,
archive checksums, physical saved state, stopped ledger ownership and preserved
failed fleet states. Requested roles must form the exact remaining ordered suffix
and have no allocation claims. Even a complete initial matrix now rejects an
already-claimed role before allocation. No accepted smoke is repeated; all study
charges and the original baseline remain in force.

## Bounded research-geometry profile worker

`training/task_profile_worker.py` runs4096 transitions/3096 updates for one fixed
role using research batch256, warmup1000 and512-session training windows, seed11.
It times learning separately from one original26 training-window evaluation,
checks unchanged inference state, exports initial/final models durably and records
actual optimizer counts, sampling and head activity. No performance scores or
checkpoint selection are exported. It rejects local execution before model/output
creation; no local optimization is used for verification.

This worker is implemented but unrun. Remote registration, complete smoke
acceptance, physical profile verification and shared-budget allocation admission
must precede profiling. Measured timings must determine a bounded100k execution
plan including setup/evaluation/retrieval/cleanup; smoke wall time alone cannot
justify a research duration. The two-unit total family allowance is unchanged.

The version45 remote profile contract runs the three fixed roles sequentially in
one CPU allocation,4096 transitions each. It pins the same dataset/type evidence
and seed11, retains a two-hour allocation ceiling and360-second shutdown reserve,
and exports completion/source/kernel metadata after all roles finish. Canonical
validation rejects omitted roles, hardware changes or extended deadlines. This
registration does not enable allocation: complete learned smoke acceptance and a
physical downloaded-profile verifier are still required. The profile shares the
existing task-study allowance and cannot start a new budget baseline.

The downloaded version45 profile verifier requires all three roles' initial/final
models and input/profile records within a64MiB/24-member safe archive. It verifies
CPU/default-kernel clean provenance, actual research batch/warmup/replay capacity,
4096 transitions/3096 Adam steps, finite changed weights, finite positive learning/
evaluation/inclusive timings and identical fitted holding paths. Its fleet result
includes the required `bundle_id` (the profile-record identity; no frozen prediction
bundle is produced by profiling). Untrained success claims and invalid timings are
rejected without local optimization. Actual learned positive-path profile acceptance
remains unrun; complete smoke admission and budget integration are still required.

Profile fleet admission now independently rechecks both reviewed prefix archives
and the routed role's successful fleet/bundle identity, stopped ledger ownership
and cleanup before permitting one dedicated sequential CPU allocation. A durable
profile claim forbids automatic retries. It retains the original task-study
baseline and aggregate account/project protection, uses bounded artifact downloads
and disables execution retry. Missing immutable inputs or incomplete smoke evidence
reject admission before allocation. Actual profile execution remains pending the
routed smoke's successful completion and cleanup.

## Complete learned smoke acceptance — 2026-11-01

All three roles completed1024 transitions,960 updates per optimizer, physical
checkpoint/replay restoration and exact frozen prediction parity across111 symbols.
The routed smoke confirms actual stock and ETF actor/critic head gradients. Final
training-window returns are masked+2.23%, tagged shared−7.30%, routed−2.08%; these
short fitted runs are engineering acceptance, not evidence selecting a winner or
opening development. The combined `complete-smoke-acceptance.json` retains each
archive checksum, source, curve, head activity and acceptance scope.

All three owned sessions are stopped; official usage reports zero assignments and
548.25 account units. Recorded cumulative project usage is39.960425 units, including
all setup/failures/retrieval. Shared task-study consumption is0.034566 of2 units.
The original failed fleet records remain retained. Research-geometry profiling is
now admitted;100k training and development access remain gated.

## Predeclared timing projection

`training/task_projection.py` scales each role's complete4096-step learning time
by99000/3096 research updates and includes ten measured training-window evaluations,
then applies a1.5 margin. Add900 seconds setup and600 seconds retrieval/cleanup per
allocation. Scaling the whole learning interval also recharges warmup/environment
work conservatively. Each projected allocation must fit7200 seconds. Sequential
aggregate cost uses at least the observed profile rate plus the ledger's0.01/hr
margin, and must fit the original remaining two-unit allowance. A positive reliable
rate is required even for CPU. Any failure rejects execution; no duration/allowance
is enlarged automatically. This arithmetic is not physical profile acceptance or
performance evidence, and has not admitted any100k run yet.

## Fixed100k training cell implementation

`training/task_screen_worker.py` implements an unregistered seed11 cell using the
unchanged research SAC/sampler/simulator. It saves every10k checkpoint after the
actual optimizer update, durably exports weights before replay, retains full
training-window replay JSON and compact incremental curves, and selects only from
the complete ordered training curve. Each checkpoint evaluation must leave policy,
optimizer and entropy state unchanged. All111 training symbols require measured
starts/transitions; routed actor/critic heads require actual nonzero gradients.

It retains99000 optimizer updates, final replay and exact saved-state restoration,
selected/final training reports and separate frozen bundles, plus a final training
cell for the three-role fitted admission gate. No development/test evaluation or
promotion is performed. Local execution rejects before creating outputs. Relevant
unit checks cover the guard, selection and existing optimizer/gate machinery; actual
100k positive-path execution remains unrun. Remote allocation/physical full-run
acceptance and measured profile binding remain to implement before any screen run.

## Independent full-cell replay verification

`training/task_screen_verify.py` loads all ten learned checkpoints, verifies their
actual Adam counts/moments and research geometry, and independently reruns the
fixed original26 training-window evaluation. Retained full results, compact curves,
matched holding identities, earliest-max selection and final gate cell must match
exactly. It also checks complete replay,111-symbol coverage, selected/final frozen
weights and111-symbol prediction parity. No local optimizer call is made.

The unit rejection fixture deliberately pairs a complete success/selection record
with untrained model files and confirms rejection from saved optimizer counters.
Actual100k positive-path acceptance remains unrun. This verifier can take several
minutes and must run asynchronously under the fleet's continuing minute metering;
remote transport/async registration remain pending before allocation admission.

## Full-cell transport contract

Version46 fixes seed11/CPU/100k and requires a profile-acceptance identity; canonical
validation rejects extended steps/deadlines. Fleet allocation is still explicitly
blocked pending measured profile binding and async verifier wiring. The transport
extractor requires all ten models and full fitted replays, final replay, curve,
selected/final reports with JSON sidecars and both frozen bundles. It rejects extra,
missing, duplicate, unsafe/linked members and mismatched completion provenance
before extraction, bounded to64 members and1GiB expanded. The higher byte bound
accounts for the measured approximately51MB full26-symbol JSON replay per checkpoint
and retained selected/final copies. Transport acceptance does not accept performance;
independent checkpoint replay remains mandatory. No100k job has run.

## Asynchronous full-cell verification

The full-cell verifier has a separate subprocess entry that disables learn/train,
backward and Adam step locally, restores APIs on interruption, extracts only checked
transport, independently replays saved models and publishes proof only on success.
Fleet routing permits one local verifier at a time while minute metering continues.
The subprocess is bounded to1800 seconds and starts only when budget/deadline checks
cover that duration plus360 seconds shutdown reserve. Full allocation remains off
until measured profile binding is implemented.

Timing projections now explicitly include independent verification:1.5 times ten
measured evaluation durations plus600 seconds loading/I/O allowance. This must fit
the1800-second verifier limit and is added to the already declared remote compute,
setup and retrieval/cleanup allowances before the7200-second allocation check.
This corrects omitted verification work before any100k allocation is admitted;
no learning setting, performance threshold or credit allowance is changed.

## Profile compute-source binding

`training/task_profile_sources.py` compares current policy/SAC configuration and
factory, simulator/accounting/sampling, features, dataset/validation/type handling,
evaluation, task environment/optimizer helpers and dependency lock against the exact
profile Git revision. Changed content, added/removed compute files or source symlinks
reject reuse. It returns SHA256 evidence for the matched files. Actual comparison
against the running profile's captured clean revision matches103 files. New screen
orchestration/verifier code is outside that compute inventory and requires its own
checks; this does not claim it was timed by the earlier profiling worker. The helper
is implemented for admission binding; full-screen fleet admission remains disabled.

## Accepted research profile and failed initial duration admission — 2026-11-01

All roles pass physical4096-transition/3096-update profiling and103-file compute
source binding; owned shutdown is verified. Learning/evaluation seconds are masked
71.561/41.666, tagged shared75.518/42.770 and routed96.716/38.779. The retained
`research-profile-acceptance.json` includes exact archives, source evidence, actual
records, aggregate rate and original study charges.

The conservative full-allocation projections are6783/7006/7903 seconds. Routed
exceeds the currently registered7200-second limit, so full-screen admission fails;
that failed result is retained and no100k job was allocated. All verification
projections fit1800 seconds. Projected study execution cost0.5423 units fits the
unchanged remaining allowance. This is a duration failure, not a performance result.
The next action is an explicit execution-only review of a measured subsequent-run
duration; the two-hour initial smoke/profile requirement was met. No learning,
selection, performance gate, study baseline or project/account limit may change as
part of that review, and the original failed projection must remain preserved.

Official post-cleanup account548.23/rate0/no assignments; cumulative project
39.976410/150, task study0.050551/2 units. No development or retired-test scoring.

## Explicit subsequent-screen execution review

`configs/experiments/task_screen_execution_review.json` pins the accepted profile
receipt and preserves its failed initial two-hour projection. The initial smoke and
profile allocations already met the user-required two-hour initial deadline. This
separate reviewed subsequent-run contract permits7200 seconds for each shared role
and9000 seconds for routed, covering its7903-second conservative projection. It
changes no training steps, model/optimizer, selection, performance gate, study
baseline, credit ceiling or account floor. No automatic runtime extension occurs.

`training/task_execution.py` physically rechecks all stopped smokes/profile, exact
receipts/archives and103 compute dependencies, recomputes original projections and
checks the complete reviewed durations against remaining shared allowance. The
full allocation ceilings cost0.585 units at the observed rate plus ledger margin,
within the original two-unit family allowance. Per-role persistent claims prohibit
retries. Actual production admission passed these checks without allocation or
ledger reset. Remote registration/fleet wiring for this reviewed contract remains
pending; the original version46 plan stays blocked and unchanged.

## Reviewed screen launch wiring

Version47 implements the explicit execution review with fixed7200/7200/9000-second
role allocations, original seed11/100k methods and profile identity. The complete
ordered matrix is required; a dedicated aggregate fleet supports the9000-second
maximum and records each role's one-time claim. There is no automatic execution
retry. The asynchronous verifier retains1800 seconds plus shutdown reserve, and
all roles still share the original two-unit baseline and account/project protections.
Version46's original two-hour contract stays blocked and preserved.

From clean committed source, prepare without allocating:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_task_screen_jobs.py --output .runtime/task-screen-jobs-v1.json
```

Run through `python -m rl_stocks.runtime.colab.fleet` with the prepared jobs, existing
CLI interpreter and a fresh state-output. The builder and fleet each recheck physical
profile/source evidence before allocation. Remote dispatch verifies input identities,
executes the fixed training-only cell and exports completion metadata; downloaded
cells undergo independent saved-checkpoint replay. No development evaluation is
performed in this stage, regardless of training returns.

## Masked100k completion and exact CPU replay diagnosis — 2026-11-01

Masked shared completed100k transitions/99000 updates. Original fleet stopped at
local validation because host-default replay differed from retained results; owned
shutdown was verified, and no other screen role was allocated. At10k, maximum raw
action difference was2.384e-7 and equity difference$0.001912; no tolerance was widened.
Matching subprocess `ATEN_CPU_CAPABILITY=avx2` and `MKL_CBWR=AVX2` restored exact first
predictions for26/26 symbols and then exact independent replay for all ten saved
checkpoints, including curves, selection, final cell and learned frozen parity.
No model was retrained or optimized locally. This supports an arithmetic-mode cause;
it is not a claim of bit portability across all CPU backends.

Future local full-cell verifier children explicitly set this arithmetic mode before
Torch import, require actual AVX2 capability and retain single-thread execution.
The parent environment and remote training remain unchanged. Original failed fleet
and default-mode diagnosis stay retained. `masked-screen-acceptance.json` pins the
same archive SHA, exact proof and diagnostic evidence. Selected50k training-window
return40.52%; final100k27.61%. These fitted returns do not open development: both
unallocated roles and the complete three-role training admission gate are pending.
Project40.065123/150; official account548.15/rate0/no assignments after cleanup.

## Reviewed remaining100k roles after exact masked acceptance

The builder accepts `--resume-unallocated` only for tagged shared/routed after
checking the pinned masked exact-replay receipt, original archive SHA, retained
failed fleet/claim and stopped ownership. A claimed requested role rejects before
allocation. The fleet supports this exact suffix and keeps the original baseline,
measured profile/source checks, role durations and per-role claims. It does not
retry masked training or overwrite original failure state. Current production
admission passed with0.405 projected units and1.860736 study units remaining.
Use a fresh jobs output and fleet state file for the suffix; development stays shut
until the complete three-role final fitted gate passes.

## Complete verified final training gate publication

After the suffix fleet completes and all owned sessions are stopped, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/accept_task_screen.py --output results/task-conditioning-2026-10-31/training-admission.json
```

The reader binds the pinned masked exact proof and both suffix verification files
to retained archives, final training-cell metrics, selected steps and frozen bundle
identities. It requires complete ten-checkpoint replay proofs and stopped verified
ownership before calling the original three-role training gate. This inexpensive
binding consumes independent replay evidence; it does not replace model replay.

The decision is written once. Failure also publishes a persistent family closure;
shared task-budget admission then rejects further allocations without altering
cumulative usage or global budget-stop state. Development remains forbidden on
failure. Unit checks reject modified metrics and incomplete proofs and verify
closure enforcement. This reader has not yet consumed a complete actual matrix;
tagged shared/routed100k outcomes remain pending. All103 profiled compute files
remain unchanged by these orchestration additions.

## Accepted masked training-only directional diagnosis

`masked-training-directional.json` derives side-specific contributions directly
from the checksum-pinned, exactly replayed reports, without fitting or checkpoint
changes. Long+short P&L reconciles to net equity for all26 symbols at selected/final.
Selected50k mean return40.52% comprises+43.35pp long and−2.82pp short; final100k
27.61% comprises+43.29pp long and−15.68pp short. Matched holding is56.45% over
this fitted window; it is not the30.60% development baseline. Contributions are
cash P&L divided by initial equity, not separately compounded portfolio returns.

Only two stocks have negative full-window holding returns. Selected has one with
positive total and positive short contribution; final has none with positive total,
although one retains positive short P&L. These are reporting-only ex-post cohorts,
never observations/sampling/rewards/checkpoint criteria. The model therefore does
not satisfy the requested declining-stock outcome in this control. This diagnosis
cannot choose or modify the unfinished tagged comparison; its fixed gate remains.

## Tagged100k exact replay failure and fixed-mode limitation

Tagged shared completed100k, selected50k return50.27%, final46.03% on the fitted
window, but remains unaccepted: independent exact replay failed even with AVX2.
The controller salvaged the full archive and verified shutdown; routed was never
allocated. `tagged-screen-numerical-diagnosis.json` pins archive/checkpoint hashes,
original failed state and nine bounded first-decision probes. Host/default, AVX2
and baseline Torch dispatch crossed with default/AVX2/COMPATIBLE MKL modes yield
14–18 exact symbols out of26, with maximum action difference1.192e-7. No mode
reproduces all first decisions. No performance outcome selected a backend.

The masked AVX2 exact proof remains valid, but the assumption that one fixed mode
would reproduce every Colab session is contradicted. No tolerance, gate or learning
configuration was changed; no training was repeated and the suffix was not
restarted. Next is a retained-artifact numerical execution investigation. Future
runtime provenance must describe the relevant arithmetic/backend rather than imply
universal CPU bit portability. Current project40.121163/150; official post-cleanup
account548.11/rate0/no assignments. Development and routed allocation remain shut.
