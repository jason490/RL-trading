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
