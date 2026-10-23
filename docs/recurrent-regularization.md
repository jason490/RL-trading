# Fixed recurrent PPO regularization comparison

Status: distinct numeric protocol and actual untrained optimizer construction
prepared; dedicated validation/integration, actual environment checkers and Colab
smokes remain before training. No allocation or performance result.

## Hypothesis and scope

[Saved model evidence](../results/history-optimizer-2026-10-23/README.md) shows finite
Adam states and training/development differences without an established cause.
Weight norms/KL/displacement do not prove that regularization helps. Existing
configs contain supervised/SAC weight decay experiments, not a direct recurrent
PPO comparison. Their results do not establish this distinct learner's outcome.
The failed history, capacity and KL-guard families remain closed.

[Fixed protocol](../configs/experiments/recurrent_regularization.json), canonical
SHA-256 `f47e93a8fbe3911a2bcd872b7e26e71a7f5f7730a6cd6bd09daace9c5b40f41a`.

| Arm | Coupled Adam weight decay | Adam epsilon | Parameters |
| --- | ---: | ---: | ---: |
| Control |0|0.00001|61,513|
| Candidate |0.0001|0.00001|61,513|

Locked Torch2.8 Adam uses coupled decay by default; actual construction verifies
`decoupled_weight_decay=False`. All owned actor/critic parameters, including
biases and Gaussian log_std, are included. This changes the learning objective,
not simulator transaction costs or net-log-equity reward. Because log_std is
included, it is not an exploration-neutral directional-only intervention.

SB3 supplies eps1e-5 only when optimizer_kwargs is absent; explicit kwargs must
preserve it in both arms. Otherwise adding decay also changes default epsilon
to1e-8. Both actual fresh constructions verify epsilon/decay/coupling and exact
common untrained tensors. No backpropagation, optimizer steps or fitting occurred.
[Actual probe](../results/recurrent-regularization-2026-10-23/untrained-probe.json).

## Controls and fixed gates

Both use immutable full-history expanded111 training/original26 development data,
original seven features and normalization,21 masked observations, separate
one-layer64 actor/critic LSTMs,64×64 heads and native Gaussian actions. Same
net-log-equity reward, costs, whole shares,100% cap,20-session limit, next-session
execution, learned protections and short capability. Training adaptive exposure,
evaluation/frozen unit-sign execution remains the declared contract. No recent
subset, reward/features/architecture/discount/KL/lr/epoch change.

Fresh seed11 each100k, rollout1000/batch50/10 epochs,10k checkpoints; selected and
final post-update weights both required. Candidate must gain2pp in both phases,
not worsen either final stock/ETF mean, and worsen mean drawdown by at most2pp.
Selected and final stock/ETF excess and median stock excess each≥−2pp versus
matched holding. Failure closes the fixed recurrent-L2 family: no coefficient,
optimizer, model, reward or step grid/extension/confirmation/promotion. Only
passing allows fixed paired11/23/37 confirmation, then frozen prospective evidence.
No development-only result completes the overall objective.

Same checksum-pinned full64 untrained tensors/CPU RNG per seed from the capacity
control entries; no trained weights reused. Smokes512 use the same full64
architecture,128 rollout/batch64 and128-session windows. Require actual80 Adam
updates/40 nominal epochs, correct coefficients/coupled semantics, exact input
snapshot, restored selected/final models/moments/checkpoints,156 frozen recurrent
checks and measured year logs. Reuse common record validation but preserve older
optimizer metadata contracts; add explicit new-study coupling evidence without
silently changing historical records.

## Budget and next implementation

One shared2-unit incremental allowance starts at project38.68329334472413 across
smokes/screen/setup/idle/transfers/failures/retries/shutdown. Original cumulative
project150/130 and account150/170 rules remain. CPU preferred, no GPU allocation
implied. Existing central fleet, shared ledger, ownership/deadlines/downloads and
verified teardown must be reused.

Before100k require actual accepted smoke bytes and runtime projection:
`2×1063.47690073×max(1,l2_smoke_seconds/control_smoke_seconds)+1200≤7200`.
Reference metadata hash is pinned in protocol. Mixed smoke-duration ratios are
heuristics, not hardware speed evidence or deadline guarantees. Reject missing,
stale, failed or over-deadline acceptance; no automatic hardware restart.

Next add strict fixed protocol/parameter/model/coefficient contracts and dedicated
plan/worker/runner/artifact/report wiring, preserving legacy paths. Run actual
data/checkers and regression checks, commit/freeze source, verify official usage
and current Colab Python, then two512 CPU smokes before any100k allocation.
Project/account unchanged:38.68329334 and last verified349.27 zero rate/assignments.
No retired2024+ access or orders. Declining-stock cohort remains reporting-only.
