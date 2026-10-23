# Fixed recurrent PPO regularization comparison

Status: completed screen failed fixed gates; recurrent L2 family closed. Latest
verified outcome below, earlier sections retain preparation history.

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

## Version29 integration and actual construction acceptance

Fixed `l2_plan`/`l2_study` contracts bind data, coefficients, initialization and
runtime approval. The runner records initial/selected/final L2-specific model
records, actual coupled-Adam defaults/group flags and exact `l2-initialization.npz`.
`runtime/colab/l2_artifacts.py` validates required measured years, coefficient,
epsilon, coupling, optimizer moments/steps, snapshot, checkpoint, execution,
normalization and frozen identity. Older record formats remain unchanged; shared
validation defaults to zero decay and the new caller supplies its fixed value.

138 focused tests pass, including wrong coefficient/epsilon/coupling/boolean
flags, missing-year/checkpoint/snapshot, runtime-profile stale/failed/overdue
rejection, report routing, conditional confirmation, and prior capacity/history/
guard/account-floor regressions. Actual two immutable smoke tasks validate before
allocation. Both actual data environments pass Gymnasium/SB3 checkers; fresh full64
recurrent forwards, matching pinned tensors/RNG and exact optimizer records pass.
[Construction evidence](../results/recurrent-regularization-2026-10-23/construction.json).
No local learning, backpropagation or optimizer steps; source compiles.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python \
 scripts/build_l2_jobs.py --smoke --output .runtime/l2-smoke-jobs.json
```

Commit/freeze source, check official usage and actual default Python, then run
the two512 CPU tasks under the existing central fleet, original ledger and
shared study baseline38.68329334472413/2-unit allowance. After retrieval,
156 frozen checks, actual saved models/moments/coupling/year proof and owned
shutdown, accept without changing settings based on smoke returns:

```bash
PYTHONPATH=src .venv/bin/python scripts/accept_l2_smokes.py \
 --state .runtime/l2-smoke-state.json \
 --restoration .runtime/l2-smoke-restoration.json \
 --source-proof .runtime/l2-source-v1/export-proof.json \
 --output .runtime/l2-smoke-acceptance.json
PYTHONPATH=src .venv/bin/python scripts/build_l2_jobs.py \
 --output .runtime/l2-screen-jobs.json
```

The exact accepted file bytes bind generated runtime profiles, checked against
actual acceptance before controller allocation. Long plans reject missing/stale/
failed/over-deadline approval; confirmation requires a passing same-protocol
screen. Strict final report uses `scripts/report_recurrent_l2.py --state ...
--runtime-profile .runtime/l2-screen-jobs.runtime-profile.json --output ...`.
Actual two512 Colab smokes remain before100k; no allocation or new usage yet.

## Actual Colab smoke acceptance and runtime approval

Two512 CPU/defaultPython3.13.16 runs from1d640ee completed/retrieved and verified
stopped.156 exact frozen sequence/repeat/reset predictions, normalization/input
rejection and actual model/optimizer/checkpoint restoration pass. Both full64
architectures61,513 parameters,80 Adam updates/40 nominal epochs. Correct coupled
Adam0/0.0001 and explicit eps1e-5 verified in actual saved models; same initial
weights/CPU RNG and actual sampling/year margins. No smoke-return selection.
Durations144.9779/151.7729s give conservative3426.6427s/57.1min projection with
1200s reserve, under7200s. Exact actual acceptance binds prepared100k plans.
Same-source seed11 screen next. Project38.70365471, study increment0.02036137,
official account349.26 zero rate/assignments; original baseline/reserves retained.
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
