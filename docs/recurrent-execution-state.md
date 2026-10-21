# Recurrent execution-state research

## Evidence and hypothesis

[The retained-model audit](../results/recurrent-contract-2026-10-21/README.md)
confirms the earlier recurrent study used12 inputs and working memory resets.
It did not expose held protective-order geometry and partial-exit state directly.
That study failed; its three-seed selected/final mean13.27%/6.13% versus holding
30.60% stays unchanged. Missing explicit state is a testable hypothesis, not a
proven explanation. This study does not extend the failed discount protocol.

## Fixed comparison

[Checksummed protocol](../configs/experiments/recurrent_state.json) declares:

| Arm | Algorithm | Inputs |
| --- | --- | --- |
| feedforward_full | PPO | Seven market, five legacy portfolio, nine execution |
| recurrent_masked | RecurrentPPO | Same dimensions, nine execution fields zeroed |
| recurrent_full | RecurrentPPO | Full21 inputs |

Recurrent controls have identical initial tensors at each seed; actor and critic
use separate one-layer64-unit LSTMs and64×64 heads. Feed-forward PPO has64×64
heads and a different parameter count. It is an algorithm/architecture control,
not a parameter-count-matched proof of memory. Masked/full recurrence isolates
explicit information with equal architecture and initialization.

All arms use expanded111 training symbols, the original26 development episodes,
original immutable seven-feature normalization, stock/ETF shuffled group cycles,
512-transition episode windows and actual per-symbol/group sampling counts.
Dataset cdb93a33… is immutable; no further downloads are needed. Fresh100k
seed11 cells use rollout1000, batch50, ten epochs, learning rate0.0003,
gamma0.99, GAE0.95, clip0.2 and entropy coefficient0. No reward, cost, indicator,
window-length or protection changes. The recurrent library handles sequence
boundaries and masks; no backpropagation-length sweep is authorized by this
protocol. CPU is preferred for the small networks, with three-way concurrency
allowed under centralized accounting.

## Acceptance and gates

Before the screen, three512-transition CPU smokes must verify actual21-input
models, masked/full recurrent initialization equality, memory carry, episode
resets, deterministic frozen sequence parity, save/load, immutable preprocessing,
strict dataset identity and current checkpoint/report hashes. Smoke uses128
rollout, batch64,32-unit LSTM/heads; smoke returns never tune configuration.
Training remains in Colab. Local construction tests perform no optimizer updates.

Evaluate every10k. Candidate selected AND final return must each exceed BOTH
controls by2pp, final stock/ETF gains against each control must be nonnegative,
and final mean drawdown must not worsen by more than2pp. Selected AND final
stock/ETF excess and median stock excess must each be at least−2pp versus
matched holding. Failure closes this fixed family: no extra steps, architecture
sweep, reward adjustment, confirmation or promotion. Only a passing screen permits
fixed seeds11/23/37 confirmation under the protocol's paired and absolute gates.
Even then a frozen prospective comparison is required before success claims.

Report positive total return AND positive net short P&L on reporting-only declining
stocks; their future labels never enter inputs, sampling, reward or selection.
Repeated2021–2023 development evidence cannot become independent test evidence.
Retired2024–September2026 remains unopened. Survivorship/delisting limitations
remain; no paper/live broker orders are enabled.

## Runtime and implementation status

Study allowance3 incremental units includes all smokes, setup, idle allocation,
retrieval, failures and retries. Cumulative project ceiling150/teardown130 and
account floor150/teardown170 both apply; current project37.897158 is retained.
Each allocation has7200 seconds including retrieval. Check actual default Python,
monitor aggregate usage every minute, checkpoint durably, stop/verify all owned
sessions on failure or unreliable metering, and never restart automatically after
a budget stop. Verify published hashes before duplicate cleanup.

Protocol and construction contracts are implemented in
[recurrent_state.py](../src/rl_stocks/training/recurrent_state.py).
Seven tests cover actual fresh model construction, equal recurrent tensor hashes,
unchanged caller overrides, invalid actual records, untrained save/load sequence
and reset parity, and mandatory relative/absolute gates. No training or allocation
has occurred for this study. Version22 job, worker, artifact and restored-report integration is now implemented
and checked. Actual immutable smoke payloads validate. In particular,
selected steps must remain distinct from final100k post-update identity; never
infer the latest report from a legacy selected validation file.

## Integrated remote acceptance

Exact version22 plans reject unregistered changes and bind the account150/170
contract. Controller and worker validate expanded111 immutable data before fitting.
The runner records actual initial/selected/final model settings, dimensions, LSTM
configuration, steps and policy tensor hashes. Archives require current checkpoint
identities, fixed preprocessing, source metadata, exact sampling totals and saved
records.15 archive fixtures include semantic corruption despite recomputed hashes.
69 focused model/plan/archive/checkpoint/account/fleet/report checks pass. These
checks do not replace actual training or frozen recurrent sequence acceptance.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_recurrent_state_jobs.py \
 --smoke --output .runtime/recurrent-state-smoke-jobs.json
```

The builder does not allocate. Freeze and push clean source, check official CLI
usage and actual default Python, then run three512 CPU smokes using the existing
owned fleet. Accept sequence/reset/frozen parity and exact initial recurrent
records before any100k screen. After verified screen shutdown,
`scripts/report_recurrent_state.py --state STATE --output NEW_JSON` restores
actual weights, revalidates paths against immutable development data and evaluates
predefined gates. It refuses incomplete states or mismatched sampling/source.
No new Colab study session has been provisioned at this integration milestone.

Recurrent frozen acceptance now checks action AND hidden/cell-state equality on
causal three-step prefixes of every development symbol, deterministic repeats
from the same prior state, caller-state immutability and fresh episode resets.
Regression probes reject lost memory and state mutation;4 focused checks pass.
`scripts/accept_recurrent_smokes.py` requires three complete cleaned runs, actual
restored selected/final records, exact source proof, equal recurrent initial
tensors/sampling and234 exact frozen predictions. Actual smoke acceptance remains
pending while the original frozen-source Colab controller executes. Local verifier
changes do not alter running workers or fit parameters.

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
