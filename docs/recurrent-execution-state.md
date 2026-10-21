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
