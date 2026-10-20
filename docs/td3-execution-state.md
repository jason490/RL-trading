# TD3 execution-state research

The [saved audit](../results/td3-diagnostic-2026-10-20/README.md) verifies three
baseline TD3 seeds used 12 legacy inputs, fixed Gaussian action noise sigma 0.1,
and delayed actor/target updates. It loads only owned saved models and validation
reports, without data access or fitting. Prior final stock/ETF means -0.08%/-0.37%
underperform holding 32.63%/19.47%, with substantial cost and non-cost gaps.
Input omission is measured; its performance effect remains unproven.

TD3 uses clipped double critics, delayed policy updates and target smoothing.
The project's explicit action noise remains enabled. These mechanics differ
from SAC's learned entropy, but do not predict better trading results.
[Official pinned SB3 2.7.0 documentation](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/td3.html).

## Fixed protocol

[td3_execution_state.json](../configs/experiments/td3_execution_state.json) is
checksummed by `training/td3_state.py`. Three fresh equal-budget arms compare
legacy 12 inputs, masked 21 inputs and causal full 21 inputs. Masked/full share
capacity and seeded actor/target/critic initialization; smoke must verify those
actual initial fingerprints, noise and dimensions. Keep seven indicators,
original immutable dataset 672a2162…, training-only preprocessing, net-log reward
scale 1, costs and four learned action components. Train only through 2020;
2021–2023 remains repeatedly used development. Never open the retired test.

Use existing parameters: 100k replay, batch 256, warmup 1,000, one critic update
per collected step, policy delay 2, target noise 0.2 clipped at 0.5, tau 0.005,
learning rate 0.0003, gamma 0.99 and 64×64 networks. Collection action noise is
NormalActionNoise, zero mean and sigma 0.1 on all four normalized actions.
No hyperparameter sweep is combined. Existing 1,024-step smokes use locked
32×32 networks, batch 64 and warmup 64; smoke return never selects settings.

After three verified Colab CPU smokes, run three 50k seed-11 cells with 10k
checkpoint curves and distinct selected/final reports. Candidate must gain 2pp
against both controls in both phases, avoid final stock/ETF deterioration and
limit extra final mean drawdown to 2pp. Each phase must have stock and ETF excess
and median stock excess at least -2pp. Failure stops this family, without
confirmation seeds, extensions or parameter rescue. Only a pass permits matched
23/37 confirmation under the protocol's multi-seed criteria; prospective frozen
confirmation remains required. Joint positive total/short returns on declining
stocks are reported separately, with future labels never used in selection.

## Runtime and evidence requirements

Use the existing fleet and centralized ledger, Colab default Python verification,
CPU profiling precedent and locked dependencies. One study's 2-unit allowance
includes all smokes, setup, idle time, retrieval, failures and retries. Keep the
project ceiling 150, working stop 130 and per-allocation deadline 2 hours.
Persist every owned session ID, retrieve durably with bounded retries and stop/
verify all owned runtimes. Checksum publication before duplicate-only cleanup;
preserve weights, replay, curves, source and data. Do not allocate until job,
worker, archive and initial/noise contracts pass preflight.

Current status: audit and preregistration complete; 12 focused protocol and
saved-path accounting checks pass. Integration/smoke/screen remain pending.
No new Colab allocation. Project 36.069198/150, objective active and unmet.
