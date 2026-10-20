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

## Version 18 preflight

Job builder, preallocation payload validation, remote worker and archive contracts
now enforce the fixed protocol. Runner records actual initial actor, target actor,
critic and target critic hashes, Gaussian exploration and delayed-update settings
without action sampling. Masked/full network equality remains an actual Colab
smoke requirement, not inferred from fixtures. The strict reporter validates
source/configuration, complete causal report accounting, matching holding paths,
selected/final checkpoint scores and fixed gates; no candidate is yet promoted.

Fifty TD3/SAC/temperature/fleet/budget checks pass; three immutable smoke payloads
validate without allocation or fitting. Archive tests cover both 1,024 and 50k
contracts and semantic tampering despite recomputed checksums. Existing off-policy
parameter checking now treats TD3 as off-policy rather than injecting PPO rollout
parameters. Scripts compile. Next freeze committed source, then three metered CPU
smokes under one unchanged study allowance; do not screen before actual parity.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_td3_state_jobs.py --smoke \
  --output .runtime/td3-state-smoke-jobs.json
PYTHONPATH=src .venv/bin/python scripts/report_td3_state.py \
  --state .runtime/td3-state-screen-state.json \
  --output artifacts/td3-state/screen-report.json
```

## Actual TD3 Colab smoke acceptance

Three1024-transition CPU smokes pass on default Python3.13.16 from source76151d1.
Actual masked/full initial actor, actor-target, critic and critic-target hashes
match; Gaussian noise and delay/smoothing match the fixed protocol.234 exact
frozen-prefix predictions, immutable normalization and incompatible-schema
rejection pass. All3 owned sessions stopped. Project36.090878/150; study start
36.069198 and2-unit allowance stay unchanged. Proceed only to the fixed3x50k
seed11 screen; no smoke-return tuning or retired-test reuse.
[Evidence](../results/td3-execution-state-2026-10-20/README.md).

## Completed TD3 state comparison

The fixed three-arm50k screen fails. Selected/final returns: legacy4.59%/2.87%,
masked9.94%/0.45%, full6.19%/6.19%, versus holding30.60%. Full final stocks5.38%
and ETFs10.65% miss holding32.63%/19.47%; median stock excess-16.16pp fails.
Full gains3.32/5.75pp final over controls but misses both selected-gain gates.
Final drawdown-24.59% cannot override return failures. All arms/phases have0/3
joint falling-stock positive-total/positive-short cases. No extension, noise
sweep, confirmation or promotion.

Actual initial masked/full four-network hashes match;468 smoke/screen frozen
prefix predictions are exact with immutable normalization/schema rejection.
All6 owned sessions stopped;13 server assets verified. Duplicate-only cleanup
removed21 files (241,549,725 bytes), preserving405 file hashes including models,
replay, raw data, frozen source, ledger and plan.html. Post-cleanup report is
byte-identical;28 focused TD3/report-contract checks pass. Project36.149107/150;
separate account350.42 with zero rate/assignments.
[Complete evidence](../results/td3-execution-state-2026-10-20/README.md).

Before another algorithm or feature study, preregister a saved-policy replay on
a fixed training-only interval to compare fitted-period performance with the
known development failures. This can distinguish weak fitting from a
generalization gap without new fitting or Colab. Preserve normalization/models
and do not select checkpoints from the diagnostic. Earlier state-study failures
remain negative evidence, not proof that all execution information is useless.
Prospective confirmation and the full objective remain unmet.
