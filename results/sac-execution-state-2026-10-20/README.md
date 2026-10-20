# SAC execution-state comparison — completed, gate failed

Three fixed50k-transition SAC seed11 screens ran on Colab CPU after three1024-
transition smokes. Actual kernel Python3.13.16, source
`bb0f3e25bcf60765df158401434959458508da7c`. Same immutable training-only
preprocessing, seven indicators,26-symbol development panel, costs and actions.

| Observation arm | Selected mean return | Final post-update mean return |
| --- | ---: | ---: |
| Legacy12 inputs | 7.44% | 7.39% |
| Masked21-input control | 17.84% | 17.69% |
| Full21-input execution state | 7.93% | 7.93% |
| Matched holding | 30.60% | 30.60% |

These are mean net returns across separate2021–2023 development episodes, not
annualized returns or a combined deployable portfolio. Cached pre-final-update
checkpoints gave7.44%/17.84% for legacy/masked; the verified final reports above
include the final optimizer update. Selected and final must not be conflated.

Full state fails: gains only0.48pp selected/0.53pp final versus legacy and
loses9.91pp/9.77pp versus masked (required+2pp against both). Final full-state
stock8.68%/ETF3.77% trails holding32.63%/19.47%, median stock excess−17.40pp.
Mean final drawdown improves to−25.91% versus legacy−33.08% and masked−30.18%,
but that does not override the return gates. No confirmation, longer run or
promotion follows. More execution information did not improve this fixed SAC
configuration; this does not prove such information is universally useless.

## Short-side results

Full state achieves joint positive total return and net short P&L on1/3 falling
stocks: AMZN+11.62% total and+$15,115.14 net short P&L per$100k initial equity.
DIS−35.26% andNEE−2.94% fail, with negative short contribution. Legacy succeeds
on0/3; masked1/3 (NEE). These future decline cohorts are reporting-only, never
inputs, rewards, sampling labels or checkpoint-selection criteria.

## Evidence and operations

[Exact gates and group/symbol results](summary.json),
[interactive comparison](https://github.com/jason490/RL-trading/releases/download/sac-execution-state-2026-10-20/comparison.html),
[compact10k learning records](learning-curves.csv),
[screen frozen restoration](restoration.json), [smoke records](smoke-summary.json),
[compute](compute.json), [protocol](../../configs/experiments/sac_execution_state.json),
[guide](../../docs/sac-execution-state.md).

All468 smoke/screen selected-bundle prefix predictions are exact, with immutable
normalization and incompatible-input rejection. All six owned sessions stopped,
cleanup verified. Study conservative increment0.09215709 units; cumulative
36.00969223/150, remaining113.99030777. Separate account350.51, zero assignments
and rate verified after shutdown. Original ledger includes setup/idle/retrieval
and is never reset. Retired2024–September2026 data was not opened or evaluated.

[Download all six models and derived reports](https://github.com/jason490/RL-trading/releases/tag/sac-execution-state-2026-10-20),
[release verification](release-verification.json), [cleanup](cleanup.json).
Keep unsuccessful models, unique checkpoints and survivor/vintage limitations.
The overall benchmark and profitable-short objective remains unachieved.

All13 release assets passed server SHA-256/size verification. Duplicate-only
cleanup removed21 files (238,984,076 bytes,227.91MiB); all390 retained hashes
remain unchanged. Replay from retained files reproduces the report byte-for-byte.
[Reproduction](reproduction.json). No model, unique checkpoint, source data or
ledger was removed. No further allocation follows this failed screen.
