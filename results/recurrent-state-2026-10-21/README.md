# Recurrent execution-state comparison — complete, gates failed

Fresh fixed100k seed11 Colab CPU runs compare feed-forward full21, recurrent
masked21 and recurrent full21. Same expanded111 training source, original26
2021–2023 development episodes, immutable normalization, sampler, reward, costs,
PPO rollout/batch/epochs and gamma. Recurrent initial weights match exactly.

| Arm | Selected return | Final post-update return |
| --- | ---: | ---: |
| Feed-forward full | 4.46% | 3.34% |
| Recurrent masked | 24.99% | 24.04% |
| Recurrent full candidate | 3.95% | −3.11% |
| Matched holding | 30.60% | 30.60% |

Means of separate cumulative development episodes, not annualized returns or an
investable portfolio. Candidate loses0.51/6.45pp selected/final to feed-forward
and21.05/27.16pp to recurrent masked. Required+2pp gains fail. Candidate final
stocks−2.25% versus holding32.63%, ETFs−7.84% versus19.47%; median stock excess
−29.25pp. Drawdown cannot override failed return gates. No extension, recurrent
architecture/sequence sweep, confirmation or promotion follows this protocol.
Masked control is the best arm here but still misses absolute holding targets;
it is not retrospectively promoted. Repeated development evidence is not fresh
out-of-sample proof, and the overall objective remains unmet.

## Short side and limits

Candidate selected/final0/3 joint positive total return AND positive net short
P&L on reporting-only declining stocks; feed-forward0/3, masked1/3 selected and
0/3 final. Positive individual short contributions do not establish profitable
whole strategies. Future decline labels never enter learning or selection.
Explicit execution state did not help this configuration. One seed cannot prove
memory or state universally harmful; feed-forward and LSTM parameter counts differ.
Retired2024–September2026 stays unopened. Survivor-data limitations remain.

## Integrity and operations

Three512 smokes plus three100k screens complete on actual Python3.13.16, clean
workerdd3b256fe0b830fb3d662e8cff75077a88e5aa10. Actual initial/selected/final model
settings, tensors and steps restore correctly; checkpoint identities bind weights
and reports. All-arm sampling counts match; all111 symbols receive transitions.
468 smoke/screen frozen predictions are exact, including recurrent hidden/cell
carry, repeats, independent resets, immutable normalization and schema rejection.

All6 owned sessions stopped, artifacts retrieved and shutdown verified. Project
38.01440797/150, remaining111.98559203; study increment0.11725006 including setup,
idle, retrieval and all attempts. Post-shutdown official account349.80, zero rate
and assignments. Existing CLI reused, ledger not reset; project150/130 and
account150/170 protections remain. No broker orders or prospective success.

[Exact gates and symbols](summary.json), [10k curves](learning-curves.csv),
[screen restoration](restoration.json), [smokes](smoke-summary.json),
[compute](compute.json), [protocol](../../configs/experiments/recurrent_state.json),
[guide](../../docs/recurrent-execution-state.md),
[interactive comparison](https://github.com/jason490/RL-trading/releases/download/recurrent-state-2026-10-21/comparison.html),
[models and reports](https://github.com/jason490/RL-trading/releases/tag/recurrent-state-2026-10-21).

Next diagnose saved selected/final costs and directional participation without
fitting before choosing a distinct hypothesis. Preserve every failed gate.

Publication verifies13 server assets. Cleanup removes47,914,914 duplicate bytes
with439 retained hashes unchanged. Post-cleanup strict report reproduces byte
for byte. All selected/final/checkpoint/frozen/source/data evidence remains.
[Publication](release-verification.json), [cleanup](cleanup.json),
[reproduction](reproduction.json).
