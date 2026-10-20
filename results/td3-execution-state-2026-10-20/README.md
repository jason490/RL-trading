# TD3 execution-state comparison — completed, gate failed

Three fixed 50k-transition TD3 seed-11 screens ran on Colab CPU after three
1,024-transition smokes. Default Python 3.13.16; frozen training source
`76151d145250519aacedc05a6314436e7868a15c`. Same seven market indicators,
immutable preprocessing, data, reward, costs and four learned action components.

| Observation arm | Selected mean return | Final post-update mean return |
| --- | ---: | ---: |
| Legacy 12 inputs | 4.59% | 2.87% |
| Masked 21 inputs | 9.94% | 0.45% |
| Full execution state, 21 inputs | 6.19% | 6.19% |
| Matched holding | 30.60% | 30.60% |

These are cumulative net-return averages across separate 2021–2023 development
episodes, not annual returns or a deployable portfolio. Selected and final
weights are distinct outcomes. Legacy/masked 50k on-step cached scores were
2.67%/0.77%; the table uses verified post-update final reports.

Full state gains 1.60pp selected/3.32pp final over legacy and loses 3.75pp
selected/gains 5.75pp final versus masked. It misses the required selected gain
against both controls. Final stocks 5.38%/ETFs 10.65% trail holding 32.63%/19.47%;
median stock excess -16.16pp also fails. Mean final maximum drawdown -24.59%
versus legacy -30.54%/masked -24.50% cannot override return failures. No
confirmation seeds, extension, noise sweep or promotion follows this failure.

## Short-side outcome

Full state earns positive total return AND positive net short P&L on 0/3
falling stocks. AMZN +4.42% total has negative short P&L (-$637.68 per $100k);
DIS -46.94% has negative short P&L (-$1,634.46), and NEE -6.75% has positive
short P&L (+$996.98) but negative total return. Neither counts as joint success.
Legacy and masked also succeed on 0/3 in both phases. Future decline labels
remain reporting-only, never observations, rewards, sampling or selection.

## Evidence and operations

Actual initial masked/full actor, target actor, critic and target critic hashes
match exactly. Gaussian exploration sigma 0.1 and delayed-update/target-noise
settings match the protocol. All 468 smoke/screen selected-bundle prefix
predictions are exact, normalization unchanged and incompatible inputs rejected.

[Exact gates and groups](summary.json), [10k learning records](learning-curves.csv),
[screen restoration](restoration.json), [smoke evidence](smoke-summary.json),
[compute](compute.json), [protocol](../../configs/experiments/td3_execution_state.json),
[guide](../../docs/td3-execution-state.md),
[interactive comparison](https://github.com/jason490/RL-trading/releases/download/td3-execution-state-2026-10-20/comparison.html),
[models and derived reports](https://github.com/jason490/RL-trading/releases/tag/td3-execution-state-2026-10-20),
[release verification](release-verification.json), [cleanup](cleanup.json).

All six owned sessions stopped and cleanup verified. Study conservative increment
0.07990875 units; cumulative 36.14910675/150, remaining 113.85089325. Separate
account balance 350.42 with zero reported rate and assignments after shutdown.
The ledger never reset; setup/idle/downloads/failures/retries count. Retired
2024–September 2026 data was not opened or evaluated. Survivor/vintage limitations
and repeated development use remain. The full research objective is unmet.
