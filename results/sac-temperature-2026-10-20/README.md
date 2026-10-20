# SAC initial-temperature comparison — completed, gate failed

Two fixed 50,000-transition SAC seed-11 screens ran on Colab CPU after two
1,024-transition smokes. Actual Python 3.13.16; frozen training source
`0f04e6b3ad96a4afb30243dc8f9066fcf8dfb4f9`. Only initial learned alpha changed
from 1 to 0.001; full execution state, target entropy -4, reward scale 1,
data, costs, actor/critic initialization and all remaining settings match.

| Arm | Selected mean return | Final post-update mean return |
| --- | ---: | ---: |
| Default alpha 1 | 7.93% | 7.93% |
| Initial alpha 0.001 | 19.67% | 9.84% |
| Matched holding | 30.60% | 30.60% |

These are average cumulative net returns across separate 2021–2023 development
episodes (22 stocks, 4 ETFs), not annual returns or a deployable portfolio.
The candidate peaks at 10k. Its cached 50k on-step score 11.48% precedes the
last optimizer update; the actual final report is 9.84%. It gains 11.74pp
selected but only 1.92pp final versus default (required 2pp), and mean maximum
drawdown worsens 4.27pp (allowed 2pp): -30.18% versus -25.91%.

Candidate final stocks 10.78% and ETFs 4.72% trail matched holding 32.63% and
19.47%. Even selected stocks 21.25% and ETFs 10.97% fail absolute gates;
median stock excess is -14.99pp selected, -10.79pp final. The protocol fails:
no confirmation seeds, extension, temperature grid or promotion follows.

## Short-side results

Joint positive total return and net short P&L succeeds on 1/3 falling stocks
at the selected checkpoint (AMZN +38.14%, short +$11,344.78 per $100k), but
0/3 final. Final AMZN +11.70% has negative short P&L; DIS -57.38% and NEE
-16.26% have negative total returns. The desired profitable-short objective
remains unmet. Future decline labels are reporting-only, never model inputs,
rewards, sampling or checkpoint-selection criteria.

## Verification and artifacts

Fresh default configuration, all checkpoint scores, selected/final policy,
critic/target tensors and learned alpha reproduce the prior full-state SAC
control exactly. Initial actor/critic/target fingerprints match across arms.
All 312 smoke/screen restored selected-bundle prefix predictions are exact;
normalization remains immutable and incompatible inputs are rejected.

[Exact results and gates](summary.json), [compact 10k curves](learning-curves.csv),
[restoration](restoration.json), [compute](compute.json),
[protocol](../../configs/experiments/sac_temperature.json),
[interactive comparison](https://github.com/jason490/RL-trading/releases/download/sac-temperature-2026-10-20/comparison.html),
[models and derived reports](https://github.com/jason490/RL-trading/releases/tag/sac-temperature-2026-10-20),
[release verification](release-verification.json), [cleanup](cleanup.json).

All four owned sessions stopped with verified cleanup. Study conservative
increment 0.05950577 units; cumulative 36.06919800/150, remaining 113.93080200.
Separate account balance 350.48; reported rate zero and assignments zero after
shutdown. The shared ledger was never reset. Retired 2024–September 2026 data
was not opened/evaluated. This repeatedly used development panel cannot prove
prospective performance; the full research objective remains active.
