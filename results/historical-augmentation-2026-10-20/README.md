# Historical ETF augmentation — completed, gate failed

The fixed three-arm seed11 screen ran50,000 PPO transitions per arm on Google
Colab CPU, after three512-transition smokes. Actual kernel Python3.13.16,
source `4be5219e782de018ddb5e28cf2fd9e3471780fc5`, shared original ledger.

| Training arm | Selected mean return | Final mean return |
| --- | ---: | ---: |
| Original | 19.23% | 15.01% |
| Recent-ETF oversampling | 15.33% | 15.33% |
| Earlier-ETF augmentation | 17.06% | 17.06% |
| Matched holding | 30.60% | 30.60% |

These are mean net returns across separate2021–2023 development episodes,
not annualized returns or a combined deployable portfolio. Repeated development
validation is not fresh market evidence. No retired-test interval was used.

Earlier augmentation fails both control comparisons: selected return−2.17pp
versus original, and only+1.73pp versus recent oversampling (required+2).
Its final stock17.87%/ETF12.56% trails holding32.63%/19.47%; median stock
excess−7.42pp also fails. Final mean drawdown−27.79% is better than original
−29.18% but worse than recent−26.76%. No confirmation, extra steps or model
promotion follows this failed fixed screen.

## Profitable shorts

Earlier augmentation achieves joint positive total return and net short P&L in
one of three falling stocks: AMZN+3.81% total, net short P&L+$9,560.76 per$100k
initial equity. DIS−45.50% andNEE−11.07% fail despite positive short P&L. Original
has0/3 joint successes; recent oversampling1/3. These are reporting-only cohorts,
never observations, rewards, sampling labels or checkpoint-selection criteria.
Loss reduction must not be described as profitable total performance.

## Matched evidence and artifacts

All original observations, training-only normalization and validation episodes
are unchanged. Earlier training adds8,776 separately warmed/purged rows. The
augmented arms have exactly equal realized transition counts per symbol;
7,680 older transitions (15.36% of50k), with14,336 total ETF transitions per
augmented arm (28.672%). Expected ETF probability was8/30; actual counts are
reported rather than presumed. No outcome labels enter the full-window sampler.

[Metrics and exact fixed gates](summary.json), [interactive curves](https://github.com/jason490/RL-trading/releases/download/historical-augmentation-2026-10-20/comparison.html),
[compact10k learning records](learning-curves.csv), [sampling parity](sampling-parity.json),
[frozen restoration](restoration.json), [compute](compute.json),
[preparation](preflight.json), [smoke evidence](smoke-summary.json).
234 screen-prefix and234 smoke-prefix frozen predictions are exact; preprocessing
is unchanged and incompatible inputs rejected. All six owned sessions stopped
and teardown verified. Study conservative increment0.04629467 units; cumulative
35.91753514/150, remaining114.08246486. Separate account balance350.58, rate0,
zero active assignments verified after shutdown. Metering includes setup/idle,
retrieval and failures; billing may lag and the ledger was never reset.

[Download all six models and derived reports](https://github.com/jason490/RL-trading/releases/tag/historical-augmentation-2026-10-20).
[Protocol](../../configs/experiments/historical_augmentation.json),
[guide](../../docs/historical-augmentation.md), [cleanup](cleanup.json).
All15 uploaded assets passed server SHA-256/size verification. Duplicate-only
cleanup removed21 files (195,506,959 bytes,186.45MiB); all377 retained hashes
are unchanged. Unique snapshots, all weights/reports, raw data and ledger remain.
[Publication verification](release-verification.json), [cleanup](cleanup.json).
Retain complete unsuccessful evidence and stock-survivorship/vintage limitations.
The full stock/ETF and profitable-short objective remains unachieved. Diagnose
remaining signal/execution failures before committing a distinct hypothesis;
this screen does not justify more history or training by default.

Post-cleanup report replay reconstructs transport validation temporarily and
reproduces the summary byte-for-byte; all377 retained hashes remain unchanged.
[Reproduction proof](reproduction.json). No training or new allocation.
