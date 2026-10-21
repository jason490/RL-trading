# SAC discount horizon — complete, gates failed

Fresh fixed100k seed11 SAC CPU runs compare gamma0.99 control and0.95 candidate.
Same expanded111 training symbols, original26 development episodes, preprocessing,
21 full-state inputs, seeded networks/alpha, sampler, reward, costs and Adam0 decay.
Actual initial model settings and saved selected/final records restore correctly.

| Arm | Selected mean return | Final post-update return |
| --- | ---: | ---: |
| Control gamma0.99 | 24.19% | 24.19% |
| Candidate gamma0.95 | 22.79% | 22.16% |
| Matched holding | 30.60% | 30.60% |

Separate2021–2023 cumulative development episodes, not annual returns or an
investable portfolio. Candidate selected/final loses1.39/2.03pp against control,
missing the required+2pp. Final stock23.88%/ETF12.64% trail holding32.63%/19.47%;
median stock excess−9.78pp. Final mean maximum drawdown−32.75% improves only
0.13pp versus control, insufficient to override failed return gates. No extension,
gamma sweep, confirmation seeds or promotion. Overall objective remains unmet.

## Short-side result

Candidate selected/final achieves1/3 joint positive total return AND positive net
short P&L on reporting-only declining stocks. Final AMZN+32.22% with+$6344.06
short contribution passes; DIS−38.74% with+$2404.43 and NEE−8.17% with+$2793.90
fail because total return is negative. Control0/3. Positive shorts alone cannot
claim all falling stocks were traded profitably. Future decline labels never enter
observations, rewards, sampling or checkpoint selection.

## Coverage and integrity

Both arms have exactly matching actual symbol/group counts:80,032 stock and
19,968 ETF transitions (19.968% ETF); all111 symbols receive transitions.
Fresh initial actor/critic/target/alpha match; only discount differs. New current
checkpoint identities bind saved weights/reports to steps/phases/hashes. Both
complete screens and both smokes pass strict archives.312 smoke/screen frozen
prefix predictions are exact with immutable normalization and schema rejection.
Retired2024–September2026 remains unopened. No prospective success claimed.

## Operations and artifacts

All4 owned CPU sessions stopped and cleanup verified. Actual default Python3.13.16;
existing CLI0.7.4 reused without upgrade. Study conservative increment0.10994451;
project37.89715791/150, remaining112.10284209. Post-shutdown account349.88,
rate0/assignments0. Account floor150/teardown170 and project150/130 remain active;
ledger never reset and prior failed/interrupted evidence stays retained.

[Exact gates and per-symbol results](summary.json), [10k curves](learning-curves.csv),
[screen restoration](restoration.json), [smoke evidence](smoke-summary.json),
[compute](compute.json), [protocol](../../configs/experiments/sac_discount.json),
[guide](../../docs/sac-discount.md),
[interactive comparison](https://github.com/jason490/RL-trading/releases/download/sac-discount-2026-10-21/comparison.html),
[models and derived reports](https://github.com/jason490/RL-trading/releases/tag/sac-discount-2026-10-21).

Next inspect saved selected/final long/short exposure and cost contributions
before a distinct hypothesis. No fitting, provider calls or Colab needed for
that diagnosis. Failed gates and repeatedly used development limitations remain.

Publication verifies13 server assets. Cleanup removes31,529,571 duplicate bytes,
398 retained hashes unchanged; strict report reproduces byte-for-byte.35 focused
checks pass. [Publication](release-verification.json), [cleanup](cleanup.json),
[reproduction](reproduction.json).
