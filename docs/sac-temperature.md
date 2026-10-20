# SAC initial-temperature research

The [saved learning audit](../results/sac-learning-diagnostic-2026-10-20/README.md)
inspects three final policies and their retained training replay. It uses no
optimizer steps, opens no dataset and allocates no Colab. Each replay has50k
transitions;4,096 unique rows are sampled using seed11. Actor noise uses an
isolated/restored Torch RNG scope. Weight/temperature hashes and original files
are unchanged. Final-critic diagnostics cannot reconstruct early critics.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python scripts/diagnose_sac_learning.py \
  --output artifacts/sac-learning/new-summary.json
```

## Observations and limits

Replay mean absolute rewards are0.0055–0.0059. Logged learned alpha is about
0.067 at10k and near1e-5 at50k; default initialization is1. Saved final Q means
are near4, while sampled one-step RMSE is0.0082–0.0099. The diagnostic separately
records reward, discounted target-Q and discounted entropy, respecting timeout
bootstrapping. Final log densities are positive, so final entropy contributions
are negative and small. Early entropy dominance is a hypothesis, not measured
early-state target decomposition or proof of performance failure.

Large value offsets alone do not establish bad action gradients. Soft values
include entropy and historical learning dynamics; low TD error does not certify
financial calibration or profitable trading. There is no checkpoint selection,
forecast correction, reward change or promotion in this analysis.

Implementation was checked against installed SB3 2.7.0 and the
[official SAC source](https://stable-baselines3.readthedocs.io/en/v2.7.0/_modules/stable_baselines3/sac/sac.html).

## Fixed next test

[Protocol](../configs/experiments/sac_temperature.json) compares default `auto`
(initial alpha1) with `auto_0.001` (initial alpha0.001), retaining automatic
adaptation, target entropy−4, full execution observations, standard seeded
actor/critic initialization and all other parameters. Keep net-log-equity reward
scale1, default costs, seven market indicators and the immutable original
train/validation-only data. No reward scaling or temperature grid is combined.
The initial exploration/soft-target trajectory changes; this is not claimed
as merely cosmetic numerical conditioning.

Two1024-transition Colab CPU smokes must verify alpha and equal initial seeded
actor/critic tensors before two50k seed11 screens. Report10k curves and selected/
final weights. Fresh default scores/tensors must be checked against the previous
full-state control before interpreting the candidate. Candidate must improve
both selected/final mean returns by2pp, avoid final stock/ETF deterioration and
keep relative drawdown within2pp; both phases also require group holding excess
and median stock excess at least−2pp. Failure stops this fixed configuration.
Only a passing screen proceeds to seeds23/37 with controls and the declared
multi-seed gates. Prospective confirmation remains required for the full goal.

Use the existing Colab CLI/fleet, shared150-unit ceiling and130-unit working
stop,2-hour owned deadlines and a2-unit incremental study allowance including
smoke/setup/idle/retrieval/retries. Match actual default Colab Python before
dependency sync. Retrieve and verify publication/owned teardown before removing
duplicates. No retired-test access, future decline-label selection or broker
orders. Short profits must accompany positive total returns in the separate
falling-stock report. No model is promoted by this diagnostic.

Current stage: six numeric tests and actual replay inspection pass; protocol
fixed, runtime/job/archive integration pending. No new allocation or training.
Project remains36.009692/150; keep the full research goal active.

Version17 integration now locks the initial-temperature override, dataset and
protocol in task/worker/archive checks. Runner records actual initial alpha,
target entropy and actor/critic/target fingerprints before learning. Valid-case
fixtures and semantic-tampering rejection pass;49 focused temperature/SAC/fleet/
budget/artifact checks pass without fitting. Two immutable smoke payloads
validate. Actual cross-arm initial-weight equality and Colab parity remain
required before screening; fixture fingerprints are not experimental evidence.

The two smokes have passed from frozen source0f04e6b, actual Python3.13.16.
Initial actor/critic/target fingerprints match exactly; alpha1/0.001 and target
entropy−4 are verified.156 restored predictions are exact, and both owned
sessions stopped. The fixed50k screen is now launched under the original
ledger/lock and2-unit allowance including smoke. The screen reporter is implemented:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python scripts/report_temperature.py \
  --state .runtime/temperature-screen-state.json \
  --output artifacts/temperature/screen-report.json
```

It validates cross-arm initial fingerprints and normalized matched settings,
then checks fresh default configuration, all checkpoint scores and selected/
final model tensors against the prior full-state control. Any discrepancy
disables promotion and is reported; it never triggers automatic retries.
Fixed two-arm gates and interactive10k charts reuse the verified common
reporting code. Ten focused gate/archive/initialization tests pass without
fitting. Actual screen/reproduction acceptance is still pending.

[Actual Colab evidence](../results/sac-temperature-2026-10-20/README.md):
project36.022140 after smoke; screen/reproduction/gates remain pending.

## Completed initial-temperature comparison

The two 50k SAC seed-11 screens are complete and fail the preregistered gates.
Initial alpha 0.001 improves selected mean return to 19.67% versus default
7.93%, but final falls to 9.84%; the final gain of 1.92pp misses 2pp and
drawdown worsens 4.27pp. Final stocks/ETFs 10.78%/4.72% trail matched holding
32.63%/19.47%; joint falling-stock total/short positives are 1/3 selected,
0/3 final. No extension, confirmation seeds, temperature sweep or promotion.

Fresh default configuration, all checkpoint scores and selected/final network
tensors reproduce the prior full-state SAC exactly. Initial network hashes
match across arms; 312 smoke/screen selected-bundle prefix predictions are exact.
All four owned sessions stopped. Thirteen release assets have verified server
digests; duplicate-only cleanup removes 15 files (167,684,918 bytes), preserving
354 file hashes including models, replay, data, source, ledger and plan.html.
The post-cleanup report reproduces byte for byte. Eighteen focused temperature
and report-contract checks pass. Cumulative project 36.069198/150; account
350.48 with zero assignments/rate after shutdown. Retired test stays isolated.
[Complete evidence](../results/sac-temperature-2026-10-20/README.md).

Next inspect saved selected/final action paths, directional contributions,
exposure and cost differences to distinguish deterioration mechanisms before
preregistering another bounded hypothesis. This diagnostic needs no fitting,
provider calls or Colab allocation. Development results remain repeatedly used;
prospective confirmation and the overall benchmark/short objective remain unmet.

## Saved temperature-path outcome

Candidate selected-to-final stock/ETF net return changes -10.47/-6.25pp split
into long -7.34/-3.45pp and short -3.13/-2.80pp. Cost drag rises only0.91pp
for stocks and falls0.75pp for ETFs. Saved-path non-cost changes -9.56/-7.00pp
show costs alone cannot account for deterioration; exposure rises while both
side contributions worsen. This is descriptive accounting, not a causal
cost-free replay or proof of an optimization mechanism. No fitting/data access/
Colab and source hashes unchanged;11 existing numeric/contract checks pass.
[Evidence](../results/sac-temperature-paths-2026-10-20/README.md).

Before new allocation, audit existing TD3 input/exploration contracts and whether
a fixed full-state delayed-update comparison provides a distinct hypothesis.
Prior weak TD3 results stay visible. No promise of improvement, automatic
allocation, temperature rescue or relaxation of stock/ETF/short/prospective
requirements. Project remains36.069198/150.
