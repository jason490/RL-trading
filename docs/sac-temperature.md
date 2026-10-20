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
