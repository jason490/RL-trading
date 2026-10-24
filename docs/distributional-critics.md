# Distributional critic research review

TQC is a candidate for a distinct algorithm study, not an implemented project
algorithm or a current trading recommendation. The installed `sb3-contrib==2.7.0`
exports TQC without a dependency change. The project factory/frozen runtime do not
yet register it. No TQC policy has been trained, evaluated or promoted here.

TQC estimates distributions of action values with multiple quantile critics and
truncates high target quantiles. Its research motivation is controlling value
overestimation in continuous control. The primary paper provides continuous-control
benchmarks, not stock-trading evidence. See the [original paper](https://proceedings.mlr.press/v119/kuznetsov20a.html)
and [version-matched implementation documentation](https://sb3-contrib.readthedocs.io/en/v2.7.0/modules/tqc.html).
That implementation supports continuous Box actions and does not supply recurrent
policies. Its documented example uses two critics,25 quantiles and two dropped
quantiles per critic; these are possible fixed study settings, not trading optima.

## Evidence needed before choosing the experiment

The existing [SAC learning diagnostic](sac-temperature.md) records large final
soft-value offsets and much smaller one-step TD errors. It explicitly does not
establish overestimation, inaccurate action gradients or the cause of return
failure. Soft critic values include entropy and discounted continuation. Comparing
them directly to raw profit or deterministic episode return would be invalid.
Failed SAC temperature, state, discount and regularization families remain closed.
TQC must not become a pretext for reopening their parameter searches.

The next useful review is an inference-only calibration diagnostic of a fixed saved
SAC policy on training-only windows. Before executing it, fix the saved model/data
identities, starts, stochastic rollout seeds, rollout count and horizon; account
for reward scale, entropy, action-conditioned first-step values, termination and
timeout bootstrapping consistently. Preserve checkpoints, optimizer state,
normalization and RNG; no local fitting. Report sampling uncertainty and signed
calibration differences. Any inability to reconstruct the policy's exact training
MDP or reward/observation contract must stop that diagnostic, not be filled with
an approximation. Calibration evidence could motivate a critic study but would
not prove a causal performance fix.

## Requirements if a matched study is justified

Freeze a new protocol before any performance scoring or Colab allocation. Candidate
arms are native SAC, distributional TQC with zero truncation, and otherwise matched
TQC with one fixed truncation setting. The zero-truncation control distinguishes
critic-distribution effects from truncation. Use the same native market inputs,
execution state, data, episode sampling, reward, costs, actor architecture,
learning rate, replay/gradient budget and temperature behavior. Record the critic
architecture difference rather than calling the complete networks identical.
Verify shared initial actor weights and RNG setup with actual tensors; do not
claim identical training trajectories across different critic objectives.

Implementation would need factory/configuration registration, input and quantile
geometry validation, actual optimizer/critic restoration, frozen deterministic
save/load parity and report/archive identity support. First run bounded Colab CPU
smokes with actual kernel-version inspection and profiling. Any screen must retain
selected and final checkpoints, compact step/return curves, matched stock/ETF
holding, drawdown and short-side diagnostics. Reuse the established relative and
absolute development gates; only passing screens proceed to seeds11/23/37 and
prospective confirmation. No retired-test access, coefficient/quantile grid,
reward changes or extensions after failure. Use the existing shared project and
account ledger/stop thresholds; no new allocation is authorized by this review
alone without the normal preflight and frozen study protocol.

This review runs independently of financial-source feasibility. No SEC feature
is enabled, no financial coverage failure is hidden, and no trading-return
improvement is claimed.

## Fixed saved-SAC soft-consistency diagnostic

[Protocol](../configs/experiments/sac_soft_consistency.json) now fixes the final
50k full-execution SAC seed11 model, original26-symbol train/development-only
source, exact model/metadata/source-state hashes and frozen source revision
`bb0f3e25bcf60765df158401434959458508da7c`. The frozen package identity is verified
before execution. The current environment has changed since training; the worker
must import the retained original package instead of approximating its behavior.

64 distinct reset seeds sample the original `legacy_uniform` training-start
contract; four fixed action-noise seeds repeat each flat-account state. Each
stochastic frozen-policy rollout ends at the original512-session timeout, data
cutoff or true termination. Its action-conditioned soft target is:

```text
r0 + sum(t=1..H-1, gamma^t * (rt - alpha * logpi_t))
   + gamma^H * (min_target_critic_Q - alpha * logpi_tail)
```

The tail is removed only on true termination; timeouts/data cutoffs bootstrap
exactly as training does. Initial-action entropy is excluded from Q(s0,a0), and
the tail uses the saved target critics. Alpha is the saved final value, never
adapted. This is a multistep soft-Bellman consistency diagnostic with a **learned**
bootstrap, not an unbiased true-value estimator. Flat reset states do not establish
held-position/replay-wide calibration or identify wrong action gradients.

Window means are the sampling units. The predeclared descriptive evidence flag
requires mean(minQ−soft-target) minus two conditional sampling standard errors to
be at least0.1 soft-reward units and at least75% of trajectories to reach512 steps.
These are not profit percentages or confidence bounds for future markets. A
passing flag only motivates separately preregistering a matched critic study;
no model is promoted and no trading screen is automatically launched. A failed
flag closes this particular diagnostic rationale without changing seeds/horizon.

Run locally for **inference only**, with the original frozen package:

```bash
PYTHONPATH=.runtime/sac-execution-source/src \
MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
.venv/bin/python scripts/diagnose_soft_consistency.py \
  --output results/sac-soft-consistency-2026-10-24/summary.json \
  --artifacts artifacts/sac-soft-consistency-v1
```

The worker has a900-second execution bound, a durable per-window journal and no
automatic restart. It verifies source hashes, weights, all three optimizer states,
update count, normalization and Torch/NumPy RNG immutability. No replay buffer is
loaded, normalization fitted, optimizer stepped or validation/test policy scored.
11 focused target/termination/timeout/RNG/deadline tests and targeted Ruff pass
before the first actual diagnostic; no outcome is claimed by this preparation.
