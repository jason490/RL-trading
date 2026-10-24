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
