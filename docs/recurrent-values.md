# Saved recurrent value-target audit

This is an inference-only diagnostic of the accepted legacy RecurrentPPO control,
not another VIX experiment or a training allocation. Its selected80k and final100k
weights are pinned by checksum before execution. All failed families remain closed.

## Question and fixed scope

Sparse SB3 explained-variance logs describe sampled training rollouts. They do not
establish expected-return calibration or explain why development performance drops.
The audit compares saved value predictions with later observed, discounted rewards
on one fixed training window per original26 stock/ETF symbols. No outcome chooses
the symbols, window, checkpoints or random seeds.

The [protocol](../configs/experiments/recurrent_value_audit.json) fixes the original
seven-field111-symbol dataset, original26-symbol panel, last513 observations of
each unique training episode ending2020-12-02,512 transitions, first256 decisions
for primary summaries, gamma0.99 and GAE lambda0.95. The default simulator,
execution-masked21 inputs and training's unrestricted exposure remain unchanged.
The stock/ETF groups are reported separately.

## Collection and targets

Use native stochastic Gaussian actions from the actual saved policy, carrying
both actor and critic hidden/cell state. Reset states and Torch draws per symbol;
seed1911 plus fixed panel index is reused across phases. Policy-conditioned paths
can differ even with common underlying draws. Clip raw components to[-1,1] just
as installed RecurrentPPO does; record per-component clipping fractions. Cast
rewards to float32 as SB3 vector collection does. Independently check every
forward value against `predict_values` from the same previous critic state.

Reconstruct observed zero-tail discounted returns, critic-bootstrapped discounted
returns and GAE lambda returns. Time-limit/data cutoffs keep marked equity and
bootstrap; true insolvency termination uses zero tail. Do not add liquidation
costs at truncation. Tests compare GAE targets with SB3 and discounted sums with
independent direct arithmetic; no buffer training or optimizer step occurs.

## Limits on interpretation

One stochastic path per phase/symbol is not an unbiased expected-return estimate.
Overlapping targets are correlated. Zero-tail returns are censored; bootstrapped
and lambda targets depend partly on the critic itself. First256 decisions reduce
end-of-window effects but do not remove these limits. Report bias, RMSE, MAE,
target variability, descriptive explained variance and bootstrap sensitivity.
These are target-consistency diagnostics, not independent calibration proof,
causal optimizer attribution, model selection or permission to change rewards.

## Integrity and operation

Read only completed, owned-cleaned sources. Verify actual saved policy/Adam
records, source metadata, weights, immutable data and normalization. Keep model
counters, learned tensors/moments and all source files unchanged; restore caller
Torch/NumPy/Python RNG and CPU thread count. Fixtures forbid `.learn` and
`Adam.step`, check stochastic repeatability and reject inconsistent value heads.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python scripts/diagnose_recurrent_values.py \
  --output artifacts/recurrent-values-2026-10-25
```

The worker has a600-second inclusive deadline, atomic progress journal, new-only
output directory and retained compact per-phase JSONL traces. Failure records its
stage and does not restart automatically. No provider requests, development replay,
retired-test access, new checkpoint selection, Colab allocation or broker orders.
