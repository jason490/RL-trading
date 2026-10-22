# Saved optimizer diagnostics

Use saved weights, optimizer state and compact training logs before attributing poor returns to optimizer behavior. The current [cash-study audit](../results/cash-optimizer-2026-10-22/README.md) restores eight selected/final models without fitting or opening datasets.

## Counters and finite state

`reporting/optimizer.py` verifies exact trainable-parameter ownership, Adam state shapes, finite weights/moments, nonnegative second moments and integer per-parameter step counters. It records learning rate, epsilon, betas, weight decay, nominal PPO epoch counter and actual Adam steps separately. Missing state is visible rather than inferred from an epoch counter.

SB3-contrib2.7.0 increments `_n_updates` by configured epochs per training call, including calls shortened by KL stopping. Therefore this is a nominal counter, not a trustworthy count of actual minibatch updates. Current no-KL-stop final100k models each have1,000 nominal epochs and20,000 actual Adam steps. The selected on-step checkpoints exclude the pending rollout update.

## Logging limits

Keep the last evaluation at each step; final post-update supersedes on-step. Saved10k points describe only the latest rollout update. Approximate KL is the final epoch's minibatch mean; clipping and entropy summarize update minibatches. Do not treat sparse logs as every-step measurements or infer causal effects from their relationship with validation return. Native/hybrid joint entropies have different components, and policy-space KL differs from executed trades after transforms, clipping and risk approval.

## Current finding and next research

All eight models have finite accepted states and expected updates. Late KL/clipping generally increase, including the successful one-seed posthoc control, so these logs alone do not explain deterioration. Cash/binary direction drift is documented separately in [participation diagnostics](participation-diagnostics.md). Any policy-update safeguard must be a distinct preregistered hypothesis with matched controls and actual Adam-step logging; no closed-family rescue, threshold grid or retrospective promotion. Inspect fitted-period versus development behavior before another training decision.

The CLI requires complete/cleaned accepted study state and published summary identity, checks original file hashes, restores all models read-only and rejects changes. Tests use synthetic saved state and never call optimizer.step or learn. See the linked audit for actual checks, CSVs, source hashes, reproduction command and limitations.
