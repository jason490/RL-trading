# Saved cash-study optimizer audit

Read-only restoration of all eight selected/final models from the [completed study](../recurrent-cash-2026-10-22/README.md). Original model and metadata hashes must match the retained publication inventory; accepted summary must match the verified GitHub asset. All source files remain unchanged. No fitting, optimizer steps, dataset opening, provider calls, Colab allocation or retired-test evaluation.

## Actual updates and numerical state

All models own each trainable parameter exactly once. Every trained optimizer has21 parameter states; weights and Adam first/second moments are finite with valid shapes and nonnegative second moments. Saved settings have no KL stopping (`target_kl=None`). Every final100k model has20,000 Adam minibatch steps for every parameter, matching100 rollouts ×20 minibatches ×10 epochs.

| Arm | Selected checkpoint | Selected nominal epoch counter | Selected Adam steps | Final nominal epoch counter | Final Adam steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Posthoc Gaussian unit | 100k post-update | 1,000 | 20,000 | 1,000 | 20,000 |
| Aligned Gaussian unit | 70k on-step | 690 | 13,800 | 1,000 | 20,000 |
| Binary categorical | 40k on-step | 390 | 7,800 | 1,000 | 20,000 |
| Cash categorical | 10k on-step | 90 | 1,800 | 1,000 | 20,000 |

SB3's `_n_updates` is a nominal epoch counter, not individual Adam steps. The installed sb3-contrib2.7.0 increments it by configured `n_epochs` after each training call; if KL stopping is enabled, it can overcount epochs actually executed. This study has no KL stopping, so the above expected counts hold. On-step checkpoints precede the current rollout update, explaining90/390/690 rather than100/400/700. The final post-update checkpoint includes all updates.

## Saved policy-change diagnostics

| Arm | KL at10k | KL at final | Clip fraction at10k | Clip fraction at final |
| --- | ---: | ---: | ---: | ---: |
| Posthoc Gaussian | 0.0095 | 0.0377 | 11.12% | 29.22% |
| Aligned Gaussian | 0.0109 | 0.0408 | 10.61% | 32.10% |
| Binary | 0.0075 | 0.0385 | 10.95% | 28.11% |
| Cash | 0.0081 | 0.0353 | 8.23% | 27.50% |

Each point logs only the most recent rollout, at10k spacing. Approximate reverse KL is the last epoch's mean minibatch estimate; clip fraction and entropy loss average the update minibatches. On-step logs precede that checkpoint's pending update; final logs follow the final update. They do not provide a full trajectory of every optimizer step, a guaranteed KL bound, or a causal correlation test.

These are policy-space changes, not changes in the final executed trade distribution. Gaussian unit execution transforms its first output; protections are clipped/risk approved. Joint entropy includes different distributions/components across native and hybrid policies, so it cannot be directly ranked as categorical direction uncertainty. Learned Gaussian deviations remain positive (final means about0.71–0.75), not collapsed to zero.

Both failed categorical policies show higher late KL/clipping, but the posthoc control also does and finishes at32.80%. The observed policy drift and excess losses therefore cannot be attributed to KL magnitude alone. No saved-state corruption or unexpected update count was found; sparse diagnostics do not rule out every numerical/optimization issue. A policy-update safeguard is a possible separate hypothesis, not an established cure or permission to rescue the failed cash family. Inspect fitted-period/generalization behavior before deciding the next bounded intervention.

## Evidence and reproduction

[summary.json](summary.json) records source hashes, actual settings, per-parameter Adam steps, finite moment checks and40 compact training points. [optimizer-counters.csv](optimizer-counters.csv) and [training-diagnostics.csv](training-diagnostics.csv) support comparison/graphs.

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_cash_optimizer.py \
 --output /tmp/cash-optimizer-new.json
```

The output path must be new; retained local study files and private hash inventory are required. Twelve focused helper checks cover counter semantics, no input mutation, nonfinite/malformed/foreign optimizer states, missing ownership and invalid training logs. Actual restoration verifies all eight models and all recorded source hashes. No model is promoted. Project38.309006 and last official account349.56 remain; all12 study sessions are stopped.
