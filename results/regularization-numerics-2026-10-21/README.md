# Retained SAC numerical/performance diagnosis

No fitting, backward pass, optimizer update, dataset access, Colab allocation or
retired-test evaluation. The complete control100k and incomplete candidate80k
models are the exact checksummed artifacts of the interrupted regularization
study. This diagnostic cannot complete or promote that study.

| Saved model | Policy tensor elements | Subnormal values | Forward normal arithmetic | Forward flush mode | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control100k | 29,644 | 0 | 0.447ms | 0.438ms | 1.02× |
| Regularization80k | 29,644 | 5,942 | 6.266ms | 0.444ms | 14.10× |

Same local CPU, one PyTorch thread, fixed synthetic256×21 float32 batch. Each
mode runs10 warmups then3×100 actor+critic+target forward passes with gradients
disabled. Ratios are this host's measurements, not Colab training benchmarks.
No timing threshold is used as a test or model-selection criterion.

Actor/critic optimizer moments also contain subnormals:control328/1766 versus
candidate4230/3413. They were counted only, not updated. Both modes produce
bit-identical actor/critic/target outputs on this batch; maximum difference0.
This is finite-batch evidence, not a guarantee over all inputs or backpropagation.
Policy tensor hashes and optimizer update counters remain unchanged, as do all
source files. Process-local arithmetic changes are discarded at process exit;
no model or training configuration is modified/exported.

Saved remote elapsed intervals include learning, evaluation and snapshot export.
Control intervals are about104–108 seconds per10k after its first checkpoint;
candidate50k→60k/60k→70k/70k→80k take486/494/485 seconds. Different Colab hosts
confound cross-arm comparisons. Same-host forward results support subnormal
handling as a possible contributor to slowdown, not proof of its entire training
cost or the cause of poor returns. An optimizer/backward benchmark remains
unperformed; all fitting still requires Colab.

[Exact timings, counts, curves and source identities](summary.json),
[script](../../scripts/diagnose_regularization_numerics.py),
[interrupted outcome](../sac-regularization-2026-10-20/README.md).
Eleven numerical/construction/checkpoint tests pass without local fitting.
Project conservative37.787213/150 and stop flag unchanged; all owned sessions
remain stopped. This is not authorization to restart after the metering stop.

Next, if further allocation is explicitly authorized, profile any proposed CPU
arithmetic change in a separate bounded matched numerical preflight before
adopting it. Preserve original failed/incomplete return gates and use new source
identity; never silently retrofit old results or assume speed improves returns.
