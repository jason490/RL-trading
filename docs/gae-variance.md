# Saved GAE target variability

The [fixed audit](../configs/experiments/gae_variance_audit.json) follows the
completed failed lambda0.95/1 study. It uses all100 actual pre-update1000-step
training buffers from each seed11 arm. It does not replay models, read market
prices or use development outcomes to choose a new setting.

Within each stored buffer, recompute both original traces with identical rewards,
critic values, reset boundaries and endpoint. Report population variances and
paired lambda1/lambda0.95 ratios; zero denominators remain explicit nulls. Fixed
all/first50/last50 blocks are descriptive. A ratio above1 describes greater raw
target variation; it is not a financial or statistical promotion gate.

Separately decompose lambda1 returns into discounted raw rewards and the endpoint
critic contribution, plus the explicit stored float32 reward-adjustment rounding
residual. Keeping that residual separate avoids mislabeling representation error
as critic dependence. True terminal tails are zero; timeout values and rollout-end
values remain explicit. This identity retains gamma0.99 and does not create an
unbiased full-horizon target. It does not establish critic calibration.

SB3 normalizes advantages inside recurrent minibatches, whose actual shuffled
sequence order and masks are not retained by these scalar arrays. This audit
therefore does not estimate actual normalized gradient noise or explain policy
return causally. Different source policies induce different reward/value paths;
only the two recomputations on each individual buffer are path-paired.

Source summary bytes, every actual NPZ and JSONL record must match the retained
accepted hashes. Original arrays/math are revalidated; files and ledger remain
unchanged. No optimization, new Colab allocation, provider request, retired-test
access or new GAE hyperparameter search. Implementation and synthetic arithmetic
checks precede the actual audit; no variance result is claimed yet.


Run after source acceptance:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/audit_gae_variance.py \
  --output results/gae-variance-2026-10-25/summary.json
```

Synthetic checks verify closed-form timeout/terminal/end-of-rollout identities,
unchanged input arrays, one-step TD, null denominator handling and population
variance aggregation. Protocol changes and running/incomplete sources reject
before array access. Existing rollout and financial-gate regressions remain.
Actual audit is now complete; see the accepted results below.


## Actual result

[Completed audit](../results/gae-variance-2026-10-25/README.md): median within-buffer
advantage variance ratios4.408252 control/4.018140 candidate; return ratios2.779849/
2.422366. Raw advantage variation rises in every buffer. Both arms have195 timeouts
and no true terminations, retaining endpoint dependence. Independent matrix
verification agrees within1.05e−17 variance/1.60e−14 ratio error across all200
buffers. All221 protected source files/ledger unchanged; no new units or training.
These raw-target results do not prove normalized gradient noise or a cause of
poor return. The failed trace family remains closed.
