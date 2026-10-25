# Fixed benchmark variance audit

This follow-up reuses the52 saved stochastic training trajectories from the
[recurrent value audit](recurrent-values.md). It does not replay or optimize a
model. The question is whether subtracting one unit of the existing matched
buy-and-hold reward lowers observed reward/discounted-target variation enough
to justify proposing a separate experiment.

## Fixed method

The [protocol](../configs/experiments/benchmark_variance_audit.json) pins prior
summary/trace/protocol hashes, original immutable dataset and implementation
hashes. Use exactly the same26 training windows ending2020-12-02. The benchmark
buys whole shares at the next open, with unchanged capital, costs, splits and
dividends; retain cash dividends and marked equity at the cutoff. Benchmark
rewards use net log-equity changes, rounded to float32 like stored policy rewards.
No future data is a model input; future observed sums are reporting-only targets.

Coefficient is exactly one. Subtract rewards in float64 and reconstruct zero-tail
discounted sums at gamma0.99; primary target summaries use first256 decisions.
Verify linearity and `var(P-B)=var(P)+var(B)-2cov(P,B)`. Report each symbol,
equal-weight mean within-episode variances and pooled descriptive statistics.
The gate uses within-episode variation, avoiding apparent reductions caused only
by different symbol means.

Every selected/final × stock/ETF × reward/target cell must have residual variance
at most90% of original policy variance. All eight conditions must pass; zero
policy variance fails feasibility. There is no fitted coefficient, benchmark,
window or discount search. Failure closes this fixed subtraction hypothesis.
A pass is not economic improvement, model promotion or training authorization;
a distinct preregistered model experiment would still be required.

## Limits and operation

Observed variation is not irreducible noise or gradient variance. There is only
one stochastic trajectory per phase/symbol, overlapping targets are correlated,
and finite horizons are censored. Subtracting the same realized benchmark from
both a prediction and target leaves prediction error unchanged. Do not claim
policy invariance with action-dependent termination or learned bootstrap values.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python scripts/audit_benchmark_variance.py \
  --output artifacts/benchmark-variance-2026-10-25
```

The180-second worker validates complete trace/window alignment, preserves every
source/data/ledger hash, writes a progress journal and refuses existing outputs.
No training, replay, new provider request, development evaluation, retired-test
access, Colab allocation or broker orders. All prior failed families remain closed.

## Actual outcome — failed

[Completed evidence](../results/benchmark-variance-2026-10-25/README.md) records
only3/8 passing conditions. Selected stock/ETF reward ratios1.0159/1.4952 and
final ETF reward ratio1.1792 fail. ETF target ratios1.4020 selected/1.1167 final
also fail, despite stock target reductions. Independent cash/share accounting
reproduces benchmark rewards/equity exactly; covariance/discount/ratio checks
agree within1.12e−15. Worker exited0 in5.22s; all source/model/data/ledger hashes
unchanged. No optimization/replay/new units. The fixed subtraction hypothesis
is closed without a coefficient/benchmark/window/discount search or training.
