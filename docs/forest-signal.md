# Tabular random-forest signal feasibility

The fixed MLPs and their common-component transformation failed the predictive
gates. This new experiment changes the learner to a random forest while retaining
the same target payload, folds, feature arms and thresholds. It is a forecast
feasibility study; it does not implement a trading policy or relax the benchmark
goal.

## Fixed comparison

[The protocol](../configs/experiments/forest_signal.json) declares masked/full
10-feature forests with seeds 11/23/37, plus the same constant and ridge controls.
Each final forest has 128 trees, maximum depth 8, at least 50 observations per
leaf, 80% feature sampling, 80% bootstrap samples and one CPU worker. There is
no probe-based parameter search or checkpoint selection. Two discarded prefix
trees precede 768 full-fit trees, for 770 fitted trees in total. This is a tree
work count, not PPO transitions or optimizer updates.

The learner is stochastic through bootstrap/feature sampling. The three seeds
are three forest fits, not 384 independent experiments. Paired feature arms use
the same settings and seed; realized tree structures may differ with information.
Both arms must retain the fit-only preprocessing and unchanged completed-label
payload from [supervised signal preparation](supervised-signal.md).

The constant, ridge, MSE, yearly, IC and passing-seed criteria are unchanged.
Forecast accuracy cannot establish profitable shorts, stock/ETF outperformance,
or permission to transfer these models into a policy. Any subsequent trading
experiment needs its own matched four-action/protection contract and prospective
evidence. Failure stops this forest configuration.

## Dependency and Python preflight

The latest inspected Colab default is Python 3.13.16. The local environment uses
that version, and remote bootstrap must check the actual current kernel before
synchronizing dependencies; a changed interpreter requires project alignment,
not installation of an older Python.

The optional `trees` group pins scikit-learn 1.9.1 and SciPy 1.17.1. Hash-lock
resolution adds joblib 1.6.0 and threadpoolctl 3.7.0 while retaining every existing
package version. Official PyPI metadata provides CPython 3.13 Linux x86_64 wheels
and compatible NumPy requirements. Wheel-only, hash-verified local installation,
imports, core dependency checks and `uv pip check` pass.

Sources: [scikit-learn installation](https://scikit-learn.org/stable/install.html),
[RandomForestRegressor API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html),
[scikit-learn release files](https://pypi.org/project/scikit-learn/1.9.1/),
[SciPy release files](https://pypi.org/project/scipy/1.17.1/).

`requirements.lock` contains the full research environment, including optional
provider, development and tree packages. `requirements.txt` remains the smaller
core list. To reproduce the reviewed full environment with the existing uv:

```bash
uv pip install --python .venv/bin/python --require-hashes \
  --only-binary :all: -r requirements.lock
PYTHONPATH=src .venv/bin/python scripts/preflight_forest_signal.py \
  --output artifacts/forest-signal/preflight.json
```

The lock script now includes `--extra trees`. In this agent environment the uv
Snap launcher rejected its transient scope, and its default cache was outside
the writable roots. The existing binary worked with `RL_UV_BIN` and a workspace
`UV_CACHE_DIR`; no second uv or Colab CLI was installed or upgraded. This is a
packaging/cache setting, unrelated to Python-version compatibility.

## Portable tree representation

`inference/forest_arrays.py` exports numerical tree arrays to NPZ without pickle
or an estimator dependency for prediction. It checks field sets, integer node
indices, finite values, unique reachable nodes, leaf structure, maximum depth
and a bounded tree count. Inputs must be finite 10-field matrices; conversion
to float32 matches sklearn's prediction input convention. Leaf values and forest
averaging remain float64. NaN routing is deliberately unsupported because the
research feature contract rejects missing/nonfinite observations.

Tests use explicit tree fixtures, not locally fitted forests, to verify threshold
ordering, equal averaging, serialization parity, malformed/cyclic nodes,
incorrect file sets and invalid inputs. Actual sklearn-to-array parity must still
pass in the Colab prefix before full fits. A complete forest bundle, fitting
worker and strict metered job/artifact integration are the next implementation
stage; their acceptance is not implied by these array tests.

## Current evidence

[Preflight](../results/forest-signal-2026-10-20/preflight.json) confirms Python,
locked dependencies, immutable folds, constructor settings and unfitted state.
There are 45,266 fit and 18,616 probe rows. No forest model has been fitted,
no new Colab allocation occurred, and no performance has been measured.

264 non-training regression tests and three focused forest checks pass. The
existing training integration suite was excluded to keep fitting in Colab.
Project usage remains 35.861240/150 units. The planned one-CPU run remains capped
at one additional unit, shares the original ledger and enforces the two-hour
allocation deadline and 130-unit project working stop.

## Implemented Colab worker and restoration

Version14 binds the prepared signal identity and complete protocol hash. The
worker fits only on the completed pre-2018 fold, with one CPU thread, checks two
prefix trees before full fitting, and saves a durable checkpoint after each of
six complete forests. Paired feature arms record identical per-tree seed
fingerprints. Scores use only final 128-tree forests; fit MSE is diagnostic and
does not select checkpoints. Work metadata explicitly records 770 fitted trees.

`ForestSignalBundle` shares the strict name/source/frozen-preprocessing contract
with neural forecasts and restores six numerical forests, ridge and constant
controls. Its manifest includes the dependency lock and actual package versions.
The archive verifier recomputes all MSE/IC/year metrics and gate decisions from
returned predictions, checks complete model/date/symbol panels, counts and seed
fingerprints, and verifies the dependency-manifest hash.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_signal_jobs.py --forest \
  --output .runtime/forest-feasibility-jobs.json
# Run through the existing metered fleet after committed-source export.
PYTHONPATH=src .venv/bin/python scripts/verify_signal_runs.py \
  artifacts/colab/OWNED_SESSION/retrieved/artifacts \
  --output artifacts/forest-signal/restoration.json
```

Post-download verification compares every restored forest forecast with the
recorded sklearn prediction to absolute tolerance 1e-10, requires exact local
repeat predictions, checks immutable targets/sessions/symbols and fit-only
normalization, and rejects incompatible schema/source/nonfinite inputs.

42 focused forest, existing neural, target, fleet, budget and artifact checks pass
without local fitting. The actual Colab fit is next; this is wiring evidence,
not predictive performance or a successful trading model.
