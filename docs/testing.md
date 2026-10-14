# Tests and acceptance

Run the standard-library suite from the repository root with Python 3.13:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m compileall -q src scripts tests
PYTHONPATH=src .venv/bin/python scripts/offline_audit.py
```

The first command discovers both offline and dependency-backed tests. It reports explicit skips when NumPy/Gymnasium/torch/SB3/Plotly are absent. A skipped integration test is not a passing RL smoke. See [verification.md](verification.md) for the exact latest result.

## Offline coverage

| Area | Exercised behavior |
| --- | --- |
| Provider contracts | Independent unsupported-family rejection; deterministic synthetic seeds; CSV interchangeability |
| Data quality | UTC chronology, duplicates/order, OHLC/nonfinite failures, missing-session reports, mixed feed rejection |
| Storage | Idempotent immutable content, corruption detection and resumable chunk reuse |
| Features | Prefix/future-suffix causality, prior-only volume denominator, weekly prefix parity and effective splits |
| Supply/demand | Both directions, break priority, mitigation snapshot immutability, actual weekly impulse release, readiness and raw split continuity |
| Sectors/universe | Configurable roles, current-snapshot no-backdating, dated effective intervals, immutable metadata, momentum ranks, dividends, split/gap/late-bar masks and training-era label gate |
| Datasets | Global date partitions, 20-session purging, explicit unverified override and training-only means |
| Normalization | Frozen statistics; schema, nonfinite and ragged-matrix rejection |
| Accounting | Long/short P&L, costs, dividends, splits, quantity caps and repeated-target reconciliation |
| Execution | Invalid entry gaps, worse-price gap stops, stop-first race, both zone boundaries, short scale-out, reversals and holding limits |
| Risk/timing | Both price geometries, four finite actions, prior availability and split-adjusted targets |
| Reports | Real HTML/JSON fallback generation and comparison metadata including failures |
| Budget/recovery | Exact usage parser, idle/rate estimates, delayed measurement, cumulative retries, two-hour deadline, working-limit stop/no restart, existing-assignment refusal, metering failure and verified/failed teardown |

Tests use synthetic fixtures and mocked runtime behavior; they never create a session, fetch data, need credentials or send orders. Failed-cleanup tests deliberately print a mock recovery command to verify that such failures remain visible.

## Dependency-backed integration

`tests/test_integration.py` uses the actual Gymnasium/SB3/torch packages after installation:

- Gymnasium and SB3 checkers, seeded reset parity, correct time-limit truncation.
- Exactly 512 PPO and 1024 SAC/TD3 transitions on the same dataset/costs/seed.
- Validation reports/checkpoints, saved replay where applicable and deterministic save/load parity.
- Frozen normalization immutability, feature/feed/nonfinite rejection and bundle corruption detection.
- Default training does not create a final-test report.

Synthetic wiring checks do not establish market performance or provider coverage. Actual Yahoo coverage and CPU/A100 training are recorded in [verification](verification.md). Actual Alpaca account/feed/dividend audits remain separate acceptance steps.

## Before a later Colab run

Generate/review the transitive hash lock; run `pip check` and the local integration suite; inspect source coverage/actions/availability; verify a CPU smoke/profile; then use the owned budget-controlled runtime. A100 hardware, remote package compatibility, measured consumption, artifact retrieval and actual backend termination require a real remote run; mock lifecycle tests alone cannot verify them. See [verification.md](verification.md) for actual execution evidence.

When changing interfaces, add targeted tests for causal timing, quantity/risk gates, accounting or serialization behavior that can actually fail. Preserve user changes and report checks that ran, including skips and blockers. Do not add redundant tests solely to mirror a reversible presentation change.
