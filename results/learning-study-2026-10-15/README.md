# Controlled CPU learning study

The four-stage protocol completed 15 SAC runs (seeds 11/23/37), 1.95 million
transitions, with fixed datasets/costs and validation-only checkpoint selection.
No stage passed the preregistered risk gate. No new test audit or broker orders ran.

| Stage / inputs | Reward scale | Steps per seed | Mean selected return | Mean final return |
|---|---:|---:|---:|---:|
| Baseline | 1 | 100k | 19.70% | 16.50% |
| Combined zones + sector context | 1 | 100k | 11.78% | 0.94% |
| Baseline | 10 | 100k | 11.46% | 7.38% |
| Baseline | 100 | 100k | 3.98% | −0.99% |
| Baseline | 1 | 250k | **23.96%** | **14.46%** |

Buy-and-hold averaged 30.60%; the constraint-matched passive-long control returned
26.37%. Returns are net of modeled costs in separate stock episodes. Selected
models use the same validation window that selects them; these are not new
out-of-sample claims. The 250k stage improved selected checkpoints by 4.26 points
versus 100k, but final returns declined and drawdown worsened 3.54 points versus
50k anchors, above the two-point limit. Do not change the gate after this result.

## Evidence by category

- [Protocol](../../configs/experiments/learning_study.json), [long-stage decision](long-stage-decision.json), [complete summary](summary.json).
- [Initial behavior diagnostics](behavior.json) and [passive-long control](constrained-long.json).
- [100k baseline/combined](cpu-100k/progress.json), [scale 10](scale-10/progress.json), [scale 100](scale-100/progress.json), [250k](long-250k/progress.json): per-seed checkpoint CSVs and aggregate CSVs sit beside each progress file.
- [100k per-stock comparison](cpu-100k/per-stock-comparison.csv) and [250k comparison](long-250k/per-stock-comparison.csv), averaging three selected seeds per stock.
- [Frozen bundle verification](bundle-verification.json): all 15 passed exact CPU prediction parity, unchanged normalization and schema/shape rejection.
- [Operating guide](../../docs/learning-experiments.md) and [verification](../../docs/verification.md).

## Graphs and models

Local interactive graphs are under `artifacts/public/learning-study-2026-10-15/`.
Each stage has a `STAGE-curves.html` report; `long-250k-curves.html` shows the final
three-seed curves. Individual runs retain JSONL/CSV/HTML curves every 10k steps.
Records distinguish current returns, selected returns and incremental changes;
changes are performance differences on the same window, not additive profits.

Public assets are published with the
[learning-study release](https://github.com/jason490/RL-trading/releases/tag/learning-study-2026-10-15):
frozen models, best/latest weights, explicit milestone weights, standalone graphs,
validation equity reports, compact research evidence and SHA256SUMS. No raw bars,
replay buffers or private runtime/authentication logs are included.

The project ledger remains cumulative: 34.032604 conservative units used out of
150; 115.967396 remain. This series added approximately 0.941842 units. All owned
runtimes are stopped. Detailed usage and artifact cleanup evidence accompany the
release and results. The earlier 36-run study and retired final-test audit remain
unchanged. No model is paper-eligible.
