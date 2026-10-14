# Reports and run comparisons

`reporting/html.py` writes self-contained HTML and a neighboring machine-readable JSON file. Plotly is used when installed. Otherwise `reporting/offline.py` produces a clearly labelled native SVG report using only the standard library; hover points show values and expandable tables expose fills/decisions.

## Single-run contents

- Policy equity and cost-aware buy-and-hold equity.
- Underwater/drawdown curve with the same initial capital.
- Daily prices, stop/near/far levels and learned zone width.
- Fill markers and reasons in Plotly; complete fill tables in the SVG audit.
- Marked exposure, cumulative cost and turnover metrics.
- Holding-session and exit-fill P&L distributions.
- Raw/clipped decisions, execution approvals/rejections and accounting diagnostics.
- Validation checkpoint score versus transitions, labelled as selection data.

Every view names its partition and source context. Synthetic reports say that no trained policy or market evidence is present. Final-test reports are audits of frozen validation selection. Current reports do not represent paper/live fills.

HTML embeds Plotly JavaScript when available rather than relying on a third-party CDN at viewing time. JSON contains rows, fill events, metrics and baselines; it can support later custom analyses without scraping graphs. PNG/PDF automation is not implemented; use browser print/export for the current standalone HTML or add an audited plotting exporter when publication-ready static artifacts are required.

## Comparisons

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local compare artifacts/runs --output artifacts/comparison.html
```

The comparison reads only run metadata under `runs/*/metadata.json`. It groups by dataset, simulation config, requested transition budget, smoke/long mode and selection metric. Per algorithm/group it lists completed/failed runs, seeds, mean selected validation score and sample standard deviation if more than one completed score exists. A single seed has no variability estimate. All run metadata remains visible, including failures.

Repeated runs of the same seed are visible entries rather than independent evidence; review the seed list before interpreting variability. Device and actual step counts remain in metadata. Unmatched groups are separate; the report does not silently combine different datasets/costs/budgets. Comparison does not select weights or open the final test.

The offline audit script produces a reproducible validation-segment accounting report with a fixed alternating exposure fixture. Its purpose is to validate storage/features/simulation/report generation before dependencies or remote jobs are available. It is not a substitute for the PPO/SAC/TD3 acceptance suite.
