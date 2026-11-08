# Capability overview

This guide summarizes implementation scope. [Verification](verification.md)
records evidence and limitations; the [research results index](../results/README.md)
records actual experiments. Implemented software does not establish profitable
trading. No robust multi-seed/prospective winner is established and broker-order
execution remains disabled.

| Area | Scope and guide | Status and limitations |
| --- | --- | --- |
| Data access | [Providers](providers.md), [storage](storage.md) | Synthetic, file, Yahoo and Alpaca adapters; immutable provenance and resumable collection. Feed, dividend, calendar and historical coverage need separate audits. |
| Causal datasets | [Features](features.md), [datasets](datasets.md) | Daily indicators, optional completed weekly context, chronological splits, purging and frozen training-only normalization. |
| Simulation | [Simulation](simulation.md), [actions and risk](actions-risk.md) | Long/short accounting, next-session fills, costs, stops, learned target zones, adjustments and reversals; modeled fills are not broker execution evidence. |
| Learning | [Training](training.md), [recurrent comparison](recurrent-sac-comparison.md) | PPO/SAC/TD3 and separately controlled recurrent research, checkpoints and save/load checks. Experimental algorithms require their own admission. |
| Diagnostics | [Reporting](reporting.md), [evaluation](evaluation.md), [short side](short-side.md) | Equity/baseline curves, drawdowns, costs, action/exit diagnostics and matched comparisons. Development-selected results are not untouched tests. |
| Runtime | [Colab](colab.md), [concurrency](colab-concurrency.md) | Metered CPU/GPU jobs, cumulative shared budget, bounded retrieval, persisted ownership and verified cleanup. CPU is preferred for current small networks. |
| Frozen inference | [Inference](inference.md), [reference arithmetic](trpo-reference-inference.md) | Versioned/checksummed research bundles, immutable normalization and schema checks. Numerical acceptance is algorithm/contract specific. |
| Optional inputs | [Supply/demand](supply-demand.md), [sectors](sectors-universe.md) | Implemented causal feature families with completed negative as well as positive intermediate results; inputs are not universally enabled. |
| Wider stock data | [Latest acquisition](../results/stock-context-expansion-2026-11-08/README.md) | Configured132-stock universe plus ETFs;22 latest stock histories downloaded. Current labels and surviving-stock samples do not prove historical membership. |
| Cross-stock context | [Relationships](cross-stock-relationships.md), [MOVE/VIX](macro-index-context.md) | Optional aligned correlation/relative-momentum component and retained index history. New inputs remain detached from the current policy pending admission. |
| Historical coverage | [Historical data](historical-data.md) | Import/audit infrastructure; representative historical membership, dead-stock outcomes and episode integration remain separate gates. |
| Shadow and broker trading | [Shadow inference](shadow-inference.md) | Persistent simulated forward-decision infrastructure; frozen campaign activation is paused. Alpaca paper-order lifecycle remains pending. |

## Not yet established

- Consistent selected/final multi-seed performance comparable to matched stock/ETF holding, followed by prospective confirmation.
- Historical sector/membership and delisting coverage sufficient for broad population claims.
- Current policy inputs for [earnings](earnings-context.md) or a successful [auxiliary forecast model](auxiliary-model-inputs.md).
- TRPO learned-policy/profile and full-path acceptance: engineering components are being built; no TRPO performance result is claimed.
- Broker submission, protective-order reconciliation and operational paper/live execution acceptance.

See the [plan](../plan.md) and individual category guides before interpreting an
implemented component as an enabled feature or a completed research milestone.
