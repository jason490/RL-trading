# Documentation index

The system includes local research, verified CPU training and a budget-controlled Colab backend. Completed checks and remote execution status are recorded in [verification.md](verification.md). `plan.md` retains the research roadmap; `plan.html` is preserved.

Read [architecture](architecture.md), [setup](setup.md), [data providers](providers.md), [datasets](datasets.md), then [simulation](simulation.md) and [training](training.md). Each guide owns one category; cross-links point to the category responsible for further detail.

| Category | Guide | What it covers |
| --- | --- | --- |
| System architecture | [architecture.md](architecture.md) | Ownership, data flow, boundaries and extension rules |
| Installation and dependencies | [setup.md](setup.md) | Python 3.13, pins, hash-lock generation and optional SDKs |
| Shared contracts | [contracts.md](contracts.md) | Canonical bars, capability types, feature/action schemas and diagnostics |
| Provider access | [providers.md](providers.md) | Synthetic, files, Yahoo, Alpaca, audit status and entitlements |
| Storage and accumulation | [storage.md](storage.md) | Immutable identities, source chunks, cursor recovery and revisions |
| Dataset construction | [datasets.md](datasets.md) | Coverage gates, splits, purging, warmup and normalization |
| Causal feature families | [features.md](features.md) | Indicator formulas, availability, splits and weekly context |
| Multi-timeframe supply/demand | [supply-demand.md](supply-demand.md) | Planned base/impulse zones, lifecycle, availability, RL inputs and ablations |
| Trade simulation | [simulation.md](simulation.md) | Cash/P&L, fills, costs, gaps, stops, scale-outs and corporate actions |
| Policy outputs and risk | [actions-risk.md](actions-risk.md) | Four-component decoding, geometry, desired quantities and clipping |
| Gymnasium interface | [environment.md](environment.md) | Observations, seeded reset, step and episode boundaries |
| RL training and checkpoints | [training.md](training.md) | PPO/SAC/TD3 settings, seeds, selection and continuation |
| Research evaluation | [evaluation.md](evaluation.md) | Baselines, portfolio metrics and final-test isolation |
| Visual and run reports | [reporting.md](reporting.md) | Plotly/SVG reports, trade diagnostics and seed comparisons |
| Frozen inference | [inference.md](inference.md) | Checksummed bundles, deterministic predictions and schema/staleness rejection |
| Colab runtime operations | [colab.md](colab.md) | CPU/T4 preflight, A100 training, compute ledger, retrieval and recovery |
| Tests and acceptance | [testing.md](testing.md) | Offline checks, optional integration and future smoke acceptance |
| Current verification | [verification.md](verification.md) | Checks actually performed, artifacts and pending acceptance |

No broker-order adapter is enabled. Advanced data families and extra reward terms remain gated behind separate matched experiments.
