# Documentation index

The system includes local research, verified CPU/A100 training and a budget-controlled Colab backend. Completed checks and remote execution status are recorded in [verification.md](verification.md). `plan.md` retains the research roadmap; `plan.html` is preserved.

Read [architecture](architecture.md), [setup](setup.md), [data providers](providers.md), [datasets](datasets.md), then [simulation](simulation.md) and [training](training.md). Each guide owns one category; cross-links point to the category responsible for further detail.

| Category | Guide | What it covers |
| --- | --- | --- |
| Project operating guidance | [project-guidance.md](project-guidance.md) | Research requirements, agent workflow, runtime policies and primary references |
| System architecture | [architecture.md](architecture.md) | Ownership, data flow, boundaries and extension rules |
| Installation and dependencies | [setup.md](setup.md) | Python 3.13, pins, hash-lock generation and optional SDKs |
| Shared contracts | [contracts.md](contracts.md) | Canonical bars, capability types, feature/action schemas and diagnostics |
| Provider access | [providers.md](providers.md) | Synthetic, files, Yahoo, Alpaca, audit status and entitlements |
| Storage and accumulation | [storage.md](storage.md) | Immutable identities, source chunks, cursor recovery and revisions |
| Dataset construction | [datasets.md](datasets.md) | Coverage gates, splits, purging, warmup and normalization |
| Causal feature families | [features.md](features.md) | Indicator formulas, availability, splits and weekly context |
| Multi-timeframe supply/demand | [supply-demand.md](supply-demand.md) | Implemented D1/W1 base/impulse zones, lifecycle, availability, RL inputs and ablations |
| Stock universe and sector rotation | [sectors-universe.md](sectors-universe.md) | Configurable symbols, dated classifications, historical ETF rotation, scaling and commands |
| Historical universe coverage | [historical-universe.md](historical-universe.md) | Membership, delistings, provider audit and training-only decline coverage before expansion |
| Historical evidence import and audit | [historical-data.md](historical-data.md) | Stable IDs, causal membership events, delisting outcomes, immutable imports, listing discovery and coverage reports |
| Matched universe expansion | [universe-expansion.md](universe-expansion.md) | Fixed 22-versus-55-stock training comparison, common validation panel, no retired-test reuse, and Colab operations |
| Trade simulation | [simulation.md](simulation.md) | Cash/P&L, fills, costs, gaps, stops, scale-outs and corporate actions |
| Policy outputs and risk | [actions-risk.md](actions-risk.md) | Four-component decoding, geometry, desired quantities and clipping |
| Execution-state observations | [execution-observations.md](execution-observations.md) | Versioned held protection, masked controls, legacy compatibility and causal state |
| Gymnasium interface | [environment.md](environment.md) | Observations, seeded reset, step and episode boundaries |
| RL training and checkpoints | [training.md](training.md) | PPO/SAC/TD3 settings, seeds, selection and continuation |
| Matched research experiments | [experiments.md](experiments.md) | Four datasets, three algorithms/seeds, declared selection, held-out protocol and GitHub assets |
| Learning diagnostics and improvement | [learning-experiments.md](learning-experiments.md) | Trading behavior, compact checkpoint curves, controlled CPU comparisons and sequential learning gates |
| Matched 250k indicator follow-up | [indicator-experiments.md](indicator-experiments.md) | Combined, zones-only and sector-only comparisons against the completed 250k baseline |
| Algorithm stability and deterioration | [algorithm-stability.md](algorithm-stability.md) | Saved-path diagnosis and bounded PPO-versus-SAC CPU comparison |
| Research evaluation | [evaluation.md](evaluation.md) | Baselines, portfolio metrics and final-test isolation |
| Short-side performance | [short-side.md](short-side.md) | Reconciled long/short P&L, falling-stock reports and positive-return requirements |
| Autonomous improvement research | [research-loop.md](research-loop.md) | Bounded experiments, stock/ETF gates, token efficiency and prospective confirmation |
| PPO action initialization | [action-initialization.md](action-initialization.md) | Factorial exploration/protection screen, matched controls, promotion and bounded Colab operations |
| Causal training sampling | [training-sampling.md](training-sampling.md) | Training-only downtrend starts, readiness-matched controls, universe factorial study and actual transition balance |
| Visual and run reports | [reporting.md](reporting.md) | Plotly/SVG reports, trade diagnostics and seed comparisons |
| Frozen inference | [inference.md](inference.md) | Checksummed bundles, deterministic predictions and schema/staleness rejection |
| Forward shadow inference | [shadow-inference.md](shadow-inference.md) | Frozen campaign registration, durable decisions, simulated fills, restart recovery and forward reports |
| GitHub model releases | [releases.md](releases.md) | Downloadable models, derived reports, checksums and training provenance |
| Artifact retention and cleanup | [artifact-retention.md](artifact-retention.md) | Required research evidence, release verification, temporary-file cleanup and recovery state |
| Colab runtime operations | [colab.md](colab.md) | CPU/T4 preflight, A100 training, compute ledger, retrieval and recovery |
| Concurrent Colab runtimes | [colab-concurrency.md](colab-concurrency.md) | Configurable pool, shared account metering, owned teardown and recovery |
| Tests and acceptance | [testing.md](testing.md) | Offline checks, optional integration and future smoke acceptance |
| Current verification | [verification.md](verification.md) | Checks actually performed, artifacts, usage and remaining research |

No broker-order adapter is enabled. Advanced data families and extra reward terms remain gated behind separate matched experiments.
