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
| Longer-horizon price context | [trend-context.md](trend-context.md) | Causal63/252-session price returns, trailing-high drawdown and matched warmup/masked controls |
| Multi-timeframe supply/demand | [supply-demand.md](supply-demand.md) | Implemented D1/W1 base/impulse zones, lifecycle, availability, RL inputs and ablations |
| Stock universe and sector rotation | [sectors-universe.md](sectors-universe.md) | Configurable symbols, dated classifications, historical ETF rotation, scaling and commands |
| Historical training augmentation | [historical-augmentation.md](historical-augmentation.md) | Matched original/recent/earlier ETF arms, fixed windows and strict promotion gates |
| Historical market-regime coverage | [historical-regimes.md](historical-regimes.md) | Fixed chronological pre-2010 ETF source/action/coverage audit before training |
| Historical universe coverage | [historical-universe.md](historical-universe.md) | Membership, delistings, provider audit and training-only decline coverage before expansion |
| Historical evidence import and audit | [historical-data.md](historical-data.md) | Stable IDs, causal membership events, delisting outcomes, immutable imports, listing discovery and coverage reports |
| Matched universe expansion | [universe-expansion.md](universe-expansion.md) | Fixed 22-versus-55-stock training comparison, common validation panel, no retired-test reuse, and Colab operations |
| Trade simulation | [simulation.md](simulation.md) | Cash/P&L, fills, costs, gaps, stops, scale-outs and corporate actions |
| Policy outputs and risk | [actions-risk.md](actions-risk.md) | Four-component decoding, geometry, desired quantities and clipping |
| Execution-state observations | [execution-observations.md](execution-observations.md) | Versioned held protection, masked controls, legacy compatibility and causal state |
| Gymnasium interface | [environment.md](environment.md) | Observations, seeded reset, step and episode boundaries |
| PPO policy architecture | [policy-architecture.md](policy-architecture.md) | Market-only exposure, account-aware protection/critic, matched branch controls and frozen contracts |
| Reward objective research | [reward-objectives.md](reward-objectives.md) | Optional initial-capital equity changes, matched log control, discounting limits and locked Colab protocol |
| Random-forest signal feasibility | [forest-signal.md](forest-signal.md) | Locked tree dependencies, fixed comparison and portable non-pickle tree arrays |
| Supervised signal feasibility | [supervised-signal.md](supervised-signal.md) | Completed training-only targets, internal chronological probe, frozen preprocessing and gated warm-start research |
| Recurrent policy-update safeguard | [recurrent-update-guard.md](recurrent-update-guard.md) | Fixed KL guard versus strongest Gaussian control, actual Adam steps and gated Colab acceptance |
| Saved optimizer diagnostics | [optimizer-diagnostics.md](optimizer-diagnostics.md) | Actual Adam steps, nominal PPO epochs, finite moments and sparse KL/clipping interpretation |
| Numerical performance | [numerical-performance.md](numerical-performance.md) | Saved-weight subnormal counts and isolated forward timing without fitting |
| Recurrent cash availability | [recurrent-cash-choice.md](recurrent-cash-choice.md) | Fixed cash/binary/Gaussian controls, joint action likelihood and Colab acceptance gates |
| Recurrent sizing alignment | [recurrent-sizing-alignment.md](recurrent-sizing-alignment.md) | Fresh adaptive/posthoc/aligned execution controls, shared action contracts and frozen prediction |
| Fixed recurrent sizing replay | [recurrent-sizing.md](recurrent-sizing.md) | Sign-preserving unit exposure, unchanged protections, strict path parity and fixed diagnostic gates |
| Recurrent execution-state research | [recurrent-execution-state.md](recurrent-execution-state.md) | Fixed masked/full memory comparison, feed-forward control, sequence acceptance and gates |
| SAC discount horizon | [sac-discount.md](sac-discount.md) | Fixed0.99/0.95 comparison, unchanged execution and mandatory checkpoint identity |
| SAC regularization | [sac-regularization.md](sac-regularization.md) | Fixed actor/critic L2 protocol, actual optimizer acceptance and generalization gates |
| SAC training diversity | [sac-diversity.md](sac-diversity.md) | Cached expanded-universe audit, fixed preprocessing/group mass and bounded generalization comparison |
| Fitted-period policy diagnosis | [training-fit-diagnostic.md](training-fit-diagnostic.md) | Fixed saved-policy training-window replay, unchanged preprocessing and inference-only safeguards |
| TD3 execution-state research | [td3-execution-state.md](td3-execution-state.md) | Saved deterministic off-policy audit, fixed legacy/masked/full controls and strict gates |
| SAC initial-temperature research | [sac-temperature.md](sac-temperature.md) | Saved reward/Q/entropy diagnosis and fixed startup-temperature comparison |
| SAC execution-state research | [sac-execution-state.md](sac-execution-state.md) | Saved SAC input/participation audit and fixed legacy/masked/full off-policy comparison |
| RL training and checkpoints | [training.md](training.md) | PPO/SAC/TD3 settings, seeds, selection and continuation |
| Matched research experiments | [experiments.md](experiments.md) | Four datasets, three algorithms/seeds, declared selection, held-out protocol and GitHub assets |
| Learning diagnostics and improvement | [learning-experiments.md](learning-experiments.md) | Trading behavior, compact checkpoint curves, controlled CPU comparisons and sequential learning gates |
| Matched 250k indicator follow-up | [indicator-experiments.md](indicator-experiments.md) | Combined, zones-only and sector-only comparisons against the completed 250k baseline |
| Algorithm stability and deterioration | [algorithm-stability.md](algorithm-stability.md) | Saved-path diagnosis and bounded PPO-versus-SAC CPU comparison |
| Participation and cost diagnosis | [participation-diagnostics.md](participation-diagnostics.md) | Saved-path cost identity, close versus intraday exposure, resize costs and exit behavior |
| Research evaluation | [evaluation.md](evaluation.md) | Baselines, portfolio metrics and final-test isolation |
| Fixed-policy direction replay | [direction-replay.md](direction-replay.md) | Universal sign filters, counterfactual paths, immutable inputs and paired phase reuse |
| Short-side performance | [short-side.md](short-side.md) | Reconciled long/short P&L, falling-stock reports and positive-return requirements |
| Autonomous improvement research | [research-loop.md](research-loop.md) | Bounded experiments, stock/ETF gates, token efficiency and prospective confirmation |
| PPO action initialization | [action-initialization.md](action-initialization.md) | Factorial exploration/protection screen, matched controls, promotion and bounded Colab operations |
| Causal training sampling | [training-sampling.md](training-sampling.md) | Training-only downtrend starts, readiness-matched controls, universe factorial study and actual transition balance |
| Visual and run reports | [reporting.md](reporting.md) | Plotly/SVG reports, trade diagnostics and seed comparisons |
| Frozen seed ensembles | [ensembles.md](ensembles.md) | Uniform action averaging, shared state, explicit data bridge and immutable recipes |
| Frozen forecast centering | [signal-centering.md](signal-centering.md) | Causal complete-panel level anchoring, unchanged ranks and diagnostic-only gates |
| Frozen inference | [inference.md](inference.md) | Checksummed bundles, deterministic predictions and schema/staleness rejection |
| Forward shadow inference | [shadow-inference.md](shadow-inference.md) | Frozen campaign registration, durable decisions, simulated fills, restart recovery and forward reports |
| GitHub model releases | [releases.md](releases.md) | Downloadable models, derived reports, checksums and training provenance |
| Artifact retention and cleanup | [artifact-retention.md](artifact-retention.md) | Required research evidence, release verification, temporary-file cleanup and recovery state |
| Colab runtime operations | [colab.md](colab.md) | CPU/T4 preflight, A100 training, compute ledger, retrieval and recovery |
| Concurrent Colab runtimes | [colab-concurrency.md](colab-concurrency.md) | Configurable pool, shared account metering, owned teardown and recovery |
| Tests and acceptance | [testing.md](testing.md) | Offline checks, optional integration and future smoke acceptance |
| Current verification | [verification.md](verification.md) | Checks actually performed, artifacts, usage and remaining research |

No broker-order adapter is enabled. Advanced data families and extra reward terms remain gated behind separate matched experiments.

[Overnight/intraday P&L bridge](pnl-bridge.md) documents exact saved-fill
cash-flow identities, immutable-bar reconciliation and limitations.

[Completed-session decomposition](session-context.md) describes optional causal
overnight/intraday fields, same-dimension masks and pending research preflight.

[Current research results](../results/README.md) indexes recent comparisons,
diagnostics, pending training and artifact locations.

[Recurrent-capacity comparison](recurrent-capacity.md) specifies the fixed
64/128-width hypothesis, shared untrained RNG starts and resource gates.

[Training-history composition](training-history.md) documents date-only coverage,
full-window year expectations and prerequisites for a matched recency study.

[Recurrent PPO regularization](recurrent-regularization.md) specifies a distinct
fixed coupled-Adam coefficient comparison, preserved epsilon, controls and gates.

[Action-component crossover](action-crossover.md) documents fixed read-only
selected/final direction/protection mixing with independent recurrent state and
native-control reproduction requirements.

[Feature preprocessing](feature-preprocessing.md) owns frozen-scale distribution
audits and requirements for any optional, explicitly versioned input transform.

- [Gaussian action boundaries](action-boundaries.md): fixed training-prefix
  distribution audit, probability interpretation and bounded-policy prerequisites.
- [Compact broad-market context](market-context.md): fixed cached-data coverage,
  synchronized availability and causal SPY-relative feature preflight.
- [Causal filing data](filing-data.md): SEC source-audit scope, accession vintages,
  retrospective-field exclusion and prerequisites before financial features.

- [Distributional critic research review](distributional-critics.md): candidate
  TQC rationale, missing calibration evidence and requirements before a new study.

[Historical implied-volatility context](implied-volatility.md) covers the optional date-bounded VIX source, explicit lag assumptions, strict coverage gates and disabled model-input status.

[Saved recurrent value targets](recurrent-values.md) describes fixed training-only
stochastic replay, censoring/bootstrap limits, SB3-equivalent target checks and
read-only model/Adam/RNG preservation.

[Fixed benchmark variance](benchmark-variance.md) covers the read-only coefficient-one
reward decomposition, within-episode feasibility gate and limits on interpretation.

[Recurrent GAE traces](gae-traces.md) specifies the fixed lambda0.95/1 comparison,
finite-buffer return identities, unchanged controls and the completed failed screen.

[Saved GAE target variability](gae-variance.md) describes the fixed read-only
within-buffer trace comparison, endpoint decomposition and gradient-noise limits.

[Lagged Treasury context](treasury-context.md) owns the disabled two-coordinate
rate proposal, official annual XML source, timing/vintage assumptions and fixed
coverage/operation gates. Offline parser only; no yield download or model input yet.
