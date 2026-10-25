# Modular swing-trading RL project plan

Status: Python 3.13 aligned with Colab; 101 tests pass. Updated 2026-10-15. All 36 expanded-universe 50,000-transition runs and the frozen three-seed held-out audit are complete. Combined SAC averaged 10.83% on test versus buy-and-hold 48.46%; no benchmark advantage demonstrated. Models/reports are published, all owned runtimes stopped, cumulative conservative usage 33.090762/150 units, and 7.82 GiB recovered through verified cleanup. CPU is now the default per the user-directed hardware change.

## 1. Intended outcome and starting assumptions

Build a provider-independent research pipeline that accumulates market and company data, generates causal features, produces chronological training/validation/test datasets, trains interchangeable PPO/SAC/TD3 agents, and exports frozen models for inference against live market data and Alpaca paper trading.

The final model proposes position size, long/short direction, stop loss, a learned take-profit zone, and position adjustments. Execution and risk controls validate those proposals before sending orders. Add features and reward terms gradually, measuring each addition against the previous version.

Initial planning defaults, configurable before implementation:

- US equities and swing trading; begin with daily decisions and completed daily bars, then add completed weekly context. Track holding periods in sessions. A research target of roughly 2–20 sessions is an observation goal, not a rule that blocks protective exits.
- Generate a signal after the completed session is available; simulate entry/adjustment at the next regular-session open with costs. Paper inference uses the same timing and accounts for feed publication delays. Protective orders can operate between daily decisions.
- Start with a small declared universe and one traded symbol per episode. Share the policy across symbols with separate episodes; add simultaneous portfolio allocation as a separate environment version with explicit allocation constraints.
- Begin with price/volume and a simple reward. Add learned stops/targets before the paper-execution milestone; introduce shorting after long-only accounting works.
- Offline research, then live-data shadow inference, then Alpaca paper orders. Real-money deployment is a separate future task. Feature growth is implemented incrementally, with acceptance recorded below.

## 2. Proposed folder and module ownership

This is the ownership map; some advanced paths remain planned, while implemented modules are recorded below. Each category owns its code, configuration, and tests; provider-specific types stay inside adapters.

```text
AGENTS.md
plan.md
pyproject.toml                      # package metadata and dependency management
src/rl_stocks/
  core/                            # shared typed contracts, timestamps, identifiers
  data/
    providers/                     # common provider interfaces/capabilities
      alpaca/
      yahoo/
      sec_edgar/
      alpha_vantage/
      twelve_data/
      tradier/
      gdelt/
      files/                       # CSV/Parquet imports independent of an API
    accumulation/                  # backfill, incremental sync, snapshot schedules
    validation/                    # schema, coverage, quality, provenance
    storage/                       # immutable raw and versioned normalized datasets
    universe/                      # symbol identity, calendars, historical membership
    generation/                    # splits, feature assembly, manifests, fixtures
  features/
    registry/                      # dependencies, versions, availability contracts
    price/                         # returns, volatility, ranges
    trend/                         # EMA and related features
    momentum/                      # RSI and related features
    volume/                        # relative volume, VWAP, volume-profile features
    timeframes/                    # session-aware resampling and causal alignment
    market_structure/              # causal D1/W1 supply/demand zones
    sectors/                       # effective-dated classification and context
    fundamentals/                  # filing-derived financial features
    options/                       # OI concentrations and wall proxies
    order_flow/                    # trade/quote-derived proxies; depth if available
    sentiment/                     # news/entity mapping and timestamped scores
  environment/
    trading/                       # Gymnasium Env and wrappers
    accounting/                    # equity, cash, holdings, borrowing, dividends
    execution/                     # simulated orders, stops, target-zone fills
    rewards/                       # independently configurable reward components
  rl/
    registry/                      # common model factory
    ppo/
    sac/
    td3/
  training/                        # runner, callbacks, checkpointing, seeds
  evaluation/                      # validation, held-out test, baselines, metrics
  inference/                       # frozen bundle loading and decision decoding
  risk/                            # limits and validated executable trade intents
  execution/
    interfaces/                    # broker-independent order lifecycle
    alpaca/                        # paper adapter, fills, reconciliation
    orchestration/                 # scheduling, recovery, decision idempotency
  reporting/                       # plots, saved HTML reports, experiment comparison
  runtime/
    local/
    colab/                         # CLI packaging, launch, recovery, teardown
configs/
  providers/ features/ datasets/ environments/ rewards/
  algorithms/ training/ evaluation/ risk/ execution/ runtime/
tests/
  data/ features/ environment/ rl/ training/
  evaluation/ inference/ risk/ execution/ reporting/
notebooks/                         # thin exploratory views
scripts/                           # thin CLI entry points
data/                              # ignored generated storage
  raw/ normalized/ features/ datasets/ manifests/
artifacts/                         # ignored outputs, unique run IDs
  runs/ frozen/ paper/
docs/                              # schemas, provider audits, experiment decisions
```

Do not create every provider or feature module immediately. Add each implementation when its milestone needs it. Keep dependencies optional where possible so a price-only experiment does not require sentiment/options packages.

Planned flow: provider adapters → accumulator → normalized records → causal feature registry → dataset generator → environment → RL factory/training → validation/reporting → frozen bundle → inference → risk → paper broker adapter.

## 3. Modular data accumulator and dataset generator

### Contracts and storage

Define separate record types for bars, trades, quotes, company classifications, filings/facts, options chains, news, and corporate actions. Provider capabilities describe supported record types, date ranges, granularity, feeds, delays, free-tier quotas, licensing, and point-in-time guarantees. Unsupported requests fail clearly rather than returning a silently substituted feed.

Every record includes symbol/instrument ID, provider, source event time, public availability time, retrieval time, revision/as-of identity, and applicable currency/units. For initial live snapshots, availability is no earlier than actual retrieval. Historical adapters may reconstruct earlier availability only from documented publication/filing metadata; uncertain timestamps require a conservative lag or exclusion. Record uncertainty in the manifest.

Accumulate immutable raw responses plus checksums. Normalize into partitioned Parquet, cache locally, and query with DuckDB or an equivalent light engine. Incremental jobs use resumable cursors, pagination, deduplication, bounded retries, and provider-specific quota budgets. Revised data creates a new version rather than mutating an experiment's inputs.

The generator builds aligned feature matrices, validity/age masks, chronological splits, fitted preprocessing, and a dataset manifest. Synthetic data generators produce deterministic accounting/leakage fixtures and stress scenarios; synthetic profits are not evidence of a market edge. Training episodes may be sampled within the training split without mixing chronological split membership.

### Provider selection and free-data limitations

Selections below are for a small personal research prototype. Recheck actual account access, historical coverage, and terms with a small sample before implementing each adapter. Free access to a web page does not guarantee a stable API or redistribution rights.

| Category | Initial selection or candidate | Planned use and limitation |
| --- | --- | --- |
| Recent bars, quotes/trades, paper execution | Alpaca | Official client and consistent broker integration. Explicitly select feeds: free live IEX is one exchange, not consolidated SIP volume. Historical feed access differs from real-time access. Verify earliest usable dates per record type; an older Alpaca support page documents the 2016 limit. [Feed documentation](https://docs.alpaca.markets/us/docs/market-data-faq), [historical coverage note](https://alpaca.markets/support/alpaca-data-timeline). |
| Longer daily equity history | Yahoo Finance through `yfinance`, plus local imports | First free research adapter to sample for pre-2016 daily bars and corporate actions. Actual history is symbol-dependent; intraday backfill is limited. This is an unofficial personal-use tool, not a production data SLA. [Project](https://github.com/ranaroussi/yfinance), [history API](https://ranaroussi.github.io/yfinance/reference/yfinance.functions.html). |
| Sector/industry metadata | Yahoo/`yfinance`; SEC SIC fallback | Choose Yahoo's conventional sector/industry fields for small-universe current research. Archive dated snapshots; current classifications must not be inserted into old backtests. SEC SIC provides an authoritative industry fallback, with an explicit project mapping to broad sectors, not an assertion of GICS equivalence. Historical features require historical filing evidence or are omitted. [Sector API](https://ranaroussi.github.io/yfinance/reference/yfinance.sector_industry.html), [SEC SIC list](https://www.sec.gov/search-filings/standard-industrial-classification-sic-code-list). |
| Financial statements | SEC EDGAR submissions and Company Facts | Preferred free authoritative US filings source, with no API key. Preserve original filings, accession IDs, units, and public filing times; never join by fiscal period end alone. [SEC APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces). |
| Options call/put concentrations | Yahoo/`yfinance` current chains, accumulated forward | First free prototype candidate for contract strike/expiry/type/open-interest snapshots, subject to sample completeness and source delays. Does not provide a verified long historical chain archive. Persist source dates and first observed time; don't backfill today's OI into past decisions. [Chain interface](https://ranaroussi.github.io/yfinance/reference/yfinance.ticker_tickers.html). |
| Alternative options source | Tradier | Evaluate account-eligible delayed sandbox chains as an alternative, checking OI fields and access before adopting. Sandbox equity/options data is delayed and Greeks are unavailable there; real-time access requires a brokerage account. This does not solve historical chain coverage. [Market-data access](https://docs.tradier.com/docs/market-data), [chains](https://docs.tradier.com/reference/brokerage-api-markets-get-options-chains). |
| Supplemental indicators/data | Alpha Vantage | Keep an optional adapter, not the default bulk archive. Free quota is 25 requests/day; historical options, company overview, and news sentiment are currently marked premium. Do not assume its advertised long history is free for the needed endpoint. Compute EMA/RSI locally. [API documentation](https://www.alphavantage.co/documentation/), [quota](https://www.alphavantage.co/premium/). |
| Additional bar/indicator samples | Twelve Data; Stooq file imports | Twelve Data Basic advertises 8 credits/minute and 800/day; verify endpoint entitlements and actual stock history. Stooq is a daily-download candidate awaiting a coverage/adjustment/terms audit, not a selected live provider. [Twelve Data plans](https://twelvedata.com/pricing), [Stooq downloads](https://stooq.com/db/h/). |
| News/sentiment | GDELT archives, then dated issuer/news feeds | Free candidate for news/entity activity and tone; GDELT provides downloadable update files. Validate symbol mapping, first-seen timestamps, coverage, and duplicate stories. Tone is not automatically a reliable financial sentiment signal. [GDELT data](https://gdeltproject.org/data.html). |
| Order flow and price-level volume | Alpaca trades/quotes where entitled; imported datasets | Compute explicitly labelled trade/quote proxies. Bars alone cannot recover true order flow, aggressor identity, or exact volume-at-price. Complete equity depth/history is not assumed free. Defer depth features until a usable source exists. [Alpaca historical API](https://docs.alpaca.markets/us/docs/historical-api). |

No verified free source in this research establishes a complete, long, point-in-time options-chain/OI archive or equity order-book archive. These are conditional later milestones: accumulate forward, accept an audited user-supplied archive, or revisit paid data separately. Optional data must not shorten the baseline dataset silently.

Never splice incompatible price/volume series into one experiment without an audited harmonization policy. Keep execution prices and corporate actions consistent; do not treat today's dividend-adjusted historical prices as raw executable prices. Cross-provider experiments record their feed differences, especially relative volume and flow. For initial paper deployment, either train with matching Alpaca features/feed or pass an explicit cross-feed drift evaluation.

## 4. Preventing lookahead bias

- Use availability-time backward as-of joins: a feature used at decision time must have been observable by that time. Store bar start/end separately, including publication delay.
- Daily indicators use completed sessions. Weekly/monthly features become available only after those periods close; Monday's signal cannot use Friday's future close. Intraday resampling respects exchange sessions, holidays, and DST. No backfilling incomplete higher-timeframe indicators.
- EMA, RSI, ATR, returns, and rolling volume use trailing windows. Swing pivots that need future bars become available only after confirmation; avoid centered rolling windows and future peak/trough labels as observations.
- Financial features use original filing availability and the latest eligible vintage at that decision. Later amendments/restatements and today's Company Facts summary must not overwrite what the agent knew then. Resolve period duration and duplicate concepts/units explicitly.
- Options OI uses its own published as-of date and availability, not quote time. News uses first-seen/publication times with lag and revision handling. Archive sector classifications and universe membership; current winners/constituents are not a historical universe.
- Split globally by calendar time across all symbols, initially approximately 60/20/20 subject to enough history. Training-only preprocessing; freeze categorical vocabulary and handle unknown sectors. Validation/test may use earlier historical bars only as causal indicator warmup, never future outcomes or fitted data.
- Define a conservative purge at split ends for positions/labels whose outcome would cross into the next split; use a declared research maximum holding horizon for this purpose. Keep portfolio states and replay buffers separate. Audit every split's observable times and episode bounds.
- Repeated checkpoint selection uses validation and designated walk-forward folds. Unlock the final test only after features, reward, and selection rules are frozen. If test results influence redesign, retire that test as a development interval and reserve a new holdout.

## 5. Feature rollout and ablation experiments

Each feature has name/version, inputs, parameters, units, timeframe, warmup, availability lag, missing-value policy, and incremental/batch parity checks. One family is introduced per experiment; snapshot feature schema changes require retraining and a new model bundle.

| Order | Feature family | First experiment and caveat |
| --- | --- | --- |
| A | Price/volume and portfolio state | Lagged returns, trailing volatility/range, current exposure, cash, holding age, unrealized P&L. Small initial input set. |
| B | EMA, RSI, ATR, relative volume | Try EMA 20/50, RSI 14, ATR 14, and completed daily volume divided by prior 20-session mean. Exclude the current session from the denominator. Indicators are hypotheses, not guaranteed predictive variables. |
| C | Multiple timeframes | Add completed weekly trend/volatility first; monthly or intraday context later. Same underlying causal feature functions. |
| C.1 | Multi-timeframe supply/demand | Base consolidation plus directional impulse, causal zone lifecycle, ATR-scaled distances/widths, freshness and age; D1/W1 first, M1 later. Opt-in D1/W1 implemented and tested in matched 50k/250k studies; no robust advantage established. M1 remains gated. |
| D | Market/sector context | Benchmark/sector returns and relative strength; introduce sector categories only where point-in-time metadata is supportable. |
| E | Financials | A small filing-derived family such as revenue growth and leverage, joined on public availability. No current analyst forecasts in historical observations. |
| F | Price-level volume | Trailing volume profiles from actual historical trade prices/sizes where available; expose point of control/value area with causal window boundaries. Bar-derived profiles are explicitly approximations. |
| G | Sentiment | Timestamped news count/tone first, then an independently versioned sentiment model if useful. Filter duplicated/entity-ambiguous stories and measure latency. |
| H | Options | OI/volume concentrations by strike/expiry, put/call ratios, distance to candidate walls. Validate data coverage before training; keep missing/age masks. |
| I | Order flow | Quote imbalance and signed-trade estimates, clearly labelled as estimates. Full-depth features require a new source/capability audit. |

Define a call/put wall initially as a reproducible OI concentration proxy: aggregate separately by type and a configured expiry bucket, find selected high-OI strikes, and express their distance from spot. Save aggregation/tie rules and stale-data limits. OI concentrations do not reveal dealer positioning or guarantee support/resistance; gamma-weighted variants need additional justified data and assumptions.

Compare each addition against the same baseline on matched periods/seeds/costs. Optional snapshots with short histories get a separate matched-period study and cannot be credited for improvements caused merely by a different test window.

### Multi-timeframe supply/demand input

Use the reviewed [tradingAlgo supply/demand implementation](https://github.com/jason490/tradingAlgo/blob/b78d578c0b03433e7ddcb4f0fe64658cf9eaebae/screener/indicators/supply_demand.py)
as the detection reference: short consolidation bases, directional impulses and
subsequent zone lifecycle. The full specification is in
[docs/supply-demand.md](docs/supply-demand.md). Compute each symbol/timeframe
independently on completed D1 and W1 bars, with M1 a separate later study.
Real intraday history is required before adding H1/H4; daily candles cannot supply it.

Emit state only as of each daily decision. Full-history lifecycle resolution must
not remove earlier zones because they break later, or assign final freshness to
past rows. Use completed-period availability, deterministic live-zone ranking,
presence/readiness masks, daily-ATR-scaled geometry and training-only frozen
preprocessing. Start without overlap merging; a later merge variant must update
geometry and preserve causal component provenance. Volume/RSI confirmations and
cross-timeframe confluence are separate ablations with recorded definitions.

Create a new feature schema/dataset/bundle and compare D1, W1 and D1+W1 against
the same baseline periods, costs, seeds 11/23/37 and 50,000-transition budgets.
The indicator supplies context to the policy; it does not replace learned stops
or take-profit zones. D1/W1 dataset inputs and sector-rotation context are implemented; the current
trained model remains the daily-indicators baseline, and final-test data remains reserved.

## 6. Environment and learned trade outputs

### Action and decision contracts

Use a continuous action design supported by all three algorithms. First implement a position-only baseline; the final action schema is a normalized four-component `Box`:

| Component | Decoded meaning |
| --- | --- |
| Signed target exposure | Negative = short, zero = flat, positive = long; magnitude specifies requested equity allocation. |
| Stop distance | Positive bounded volatility-relative distance; produces a stop below a long or above a short reference price. |
| Take-profit near distance | Learned positive distance to the near boundary of the profit zone, above a long or below a short reference price. |
| Take-profit zone width | Learned positive width beyond the near boundary; defines the far boundary. |

The network determines the target zone; no hardcoded reward-to-risk ratio substitutes for that prediction. Use causal ATR or a declared volatility scale to decode distances, with configurable bounds, minimum ticks, and rounding. Save both raw policy outputs and validated outputs. Warmup/invalid ATR prevents a trade. Actual ATR bounds and exposure limits are experiment settings, not proven optimal values.

Export a typed decision with symbol, decision/expiry timestamps, model/schema IDs, reference price, requested/approved exposure, share/notional amount, long/short/flat direction, stop price, target-zone lower/upper prices, and resulting add/reduce/close/reverse/hold intent. Size is calculated from known equity and prices, then capped by risk/buying-power/liquidity constraints. Repeated unchanged target exposure must not add another position.

Keep existing protections anchored to the existing entry unless a model decision explicitly requests a valid update. Define a deadband/minimum order size to avoid tiny adjustments. Record any rejected or clipped proposal. Direction reversals close and reconcile the existing position before opening the other direction; protective exits override holding-period preferences.

### Making a zone affect outcomes

Start learned exits with stop plus one learned target; then promote to the four-component zone schema. For zone experiments, use a fixed, configurable scale-out schedule such as half at the near boundary and half at the far boundary, with protection maintained on remaining shares. Both boundaries must influence outcomes so the network has a reason to learn zone width. Whole-share rounding and small-position fallback are explicit. Any future partial-exit allocations are a separate action-schema change.

Simulate fills, gaps, spread/slippage, commissions/fees, borrow costs, short availability, overnight exposure, dividends, and corporate actions. The next open is a future fill outcome, never an input to yesterday's decision. Stops/targets are decoded from the recorded decision reference; if a gap invalidates their geometry before entry, follow a declared reject/skip policy shared by simulation and paper execution.

Use finer bars/trades for execution replay when available while retaining daily decisions. If one OHLC bar hits a stop and target with unknown order, use a declared conservative stop-first rule and report sensitivity to that assumption. Do not claim stop prices guarantee fill prices during gaps. Episode termination/truncation and final valuation must be documented and checker-compatible.

## 7. Reward design

Begin with per-step log change in equity after modeled execution/holding costs. Positive and negative P&L are already rewarded/punished; do not double-count a second realized-trade profit reward. Invalid/ruined equity requires explicit terminal handling rather than an undefined log. Account equity is the source of truth, with a decomposed reward log.

| Stage | Candidate addition | Reason and control |
| --- | --- | --- |
| R0 | Net equity return only | Establish accounting and a transparent baseline. |
| R1 | Modest turnover penalty | Discourage excess adjustment if real costs alone do not. Penalize traded notional, not every inference call; record how much exceeds actual costs. |
| R2 | Drawdown/downside-exposure term | Test one at a time for large adverse positions. Avoid summing realized/unrealized losses twice or repeatedly penalizing the same unchanged drawdown. |
| R3 | Holding-frequency preference | Consider only after holding-duration and turnover reports show a need. Holding age belongs in state; swing behavior can emerge from daily cadence, costs, and trading constraints. No reward for merely staying in a losing trade. |
| Diagnostic | Win/loss ratio, profit factor, expectancy | Report initially, not optimize directly. High win rate can hide large losses; trade-count ratios are sparse, unstable with few trades, and easy to game. Any later reward experiment must show improvement in net risk-adjusted results. |

Fix hard position/order/exposure limits outside reward shaping; a learned penalty is not a substitute for enforcement. Freeze reward coefficients before the final test and record every change as a new experiment.

## 8. RL defaults and tuning

Start with SB3 `MlpPolicy`, a normalized flat feature vector, and independent algorithm configurations. The following are proposed baseline settings, not trading-specific optimal hyperparameters. Verify against the pinned stable release; current reference docs include development versions.

| Setting | PPO | SAC | TD3 |
| --- | --- | --- | --- |
| Learning rate | 3e-4 | 3e-4 | 3e-4 (project starting choice) |
| Discount gamma | 0.99 | 0.99 | 0.99 |
| Initial environments | 1, then profile vectorization | 1 | 1 |
| Batch size | 64 | 256 | 256 |
| Rollout/replay | n_steps=2048 | buffer_size=100000 | buffer_size=100000 |
| Updates | n_epochs=10 | train_freq=1, gradient_steps=1 | train_freq=1, gradient_steps=1 |
| Other defaults | gae_lambda=0.95, clip_range=0.2, ent_coef=0 | learning_starts=1000, tau=0.005, ent_coef=auto | learning_starts=1000, tau=0.005, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5 |
| Exploration | stochastic policy | learned entropy | Gaussian action noise, starting sigma=0.1 in normalized units; assess each component |

References: [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html), [TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html). Use small networks initially, with architecture recorded separately for each algorithm. Reward/observation scales, daily holding horizon, and buffer memory determine whether defaults are sensible.

Run a tiny CPU wiring check before longer experiments, then a declared starting budget such as 50k transitions per algorithm/seed. Use seeds 11/23/37 initially. More simulated timesteps replay the same historical evidence; they do not add new independent market observations. Increase budgets only with validation evidence and resource estimates. PPO rollout/batch sizes must be compatible; off-policy warmup must finish before assessing learning.

Tune on training/validation walk-forward folds with recorded search budgets, covering learning rate, discount, capacity, and algorithm-specific settings. Features/reward changes are separate experiments. Match data and environment-step budgets, also report wall time and compute, and avoid selecting a winner from one lucky seed. Consider Optuna only after the fixed-default baseline is reproducible.

## 9. Checkpoint evaluation and visual reports

At each configured checkpoint (initially every 10k environment transitions), evaluate a separate validation environment with frozen preprocessing and deterministic inference. Account for vectorization when scheduling callbacks. Save latest, periodic, and best-by-validation checkpoints; do not repeatedly run the final test during training. A custom selector should use declared portfolio metrics, not automatically treat SB3's highest shaped reward as the best strategy. [SB3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html).

Create saved Plotly HTML reports plus PNG/PDF exports where needed, backed by machine-readable tables:

- Validation performance versus training steps, with all checkpoints/seeds and the selected checkpoint marked.
- Equity, benchmark equity, underwater/drawdown curves, net returns and risk metrics.
- Price charts with long/short entries, adjustments, exits, stop levels, and learned target-zone bands; track changes through time.
- Holding-time and trade-return distributions, win rate plus average win/loss and expectancy, turnover/cost breakdown, exposure and rejected decisions.
- Algorithm/config/feature/reward comparisons on matched windows, with variability across seeds and market regimes.
- Later paper reports comparing expected and actual fills/latency/feature drift.

Reports must label training, validation, held-out test, and paper results distinctly, and show weak/failed runs as well as the winner. The final test is an unbiased audit of an already selected frozen candidate; it does not select a new best model.

## 10. Saving and freezing the model

Freeze a deployable inference bundle containing model weights, fitted preprocessing/normalization state, exact feature/action/reward versions, symbol universe and provider/feed assumptions, resolved configs, dependency lock, data/split manifests, validation-selection rule/results, and checksums. Attach final test results as a versioned report without retuning the selected weights. Mark failures as candidates that are not eligible for paper execution.

Inference never calls `learn()`, updates normalization, or appends to a training replay buffer. Use deterministic predictions for the first paper protocol. Verify save/load decision parity on fixed observations, reject schema/provider mismatches, and record the immutable bundle ID with each signal. Store resumable training checkpoints separately from deployment bundles. [SB3 normalization](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html).

## 11. Live-data evaluation and Alpaca paper execution

Use a persistent local/server inference process; Colab is the training backend and is not assumed to provide unattended execution uptime. Consume live market data through the same feature code, honoring delays and completed-bar boundaries. Shadow mode records decisions without orders and checks historical/incremental feature parity, scheduling, latency, and missing/stale data behavior.

Then use Alpaca's explicit paper mode and verify the endpoint/account. Query tradability, shortability/borrow eligibility, buying power, tick/quantity restrictions, and supported order combinations. Shorting is conditional on actual account and asset eligibility. Never assume arbitrary fractional short orders or short-borrow history is available. [Margin/short selling](https://docs.alpaca.markets/us/docs/margin-and-short-selling).

Use broker-held protection where supported. Alpaca brackets have a single take-profit leg and do not support extended hours; a two-boundary scale-out zone needs a separately verified lifecycle design. Do not pretend a normal bracket implements two profit targets. Test stop coverage, partial fills, leg quantity updates, replacement/cancellation races, and rejection recovery before enabling zone execution. On adjustments, reconcile positions and outstanding exits before changing quantities. [Alpaca order capabilities](https://docs.alpaca.markets/us/docs/orders-at-alpaca).

Persist decision/order IDs, desired and actual positions, executions, and bundle IDs. Use idempotent decision processing, reconnect/backfill handling, bounded retries, and reconcile uncertain order submissions before resubmission. On restart reconstruct state from broker positions/open orders plus durable records. Stop new exposure on stale data, schema mismatch, failed reconciliation, or configured risk-limit breaches while maintaining existing protection.

Predeclare position/exposure/turnover/loss limits and paper evaluation criteria before a run. Report net returns, drawdowns, holding duration, trade count, costs, operational errors, and drift over enough time/trades to evaluate the protocol; there is no universal duration that proves profitability. Paper fills omit some real market effects. [Alpaca paper limitations](https://docs.alpaca.markets/us/docs/paper-trading).

## 12. Colab, efficiency, and reproducibility

Use the official Google Colab CLI with pinned version and checked subcommand help. Package/upload or install the project and its config/data explicitly. Record runtime hardware and dependencies; begin with CPU and profile before requesting GPU. Persist periodic checkpoints and export artifacts/logs before releasing only the sessions created for the job. [Official CLI](https://github.com/googlecolab/google-colab-cli), [runtime limits](https://research.google.com/colaboratory/faq.html).

Precompute causal features once per dataset version, use arrays in the environment loop, query partitioned storage for research, and batch/cache provider calls within quotas. Benchmark single-environment versus vectorized runs rather than assuming subprocesses help. Bound replay memory from actual observation/action dimensions; tune collection/update ratios with environment count. Keep secrets out of scripts, CLI arguments, outputs, and bundles.

## 13. Implementation milestones and completion criteria

1. **Contracts and provider audit:** Implement minimal package/config boundaries, capability manifests, file import and synthetic fixtures. Sample Alpaca and Yahoo bar history including pre-2016 requests; record actual access, coverage, adjustments, and quotas. Complete when a provider can be replaced without changing the environment and gaps/unsupported types are explicit.
2. **Accumulator and dataset baseline:** Add immutable storage, incremental recovery, quality checks, global splits, simple causal features, and train-only preprocessing. Complete when a manifest reproduces all three datasets and leakage checks cover timing, periods, revisions, and splits. Start optional snapshot archives when their source/access is verified.
3. **Position-only environment:** Implement cash/long accounting, next-session fills, costs, Gymnasium contract, and cash/buy-and-hold baselines. Then add short accounting and borrow assumptions. Complete when deterministic fixtures validate cash/P&L/constraints and environment checkers pass.
4. **Learned trade management:** Add stop/single-target actions, then the full learned zone and position-adjustment schema, partial exits, gaps, and conservative execution replay. Complete when both directions, changes, reversals, and stop/target races behave deterministically and constraints are enforced.
5. **Modular training and reporting:** Add PPO, then SAC/TD3 with fixed defaults, checkpoint validation, save/load checks, and visual reports. Complete when each algorithm runs the same versioned experiment and reports all checkpoints and seeds.
6. **Controlled feature/reward growth:** Introduce one family or reward term per study, keeping accepted changes only with matched validation evidence. Financials, options, sentiment, and flow each have separate data-readiness gates. Complete each study with an ablation report and a recorded decision, including negative results.
7. **Select, freeze, and audit:** Predeclare selection criteria, select with validation, export the immutable bundle, and evaluate once on held-out test. Complete when inference is reproducible and any failure is reported without test-driven retuning.
8. **Live-data shadow evaluation:** Run the frozen bundle with Alpaca market data, record decisions and drift, and exercise reconnect/restart scenarios. Complete when feature timing, state recovery, and policy outputs match the specified protocol.
9. **Alpaca paper execution:** Enable eligible long/short and learned stop/target orders with tested reconciliation and protection handling. Complete when the predeclared operational and performance criteria are reported, including divergence from the simulator. This milestone does not authorize live-money deployment.

Milestones 1–5 have baseline implementations and initial training acceptance as recorded below. Next research compares matched algorithm/seed runs and introduces causal supply/demand inputs through milestone 6. Before further remote jobs, recheck provider entitlements and the cumulative compute budget. All choices remain configurable; no stage is presented as evidence of expected profit.


## 14. Implementation status — 2026-10-15

The local system follows the ownership map above. Detailed category documentation is indexed in [docs/README.md](docs/README.md); checks actually performed are recorded in [docs/verification.md](docs/verification.md). `plan.html` is preserved.

| Milestone | Implemented work | Remaining acceptance |
| --- | --- | --- |
| 1. Contracts/providers | Typed capabilities; canonical bars; synthetic/CSV/Yahoo/Alpaca adapters | Yahoo coverage verified; Alpaca endpoint/feed/dividend audit remains |
| 2. Accumulation/datasets | Resumable source chunks, immutable hashes, coverage gates, causal features, splits/purge, frozen normalization | Completed for Yahoo; optional provider audits remain |
| 3. Accounting/environment | Long/short cash and costs, next-open simulation, Gymnasium adapter, baselines | CPU Gymnasium/SB3 checks pass; execution sensitivity studies remain |
| 4. Learned trade management | Four-component policy zone, adjustments/reversals, gap/stop ordering, both target boundaries, corporate actions | Neural-policy integration passes; execution sensitivity studies remain |
| 5. Training/reporting | PPO/SAC/TD3 factory, checkpoint validation, run metadata, interactive HTML/JSON and seed/failure comparisons | CPU/A100 acceptance and all 36 multi-seed research cells complete; indexed reports and model assets published |
| 6. Feature/reward growth | Required daily families and optional completed weekly context; net-equity reward only | D1/W1, configurable 22-stock/11-sector expansion and causal rotation trained in the four-dataset study; mixed CPU/CUDA confound disclosed; further ablations remain |
| 7. Freeze/audit | Checksummed research bundle and immutable deterministic inference API; optional post-freeze test path | All 36 bundles pass frozen CPU prediction parity; selected-family three-seed held-out audit complete and underperforms buy-and-hold; paper-ineligible |
| 8. Shadow inference | Shared causal feature/decision contracts and staleness gates | Persistent live-data orchestration/recovery/drift study |
| 9. Paper execution | Intentionally deferred; no broker-order implementation | Future explicit milestone with tested lifecycle and protection |

Remote smoke scripts preserve the 150-unit ceiling, 130-unit working limit, two-hour initial allocation cap, cumulative metering ledger, durable retrieval and verified cleanup. Colab attempts and cleanup are recorded in docs/verification.md. Longer experiment configurations are 50,000 transitions and seeds 11/23/37; an opt-in research job follows successful smoke acceptance. The user now requires matching Colab’s default Python (currently 3.13.15/3.13.16 across inspected runtime images), superseding the original Python 3.12 specification.

Final selection was committed in `179b358` before the held-out audit. The
[results](results/sector-study-2026-10-14/README.md) and
[verification](docs/verification.md) record all seeds, unsuccessful outcomes,
source/hardware caveats, release hashes, compute and cleanup. No final-test
result selected a checkpoint, seed or feature set. Future research should
control CPU hardware, profile sequential evaluation, test turnover/cost changes
separately and reserve a fresh audit interval. Historical membership/sector
vintages precede scaling; shadow inference and broker orders remain later work.
The immutable `plan.html` is unchanged.

## 15. Sequential learning study — completed, 2026-10-15

The preregistered four-stage CPU SAC protocol completed 15 cells and 1.95 million
training transitions: baseline/combined at 100k; baseline reward scales 10 and
100 at 100k; then baseline with the original reward at 250k, seeds 11/23/37.
Compact 10k-step curves and explicit post-update milestone weights were retained.

Selected mean validation return improved from 19.70% to 23.96%; final mean fell
from 16.50% to 14.46%. The 250k stage failed the unchanged gate because average
maximum drawdown worsened 3.54 percentage points versus the paired 50k anchors
(limit 2). Buy-and-hold was 30.60%; the constraint-matched passive-long control
was 26.37%. No stage passed and no paper candidate is approved. The bounded
protocol is exhausted; do not extend its search or change its gate after results.

All 15 frozen bundles passed exact CPU prediction parity, immutable normalization
and schema rejection checks. Full acceptance passed 115 tests. Colab runtimes
were stopped and verified; the cumulative project ledger reached 34.032604 units.
See [results](results/learning-study-2026-10-15/README.md) and
[learning experiments](docs/learning-experiments.md) for evidence and artifacts.

Next proposed work is a separate matched universe experiment covering historical
membership, declining stocks and verified delisting returns, chosen without
future knowledge. Keep the reward fixed first; test turnover controls or reward
penalties independently. The inspected final-test interval remains retired.
Persistent shadow inference and paper-order lifecycle work remain later milestones.

## 16. Matched indicator follow-up — completed, 2026-10-16

The user requested the missing combined-input 250k comparison and next steps.
The [new protocol](configs/experiments/indicator_followup.json) declares combined,
supply/demand-only and rotation-only CPU SAC families at 250k, seeds 11/23/37,
original reward and unchanged data/costs. The completed basic-input 250k family
is the control, subject to clean learning-source/dependency proofs. The previous
exhausted study remains unchanged. Compact 10k-step curves and durable retrieval
continue under the existing cumulative project budget. Feature benefit is assessed
by a separate declared paired gate; no new final-test audit or orders occur.
Historical-universe coverage and churn experiments follow attribution as separate
work. See [indicator experiments](docs/indicator-experiments.md).

The nine indicator cells finished at 250k each. Selected/final means were
46.82%/−1.73% combined, 18.39%/11.83% zones-only and 30.78%/−11.30% sector-only,
versus 23.96%/14.46% basic controls. All paired feature gates failed; no additional
steps or seeds were added to this protocol. All 12 control/new bundles passed
frozen checks. Project usage reached 35.098226 units and all owned sessions stopped.

## Earlier algorithm recommendations — superseded by the declared comparison below

After the indicator follow-up, test alternatives to SAC in a separate bounded
study. Strong selected checkpoints followed by deterioration motivate this
comparison, but do not establish that SAC or value overestimation causes the
instability. No alternative is assumed to improve trading performance.

1. **TQC first:** Compare Truncated Quantile Critics with SAC on the same fixed
   input dataset. TQC builds on SAC and truncates high critic quantiles to control
   value overestimation. This is a hypothesis to test, not a diagnosis of the
   current runs. See the [official implementation documentation](https://sb3-contrib.readthedocs.io/en/v2.8.0/modules/tqc.html).
2. **PPO with recent observations next:** Compare ordinary PPO with a fixed,
   predeclared window of stacked causal observations. Keep an unstacked PPO
   control so changes from algorithm choice and added history can be separated.
3. **Recurrent PPO afterward:** Compare an LSTM policy with the PPO controls to
   test whether learned memory adds value beyond existing indicators and stacked
   observations. Reset recurrent state at episode/symbol boundaries and never
   carry history across data splits. Account for its additional training and
   frozen-inference complexity. See [Recurrent PPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html).

Keep net-equity reward, universe, feature family, dates, preprocessing, action
contract, execution rules and costs fixed within each comparison. Use seeds
11/23/37, equal transition budgets and checkpoint-evaluation opportunities, and
record wall time, compute units, hardware and algorithm-specific update settings.
Predeclare configurations, search limits, chronological validation folds and
acceptance criteria before training; include selected and final checkpoints,
drawdown, costs, turnover and seed variability. Retain cash, buy-and-hold and the
constraint-matched passive-long controls. Do not reuse the retired final-test
interval or change gates after seeing results.

Before implementation, verify SB3-Contrib compatibility with Colab's actual
default Python and the dependency lock. Any execution protocol must fit the
remaining cumulative project budget and existing metering/cleanup limits. This
recommendation adds no jobs to the running indicator study and does not initiate
training. Historical-universe coverage remains a separate data-readiness task;
reward changes must be tested independently. Guidance on multiple seeds and
evaluation is in [SB3's experiment guide](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).

## 17. Diagnose deterioration and compare PPO variants — completed, 2026-10-16

After the last indicator cell, diagnose selected/final saved paths and run a
separate nine-cell PPO/RecurrentPPO/SAC CPU stability comparison if the evidence supports PPO.
The saved outputs show large directional/exposure changes whose P&L impact exceeds
additional costs; these observations do not prove an optimization cause. The
[protocol](configs/experiments/algorithm_stability.json) fixes basic inputs, the
original reward, 100k transitions, seeds 11/23/37 and fresh paired SAC controls.
No new final-test evaluation, extra reward, hyperparameter search or stock
selection occurs. Finish and publish the indicator study and clean its temporary
duplicates before allocating new sessions. See [guide](docs/algorithm-stability.md).


Concurrent execution is now implemented with a configurable runtime pool, isolated
payloads and one aggregate account meter. The user explicitly removed the old
one-session rule and fixed two-session/CPU-only limits; hardware remains chosen
by profiling and cost. Local acceptance passes 144 tests, including four mixed-
hardware mocked concurrent cells. Run the three-model CPU concurrent smoke
preflight before the nine 100k algorithm/memory comparison cells. All credit
usage, retries and cleanup share the preserved project ledger.

The actual concurrent preflight accepted all three algorithms after a reviewed
transport-timeout fix; full local acceptance passed 146 tests. All nine research
cells then completed 100k transitions on Colab CPU, with peak concurrency three.
Selected/final validation means were SAC 19.70%/16.50%, PPO 18.69%/16.90%, and
RecurrentPPO 13.27%/6.13%, versus buy-and-hold 30.60%. All declared gates failed.
PPO narrowed the selected-to-final gap but did not provide the required return
gains; the recurrent configuration deteriorated more. No additional training or
test selection was added. All nine downloaded bundles passed exact frozen
sequence parity, normalization and schema rejection checks. Every owned session
stopped and cleanup was verified; conservative cumulative usage is 35.389379
units. See [results](results/algorithm-stability-2026-10-16/README.md).

Next work should establish historical membership and delisting-return coverage
before a separately declared universe experiment. A matched turnover intervention
is another separate hypothesis, especially for PPO seed 37; it should not be
treated as a general explanation for all deterioration. TQC and stacked PPO
remain unexecuted ideas, not queued jobs. Persistent shadow inference and paper
execution remain later milestones. The inspected final-test interval is retired.

## 18. Persistent frozen shadow tracking — implemented; feed activation blocked, 2026-10-16

The user authorized the next step: freeze the promising validation checkpoint and
observe future behavior without more training. A dedicated local/server runtime
now registers exact combined-SAC seeds 11/23/37 plus basic PPO/SAC controls,
freezes costs, data/features and a 126-session research-review horizon, and scores
only bars closing after registration. The retired 2024–September 2026 interval is
warmup-only and cannot become a fresh test.

The SQLite journal atomically persists all arm/symbol positions, pending actions,
protective orders, fills and recurrent memory. Next-session simulated execution
requires a decision journaled before the matching open. Restarts are idempotent;
missing sessions, history revisions, stale data and feed/calendar mismatches halt
for review. Read-only collection preserves immutable receipts; interactive
reports compare separate stock accounts with buy-and-hold and cash. There is no
broker connection or automatic paper eligibility. See
[shadow operations](docs/shadow-inference.md).

Software acceptance and registration are distinct from forward market evidence.
Actual unattended uptime depends on a persistent host/process; the service template
does not install or enable itself. Historical membership/delisting expansion,
borrow eligibility and order lifecycle remain unimplemented later milestones.

Full acceptance passed 156 tests; all nine actual bundles passed feature/ATR and
training-observation parity for 26 symbols. The fixed campaign was registered
from clean source and its initial waiting/report cycle passed. Fresh Yahoo data
failed OHLC geometry on October 16 SPY, confirmed by a fresh retry. Automatic
polling remains disabled; zero forward decisions/outcomes exist. Resolve the
upstream bar or audit an explicit data correction before activation, without
silently relaxing validation. The first eligible close is October 19; do not
backdate a missed prospective start. See [actual status](results/forward-shadow-2026-10-16/status.json).

The user then explicitly approved correcting the insignificant October 16 SPY
high/open inconsistency. The opt-in exception pins the exact source OHLC checksum,
provider/feed/date/symbol and maximum $0.011 increase; raw evidence and a receipt
audit remain immutable. Real preflight passes all 37 symbols with one correction.
Full acceptance passes 160 tests. Retain the initial zero-outcome campaign and
register its replacement with the correction policy before prospective signals.

A later Yahoo response changed the same SPY bar's required Close to NaN. The
approved high-only exception does not impute it, and polling stays disabled.
Optional missing source floats are tagged losslessly in raw receipts; collection
requests stop at the latest completed exchange session. Prior successful receipts
and every empty predecessor campaign remain retained. No new training or credit
consumption occurred. See the latest feed diagnosis beside the campaign records.

## 19. Historical membership and delisting evidence — software implemented; real data pending, 2026-10-17

The user's requested evidence layer now imports stable security/issuer identities,
versioned ticker and membership events, raw bars/actions and verified or explicitly
unresolved terminal outcomes. Selection requires effective AND public availability
times; future outcomes do not select members. Immutable imports and coverage
reports retain missing/dead-stock failures, last-trade references, unresolved
settlements and unsupported non-cash outcomes. Alpha Vantage listing discovery is
read-only and never auto-certifies index membership or terminal returns.

Full acceptance passed 170 tests, plus final targeted historical validation.
A reproducible synthetic demo covers ticker reuse and a zero-value failed security;
it is not real market evidence. No credentialed vendor history is configured, so
real membership/identity/terminal verification is still pending. The old 26-tradable
universe, frozen models and retired test are unchanged; no training or credits
were consumed. Before an expanded experiment, obtain source evidence and integrate
historical episodes, membership exits, suspensions, delayed/non-cash settlements
and matching benchmarks. The current audit intentionally reports
`training_ready=false`. See [guide](docs/historical-data.md) and
[status](results/historical-coverage-2026-10-17/current-data-readiness.json).
## Alpaca historical access check — 2026-10-17

Authenticated Basic-plan historical SIP access is verified for a three-session,
four-ETF sample. The adapter accepts ignored `.env` credentials and spaces all
real SDK HTTP attempts, including pages and retries, below 180/minute against
the user's 200/minute account limit. Run one downloader process at a time;
independent process quotas are not coordinated. This is a data-access check;
full historical coverage, dividend evidence, historical memberships and terminal
outcomes remain separate gates. Existing Yahoo models and shadow inference are
unchanged. See [provider operations](docs/providers.md) and
[access evidence](results/alpaca-access-2026-10-17.json).

## 20. Matched retrospective training-universe expansion — 2026-10-17

The user authorized training/testing with more data. A declared six-cell CPU PPO
comparison uses seeds 11/23/37 and 50,000 steps per cell, expanding 22 to 55 stocks
plus the same four ETFs. Both arms score the identical original 26-symbol
validation panel. A fresh common Yahoo snapshot supplies matched prices and
cash dividends; each arm fits preprocessing on its own training observations.
The datasets contain training through 2020 and validation in 2021–2023 only;
the retired test interval is neither requested nor evaluated. Rewards, basic
features, simulator and costs are fixed. This is retrospective survivor research,
not the separately gated historical-membership/delisting experiment.

Data acceptance passes all 70 configured symbols: 70,200 versus 158,546 training
rows, with identical 19,058-row validation panels. A separate Alpaca 2016–2023
audit returned 139,508 SIP bars but found 713 missing LIN sessions; corporate
actions remain unverified. Keep that feed outside this dividend-aware comparison.
Use the existing aggregate-budget Colab pool, CPU smoke before research, and
verify all owned teardown. See [operations](docs/universe-expansion.md) and
[results](results/universe-expansion-2026-10-17/README.md).

All six declared 50k CPU PPO cells completed. Mean selected/final validation
returns were control 11.77%/10.49% and expanded 10.53%/5.60%, versus buy-and-hold
30.60%. Expansion failed the gate and lost on final returns in all three seeds;
no longer runs or reward changes were added. Six frozen bundles passed 468 exact
CPU prediction comparisons plus immutable normalization/data/schema checks.
All seven owned allocations including smoke stopped with verified cleanup.
Conservative study usage was 0.068809 units, cumulative project usage 35.458188
of 150. Keep the smaller panel as a research control, resolve provider/action and
historical membership/terminal evidence, and predeclare walk-forward development
before further search. The retired test and existing shadow campaign are unchanged.

## 21. Additional stock acquisition — 2026-10-19

The user clarified that the next step is to pull more stocks, rather than repeat
evaluation or start longer training. The declared universe now contains 110
stocks (55 added; ten names per annotated sector), plus the same 15 ETFs. Yahoo
provided 425,350 daily bars for 107 stocks plus the ETFs over 2010–2023. Missing
EA, AVB and EQR requests remain recorded; separate Alpaca SIP snapshots supplied
6,036 bars over 2016–2023. These feeds are not merged or training-ready, and the
supplement does not cover 2010–2015 or verified dividend cash flows.

All 110 stocks have some history, with 189,180 new stock bars acquired. Raw data
remain immutable locally, with manifests and coverage reports in Git. Explicit
CLI date bounds and opt-in failure-preserving batch collection are implemented;
default fail-fast and dataset quality gates remain unchanged. Full acceptance
passed 177 tests. No model evaluation, research training or Colab allocation was
added; the cumulative project ledger remains 35.458188/150 units. See
[acquisition evidence](results/stock-pull-2026-10-19/README.md).

## 22. Bounded improvement and short-side evidence — 2026-10-19

The user authorized continued bounded research toward comparable/better stock
and ETF returns, and requested profitable shorting on declining stocks. Follow
[the token-efficient research loop](docs/research-loop.md), preserve the shared
150-unit ceiling/130-unit teardown rule, and never select on the retired test.
Future declining-stock labels are reporting-only.

Evaluation now reconstructs separate long/short marked P&L including entry and
exit costs, dividends, split cash in lieu and borrow, reconciling both sides to
equity each session. A saved-path audit of nine existing matched 100k models
finds inconsistent results: final PPO/SAC/recurrent PPO average returns on the
three declining stocks are -20.62%/-18.99%/-21.64% versus holding -23.25%.
Positive policy and positive short contribution occur in only 1/9, 2/9 and 1/9
stock-seed observations respectively. The simulator supports profitable shorts;
the learned policies have not achieved this objective reliably. No training or
Colab allocation was added by this diagnostic. Next: causal execution-feasibility
baselines, then a preregistered matched direction/timing experiment. See
[audit evidence](results/short-side-2026-10-19/README.md).

The causal execution diagnostic is also complete on the existing 26-symbol
validation panel: fixed +95% exposure yields 27.61% stocks/11.64% ETFs versus
holding 32.63%/19.47%; the tested EMA trend/flat rule loses 16.26%/29.59%.
It is not a suitable profitable teacher by assumption. Preserve these negative
results and investigate action initialization/exploration and protection churn
with a matched bounded experiment before longer training. Diagnostic simulation
used no Colab allocation; the cumulative project ledger is unchanged.

## 23. PPO action initialization — complete 2026-10-19

Four seed-11 factorial arms screened initial exploration and protective means;
only wide protection and standard control received confirmation seeds 23/37.
All eight 50k research cells and four 512-step CPU smokes completed on Colab
Python 3.13.16. All 12 owned sessions stopped and 936 exact frozen predictions
passed. The three standard controls reproduce 39 tensors/18 checkpoint returns.

Mean selected/final returns improve from 11.77%/10.49% to 14.23%/13.89%, versus
holding 30.60%; all three final paired seeds improve. The relative gate passes,
absolute stock/ETF targets fail. Declining-stock final returns worsen from
-12.20% to -16.50%, with wide producing zero of nine positive-policy/positive-short
observations. No shorting-success or independent-performance claim is supported.

Conservative study usage is 0.104793 units; cumulative project 35.562980/150,
with 114.437020 remaining. Source is pinned to the screen's committed snapshot
through a verified export sharing the original ledger/lock. Seventeen release
assets passed GitHub SHA-256/size verification; 52 duplicate files (612.30 MiB
logical) were removed with 228 retained hashes unchanged. Next: predeclare a
causal down-regime sampling/universe experiment, not longer blind training. See
[results](results/action-initialization-2026-10-19/README.md) and
[operations](docs/action-initialization.md).

## 24. Explicit complete-case dataset — 2026-10-19

An explicit 107-stock Yahoo configuration records its 110-stock parent and
exclusions EA/AVB/EQR; partial Alpaca data remains separate. Opt-in universe
coverage checking gates every configured stock/ETF/context while retaining
the source's full failed-request provenance. Default snapshot-wide checks and
existing default dataset identities remain unchanged; no unverified-data
override is used. CLI development-only/common-panel options are now exposed.

Dataset `c2e6c58c4d4b0a9853b5b37851c731f559f42e597e9eefcf77796d874db31398`
contains 298,946 training feature rows across 111 tradables and the exact original
19,058-row/26-symbol validation panel, with zero test rows. Expanded preprocessing
fits training only; old models retain their original normalization. No new
training/provider requests or separate allocation were used for preparation.
Full local acceptance passes 193 tests. See
[data evidence](results/complete-case-107-2026-10-19/README.md).

Training-only trailing-price diagnostics find 93/111 expanded tradables with
persistent decline windows versus 16/26 control; their observation shares are
5.15% versus 2.31%. These are causal trailing labels, not future-recovery labels
or proof that expansion improves returns. Exact diagnostics remain in the data
evidence folder for the next separately declared research experiment.

## 25. Causal sampling and broader universe — complete 2026-10-19

Four 50k seed-11 PPO cells and four 512-step CPU smokes completed on Python
3.13.16, using matched readiness pools and the exact original validation panel.
Balanced sampling raised actual downtrend transitions from 2.9% to 11.3% on
small data and 5.1% to 15.5% on large data. Small balanced selected/final return
was 16.20%/16.20% versus reference 2.22%/-5.43%; large balanced was 10.19%/4.05%.
All three candidates worsened ETF returns, so the declared screen failed and no
confirmation/longer training was added. Holding returns 30.60% overall.

Small balanced remained mostly long and failed profitable shorting on the three
declining stocks; large balanced had one positive-policy/positive-short example
(NEE) but still lost on AMZN and DIS. Training-only forward diagnostics show the
severe-decline start class often rebounds over 20 sessions; past decline labels
cannot substitute for future direction. A synthetic witness also demonstrates
execution-state aliasing: identical current observations can hide different
protective stops and yield different next-step rewards/positions.

Next: a versioned execution-state observation with masked same-dimension and
legacy controls, preserving frozen inference compatibility, actions and rewards.
Eight frozen bundles passed 624 exact predictions and all eight owned sessions
stopped. Conservative study usage 0.075355; cumulative project 35.638336/150.
Full acceptance passes 207 tests. Seventeen GitHub asset hashes/sizes are verified;
36 duplicates (512.95 MiB logical) were removed with 160 retained hashes unchanged. See [results](results/causal-sampling-2026-10-19/README.md) and
[methodology](docs/training-sampling.md).

## 26. Versioned execution observations — complete, failed promotion

A synthetic witness proves that the current 12-value observation can alias two
states with different held protection and different next-step outcomes. Expose
actual stop/target distances, partial-exit status, last protective actions and
share-unit scale through an explicit versioned observation profile. Keep a
same-dimension masked control and a legacy reference; preserve all existing
frozen-model input contracts and deterministic prediction parity. Do not change
the four-action contract, reward, costs or simulator behavior in this experiment.

Test observability, shape/schema rejection, legacy parity and Gymnasium/SB3
checkers before any new Colab allocation. Declare the matched multi-seed protocol
and limits first, retain retired-test isolation, and share the original project
ledger. Better observability is a structural hypothesis, not a claim of improved
returns or a fully observed market process.

Implementation now includes legacy/full/masked observation profiles, ordered
schemas in frozen metadata, model-shape rejection and shared training/evaluation/
shadow state construction. Actual checkers, stop-alias witness, legacy RNG/state
parity, short geometry, split scaling and frozen save/load parity pass. Full
regression: 214 tests passed before the final protocol tests; the additional
strict remote-plan and dual-control gate checks are run separately. The declared
three-arm protocol preserves actions/rewards and permits confirmation only after
full beats both controls. See [execution observations](docs/execution-observations.md).

The three-arm seed-11 50k Colab screen completed: full 22.59%, masked 15.81%,
legacy 7.96%, holding 30.60%. Full failed ETF/drawdown constraints against the
controls, so confirmation was not run. Declining-stock total returns remain
negative despite positive short contributions. Six bundles pass 468 exact
frozen predictions; six owned sessions stopped, project 35.687823/150.
Sixteen GitHub asset hashes/sizes are verified; 27 duplicate archives/markers
(291.41 MiB logical) were removed with 314 retained hashes unchanged.
See [results](results/execution-observations-2026-10-19/README.md).

## 27. Universal direction-filter replay — complete; no all-market solution

Use fixed selected/final policies from the completed three-profile study. Replay
universal long-only and short-only exposure filters on every validation symbol,
retaining original protective components and recomputing observations along the
counterfactual path. Report all-symbol stock/ETF results and reconciled short
P&L; negative-holding labels remain reporting-only and never activate a filter.
No checkpoint selection, fitting, training or Colab allocation. Establish whether
profitable short contributions survive removal of long trades before proposing
a direction-specialist or entry-gating experiment. Preserve original reports,
weights, normalization and retired-test isolation.

The reusable evaluator now supports universal sign restrictions and preserves
original policy risk outputs in filtered-path audit rows. A strict committed
protocol covers all 18 profile/phase/mode cells. Tests verify state feedback,
cost-aware short profits, account reconciliation, reference-path rejection and
exact policy fingerprints for deterministic phase reuse.

The fixed diagnostic now covers 30 profile/phase/mode labels, 21 unique replays
and two reference-only calibration paths. Three-seed final unrestricted 13.89%,
long-only 15.49%, short-only -1.19%, holding 30.60%. Universal short-only is
positive on six of nine declining-stock/seed episodes (AMZN/DIS/NEE means
+2.12%/+6.49%/+2.87%), but not the full panel. Seed-37 selected strict reference
failed on two one-share rows; a separately committed <=$2/one-share bridge was
measured and declared before filtered replay. Original failure stays explicit.
No training or Colab units were added.
Nineteen release asset hashes/sizes are verified; 18 publication duplicates
(9.85 MiB logical) were removed with 597 retained hashes unchanged.

## 28. Market-only exposure head — complete, failed promotion

Current policy direction changes with account-path inputs even when market
features and weights are fixed; full-state seed11 flips direction on 14.70% of
common unrestricted/short-only decisions. This can reflect valid cost/risk
behavior, so test it as an ablation rather than assume it causes losses.

Propose an SB3 PPO custom policy in `rl/ppo/policies.py`: a market-only actor
branch and a full-observation actor branch feed a four-output Gaussian action
head; the critic retains full observations. In the treatment, a fixed connection
mask prevents signed exposure (component zero) from accessing the account branch.
Protective outputs remain learned and can use both branches. Compare against an
identical-capacity unmasked two-branch control and native full-state PPO; use
execution_full observations, seven existing market features, wide initialization,
unchanged net-equity reward, costs and legacy-uniform starts. Preserve native
policy/RNG/default bundle behavior. No universal direction filter during training.

Before allocation, verify the exposure output is exactly independent of account
perturbations while risk/value outputs retain access; verify masks/gradients,
parameter matching, serialization/frozen parity and strict schema rejection.
Predeclare a three-arm seed11 50k CPU screen with 512-step smokes, and confirm
23/37 only if treatment clears the selected/final, stock/ETF and drawdown gates
against both controls. Use one frozen source, original cumulative ledger,
two-unit study allowance, per-allocation two-hour deadline and actual Colab
Python checks. Directional P&L remains reporting-only; independent prospective
evidence is still required for success.

The custom policy, optional factory/training argument, frozen architecture checks
and version-10 CPU plans are implemented. Branch controls keep identical
initialized tensors and allocated parameter counts; fixed masks remove active
exposure/account connections. Native model/RNG parity and save/freeze/load
checks pass locally. Numeric protocol and shared dual-control reporter are
frozen before allocation. See [policy architecture](docs/policy-architecture.md).

All three 512-step CPU smokes and three 50k cells completed. Native final 22.59%,
unmasked branches 5.41%, market-only exposure -3.64%, holding 30.60%. Treatment
failed both control gates and made losses on AMZN/DIS/NEE, including short-side
losses. No confirmation was launched. Native selected/final tensors and all
checkpoint scores exactly reproduce prior controls; six bundles pass 468 exact
frozen predictions. Six owned sessions stopped; project 35.748563/150.
All 18 release assets are verified; 27 duplicate archives/markers (300.33 MiB
logical) were removed, with 326 retained hashes unchanged.
[Full result](results/exposure-architecture-2026-10-19/README.md).

## 29. Frozen seed ensembles — diagnostic complete; absolute gates failed

Before more training, preregister uniform averages of existing stateless PPO/SAC
three-seed policies, retaining account inputs. Inspect strongest same-data
confirmed baselines and verify feature/action/feed/normalizer compatibility.
Fix all member lists and equal weights before measuring ensemble performance;
report PPO-only, SAC-only and combined uniform candidates rather than choose
weights retrospectively. Apply all members to one shared evolving account and
average their four bounded outputs, not independent equity returns. Preserve
learned protective components, costs and next-session execution.

Use a development-only evaluation snapshot; inspect source manifests/receipts
without loading retired-test bars, and explicitly verify any dataset provenance
bridge. Include selected/final phases, immutable input hashes, individual
references and all stock/ETF/short diagnostics. This is local frozen inference,
not training or a new Colab allocation. Distinguish one ensemble made from three
seeds from three independent ensemble repetitions. No checkpoint/member is
selected using future declining-stock labels, and no deployment/goal completion
is implied by a single development result.

Uniform action averaging, strict member contracts and checksummed recipe
restoration are implemented. Preflight confirms six source PPO/SAC models match
the development snapshot's schema/normalizer/feed and exact validation dates/
closing prices. The older source dataset remains unopened. Fresh individual
references and all three fixed ensemble candidates will use one shared account
per symbol, with original path drift bounded before ensemble evaluation.

All 18 cases completed with immutable sources and recipe parity. Combined final
22.16% versus holding 30.60%; its relative descriptive gate passes but absolute
gates fail. Original SAC11 cost-reference failure remains failed; promotion is
disabled. Combined declining-stock joint total/short success is 0/3, versus 1/3
for PPO-only and SAC-only. No new training or Colab units; project remains
35.748563/150. See [complete evidence](results/frozen-ensembles-2026-10-20/README.md).

## 30. Reward objective alignment — screen complete; rejected

Add optional initial-capital net equity changes against unchanged net-log reward.
Keep native full-state wide PPO, data/costs/actions/discount/starts fixed. Two CPU
512-step smokes precede a matched seed11 50k screen; confirm23/37 only on the
predeclared gate. Preserve all failed results and short-side joint total/P&L
reporting. Undiscounted telescoping does not imply discounted PPO equivalence.
See [protocol](configs/experiments/reward_objectives.json) and
[reward guide](docs/reward-objectives.md). No new allocation yet.

Both512-step CPU smokes and matched50k cells completed. Candidate selected23.38%,
final20.43%, control22.59%, holding30.60%. Selected gain0.79pp and final loss2.16pp
fail the gate; stocks/ETFs worsen, declining-stock joint success0/3. No confirmation.
Four sessions stopped,312 frozen predictions exact, native control fully reproduced.
Project35.795378/150; [evidence](results/reward-objectives-2026-10-20/README.md).

## 31. Longer-horizon causal price context — screen complete; rejected

Add optional63/252-session log price returns and252-close drawdown. Preserve
forward-only split continuity, completed-bar availability and training-only fit.
Require253 contiguous observations; reset across known gaps. Prepare legacy7
matched-warmup,10-dimensional masked and full controls with identical raw panels
and first7 preprocessing. Default `none` must reproduce the original dataset.
This is an information hypothesis, not evidence of predictive power. No allocation
until data preflight and a committed numeric dual-control protocol; CPU smokes
then seed11 screen, confirmation23/37 only if both control gates pass. Preserve
short-side joint total/P&L reports and retired-test isolation.
See [feature guide](docs/trend-context.md).

Version12 context jobs now lock dataset/feature contracts before allocation,
inside the worker and on artifact retrieval. Matched reporting handles the fixed
3-dataset mapping and both gates. Focused job/fleet/report/artifact checks pass;
CPU smokes are next after committed-source export and account preflight.

All3 CPU smokes and3 matched50k cells completed. Full selected14.26%/final13.89%,
masked13.93%, baseline0.83%, holding30.60%. Full fails both gates, ETFs worsen and
joint declining-stock total/short success is0/3; no confirmation. All6 owned
sessions stopped,468 frozen predictions exact, project35.856639/150.
[Evidence](results/trend-context-2026-10-20/README.md).

## 32. Saved-path participation diagnosis — complete

Audit5 completed50k models and both phases using saved development reports only.
Reconcile costs and side P&L; separate accounting cost drag from non-cost excess,
close exposure from intraday activity, and holding-limit/protective re-entries.
Native stock/ETF gaps7.49/10.83pp include4.85/3.88pp cost drag. Full-context ETF
short contribution−6.79pp versus−0.15pp masked; costs are lower. Small same-side
resizes incur only0.05–0.12pp historical costs, so a small band is low priority.
No training, data access, test use or Colab allocation. Next investigate strictly
training-only supervised signal/warm-start feasibility before a new bounded
protocol. [Evidence](results/participation-diagnostic-2026-10-20/README.md).

## 33. Training-only supervised signal feasibility — completed; rejected

Prepare gross20-session next-open-to-close unit total-return labels, with later
splits/dividends and conservative completion availability. Fit labels end2017,
internal probe spans2018–2020; discard boundary-crossing windows. Never consume
project-validation features/labels; ignore pooled preprocessing and refit on fit
rows only. Immutable derived payload has45,266 fit and18,616 probe examples.
Matched10-input masked/full models, constant and fixed ridge controls,3 seeds,
fixed-final MLP updates and predictive gates are committed before fitting.
No profitability or PPO initialization claim; Colab CPU fitting/verification
requires its own strict job/artifact integration. One-unit allowance, original
ledger, actual default Python and owned cleanup remain required.
[Protocol](configs/experiments/supervised_signal.json), [guide](docs/supervised-signal.md).

Version13 feasibility jobs and the CPU worker are implemented, with six final
MLPs, constant/ridge baselines, durable checkpoints and a separate checksummed
forecast-only bundle. Archive acceptance recomputes metrics/gates from predictions;
local restoration compares every probe forecast with immutable inputs. Two
discarded prefix updates plus12000 final-fit updates are explicitly counted.
Thirty-six focused tests pass without local fitting. Actual Colab is next.

One CPU allocation completed all six MLP fits, two ridge and constant controls.
Masked/full MLP MSE is 5.52%/12.56% worse than constant; zero passing seeds and
failed yearly gates. No PPO warm start. All111,696 forecasts verified against
immutable probe data, normalization/schema checks pass, session stopped.
Project35.861240/150; [evidence](results/supervised-signal-2026-10-20/README.md).

## 34. Fixed-forecast common-component diagnostic — completed; rejected

Preregister same-session MLP demeaning plus same-arm frozen ridge mean anchoring.
Use all26 symbols exactly once per session; enforce availability before every
next entry session. No labels in the transformation, fitted coefficient, model
update or new Colab allocation. Source weights/predictions/data are checksummed;
original failed gate remains failed and promotion disabled. Reuse unchanged
numeric predictive gates for descriptive scoring only after committed preflight.
[Protocol](configs/experiments/signal_centering.json), [guide](docs/signal-centering.md).

Availability preflight was committed before scoring. Centering reduces raw MLP
MSE7.39% masked/12.96% full, but is0.38%/0.70% worse than ridge; zero passing
seeds and yearly constraints fail. Original gate remains failed, no promotion
or PPO allocation. All source hashes unchanged; no new units. Next investigate
a distinct tabular tree learner under a new compatible, bounded CPU protocol.
[Evidence](results/signal-centering-2026-10-20/README.md).

## 35. Tabular forest signal feasibility — completed, gate failed

Six fixed128-tree forests plus2 prefix trees were fitted on Colab CPU using
Python3.13.16 and protocol49aa716b. Both predictive arms fail: masked/full mean
MSE0.00672440/0.00679195 is16.23%/17.47% worse than own ridge, zero passing seeds,
and2020 constraints fail. All111,696 restored forecasts pass parity and frozen
input checks. No PPO transfer or model promotion; profitable shorting and the
stock/ETF objective remain unachieved. Owned shutdown verified, project usage
35.871240/150, zero active assignments. Preserve all models and evidence; stop
this configuration rather than enlarging the failed fit.
[Results](results/forest-signal-2026-10-20/README.md),
[protocol](configs/experiments/forest_signal.json), [guide](docs/forest-signal.md).

## 36. Chronological historical-regime audit — preregistered

Audit fixedSPY/QQQ/IWM/DIA2001–2009 source consistency and all-session coverage
before another learner. No future losing-symbol selection, automatic merging,
Colab allocation or fitting. Past-only63/252-session decline counts are
descriptive. Source splits retained without reapplying; dividend/source fields
reconciled and unsupported capital gains rejected. A later matched protocol
must separate recent-ETF sampling mass from old-history information and preserve
all existing stock/ETF, short-side and prospective gates.
[Protocol](configs/experiments/historical_regime_audit.json),
[guide](docs/historical-regimes.md).

Milestone36 source audit accepted:9,052 bars,2,263 sessions per ETF, exact
raw-field/action agreement and identical cached readback. Earlier joint trailing
decline coverage23.55% versus8.61% recent under the same past-only rule. No merge,
training or Colab. Next preregister overlap/price-unit checks before matched
training preparation. [Evidence](results/historical-regimes-2026-10-20/README.md).

The separate seam protocol fixesDec2009–Mar2010, four ETFs, source/action and
exact row-set checks, unchanged numeric price/dividend tolerances, exact volume,
splits and event/availability times.19 history/overlap tests pass before provider
acquisition. Failures retain evidence and stop merging; no automatic rescaling.
[Protocol](configs/experiments/historical_overlap.json).

Seam overlap accepted:332 rows, zero differences, matching action/time contracts
and identical readback;69 related tests pass. No rescaling, merge or training.
Next prepare separate old-history episodes, fixed512-session training windows
and matched recent/old ETF sampling mass while preserving original observations,
validation and normalization. Freeze identities/gates before Colab.
[Evidence](results/historical-regimes-2026-10-20/overlap.json).

## 37. Matched historical ETF augmentation — prepared, fitting pending

Original/recent-ETF/earlier-ETF datasets preserve original observations,
normalization and validation exactly. Rows70,200/81,000/78,976; augmented arms
each have8/30 expected ETF episode mass. Full-window sampling logs actual
symbol/period starts and transitions; legacy behavior preserved. Fixed native
wide/full-state PPO50k seed11 screen must beat both controls and satisfy both
stock/ETF selected/final targets before23/37 confirmation.16 focused preparation/
sampling tests and Gym/SB3 checkers pass without fitting. No Colab allocation;
worker/job/artifact validation and committed source freeze pending.
[Protocol](configs/experiments/historical_augmentation.json),
[guide](docs/historical-augmentation.md),
[prepared identities](results/historical-augmentation-2026-10-20/preflight.json).

Historical version15 job/worker/artifact integration implemented. Protocol/data
identities, full-window summaries, sampling totals and frozen metadata checked.
Fix nested sampling snapshots changing during parity reset; regression covers it.
Three immutable smoke tasks validate before allocation; freeze committed source
then run metered CPU smokes under the shared2-unit study allowance.

Historical smokes passed in Colab:3x512 transitions,234 exact frozen predictions,
Python3.13.16 and3 owned shutdowns verified. Screen launched:3x50k seed11,
same4be5219 source, original2-unit study allowance including smoke.
[Acceptance](results/historical-augmentation-2026-10-20/smoke-summary.json).

Milestone37 completed, gate failed. Earlier17.06% selected/final versus
original19.23%/15.01%, recent15.33%/15.33%, holding30.60%. No confirmation or
extension. Augmented symbol transition counts match exactly, so nominal
sampling imbalance does not explain this comparison.468 frozen checks pass;
6 owned shutdowns,15 verified assets,186.45MiB duplicate cleanup with377 retained
hashes unchanged. Project35.917535/150.
[Results](results/historical-augmentation-2026-10-20/README.md).

## 38. SAC execution-state diagnosis and protocol — preregistered

Three100k frozen SAC models load with12 legacy inputs;9 execution fields omitted.
Six saved phases audited without dataset access or fitting. Final stock/ETF
returns17.69%/9.97% versus holding32.63%/19.47%, close shorts2.53%/0.25%;
costs explain5.98/3.91pp of14.94/9.50pp gaps. No fabricated direction P&L.
New fixed SAC legacy/masked/full50k comparison uses current train/validation-only
data, same params/costs/reward and both controls. Strict selected/final absolute
stock/ETF gates before23/37 confirmation; prospective evidence still required.
Ten focused tests pass; no new Colab allocation. Job/worker acceptance pending.
[Diagnosis](results/sac-diagnostic-2026-10-20/README.md),
[protocol](configs/experiments/sac_execution_state.json),
[guide](docs/sac-execution-state.md).

SAC state version16 metered job/worker/artifact contracts implemented. Three
immutable1024-transition CPU smoke tasks validate locally; full/legacy/masked
archive fixtures pass and semantic tampering is rejected. Freeze committed
source before allocation; share original ledger and2-unit study allowance.

SAC state Colab smokes pass:3x1024 transitions,Python3.13.16,234 exact frozen
predictions and3 verified owned stops.3x50k seed11 screen launched from the same
bb0f3e2 source and shared original2-unit allowance. No performance verdict.
[Evidence](results/sac-execution-state-2026-10-20/README.md).

Milestone38 completed, gate failed. Full-state SAC selected/final7.93% versus
legacy7.44%/7.39%, masked17.84%/17.69% and holding30.60%. Cached last pre-update
checkpoints are not final reports. No confirmation, extension or promotion.
All6 owned sessions stopped,468 exact frozen checks,13 release assets verified;
227.91MiB duplicate cleanup with390 retained hashes unchanged and byte-identical
report replay. Project36.009692/150.
[Results](results/sac-execution-state-2026-10-20/README.md).

## 39. SAC saved-learning audit and initial-temperature protocol — fixed

Three final50k models/replays,4096 fixed sampled transitions each, zero optimizer
steps. Mean absolute rewards0.0055–0.0059, logged alpha0.067 at10k to~1e-5 at50k,
final Q means~4 and sampled one-step RMSE0.0082–0.0099. Tensor/file hashes
unchanged, no dataset/test access. Value offset alone is not causal proof.
Preregister one initial-alpha candidate0.001 against default1 with learned
adaptation and all reward/data/full-state/network/cost settings fixed. Two
1024 smokes then50k seed11 screens; strict relative and absolute benchmarks
before23/37 confirmation, no grid/longer runs on failure. Fresh default must
reproduce previous full-state control. Six numeric tests pass; no allocation.
[Diagnosis](results/sac-learning-diagnostic-2026-10-20/README.md),
[protocol](configs/experiments/sac_temperature.json),
[guide](docs/sac-temperature.md).

Initial-temperature version17 integration implemented: fixed override/identity
checks, actual initial alpha/target-entropy and network fingerprints captured
before learning.49 focused regressions pass; two1024 CPU smoke payloads validate.
Freeze committed source before metered allocation; no assumption that fixture
fingerprints prove real cross-arm parity.

Temperature smokes passed:2x1024 Colab CPU transitions,Python3.13.16, exact
initial network fingerprints, alpha1/0.001, target entropy−4,156 frozen-prefix
predictions and2 owned stops. Fixed2x50k seed11 screen launched, same0f04e6b
source and original2-unit allowance. No performance/reproduction verdict yet.
[Acceptance](results/sac-temperature-2026-10-20/smoke-summary.json).

## 40. Initial-temperature outcome — failed; saved-path diagnosis next

The two 50k SAC seed-11 screens are complete and fail the preregistered gates.
Initial alpha 0.001 improves selected mean return to 19.67% versus default
7.93%, but final falls to 9.84%; the final gain of 1.92pp misses 2pp and
drawdown worsens 4.27pp. Final stocks/ETFs 10.78%/4.72% trail matched holding
32.63%/19.47%; joint falling-stock total/short positives are 1/3 selected,
0/3 final. No extension, confirmation seeds, temperature sweep or promotion.

Fresh default configuration, all checkpoint scores and selected/final network
tensors reproduce the prior full-state SAC exactly. Initial network hashes
match across arms; 312 smoke/screen selected-bundle prefix predictions are exact.
All four owned sessions stopped. Thirteen release assets have verified server
digests; duplicate-only cleanup removes 15 files (167,684,918 bytes), preserving
354 file hashes including models, replay, data, source, ledger and plan.html.
The post-cleanup report reproduces byte for byte. Eighteen focused temperature
and report-contract checks pass. Cumulative project 36.069198/150; account
350.48 with zero assignments/rate after shutdown. Retired test stays isolated.
[Complete evidence](results/sac-temperature-2026-10-20/README.md).

Next inspect saved selected/final action paths, directional contributions,
exposure and cost differences to distinguish deterioration mechanisms before
preregistering another bounded hypothesis. This diagnostic needs no fitting,
provider calls or Colab allocation. Development results remain repeatedly used;
prospective confirmation and the overall benchmark/short objective remain unmet.

## 41. Saved TD3 audit and fixed execution-state protocol

Three saved TD3 seeds omit execution state despite verified Gaussian noise.
Final stock/ETF means -0.08%/-0.37%, so no performance claim. A fixed three-arm
legacy/masked/full test retains prior failed PPO/SAC evidence and absolute
benchmark requirements. Audit/protocol checks12 pass without training or data
access. Runtime integration and metered Colab smokes/screens remain pending.
[Guide](docs/td3-execution-state.md),
[protocol](configs/experiments/td3_execution_state.json).

TD3 version18 runtime/report integration complete:50 focused checks, actual
immutable smoke payload validation and script compilation pass without fitting.
Freeze committed source before three Colab CPU smokes. Actual network/noise and
frozen-prefix parity must precede the fixed50k screen.

TD3 actual Colab smokes3x1024 pass on Python3.13.16; matching initial four-network
hashes/noise,234 exact frozen predictions and3 owned stops verified. Fixed3x50k
screen next from same76151d1 source and original2-unit allowance.
[Acceptance](results/td3-execution-state-2026-10-20/smoke-summary.json).

## 42. TD3 state screen — failed, retained and cleaned

Full selected/final6.19% versus holding30.60%; stocks5.38%/ETF10.65% and median
stock excess-16.16pp fail. No extension/confirmation.468 frozen-prefix checks,
6 verified owned stops,13 asset digests and post-cleanup report reproduction
pass. Removed21 duplicate files preserving405 hashes. Project36.149107/150.
Before new fitting, predeclare fixed training-only saved-policy replay to
distinguish weak fitting from a generalization gap; no diagnostic selection.
[Evidence](results/td3-execution-state-2026-10-20/README.md).

## 43. Fixed fitted-period replay — completed, no promotion

Final SAC passes fitted2018–2020 stock/ETF/median comparability but fails the
known development gates; PPO/TD3 fail training flags. No universal optimizer or
overfitting diagnosis.20 focused checks, immutable source/tensor/normalizer
proofs and3 release asset digests pass, no fitting/Colab. Before another study,
audit existing expanded data for SAC training diversity with original
normalization/validation fixed. Project36.149107/150, objective unmet.
[Evidence](results/training-fit-2026-10-20/README.md).

## 44. SAC training-diversity data and fixed protocol — prepared

Cached expansion audit preserves all originals and validation; new cdb93a33…
uses original preprocessing.18 checks pass, source hashes unchanged, no fitting/
Colab. Fixed fresh two-arm100k comparison holds stock/ETF sampling mass; group
cycles and actual coverage are required before interpreting the result. Sampler/
runtime integration and smokes next; no relaxation of absolute or prospective
requirements. [Guide](docs/sac-diversity.md),
[protocol](configs/experiments/sac_diversity.json).

Diversity version19 integration complete:82 focused checks, actual immutable
payload checks and script compilation pass. Source freeze then2x1024 Colab CPU
smokes next, before matched2x100k screen. Positive-transition coverage and
actual initial network/normalizer parity remain required.

Diversity actual smokes pass2x1024 CPU transitions, initial network/alpha and
normalization equality,156 exact frozen predictions and2 verified stops. Fixed
2x100k screen next under same study allowance, worker5cad3fa/controllerc6f69d4.
[Acceptance](results/sac-diversity-2026-10-20/smoke-summary.json).

## 45. SAC training diversity — complete, gates failed

Both100k seed11 Colab CPU screens are complete. Original selected/final mean
returns24.35%/10.38%; expanded24.19%/24.19%; matched holding30.60%.
Expanded final stocks26.06%/ETFs13.88% trail holding32.63%/19.47%; median
stock excess−5.49pp. Selected gain−0.16pp fails the required2pp. All absolute
gates fail and expanded joint profitable-total/short cases are0/3. No extension,
confirmation seeds or promotion under this protocol.

All training symbols receive transitions. Intended ETF probability matches,
but actual ETF transitions differ15.36% versus19.97%; added stocks alone cannot
explain the observed return difference. Both survivor-data and repeatedly-used
development limitations remain. All312 frozen prefix predictions are exact,
all4 owned sessions stopped,14 release asset digests verified. Duplicate cleanup
removes16 files/248,893,520 bytes with643 retained hashes unchanged; the complete
report reproduces byte for byte afterward. Thirty-six focused sampling/job/
archive/gate/report tests pass. Project usage36.263895/150, remaining113.736105;
post-shutdown account350.31, zero rate/assignments. Retired test stays isolated.
[Evidence and interactive artifacts](../results/sac-diversity-2026-10-20/README.md).

Next inspect saved selected/final cost, exposure and directional contributions
before a distinct hypothesis. This requires no fitting/provider/Colab; do not
extend the failed family solely because its late checkpoint improved.

## 46. SAC fitted-window generalization diagnosis — complete

Final100k original/expanded SAC fitted mean91.72%/62.43% versus56.45% holding;
both pass descriptive training comparability but original development gates
remain failed. Existing50k and new100k decline/rebound reporting uses future
cohorts only descriptively. Source files, weights, updates and preprocessing
unchanged;12 tests pass. Full derived paths retained, no fitting/Colab. Project
36.263895/150. Next a separately fixed regularization hypothesis with fresh
matched expanded control, not blind extension or promotion.
[Evidence](results/sac-diversity-fit-2026-10-20/README.md).

## 47. SAC regularization — protocol and optimizer acceptance prepared

Fixed Adam actor/critic decay0 versus1e-4, entropy optimizer unchanged;
identical expanded-data100k cells, strict selected/final benchmark gates.
Six checks pass including actual Adam settings and initial tensor equality;
no fitting or Colab. Runtime/archive/report integration then two1024 smokes
required before screen. [Guide](docs/sac-regularization.md).

Regularization version20 integration complete:40 focused tests, actual immutable
payload checks and script compilation pass. Strict reports restore selected/
latest optimizers and require matching actual sampling counts. Freeze source,
check account usage, then two1024 Colab CPU smokes; no screen before acceptance.

Regularization replacement2x1024 Colab CPU smokes accepted:156 exact frozen
predictions, actual optimizer save/load, identical initial tensors/alpha and
actual sampling;4 total attempt sessions stopped. Failed mutation attempt kept,
fix6792d2b. Project36.297597/150, fixed paired100k screening next.

## 48. Regularization screen interrupted; recovery complete

Metering failure stops the owned fleet. Control finishes21.63%; candidate80k
curve5.10% lacks final100k, so no promotion/confirmation and no automatic restart.
Exact-fleet recovery on2026-10-21 verifies all6 attempt sessions stopped.13 release
assets verified,3 duplicate archives removed/656 retained hashes unchanged.
Conservative ledger37.787213/150 includes monitoring-gap extrapolation; measured
screen debit0.32 differs from estimate1.489616. Preserve stop flag; further
allocation requires explicit reviewed authorization. Offline diagnostics allowed.
[Full interruption evidence](results/sac-regularization-2026-10-20/README.md).

## 49. Explicit interruption checkpoint identity — implemented

Every new checkpoint has current validation JSON and an atomic manifest binding
latest/selected weights/reports to counts, phases and hashes. Complete archive
checks reject corruption/non-final markers; legacy reports stay compatible and
require separate alignment evidence.33 tests pass without fitting, no Colab or
ledger reset. Remote acceptance pending authorization after metering stop.
[Contract](docs/training.md#checkpoint-identity-during-interruptions).

## 50. Retained regularization numerical diagnosis — complete

Same-host inference-only audit finds5942 subnormal policy values and14.10×
flush-mode forward speedup versus control0/1.02×, fixed-batch outputs exact,
weights/update/source hashes unchanged. No training/backward/optimizer benchmark,
no Colab or arithmetic-setting adoption. Eleven focused checks pass. Project
37.787213/150 and budget stop preserved. Future matched numerical preflight
requires reviewed allocation authorization; incomplete return gates unchanged.
[Evidence](results/regularization-numerics-2026-10-21/README.md).

## 51. Fixed80k regularization replay — diagnostic, tolerance failed

Recovered diagnostic paths from retained intermediate weights, originals intact.
Score differs2.21088e-8 from logged80k and fails declared1e-10 tolerance; no
relaxation or parity claim. Stocks6.92%/ETFs−4.87%, zero short exposure/P&L,
0/3 joint positives. Weights/optimizer/normalization/source unchanged;20 tests
pass, no fitting/Colab. Incomplete screen and budget stop preserved.
[Evidence](results/regularization-80k-replay-2026-10-21/README.md).

## 52. Reviewed resumption and account credit protection — implemented

Explicit user approval allows research to resume while account balance stays150
or higher. Allocation/projection guards now begin teardown170 account units,
retain150 project ceiling/130 project teardown, and preserve37.787213 prior
charges. Separate locked review requires exact stopped fleet, verified cleanup,
idle metering, headroom and explicit authorization. Historical failures remain.
20 checks pass; account349.97, no allocation. Future fleets need newly frozen
floor-aware code. [Review evidence](results/budget-review-2026-10-21/README.md).

## 53. Fixed SAC discount-horizon study — integrated preparation

Distinct0.99/0.95 hypothesis, fresh identical expanded-data100k cells; no reward,
feature, numerical, temperature, initialization, sampler or execution changes.
Runtime contracts require actual model gamma/optimizer fingerprints and mandatory
current checkpoint identity; allocation requires floor-aware150/170 controller.
Two1024 CPU smokes precede screen; no allocations at preparation. Original
37.787213 project charges retained,2-unit study allowance, absolute and seed/
prospective gates unchanged. [Guide](docs/sac-discount.md).

Discount preparation acceptance:59 focused checks, actual immutable payloads and
script compilation pass; actual saved-step records required. Freeze source and
check official account usage before two1024 CPU smokes. No performance claim.

Discount actual2x1024 Colab CPU smokes accepted:Python3.13.16, actual gamma/
optimizer/step restoration, matching initial tensors/alpha/sampling, new checkpoint
identities and156 frozen predictions. Both owned sessions stopped; project
37.805591/150, account349.96 zero rate/assignments. Same-source paired100k screen
next under existing2-unit study cap and account floor150/teardown170.

### Milestone54: discount screen completed, performance gates failed


Fresh100k CPU candidate gamma0.95 selected/final22.79%/22.16% trails gamma0.99
control24.19%/24.19% and matched holding30.60%. Final stocks23.88% and ETFs12.64%
trail holding32.63%/19.47%; median stock excess−9.78pp. Candidate joint short
success1/3 versus control0/3 does not rescue failed return gates. No extension,
gamma sweep, confirmation seeds or promotion. Reused development is not independent
evidence; retired test remains unopened and the overall objective is unmet.

All4 owned sessions stopped;312 exact frozen predictions and35 focused checks
pass.13 server asset digests verified; duplicate publication archives totaling
31,529,571 bytes removed with398 retained file hashes unchanged. Strict report
reproduces byte-for-byte. Project37.897158/150, remaining112.102842; last account
349.88 with zero assignments/rate. Account150/170 and project150/130 protections
remain. Next inspect saved selected/final cost and directional contributions,
without fitting or allocation, before any distinct hypothesis.
[Detailed evidence and release](results/sac-discount-2026-10-21/README.md).

### Milestone55: retained recurrent contracts audited


Six retained100k recurrent selected/final models use legacy12 inputs, omitting9
explicit execution fields.96 fixed synthetic sequence observations show exact
direct-call/stream parity and reset replay; resetting each observation changes
actions, so memory is active. One regression passes, weights/updates/source
hashes unchanged; no fitting/dataset/test/Colab access. Missing explicit state
is a distinct testable hypothesis, not proof of the previous performance cause.
Next freeze a matched recurrent masked21/full21 comparison with feed-forward
control and unchanged absolute/relative gates before smoke acceptance.
[Audit](results/recurrent-contract-2026-10-21/README.md).

### Milestone56: recurrent execution-state protocol and construction contracts


Distinct fixed three-arm comparison: recurrent_full candidate, recurrent_masked
and feedforward_full controls; expanded111 data, unchanged normalization/sampler/
reward/costs, fresh100k seed11 screen after three512 CPU smokes. Same recurrent
initial tensors; feed-forward is not parameter-count matched. Both relative and
absolute selected/final gates required before three-seed confirmation/prospective
evidence. Seven actual-construction/save-load/sequence/reset/gate checks pass
without fitting. No allocation: job/worker/archive/restored-report integration
remains required. Project37.897158 retained; incremental study allowance3 includes
all attempts, account150/170 and project150/130 enforced at future allocation.
[Protocol and status](docs/recurrent-execution-state.md).

Recurrent remote integration:69 focused checks, three immutable smoke payloads
and compilation pass. Exact version22 plans/worker/archives/report implemented;
actual512 CPU smokes and frozen sequence acceptance remain next. No allocation.

Recurrent actual3x512 CPU acceptance passes:234 exact frozen predictions, model
records/checkpoint identity, equal initial recurrent tensors and all-arm sampling.
Three sessions stopped; project37.927940/150, account349.86 idle. Fixed same-worker
three100k seed11 screen next; no performance or prospective success claimed.

### Milestone57: recurrent screen complete, return gates failed


Fresh100k seed11 selected/final means:feed-forward4.46%/3.34%, recurrent masked
24.99%/24.04%, full-state recurrent candidate3.95%/−3.11%, holding30.60%. Candidate
loses0.51/6.45pp to feed-forward and21.05/27.16pp to masked. Candidate final
stock−2.25% versus holding32.63%, ETF−7.84% versus19.47%, median stock excess
−29.25pp. Candidate selected/final joint profitable-short outcomes0/3; masked
1/3 selected,0/3 final. Relative AND absolute gates fail. No extension, sweep,
confirmation or promotion; masked control cannot be retrospectively promoted.

All6 owned smoke/screen sessions stopped and cleanup verified.468 exact frozen
predictions include recurrent action/hidden/cell/reset parity and immutable
normalization. Actual selected/final records/checkpoint identities, equal initial
recurrent tensors, all-arm sampling counts and matched holding paths pass. Project
38.014408/150, remaining111.985592; study increment0.11725006. Official account
349.80, zero rate/assignments; floor150/teardown170 retained. No retired-test use.
Next saved-path cost/directional diagnosis without fitting before a distinct
hypothesis. Overall multi-seed/prospective objective remains unmet.
[Evidence and release](results/recurrent-state-2026-10-21/README.md).

### Milestone58: fixed recurrent sizing diagnostic prepared

Four frozen-policy cells, unchanged signs/protections/costs/holding cap; strict
unrestricted-path parity precedes candidates.12 filter/state/reference checks,
no fitting or replay yet. [Guide](docs/recurrent-sizing.md).

Sizing diagnostic complete: final32.80% versus holding30.60%, but selected
ETF/median and3.21pp drawdown worsening fail original gates. No promotion or
threshold sweep.8 verified release assets, four saved-path metric reproductions,
source/model/preprocessing unchanged. No Colab units. See
[full outcome](results/recurrent-unit-sizing-2026-10-21/README.md).

### Milestone59: sizing alignment protocol and execution contracts

Three fresh masked recurrent arms isolate training versus evaluation unit sizing.
Shared training/reward/validation/frozen execution routing and43 focused checks
pass without fitting. Job/archive/report/counters and actual CPU smokes remain;
no allocation or old-protocol promotion. [Guide](docs/recurrent-sizing-alignment.md).

Alignment integration complete:79 checks, actual immutable payload validation
and compilation pass. Strict version23 modes/counters/archives/restored reports
implemented; actual3x512 CPU smokes/frozen parity/equal-control fingerprints next.
No allocation yet. Original failed replay remains closed.

Alignment actual3x512 CPU smokes accepted:234 exact frozen mapped-action/state
predictions, actual execution counters, equal initial/control-final tensors and
sampling. Three owned sessions stopped; project38.047855/150, account349.78 idle.
Same frozen-worker three100k seed11 screen next; no performance claim.

### Milestone60: alignment screen complete, candidate gates failed


Fresh100k seed11 selected/final means:adaptive24.99%/24.04%, posthoc unit32.80%/
32.80%, aligned unit20.04%/13.93%, holding30.60%. Adaptive/posthoc final tensors
match exactly. Candidate selected/final loses4.95/10.12pp to adaptive and12.76/
18.88pp to posthoc. Candidate final stocks14.86% versus holding32.63%, ETFs8.77%
versus19.47%, median stock excess−24.77pp. Final mean drawdown−37.41% worsens
7.42pp versus adaptive and4.21pp versus posthoc. Selected joint short outcomes
2/3 deteriorate to0/3 final. Relative/absolute/risk gates fail: no extension,
threshold/architecture sweep, confirmation or promotion of this family.

All6 owned sessions stopped;468 exact frozen mapped-action/state predictions,
actual execution counters/traces, restored models and checkpoint identities pass.
All initial tensors and actual sampling counts match; all111 symbols receive
training transitions. Unit training maps62,293 sizes, processes0 non-unit sizes,
changes0 protection components; all100,000 requests nonzero. These facts establish
execution, not a proven failure cause. Posthoc remains a one-seed development
control, not robust/prospective success; original failed replay is unchanged.

Project38.136658/150, remaining111.863342; study conservative0.12225039 (ledger
observed debits0.08; rounded account decrease0.09 includes meter delay). Account
349.71 zero rate/assignments, account150/170 and project150/130 retained. No retired
test or orders. Overall goal unmet. [Full evidence](results/recurrent-alignment-2026-10-21/README.md).

### Recurrent cash-choice research (2026-10-22)

Fixed four-arm action-distribution/cash-availability protocol and hybrid recurrent
policy prepared; integration, actual Colab smokes, screen and any gated
confirmation/prospective comparison remain. No allocation at this milestone.
See [category guide](docs/recurrent-cash-choice.md). Prior failed alignment remains
closed; this is a separate hypothesis, not a promoted result.

Cash-choice integration preflight now passes96 focused no-fit checks and actual
immutable payload validation. Version24 plans/worker/archive/smoke/report paths
are implemented; actual Colab smoke acceptance remains before the fixed screen.

Cash-choice fixed100k seed11 screen completed and failed return gates: selected/
final cash29.24%/6.63%, binary34.38%/−1.25%, posthoc32.80%/32.80%, holding30.60%.
All12 owned sessions stopped;624 exact smoke/screen frozen predictions. No
confirmation/promotion; publication and verified duplicate cleanup remain.
[Outcome](results/recurrent-cash-2026-10-22/README.md).

Cash-choice publication/cleanup accepted:15 server assets verified,58,832,238
duplicate bytes removed,249 retained hashes unchanged; strict report reproduces
byte for byte. All12 owned sessions stopped. Saved behavior/optimizer audit next,
no family extension or promotion.

Recurrent selected/final fitted-period diagnosis prepared with a fixed common
2018–2020 window and original26 panel. Commit before eight read-only replays;
no new training, provider calls or retired-test access. See
[method](docs/training-fit-diagnostic.md).

Recurrent fitted replay complete: eight policies,153,088 execution rows total;
all fitted comparability flags fail, sources/tensors/full optimizer/normalization
unchanged. Cash fits this window better late but development worsens; no simple
optimizer-cause claim. See [results](results/recurrent-fit-2026-10-22/README.md).

Separate fixed Gaussian update-guard protocol/construction prepared; target_kl
0.01 versus unchanged fresh strongest control, same100k budget. No actual
allocation; dedicated Colab integration and smokes required. See
[guide](docs/recurrent-update-guard.md).

Update-guard version25 integration passes75 focused checks and actual smoke
payload validation; actual two512 CPU smokes remain before fixed100k screen.
No allocation at integration milestone.

Guard protocol2 shared untrained tensor/RNG initialization prepared after
version1 rejected unequal hashes.94 focused checks and revised payload validation
pass. Original evidence/budget retained; actual retry smokes remain.
[Details](docs/recurrent-update-guard.md).

Update-guard100k screen complete: relative improvement passes, absolute targets
fail; no promotion/confirmation. Six owned sessions stopped, frozen parity
verified, project38.39201565/150. Publication/cleanup then saved-path diagnosis.
[Evidence](results/recurrent-guard-2026-10-22/README.md).

Optional completed-session context implemented/tested (53 focused checks);
real-data preflight, fixed protocol and Colab acceptance remain before training.
[Method](docs/session-context.md). Defaults unchanged, no new usage.

Session-context retained-data preflight accepted: exact111/26 symbol panels,
original normalization and298,946/19,058 rows. Numeric protocol, initialization
and Colab acceptance remain. [Evidence](results/session-context-2026-10-22/README.md).

Session-context fixed numeric protocol/shared untrained initialization prepared;
31 focused checks pass. Dedicated Colab integration/checkers/smokes pending.
[Method](docs/session-context.md). No new usage.

Session-context Colab integration and real environment construction pass;76
focused checks. Actual three512 remote smokes remain before100k screen.
[Evidence](results/session-context-2026-10-22/README.md). No new usage.

Session-context three512 Colab CPU smokes accepted:234 frozen predictions and
actual saved models/initialization/sampling pass; all sessions stopped. Same-source
three100k screen next within original allowance. Project38.42515390/150.
[Evidence](results/session-context-2026-10-22/README.md).

Session-context100k screen complete; both control and absolute gates fail, no
promotion/confirmation. Six owned sessions stopped;468 total frozen checks pass.
Project38.51487741/150. Publication/checksummed duplicate cleanup next.
[Evidence](results/session-context-2026-10-22/README.md).

Distinct fixed64/128 recurrent-capacity protocol/shared starts prepared;65
focused construction/regression checks pass. Integration/checkers/Colab smokes
remain, runtime projection required before100k. No new usage.
[Method](docs/recurrent-capacity.md).

Capacity Colab integration/full-architecture checkers pass;96 focused tests and
actual immutable tasks validate. Actual two512 smokes/runtime profile remain
before100k. [Evidence](results/recurrent-capacity-2026-10-22/README.md). No new usage.

Capacity full64/128 Colab smokes accepted:156 frozen checks and saved model/
optimizer/sampling/RNG evidence pass; sessions stopped. Runtime projection50.3min
passes two-hour limit; same-source two100k screen next. Project38.53588481/150.
[Evidence](results/recurrent-capacity-2026-10-22/README.md).


## Completed capacity screen — 2026-10-23

Both fixed seed11 runs completed100,000 transitions in Colab CPU, defaultPython
3.13.16, from frozen clean source39f94a001044d9a0f24e4f2b1384014495f835e1.

| Arm | Parameters | Selected return | Final return |
| --- | ---: | ---: | ---: |
| Control64 |61,513|19.6245%|18.4844%|
| Larger128 |221,321|28.1508%|13.4278%|
| Matched holding |—|30.6019%|30.6019%|

Separate26-symbol mean net total returns on repeatedly reused2021–2023
 development history; neither annualized nor a combined portfolio or independent
evidence. Larger selected at60k versus control80k. Selected improvement8.5263pp,
final change−5.0565pp; final stock gain−8.9396pp and ETF gain+16.3002pp versus
control. Mean drawdown worsens0.5327pp. Larger selected stock excess−3.2133pp
misses−2pp; final stock/ETF excess−19.7987/−2.7383pp and median stock
excess−13.4832pp fail. Fixed capacity family closes: no width/depth/step grid,
extension, confirmation or promotion. No robust accepted winner established.

Strict source/dataset/accounting/holding checks pass. Both selected/final actual
weights, settings and Adam states restore; each final has20,000 actual Adam
steps/1000 nominal epochs. Same actual111-symbol training sampling and common
initial CPU RNG verified, with different pinned per-width tensors as declared.
156 screen frozen action/hidden/cell/repeat/reset predictions pass;312 including
smokes. Normalization immutable, schema/feed/nonfinite rejection passes. Retired
2024+ data unopened; no fitting by local reports or broker orders.

All four owned smoke/screen sessions stopped and cleanup verified. Official
account349.34, zero rate/assignments. Cumulative conservative project38.60166524,
remaining111.39833476; study increment0.08678784 includes setup/idle/downloads/
shutdown. Observed account debit0.07 versus conservative charge0.08678784.
Project150/130 and account150/170 safeguards unchanged; original ledger retained.


[Evidence](results/recurrent-capacity-2026-10-22/README.md).


Capacity publication verified:15 GitHub assets match server sizes/SHA-256,
four accepted model bundles and interactive reports retained. Only two duplicate
publication archives removed (45,880,597 bytes),142 retained hashes unchanged.
Original transports/source/data/models/checkpoints/ledger/plan.html preserved.
All four sessions stopped; project38.60166524 unchanged. Fixed family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/recurrent-capacity-2026-10-22).

2026-10-23: training-history composition protocol and immutable2016-onward
suffix prepared.111 symbols/137,529 training rows,19,058 unchanged development
rows; original features/normalization frozen. Measured year logs implemented,
23 subset/coverage checks pass. Fixed same64-unit100k full/recent comparison,
unchanged gates and2-unit allowance from38.60166524; dedicated Colab integration/
actual construction/smokes required before training. No allocation or performance
claim. [Method](docs/training-history.md), [preflight](results/training-history-2026-10-23/preflight.json).

2026-10-23: training-history actual two512 CPU smokes accepted.156 exact frozen
checks, measured year margins, equal starts and restored model/optimizer proof;
both sessions stopped. Runtime projection3504s passes7200s; same-source two100k
seed11 screen next. Project38.62221390/150, account349.32 zero rate/assignments.
[Evidence](results/training-history-2026-10-23/README.md).


## Completed history-composition screen — 2026-10-23

Status: fixed seed11 comparison failed; family closed. No confirmation,
extension, start-year/window grid, promotion or performance goal achieved.
Both fresh100k runs completed in Google Colab CPU/defaultPython3.13.16 from
frozen clean2dfa1fbd45d6a3c120fb11311c9d285781556b57. Earlier sections retain
preparation/smoke history; this is the latest outcome.

| Arm | Selected return | Selected step | Final return |
| --- | ---: | ---: | ---: |
| Full2010–2020 |19.6245%|80,000|18.4844%|
| Recent2016–2020 |20.9523%|50,000|5.5976%|
| Matched holding |30.6019%|—|30.6019%|

Mean net total returns across separate26-symbol episodes on repeatedly reused
2021–2023 development history; neither annualized nor a combined portfolio or
independent test. Candidate selected improvement1.3278pp misses the2pp gate;
final change−12.8868pp fails. Candidate final stock/ETF4.4602/11.8533% versus
holding32.6265/19.4660%. Its final stock change−17.3073pp versus control fails
the nonworsening group gate; ETF improves11.4258pp. Mean drawdown worsening
0.0127pp is within risk tolerance but cannot compensate for failed return gates.
Both phases fail stock/ETF/median−2pp absolute targets. Selected stock/ETF excess
−10.3746/−5.6617pp, median−8.5500pp; final−28.1664/−7.6127pp, median−15.6457pp.
Reporting-only declining-stock joint positive total AND net short success1/3
selected→0/3 final; no robust profitable-short capability established.

## Actual training years and verification

Measured2020 execution share rises as intended: full stocks1.978%/ETFs3.005%,
recent stocks6.637%/ETFs8.263%. These realized finite samples differ from exact
complete-window expectations. Per-symbol/global/decision/execution/pair margins
reconcile; actual symbol draw counts match. Recent history changed composition
without providing stable development improvement. It does not prove all recency
methods fail, or establish a sole cause of deterioration. This fixed family is
closed; a future study needs a distinct preregistered hypothesis, not a year grid.

Full control reproduces the prior capacity control's exact selected/final tensor
hashes and symbol sampling, supporting instrumentation invariance. It was trained
fresh, not resumed or selected from old weights. Both arms restore identical
pinned untrained tensors/RNG. Strict source/data/accounting/holding checks and
actual selected/final models/Adam records restore. Final20,000 actual Adam steps,
1000 nominal epochs,61,513 parameters.156 screen frozen recurrent action/state/
repeat/reset checks pass;312 including smokes. Normalization remains immutable;
schema/feed/nonfinite rejection passes. Retired2024+ unopened; no local fitting,
new provider call or orders in reporting.

## Compute, artifacts and next work

All four smoke/screen sessions verified stopped. Official account349.27, zero
rate/assignments. Original project ledger38.68329334/150, remaining111.31670666;
study conservative increment0.08162810 versus observed account debit0.07,
including setup/idle/transfers/shutdown. Original baseline38.60166524433091/shared
2-unit allowance and project150/130/account150/170 safeguards retained.

Strict summary, compact10k learning curves with incremental changes, measured
year CSV, restored models, checkpoints, complete reports/source/datasets/ledger
retained. Interactive comparison is `artifacts/training-history/summary.html`.
Verified GitHub publication and duplicate-only cleanup follow this outcome commit.
Next diagnose accepted saved paths before choosing a distinct experiment; no
automatic training restart or retired-test evaluation.

[Evidence](results/training-history-2026-10-23/README.md).


Training-history publication verified:17 GitHub asset sizes/digests match, four
frozen bundles and interactive reports public. Only two duplicate publication
archives removed (35,931,249 bytes),148 retained hashes unchanged. Original
transports/models/source/both datasets/ledger/plan.html retained. All four sessions
stopped, project38.68329334 unchanged. Fixed family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/training-history-2026-10-23).

2026-10-23: distinct fixed recurrent PPO L2 protocol prepared,0 versus0.0001 on
unchanged full-history source/64-unit model/100k. Same untrained tensors and
explicit eps1e-5/coupled Adam verified without fitting/data access. Shared2-unit
allowance from38.68329334; unchanged gates. Dedicated integration/checkers/smokes
remain before allocation; no failed-family sweep or performance claim.
[Method](docs/recurrent-regularization.md).

2026-10-23: recurrent L2 actual two512 CPU smokes accepted.156 frozen predictions,
actual coupled coefficient/epsilon/moments/model/sampling/year/checkpoint proof;
both stopped. Runtime3426.6427s passes7200s, exact acceptance binds100k plans.
Same-source two100k seed11 screen next; project38.70365471/150, account349.26
zero rate/assignments. [Evidence](results/recurrent-regularization-2026-10-23/README.md).


## Completed coupled-L2 screen — 2026-10-23

Status: fixed recurrent PPO L2 comparison failed; family closed. No coefficient,
optimizer/model/reward/step grid, extension, confirmation or promotion. Earlier
sections retain preparation/smoke history. Both100k seed11 CPU runs completed in
Google Colab/defaultPython3.13.16 from clean frozen
1d640eebf0ad604461d0d510f584477efee75844.

| Arm | Selected return | Selected step | Final return |
| --- | ---: | ---: | ---: |
| Control decay0 |19.6245%|80,000|18.4844%|
| Coupled L2 decay0.0001 |30.4487%|50,000|18.4155%|
| Matched holding |30.6019%|—|30.6019%|

Separate26-symbol mean net total returns on repeatedly reused2021–2023
development history; neither annualized, combined portfolio nor independent test.
Selected relative gain+10.8241pp passes2pp, final−0.0689pp fails. Final stock
change−2.4975pp fails nonworsening; ETF+13.2886pp improves. Mean symbol drawdown
improves1.3741pp but cannot compensate for failed return gates.

Candidate selected stocks33.9981% versus holding32.6265%, but ETFs10.9265%
versus19.4660%; selected median stock excess−3.1679pp. Thus the near-matched
selected overall mean does not pass separate stock/ETF/median−2pp gates. Final
stocks19.2699%/ETFs13.7160%, excess−13.3566/−5.7500pp and median−6.7186pp,
all fail. Reporting-only declining-stock joint positive total AND net short P&L
1/3 selected→0/3 final. No robust profitable-short capability established.

## Actual optimizer/data/frozen evidence

Both actual selected/final models, moments, checkpoint IDs and default accounting/
holding paths restore and reconcile. Actual coupled Adam defaults and group
flagsFalse, coefficient0/0.0001 and explicit eps1e-5 verified. Final20,000 actual
Adam updates/1000 nominal epochs,61,513 parameters each. Exact same pinned
untrained tensors/CPU RNG, actual symbol sampling and measured year margins pass.
Fresh control reproduces prior full-history control's exact selected/final
tensor hashes and sampling, supporting preserved epsilon/default behavior; model
ZIP bytes need not match across saved provenance. No trained weights reused.

156 exact screen frozen action/hidden/cell/repeat/reset predictions,312 including
smokes; immutable normalization/schema/feed/nonfinite rejection pass. Source/data/
reward/action/model architecture/steps remain fixed. No retired2024+ access,
local fitting, new provider calls or broker orders in report generation.

## Compute, retained artifacts and next work

All four smoke/screen sessions verified stopped; official account349.23, zero
rate/assignments. Original project ledger38.74617914/150, remaining111.25382086;
study conservative increment0.06288579 versus observed rounded account debit0.04,
including setup/idle/transfers/shutdown. Original baseline38.68329334472413/shared
2-unit allowance and project150/130/account150/170 safeguards retained.

Strict summary, compact10k curves with incremental changes, actual optimizer/
initialization/year/checkpoint records, full models/reports/source/dataset/ledger
retained. Interactive comparison: `artifacts/recurrent-l2/summary.html`. Verified
GitHub publication and duplicate-only cleanup follow this outcome commit, then
saved-path diagnosis before any distinct hypothesis. No automatic restart or
relaxed gates. The full research objective remains unachieved.

[Evidence](results/recurrent-regularization-2026-10-23/README.md).


Recurrent L2 publication complete:16 server assets verified, four model bundles
and interactive reports public. Only two duplicate archives removed (35,744,305
bytes),145 retained hashes unchanged; original source/data/models/checkpoints/
ledger/plan.html preserved. Strict summary reproduces byte for byte from retained
transports with actual models/Adam/coupling restored. All four sessions stopped;
project38.74617914 unchanged. Fixed L2 family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/recurrent-regularization-2026-10-23).


L2 crossover native control rejected fixed tolerance before any mixed case.
Bounded diagnostic retry retained exact failure; direct unwrapped replay equals
wrapper result exactly. Both differ from original Colab by at most$0.00467059
daily equity, with no structural/count/quantity/date mismatch. Runtime rounding
is plausible, not established cause. No tolerance widening, mixed-policy returns,
hybrid selection/promotion or family rescue.41 tests pass; source/model/optimizer/
normalizer unchanged, no training/provider/Colab/test access or new credits.
[Evidence](results/l2-crossover-2026-10-23/README.md).

2026-10-23: fixed cap5 market-input protocol/preprocessing/payload prepared.
Candidatec895d6af… preserves raw111/26 data/means/scales, explicit new schema;
legacy defaultc2e6c58c… identity and default bundles unchanged.83 focused tests
and actual untrained frozen/accounting checks pass. Original cap/gates/budget
unchanged in finalized protocol; dedicated Colab integration/checkers/smokes
remain before training. No allocation/new units.
[Method](docs/feature-preprocessing.md), [evidence](results/input-clipping-2026-10-23/README.md).

### 2026-10-23 — fixed clipping integration

Version30 input clipping plans/worker/runner/artifact checks, actual model and
optimizer records, initialization snapshots, strict report and smoke acceptance
implemented.156 focused checks pass, and both immutable datasets pass full64
untrained construction/environment checkers. No local training or retired-test
access. Colab512 smokes/runtime approval required before fixed100k paired screen;
original gates,2-unit study allowance,150/130 project and150/170 account safeguards
unchanged. Overall robust-return objective remains unachieved.

Actual clipping smokes subsequently accepted: two512 CPU Colab runs/defaultPython
3.13.16,156 exact frozen predictions, actual Adam/models and matched initialization/
sampling restored. Both owned sessions retrieved/stopped. Project38.76724093,
account349.22 zero assignments. Runtime projection3970.4465s permits prepared fixed
100k screen plans; screen not yet allocated, no clipping return claim.


### 2026-10-23 — fixed market-input clipping screen failed

Two100k Colab CPU runs completed; control selected/final19.6245/18.4844%,
clip5 selected/final24.7028/18.5720%, matched holding30.6019%. Selectedrelative
+5.0783pp but final+0.0877pp fails2pp; stock nonworsening and all absolute
stock/ETF/median gates fail. Family closed, no cap grid/extension/confirmation.
312 frozen checks including smokes pass; actual models/Adam/initialization/year
sampling restore; fresh control exactly reproduces prior tensors/sampling.
Control CLI disconnect recovered via advancing checkpoints/final export without
restarting training. All four owned sessions retrieved/stopped. Project38.81646887,
account349.17 zero rate/assignments. Full artifacts retained; publication and
saved-path diagnosis next, robust-return objective unachieved.

[Detailed evidence](results/input-clipping-2026-10-23/README.md).


Clipping saved-action audit: four accepted phases/104 episodes,31 focused tests.
Selected→final closing short exposure stock3.47→6.87%/ETF1.15→10.59%; net short
contribution−0.99→−4.04pp/+1.73→−5.98pp, incremental cost drag5.11→5.92pp/
4.90→6.50pp. Observational relationships, not causal attribution or justification
for disabling shorts/reward edits. Exact source hashes preserved; no fitting/data
access/Colab allocation. Family remains closed; distinct hypothesis review before
new training. [Evidence](results/clipping-actions-2026-10-23/README.md).


### Completed action-boundary diagnostic

Three fixed Gaussian policies ×111 training prefixes ×32 observations audited
without fitting/policy-action execution/validation-return scoring. Conditional
any-component mass beyond[-1,1]:untrained78.28%, selected79.65%, final79.38%.
Final marginals exposure36.23%, stop32.44%, near37.67%, width24.06%. Full model/
Adam/RNG/normalizer/source invariants pass;54 focused tests. These are conditional
probabilities, not historical rollout frequencies or a causal return diagnosis.
Next separately preregister a same-network squashed-diagonal-Gaussian comparison;
do not conflate with failed market-input clipping or add gSDE simultaneously.
No allocation or extra units. [Evidence](results/action-boundaries-2026-10-23/README.md).

### 2026-10-23 — squashed-action implementation and integration

Fixed protocol2195938b… implemented with explicit tanh diagonal-Gaussian rollout/
likelihood/Jacobian, same61,513 parameters/full64 tensors and original data/reward/
optimizer. Version31 worker/runner/strict artifacts/report and runtime gates added;
125 focused checks plus separate62 overlapping controller/frozen/hybrid checks pass.
Actual immutable source/checkers/full64 forwards pass. No local optimizer updates.
Two512 Colab smokes remain before100k; project38.81646887/account349.17 safeguards
unchanged. [Evidence](results/squashed-actions-2026-10-23/README.md).


### Actual squashed-action Colab smokes accepted

Both512 CPU/defaultPython3.13.16 runs from cleanfa63d6c… complete;156 exact frozen
recurrent sequence/reset predictions, immutable normalization, actual saved models/
Adam, distribution flags/Jacobian and identical initial tensors/RNG/symbol/year
sampling pass.80 actual updates/40 nominal epochs,61,513 parameters each. Both
owned sessions retrieved/stopped. Runtime129.7219s control/159.4628s candidate gives
fixed3472.1638s projection including1200s reserve, below7200s. Exact accepted smoke
bytes bind prepared100k plans, not yet allocated. Project38.83857572/account349.16
zero rate/assignments, shared2-unit allowance/original project/account limits
unchanged. No smoke-return selection, promotion or retired-test access.
[Evidence](results/squashed-actions-2026-10-23/README.md).


### 2026-10-23 — fixed squashed-action screen failed

Two100k Colab CPU runs completed; native control selected/final19.6245/18.4844%,
squashed15.4740/5.1381%, matched holding30.6019%. Relative and absolute stock/ETF/
median gates fail; final stock6.4716%/ETF−2.1964%. Fixed family closed with no
transform/temperature/log_std grid/extension/confirmation.312 frozen checks
including smokes, actual model/Adam/distribution/initialization/year proofs pass;
fresh control exactly reproduces prior tensors/sampling. All four owned sessions
retrieved/stopped. Project38.90354117/account349.10 zero assignments. Large
conditional boundary mass did not demonstrate a causal cure or policy improvement.
Full artifacts retained; publication and saved-path diagnosis next. Robust-return
objective unachieved. [Evidence](results/squashed-actions-2026-10-23/README.md).


Squashed saved-path diagnosis: four accepted phases/104 episodes,32 focused tests.
Selected→final closing short exposure stock4.59→10.14%/ETF4.34→9.26%; net short
contribution−5.38→−11.18pp/−1.24→−5.41pp; long contributions also deteriorate.
Incremental cost drag3.88→5.05pp/3.54→4.56pp is not the whole gap. Observational
relationships, not a cause or justification for disabling shorts/reward edits.
Exact source hashes preserved; no fitting/data access/Colab allocation. Fixed
family closed; distinct hypothesis review before new training.
[Evidence](results/squashed-actions-diagnosis-2026-10-23/README.md).

### 2026-10-23 — compact broad-market context coverage

Fixed three-field SPY20-session price return/volatility and own-relative return
reader implemented with matched sessions/availability/known-gap and forward-split
barriers.60 focused checks pass; all298,946 training/19,058 development source
rows available and raw/source/protocol/implementation hashes unchanged. No new
dataset, preprocessing fit, model/inference/performance scoring or Colab units.
Separately preregister matched full/masked/legacy-control study before allocation.
[Evidence](results/market-context-2026-10-23/README.md). Objective unachieved.

Market-context immutable payload stage subsequently passed: masked/full preserve
298,946 training/19,058 development rows, original bars/ATR/features/first-seven
normalizer exact. Full extra-field fit training-only; masked zero/identity and
frozen rejection of nonzero masks.48 focused checks and156 actual legacy frozen
predictions pass; no local policy learning/updates or allocation. Shared untrained
initialization assets, final protocol and dedicated training integration/smokes
remain. [Evidence](results/market-context-2026-10-23/payload-preflight.json).

Market-context shared initializations/final protocol subsequently frozen: three
mapped24-input untrained snapshots, original common/input/account tensors and CPU
RNG exact; full/masked identical63,049 parameters, legacy61,513.54 focused checks
pass; no observed-data access/fitting/optimizer updates or allocation. Final
protocolf9045fe8… preserves preregistered settings. Dedicated Colab integration,
actual source checkers and three512 smokes remain. [Evidence](results/market-context-2026-10-23/initialization.json).

Market-context version32 integration subsequently implemented: three-arm protocol,
plans/worker/runner/artifact/report and required actual masked-input frozen check.
All immutable-dataset model/environment checkers pass;53 initial new,156 shared/
legacy and updated121 focused checks pass with overlap. Three512 CPU smoke plans
prepared, actual Colab acceptance still required before100k. No local learning/
updates, project38.90354117/account349.10 unchanged. [Evidence](results/market-context-2026-10-23/construction.json).


### Actual market-context Colab smokes accepted — 2026-10-24 UTC

Three512 CPU/defaultPython3.13.16 runs from clean88fb98c… complete.234 exact frozen
sequence/reset predictions and actual masked nonzero-input rejection pass; saved
models/Adam, mapped initial tensors/RNG and actual symbol/year sampling restore.
80 actual updates/40 nominal epochs each, legacy61,513/augmented63,049 parameters.
All three sessions retrieved/stopped. Durations138.0922/138.7362/136.6898s give
fixed3406.0216s projection including1200s reserve, below7200s. Exact accepted bytes
bind prepared three100k seed11 plans, not allocated. Project38.93466769/account
349.08 zero rate/assignments; original2-unit allowance and project/account limits
unchanged. No smoke-return tuning or retired-test access. [Evidence](results/market-context-2026-10-23/README.md).


### Compact market-context screen failed — 2026-10-24 UTC

Three100k Colab CPU runs complete: legacy selected/final19.6245/18.4844%, masked
18.7314/18.7314%, full7.6978/−3.7731%, holding30.6019%. Full fails both-control
relative/stock/ETF/risk and all absolute/median gates. Final stock−3.4809%/ETF−5.3803%;
drawdown worsens4.7250/4.9084pp versus controls. Family closed, no benchmark/window/
subset/model/reward/step grid or extension/confirmation.468 frozen checks including
smokes, actual models/Adam/mapped tensors/RNG/year sampling restore; legacy control
reproduces prior native tensors/sampling. All six sessions retrieved/stopped.
Project39.01437789/account349.02 zero assignments. Coverage did not imply alpha;
full artifacts retained, publication/diagnosis next, objective unachieved.
[Evidence](results/market-context-2026-10-23/README.md).


Market-context saved-action audit: six accepted phases/156 episodes,33 focused
checks. Full selected→final short exposure decreases stock15.12→8.55%/ETF4.65→1.53%,
but net short contribution remains negative/slightly worsens; net long contribution
falls19.74→8.53pp/5.69→−1.64pp. Cost drag4.96→5.96pp/3.20→4.63pp is not the whole
gap. Observational accounting, not a sole cause or reason to ban shorts/change
rewards. Source hashes unchanged; no fitting/data/provider access or new units.
Family closed; distinct hypothesis review before new allocation.
[Evidence](results/market-actions-2026-10-24/README.md).

### Filing-family feasibility update — 2026-10-24

SEC acquisition/structural inspection and current-ID discovery are implemented;
51 focused checks pass. First2020q1 archive retained, strict future-period failure
preserved, original forecast context resolved, explicit forward-period quarantine
inspected3,096,249 numeric records. Current ticker discovery107/107 yields102
candidate CIKs represented in that archive, with5 absences requiring source/issuer
review. No historical identity map, causal financial dataset or financial model
has been accepted. Publication/vintage/concept/coverage checks and a separate
matched training protocol remain mandatory. No Colab units used for this family.
See[filing guide](docs/filing-data.md) and[source evidence](results/filing-source-review-2026-10-24/README.md).

SEC source collection update:59/59 registered quarterly archives retained,
4.61GB/601.7sec. Chronological SUB inventory verifies366,856 unique records,
342,981 qualifying financial filings/14,100 issuers, no cross-archive accession
conflicts.81 focused checks pass; candidate-year visual report saved. Source
coverage is not financial-feature acceptance: issuer lineage, exact concepts/
periods/vintages and public availability remain pending. No financial model
training or additional Colab units. Source retention/terminal collection verified.

Master-index source update:59/59 registered compressed indexes collected,
186.76MB/157.9sec. Comparison scans14.88million metadata rows and finds6,412
issuer/accession pairs/6,410 unique documents for112 candidate CIKs. All6,161
financial-table pairs recovered;251 extra pairs include249 additional documents
and2 joint-filer links. Six date discrepancies retained, no form conflict.
131 focused checks pass. This is current-index metadata scope completeness,
not original-fact/vintage/price-link approval; financial models remain disabled.
Source retention/terminal cleanup verified, no additional Colab units.

Financial source-concept update:fixed metadata audit of148.96million NUM rows
complete. Required current concepts all present/commonflow in43.7% of6,161
assessed pairs; direct liabilities62.9%,251 master-only pairs unassessed. These
are not original arithmetic/price coverage results.144 focused checks pass,
local workers stopped, evidence retained. Accounting-identity/original-concept
validation remains before bulk extraction or model training; no new Colab units.


Original balance-identity audit: four retained IBM/Kroger annual instances reproduce
exactly; three Assets−Liabilities−equity-including-NCI residuals are zero, IBM2019
is −$1million and remains nonzero. Parent-equity residuals differ in all four.
Missing temporary equity is unknown, and missing liabilities are not inferred.
Default money extraction/financial rules unchanged; 152 focused checks and targeted
Ruff pass. No training/new Colab units. Financial dataset acceptance and a robust
model winner remain unachieved. See [accounting evidence](results/filing-source-review-2026-10-24/balance-identity-inspection.json).


Fixed original-source missing-liabilities probe: earliest of33 eligible2020 annual
filings is Intel0000050863-20-000011. Original end2019-12-28 differs from archive
20191231; original also lacks qualifying current direct liabilities, so unchanged
snapshot remains unready. One case only; no general coverage estimate, imputation,
financial-input activation, model return or Colab units. Offline source/selection/
report reproduction verified. [Evidence](results/filing-source-review-2026-10-24/missing-liabilities-probe.json).


Distinct flow-only financial family fixed/implemented: same-accession comparable
revenue growth and parent margin, no liabilities substitution or income aliases.
Five retained originals yield3 arithmetic-ready/2 unready; legacy IBM/Kroger
snapshots unchanged, ready ratios independently checked by Fraction arithmetic.
Current common-flow tag presence4,512/6,161 (73.2%) is metadata only, not causal
stock coverage.162 focused checks/Ruff pass; family remains disabled, no training
or Colab units. Original-source/issuer-price-day coverage gates remain before any
matched model experiment. [Evidence](results/filing-source-review-2026-10-24/flow-feasibility-inspection.json).


Cross-era original-flow pilot preregistered/prepared:319 selected annual instances
across112 candidate CIKs/three fixed eras,17 absent issuer-era slots.176 focused
checks pass. Sequential bounded acquisition, terminal journal and no automatic
restart;80%overall/70%per-era source arithmetic gate fixed before collection.
No financial dataset/training approval, original vintage/price links unresolved.
See[protocol](configs/experiments/filing_original_flow_pilot.json).


Original-flow pilot first collection stopped at the20MB source cap on eighth
selected document(AEP0000004904-15-000008). Seven complete originals/26,924,787
bytes retained and reverified;6 arithmetic-ready/1 unready are not a coverage
estimate.312 cases unfinished, gate not evaluated. Terminal failure preserved,
collector exited, no retry/replacement/restart or Colab units.183 focused checks
pass; source hashes/extraction/snapshots reproduce. A reviewed continuation must
retain the failure and count oversized unsupported sources in the fixed sample.
[Evidence](results/filing-source-review-2026-10-24/original-flow-pilot-summary.json).


Reviewed original-flow continuation implemented: separate journal preserves the
initial source-cap stop, counts oversized selected sources as unsupported without
retry/replacement and retains original sample/20MBcap/80%overall/70%era gates.
Original3600s wall-clock deadline and byte accounting preserved.187 focused
checks pass; initial terminal report unchanged. No training/Colab allocation.

Distinct critic-algorithm review: installedSB3Contrib2.7.0 providesTQC, but the
project factory/runtime do not yet support it. Primary-source review distinguishes
value-distribution/truncation research from closed SAC parameter families. No
trading evidence or overestimation-cause claim; fixed saved-policy calibration
review is required before any study protocol or training allocation. See
[review](docs/distributional-critics.md). No model fitting/new Colab units.


Cross-era original-flow pilot complete, **failed**:210/319 ready(65.83%) versus80%
overall; eras69/104(66.35%),66/107(61.68%),75/108(69.44%) all miss70%.297 inspected
include67missingflow/19conflict/1prior failure;19oversized+3unsupported stay in
fixed denominator.300 complete originals/1.44GB retained;210 ratios independently
verified, all297 snapshots reproduce. Initial stop preserved; collector exited0,
locks released/no task temporaries.188 checks/Ruff pass. No bulk financial data
acceptance, training, threshold loosening or new Colab units. A distinct critic
calibration review is the next candidate, not a promoted model. [Evidence](results/filing-source-review-2026-10-24/original-flow-pilot-reviewed-summary.json).

Saved-SAC soft-consistency diagnostic fixed/implemented: exact original source
export/model/data identities;64 flat training reset states×4 stochastic repeats,
original512 horizon, proper future entropy/target-critic timeout bootstrap.11
focused tests/Ruff pass; no local optimization or Colab allocation. Descriptive
conditional inconsistency flag is not true-Q calibration, trading return or model
promotion. Actual run pending. See[protocol](configs/experiments/sac_soft_consistency.json).

Saved-SAC soft-consistency complete: mean minimumQ4.031877 versus multistep soft
target0.234971 on64 fixed flat training reset states×4 stochastic repetitions.
Descriptive gap lower-two-SE3.625062≥0.1 and204/256 full512 trajectories≥75%, flag
passes. All64 window means positive; soft units, not trading returns or true-Q
calibration proof. All weights/optimizer states/updates/normalization/RNG and
sources unchanged; worker exited0/environment closed, zero local optimization or
Colab allocation.199 combined focused checks pass. Next preregister a distinct
three-arm critic study and implement its required acceptance, then Colab smokes;
no TQC model/promotion exists. [Evidence](results/sac-soft-consistency-2026-10-24/README.md).

Distinct distributional-critic study preregistered after the saved-policy flag:
nativeSAC versus2x25quantileTQCdrop0/drop2, matched original26-symbol source,
50k seed11/10k curves and fixed shared actor/nativeRNG acceptance. Candidate must
pass relative gains versusbothcontrols plus unchanged absolute stock/ETF/risk
gates; onlythen11/23/37/prospective.2-unit incremental allowance inside existing
project/account/owned-deadline limits. No TQC construction/training, implementation
and Colabsmokes pending. [Protocol](configs/experiments/distributional_critics.json).

Distributional-critic base implementation: TQC factory/configuration, strict
quantile/truncation validation, actual critic/target geometry and mandatory frozen
critic declaration. Existing SAC defaults/weights/RNG preserved; SAC/TQC actors
and TQC0/TQC2 initial tensors verified at32/64 widths.23 TQC checks within42
passing algorithm/frozen checks; no optimizer steps/Colab allocation. Nonempty
Adam serialization fixtures are not learned acceptance. Shared nativeRNG payload,
study/runtime/archive/gate integration and actual Colabsmokes are next. No TQC
trained model or performance result. [Guide](docs/distributional-critics.md).

Distributional-critic initialization accepted:12 pinned untrained archives for
three arms×(smoke11/full11/23/37). NativeSAC standard initialization unchanged;
actors identical, TQC0/TQC2 full tensors/empty optimizers identical, globalTorch/
NumPy/Python native sampling RNG restored. Fresh off-policy/replay/seed/unit/Adam
checks reject before mutation;61 focused tests pass, no optimizer steps or market
data access. Binding/source/package/hash evidence retained. Training metadata,
task/archive/controller integration and actual Colabsmokes still pending. No
new model return or Colab units. [Evidence](results/distributional-critics-2026-10-24/README.md).


Distributional-critic version33 integration ready: training/task/archive/fleet
contracts, actual saved-model/replay restoration and persistent shared two-unit
study baseline implemented.12 checksum-identical initializers now ship in tracked
configs.115 focused checks pass without local optimization; actual Colab smokes
and all new performance gates remain pending. No new return or promoted model.
See[study guide](docs/distributional-critics.md).


Distributional-critic actual Colab smokes accepted:3×1024 CPU transitions, actual
Python3.13.16, initial tensor/RNG and learned Adam/replay restoration,192 exact
frozen predictions and immutable normalization. All3owned sessions stopped.
Conservative project39.039379/150, incremental0.025001, official account349.00
with zero assignments. Fixed screen runtime projection3717.83<7200seconds;
50k seed11 three-arm plans prepared. No completed screen, return improvement,
retired-test access or promotion. Evidence:results/distributional-critics-2026-10-24/smokes.json.


Fixed critic50k seed11 screens launched in three centrally metered Colab CPU
sessions from immutable8c27af4 source after actual smoke acceptance; all actual
kernels report3.13.16. No completed performance result yet. Reporting now verifies
owned-cleaned archives, actual replay/optimizer/frozen state, matched baseline
paths, selected/final curves and declining-stock diagnostics, checksum-pinned
historical SAC metadata and unchanged0.01pp/0.05pp reproduction tolerances.
118 focused checks/Ruff pass; report-time checks perform no local learning.
Confirmation remains blocked on all fixed gates, not elapsed training steps.


Completed fixed critic50k screen failed: nativeSAC selected/final7.9281/7.9281%,
untruncatedTQC1.9991/1.9991%, truncatedTQC9.7018/7.6975%, holding30.6019%.
Candidate fails both-control/absolute/risk gates; no confirmation, grid or
extension. NativeSAC reproduces every checkpoint and selected/final symbol
exactly.384 smoke/full frozen comparisons and actual learned optimizer/replay
checks pass; six owned sessions stopped. Conservative project39.154379/150,
study0.140001/2, official account348.89 with zero rate/assignments. Retired-test
isolation and immutable normalization preserved; objective remains unmet. See
results/distributional-critics-2026-10-24/screen.json and compute.json. Publication
and scoped duplicate cleanup remain next; no promoted model.


Distributional-critic publication/cleanup verified:12 GitHub release asset hashes
and sizes match;18 duplicate transports/upload payloads removed (520,262,328
bytes),191 retained files rehashed at cleanup. All models/replay/checkpoints,
source/data, curves and owned recovery records retained. Subsequent README
publication update is explicitly recorded in cleanup evidence. Six owned sessions
stopped, zero assignments; no additional units or training. Fixed critic study
failed and remains closed; next work is bounded saved-policy/execution sensitivity
diagnosis, not extension or promotion. Release:
https://github.com/jason490/RL-trading/releases/tag/distributional-critics-2026-10-24


[Saved critic final-update sensitivity](results/critic-update-2026-10-24/README.md) complete: three one-update pairs on
256 fixed candidate training-replay observations; truncatedTQC actor relativeL2
0.03356%, exposure mean absolute delta0.001462, GaussianKL0.001133 and zero
exposure-sign flips. These are normalized action/tensor diagnostics, not returns
or a cause of development deterioration. Independent Torch arithmetic agrees;
all protected files/model/Adam state unchanged, worker exited0, no local
optimization or Colab units. First phase-filename rejection and reviewed v2
correction retained; sample/metrics unchanged. Fixed failed family remains
closed. Next bounded saved-development-path audit before another training
hypothesis; no promotion or retired-test access.


[Saved TQC path audit](results/critic-paths-2026-10-24/README.md) complete: all26 selected/final development paths
reconcile−2.004248pp. Stocks closing quantities differ72.94% of sessions, sides
2.97% and protective reasons2.24%; stock long contribution change−2.5417pp, short
−0.1624pp. Same-day partitions are observations, not counterfactual causes or
proof that earlier protection effects are absent. Initial legitimate-null
rejection and reviewed schema correction retained.27 tests and independent
NumPy verification pass, worker exited0/source unchanged, no models/training/
new Colab units or retired data. Current account348.89,0rate/assignments; project
39.154379/150. Failed critic family stays closed. Next candidate is a bounded
source/availability audit of historical option-implied volatility, not another
architecture/steps rescue; no new inputs or training authorized by feasibility
evidence alone.


[Implied-volatility source feasibility](results/vix-source-2026-10-25/README.md) passed: single date-bounded Yahoo
request retained3,774 observations2009–2023; lagged level/change20 context covers
all3,433 original-dataset decisions (100% overall/eras/years), with exact prior
sessions and assumed availability no later than decisions. Four tests and
independent pandas/source/dataset/normalizer verification pass; worker exited0.
Historical publication vintages unverified; source values remain local and model
inputs disabled. No training, new return or Colab units. Next audit the accepted
111-symbol consumer and preregister a separate matched feature comparison only
after coverage; failed families and retired-test isolation remain unchanged.


[VIX information comparison preregistered](results/vix-context-2026-10-25/README.md) after expanded-consumer coverage
passes all298,946train/19,058development rows (111/26symbols,100% unique-date/
era/year readiness). Independent pandas/source/normalizer verification passes;
consumer worker exited0, no requests or new units. Fixed legacy/masked/full
21/23-input recurrent controls,100k seed11 steps, original common tensors/RNG
and zero new LSTM columns, immutable seven-field preprocessing, fixed both-
control/absolute/risk gates and historical reproduction. Only passing512smokes
then screens permit11/23/37/prospective; study2units inside original budget.
Nine tests pass; no new policy construction, training, performance score or
Colab allocation. Payload/initializer/frozen/runtime implementation is next;
historical vintage and retrospective universe limitations remain unverified.


[VIX immutable payload/initialization acceptance](results/vix-context-2026-10-25/README.md): full/masked payloads
reloaded and exactly reconstructed; original bars/ATR/splits/seven fields and
preprocessing unchanged. Only full's2 new fields fit298,946 training rows;
independent pandas/NumPy verification passes.12 initializer entries/3new23-input
archives retain exact original common tensors/CPU RNG and zero new actor/critic
columns; seeded NumPy/Python fingerprints verified. Binding/portable artifacts
are pinned,70 focused tests pass, source workers exited0. No policy optimization
or Colab units. Learned/runtime acceptance remains pending. Next explicit family
dispatch plus task/archive/report/runtime integration and actual Colabsmokes;
legacy-overlap must not change previous controls. No new performance result.

VIX runner integration now enforces explicit family dispatch and fixed data,
preprocessing, request and shared-start contracts before model construction.
Initial/selected/final tensor and Adam evidence plus a checksummed initialization
snapshot are retained.90 focused non-optimizing checks pass, including actual
recurrent serialization; job/fleet/archive integration and actual Colab smokes
remain. No new trading return, Colab allocation or promotion at this milestone.

VIX version34 Colab plans, preallocation/remote payload validation, persistent
shared two-unit study baseline, deadline projection and complete archive contracts
are implemented.118 focused non-optimizing job/archive/runner/fleet regressions
pass. Physical learned-artifact restoration and actual512-transition CPU smoke
acceptance remain before100k screens. No allocation or performance result yet.

VIX physical latest/selected/frozen Adam/recurrent restoration and strict smoke
acceptance are implemented: at least78 causal sequence checks per arm, immutable
normalization/data/files and caller RNG/state preservation.93 combined regressions
plus2 acceptance boundaries pass without optimization. Actual Colab512-transition
CPU smokes remain next; serialization fixtures are not learned acceptance.

VIX actual512-transition Colab CPU smokes all pass learned optimizer/recurrent
restoration,234 exact frozen predictions, immutable normalization/data and matched
development accounting. All actual default kernels3.13.16 and all three owned
sessions stopped. Conservative cumulative project39.182288 units; official balance
348.88, zero rate/assignments. Fixed screen projection3471.70s fits7200s; next
predefined100k seed11 comparison and independent historical-control reproduction.
No profitability, confirmation or promotion claimed; retired test stays isolated.

VIX fixed100k seed11 legacy/masked/full screen is running in three centrally
metered Colab CPU sessions from accepted33dbb56 source, each kernel3.13.16.
Strict post-cleanup reporting now verifies historical source/report/checkpoint
reproduction, actual model/Adam/frozen state, matching source/data/sampling and
selected/final accounting before unchanged gates.58 focused checks pass; actual
full-screen outcomes remain pending. No VIX performance improvement claimed.

VIX fixed100k screen completed failed: full selected/final24.09/14.44%, legacy
19.62/18.48%, masked24.36/9.20%, matched holding30.60%. Historical legacy reproduces
exactly at every checkpoint/per-symbol;468 actual frozen sequence checks and
independent NumPy/accounting acceptance pass. All six sessions stopped; cumulative
project39.272458, official account348.80 idle. Fixed family closed without tuning,
extension, confirmation or promotion. Prospective/robust trading objective unmet.

VIX publication/cleanup complete:14 server-verified release assets,13 exact
duplicates/419.93MiB removed,489 retained hashes unchanged. Six differing snapshots
preserved alongside all models/data/source/ledger/reports; all six sessions stopped.
No recovery needed. Fixed family stays closed; distinct saved-evidence diagnosis
must precede another hypothesis. Robust/prospective objective remains unmet.

Next distinct diagnostic is fixed before replay: saved legacy RecurrentPPO
selected/final value-target consistency on26 original training windows ending
2020-12-02,512 transitions each, native stochastic/unrestricted execution.
Protocol pins source weights/data/seeds; no outcome selection, optimization or
Colab allocation.16 focused target, collector, restoration and state tests pass.
Censored/self-bootstrapped targets are not independent calibration proof; all
failed families remain closed. Actual audit execution remains next.

Saved legacy recurrent value-target audit completed52 fixed stochastic training
windows/26,624 rows; worker exited0 in40.44s. Independent NumPy discount/SB3 GAE
verification passes, source model/Adam/data/ledger unchanged. Zero-tail descriptive
EV0.029 selected/−0.084 final, versus self-bootstrapped GAE0.491/0.549; no unbiased
calibration or causal performance claim. No new optimization/Colab units/retired
rows. Next distinct training-only benchmark/noise decomposition before another
hypothesis; robust/prospective objective remains unmet.

Fixed benchmark/noise follow-up is preregistered before execution: reuse the52
saved training traces, matched causal holding rewards, coefficient exactly1,
gamma0.99 and unchanged windows. All eight phase/group/reward/target variance
ratios must be<=0.9 using mean within-episode variances. No fitted coefficient,
grid, replay, model update or Colab allocation. Eight focused covariance/linearity,
corporate-action/bounds and recurrent-target tests pass; actual decomposition next.

Fixed benchmark variance audit completed failed:3/8 conditions pass; ETF reward
and target variation increase in both phases. Independent benchmark accounting
exact, covariance/discount/ratio error1.12e−15, source model/data/ledger unchanged,
worker terminal0 in5.22s. No replay/training/new units. Fixed coefficient-one
subtraction hypothesis closed without grid or allocation; objective remains unmet.

GAE trace comparison preregistered: one lambda1 candidate versus original0.95
native recurrent control, same data/reward/gamma/architecture/initialization/
sampler/100k steps and fixed performance gates. Parameter equality and actual
SB3 finite-buffer telescoping/timeout/boundary tests pass (12 focused checks with
prior diagnostics). No new policy constructed or trained. Shared-start/runtime/
actual-rollout/archive integration and512-transition Colab smokes are next.

GAE shared original full64 starts/model records and standalone actual pre-update
rollout checker implemented;61 focused non-optimizing initialization/buffer/
serialization regressions pass. Timeout, boundary, GAE and lambda1 segmented
return arithmetic validates with compact retained NPZ/JSONL evidence. Runner
request/callback wiring, version35 budget/archive and learned acceptance remain
before Colab smokes. No allocation, new return, source data or promotion.
