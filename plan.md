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
| C.1 | Multi-timeframe supply/demand | Base consolidation plus directional impulse, causal zone lifecycle, ATR-scaled distances/widths, freshness and age; D1/W1 first, M1 later. Implemented opt-in D1/W1; matched ablations and new-model training pending. |
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

## 16. Matched indicator follow-up — authorized, 2026-10-16

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

## 17. Recommended algorithm comparisons — proposed

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

## 17. Diagnose deterioration and compare PPO — authorized, 2026-10-16

After the last indicator cell, diagnose selected/final saved paths and run a
separate six-cell PPO/SAC CPU stability comparison if the evidence supports PPO.
The saved outputs show large directional/exposure changes whose P&L impact exceeds
additional costs; these observations do not prove an optimization cause. The
[protocol](configs/experiments/algorithm_stability.json) fixes basic inputs, the
original reward, 100k transitions, seeds 11/23/37 and fresh paired SAC controls.
No new final-test evaluation, extra reward, hyperparameter search or stock
selection occurs. Finish and publish the indicator study and clean its temporary
duplicates before allocating new sessions. See [guide](docs/algorithm-stability.md).
