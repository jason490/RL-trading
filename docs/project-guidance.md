# Project operating guidance

This guide preserves the detailed guidance moved from `AGENTS.md`. It records research requirements, operating policies, and reference links; the category guides linked in [README.md](README.md) describe implementation details. The project-scope verification statements below are a snapshot at migration; [verification.md](verification.md) records current evidence and remaining work. Standing agent authorization and essential constraints remain in [AGENTS.md](../AGENTS.md).

## Project scope

This repository is intended for modular swing-trading research using Google Colab CLI, interchangeable market-data providers, a custom Gymnasium environment, and PPO, SAC, and TD3. Alpaca is the initial paper-execution adapter, not a required historical-data dependency. The implementation includes provider-independent research modules, a tested simulator, PPO/SAC/TD3 training, reports, frozen research inference, and a budget-controlled Colab runner. Dependency-backed CPU acceptance and remote CPU/A100 smoke acceptance pass; Yahoo daily coverage is verified. The initial A100 PPO run completed 50,000 transitions, seed 11, with downloaded frozen artifacts and verified shutdown. Multi-seed studies and held-out testing remain pending. The user has authorized Colab training within the existing 150-unit project ceiling, with teardown by 130 units and a two-hour initial allocation deadline. Read `plan.md` for milestones and `docs/README.md` for the detailed category guides and actual verification status. Inspect repository contents and nested `AGENTS.md` files before future edits. Preserve user changes, follow existing tooling, and report only checks actually performed.

## Git workflow

- After each completed feature addition or bug fix, run the relevant checks, create a focused commit for that change, and push it to the current branch's configured upstream. Include related tests and documentation in the same commit. This is standing authorization to commit and push; do not ask again for routine feature/fix commits.
- Inspect Git status before staging. Stage only files belonging to the completed change, preserve unrelated user changes, and use a clear commit message describing the feature or fix.
- Do not force-push, rewrite published history, or amend unrelated commits. If committing or pushing fails, preserve the work and report the failure and any unpushed commit IDs; do not claim it was pushed.

## Architecture

- Prefer maintained Stable-Baselines3 implementations and Alpaca's official `alpaca-py` SDK. Check compatible Python, PyTorch, Gymnasium, SB3, and SDK versions together, then pin dependencies.
- Separate data access, causal features, portfolio accounting/environment, algorithm selection, training, evaluation, and broker execution. Keep reusable logic in modules and notebooks thin.
- Select PPO/SAC/TD3 through validated configuration and a shared factory while preserving algorithm-specific settings. Avoid API calls at import time, mutable global state, and notebook-only logic.
- Training and evaluation use local versioned data. Environment `reset()` and `step()` never fetch data or submit orders.
- Keep each category in its own folder/module following the ownership map in `plan.md`: provider adapters, accumulation, validation/storage/generation, individual feature families, accounting/environment/rewards, individual algorithms, training, evaluation, inference, risk, execution, reporting, and runtimes. Add modules incrementally when needed.
- Require typed provider capabilities and canonical schemas. Support bars, trades/quotes, sectors, filings, options, and news independently; optional data or SDKs must not become baseline requirements. Include a local-file adapter and deterministic synthetic generators.

## Custom Gymnasium environment

- Subclass `gymnasium.Env`. `reset(*, seed=None, options=None)` returns `(observation, info)`, calls `super().reset(seed=seed)`, and uses `self.np_random` for environment randomness.
- `step(action)` returns `(observation, reward, terminated, truncated, info)`. Distinguish true terminal conditions from time limits and data cutoffs; preserve correct bootstrapping semantics.
- Declare spaces explicitly. Observations must match shape, dtype, bounds, and finite-value expectations. Prefer float32 numeric features and include relevant portfolio state.
- For a shared PPO/SAC/TD3 interface, use a bounded continuous `Box`, preferably normalized to `[-1, 1]`. SAC and TD3 require continuous actions. Document action-to-position/weight mapping and enforce cash, leverage, shorting, and position constraints.
- Define data availability, decision time, and fill time to prevent look-ahead. Model fees, spread, slippage, turnover, cash, holdings, corporate actions, missing bars, and episode-end valuation consistently; document assumptions.
- Base rewards on portfolio-value changes after costs. Make shaping/scaling explicit and log raw returns separately. Expose accounting diagnostics in `info`.
- Run Gymnasium and SB3 environment checkers. Keep raw Gymnasium and SB3 vector-environment interfaces distinct and use supported adapters.
- Use daily swing decisions initially with next-session execution and completed weekly context later. Keep inference cadence separate from broker-held protective orders and finer-granularity execution replay.
- The final policy outputs signed target exposure, stop distance, take-profit near distance, and zone width. Decode these into position amount, long/short direction, stop, a learned target zone, and add/reduce/close/reverse/hold intent. Start with simpler action versions and explicitly version every change; target zones must be chosen by the policy, not replaced by a fixed reward-to-risk formula.
- Target exposure is desired total exposure, not a repeated order size. Reconcile changes with actual positions. Validate both long and short price geometry and quantity/risk limits; record raw and approved outputs. Zone boundaries must affect execution outcomes, with explicit partial-exit rules.
- Define gap handling and stop/target ordering when both are touched in a bar. Default to conservative ordering when finer data is absent; protective stops do not guarantee their trigger price as the fill.

## Data and evaluation integrity

- Use chronological train/validation/test splits or walk-forward evaluation. Never shuffle observations across boundaries or tune against the final test interval.
- Use causal features, fit preprocessing on training data only, and freeze it during validation/test/inference. Prevent leakage through lookbacks and overlapping reward horizons; add split gaps when needed.
- Validate timestamps, ordering, duplicates, missing bars, calendars, symbol alignment, and corporate-action handling. Store UTC timestamps and handle exchange time zones explicitly. Do not silently fill arbitrary gaps or substitute another data feed.
- Record universe, dates, interval, feed/adjustments, retrieval time, dataset hash/version, and split boundaries. Address survivorship bias before performance claims.
- Compare algorithms using identical data, cost assumptions, seeds, budgets, and evaluation protocol. Include cash/no-trade and buy-and-hold baselines. Report net return, drawdown, volatility, turnover, costs, and Sharpe with annualization assumptions and variability over multiple seeds. Training reward alone is not performance evidence.
- Preserve event time, public availability time, retrieval time, source/version, and revision identity. Backward as-of joins may expose only information available at the decision. Weekly/monthly values require completed periods; filings require public availability and historical vintages; OI and sector/news snapshots require their own as-of rules. Uncertain publication timestamps need conservative treatment.
- Add EMA/RSI/ATR/relative-volume and each later feature family separately with matched ablations. Compute indicators locally where possible. Never label bar-derived volume profiles or estimated signed trades as exact volume-at-price/order flow.
- Use Yahoo/`yfinance` as the initial free sector/current-chain and longer-daily-history research candidate, with personal-use and reliability limitations; SEC filings provide financials and a SIC fallback. Snapshot-only metadata/options cannot be inserted into historical backtests. Alpha Vantage, Twelve Data, Tradier, GDELT, and file imports are optional adapters audited per endpoint, entitlement, date range, and terms before adoption.
- Do not promise free long historical options/OI or order-book archives. Accumulate verified snapshots forward or keep these features disabled pending adequate data. OI call/put walls are concentration proxies, not known dealer positions or guaranteed price barriers.

## Training and artifacts

- PPO is on-policy: configure rollout length, environment count, batch size, learning rate, and epochs deliberately; check rollout/batch compatibility. Benchmark CPU for small MLPs before assuming a GPU helps.
- SAC/TD3 are off-policy: configure warmup, replay capacity, collection frequency, gradient steps/update ratio, batch size, and target updates. Keep SAC entropy settings and TD3 exploration noise, policy delay, and target smoothing algorithm-specific.
- Record resolved config, Git revision/dirty state, package versions, seeds, device, data identity, and duration. Seed Python, NumPy, PyTorch, spaces, and environments; document hardware/backend nondeterminism.
- Use unique run directories. Save model, preprocessing/normalization state, metadata, and evaluation results together. Save SAC/TD3 replay buffers when needed for resume. Distinguish continued learning from exact restoration of RNG/environment state.
- Checkpoint periodically to durable storage, fail clearly on nonfinite data/rewards/losses, and keep large datasets and generated artifacts out of Git.
- Use the documented baseline configurations in `plan.md` as hypotheses, not proven trading optima. Tune only on training/validation folds with recorded search budgets. More passes over the same history do not add independent market evidence.
- Begin reward design with net equity change after modeled costs. Add turnover/downside terms one at a time. Report win/loss ratio, profit factor, and expectancy as diagnostics initially; avoid optimizing trade win rate alone or counting P&L twice. Holding age belongs in state; swing preferences must never block protective exits.
- Evaluate validation at configured checkpoints and create saved equity/drawdown charts, checkpoint/seed comparisons, trades with stop/target-zone overlays, holding-time distributions, and cost/exposure reports. Declare the selection metric; distinguish validation selection from final-test audit and include unsuccessful runs.
- Freeze deployment bundles with weights, preprocessing, feature/action schemas, configs, provider assumptions, dependency/data manifests, and checksums. Inference must not train, update normalization, or modify replay buffers. Check save/load parity and reject incompatible feature/feed schemas.

## Google Colab CLI

- Always match the current default Python version provided by Colab. Before dependency changes or training, inspect the actual Colab kernel version and align local environments, package Python constraints, dependency-lock targets, runtime checks, scripts, tests and documentation with it. Verify compatible dependencies and run acceptance on that version. Do not install an older Python interpreter in Colab or add compatibility workarounds merely to preserve an outdated project pin. A standard isolated dependency environment must use Colab's own interpreter. If Colab changes its Python version, update and verify the project before training.
- Reuse the existing official Colab CLI installation. Do not install a second copy or reinstall/upgrade it unless explicitly requested or an identified compatibility problem requires it; explain that problem before changing the installation.
- Use Google's official [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli), whose executable is `colab`; similarly named tools differ. Check `colab version` and installed-release command help before automation.
- Once scripts and dependency files exist, use explicit named sessions and documented `colab new`, `colab install`, `colab exec`, `colab upload`/`download`, and `colab stop` workflows. Ensure package, configuration, and data dependencies are available remotely; executing a local script transmits its content but does not sync the whole repository.
- Use lower-credit runtimes for setup, debugging and validation before allocating higher-credit hardware. Start with local CPU or a Colab CPU runtime for dependency checks, environment checkers and tiny train/evaluate/save/load smoke tests; use T4 or another suitable lower-credit GPU when GPU-specific smoke checks are needed. Check actual reported consumption rates rather than assuming a fixed price by GPU name. Running CPU code inside an allocated A100 does not count as a lower-credit preflight.
- Prefer Colab CPU for the current small-network research workload, per the user's 2026-10-15 direction, which supersedes the earlier A100 requirement. Use the cheapest suitable GPU only when profiling justifies it, after lower-credit preflight. Do not allocate A100/H100 merely for installation, debugging or smoke validation. Prepare code, data and dependency locks beforehand, retain the cumulative 150-unit project ledger across runtime changes, and include setup, idle time, downloads, failures and retries in the budget. Colab VMs are ephemeral: checkpoint durably, retrieve artifacts/logs before teardown, and stop only sessions created for the task. Respect usage limits.
- Use Colab for training jobs. Plan live-data inference and paper orchestration on a persistent local/server process with durable recovery state; do not assume a Colab VM provides unattended trading uptime.

## Alpaca and credentials

- Separate historical-data and trading clients. Explicitly set entitled feeds and adjustments; handle pagination, timeouts, rate limits, and bounded retries. Cache reproducible historical snapshots.
- Offline simulation is the default. Any broker adapter must explicitly select paper trading (for example `TradingClient(..., paper=True)`) and verify the paper endpoint/account. Live trading requires an explicit user request; training or backtesting does not authorize orders.
- Read credentials from environment variables or runtime secret storage. Never commit or print API keys, OAuth tokens, `.env` contents, authenticated outputs, or secrets in notebooks, CLI arguments, logs, or model metadata.
- Reconcile orders, partial fills, cancellations, positions, and rejections. Use stable client order IDs and reconcile ambiguous submissions before retrying to prevent duplicates. Respect account/asset restrictions. Paper results do not prove live performance.
- Move frozen models through live-data shadow inference before paper orders. Reuse causal feature/decision code, enforce staleness/schema checks, log bundle IDs, and test restart/reconnect recovery. Validate actual short/borrow eligibility and feed differences before enabling short paper execution.
- Alpaca brackets have one take-profit leg; verify a separate order lifecycle for zone scale-outs and position adjustments. Maintain protection for remaining quantities through partial fills/cancellation/replacement. Enforce explicit limits independently of reward shaping.

## Validation

- Use synthetic fixtures and mocked Alpaca clients so routine checks need no network, credentials, runtime provisioning, or orders.
- Test accounting, action constraints, timing/leakage, episode boundaries, seeded reset, normalization freeze, and checkpoint restoration. Before long jobs, run environment checkers and tiny CPU train/evaluate/save/load smoke tests for each algorithm. Smoke tests validate wiring, not profitability.
- Follow existing lint/type/test tooling. Run relevant checks after implementation and report failures or blockers accurately. Update runnable documentation when interfaces change.

## Primary documentation

Read 2026-10-13; recheck docs matching installed releases when APIs change. SB3 `master` may describe unreleased APIs.

- [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)
- [Colab FAQ/runtime lifecycle](https://research.google.com/colaboratory/faq.html)
- [Gymnasium Env API](https://gymnasium.farama.org/api/env/)
- [Gymnasium custom environment tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
- [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html), [TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
- [Alpaca Python SDK](https://alpaca.markets/sdks/python/)
- [Alpaca TradingClient](https://alpaca.markets/sdks/python/api_reference/trading/trading-client.html)
- [Alpaca market-data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq)
- [Alpaca paper trading](https://docs.alpaca.markets/us/docs/paper-trading)
- [Alpaca order lifecycle](https://docs.alpaca.markets/us/docs/orders-at-alpaca)
- [Alpaca margin and short selling](https://docs.alpaca.markets/us/docs/margin-and-short-selling)
- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [yfinance research adapter and terms](https://github.com/ranaroussi/yfinance)
- [Alpha Vantage endpoint documentation](https://www.alphavantage.co/documentation/)
- [Twelve Data free-tier plans](https://twelvedata.com/pricing)
- [Tradier market-data access](https://docs.tradier.com/docs/market-data)
- [GDELT archives](https://gdeltproject.org/data.html)
