# rl-stocks

A modular research toolkit for daily swing-trading simulations with Stable-Baselines3 PPO, SAC and TD3. It separates market-data access, causal features, portfolio accounting, training, evaluation, reporting and frozen inference.

The default research universe is **SPY, QQQ, IWM and DIA**, with one symbol traded per episode. An optional [configurable sector universe](docs/sectors-universe.md) adds 22 stocks across 11 sectors and sector-ETF rotation context. The project is a research prototype: software checks and synthetic results do not establish profitable trading performance. Broker-order execution is not implemented.

## Current capabilities and validation

| Area | Implementation | Validation status |
| --- | --- | --- |
| Synthetic/CSV data, immutable storage and resumable accumulation | Implemented | Offline tests pass |
| Causal daily features, optional completed weekly context, chronological splits and frozen normalization | Implemented | Offline tests pass |
| Long/short accounting, next-open fills, costs, stops, learned target zones and reversals | Implemented | Offline simulator tests pass |
| Reports and matched run comparisons | Implemented | SVG/JSON and Plotly integration tested |
| Yahoo/Alpaca historical adapters | Implemented | Yahoo daily coverage verified; authenticated Alpaca SIP access passes, with historical/action gaps explicitly reported |
| Gymnasium environment, PPO/SAC/TD3 training, checkpoints and frozen neural inference | Implemented | CPU acceptance passes for all three algorithms |
| Budget-controlled Colab launch, metering, retrieval and cleanup | Implemented | CPU/A100 acceptance and matched research studies complete; artifacts retrieved and owned shutdown verified |
| Multi-timeframe supply/demand inputs and sector rotation | Implemented, opt-in | Causality/metadata tests, CPU smoke, 36-run comparison and frozen held-out audit complete |
| Live-data shadow orchestration and Alpaca paper orders | Shadow implemented; paper orders remain a future milestone | Frozen shadow activation is paused; no orders submitted |

The early baseline verification recorded **207 passing tests with no skips**; later features have their own focused acceptance evidence. Direct and transitive dependencies are pinned in a hash lock. See [verification status](docs/verification.md) for the evidence, limitations and generated artifacts, and [the documentation index](docs/README.md) for detailed category guides.

The [matched training-universe comparison](docs/universe-expansion.md) tests 22
versus 55 stocks, plus four ETFs, on the same validation panel. Its six Colab CPU
PPO runs did not show an improvement from expansion: mean selected/final returns
were 10.53%/5.60% versus control 11.77%/10.49% and buy-and-hold 30.60%. The sample
remains retrospective; this is development validation, not fresh test evidence.

Current experiment outcomes and pending work are indexed in [research results](results/README.md).

## Controlled learning experiments

[Controlled CPU learning experiments](docs/learning-experiments.md) completed 15 SAC
runs on CPU, comparing inputs, reward scales and budgets up to 250k transitions.
Longer training raised mean selected validation return from 19.70% to 23.96%,
but final checkpoints averaged 14.46% and drawdown failed the declared risk gate.
Buy-and-hold returned 30.60%. No configuration is paper-eligible.

Compact CSV/JSONL records and interactive graphs track validation returns every
10k steps, including incremental percentage-point changes, drawdown and costs.
See the [learning results](results/learning-study-2026-10-15/README.md).
The original held-out audit remains unchanged; future tuning requires a fresh
locked interval for new out-of-sample claims.

## Matched indicator follow-up

The [250k indicator follow-up](docs/indicator-experiments.md) compares combined,
supply/demand-only and sector-only inputs against the completed 250k baseline.
The original reward and costs stay fixed. See [current comparison evidence](results/indicator-followup-2026-10-16/README.md).
A training-only decline audit is available; [historical universe coverage](docs/historical-universe.md)
remains a separate data-verification requirement before expanding the sample.

## Algorithm and memory comparison

The [completed 100k comparison](results/algorithm-stability-2026-10-16/README.md)
trained SAC, PPO and RecurrentPPO on Colab CPU with three seeds each. Mean final
validation returns were 16.50%, 16.90% and 6.13%, respectively, versus 30.60%
buy-and-hold. All improvement gates failed; recurrent memory did not help this
configuration. [Download interactive curves and frozen models](https://github.com/jason490/RL-trading/releases/tag/algorithm-stability-2026-10-16).
These are separate stock episodes, not portfolio or new held-out results.

## Published research results

The [completed study](results/sector-study-2026-10-14/README.md) compares four input
sets, three algorithms and seeds 11/23/37 at 50,000 transitions each. Validation
selected combined-feature SAC; its three-seed mean return was 19.79%, versus
30.60% for buy-and-hold, with substantial seed variation.

The separately frozen 2024–September 2026 audit returned **10.83%** on average
across seeds versus **48.46%** for buy-and-hold. The preselected seed returned
13.90%. These are total returns from separate symbol episodes, not a portfolio
or annualized results. The models have not demonstrated a benchmark advantage. Subsequent controlled studies and negative results are recorded in the [research log](docs/research-loop.md), including [historical ETF augmentation](results/historical-augmentation-2026-10-20/README.md) and [SAC execution-state comparison](results/sac-execution-state-2026-10-20/README.md).
The first 29 runs used CUDA and the last seven CPU; hardware differences limit
a strict feature-only interpretation. The universe is a retrospective surviving-stock sample.

[Download models and visual reports](https://github.com/jason490/RL-trading/releases/tag/sector-study-2026-10-14).
Extract `validation-reports.tar.gz` and open its index; open `test.html` for the
held-out charts. All ten uploaded asset digests were verified. See
[artifact retention](docs/artifact-retention.md) for local evidence and cleanup.

## Forward shadow tracking

[Persistent shadow inference](docs/shadow-inference.md) freezes saved policies
and records future daily decisions and simulated fills in a durable journal.
The supplied campaign includes all three combined-input SAC seeds plus basic
PPO/SAC controls, with buy-and-hold/cash comparisons and interactive reports.
It runs on a local/server process, makes no broker orders and does not train.
Registration and software acceptance are separate from future market results.

## Historical-universe evidence

[Historical data import and audit](docs/historical-data.md) tracks stable security
IDs, dated membership/ticker changes, missing member prices and delisting outcomes.
It includes immutable file imports, coverage reports and optional read-only
Alpha Vantage listing discovery. Real historical coverage is still unverified;
the bundled example is synthetic, and episode/terminal integration remains a
separate gate before training an expanded universe.

## Requirements and installation

Use **Python 3.13**. Installation and real-provider retrieval require network access. The baseline does not need Alpaca credentials; Yahoo and Alpaca SDKs are optional package extras. Run commands from the repository root.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install uv

# Install the committed lock; regenerate it only for reviewed dependency changes.
export UV_CACHE_DIR="$PWD/.runtime/uv-cache"
mkdir -p .runtime
.venv/bin/uv pip sync --python .venv/bin/python requirements.lock
.venv/bin/uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m pip check

PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Review and retain `requirements.lock`. `requirements.txt` contains direct pins and is not a complete transitive lock. Training checks actual core package versions against the reviewed release set. Full setup instructions and compatibility assumptions are in [setup.md](docs/setup.md).

## Offline quick start

The standard-library audit needs Python 3.13 but no API credentials, RL packages or remote runtime:

```bash
PYTHONPATH=src .venv/bin/python scripts/offline_audit.py
```

It prints paths to a synthetic snapshot, a reproducible dataset and a standalone HTML report. The report uses a fixed exposure fixture, not a trained policy. Synthetic calendars are explicitly unverified; their results are only accounting and wiring checks.

## Historical data and datasets

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local ingest --provider yahoo
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local dataset data/raw/SNAPSHOT_HASH
```

Replace `SNAPSHOT_HASH` with the directory printed by ingestion. The default request covers 2010-01-01 through September 2026 (`2026-10-01` exclusive). Provider adapters can also be called with explicit requests through the Python API; the dataset generator supports both the default four ETFs and `--universe configs/universes/sector_research.json`.

Available providers are `synthetic`, `files`, `yahoo` and `alpaca`. Files use `--file bars.csv`; Alpaca requires an explicit IEX/SIP feed and environment credentials. Alpaca bar-only data lacks verified dividend cash flows and cannot pass the default dataset gate without further action-data auditing. Sources are never silently substituted or mixed.

Features include log returns, trailing volatility, EMA 20/50, RSI 14, ATR 14 and prior-window relative volume. `dataset --weekly` adds completed weekly context. Training ends in 2020, validation covers 2021–2023, and testing starts in 2024. Training/validation ends have a 20-session purge; preprocessing fits only retained training rows.

See [providers](docs/providers.md), [storage](docs/storage.md), [features](docs/features.md) and [datasets](docs/datasets.md) for source limitations, availability conventions, coverage gates and schemas.

## Training and evaluation

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local train data/datasets/DATASET_HASH --algorithm ppo
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local train data/datasets/DATASET_HASH --algorithm sac
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local train data/datasets/DATASET_HASH --algorithm td3
```

Default smoke budgets are exactly **512 PPO / 1,024 SAC / 1,024 TD3 transitions**. Training runs Gymnasium/SB3 checkers, records seeds/configuration/data/package identities, evaluates validation checkpoints and exports a checksummed research bundle. Smoke acceptance verifies software wiring, not profitability.

`--long --seed 11` selects a 50,000-transition experiment; the planned seed set is 11/23/37. Complete smoke acceptance before longer jobs. `--final-test` audits selected weights only after validation selection and freezing; it never selects checkpoints. Keep the final test untouched while changing features, rewards or hyperparameters.

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local compare artifacts/runs --output artifacts/comparison.html
```

Reports include equity/baselines, drawdown, exposure, protective zones, fills, costs and holding-time diagnostics. Run comparisons retain failures and separate unmatched configurations. Frozen inference rejects incompatible schemas/feeds and invalid or stale inputs. Detailed behavior is documented in [training](docs/training.md), [evaluation](docs/evaluation.md), [reporting](docs/reporting.md) and [inference](docs/inference.md).

## Optional Colab training

Check the actual Colab kernel Python before dependency changes or training. Match its default interpreter and verify compatible dependencies rather than installing an older Python. The inspected images use Python 3.13.

Colab is an optional training backend. Use Google's official [google-colab-cli](https://github.com/googlecolab/google-colab-cli), authenticate interactively before automation, and verify that `usage` returns balance/rate information. The runtime code was inspected against CLI 0.7.4; check installed-release help before changing versions.

```bash
# CLI_PYTHON is the interpreter containing the official google-colab-cli package.
/path/to/CLI_PYTHON scripts/colab_safe.py version
/path/to/CLI_PYTHON scripts/colab_safe.py usage

# Run only after lock generation, local integration and data audits pass.
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.runner data/datasets/DATASET_HASH --cli-python /path/to/CLI_PYTHON
```

The runner defaults to CPU. Without a research flag it runs smoke acceptance; `--research ppo` or `--study --study-algorithm ppo` requests full CPU research. Explicitly select `--accelerator T4` or `--accelerator A100` when profiling justifies GPU use. It owns one uniquely named session at a time, persists a cumulative usage ledger, monitors usage every minute, retrieves artifacts and verifies cleanup. Project defaults are a **150-unit ceiling**, teardown beginning by **130 units**, and an initial allocation deadline of at most **two hours**, including setup and retrieval. Unreliable metering prevents or stops allocation; budget stops do not trigger automatic restart.

The wrapper redirects logging/settings/history/session state to private `.runtime/` storage without changing authentication. See [Colab operations and troubleshooting](docs/colab.md) for reusing an existing installation, authentication, network diagnosis, budget accounting and recovery. Recorded remote acceptance includes all three algorithm smokes, 36 multi-seed research runs and a frozen held-out audit; artifacts and budget evidence are listed in [verification](docs/verification.md).

## Recommended development sequence

1. Review and install the dependency lock, then pass the integration suite without missing-dependency skips.
2. Retrieve and audit actual provider history, adjustments, calendars and availability; build a versioned dataset.
3. Run short CPU train/evaluate/save/load checks for all three algorithms.
4. Verify Colab authentication and metering, then run bounded CPU/T4 preflight before choosing a full-training runtime from measured throughput and compute cost.
5. Compare matched 50,000-transition runs across seeds 11/23/37 using validation only.
6. Freeze a selected candidate and perform a held-out test audit.
7. Add persistent shadow inference before designing paper-order execution.

Advanced data families, additional reward terms and broker-order lifecycle work remain separate milestones. The [supply/demand guide](docs/supply-demand.md) describes implemented causal daily/weekly inputs; [sector rotation](docs/sectors-universe.md) documents the expanded universe and classification gates. Keep generated data, artifacts, runtime state and credentials out of Git. [plan.md](plan.md) contains the roadmap; [docs/README.md](docs/README.md) indexes the detailed guides.

The next research priorities are representative historical stock coverage,
profiling/batching simulation and evaluation, and separately declared experiments
to reduce unnecessary turnover and costs. Keep this completed test audit separate from future tuning;
new claims need a fresh locked audit interval. Historical universe membership,
delistings and classification vintages are prerequisites for broader stock
scaling. Persistent shadow inference and paper execution remain later milestones.
