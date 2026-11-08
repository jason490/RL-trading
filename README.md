# rl-stocks

A Python toolkit for researching daily swing-trading strategies with reinforcement learning. Start with historical data, build reproducible datasets, and compare simulated trading against matched buy-and-hold baselines.

**Research only:** no model has established robust, prospective benchmark-beating performance. Broker-order execution is not enabled. See [research results](results/README.md) for measured outcomes and limitations.

## Getting started

Run these commands from the repository root on Linux/macOS or WSL. You need Git, Python **3.13**, [uv](https://docs.astral.sh/uv/) and enough disk space for the pinned PyTorch dependencies. Before Colab training, check its actual default kernel version and match it; the latest verified kernel is Python 3.13.18. See [setup](docs/setup.md) for compatibility and dependency management.

### 1. Clone and install

```bash
git clone https://github.com/jason490/RL-trading.git
cd RL-trading
python3.13 --version
python3.13 -m venv .venv
mkdir -p .runtime
export UV_CACHE_DIR="$PWD/.runtime/uv-cache"
uv pip sync --python .venv/bin/python requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
```

Use the committed dependency lock; regenerating it is a separate, reviewed dependency change. The lock includes the optional Yahoo/Alpaca providers and development tools.

### 2. Download historical data

Start with **SPY, QQQ, IWM and DIA**. Yahoo retrieval needs internet access but no API key. These explicit dates download development history only; `--end` is exclusive.

```bash
.venv/bin/python -m rl_stocks.runtime.local ingest \
  --provider yahoo --start 2010-01-01 --end 2024-01-01 --full-range
```

The command prints a content-addressed snapshot directory under `data/raw/`. Downloads retain source provenance and resume verified cached chunks. Yahoo history can be revised or unavailable; acquisition alone does not certify coverage or realistic returns.

For a larger, separately declared universe, add:

```bash
--universe configs/universes/stock_research_132.json
```

This requests 132 stocks, four broad ETFs and eleven sector ETFs. Short listing histories and provider gaps need explicit auditing before dataset construction. MOVE/VIX are separate context sources, not tradable symbols in that universe. See [data providers](docs/providers.md), [stock universes](docs/sectors-universe.md), and [MOVE/VIX context](docs/macro-index-context.md). Alpaca credentials and its request limits are documented in the provider guide.

### 3. Build a development dataset

Replace `SNAPSHOT_HASH` with the directory printed by ingestion:

```bash
.venv/bin/python -m rl_stocks.runtime.local dataset \
  data/raw/SNAPSHOT_HASH --development-only
```

This prints a dataset directory under `data/datasets/`. Training ends in 2020; validation covers 2021–2023. Normalization fits training rows only, and split ends are purged. Quality checks reject missing coverage or unverified adjustments; investigate failures rather than bypassing them. Keep the retired 2024+ test history out of ongoing tuning. See [datasets](docs/datasets.md).

For a wider snapshot, pass the same `--universe` when building the dataset. Some symbols need a separately audited complete-case or historical-membership workflow; the quick start does not automatically admit every downloaded stock.

### 4. Train in Colab and review reports

Use the existing official Colab CLI installation, authenticate it, and check its balance and usage rate. Follow [Colab setup and preflight](docs/colab.md), then run the bounded CPU smoke checks with the dataset path from step 3:

```bash
# CLI_PYTHON is the Python executable containing your installed google-colab-cli.
/path/to/CLI_PYTHON scripts/colab_safe.py usage
.venv/bin/python -m rl_stocks.runtime.colab.runner \
  data/datasets/DATASET_HASH --cli-python /path/to/CLI_PYTHON --accelerator CPU
```

This allocates a metered remote session. The runner records usage, retrieves artifacts and verifies owned-session shutdown. Project limits are a cumulative 150 compute units, teardown by 130 units, and an initial allocation deadline of at most two hours. The shared ledger also protects a minimum account balance of 150 units, with teardown by 170. Do not reset the ledger or restart after a budget stop.

Use [training](docs/training.md) and [evaluation](docs/evaluation.md) to design longer matched experiments after smoke acceptance. Open downloaded HTML reports using the [reporting guide](docs/reporting.md); frozen model artifacts and checksums stay with each run. Smoke checks establish software wiring, not profitability.

## Documentation

- [Documentation index](docs/README.md): category guides and operating procedures.
- [Capability overview](docs/capabilities.md): implemented components and pending work.
- [Verification](docs/verification.md): checks actually performed and artifact evidence.
- [Research results](results/README.md): completed comparisons, including failed experiments.
- [Research workflow](docs/research-loop.md): bounded experiments, selection rules and prospective gates.
- [Project plan](plan.md): milestones and remaining work.

Keep downloaded data, generated artifacts, runtime state and credentials out of Git. See [artifact retention](docs/artifact-retention.md) before cleaning research outputs.
