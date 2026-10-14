# Installation and dependency management

Use Python **3.12**. The package declares `>=3.12,<3.13`; the local interpreter used for dependency-free verification is 3.12.14. Python 3.14 on the host is not the intended research runtime.

## Direct pins

| Package | Pin | Role |
| --- | --- | --- |
| NumPy | 2.2.6 | Numeric observations and SB3 input |
| PyTorch | 2.8.0 | Policy/critic networks on CPU or CUDA |
| Gymnasium | 1.2.0 | Environment API and checker |
| Stable-Baselines3 | 2.7.0 | Maintained PPO/SAC/TD3 |
| pandas | 2.3.2 | Optional frame/Parquet operations and SB3 dependency |
| PyArrow | 21.0.0 | Derived normalized Parquet |
| Plotly | 6.3.0 | Interactive reports |
| exchange-calendars | 4.11.1 | XNYS session/open/close coverage |
| yfinance, optional | 0.2.65 | Yahoo research adapter |
| alpaca-py, optional | 0.42.0 | Historical Alpaca adapter |
| pytest / Ruff, development | 8.4.2 / 0.13.0 | Optional development tools |

SB3 2.7.0 metadata declares Gymnasium >=0.29.1,<1.3; NumPy >=1.20,<3; and torch >=2.3,<3. These direct pins satisfy the declared bounds. This is **metadata verification**, not proof that all native libraries import successfully on a particular machine. [Pinned SB3 release metadata](https://pypi.org/pypi/stable-baselines3/2.7.0/json)

## Create the environment and hash lock

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install uv
RL_UV_BIN=.venv/bin/uv bash scripts/lock.sh
.venv/bin/uv pip sync --python .venv/bin/python requirements.lock
.venv/bin/uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m pip check
```

The lock script resolves runtime plus Yahoo/Alpaca/development extras and generates hashes for transitive packages. It uses `RL_UV_BIN` so a working project-local uv executable can replace a restricted Snap executable. Once generated, preserve/review `requirements.lock` in the repository and include it in runtime payloads and frozen bundles. Never label the direct-only `requirements.txt` as a fully resolved lock.

A minimal package installation can omit provider extras: `pip install -e .`; install `.[yahoo]` or `.[alpaca]` only when those adapters are needed. For reproducible research use the reviewed lock, not an unconstrained fresh resolution. The full lock includes extras for convenience without making SDK imports a baseline requirement.

The local sandbox has no installable RL packages and cannot reach package repositories, so transitive resolution, `pip check`, real Gymnasium/SB3 checkers and neural training remain pending. Colab allocation refuses a missing lock. No fake lock or substituted RL implementation is supplied.

## Dependency-free work

Data contracts, CSV imports, synthetic generation, indicators, JSON storage/datasets, portfolio accounting, budget logic and SVG reports work with Python's standard library:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/offline_audit.py
```

The integration suite explicitly skips when NumPy/Gymnasium/torch/SB3/Plotly are missing. Install the real dependencies and rerun it before claiming training acceptance. Package downloads need a network-enabled execution environment; Colab execution is currently deferred.

Training validates Python and actual installed core package versions against the reviewed direct pins before running checkers or learning. Local torch build suffixes (CPU/CUDA) are allowed; a different base release is rejected. This gate detects drift but does not replace the transitive lock or `pip check`.
