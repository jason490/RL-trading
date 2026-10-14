# Colab execution, budget and recovery

Remote execution is explicitly authorized for the current training task. Actual run results and cleanup status belong in [verification.md](verification.md); this guide describes the launch and recovery procedure.

Use Google's official `google-colab-cli` executable `colab`. The installed release inspected here is 0.7.4. Its usage command prints exact labelled account balance, hourly consumption and active assignment count. Do not assume a fixed conversion between 150 compute units and wall hours. [Official CLI](https://github.com/googlecolab/google-colab-cli)

## Runtime selection: inexpensive preflight, A100 training

Use lower-credit hardware for preparation and validation before allocating the
full-training runtime. The user has selected **A100 for full training**.

| Phase | Runtime | Required evidence before proceeding |
| --- | --- | --- |
| Prepare code, data and locks; check dependencies, accounting and environment APIs | Local CPU or Colab CPU | Verified dataset, compatible dependencies and passing checkers |
| Tiny train/evaluate/save/load tests | CPU first | PPO/SAC/TD3 smoke acceptance, frozen prediction parity and recorded duration |
| GPU-specific checks, when needed | T4 or another suitable lower-credit GPU | CUDA wiring, artifact retrieval and cleanup verified |
| Full research training | A100 | Lower-credit preflight passed; data/configuration ready; budget and deadline checked |

Do not allocate A100/H100 just to install dependencies, debug scripts or run
smoke tests. A CPU profile performed inside an allocated A100 still incurs the
A100 runtime's charges and does not replace CPU-first preflight. A100 may run
brief final sanity checks immediately before its full training job after the
lower-credit checks have already passed.

Read the account balance before allocation and the reported consumption rate
afterward. Hardware names do not establish a fixed credit rate. Record the
runtime, rate, elapsed time and measured/estimated units for each attempt. Keep
one owned session at a time; retrieve artifacts and verify shutdown before
changing runtimes. The existing 150-unit ceiling, 130-unit teardown threshold,
cumulative ledger and initial two-hour allocation limit apply across all
stages, including setup, allocated idle time, downloads, failures and retries.

The runner defaults to **CPU preflight**, supports **T4 GPU preflight**, and
requires `--research` for **A100 full training**. All modes share the same
cumulative ledger, deadline, downloads and cleanup verification:

```bash
# Remote CPU preflight: PPO/SAC/TD3 tiny train/save/load checks.
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.runner data/datasets/DATASET_HASH --cli-python /path/to/CLI_PYTHON --accelerator CPU
# Optional lower-credit CUDA preflight.
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.runner data/datasets/DATASET_HASH --cli-python /path/to/CLI_PYTHON --accelerator T4
# Full A100 training only after preflight passes.
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.runner data/datasets/DATASET_HASH --cli-python /path/to/CLI_PYTHON --research ppo
```

A reported zero consumption rate is valid for a verified CPU runtime, while
metering still runs every minute. GPU runtimes require a positive rate. Missing
or unparseable metering always stops work. Logs are private and separated by
session under `.runtime/SESSION_NAME/`; retries cannot reuse stale operation logs.

## Writable wrapper and authentication

`scripts/colab_safe.py` imports the installed official CLI. It directs logs, settings, history and session state to private project `.runtime/` paths because the default home config files are read-only in this sandbox. The wrapper keeps only warning-level logs to avoid HTTP DEBUG details. It does not modify `HOME`, change OAuth scopes, copy/print credentials or alter authentication. The official CLI remains responsible for existing credentials and refresh behavior.

Use the Python executable in the environment containing that official package:

```bash
/path/to/official-colab/python scripts/colab_safe.py version
/path/to/official-colab/python scripts/colab_safe.py usage
/path/to/official-colab/python scripts/colab_safe.py new --help
```

Inspect command help matching the installed release before changing automation. The current wrapper's entry point and metering parser are tied to the inspected version; a different output format fails closed.

## Preconditions and launch

### Python version preflight

Before launching paid work, check the local research interpreter:

```bash
.venv/bin/python --version
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
```

There are three interpreter roles: the local official CLI, Colab's notebook
kernel, and the research dependency environment. The CLI can use another Python
version; research must match Colab's kernel. The current A100 uses **Python 3.13.15**; another inspected image reports
3.13.16. Record the actual patch version rather than assuming it. The standard virtual environment isolates dependency versions while
using the kernel's own interpreter.

For manual diagnosis on an already owned session, inspect both remote versions
without allocating another session:

```bash
/path/to/CLI_PYTHON scripts/colab_safe.py exec -s OWNED_SESSION_NAME <<'PY'
import pathlib, subprocess, sys
print('Colab kernel Python:', sys.version.split()[0])
research = pathlib.Path('/content/rl-stocks/.venv/bin/python')
if research.exists():
    subprocess.run([str(research), '-c',
                    'import sys; print("Research Python:", sys.version.split()[0]); assert sys.version_info[:2] == (3, 13)'],
                   check=True)
else:
    print('Research environment is not created yet; run the isolated bootstrap.')
PY
```

The automatic runner uses Colab's own interpreter for a standard isolated
dependency environment before research package installation. Its worker checks the version before importing PyTorch/training;
the downloaded completion manifest must also report Python 3.13. If Colab changes its default Python minor version, align and verify the
project before training. Older-interpreter workarounds are prohibited. All diagnosis performed while allocated remains subject to metering
and the two-hour deadline.

### Launch prerequisites

1. Resolve/review `requirements.lock`, install it locally and pass dependency-backed acceptance.
2. Inspect the data manifest/coverage/actions and CPU smoke profile.
3. Ensure there are no other account assignments; isolated account balance metering otherwise cannot safely attribute this project.
4. Verify `usage` returns finite nonnegative values and more than 20 units of available balance.
5. Confirm the persistent project ledger has no unresolved session or prior budget stop.

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.runner data/datasets/DATASET_HASH --cli-python /path/to/official-colab/python --research ppo
```

The runner checks installed version and help for usage/new/install/exec/upload/download/stop/sessions before allocation. A missing transitive lock prevents provisioning. It acquires an exclusive local allocation lock, saves a unique owned name and initial balance, creates one selected runtime, records its endpoint when discoverable, verifies its reported hardware and obtains its consumption rate immediately. An unavailable rate causes teardown; zero is permitted only for verified CPU runtimes.

The package, configs, dataset and lock are explicitly archived/uploaded. `colab install` installs the pinned uv bootstrap tool. `colab exec -f scripts/remote_smoke.py` creates a remote dependency environment using Colab's kernel interpreter, syncs the hash lock and launches `remote_train.py` as a subprocess. Both kernel and research process use the same Colab Python 3.13 interpreter, with research dependencies isolated from preinstalled notebook packages. Executing a local script alone does not synchronize imports or data. The worker verifies Python 3.13 and selected CPU/T4/A100 hardware. CPU preflight runs exactly PPO 512 and SAC/TD3 1024 CPU transitions. GPU plans include a CPU PPO profile and those exact CUDA smoke budgets before any full research job. Longer 50,000-transition jobs are opt-in research experiments.

The installed CLI can exit zero after a remote Python exception. The runner
therefore validates the downloaded completion manifest, expected job counts,
exact transition counts, dataset/runtime identities and frozen-bundle checksums.
A failure archive or missing job is rejected even if the CLI command succeeded.

Conversely, a CLI output-stream error can occur after the worker completed.
The runner records that transport failure, retrieves the final archive and
accepts it only if every expected job, runtime identity and bundle checksum
passes validation. Budget stops and interruptions still enter teardown.

The worker environment selects Matplotlib's `Agg` backend for headless training,
sets a writable plotting cache and removes inherited notebook Python path/home
overrides. This prevents Colab's inline backend from requiring notebook-only
packages inside the research environment. Python remains Colab's own version;
interactive reports use Plotly. Bootstrap stdout/stderr and worker tracebacks
are saved in the retrieved archive for diagnosis.

To follow successful smoke acceptance with one initial research model in the same
owned session, add `--research ppo` (or `sac` / `td3`). This appends exactly 50,000
transitions with seed 11; PPO uses rollout 1000 / batch 50. The uploaded, recorded
job plan is validated remotely. A failed smoke job prevents research training.
Research does not extend the two-hour allocation deadline or compute budget.
Seeds 23/37 and final-test auditing remain separate experiments.

## Compute ledger and deadline

`.runtime/usage-ledger.json` persists all attempts. The total ceiling is **150 units**; working operations stop at **130**, leaving 20 for metering lag and shutdown. Setup, idle allocation, installation, failed runs, retries and downloads count. The charge estimate uses the larger of measured balance loss and conservative rate integration; rate integration uses the larger adjacent rate plus a small rounding allowance. Account credits cannot erase already recorded usage.

The watchdog requests usage every minute while long commands run, estimates between readings and projects the next minute/bounded operation before proceeding. Another account assignment, unparseable/missing meter, insufficient balance or projected working-limit violation enters teardown. It never automatically restarts after a budget stop.

The initial allocation deadline is at most two hours, including provisioning, installation and artifact retrieval. Rate/balance guards can stop it sooner. Individual operations have bounded waits; the watchdog can terminate their local process group while still proceeding to remote `stop`. Installing or downloading is not exempt from the budget.

## Durable retrieval and teardown

Validation checkpoints save model/report/history and an atomic remote checkpoint archive. The local watchdog tries to retrieve that archive each minute within a bounded wait. The final remote archive is built in `finally`, including partial/failure artifacts. Final retrieval has at most three 30-second attempts and runs only while the working budget permits it.

On success, failure, timeout or handled SIGINT/SIGTERM, `finally` stops the owned session with bounded attempts and queries the backend to verify absence. It never stops unrelated sessions. An ambiguous allocation failure with no known endpoint and a remaining unknown assignment is **not** declared safe. Failed cleanup remains in the ledger and blocks subsequent allocation.

Output summarizes session identity, measured attempt debit, conservative cumulative charge, remaining 150-unit project budget, retrieved paths and verified cleanup. Local runtime state is private and ignored by Git because official session state can contain tokens; only non-secret session/endpoint identifiers enter the usage ledger.

## Recovery

If cleanup fails, the runner prints the exact recovery command and persists the owned session name/endpoint. Use that identifier, not an implicit default:

```bash
/path/to/official-colab/python scripts/colab_safe.py stop -s OWNED_SESSION_NAME
/path/to/official-colab/python scripts/colab_safe.py sessions
```

Review the ledger after verified manual recovery before allowing another allocation. Do not delete/reset it to bypass a project budget stop. If a session remains orphaned after an ambiguous response, inspect backend sessions and recover the owned allocation explicitly; the software will not guess and terminate another user's session.

SIGKILL, local power loss or a total network outage cannot guarantee remote cleanup. Durable local checkpoints and persisted identifiers are the recovery mechanisms. Before the first successful checkpoint transfer, ephemeral VM data can still be lost. A Colab runtime is suitable for bounded training jobs, not unattended persistent trading uptime.

Authenticated backend balance/rate readings now work. Mock tests verify budget arithmetic and failure/cleanup branches; actual Colab lifecycle evidence and charges are recorded in [verification.md](verification.md).

## Troubleshooting and first-time setup

These steps are intended for a normal terminal with working DNS/HTTPS and a writable user credential directory. Running them inside a restricted execution sandbox may reproduce its network failure even when the host terminal works. Test from the same environment that will run the budget controller. The commands below read usage and authenticate; they do not allocate a VM.

### 1. Reuse the installed official CLI

Colab is already installed for this project; no separate installation is required. Use the Python interpreter from the environment containing that official package, keeping its dependencies separate from the project's RL environment:

```bash
/path/to/CLI_PYTHON scripts/colab_safe.py version
/path/to/CLI_PYTHON scripts/colab_safe.py usage --help
```

Replace `/path/to/CLI_PYTHON` with the existing installation's interpreter. For a uv tool installation, inspect `uv tool dir` and the `google-colab-cli/bin/python` path beneath that directory. The runtime was inspected against CLI 0.7.4; verify the installed release before automation. A similarly named package is not a substitute. Do not reinstall or upgrade a working installation merely to follow this guide.

### 2. Authenticate interactively and verify metering

```bash
/path/to/CLI_PYTHON scripts/colab_safe.py --auth oauth2 usage
```

If authentication is needed, follow the official CLI's authorization URL/code prompts locally. Never paste tokens into project configuration, command arguments, issue reports or chat. The OAuth flow needs working Google connectivity and writable credential storage; the project wrapper redirects runtime files but does not redirect the official authentication cache.

Successful metering should print these labels with actual numeric values:

```text
Current balance: ... compute units
Usage rate: .../hr
Active assignments: ...
```

A zero usage rate can be normal **before** allocation; this runner requires a positive rate after allocating a GPU; a verified CPU runtime may report zero. Its preflight also requires zero existing assignments and more than 20 units of account balance. The 150-unit project ceiling is separate from the account's available balance. [Official usage/authentication documentation](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md)

The automated runner uses noninteractive stdin, so complete authorization beforehand. An interactive sign-in that has never been completed cannot be repaired by blindly retrying automated provisioning.

### 3. Resolve failures by category

| Symptom | Meaning | Resolution |
| --- | --- | --- |
| Snap confinement error from `uv` | The Snap executable cannot start in that execution environment | Install uv in `.venv` with pip and use `RL_UV_BIN=.venv/bin/uv` for locking |
| `Read-only file system` for Colab logs/settings/history | Default CLI storage is outside the writable area | Use `scripts/colab_safe.py`; it redirects those operational files to `.runtime/` |
| Read/write failure for the authentication cache | Authentication storage is still unavailable; it is not covered by the runtime wrapper | Authenticate from a normal terminal with writable user credential storage and run the controller there |
| No `usage` command | The installed CLI release lacks required metering support | Install the inspected official release and check version/help |
| `Name or service not known`, DNS or connection failure | The process cannot reach its network destination | Fix DNS/proxy/firewall/network policy for that environment, or run outside the restricted sandbox |
| Package installation cannot find any versions during a connection failure | Package-index access failed; this does not establish an incompatible package pin | Restore package-index HTTPS access, then rerun lock generation/installation |
| Authorization prompt, expired credentials or access denied | Account sign-in/permissions need attention | Complete official interactive authentication and rerun `usage`; do not assume a network error is an auth error |
| Missing `requirements.lock` | Reproducible dependency resolution was not completed | Follow [setup.md](setup.md), review the generated hash lock and pass local integration |
| A100 allocation denied/unavailable | Account entitlement or current accelerator availability needs checking | Check the Colab account and installed CLI error; avoid unmetered repeated retries |

For network diagnosis, test package-index connectivity without displaying credentials:

```bash
.venv/bin/python -c "import urllib.request; print(urllib.request.urlopen('https://pypi.org/simple/', timeout=10).status)"
```

A package-index success does not prove Google access; `usage` is the separate authenticated Colab check. Do not enable HTTP DEBUG logs or publish raw credential files to diagnose connectivity.

### 4. Resolve dependencies and run local acceptance

Use [setup.md](setup.md) to generate `requirements.lock`, install it and run `pip check`. Run the integration suite and CPU smoke on a declared dataset before spending accelerator units. Distinguish a passing suite from one that merely skips all RL tests because packages are missing.

### 5. Launch through the budget controller when ready

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.runner data/datasets/DATASET_HASH --cli-python /path/to/CLI_PYTHON --research ppo
```

Replace the dataset placeholder with the printed content-addressed path. Preserve the ledger across attempts. This command allocates an A100 for full research training; run it only after the lower-credit preflight passes. Actual A100 eligibility and lifecycle success must be verified on each allocation.
