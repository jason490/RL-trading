# Colab execution, budget and recovery

Remote execution is explicitly authorized for the current training task. Actual run results and cleanup status belong in [verification.md](verification.md); this guide describes the launch and recovery procedure.

Use Google's official `google-colab-cli` executable `colab`. The installed release inspected here is 0.7.4. Its usage command prints exact labelled account balance, hourly consumption and active assignment count. Do not assume a fixed conversion between 150 compute units and wall hours. [Official CLI](https://github.com/googlecolab/google-colab-cli)

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

1. Resolve/review `requirements.lock`, install it locally and pass dependency-backed acceptance.
2. Inspect the data manifest/coverage/actions and CPU smoke profile.
3. Ensure there are no other account assignments; isolated account balance metering otherwise cannot safely attribute this project.
4. Verify `usage` returns finite nonnegative values and more than 20 units of available balance.
5. Confirm the persistent project ledger has no unresolved session or prior budget stop.

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.runner data/datasets/DATASET_HASH --cli-python /path/to/official-colab/python
```

The runner checks installed version and help for usage/new/install/exec/upload/download/stop/sessions before allocation. A missing transitive lock prevents provisioning. It acquires an exclusive local allocation lock, saves a unique owned name and initial balance, creates one A100, records its endpoint when discoverable, verifies reported A100 hardware and obtains its consumption rate immediately. A zero/unavailable rate causes teardown.

The package, configs, dataset and lock are explicitly archived/uploaded. Remote `colab install -r` installs the pinned environment; `colab exec -f scripts/remote_smoke.py` bootstraps that uploaded payload. Executing a local script alone does not synchronize its imports or data. The remote script requires Python 3.12 and actual A100 CUDA hardware, runs a CPU PPO profile, then exactly 512 PPO and 1024 SAC/TD3 GPU transitions. Longer 50,000-transition configs remain separate, unexecuted experiments.

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

The actual backend balance/rate could not be read in this sandbox. Mock tests verify budget arithmetic and failure/cleanup branches; they do not prove actual Colab lifecycle behavior. That validation is reserved for the later user-run smoke.

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

A zero usage rate can be normal **before** allocation; this runner requires a positive rate after allocating a paid runtime. Its preflight also requires zero existing assignments and more than 20 units of account balance. The 150-unit project ceiling is separate from the account's available balance. [Official usage/authentication documentation](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md)

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
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.runner data/datasets/DATASET_HASH --cli-python /path/to/CLI_PYTHON
```

Replace the dataset placeholder with the printed content-addressed path. Preserve the ledger across attempts. This command allocates a runtime; unlike the preceding help/usage steps, run it only when remote smoke execution is intended. Actual A100 eligibility and lifecycle success still require that later acceptance run.
