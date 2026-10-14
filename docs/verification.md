# Verification and milestone status — 2026-10-14

The project targets **Python 3.13**, matching Colab's current interpreter family.
The local acceptance environment is **3.13.16**. The current CLI-provisioned A100
reports **3.13.15**; its research environment uses that same interpreter. Patch
versions can vary by runtime image and are recorded in artifacts. No broker
orders have been submitted. `plan.html` is preserved.

## Checks actually performed

| Check | Result | Scope |
| --- | --- | --- |
| Local interpreter | Python 3.13.16 | Project `.venv`; supersedes the original 3.12 requirement |
| Hash lock and installation | Passed | 79 resolved dependencies, reviewed direct pins, editable package |
| `pip check` | Passed | No broken requirements in the research environment |
| Full unittest discovery | **68 passed; zero skips or failures** | Includes real dependency-backed RL acceptance |
| Gymnasium/SB3 checkers | Passed | Seeded reset, observation/action spaces, terminal/truncation behavior |
| CPU algorithm smoke | Passed | Exactly PPO 512, SAC 1024 and TD3 1024 transitions on synthetic fixtures |
| Checkpoints/replay/frozen inference | Passed | Save/load prediction parity, normalization immutability, schema/feed/corruption rejection, off-policy replay continuation |
| Reporting | Passed | Plotly HTML/JSON and dependency-free report behavior |
| Optional SDK wiring | Passed with mocked clients | Actual Yahoo/Alpaca package imports; no Alpaca endpoint/credential audit |
| Calendar regression | Passed | Holiday start, weekend end and exclusive session endpoints |
| Colab artifact validation | Passed | Rejects remote exceptions, missing jobs, wrong Python and corrupted weights even if CLI exits zero |
| Source/script/test compilation | Passed | Syntax compilation; no static type-check claim |
| Actual Yahoo daily download | Passed | Four ETFs, 4,211 bars each, no missing XNYS sessions |
| Real-data CPU preflight | Passed | PPO 512 transitions, validation report, saved model and frozen bundle on Yahoo data |
| A100 research execution | Awaiting retry after fix | Python/CUDA hardware verified; notebook backend import failure fixed |

Commands used for current local acceptance:

```bash
.venv/bin/python --version
.venv/bin/python -m pip check
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts tests
```

The full suite completed in 32.253 seconds on Python 3.13.16. The earlier
Python 3.12 acceptance also passed but is historical evidence, not the supported
runtime. No Ruff, mypy or pytest invocation is claimed. Integration smoke checks
verify wiring and accounting, not profitable learning.

## Verified market dataset and CPU artifacts

- Yahoo snapshot: `data/raw/7836ed5c18b90aaa6d40b43b08e43a8d86a76589ba0fd3b3822114adbd56bc75`.
- Dataset: `data/datasets/efbd5f5095604fc2c3d7e280c5d0d4c043a160525cca7daa5e24ab4374425aec`.
- Coverage: SPY, QQQ, IWM and DIA, 2010-01-04 through 2026-09-30; 4,211 bars per symbol.
- Training: 2,700 feature rows per symbol, 2010-03-16 through 2020-12-02.
- Validation: 733 rows per symbol, 2021-01-04 through 2023-11-30.
- Reserved test: 689 rows per symbol, 2024-01-02 through 2026-09-30.
- Training/validation ends have 20-session purges. Preprocessing fits training only; the unverified override is false.
- [Python 3.13 real-data CPU validation report](../artifacts/cpu313-preflight/runs/20261014T161538-ppo-11-f6ea2754/validation.html).
- [CPU run metadata](../artifacts/cpu313-preflight/runs/20261014T161538-ppo-11-f6ea2754/metadata.json).
- [CPU frozen manifest](../artifacts/cpu313-preflight/frozen/ca9d6a2324e2182099663301139272c733d872eaf0dcfb69462d4e6cad31d47e/manifest.json).

Yahoo availability is reconstructed; this download does not prove historical
publication vintages. Split-adjusted OHLC and cash-dividend semantics are recorded
in provenance. Raw source responses, coverage and immutable checksums are saved.
Generated data/artifacts are ignored by Git; these links require the local outputs.

## Remote attempts, compute and recovery

The cumulative ledger remains `.runtime/usage-ledger.json`; it is never reset
between retries. After five attempts, conservative cumulative usage is 0.722489 units,
with zero debit reported by the delayed account meter. All five are stopped.
Owned A100 allocations report 5.30 units/hour. The project ceiling stays 150 units,
with work stopping by 130 and a two-hour allocation deadline including setup,
training and retrieval.

Prior attempts:

1. `rl-stocks-smoke-f4c81ae30fb8`: allocation returned Service Unavailable; backend absence verified.
2. `rl-stocks-smoke-8099d697d068`: remote kernel failed the former Python 3.12 gate. Failure archive retrieved; cleanup verified. This exposed the CLI's zero-exit remote-exception behavior, now covered by artifact validation.
3. `rl-stocks-smoke-06fa5663634f`: older-interpreter worker failed before training. Partial archive retrieved; cleanup verified. The user then required matching Colab Python, and older-interpreter provisioning was removed.
4. `rl-stocks-smoke-68624bf171b8`: Python 3.13.15 and A100 verified; SB3 import failed because an inherited inline Matplotlib backend was unavailable in the research environment. Failure archive retrieved; cleanup verified. The launcher now selects the headless Agg backend and has a real SB3 import regression test.

5. `rl-stocks-smoke-897ccaeabfd6`: bootstrap installation failed before worker execution; cleanup verified. The runner now supports shared-ledger CPU/T4 preflight and separates logs by session.

Subprocess stdout/stderr and worker tracebacks are now retained durably in final
artifacts. Metering, downloads and teardown remain bounded; no budget-stop restart
is automatic. Final charges, retrieved models and cleanup will be recorded after
this attempt finishes.

## Remaining research and execution milestones

- Finish A100 smoke and the initial seed-11, 50,000-transition PPO run.
- Run matched 50,000-transition experiments across algorithms and seeds 11/23/37.
- Perform feature/reward ablations using training and validation only.
- Freeze a candidate before the held-out test audit; test data has not selected checkpoints.
- Audit actual Alpaca feeds/dividends, then add persistent shadow inference and paper-order lifecycle work.

Advanced feature families, extra reward terms and broker orders remain disabled.
See [testing.md](testing.md), [setup.md](setup.md) and [colab.md](colab.md) for procedures.
