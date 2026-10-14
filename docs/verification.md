# Verification and milestone status — 2026-10-14

The system is implemented for local research with explicit pending acceptance. **Colab execution is deferred at the user's request.** No remote session or broker order was created. The original `plan.html` remains preserved.

## Checks actually performed

| Check | Result | Scope |
| --- | --- | --- |
| Python runtime | 3.12.14 | Real project-local interpreter |
| Unit discovery | 51 tests; **46 passed, 5 skipped**, no failures | Standard-library core plus explicit integration skips |
| Source/script/test compilation | Passed | Syntax compilation; not dependency imports or static type checking |
| Local CLI help | Passed | ingest/dataset/train/compare command registration |
| Full offline audit | Passed | All four synthetic symbols, requested dates, immutable accumulation/datasets, validation accounting, baselines, HTML/JSON reports |
| Documentation links | Checked after writing these guides | Local relative Markdown targets |
| Official CLI wrapper | Version 0.7.4 and usage help worked | Writable log/settings/history/state; no allocation |
| Earlier backend usage attempt | Failed to provide balance/rate due network/connection failure | No account balance inferred |
| `plan.html` preservation | SHA-256 `26c4bf4b910013bf47231d47b74ccf82ece9fe05296cd5eefddccc300d3a365f` | File not edited |

Commands used:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m compileall -q src scripts tests
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local --help
PYTHONPATH=src .venv/bin/python scripts/offline_audit.py
```

The skipped methods cover algorithm configuration with real SB3 imports; Gymnasium/SB3 checkers and seeded/boundary behavior; actual PPO/SAC/TD3 train/save/load/report/bundle/replay continuation; mocked official Alpaca wiring; and mocked Yahoo wiring. NumPy, Gymnasium, torch, SB3 and Plotly are unavailable in this sandbox. Their skips do **not** count as smoke acceptance. The dependency-pin audit is tested with mocked installed-version metadata, not native package imports.

No Ruff, mypy, pytest invocation, real neural training, market-data endpoint audit or A100 run is claimed. No usable Git checkout is exposed here; run metadata records that absence and includes a package-source hash.

## Generated offline artifacts

- Snapshot: `data/raw/7529f85d10ebd9fbe02bcc283a4c786040f5fc3b99b2d454fcaa8795db64c5a2`
- Dataset: `data/datasets/967df26a85479bc63b0126a88164e3be5da49e709f2861a0d3949fe62dffb03d`
- [Interactive synthetic accounting report](../artifacts/offline-audit/20261014T054155-222304b6/report.html)
- [Machine-readable report](../artifacts/offline-audit/20261014T054155-222304b6/report.json)
- [Audit metadata](../artifacts/offline-audit/20261014T054155-222304b6/metadata.json)

The audit uses only validation segments and a fixed alternating exposure fixture. It is not a trained policy and supports no market profitability conclusion. Synthetic calendars/retrieval semantics and the explicit unverified override are recorded in the manifest. Current reports use the native SVG fallback because Plotly is unavailable.

## Compute and cleanup

Project sessions allocated: **0**. Project compute units allocated/consumed by owned sessions: **0**. The unspent project ceiling is **150 units**; the actual account balance is unknown. There are no retrieved Colab artifacts and no owned remote session requiring cleanup. `.runtime/usage-ledger.json` persists the unused ceiling and an empty attempt history for the later run.

Budget/cleanup tests use mocked sessions only. They exercise conservative estimates, cumulative attempts, working-limit/deadline refusal, metering failure, provisioning failure, verified termination and retained recovery status after failure. They do not prove actual account metering or backend termination. The runtime's handled-interruption paths and persisted identity are implemented; power loss/SIGKILL still require manual recovery.

## Pending acceptance and intentional scope

| Area | Next acceptance |
| --- | --- |
| Dependencies | Generate/review transitive hash lock; install reviewed pins; run pip check and native imports |
| Providers | Actual Yahoo pre-2016/through-September-2026 coverage; Alpaca endpoint/feed/dividend audit |
| Environment/RL | Real Gymnasium/SB3 checkers; exactly 512 PPO / 1024 SAC / 1024 TD3 CPU smoke |
| Frozen inference | Actual neural save/load parity, schema/feed/corruption rejection and replay continuation |
| Colab | Later user-run A100 smoke, metering, durable downloads and verified cleanup within the persistent 150-unit ceiling |
| Research | Matched multi-seed experiments/ablations and post-freeze held-out audit; no performance claims yet |
| Live/paper | Shadow orchestration and Alpaca order lifecycle remain separate later milestones |

Read [testing.md](testing.md) for acceptance procedures, [setup.md](setup.md) for lock installation and [colab.md](colab.md) for the deferred budget-controlled launch. The implemented baseline keeps advanced data families, extra reward terms and broker orders disabled.
