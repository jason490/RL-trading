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
| Full unittest discovery | **86 passed; zero skips or failures** | Includes real dependency-backed RL acceptance |
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
| Colab CPU acceptance | Passed | PPO 512 / SAC 1024 / TD3 1024; artifacts verified despite CLI stream failure |
| A100 smoke and research | Passed | All three CUDA smokes plus PPO 50,000, seed 11; NVIDIA A100-SXM4-40GB, Python 3.13.15 |
| Retrieved A100 inference | Passed | All five bundles: local CPU deterministic save/load parity, frozen normalization and unchanged checksums |

Commands used for current local acceptance:

```bash
.venv/bin/python --version
.venv/bin/python -m pip check
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts tests
```

The latest full suite completed in 61.773 seconds on Python 3.13.16, including
the new supply/demand and sector tests. The earlier 69-test acceptance completed
in 32.703 seconds before these additions. The earlier
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

## Expanded research inputs — local acceptance after the baseline run

The implementation now supports configurable tradable/context symbols,
incremental D1/W1 supply/demand features and sector-ETF rotation. The existing
A100 models were trained before these additions and still use seven daily market
inputs. No new Colab allocation or expanded-universe neural training was started
for this feature change; the cumulative ledger remains 1.479525 units.

Actual Yahoo retrieval produced:

- Bar snapshot `data/raw/fced493827a124d4fb499d1c0e40009dc2420e19cd8712441ef65372d589457a`: 37 symbols, 152,227 bars, no missing expected exchange sessions within declared history boundaries; source responses and per-symbol chunks retained.
- Sector snapshot `data/sectors/6612a6c493b468f58e523ead790196a3cd867416552443d02bcaaa46315d1ea6`: current dated metadata for all 22 stocks, matching the configuration's categories. These are not historical classification vintages.
- Coverage spans 2010–September 2026 for the stocks/broad ETFs and older sector ETFs. XLC begins 2018-06-19; XLRE begins 2015-10-08. Their pre-history periods are explicitly recorded and masked in features.

Four datasets were built with verified coverage, no research override, identical
episode dates, 26 tradable symbols, three chronological splits per symbol and
training-only preprocessing. Classification labels remain metadata; the default
rotation features use historical ETF prices. The dated-record gate rejects
`--sector-labels` when stocks lack training-era classifications.

| Dataset | Identity | Market inputs | Environment observation size |
| --- | --- | --- | --- |
| Expanded baseline | `dfd600f62d5f12f7aa1c62b64a3fe2a1d00a48b3264f2a7fff31354958a57059` | 7 | 12 |
| D1/W1 supply/demand | `420fead30751749541bb989e0d671b8f58849d88e6dbe4b3de54697b13b44c32` | 33 | 38 |
| Sector rotation | `49ed77771b1e4f0c16435daa2de5416eda107cac19b8bb255382d9f6268cc2c7` | 62 | 67 |
| Combined | `062ae018bf7ed0a01bed50695b824238ebc4de89bae8fa0ae09077ac007a5e25` | 88 | 93 |

All four passed real Gymnasium and SB3 checkers and three no-trade steps for
each of 26 training-symbol episodes, with finite observations/rewards and
unchanged normalization. The old frozen bundle rejects the combined schema.
Test episodes were constructed but no test performance was evaluated.
Local evidence is saved in `.runtime/expanded-datasets.json`,
`.runtime/expanded-environment-audit.json` and
`.runtime/final-expanded-acceptance.txt`. The full suite has 86 passing tests
and no skips; `pip check` and syntax compilation also passed. No new lint/type
check is claimed. Generated data/runtime evidence remains excluded from Git.

Next is lower-credit acceptance of the expanded model inputs followed by matched
50,000-transition A100 studies, using the existing cumulative budget. Review
[sector/universe scaling](sectors-universe.md#scaling-to-hundreds-of-stocks)
before adding hundreds of symbols.

## Completed A100 training and retrieved artifacts

The final owned session was `rl-stocks-a100-0343183b915b`, allocated after the
successful Colab CPU preflight. Source revision was
`85bb0badfef319c06bffc141097faeab569fcb12`, with a clean working tree at upload.
The research environment used the VM's own Python 3.13.15 and the committed hash
lock. Hardware was NVIDIA A100-SXM4-40GB. Its completion manifest and every frozen
bundle's file checksums/identity were validated before acceptance.

| Job | Device | Actual transitions | Training duration |
| --- | --- | --- | --- |
| PPO smoke | CPU | 512 | 9.83 s |
| PPO smoke | CUDA | 512 | 13.16 s |
| SAC smoke | CUDA | 1,024 | 31.98 s |
| TD3 smoke | CUDA | 1,024 | 23.27 s |
| PPO, seed 11 | CUDA | 50,000 | 140.33 s |

Durations above include training/validation/freezing, not VM setup and download
time; every allocated stage counts in the usage ledger. The CPU job inside this
A100 allocation was an additional check, not the lower-credit preflight.

The full PPO run is `20261014T165221-ppo-11-f631b986`. Equal-weight mean net
validation return selected the checkpoint observed at 30,000 transitions
(11.2429857%); final post-update 50,000-transition weights scored 5.7217291%.
These are 2021–2023 validation simulations under modeled costs, not held-out
performance. PPO's on-step callback precedes its rollout optimizer update.
`best.zip` and the frozen bundle hold the selected weights; `latest.zip` holds
the final weights. No test data was evaluated or used for checkpoint selection.

The retained baseline has seven daily market inputs and five portfolio-state
inputs, schema `daily-indicators-v1`. Supply/demand and weekly features are not
included. The selected bundle is
`e71f26e9451eb5f16a84b7807896220b8c7e39125e6e5aa9bd4c74256805a675`;
it remains `paper_eligible=false`.

Retrieved outputs, available locally:

- [Final artifact archive](../artifacts/colab/rl-stocks-a100-0343183b915b/artifacts.tar.gz), 16,681,515 bytes; periodic checkpoint archive also retrieved.
- [Completion manifest](../artifacts/colab/rl-stocks-a100-0343183b915b/retrieved/artifacts/completion.json), including every completed job.
- [Full PPO metadata](../artifacts/colab/rl-stocks-a100-0343183b915b/retrieved/artifacts/runs/20261014T165221-ppo-11-f631b986/metadata.json).
- [Selected frozen manifest](../artifacts/colab/rl-stocks-a100-0343183b915b/retrieved/artifacts/frozen/e71f26e9451eb5f16a84b7807896220b8c7e39125e6e5aa9bd4c74256805a675/manifest.json).
- [Selected-weights validation report, CPU replay](../artifacts/colab/rl-stocks-a100-0343183b915b/retrieved/artifacts/runs/20261014T165221-ppo-11-f631b986/selected-validation-cpu.html).
- [Final-weights validation report, A100](../artifacts/colab/rl-stocks-a100-0343183b915b/retrieved/artifacts/runs/20261014T165221-ppo-11-f631b986/validation.html).
- [Local inference audit](../artifacts/colab/rl-stocks-a100-0343183b915b/retrieved/artifacts/local-inference-audit.json).

All five downloaded bundles loaded on local Python 3.13.16/CPU, matched their
saved `best.zip` predictions exactly on the same device, repeated predictions
deterministically and left normalization/files unchanged. The selected PPO's
CPU validation replay scored 11.24298574%; this small backend difference is not
a claim of bit-for-bit CPU/CUDA parity. Reports are saved outside immutable
bundles. The default run's `validation.html` reflects the final validation call,
so use the separately labeled selected-weights report for the chosen bundle.

## Remote attempts, compute and recovery

The cumulative ledger remains `.runtime/usage-ledger.json`; it is never reset
between retries. After seven attempts, conservative cumulative usage is
**1.479525 units**, leaving **148.520475 units** of the total ceiling and
128.520475 units before the working stop threshold. All seven attempts are
marked stopped and backend termination/absence was verified.

Account balance fell from 382.82 to 382.03 units over these attempts, an observed
debit of **0.79 units**; the last allocation's within-attempt balance debit was
0.56 units. Delayed metering means the observed debit may not be the final bill.
Budget decisions retain the larger conservative ledger estimate. The final
`colab usage` reading reported balance 382.03, **0.00 units/hour and zero active
assignments**. Owned A100 allocations reported 5.30 units/hour; CPU reported
0.08 units/hour. The project ceiling stays 150 units,
with work stopping by 130 and a two-hour allocation deadline including setup,
training and retrieval.

Prior attempts:

1. `rl-stocks-smoke-f4c81ae30fb8`: allocation returned Service Unavailable; backend absence verified.
2. `rl-stocks-smoke-8099d697d068`: remote kernel failed the former Python 3.12 gate. Failure archive retrieved; cleanup verified. This exposed the CLI's zero-exit remote-exception behavior, now covered by artifact validation.
3. `rl-stocks-smoke-06fa5663634f`: older-interpreter worker failed before training. Partial archive retrieved; cleanup verified. The user then required matching Colab Python, and older-interpreter provisioning was removed.
4. `rl-stocks-smoke-68624bf171b8`: Python 3.13.15 and A100 verified; SB3 import failed because an inherited inline Matplotlib backend was unavailable in the research environment. Failure archive retrieved; cleanup verified. The launcher now selects the headless Agg backend and has a real SB3 import regression test.

5. `rl-stocks-smoke-897ccaeabfd6`: bootstrap installation failed before worker execution; cleanup verified. The runner now supports shared-ledger CPU/T4 preflight and separates logs by session.

6. `rl-stocks-cpu-fcb20e48ed3d`: CPU preflight passed with PPO 512 / SAC 1024 / TD3 1024 on the verified Yahoo dataset, Python 3.13.16 and 0.08 units/hour. A CLI transport error followed completion; the downloaded completion manifest and all bundle checksums were independently verified. All three frozen bundles also passed local deterministic inference and normalization-freeze checks. Cleanup verified.

7. `rl-stocks-a100-0343183b915b`: all five jobs in the table above completed. Final and checkpoint archives retrieved, completion and bundle checksums validated, local frozen inference verified. Attempt estimate 0.752320 units. Owned session stopped; backend absence and zero active assignments verified.

Subprocess stdout/stderr and worker tracebacks are now retained durably in final
artifacts. Metering, downloads and teardown remain bounded; no budget-stop restart
is automatic. There is no outstanding cleanup failure or active owned session.
Retain the ledger and downloaded artifacts for future jobs; do not reset usage.

## Remaining research and execution milestones

- Run matched 50,000-transition experiments across algorithms and seeds 11/23/37.
- Train on the implemented [causal D1/W1 supply/demand inputs](supply-demand.md) and [sector rotation](sectors-universe.md), then run matched feature ablations using training and validation only; M1 is later.
- Perform other feature/reward ablations separately.
- Freeze a candidate before the held-out test audit; test data has not selected checkpoints.
- Audit actual Alpaca feeds/dividends, then add persistent shadow inference and paper-order lifecycle work.

Supply/demand and sector rotation are opt-in; the previously trained model uses
neither. Other advanced feature families, extra reward terms and broker orders remain disabled.
See [testing.md](testing.md), [setup.md](setup.md) and [colab.md](colab.md) for procedures.
