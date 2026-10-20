# Verification and milestone status — 2026-10-19

The project targets **Python 3.13**, matching Colab's current interpreter family.
The local acceptance environment is **3.13.16**. The verified A100 image
reported **3.13.15**; its research environment uses that same interpreter. Patch
versions can vary by runtime image and are recorded in artifacts. No broker
orders have been submitted. `plan.html` is preserved.

## Checks actually performed

| Check | Result | Scope |
| --- | --- | --- |
| Local interpreter | Python 3.13.16 | Project `.venv`; supersedes the original 3.12 requirement |
| Hash lock and installation | Passed | 79 resolved dependencies, reviewed direct pins, editable package |
| `pip check` | Passed | No broken requirements in the research environment |
| Full local acceptance | **207 passed; zero skips or failures** | Includes real dependency-backed RL acceptance; latest run 87.73 seconds, 15 warnings (two additional checker warnings) |
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

The original completed-study suite ran in 67.387 seconds on Python 3.13.16 (101 tests),
including supply/demand, sector, matched-study, public-asset, graceful-handoff and streamed-transfer checks. The earlier 69-test acceptance completed
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
inputs. That feature implementation used no new Colab allocation. Subsequently the
user authorized the next steps; expanded CPU acceptance passed and matched A100
studies have started under the same cumulative ledger.

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
`.runtime/final-expanded-acceptance.txt`. The full suite now has 101 passing tests
and no skips; `pip check` and syntax compilation also passed. No new lint/type
check is claimed. Generated data/runtime evidence remains excluded from Git.

Expanded CPU session `rl-stocks-cpu-8c7f01297b8a` completed PPO 512 / SAC 1,024 /
TD3 1,024 on the combined 88-market-input dataset, Python 3.13.16. Completion,
bundle checksums and all three local CPU frozen save/load predictions were
verified; normalization remained unchanged. Final artifacts were retrieved and
backend shutdown verified. Attempt estimate: 0.016613 units; measured debit:
0.01 units. Cumulative estimate after preflight: 1.496138 units.

The committed [matched study protocol](experiments.md) now runs four datasets,
three algorithms and seeds 11/23/37. All 36 cells and the separate held-out audit
are complete. The first 29 cells used CUDA; the final seven used CPU after a
user-directed cost change. See the final study evidence below. Review
[sector/universe scaling](sectors-universe.md#scaling-to-hundreds-of-stocks)
before adding hundreds of symbols.

## Completed A100 training and retrieved artifacts

The initial four-ETF A100 session was `rl-stocks-a100-0343183b915b`, allocated after the
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

- Final and periodic archives were retrieved and validated; redundant archive copies were later removed after extracted outputs and published models were verified.
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

## Initial four-ETF attempts, compute and recovery (historical snapshot)

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

## Completed 36-run study and held-out audit — 2026-10-15

All four datasets, three algorithms and three seeds completed exactly 50,000
transitions per research cell. [Validation selection](../results/sector-study-2026-10-14/validation-study.json)
was committed and pushed in `179b358` before opening the test interval.
Learning-package files were byte-identical across source revisions; full uploaded
package hashes were independently checked against clean recorded Git trees.
All 36 frozen bundles passed same-device CPU save/load prediction parity, repeat
determinism, unchanged normalization and unchanged file checksums.

| Inputs | Algorithm | Mean selected validation return | Seed standard deviation |
| --- | --- | ---: | ---: |
| baseline | PPO | 12.97% | 3.57 pp |
| baseline | SAC | 15.60% | 5.95 pp |
| baseline | TD3 | 4.18% | 8.90 pp |
| combined | PPO | 7.82% | 15.21 pp |
| combined | SAC | 19.79% | 27.08 pp |
| combined | TD3 | 5.25% | 19.07 pp |
| rotation | PPO | -1.92% | 3.79 pp |
| rotation | SAC | 4.12% | 7.82 pp |
| rotation | TD3 | 5.09% | 10.96 pp |
| supply_demand | PPO | 9.00% | 9.46 pp |
| supply_demand | SAC | 5.93% | 8.35 pp |
| supply_demand | TD3 | 3.52% | 3.66 pp |

Matching buy-and-hold averaged 30.60%; cash returned 0%. Validation selected
combined-feature SAC, candidate seed 23 (51.05% selected validation return).
The family mean was 19.79%, with 27.08 percentage points of seed variation.
The first 29 cells used CUDA and the final seven CPU, including all combined
SAC/TD3 seeds. This backend confound limits attribution of score differences
to the input families alone; it is not a controlled hardware comparison.

The [one held-out audit](../results/sector-study-2026-10-14/test-audit.json) evaluated
all three frozen selected-family seeds on 2024–September 2026. No seed, checkpoint
or feature family was reselected using these outcomes.

| Seed | Mean net test return across 26 symbol episodes |
| --- | ---: |
| 11 | 9.22% |
| 23, preselected candidate | 13.90% |
| 37 | 9.38% |
| Three-seed mean | 10.83% |
| Buy-and-hold | 48.46% |

Seed standard deviation was 2.66 percentage points. Mean per-symbol maximum
drawdown was 26.82%, versus 25.76% for buy-and-hold. Normalization and frozen
files were unchanged throughout the audit. The models underperformed the
benchmark and remain paper-ineligible. Returns are total interval returns from
independent symbol simulations, not an annualized return or investable portfolio.

[GitHub release](https://github.com/jason490/RL-trading/releases/tag/sector-study-2026-10-14)
contains 36 research bundles and four startup-smoke bundles, weights/metadata,
indexed validation charts, held-out charts and selection/audit summaries.
All ten server-reported asset digests and sizes match local files. Raw provider
bars, replay data and private runtime logs were excluded from publication.
The [verification record](../results/sector-study-2026-10-14/release-verification.json)
contains the exact hashes. Extract the validation report archive with its
`validation/` directory beside the index.

## Final budget and cleanup

All **22 task-owned attempts** are stopped, with backend shutdown verified.
The final account reading reported zero active assignments and 0.00 units/hour.
Conservative cumulative ledger usage is **33.090762 units**; rate-integrated
estimates total 32.779824 units. The observed account balance debit is 29.97 units,
subject to delayed billing. The conservative ledger governs the budget: **116.909238
units remain** under the 150-unit ceiling, with 96.909238 before the 130-unit
working stop. No budget-stop restart occurred and the ledger was never reset.
Detailed sanitized attempts are in [usage-summary.json](../results/sector-study-2026-10-14/usage-summary.json).

CPU continuation used Python 3.13.16 and a reported 0.08 units/hour; A100 used
Python 3.13.15 and about 5.30 units/hour. CPU PPO seed 37 completed training and
validation in 150.87 seconds. An independent 12-second A100 sample during PPO
reported 0% GPU utilization and 544 MiB allocated; it may overlap CPU-heavy
validation and is not a whole-run profile. The diagnostic kernel was deleted.

After publication verification, cleanup removed **293 temporary artifact files**,
4.10 GiB of logical artifact data, the obsolete task-created Python 3.12
environment and regenerable dependency cache. Actual filesystem space recovered:
**7.82 GiB**. Final selected/latest models, final replay buffers, detailed reports,
immutable data, source, current Python 3.13 environment and usage ledger remain.
All 53 retained frozen directories and all release assets were rechecked and
unchanged; post-cleanup dependency checks and fresh-process deterministic frozen inference passed. Three archives without verified
extracted replacements were conservatively retained. See the exact
[cleanup inventory](../results/sector-study-2026-10-14/cleanup.json) and
[retention guide](artifact-retention.md).

The full 101-test suite passed in 67.387 seconds, with zero skips/failures.
After the later report-size fix, its focused publication regression passed
again. Syntax compilation passed; `plan.html` retains its original checksum.

## Remaining research and execution milestones

- Repeat controlled CPU comparisons before attributing gains to added features.
- Profile sequential simulation/evaluation; batch or vectorize where measured useful.
- Investigate turnover/cost reduction as separate training/validation experiments.
- Audit historical universe membership, delistings and sector vintages before broad scaling.
- Add D1-versus-W1 isolation, optional monthly context and other families separately.
- Use a fresh locked audit interval for future claims; do not repeatedly tune against this test.
- Audit actual Alpaca feeds/dividends, then build persistent shadow inference before paper orders.

Other advanced families, extra reward terms and broker orders remain disabled.
See [testing.md](testing.md), [setup.md](setup.md) and [colab.md](colab.md) for procedures.

## Completed CPU learning study — 2026-10-15

The [learning protocol](learning-experiments.md) completed all 15 CPU cells,
including three 250k runs. Full local acceptance passed **115 tests in 76.265
seconds**, zero skips/failures (`.runtime/learning-full-acceptance.txt`). The earlier
111-test protocol preflight preceded four additional CLI transport tests.
Coverage includes compact curve calculations, post-update replacement, reward
scaling without equity changes, segmented SAC, complete remote curve acceptance,
paired multi-seed gates, terminal connection detection and bounded prestart retry.

Every remote run passed completion, hardware/Python, step-count, curve and frozen
checksum acceptance. Independent local checks verified exact CPU prediction parity
on three observations per bundle, normalization immutability and schema/shape
rejection for all 15 bundles. Clean Git source proofs match the learning code
across all cells; dependencies and simulator assumptions match. See
[bundle evidence](../results/learning-study-2026-10-15/bundle-verification.json) and
[study summary](../results/learning-study-2026-10-15/summary.json).

Two early connection failures were stopped and retained in the cumulative usage
ledger. All 39 project-owned attempts are verified stopped. Conservative project
usage is 34.032604 units, including approximately 0.941842 for this learning series;
rate estimates sum to 33.721666 and observed account debits to 30.51. These are
alternative project-wide measures, not amounts to add together. Latest account
balance was 352.08 units; remaining project ceiling is 115.967396 units.

Selected validation return improved to 23.96%, but no stage passed the declared
risk gate. No new final-test evaluation or broker orders occurred. The earlier
36-run study and held-out audit remain historical and unchanged.

The [learning release](https://github.com/jason490/RL-trading/releases/tag/learning-study-2026-10-15)
contains 15 assets; every server-reported SHA-256 digest matched locally. Only
after that verification, cleanup removed 61 duplicate transfer/rolling-checkpoint
files (1,906,588,287 logical bytes; observed free-space increase 1,906,921,472 bytes).
All 15 frozen bundles, 33 milestone models, final replays and curve journals remain.
Post-cleanup bundle/release hashes and the immutable `plan.html` hash still match.
See [release evidence](../results/learning-study-2026-10-15/release-verification.json)
and [cleanup evidence](../results/learning-study-2026-10-15/cleanup.json).

## Indicator follow-up preflight — 2026-10-16

The new matched 250k indicator protocol is committed separately from the exhausted
learning study. Local acceptance passed **120 tests in 76.990 seconds**, zero
skips/failures (`.runtime/indicator-full-acceptance.txt`). New checks cover bounded
execution windows with retrieval/shutdown reserves, factorial input attribution,
paired gains and drawdown gates, invalid/mismatched controls, real report export
and training-only regime coverage independent of future validation data.

Colab CLI 0.7.4 was reused. Pre-allocation account balance was 352.07 units, rate
0.00/hour and active assignments zero; the project ledger remains cumulative from
34.032604 units. The active CPU kernel reports Python **3.13.16**. Immutable dataset
checks confirm identical train/validation bars across 7/33/62/88-feature families.
This is preflight evidence, not a completed new indicator-family result.
See [current results](../results/indicator-followup-2026-10-16/README.md).

The within-run 100k-anchor reporting check brings the suite to **121 tests**,
all passing in 76.908 seconds (`.runtime/indicator-full-acceptance-final.txt`).
The three combined-input 250k cells completed and passed remote artifact/source
acceptance. Selected mean validation return is 46.82%, final mean −1.73%; the
feature gate fails final-return and final-drawdown conditions. Supply/demand-only
and sector-only cells remain in progress; do not call the full follow-up complete.

The lost-stream export recovery adds three checks for authenticated probe errors,
continued metering/checkpoint retrieval, and deadline/budget stops. Full local
`pytest -q` acceptance passed **124 tests in 78.80 seconds**, with zero skips or
failures. Eleven warnings describe Gymnasium's unbounded numeric observation
space and the optional SDK's deprecated WebSocket import. No learning code or
reward changed; `plan.html` retains its original checksum.

RecurrentPPO local support passes 512-step train/evaluate/save/load acceptance,
exact multi-step action/hidden-state parity, explicit reset parity, immutable
normalization, invalid memory rejection and reversed stock-order evaluation.
Colab's active default kernel was rechecked as **3.13.16** before adding
`sb3-contrib==2.7.0`; the hash lock retains all prior pins. Full local acceptance
passed **131 tests in 90.02 seconds**, zero skips/failures (13 dependency/checker
warnings). Recurrent research training has not yet started.

The stability-report gate and explicit separate actor/critic LSTM settings pass
full local acceptance: **134 tests in 91.41 seconds**, zero skips/failures, with
13 known checker/dependency warnings. The new paired report rejects incomplete
endpoints and mismatched source/dependency/cost assumptions, and requires return
gains as well as reduced deterioration. No new comparison cell has been allocated
at this verification point.

The indicator follow-up is complete: nine new CPU SAC 250k cells plus three
reused basic-input controls. All source/dependency/data/cost comparisons match;
all nine new and three control frozen bundles pass exact CPU parity, immutable
normalization and schema/shape rejection. All feature gates fail. Cumulative
conservative usage is **35.098226 units**, with **114.901774 remaining** and all
49 owned attempts stopped. The new algorithm/memory study has not started.

The concurrent coordinator's full local acceptance passed **144 tests in
79.17 seconds**, zero skips/failures (13 known warnings). After tightening the
aggregate cleanup projection, all **14 targeted pool/control/plan checks** also
pass. Mocked lifecycle acceptance covers four concurrent mixed-hardware cells,
single-charge account metering, deadline/global stops, interrupts, metering loss,
cleanup failure and concurrent recovery, payload isolation and unowned-session
preservation. No actual concurrent VM allocation has occurred at this point.

The real concurrent CPU preflight is accepted for SAC 1,024 transitions, PPO
512 and RecurrentPPO 512. Three named Google Colab sessions were simultaneously
active; measured aggregate rate was **0.24 units/hour**. A transient completion
marker download timeout triggered checkpoint salvage and verified shutdown;
bounded marker/final-download timeout retries were fixed and tested, then only
the two unaccepted smoke cells were retried. All sessions stopped. Downloaded
bundles pass exact multi-step CPU prediction parity (including recurrent state),
immutable normalization and matching learning-source proofs. Full local
acceptance passes **146 tests in 86.94 seconds**, zero skips/failures.

## Completed concurrent algorithm study, 2026-10-16

All nine declared SAC/PPO/RecurrentPPO cells completed 100,000 transitions in
Google Colab CPU runtimes, with peak concurrency three. The coordinator retrieved
all artifacts and verified every owned session stopped. Clean learning-source and
dependency proofs match across arms. Selected/final means are SAC 19.70%/16.50%,
PPO 18.69%/16.90%, recurrent PPO 13.27%/6.13%; all improvement gates fail.
No new final-test evaluation or orders occurred. All nine frozen bundles pass
exact CPU sequence predictions, recurrent state/reset parity where applicable,
immutable normalization and schema/feed/shape rejection. Full local acceptance
remains 146 passed (86.94 seconds); no learning code changed during publication.
Cumulative conservative usage is 35.389379 units, with 114.610621 remaining.
[Results](../results/algorithm-stability-2026-10-16/README.md) retain all seeds,
10k-step curves, failed gates and saved-path deterioration attribution.

Publication verification confirms all 13 algorithm-study assets have matching
server SHA-256 digests and sizes. Cleanup removed 44 duplicate/temporary files
(678,561,987 logical bytes); 129 retained files were rehashed, including all nine
frozen bundles and nine endpoint milestones. Required replay, curves, validation
outputs, failed-preflight evidence, immutable data and recovery records remain.
No owned sessions are active. See the study's tracked release and cleanup records.

## Persistent shadow software acceptance, 2026-10-16

Full local acceptance passed **156 tests in 95.42 seconds**, zero skips/failures,
with 13 existing checker/dependency warnings. Ten new shadow tests cover restart
accounting and recurrent state, transactional rollback, duplicate prevention,
observed-before-open timing, protocol/feed/manifest/history rejection, missing
context, exchange times, future/stale bars, read-only collection and fixed-horizon
reporting. Static undefined/unused-name checks pass for the new modules.

All nine actual registered research candidates were checked on previously
inspected validation inputs for software parity: shadow features and ATR match
the stored datasets (26 symbols each), normalized observations exactly match the
training environment, frozen actions are finite and normalization is unchanged.
This is software acceptance, not fresh evaluation or forward performance.
See [acceptance evidence](../results/forward-shadow-2026-10-16/software-acceptance.json)
and [shadow operations](shadow-inference.md). No Colab session or dependency
change was required; cumulative project usage remains 35.389379 units.

The actual nine-arm campaign was registered from clean revision `e6db469` at
2026-10-16 22:55:19 UTC, and the scheduler's waiting/report cycle passed. Fresh
Yahoo preflight then failed strict OHLC geometry for the October 16 SPY bar
(high below open by approximately one cent), confirmed by a fresh retry. Prices
were not repaired or substituted. Unattended polling is **not running**; no
forward decisions or returns exist. See the [status](../results/forward-shadow-2026-10-16/status.json)
and [registration](../results/forward-shadow-2026-10-16/registration.json).
A persistent host and a successful feed audit are required before activation.

## User-approved SPY data correction, 2026-10-17

The user approved a narrowly scoped correction for Yahoo's October 16 SPY bar.
The shadow recipe pins `yahoo-spy-2026-10-16-high-v1`: only the original observed
OHLC checksum may raise high to open, capped at $0.011. Default Yahoo ingestion
and all subsequent bar validation remain strict. Immutable raw responses retain
the old high, and normalized receipts record the exact correction and revision.
A fresh real-data preflight now passes for all 37 required symbols, with exactly
one correction and unchanged historical overlap. No old training dataset or model
was modified. The original empty campaign is retained; a replacement must pin
the new receipt policy and implementation before forward signals.

Full acceptance passed **160 tests in 88.59 seconds**, zero skips/failures, with
13 existing warnings. All 14 targeted correction/shadow checks also pass, including
wrong-date/symbol/changed-source/larger-error rejection and audit persistence.
See [preflight](../results/forward-shadow-2026-10-16/correction-preflight.json).

Required missing prices remain rejected before any correction fingerprint is
computed. Optional nonfinite raw metadata is preserved with explicit JSON tags;
normalized prices and volume must still be finite. Collection caps its exclusive
request end at the latest completed exchange session plus one day, including
weekends/holidays. Fifteen targeted correction/shadow tests pass, covering these
cases and unchanged strict rejection. The earlier valid 37-symbol preflight is
retained; a later fresh Yahoo response changed October 16 SPY Close to NaN, so
activation remains blocked by missing required prices rather than the repaired
high/open inconsistency. See [latest diagnosis](../results/forward-shadow-2026-10-16/latest-feed-diagnosis.json).

Final full acceptance passes **161 tests in 87.42 seconds**, zero skips/failures,
with 13 existing checker/dependency warnings. No Colab runtime was allocated.

## Historical membership/terminal evidence layer, 2026-10-17

Full acceptance passed **170 tests in 89.47 seconds**, zero skips/failures, with
13 existing warnings. Nine targeted historical tests pass after final validation
hardening. Checks cover stable-ID ticker reuse, effective/publication-time joins,
later vintage corrections, future-outcome independence, missing failed-stock
prices, unresolved/non-cash/zero/buyout outcomes, delisting missing-return codes,
source attestations, immutable identity and secret-safe mocked API failures.
The CLI import/audit demo and HTML/CSV export passed on synthetic records only.

No real historical vendor evidence was imported or certified; no Alpha Vantage
API key or historical subscription was configured, and no API request, Colab
allocation, training, broker order or new held-out evaluation occurred. The current
retrospective 26-tradable sample and frozen policies remain unchanged. Coverage
approval is conditional on source attestations and `training_ready` remains false
until the documented historical episode/accounting integration is complete.
See [guide](historical-data.md) and [readiness record](../results/historical-coverage-2026-10-17/current-data-readiness.json).
# Alpaca credentials and quota check — 2026-10-17

Authenticated historical SIP access passed for SPY, QQQ, IWM and DIA over
2026-10-12 through 2026-10-14, with three canonical bars each, retained as an
immutable local snapshot. See [access evidence](../results/alpaca-access-2026-10-17.json).
The ignored `.env` remains untracked. Thirteen focused checks passed (three new credential/quota tests, the existing
mocked adapter test, and nine historical-evidence checks), with one existing
optional-SDK deprecation warning; unused-import checks passed. The quota test
simulates 201 HTTP requests across two clients, confirms pacing, read-only
transport, 429 cooldown and bounded handling of longer cooldowns without real
network traffic. Live acceptance used only the historical-data endpoint, with
no orders, training, Colab allocation or existing dataset/model changes.
Rate limiting is process-wide; use one downloader process for this account.

## Universe expansion preparation — 2026-10-17

The common Yahoo snapshot passed all requested calendar coverage gates for 70
symbols through 2023. Control and expanded datasets contain 70,200 and 158,546
training rows with exactly matching 19,058-row validation panels and no test
episodes. The separate Alpaca SIP audit retained 713 missing LIN sessions and
did not certify corporate actions. See [data and audit evidence](../results/universe-expansion-2026-10-17/README.md).

Full local acceptance passed **174 tests**, with 13 existing warnings, in 93.85
seconds. The final targeted two-test expansion check also passed after adding
the report rejection test: a large selected-checkpoint gain cannot pass when
final performance deteriorates, duplicate cells and mismatched baselines are
rejected, normalization fits training only, and no retired-test episodes are
created. Unused-import checks passed for changed modules/scripts/tests.
Colab execution acceptance and outcomes are recorded separately after retrieval.

## Completed universe comparison — 2026-10-17

All six declared 50k PPO cells completed on Colab CPU, Python 3.13.16, with the
same clean committed learning source. The control's five on-step checkpoints
through 50k exactly reproduce the prior seed-11 CPU run. Dataset validation
panels, baselines, features, costs and algorithm parameters are checked before
reporting. The retired test is absent and was never evaluated.

The expanded panel failed the improvement gate: selected/final mean returns
10.53%/5.60%, versus control 11.77%/10.49% and buy-and-hold 30.60%. Six actual
frozen bundles passed 468 exact CPU prediction comparisons, repeated-prediction
parity, full feature-order/normalization/data-manifest equality, immutable
normalization, and schema/feed/nonfinite rejection. The reusable verifier is
`scripts/verify_frozen_runs.py`; the final full local suite passed **175 tests**
with 13 existing warnings in 89.53 seconds, and changed-file unused-import checks
passed. No failures or skipped tests were hidden.

All seven owned allocations including smoke were stopped and verified. Measured
study debit was 0.05 units; conservative rate accounting charges 0.068809 units.
The preserved project ledger is 35.458188/150 units with 114.541812 remaining;
the working teardown threshold remains 130. All compute includes setup, allocated
idle time, transfers and teardown. No automatic longer run or new reward search
followed the failed gate. See [results and next gates](../results/universe-expansion-2026-10-17/README.md).

The [release](https://github.com/jason490/RL-trading/releases/tag/universe-expansion-2026-10-17)
has 13 assets whose server sizes and SHA-256 digests match local copies. Only
after that verification, cleanup removed 48 redundant files (394.43 MiB) and
rehashed all 146 retained files unchanged. Models, 50k endpoint weights, curves,
normalization, reports, datasets and the preserved ledger remain. The final CLI
check reports zero assignments and 0.00 units/hour. Publication and cleanup
inventories are linked from the results guide; `plan.html` is unchanged.

## Additional stock pull — 2026-10-19

The user requested more stock histories only. All 110 requested stocks now have
some history across separate sources: Yahoo supplies 107 stocks plus 15 ETFs
(425,350 bars, 2010–2023 subject to declared inceptions), and Alpaca SIP supplies
EA, AVB and EQR (6,036 bars, 2016–2023). Primary failures remain in coverage; the
feeds are not merged and training readiness is false. No research dataset,
evaluation, training or Colab allocation was added. The project ledger is unchanged.

Focused ingestion checks passed 17 tests; the full suite passed **177 tests**,
zero failures/skips and 13 existing warnings, in 88.71 seconds. New checks cover
exclusive date bounds, continued collection with failed-symbol retention, safe
error metadata, default fail-fast behavior and storage checksum rejection.
Changed-file unused-import checks passed. See the
[manifest, coverage and reproduction guide](../results/stock-pull-2026-10-19/README.md).

## Short-side attribution — 2026-10-19

Full acceptance passes 183 tests, zero skips/failures, 13 existing warnings.
New tests cover profitable falling-stock shorts after modeled costs, rising-stock
losses, disabled shorts, reversals, scale-outs, splits/dividends, gap stops,
early episode prefixes and reconciliation failures. Syntax and whitespace checks
pass. New evaluation reports include exact signed-cash-flow long/short attribution.

The saved nine-model 100k audit reconciled 468 symbol/seed/phase paths to equity;
maximum declining-cohort per-session residual was below $0.000001. Results are
in [the short-side evidence](../results/short-side-2026-10-19/README.md). No new
training or Colab allocation was required. CLI usage confirms zero active
assignments, 0.00 units/hour; the cumulative project ledger remains 35.458188.
The learned short-side objective is not reliably met. The bounded token-efficient
research workflow is documented; prospective confirmation remains outstanding.

The subsequent fixed-rule execution diagnostic completed all 104 rule/symbol
paths on the existing validation panel with no training or Colab allocation.
Two additional rule-contract tests pass; the combined direction/rule focused
suite passes 8 tests. Compilation and whitespace checks pass. The latest full
suite above remains the 183-test run before these two isolated rule tests were
added. See [execution evidence](../results/execution-rules-2026-10-19/README.md).

## Action-initialization preflight — 2026-10-19

The four-arm PPO screen is declared before allocation. Strict CPU plans and
archive checks verify initialization and final post-update curves. Native
control leaves weights and RNG unchanged; altered arms preserve other seeded
weights, both exposure signs and trainability, and pass save/load prediction
parity. Full acceptance passes 189 tests in 87.24 seconds, no skips/failures,
13 existing warnings. The subsequently added screen-gate regression and related
focused suite pass 15 tests. No real research training was performed locally.
Remote smoke and research results remain pending at this declaration.

Four distinct initialization CPU Colab smokes now pass: exactly 512 PPO
transitions each on Python 3.13.16, with valid post-update curves and frozen
bundles. Independent verification passes 312 exact CPU predictions, immutable
normalization/data manifests and incompatible schema/feed/nonfinite rejection.
All four owned sessions stopped. Measured debit was 0.0 units; conservative
estimate 0.021871; cumulative project 35.480059/150 units. The 50k research screen
is still pending; smoke returns are not model-performance evidence. See
[smoke evidence](../results/action-initialization-2026-10-19/smoke.json).

## Complete-case dataset acceptance — 2026-10-19

Explicit universe-scoped coverage retains the wider request's failures while
checking all retained stock/ETF/context coverage and verified corporate actions.
Default snapshot-wide checks and default dataset identities remain unchanged.
Full acceptance passes 193 tests in 111.38 seconds, no skips/failures and 13
existing warnings; the focused scope tests pass again with added missing-context
and absent-coverage checks. Syntax and whitespace checks pass.

The new 107-stock/four-ETF training dataset contains 298,946 feature rows, with
the exact original 26-symbol/19,058-row validation panel and no test episodes.
No unverified override, feed merge, network fetch or model training was used in
this preparation. See [evidence](../results/complete-case-107-2026-10-19/README.md).
The ongoing initialization screen retains its original dataset and captured source.

## Completed action-initialization study — 2026-10-19

Eight 50k PPO research cells and four 512-transition CPU smokes completed, all
on actual Colab Python 3.13.16. The wider-initialization variant passes its
relative three-seed gate: selected/final means 14.23%/13.89% versus standard
11.77%/10.49%, with final gains in all three seeds. Holding returns 30.60%;
absolute stock/ETF targets fail. Declining-stock returns worsen, with zero of
nine wide observations satisfying positive policy and short P&L together.

Twelve frozen bundles pass 936 exact predictions plus immutable data/normalizer
and rejection checks. Three controls exactly reproduce 39 policy tensors and
18 checkpoint returns. All 12 owned sessions stopped; final CLI rate/assignments
are zero. Conservative study charge 0.104793 units; cumulative 35.562980/150,
remaining 114.437020. No retired test was evaluated. Related confirmation-gate,
curve and variant-audit focused checks pass; the latest full suite remains the
193-test run documented above. All 17 public asset hashes/sizes match GitHub server digests. After verification,
52 duplicate archives/markers (612.30 MiB logical) were removed; all 228 retained
artifact hashes remain unchanged. Raw data, models, reports and the ledger remain.
See [complete evidence](../results/action-initialization-2026-10-19/README.md).

## Causal sampling preflight — 2026-10-19

Full local acceptance passes 207 tests in 87.73 seconds, no skips/failures and
15 warnings. New coverage verifies causal prefix labels, training isolation,
shared readiness eligibility, class balancing, exact legacy RNG draws and real
Gymnasium/SB3 checkers. Archive validation checks final sampling counters against
exact transitions; fleet allowance tests verify refusal before allocation,
registered-intent cleanup and owned teardown on a projected study-budget stop.

Preflight confirms the original raw 26-symbol/19,058-row validation episodes
match across both datasets and fingerprints all four pools. Eligible-start
persistent-downtrend shares are 2.27% (small) and 5.11% (large). Sampling modes
leave rewards, observations and inference normalization unchanged. No research
training or Colab allocation has occurred at this declaration. Protocol and
methodology are in [training-sampling.md](training-sampling.md).

All four sampler/data CPU Colab smokes now pass exact 512-step budgets, final
sampling-counter checks and owned shutdown on actual Python 3.13.16. Four frozen
bundles pass 312 exact CPU predictions, immutable normalization/data and rejection
checks across the two training normalizers. Conservative smoke usage is 0.027557
units; project cumulative 35.590538/150. No 50k result is claimed at this stage.
See [smoke evidence](../results/causal-sampling-2026-10-19/smoke.json).

## Causal sampling screen outcome — 2026-10-19

All four research cells completed 50k steps, with matching pool/source/baseline
hashes and exact training counters. No candidate passed: even the 16.20% small
balanced result deteriorated on ETFs versus the fresh control. No confirmation
or longer runs were added. Eight bundles across smoke/research pass 624 exact
CPU predictions and immutable preprocessing checks; all eight owned sessions
stopped. Conservative project usage is 35.638336/150 units (study 0.075355).
No retired test was evaluated. Training-only diagnostics and a synthetic
execution-state aliasing witness are retained in the results folder. All 17 public assets match GitHub server SHA-256/size checks. Cleanup removed
36 duplicate archives/markers (512.95 MiB logical), with all 160 retained artifact
hashes unchanged. Raw data, models, reports, source export and ledger remain. [Full evidence](../results/causal-sampling-2026-10-19/README.md).

## Execution-state observations — 2026-10-19

Versioned legacy/masked/full inputs, frozen schema/model-shape checks and shared
training/evaluation/shadow state construction are implemented. Full regression
passed 214 tests; final profile/runtime additions passed 32 focused tests and
the explicit audit fix passed ten profile/audit checks. Three CPU Colab smokes
and three 50k PPO cells completed on Python 3.13.16. Legacy model tensors and
all checkpoint scores exactly reproduce previous wide controls; six bundles
pass 468 exact frozen predictions. All six owned sessions stopped.

Full state returned 22.59% versus masked 15.81%, legacy 7.96% and holding 30.60%,
but failed both control gates on ETF/drawdown constraints. No confirmation was
run. All three declining stocks still have negative total policy returns despite
positive short P&L. Conservative cumulative usage is 35.687823/150; the study
charged 0.049487 units. This is development evidence only.
[Full results and limits](../results/execution-observations-2026-10-19/README.md).

The execution-observation release has 16 verified GitHub asset hashes/sizes.
Cleanup removed 27 duplicates (291.41 MiB logical), with 314 retained file hashes
unchanged. Final archives and payload source/data/plan contents were checked
before removal. The source export and original ledger remain available.

## Universal direction replay — 2026-10-19

Thirty fixed profile/phase/mode labels completed using 21 unique replays, with
two additional unrestricted calibration paths. No models were trained and no
Colab session allocated. Three-seed final unrestricted/long-only/short-only
means are 13.89%/15.49%/-1.19%, versus holding 30.60%. Short-only earns positive
total and short P&L in six of nine declining-stock/seed episodes, with mean
AMZN +2.12%, DIS +6.49%, NEE +2.87%, but loses over the whole panel.

Seed 37 selected required a separately declared one-share CPU bridge after its
original strict reference check failed on two rows with <$1 equity drift. The
strict failure, reference-only measurement and bounded exception are retained.
All other legacy seed/phase references pass the original strict checks. All
314 source-study and 228 older initialization retained hashes remain unchanged;
project units stay 35.687823/150 with zero active assignments. Tests cover filter
state feedback, profitable/costly shorts, parity and bounded rejection, paired
model/seed consistency and pre-filter direction diagnostics.
[Results and limits](../results/direction-replay-2026-10-19/README.md).

Direction-replay publication verified all 19 GitHub asset hashes/sizes. Cleanup
removed 18 duplicate publication copies (9.85 MiB logical) with 597 retained
hashes unchanged. All full replay/calibration paths and original models/data
remain available. Final account status has zero active assignments.

## PPO exposure architecture — completed 2026-10-20

237 offline tests and three CPU Colab 512-step smokes pass on Python 3.13.16.
The subsequent three-arm 50k seed-11 screen fails: native +22.59%, unmasked
branches +5.41%, market-only exposure -3.64% final, versus holding +30.60%.
Treatment also loses on all three declining stocks; no confirmation is run.
Six bundles pass 468 exact frozen predictions and architecture/mask checks;
native selected/final tensors and every checkpoint score exactly reproduce the
previous full-state control. All six owned sessions stopped. Study conservative
usage 0.060740; cumulative 35.748563/150. No retired test or broker orders.
[Results](../results/exposure-architecture-2026-10-19/README.md).

Architecture publication verified all 18 GitHub asset hashes/sizes. Cleanup
removed 27 duplicates (300.33 MiB logical), preserving all 326 retained file
hashes. Final archives and payload source/data/plan contents were checked before
removal. Source export, original data and cumulative ledger remain intact.

## 2026-10-20 reward objective screen

Two512-step and two50k Colab CPU cells completed with default Python3.13.16,
immutable source/ledger,312 exact frozen predictions and all4 shutdowns verified.
The candidate selected23.38% but final20.43%, below22.59% control and30.60%
holding; predefined gate fails, no confirmation. Default control tensors and all
checkpoint scores match the prior run. All13 release assets verified;15 duplicate
files removed (118.91MiB),272 retained hashes unchanged. No test evaluation or
orders. Project35.795378/150; account350.66,zero active/rate.
[Evidence](../results/reward-objectives-2026-10-20/README.md).

## 2026-10-20 longer-horizon context screen

All3 CPU512-step smokes and3 matched50k cells completed on actual defaultPython
3.13.16. Full selected14.26%/final13.89%, masked13.93%, baseline0.83%, holding
30.60%. Full fails both gates, particularly ETF performance; declining-stock
joint positive total/short success0/3. No confirmation or default-feature change.
All468 frozen prefix predictions match; all6 owned sessions stopped. All14
release assets verified,21 duplicates removed (187.35MiB),316 retained hashes
unchanged. Source/data/ledger retained; no retired test or orders. Project
35.856639/150,account350.62,rate0.00,active0.
[Evidence](../results/trend-context-2026-10-20/README.md).

## 2026-10-20 saved-path participation audit

Five source runs, ten selected/final phases, matched26-symbol development
holding panels; costs and aggregate side/equity totals reconcile. Source report
checksums remain unchanged, diagnostic code hashes recorded. Twelve relevant
tests pass, including intraday short versus flat-close distinction, cost identity,
pretrade resize scale, and bad date/horizon/accounting rejection. No weights,
datasets, retired-test reports or Colab allocations opened; no new credits used.
[Report and charts](../results/participation-diagnostic-2026-10-20/README.md).

## 2026-10-20 supervised signal data preparation

A checksummed training-only label payload contains45,266 fit examples with labels
completed by2017 and18,616 probe examples completed during2018–2020. Fit-only
preprocessing ignores pooled source statistics and all probe/project-validation
features. Eleven focused tests pass, covering completion, purging, dividends,
splits, episodes, future perturbations, delayed/UTC timestamps and normalization.
Immutable readback and protocol/data identity checks pass. No supervised model
fit, new Colab allocation, policy change or profitability result yet.
[Preflight evidence](../results/supervised-signal-2026-10-20/README.md).

## 2026-10-20 supervised feasibility completed

One Colab CPU runtime (Python3.13.16) completed2 discarded prefix updates and
12000 full-fit updates across6 models, plus ridge/constant controls. Both MLP
gates fail: masked/full mean MSE5.52%/12.56% worse than constant, no passing
seeds. No PPO warm start. All111,696 forecasts restore against immutable probe
inputs; maximum cloud/CPU error5.96e-8, local repeats exact. Metric/gate and
normalization/schema checks pass. Session stopped,12 release assets verified,
11 duplicates removed,233 retained hashes unchanged. No project-validation or
retired-test labels used; no profitability established. Project35.861240/150.
[Evidence](../results/supervised-signal-2026-10-20/README.md).

## 2026-10-20 fixed-forecast centering diagnostic

Protocol/preflight committed before scoring. Complete26-symbol panels on716
sessions satisfy same-session availability before next-entry midnight. Source
weights, normalization, forecasts and targets unchanged. Six centered forecasts
preserve daily ranks but fail the unchanged predictive gate against ridge.
No training, project-validation/test labels, Colab allocation or orders.
[Result](../results/signal-centering-2026-10-20/README.md).

## 2026-10-20 forest dependency/data preflight

Reviewed optional sklearn/scipy wheels install on local Python3.13.16, matching
the latest inspected Colab CPU kernel. Existing pins unchanged; package/core
checks pass. Immutable completed folds and unfitted constructors accepted.
Portable array fixtures test ordering, averaging, restoration and invalid/cyclic
inputs. 264 non-training regression tests plus3 focused checks pass. No forest
fit, new Colab allocation, policy or performance claim.
[Preflight](../results/forest-signal-2026-10-20/README.md).

## Completed forest feasibility (2026-10-20)

Six fixed forests trained in Colab CPU, Python3.13.16; 42 focused checks pass.
All111,696 immutable-probe forecasts pass restoration, exact local repeats,
normalization and schema/source/nonfinite checks. Both arms fail the predictive
gate (16.23%/17.47% worse than ridge, zero passing seeds), so no PPO transfer,
trading promotion or profitable-short claim. Project conservative35.871240/150;
owned shutdown verified, rate0 with zero assignments. See the
[complete evidence](../results/forest-signal-2026-10-20/README.md).

Saved forest error decomposition now verifies both exact MSE identities in
36 predictor/year cells. Six focused tests pass without fitting. Undefined
sign precision/recall is null, and unequal daily panels use row weights.
Prediction source unchanged; no gate change, new Colab usage or promotion.

## Historical ETF regime source audit (2026-10-20)

Fixed2001–2009 four-ETF panel passes all-session coverage and raw/normalized
OHLCV/action agreement. Cached readback is exact; recent2010–2020 subset passes
identical source checks.9,052 earlier bars,23.55% joint trailing declines versus
8.61% recent. No merge or fitting; cross-snapshot price continuity remains
pending. [Evidence](../results/historical-regimes-2026-10-20/README.md).

Historical seam acceptance passes:332 bars, zero OHLCV/dividend differences
against earlier/recent references, matching raw split events and executable
timestamps/factors. Cached readback exact;69 related tests pass. No merge,
training or Colab. [Evidence](../results/historical-regimes-2026-10-20/overlap.json).

## Matched historical augmentation preparation

Three immutable arms pass exact readback and unchanged original episodes,
validation and normalization checks. Full512-transition windows, reproducible
sampling and realized symbol/period counts pass tests and actual Gym/SB3
checkers.16 focused tests pass without fitting; original causal/legacy sampling
regression included. Colab training/job/artifact integration remains pending.
[Prepared identities](../results/historical-augmentation-2026-10-20/preflight.json).

Broader verification passes75 preparation/sampling/core/observation/reward tests
without fitting. Repeated preparation reproduces all three exact data identities.
Downtrend labels are not measured by full-window sampling; any compatibility
counters are explicitly marked unmeasured, not evidence of zero downtrends.

Version15 now binds dataset identities, fixed protocol, CPU jobs and full-window
sampling on the launcher and worker. Returned archives check PPO/init/state/
reward settings, checkpoint curves, complete frozen metadata and reconciled
symbol/history sampling counts. Invalid checksummed contracts fail. Sampling
statistics are copied before parity resets to prevent nested-count mutation.
Actual metered Colab acceptance is next; no fitting result is implied.

53 focused historical/sampling/context/fleet/budget/artifact checks pass before
allocation; three exact immutable smoke task payloads validate locally.

Actual historical Colab smoke acceptance:3x512 PPO transitions,Python3.13.16,
234 exact restored predictions,3 owned shutdowns verified. Project35.888374
after smoke; no performance gate assessed.50k three-arm screen launched from
same source and original2-unit study allowance.

Historical fixed screen reporter implemented before reading completed screen
metrics. Two focused gate tests cover both controls/phases, absolute stock/ETF
and median targets, drawdown, incomplete/nonfinite rejection. Actual50k screen
is still running; no gate verdict has been computed.

## Historical augmentation screen completed

Three50k PPO screens complete, kernel3.13.16; earlier selected/final17.06% fails
against original19.23%/15.01%, recent15.33%/15.33%, holding30.60%. Both stock/ETF
absolute targets fail; joint declining-stock total/short successes1/3. No
confirmation.468 exact smoke/screen predictions,6 owned stops,15 verified
release assets; duplicate cleanup21 files195,506,959 bytes,377 retained hashes
unchanged. Conservative project35.917535/150.
[Full evidence](../results/historical-augmentation-2026-10-20/README.md).

## SAC saved-path diagnosis

Three frozen100k SAC models restore with12 legacy inputs. Six development
reports reconcile costs/equity and identical holding paths without dataset
access; all original hashes unchanged. Final stock/ETF17.69%/9.97%, holding
32.63%/19.47%; close shorts2.53%/0.25% are not short-profit evidence. Ten
focused tests pass. New three-arm SAC state protocol fixed but no allocation.
[Evidence](../results/sac-diagnostic-2026-10-20/README.md).

Broader SAC diagnostic/participation/observation regression:19 tests pass,
including actual Gymnasium/SB3 observation-profile checkers, without fitting.

Version16 integration now verifies fixed protocol/data identity before allocation
and again in the worker. Metadata/frozen bundles carry the protocol hash; archive
checks cover SAC parameters, initialization, observation profile and complete
selected/final checkpoint curves. All three immutable smoke task payloads pass
local validation. Actual Colab acceptance is next, not implied by fixtures.

53 focused SAC/history/observation/fleet/budget/artifact regressions pass before
allocation, including actual Gym/SB3 observation checks without model fitting.

SAC fixed reporter and interactive chart are implemented before screen results;
13 common/SAC gate/archive checks pass. Smoke fleet3x1024 is running from
bb0f3e2; runtime/parity/shutdown acceptance pending, no performance claim.

## Actual SAC state Colab smoke acceptance

All3x1024-transition CPU smokes pass onPython3.13.16,234 frozen predictions
exact with immutable normalization and incompatible-input rejection.3 owned
sessions stopped and cleanup verified. Project35.939018 after smoke. Fixed
3x50k screen launched frombb0f3e2 with original2-unit study allowance.
[Evidence](../results/sac-execution-state-2026-10-20/README.md).

## SAC execution-state screen completed

Three50k SAC screens complete onColab CPU,Python3.13.16. Selected/final
legacy7.44%/7.39%, masked17.84%/17.69%, full7.93%/7.93%, matched holding30.60%.
Full fails both controls and absolute stock/ETF/median targets; joint declining
stock total/short positives1/3. No confirmation.468 exact smoke/screen selected
bundle predictions,6 owned shutdowns,13 verified release assets. Cleanup21 files
238,984,076 bytes,390 retained hashes unchanged; replay summary byte-identical.
Conservative project36.009692/150.
[Evidence](../results/sac-execution-state-2026-10-20/README.md).

## SAC saved replay and target-scale diagnosis

Three own final50k models/replays inspected with zero optimizer steps;4096 fixed
rows per model, timeout-aware soft-target decomposition and unchanged tensor/file
hashes. No dataset or Colab access. Six numeric tests pass. Reward/alpha/value
scales motivate a fixed initial-temperature hypothesis, not a causal failure or
performance claim. [Evidence](../results/sac-learning-diagnostic-2026-10-20/README.md).

Combined replay-target, saved-path and participation regression:16 tests pass,
with no fitting or additional runtime allocation.

Version17 integration now locks the initial-temperature override, dataset and
protocol in task/worker/archive checks. Runner records actual initial alpha,
target entropy and actor/critic/target fingerprints before learning. Valid-case
fixtures and semantic-tampering rejection pass;49 focused temperature/SAC/fleet/
budget/artifact checks pass without fitting. Two immutable smoke payloads
validate. Actual cross-arm initial-weight equality and Colab parity remain
required before screening; fixture fingerprints are not experimental evidence.

Initial-temperature reporter implemented before screen results: exact default
configuration/score/tensor reproduction and cross-arm initial fingerprint checks
precede interpretation. Ten focused tests pass. Two Colab smokes are running;
no runtime/parity acceptance or performance verdict claimed yet.

## Actual temperature Colab smoke acceptance

Two1024-transition CPU smokes pass onPython3.13.16. Initial actor/critic/target
fingerprints match exactly, initial alpha1/0.001 and target entropy−4 verified;
156 exact restored predictions with immutable normalization/input rejection.
Both owned stops verified; project36.022140 after smoke. Fixed2x50k screen
launched from same0f04e6b source and original2-unit allowance.
[Evidence](../results/sac-temperature-2026-10-20/README.md).

## Saved development-report identity/accounting checks

Report acceptance now binds exact episode keys, decision/execution dates and
prices to immutable2021–2023 validation, validates equity chains and early-stop
conditions, recomputes metrics and complete cash/holding paths, and reconciles
reported long/short attribution. A validation label alone cannot admit future
dates or altered benchmarks.14 focused tests pass;12 actual retained selected/
final SAC/history reports pass without fitting. No training-source change.

## Completed initial-temperature comparison

The two 50k SAC seed-11 screens are complete and fail the preregistered gates.
Initial alpha 0.001 improves selected mean return to 19.67% versus default
7.93%, but final falls to 9.84%; the final gain of 1.92pp misses 2pp and
drawdown worsens 4.27pp. Final stocks/ETFs 10.78%/4.72% trail matched holding
32.63%/19.47%; joint falling-stock total/short positives are 1/3 selected,
0/3 final. No extension, confirmation seeds, temperature sweep or promotion.

Fresh default configuration, all checkpoint scores and selected/final network
tensors reproduce the prior full-state SAC exactly. Initial network hashes
match across arms; 312 smoke/screen selected-bundle prefix predictions are exact.
All four owned sessions stopped. Thirteen release assets have verified server
digests; duplicate-only cleanup removes 15 files (167,684,918 bytes), preserving
354 file hashes including models, replay, data, source, ledger and plan.html.
The post-cleanup report reproduces byte for byte. Eighteen focused temperature
and report-contract checks pass. Cumulative project 36.069198/150; account
350.48 with zero assignments/rate after shutdown. Retired test stays isolated.
[Complete evidence](../results/sac-temperature-2026-10-20/README.md).

Next inspect saved selected/final action paths, directional contributions,
exposure and cost differences to distinguish deterioration mechanisms before
preregistering another bounded hypothesis. This diagnostic needs no fitting,
provider calls or Colab allocation. Development results remain repeatedly used;
prospective confirmation and the overall benchmark/short objective remain unmet.

## Saved temperature-path outcome

Candidate selected-to-final stock/ETF net return changes -10.47/-6.25pp split
into long -7.34/-3.45pp and short -3.13/-2.80pp. Cost drag rises only0.91pp
for stocks and falls0.75pp for ETFs. Saved-path non-cost changes -9.56/-7.00pp
show costs alone cannot account for deterioration; exposure rises while both
side contributions worsen. This is descriptive accounting, not a causal
cost-free replay or proof of an optimization mechanism. No fitting/data access/
Colab and source hashes unchanged;11 existing numeric/contract checks pass.
[Evidence](../results/sac-temperature-paths-2026-10-20/README.md).

Before new allocation, audit existing TD3 input/exploration contracts and whether
a fixed full-state delayed-update comparison provides a distinct hypothesis.
Prior weak TD3 results stay visible. No promise of improvement, automatic
allocation, temperature rescue or relaxation of stock/ETF/short/prospective
requirements. Project remains36.069198/150.

## TD3 audit and fixed protocol

Three owned 50k seeds verify legacy12 inputs, Gaussian sigma0.1 and delayed
updates; no training/data access/Colab. Final stocks/ETFs -0.08%/-0.37% remain
weak. Fixed legacy/masked/full TD3 comparison preregistered with unchanged data,
reward, costs, parameters and strict absolute/relative gates;12 protocol/off-policy
checks pass. No allocation before runtime/initial-noise preflight.
[Category guide](td3-execution-state.md),
[audit](../results/td3-diagnostic-2026-10-20/README.md).

TD3 version18 integration passes50 focused job/protocol/archive/gate/fleet/budget
checks; actual immutable smoke payloads validate. Runner fingerprints all four
initial networks/noise, and strict reporter enforces matched selected/final
accounting and gates. No fitting or allocation yet; actual initialization/parity
remains required. [Details](td3-execution-state.md#version-18-preflight).

## Actual TD3 Colab smoke acceptance

Three1024-transition CPU smokes pass on default Python3.13.16 from source76151d1.
Actual masked/full initial actor, actor-target, critic and critic-target hashes
match; Gaussian noise and delay/smoothing match the fixed protocol.234 exact
frozen-prefix predictions, immutable normalization and incompatible-schema
rejection pass. All3 owned sessions stopped. Project36.090878/150; study start
36.069198 and2-unit allowance stay unchanged. Proceed only to the fixed3x50k
seed11 screen; no smoke-return tuning or retired-test reuse.
[Evidence](../results/td3-execution-state-2026-10-20/README.md).

## TD3 state screen outcome

Full TD3 selected/final6.19% fails fixed controls and absolute stock/ETF gates;
all arms achieve0/3 joint profitable-short cases. No more steps/confirmation.
468 exact restored predictions,6 owned stops,13 verified release assets and
post-cleanup byte-identical report pass; project36.149107/150.
[Outcome and next fit-versus-generalization diagnostic](td3-execution-state.md#completed-td3-state-comparison).

## Training-fit diagnostic preregistration

Fixed final PPO/SAC/TD3 policies, same2018–2020 fitted slice, explicit training
labels and no optimizer/selection/Colab. Window/split/accounting tests20 pass.
Preregistered before actual replay; [guide](training-fit-diagnostic.md).

Training-fit replay completed without fitting or Colab: SAC passes fitted-window
comparability but fails development; PPO/TD3 fail fitted flags. All source,
weights, update counts and normalization remain unchanged. Three release assets
verified, complete derived paths retained. Project36.149107/150.
[Evidence and training-diversity audit next](training-fit-diagnostic.md#actual-inference-only-outcome).

## SAC training-diversity preparation

Cached107-stock plus4-ETF data preserves every original episode and all26
validation episodes exactly. New immutable cdb93a33… fixes original preprocessing
without fitting or modifying sources.18 data/slice tests pass. Fixed protocol
retains22/26 stock and4/26 ETF sampling mass, explicit full-window group cycles
and actual symbol coverage. Fresh equal100k arms allow traversal of107 stocks;
50k cannot. No sampler rescue, return-label weighting or relaxed development
gates. Prior PPO expansion failures remain negative evidence. Runtime/sampler
integration and Colab smokes pending, no allocation; project36.149107/150.
[Guide](sac-diversity.md),
[audit](../results/sac-diversity-data-2026-10-20/README.md).

SAC diversity version19 preflight passes82 focused checks and both actual
immutable payload validations. Group cycles, seeded resets, full-window counts,
positive-transition coverage and frozen normalizer/initial-state/archive gates
implemented. No fitting; real smoke parity still required.
[Details](sac-diversity.md#version19-preflight-implemented).

Observed smoke transport failure motivates bounded deferred unstarted-worker
recovery. Sixteen lifecycle/budget/retry checks pass; original deadlines and
one-retry limit preserved. Current frozen training source unchanged.
[Behavior](colab-concurrency.md#delayed-proof-of-an-unstarted-worker).
