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

Diversity actual Colab smokes pass:2x1024, Python3.13.16, matching initial
networks/alpha/preprocessing,156 exact restored predictions, sampler counts and
2 verified owned stops. Project36.169866/150.
[Evidence](../results/sac-diversity-2026-10-20/README.md).

## Completed training-diversity comparison

Both100k seed11 Colab CPU screens are complete. Original selected/final mean
returns24.35%/10.38%; expanded24.19%/24.19%; matched holding30.60%.
Expanded final stocks26.06%/ETFs13.88% trail holding32.63%/19.47%; median
stock excess−5.49pp. Selected gain−0.16pp fails the required2pp. All absolute
gates fail and expanded joint profitable-total/short cases are0/3. No extension,
confirmation seeds or promotion under this protocol.

All training symbols receive transitions. Intended ETF probability matches,
but actual ETF transitions differ15.36% versus19.97%; added stocks alone cannot
explain the observed return difference. Both survivor-data and repeatedly-used
development limitations remain. All312 frozen prefix predictions are exact,
all4 owned sessions stopped,14 release asset digests verified. Duplicate cleanup
removes16 files/248,893,520 bytes with643 retained hashes unchanged; the complete
report reproduces byte for byte afterward. Thirty-six focused sampling/job/
archive/gate/report tests pass. Project usage36.263895/150, remaining113.736105;
post-shutdown account350.31, zero rate/assignments. Retired test stays isolated.
[Evidence and interactive artifacts](../results/sac-diversity-2026-10-20/README.md).

Next inspect saved selected/final cost, exposure and directional contributions
before a distinct hypothesis. This requires no fitting/provider/Colab; do not
extend the failed family solely because its late checkpoint improved.

## Regularization interruption recovery — 2026-10-21

Fixed100k screen incomplete:control final21.63%, candidate last retrieved80k
curve5.10%, no candidate final. Strict report remains unassessable/no promotion.
All6 owned attempt sessions stopped after exact-fleet recovery; account349.97,
rate0/assignments0. Conservative ledger37.787213/150 retains monitoring-gap
estimate and stop flag. Actual checkpoint/optimizer restoration and saved-path
accounting pass;78 control plus156 smoke frozen predictions exact.13 server
asset hashes verified;3 archive duplicates removed,656 retained hashes unchanged.
[Evidence and reproduction](../results/sac-regularization-2026-10-20/README.md).

## Interruption checkpoint identity fix — 2026-10-21

New snapshots explicitly bind current and selected weights/reports to steps,
evaluation phases and checksums. A worsening non-selected checkpoint now has its
own saved paths; legacy reports are not misidentified as current. Complete
archive validation checks the manifest when present; earlier archives stay
compatible.33 focused callback/archive/curve/publication checks pass with fake
saves/evaluations and no local training. Actual new-format Colab smoke remains
pending after the reviewed budget stop. Ledger remains37.787213/150; no allocation.
[Checkpoint contract](training.md#checkpoint-identity-during-interruptions).

Discount-horizon Colab acceptance passes2x1024 CPU transitions, actual gamma/
saved-step/optimizer restoration, matching initial networks/alpha/sampling,
156 exact frozen predictions and new checkpoint identity. Both sessions stopped;
project37.805591/150, account349.96 zero rate/assignments, floor150/teardown170.
[Evidence](../results/sac-discount-2026-10-21/README.md). Paired100k screen next;
no performance result or promotion from smoke validation.

## Completed discount screen — gates failed

Fresh100k CPU candidate gamma0.95 selected/final22.79%/22.16% trails gamma0.99
control24.19%/24.19% and matched holding30.60%. Final stocks23.88% and ETFs12.64%
trail holding32.63%/19.47%; median stock excess−9.78pp. Candidate joint short
success1/3 versus control0/3 does not rescue failed return gates. No extension,
gamma sweep, confirmation seeds or promotion. Reused development is not independent
evidence; retired test remains unopened and the overall objective is unmet.

All4 owned sessions stopped;312 exact frozen predictions and35 focused checks
pass.13 server asset digests verified; duplicate publication archives totaling
31,529,571 bytes removed with398 retained file hashes unchanged. Strict report
reproduces byte-for-byte. Project37.897158/150, remaining112.102842; last account
349.88 with zero assignments/rate. Account150/170 and project150/130 protections
remain. Next inspect saved selected/final cost and directional contributions,
without fitting or allocation, before any distinct hypothesis.
[Detailed evidence and release](../results/sac-discount-2026-10-21/README.md).

Saved discount-path diagnosis finds candidate final stock/ETF excess−8.74/−6.82pp
equals non-cost excess−4.73/−3.94 minus cost drag4.01/2.89pp. Whole-panel shorts
contribute−1.42%/−1.30% despite positive shorts on reporting-only decliners.
Both costs and participation matter; no optimizer-cause claim or more-step/gamma
sweep follows.12 accounting/behavior/report checks pass, source hashes unchanged,
no fitting/data access/Colab. Next audit existing recurrent input/memory contracts
before any distinct protocol. [Evidence](../results/sac-discount-paths-2026-10-21/README.md).

## Recurrent input and memory audit

Six retained100k recurrent selected/final models use legacy12 inputs, omitting9
explicit execution fields.96 fixed synthetic sequence observations show exact
direct-call/stream parity and reset replay; resetting each observation changes
actions, so memory is active. One regression passes, weights/updates/source
hashes unchanged; no fitting/dataset/test/Colab access. Missing explicit state
is a distinct testable hypothesis, not proof of the previous performance cause.
Next freeze a matched recurrent masked21/full21 comparison with feed-forward
control and unchanged absolute/relative gates before smoke acceptance.
[Audit](../results/recurrent-contract-2026-10-21/README.md).

## Recurrent full-state protocol prepared

Distinct fixed three-arm comparison: recurrent_full candidate, recurrent_masked
and feedforward_full controls; expanded111 data, unchanged normalization/sampler/
reward/costs, fresh100k seed11 screen after three512 CPU smokes. Same recurrent
initial tensors; feed-forward is not parameter-count matched. Both relative and
absolute selected/final gates required before three-seed confirmation/prospective
evidence. Seven actual-construction/save-load/sequence/reset/gate checks pass
without fitting. No allocation: job/worker/archive/restored-report integration
remains required. Project37.897158 retained; incremental study allowance3 includes
all attempts, account150/170 and project150/130 enforced at future allocation.
[Protocol and status](recurrent-execution-state.md).

Recurrent version22 remote integration complete: fixed plans, actual dataset
preflight, floor-aware controller, saved-model fingerprints, mandatory checkpoint
identity, strict archives and restored-weight matched report.69 focused checks
pass; actual three smoke payloads validate, scripts compile. No allocation or
training yet. Freeze/push source and official usage check precede three512 CPU
smokes; frozen sequence/reset acceptance precedes any100k screen.
[Guide](recurrent-execution-state.md).

Actual three512 Colab CPU smokes accepted on Python3.13.16, clean workerdd3b256.234 exact frozen prefix predictions include carried
hidden/cell state, repeats, reset and caller-state immutability; actual saved
steps/settings/tensor fingerprints, equal recurrent initialization and all-arm
sampling pass. All three owned sessions stopped; project37.927940/150, account
349.86 zero rate/assignments. Next fixed three100k seed11 screen on same worker,
no tuning from smoke returns. [Evidence](../results/recurrent-state-2026-10-21/README.md).

## Completed recurrent screen — gates failed

Fresh100k seed11 selected/final means:feed-forward4.46%/3.34%, recurrent masked
24.99%/24.04%, full-state recurrent candidate3.95%/−3.11%, holding30.60%. Candidate
loses0.51/6.45pp to feed-forward and21.05/27.16pp to masked. Candidate final
stock−2.25% versus holding32.63%, ETF−7.84% versus19.47%, median stock excess
−29.25pp. Candidate selected/final joint profitable-short outcomes0/3; masked
1/3 selected,0/3 final. Relative AND absolute gates fail. No extension, sweep,
confirmation or promotion; masked control cannot be retrospectively promoted.

All6 owned smoke/screen sessions stopped and cleanup verified.468 exact frozen
predictions include recurrent action/hidden/cell/reset parity and immutable
normalization. Actual selected/final records/checkpoint identities, equal initial
recurrent tensors, all-arm sampling counts and matched holding paths pass. Project
38.014408/150, remaining111.985592; study increment0.11725006. Official account
349.80, zero rate/assignments; floor150/teardown170 retained. No retired-test use.
Next saved-path cost/directional diagnosis without fitting before a distinct
hypothesis. Overall multi-seed/prospective objective remains unmet.
[Evidence and release](../results/recurrent-state-2026-10-21/README.md).

Recurrent outcome publication accepted:13 server sizes/digests match; duplicate
publication archives47,914,914 bytes removed,439 retained hashes unchanged. Strict
report reproduces byte for byte after cleanup. All6 sessions stopped; all models,
frozen bundles, current checkpoint identities, source, data and ledger retained.
[Audit](../results/recurrent-state-2026-10-21/cleanup.json).

## Saved recurrent path diagnosis

Three100k runs/six phases audited without fitting/data/test/Colab access. Full
recurrent final stock/ETF gaps34.88/27.31pp include5.81/5.20pp costs; weak non-cost
participation dominates. Masked control gaps6.81/5.19pp include6.11/4.73pp costs,
non-cost excess−0.69/−0.47pp. Small-resize costs0.067/0.081pp cannot close gaps.
Whole-panel shorts contribute+0.48%/−1.52% masked,−3.78%/−6.13% full. Cost addback
is descriptive, not a cost-free replay; no cause/profitability claim.15 checks
include explicit mixed algorithm rejection, accounting and source immutability.
Next audit cost by fill reason before any distinct matched cost intervention;
failed state family stays closed. [Evidence](../results/recurrent-state-paths-2026-10-21/README.md).

## Recurrent fill-cost attribution

Observed masked final stock/ETF same-side resizing costs2.13/1.51pp, entries
1.22/0.97, protective+holding exits1.69/1.26, borrow0.64/0.77. Categories reconcile
all fees/impact/borrow and executed quantities; unknown reasons/corruption reject.
Even arithmetic addback of every resize leaves4.68/3.68pp gaps; adding all entries,
resizes, closes and reversals still leaves~2.99/2.47pp. These are saved-path
bookkeeping bounds, not counterfactual profits. Resize-only savings insufficient;
required protections/20-session exits/borrow remain.18 checks pass, source hashes
unchanged, no fitting/data/Colab. Next separate direction from protective-exit
participation before a distinct protocol. [Audit](../results/recurrent-fill-costs-2026-10-21/README.md).

## Saved action/execution relationship

Masked final stock/ETF requested long magnitude74.06%/73.15% versus end-long
67.04%/67.44%; sign agrees93.93%/94.77% of sessions. Flat-after-protection
2.09%/0.92%, flat-after-holding2.84%/3.65%; rejected rows separate. These observed
relationships do not prove a causal protection bottleneck. Nine checks reconcile
action categories and net P&L with unchanged sources; no fit/data/Colab. Next
consider preregistered fixed unit-magnitude sign-preserving replay, with original
protective outputs and exact unrestricted parity gates before interpretation;
no replay yet. [Evidence](../results/recurrent-actions-2026-10-21/README.md).

## Fixed recurrent sizing replay prepared

Preregister four saved masked100k seed11 cells: selected/final unrestricted versus
unit-magnitude sign-preserving exposure. Other three protective outputs, costs,
holding cap and causal recurrent/account updates stay intact. Both unrestricted
paths must pass exact-quantity/bounded-numeric parity before any candidate replay.
Fixed relative+absolute selected/final gates, no threshold sweep or promotion.
12 filter/state/reference checks pass; no actual replay/fitting/Colab yet. Freeze
clean source before execution. [Protocol](recurrent-sizing.md).

## Completed fixed sizing replay — full gates failed

Both unrestricted phases pass preregistered tolerances with exact quantities;
reports are not byte-identical (maximum equity differences$0.013815/$0.005020).
Four cells complete with unchanged weights/updates/normalization. Unit selected/
final means28.53%/32.80% versus control24.99%/24.04%, holding30.60%. Final stocks
35.40% (+2.77pp excess), ETFs18.55% (−0.92pp), median stock excess+0.41pp meet
final comparability targets. Selected ETF excess−3.36pp and median stock−7.56pp
fail; final mean drawdown worsens3.21pp versus allowed2. Unit joint short success
0/3 in both phases. Preserve the full failure: no threshold sweep, phase reselection,
extra training or promotion under this replay protocol. No new Colab units.

Eight published asset sizes/digests match; verified duplicates removed, retained
paths reproduce all four group/short metrics exactly.12 filter/state/reference
checks passed. Overall goal remains unmet. A separately preregistered training/
execution-alignment hypothesis may examine why deploying unit size reverses
selected/final ranking; it is not confirmation of this failed replay or a proven
cause. [Outcome and audit](../results/recurrent-unit-sizing-2026-10-21/README.md).

## Training/execution sizing alignment prepared

New fixed hypothesis distinguishes adaptive training/evaluation, adaptive training
with unit evaluation, and aligned unit training/evaluation. Same fresh masked21
RecurrentPPO/data/initialization/sampling/100k budget; both controls required.
Adaptive/posthoc final tensors must match. Shared explicit execution contract now
routes training action mapping, actual net-equity reward, validation selection
and frozen predictions consistently; default historical bundles stay adaptive.
43 focused checks including unit Gymnasium/SB3 checkers pass without fitting.
Dedicated job/archive/report/counter integration remains before three512 Colab
smokes. No allocation, project38.014408 retained, study allowance3 with account
150/170 and project150/130. Prior failed replay remains closed; this is a new
learning-contract intervention. [Protocol and status](recurrent-sizing-alignment.md).

Alignment version23 integration complete: actual processed-action counters,
strict training/evaluation modes, saved records, unit action/protection traces,
checkpoint identities and restored matched reporting.79 focused checks and actual
three immutable smoke payloads pass without fitting; scripts compile. Smoke
acceptance requires234 exact frozen action/state predictions plus adaptive/posthoc
final tensor equality. No allocation at integration, project38.014408 retained.
Freeze clean pushed source and official usage check before three512 CPU smokes.
[Guide](recurrent-sizing-alignment.md).

## Actual alignment Colab smoke acceptance

Three512 CPU smokes accepted on Python3.13.16, clean workercaddfcb.234 exact
frozen mapped-action/hidden/cell/repeat/reset predictions, immutable normalization
and input rejection pass. Actual counters:adaptive/posthoc512 transitions each,
340 non-unit raw sizes and0 changes; aligned343 changes,0 processed non-unit sizes,
0 protective changes. All initial tensors/sampling match; adaptive/posthoc final
policy tensors match exactly. Actual selected/final records/checkpoint identities
and unit traces pass. All3 sessions stopped; project38.047855/150, account349.78
zero rate/assignments. Fixed same-source three100k screen next within3-unit study
allowance; no tuning from smoke returns. [Evidence](../results/recurrent-alignment-2026-10-21/README.md).

## Completed alignment screen — candidate gates failed

Fresh100k seed11 selected/final means:adaptive24.99%/24.04%, posthoc unit32.80%/
32.80%, aligned unit20.04%/13.93%, holding30.60%. Adaptive/posthoc final tensors
match exactly. Candidate selected/final loses4.95/10.12pp to adaptive and12.76/
18.88pp to posthoc. Candidate final stocks14.86% versus holding32.63%, ETFs8.77%
versus19.47%, median stock excess−24.77pp. Final mean drawdown−37.41% worsens
7.42pp versus adaptive and4.21pp versus posthoc. Selected joint short outcomes
2/3 deteriorate to0/3 final. Relative/absolute/risk gates fail: no extension,
threshold/architecture sweep, confirmation or promotion of this family.

All6 owned sessions stopped;468 exact frozen mapped-action/state predictions,
actual execution counters/traces, restored models and checkpoint identities pass.
All initial tensors and actual sampling counts match; all111 symbols receive
training transitions. Unit training maps62,293 sizes, processes0 non-unit sizes,
changes0 protection components; all100,000 requests nonzero. These facts establish
execution, not a proven failure cause. Posthoc remains a one-seed development
control, not robust/prospective success; original failed replay is unchanged.

Project38.136658/150, remaining111.863342; study conservative0.12225039 (ledger
observed debits0.08; rounded account decrease0.09 includes meter delay). Account
349.71 zero rate/assignments, account150/170 and project150/130 retained. No retired
test or orders. Overall goal unmet. [Full evidence](../results/recurrent-alignment-2026-10-21/README.md).

Alignment publication/cleanup accepted:13 server sizes/digests match;48,871,879
duplicate bytes removed,456 retained hashes unchanged. Post-cleanup strict report
reproduces byte for byte. All6 sessions stopped and required source/data/model/
counter/curve evidence retained. [Audit](../results/recurrent-alignment-2026-10-21/cleanup.json).

## Alignment saved risk/action audit — 2026-10-22

All six accepted phases audited without fitting/data/Colab. Aligned final stock/ETF
cost drag11.08/8.37pp and noncost excess−6.68/−2.33pp both contribute to failure.
Mean symbol drawdowns−37.89%/−34.78% contain only2.87%/2.63% of peak equity in
window costs; this is descriptive bookkeeping, not a cost-free replay. All saved
policy exposure requests are nonzero; original Gaussian output and issued unit
action are now distinguished. Consider explicit categorical cash as a separate
preregistered hypothesis; no causal fix established, no allocation or promotion.
19 focused tests pass; accepted source hashes unchanged. Project38.136658 retained,
all owned sessions stopped. [Evidence](../results/alignment-risk-actions-2026-10-22/README.md).

## Recurrent cash-choice protocol and policy construction — 2026-10-22

Separate fixed four-arm hypothesis: explicit categorical short/cash/long versus
matched binary short/long, aligned Gaussian unit and strongest posthoc Gaussian
unit control. Candidate must pass relative gates against every control plus
selected/final stock/ETF/median holding and drawdown targets before11/23/37
confirmation and prospective evidence. Same data/reward/memory/sampling,100kseed11;
no rescue of failed alignment family. Hybrid policy/likelihood construction and
local no-fit checks prepared; dedicated plan/worker/archive/frozen integration and
actual four512 Colab smokes remain. No allocation; project38.136658 retained.
[Protocol, implementation and acceptance](recurrent-cash-choice.md).

Cash-choice factory/frozen integration: explicit hybrid algorithm and strict
action-distribution metadata now bind actual policy/cash support to frozen
loading; semantic corruption rejects despite valid bundle checksums. Native
legacy compatibility retained.37 focused no-fit checks pass, including branch,
checkpoint and sequence regressions. Dedicated Colab plans/worker/archive/report
integration and actual smokes remain; no new credits used.

Cash-choice version24 integration complete: fixed four-arm plans/worker/archive,
actual optimizer/parameter-count records and cash execution counters, mandatory
checkpoint identities, restored smoke/screen reporters and all-control gates.
96 focused checks and four actual immutable payload validations pass; scripts
compile. No fitting/allocation. Freeze clean pushed source, official usage check,
then four512 CPU smokes requiring312 exact sequence predictions and verified
cleanup before100k screen. Project38.136658 retained; account150/170 and
project150/130 remain. [Guide](recurrent-cash-choice.md).

## Initial smoke failure and correction

The first four512 CPU attempt failed before acceptance: the binary policy
rejected all-zero recurrent padding when sb3-contrib evaluated minibatches before
masking their losses. This is a compatibility bug, not a failed performance gate.
Only all-zero binary evaluation placeholders now receive a valid dummy direction;
valid actions and strict binary distribution/execution support remain unchanged.
A real recurrent-buffer regression exercises unequal sequence lengths, padded
likelihoods, masks and finite gradients without local fitting or optimizer steps.

All four owned sessions stopped and verified; one failure archive retained with
checksums, others had no salvage snapshot. No accepted model/performance claim.
Project38.14814485, attempt conservatively0.01148649; rounded observed debit0,
official account349.70 zero rate/assignments. Original failed state, ledger, source
and four-unit study baseline retained. Corrected source needs a new export/retry
state and repeated four512 smokes before screen.
[Failure evidence](../results/recurrent-cash-2026-10-22/README.md).

## Corrected Colab smokes accepted

Four512 CPU smokes from clean worker70aa505 on Python3.13.16 accepted:312 exact
frozen action/hidden/cell/repeat/reset predictions, immutable normalization and
input rejection, actual saved selected/final optimizer/configuration restoration,
current checkpoint identities and equal sampling. Initial tensors match within
Gaussian and categorical pairs; cross-family equality is not claimed. Cash arm
requests205 cash/307 directional samples, binary512 directional/zero cash; both
record40 PPO epochs and80 actual Adam steps, have21 optimizer parameter states
and changed weights. The saved `optimizer_updates` field is SB3’s `_n_updates`
epoch counter; it is not an Adam minibatch-step count. Restoring all four final
smoke optimizers confirms every parameter state has step80.
No sizing/protection transforms occur in categorical arms. These are wiring
checks, not performance evidence or settings selection.

All four retry sessions stopped/verified (eight including initial failure).
Project38.18725015/150; study increment0.05059178 including failure, remaining
111.81274985. Account349.67 zero rate/assignments. Source, original ledger and
failure evidence retained. Fixed same-source four100k seed11 screen next within
original4-unit study allowance; no tuning from smoke returns.

[Evidence](../results/recurrent-cash-2026-10-22/README.md).

## Completed fixed screen — cash candidate failed

Strict matched source-bar/accounting/holding validation, actual selected/final
model restoration, mandatory checkpoint identities, within-family initial tensor
equality and identical actual sampling pass. All111 training symbols receive
transitions. Mean net returns on the original26 development episodes:

| Arm | Selected | Final100k |
| --- | ---: | ---: |
| Posthoc Gaussian unit | 32.80% | 32.80% |
| Aligned Gaussian unit | 20.04% | 13.93% |
| Binary categorical | 34.38% | −1.25% |
| Cash categorical candidate | 29.24% | 6.63% |
| Matched holding | 30.60% | 30.60% |

Cash final stocks7.28% versus holding32.63%, ETFs3.03% versus19.47%; median stock
excess−25.65pp. Cash selected stocks32.02%, ETFs13.97%, median stock excess−3.25pp;
selected ETF/median targets fail. Selected/final gains versus strongest posthoc
control are−3.56/−26.18pp. Cash final mean symbol drawdown−31.03% improves2.16pp
over posthoc's−33.20%, but lower drawdown alone cannot pass return gates. Cash
loses selected5.14pp to binary, while final beats binary7.87pp; final ETF gain
versus binary is−0.45pp. All three relative control gates and both absolute
selected/final gates fail.

Cash joint positive total return AND positive net short P&L improves from1/3
selected declining stocks to2/3 final: AMZN+9.49% with+$4,299.85 short P&L,
NEE+20.35% with+$10,635.29. DIS remains−31.09%, short P&L−$6,968.94. These
future-defined cohorts are reporting-only. Partial short success does not offset
weak overall stock/ETF results. Binary joint outcomes fall2/3 selected to0/3 final;
its temporary34.38% checkpoint is not a stable winner. Posthoc remains one-seed
reused-development evidence, not robust/prospective success.

The fixed cash-choice family is closed: no extra steps, cash thresholds, entropy
or architecture grids, confirmation or promotion. No retired-test evaluation,
orders or prospective success. Next inspect saved action/participation and
optimizer diagnostics before any distinct preregistered intervention.

All12 owned sessions stopped/verified, including initial failure.624 exact frozen
predictions across accepted smokes and screens pass. Models, full reports, source,
execution/sampling counters, checkpoints and compact10k curves retained.
Publication/checksummed duplicate cleanup remain.

Project38.30900603/150, remaining111.69099397; conservative study increment
0.17234766 includes failure/setup/idle/retrieval. Ledger observed debits0.14.
Account349.56 zero rate/assignments; original account150/170 and project150/130
limits retained. Original ledger and all failure evidence preserved.

[Full evidence](../results/recurrent-cash-2026-10-22/README.md).

## Publication and cleanup accepted

[Model/report release](https://github.com/jason490/RL-trading/releases/tag/recurrent-cash-2026-10-22) contains15 assets, all server sizes and
SHA-256 digests verified. Eight accepted model bundles retain selected/latest
weights, metadata, checkpoint identities and execution/sampling counters.
`comparison.html` shows interactive10k curves; `learning-curves.csv` retains
percentage-point increments. Failure evidence remains visible.

Cleanup removed only two verified duplicate publication archives,58,832,238 bytes
(about56.1MiB). All249 inventoried retained hashes are unchanged, including
original transports, complete model/frozen/report/checkpoint evidence, source
archives, failure archive, immutable dataset and original ledger. The strict
summary reproduces byte for byte after cleanup, restoring all four screen models
and rechecking source/accounting/gates. All12 owned sessions remain stopped.
Project38.30900603 and last official account349.56 remain; no further compute
allocated. Next audit saved behavior/optimizer evidence before a distinct
preregistered intervention. No cash-family extension or promotion.

[Verification](../results/recurrent-cash-2026-10-22/release-verification.json), [cleanup](../results/recurrent-cash-2026-10-22/cleanup.json), [reproduction](../results/recurrent-cash-2026-10-22/reproduction.json).

## Cash-choice saved action audit — 2026-10-22

All eight accepted phases/208 symbol episodes audited with unchanged source hashes
and no fit/model/data/provider/Colab access. Cash stock long/short/cash requests
shift98.01/0.35/1.64%→68.78/16.30/14.92%; stock short contribution+0.83→−9.21pp.
Binary stock shorts6.06%→31.04%, contribution+3.45→−15.66pp. Cash final stock/ETF
excess−25.35/−16.43pp reconciles noncost−17.83/−8.95 minus cost7.51/7.49pp;
costs alone insufficient. Observed direction drift is not a proven optimizer
cause.28 focused tests pass, including exact registered arms/source identities,
cash requests and legacy compatibility. No family rescue or new training; saved
optimizer diagnosis next. Project38.309006 retained, all sessions stopped.
[Full audit](../results/cash-actions-2026-10-22/README.md).

## Saved cash-study optimizer audit — 2026-10-22

All eight selected/final models restored read-only against retained/publication
hashes. All weights/Adam moments finite; all21 parameter states present, expected
shapes and counts. Final100k has20,000 Adam steps, distinct from1,000 nominal PPO
epochs; selected cash/binary have1,800/7,800 actual steps. Late logged KL/clipping
rise in failed categorical models and strongest posthoc control alike, so no
causal optimizer explanation established.12 helper checks pass; no fit/dataset/
provider/Colab access, sources unchanged. Fitted-period/generalization diagnosis
next before any distinct safeguard protocol. Project38.309006 retained, all
sessions stopped. [Audit](../results/cash-optimizer-2026-10-22/README.md).

Recurrent fitted-period protocol prepared: all four accepted arms × selected/final,
original26 symbols, fixed2018–2020 common window, frozen normalization/memory and
execution rules. Explicit original-panel slicing preserves legacy rejection;
full optimizer fingerprint guards read-only replay. Commit before actual inference,
no fitting/provider/Colab; descriptive flags never promote failed cash family.
[Protocol/method](training-fit-diagnostic.md). Actual replay pending.

## Actual recurrent fitted-period replay

Eight saved selected/final models complete26 episodes/19,136 rows each, with
unchanged tensors, full Adam fingerprints, counters, normalization and source
hashes; all matched holding paths equal. No fitting/provider/Colab/test access.
All eight fitted stock/ETF/median flags fail. Fitted holding58.15% stocks,47.15%
ETFs. Posthoc final stocks27.22%, ETFs68.43%; cash final19.92%/27.07%. Cash
fitted mean improves3.22→21.02% while development29.24→6.63%; binary fitted
mean7.79→15.28% while development34.38→−1.25%. This is consistent with a
generalization tradeoff, not pure deterioration on every period; fitted benchmark
weakness remains and no cause/promotion proven.32 focused checks passed before
committed protocol6f2e10c and actual replay. All sessions stopped, project38.309006
retained. [Evidence](../results/recurrent-fit-2026-10-22/README.md).

Recurrent fitted-period charts published: [release](https://github.com/jason490/RL-trading/releases/tag/recurrent-fit-2026-10-22), six server sizes/digests verified,6,722,998 duplicate bytes removed and30 retained hashes unchanged. Original complete paths retained, no model duplication or new Colab charge. [Evidence](../results/recurrent-fit-2026-10-22/README.md).

## Separate Gaussian policy-update guard protocol — 2026-10-22

Fixed fresh control target_kl=None versus candidate0.01, same strongest Gaussian
posthoc training/evaluation contract, data/reward/memory/sampling and100kseed11.
No categorical/cash rescue. Same selected/final relative/absolute/risk gates;
actual candidate Adam-step reduction required for activation. No grids/extension.
Construction records distinguish nominal epochs from actual Adam steps, and a
forward-only high-KL library test forbids optimizer.step and proves no learning.
Dedicated Colab integration and two512 smokes remain; no allocation.
[Protocol and constraints](recurrent-update-guard.md).

Update-guard construction acceptance:27 focused checks pass, including identical
initial tensors, exact sole-setting difference, nominal/actual update rejection,
activation/nonactivation and a real library high-KL stop before any optimizer
step. No fitting or allocation; dedicated Colab integration remains.

Update-guard version25 integration complete: exact target/floor plans and worker,
actual Adam metadata and strict archives, restored smoke/screen reporting with
performance AND activation gates.75 focused checks and two actual immutable smoke
payload validations pass; scripts compile. No allocation at this milestone.
Commit/freeze source and official usage check before two512 CPU smokes requiring
156 exact frozen predictions and verified cleanup. Project38.309006 retained.
[Guide](recurrent-update-guard.md).

## Initial guard smoke comparison rejected

Both512 CPU runs completed/retrieved,156 exact frozen predictions pass individually
and both owned sessions stopped. Combined acceptance rejects different initial
tensor hashes despite matching seed/source/settings/version records. Local
untrained pair hashes agree but differ from both remote hashes; cause/magnitude
not proven, initial arrays unavailable in this attempt. Keep exact equality gate.
Next checksum-pin shared untrained tensor/RNG snapshots per seed, with explicit
revised protocol before repeat smokes; no learning/performance initialization
selection or tolerance relaxation. No100k screen allocation. Both had80 Adam
steps/40 nominal epochs; short-smoke nonactivation was allowed and is separate.
Project38.32607251, original study baseline38.309006 retained, no budget stop.
[Failure evidence](../results/recurrent-guard-2026-10-22/README.md).

## Update-guard shared initialization version2

Fixed untrained tensor/CPU-RNG artifacts enforce exact paired initialization
after version1 hash mismatch; original failed source/plans/charges retained. No
learning/data/performance selection, no tolerance/gate relaxation. Manifest and
per-file/schema/RNG checksums bind controller, worker, metadata, saved model and
retained run snapshot.94 focused checks and two revised actual immutable payloads
pass; source compilation passes. No allocation during fix. Commit/freeze new
source and repeat two512 CPU smokes using new state; original2-unit study baseline
38.309006 and project/account safeguards retained. Project38.326073 currently.
[Method](recurrent-update-guard.md).

## Version2 actual smokes accepted

Two512 CPU runs from clean workered9d834, Python3.13.16 accepted after durable
retrieval and verified shutdown.156 exact frozen action/hidden/cell/repeat/reset
predictions, immutable normalization/input rejection, restored selected/final
settings/Adam evidence and current checkpoint identities pass. Both load the
same pinned tensors and CPU RNG, with exact initial fingerprint equality and
matching actual sampling. Input snapshots are retained in both runs.

Both smokes record40 nominal epochs and80 actual Adam steps. The short guard
does not activate early stopping, explicitly allowed; actual reduction is still
required at the fixed100k screen. This is wiring acceptance, not performance
evidence or a threshold-selection exercise. Original version1 rejection remains.

All four owned smoke sessions stopped. Project38.34607251/150, remaining
111.65392749; study conservative increment0.03706648 includes rejected attempt.
Official account349.54, zero assignments/rate. Original study baseline38.309006
and2-unit allowance, project150/130 and account150/170 remain. Same-source two100k
seed11 screen next; no tuning from smoke returns.

[Evidence](../results/recurrent-guard-2026-10-22/README.md).


## Completed update-guard screen

Fixed seed11 control selected/final19.62/18.48%, guard28.07/24.44%, holding30.60%.
Guard reduces actual Adam steps20,000→6,165 and passes relative/risk gates, but
both absolute target gates fail. Final stocks27.78% versus32.63% holding;
ETFs6.11% versus19.47%; joint short/total success0/3. No confirmation, extension,
KL sweep or promotion. All six study sessions stopped,156 screen frozen checks
pass, original ledger retained. Project38.39201565/150; official account349.51,
zero rate/assignments. Publication/duplicate cleanup and saved-path diagnosis next.
[Full evidence](../results/recurrent-guard-2026-10-22/README.md).


Update-guard publication accepted:15 server assets match SHA-256/size, four
accepted bundles and interactive curves retained publicly. Only two verified
duplicate archives removed (32,215,676 bytes);167 retained hashes unchanged.
Strict summary reproduces byte for byte after cleanup. All six sessions stopped;
project38.39201565 unchanged. Saved-path diagnosis next, no guard promotion.
[Evidence](../results/recurrent-guard-2026-10-22/README.md).


## Saved update-guard action audit

Four accepted phases/104 episodes audited against exact accepted hashes/panel;
no fit/model/data/provider/Colab access and source hashes unchanged. Guard final
stock/ETF long requests99.51/98.91% versus control93.60/95.77%; cost drag falls
6.07→2.85pp stocks,5.13→2.53pp ETFs. Relative gains combine lower costs and
residual path improvements. Final ETF shortfall13.35pp still includes10.83pp
residual plus2.53pp costs; selected-to-final deterioration occurs despite lower
costs. No causal explanation established.15 focused tests pass. Guard stays
closed; next fixed exit/gap participation audit before another intervention.
Project38.39201565 unchanged, all sessions stopped.
[Evidence](../results/guard-actions-2026-10-22/README.md).


## Saved exit/inherited-position diagnosis

Four phases/104 episodes partitioned with reconciled fill/quantity/equity/holding
chains; no fit/data/Colab. Guard final ETF excess−13.35pp includes flat/no-exit
−5.13pp and inherited-short/no-exit−4.74pp, versus protection−0.03pp. Daily
observed partitions are not isolated gap P&L or causal effects. Reports lack
all opens; next preregister exact bar-backed overnight/intraday cash-flow
reconciliation before another intervention.26 focused tests pass. Guard closed;
project38.39201565 unchanged, all sessions stopped.
[Evidence](../results/guard-exits-2026-10-22/README.md).


## Actual bar-backed cash-flow bridge

Committed method96dddd9 precedes four accepted phases/104 episodes. Every
policy/holding session reconciles (worst error1.28e-10 dollars); source hashes
unchanged, no fitting/new prediction/provider/Colab/test access. Guard final
ETF excess−13.3541pp comprises overnight−7.6351, inherited intraday−1.2040,
fill-to-close−1.5044, dividends−0.4841 and fees/impact/borrow−2.5265pp.
Overnight differences are substantial, not proof of a holding-rule cause. Next
fixed partition by inherited side and preceding actual exit reason before any
intervention. Project38.39201565 unchanged, all sessions stopped.
[Evidence](../results/guard-bridge-2026-10-22/README.md).


## Actual overnight partition complete

Fixed methoda19f350, four phases/104 episodes;41 focused checks and exact
bar-backed bridge pass, sources unchanged. Guard final ETF overnight excess
−7.6351pp splits flat-after-holding−3.1580, inherited short−4.3602,
flat-after-protection−0.6304 and long+0.5135pp. Guard selected same holding
rule gives flat-after-holding−0.1277pp; no fixed performance ceiling or causal
rule-change benefit established. Both short timing and missed exposure matter.
Diagnostic sequence complete; choose a distinct preregistered learning
hypothesis before new allocation, preserving short/holding/absolute gates.
Guard closed, no promotion. Project38.39201565 unchanged, all sessions stopped.
[Evidence](../results/guard-overnight-2026-10-22/README.md).


## Distinct session-decomposition information hypothesis

Optional completed overnight/intraday log returns and EMA20 components implemented
with none/masked/full modes; defaults unchanged. Synthetic witness proves baseline
seven features omit this open/close decomposition, not predictive value.53 focused
checks pass (formulas, causality, splits, masks, training-only normalization, gaps).
No real-data preflight/training or allocation yet. Next bind expanded111/original26
datasets, exact shared controls/initialization and numeric gates before Colab
smokes. Guard family remains closed; project38.39201565 unchanged.
[Method and pending requirements](session-context.md).


## Actual session-context data preflight

Retained development-only raw source verifies; default generated identity
unchanged. New masked/full datasets preserve all bars/ATR/original features and
original first7 normalization exactly. Each has298,946 training and19,058
development rows,111 training/26 development symbols. Added-column normalization
fits training rows only; future/test data unopened.53 focused checks pass.
No training/allocation; next fixed three-arm protocol and shared untrained
initialization, then Colab integration/checkers/smokes. Project38.39201565 unchanged.
[Evidence](../results/session-context-2026-10-22/README.md).


Session-context numeric protocol and pinned untrained starts prepared: actual
legacy/masked/full dataset IDs, fixed100kseed11 screen, unchanged selected/final
relative+absolute/risk gates, conditional11/23/37 confirmation,2-unit incremental
allowance from38.39201564833781.31 focused protocol/data/shared-start checks pass,
including all fixed seed/size constructions and actual save/load without learning.
Dedicated Colab integration/checkers/smokes remain; no allocation.
[Protocol and initialization](session-context.md).


Session-context version26 Colab integration implemented;76 focused checks pass.
Actual three immutable smoke tasks validate; real Gymnasium/SB3 checkers and
untrained recurrent forwards pass at21/25/25 fields. Shared model evidence and
study-specific artifact checks are separated by ownership. Source compilation
passes. Actual remote smokes remain before screen; no allocation/usage yet.
[Details](session-context.md), [current results index](../results/README.md).


## Actual session-context smokes accepted

Three512 Colab CPU runs from cleanb1cfa00/defaultPython3.13.16 completed and
retrieved; all owned sessions verified stopped.234 exact frozen recurrent
predictions, normalization/rejection checks, actual selected/final optimizer
restoration and checkpoint identities pass. Each has80 Adam steps/40 nominal
epochs. Masked/full initial tensors/RNG/settings match exactly; actual sampling
agrees across all arms. One CLI reply timeout recovered from completed export,
without training restart. Fixed same-source three100kseed11 screen next; no
tuning from smoke returns. Project38.42515390/150, study increment0.03313825;
official account349.48 zero rate/assignments. Original2-unit study baseline
38.39201564833781 and all project/account protections retained.
[Evidence](../results/session-context-2026-10-22/README.md).


## Completed session-context screen — gates failed

Legacy selected/final19.62/18.48%, masked27.66/10.76%, full26.42/16.08%,
holding30.60%. Full fails both relative control gates and both absolute phases;
final stock/ETF16.56/13.45% versus holding32.63/19.47%. Joint short/total
reporting-only cohort2/3 selected→1/3 final. No feature/window/step/reward sweep,
confirmation or promotion; fixed family closed.

Strict immutable source/accounting/holding and actual optimizer restoration pass;
full/masked exact initialization and sampling match. Final20,000 Adam steps,
61,513 legacy/63,561 augmented trainable parameters.234 screen frozen predictions,
all six owned sessions stopped; legacy tensors/sampling reproduce prior control.
Project38.51487741/150, study increment0.12286176; official account349.41 zero
rate/assignments. Source/ledger/data retained. Publication/verified duplicate
cleanup next. [Full evidence](../results/session-context-2026-10-22/README.md).


Session-context publication complete:17 server assets verified, six accepted
model bundles and interactive reports public. Only two duplicate publication
archives removed (49,579,965 bytes);178 retained hashes unchanged. Strict summary
reproduces byte for byte, all three screen models/accounting/gates rechecked.
All six sessions stopped; project38.51487741 unchanged. Fixed family closed;
next distinct preregistered learning hypothesis, no extension/feature grid.
[Evidence](../results/session-context-2026-10-22/README.md).


## Distinct fixed recurrent-capacity hypothesis

Prior recent recurrent protocols keep64-unit widths; no direct64/128 ablation
found. Weak fitted-period benchmarks motivate a capacity hypothesis but do not
prove underfitting. Freeze control64 versus candidate128 recurrent/dense widths,
61,513/221,321 parameters, unchanged baseline seven features/data/reward/actions/
sampling and100kseed11 budget. No session-feature rescue or width/depth grid.
Original selected/final relative/absolute/risk and conditional11/23/37/prospective
gates retained. Fixed2-unit allowance from38.514877408036206.

Pinned untrained per-arm starts/common CPU RNG prepared; both smokes use full
architectures.65 focused construction/regression checks pass, including actual
save/load and invalid/runtime-deadline rejection. Conservative smoke-duration
projection must pass before long CPU allocation. Dedicated integration/checkers/
Colab smokes remain; no allocation or new usage.
[Protocol and constraints](recurrent-capacity.md).


Capacity version27 integration prepared:96 focused tests pass, actual full64/128
environment checkers/untrained forwards and two immutable smoke tasks validate.
Required runtime profile is bound to actual accepted smoke bytes and recomputed
before controller allocation; missing/stale/over-deadline profiles reject.
Worker/runner/retrieval/model/optimizer evidence and confirmation/report routing
are integrated using existing lifecycle/budget code. Source compilation passes.
Actual two512 CPU smokes still required before100k; no allocation/usage yet.
[Evidence](../results/recurrent-capacity-2026-10-22/README.md).


## Actual capacity smokes accepted and runtime projection passes

Two512 full64/128 CPU runs from39f94a0/defaultPython3.13.16 completed/retrieved
and verified stopped.156 frozen recurrent predictions, immutable normalization
and rejection checks, restored models/optimizers/checkpoints, same sampling and
common CPU RNG pass. Full61,513/221,321-parameter architectures each perform
80 Adam steps/40 nominal epochs. Durations140.96/136.94s are mixed train/evaluate,
not a reliable speed ranking. Fixed projection3016s/50.3min passes7200s; exact
acceptance bytes bind approved plans. Same-source two100kseed11 screen next.
Project38.53588481, study increment0.02100741; official account349.40 zero rate/
assignments. Original baseline38.514877408036206/2-unit allowance retained.
[Evidence](../results/recurrent-capacity-2026-10-22/README.md).


## Completed capacity screen — 2026-10-23

Both fixed seed11 runs completed100,000 transitions in Colab CPU, defaultPython
3.13.16, from frozen clean source39f94a001044d9a0f24e4f2b1384014495f835e1.

| Arm | Parameters | Selected return | Final return |
| --- | ---: | ---: | ---: |
| Control64 |61,513|19.6245%|18.4844%|
| Larger128 |221,321|28.1508%|13.4278%|
| Matched holding |—|30.6019%|30.6019%|

Separate26-symbol mean net total returns on repeatedly reused2021–2023
 development history; neither annualized nor a combined portfolio or independent
evidence. Larger selected at60k versus control80k. Selected improvement8.5263pp,
final change−5.0565pp; final stock gain−8.9396pp and ETF gain+16.3002pp versus
control. Mean drawdown worsens0.5327pp. Larger selected stock excess−3.2133pp
misses−2pp; final stock/ETF excess−19.7987/−2.7383pp and median stock
excess−13.4832pp fail. Fixed capacity family closes: no width/depth/step grid,
extension, confirmation or promotion. No robust accepted winner established.

Strict source/dataset/accounting/holding checks pass. Both selected/final actual
weights, settings and Adam states restore; each final has20,000 actual Adam
steps/1000 nominal epochs. Same actual111-symbol training sampling and common
initial CPU RNG verified, with different pinned per-width tensors as declared.
156 screen frozen action/hidden/cell/repeat/reset predictions pass;312 including
smokes. Normalization immutable, schema/feed/nonfinite rejection passes. Retired
2024+ data unopened; no fitting by local reports or broker orders.

All four owned smoke/screen sessions stopped and cleanup verified. Official
account349.34, zero rate/assignments. Cumulative conservative project38.60166524,
remaining111.39833476; study increment0.08678784 includes setup/idle/downloads/
shutdown. Observed account debit0.07 versus conservative charge0.08678784.
Project150/130 and account150/170 safeguards unchanged; original ledger retained.


[Evidence](../results/recurrent-capacity-2026-10-22/README.md).


Capacity publication verified:15 GitHub assets match server sizes/SHA-256,
four accepted model bundles and interactive reports retained. Only two duplicate
publication archives removed (45,880,597 bytes),142 retained hashes unchanged.
Original transports/source/data/models/checkpoints/ledger/plan.html preserved.
All four sessions stopped; project38.60166524 unchanged. Fixed family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/recurrent-capacity-2026-10-22).


## Capacity saved-path diagnosis — 2026-10-23

Four accepted phases/104 episodes audited without data/model fitting/Colab.
Larger stock selected→final29.41→12.83%, short contribution−0.89→−7.41pp,
cost drag3.78→5.14pp and noncost excess+0.57→−14.66pp. ETF21.21→16.73%,
cost3.01→5.13pp; neither causal explanation nor a short-ban justification.
21 focused audit/action checks pass; source hashes unchanged. Capacity family
closed. Next audit training-history coverage/sampling before preregistering a
distinct matched full-history/recent-window study; no allocation or window
selection from returns. Project38.60166524 unchanged.
[Evidence](../results/capacity-actions-2026-10-23/README.md).

## Measured training-year instrumentation — 2026-10-23

48 focused sampling/history/capacity artifact/report checks pass. Actual
Gymnasium/SB3 environment checkers pass (existing unbounded-Box warnings).
Cross-year decision/execution counts, partial-window/mid-episode resets,
immutable snapshots, RNG invariance, corrupt margins/future years and independent
symbol-total rejection tested. Existing callback/final metadata paths persist
new counters; no training/provider requests/Colab allocation. Historical artifacts
without counters remain accepted and do not prove year visitation. Project
38.60166524 unchanged. See [year-log contract](training-history.md).

Training-history version28 integration:111 focused tests pass; actual two immutable
payloads/environment Gymnasium+SB3 checkers/untrained64-unit recurrent forward
and equal initial tensor/RNG checks pass. Missing/corrupt measured-year evidence,
wrong dataset/model/optimizer/checkpoint, stale/failed runtime profile and invalid
confirmation reject. Older historical/capacity/guard/account-floor regressions
pass; compile passes. No local fitting or Colab allocation; actual two512 smokes
remain required. Project38.60166524 unchanged.
[Evidence](../results/training-history-2026-10-23/README.md).

Actual training-history smokes accepted: two512 Colab CPU runs/defaultPython
3.13.16,156 exact frozen recurrent checks and actual selected/final optimizer/
model/checkpoint restoration. Equal pinned initial tensors/RNG and symbol counts;
measured year margins pass. Both owned sessions stopped. Runtime projection
3504.0053s passes two-hour limit; exact accepted bytes bind prepared100k plans.
Project38.62221390/150, study increment0.02054866, account349.32 zero rate and
assignments. [Evidence](../results/training-history-2026-10-23/README.md).


## Completed history-composition screen — 2026-10-23

Status: fixed seed11 comparison failed; family closed. No confirmation,
extension, start-year/window grid, promotion or performance goal achieved.
Both fresh100k runs completed in Google Colab CPU/defaultPython3.13.16 from
frozen clean2dfa1fbd45d6a3c120fb11311c9d285781556b57. Earlier sections retain
preparation/smoke history; this is the latest outcome.

| Arm | Selected return | Selected step | Final return |
| --- | ---: | ---: | ---: |
| Full2010–2020 |19.6245%|80,000|18.4844%|
| Recent2016–2020 |20.9523%|50,000|5.5976%|
| Matched holding |30.6019%|—|30.6019%|

Mean net total returns across separate26-symbol episodes on repeatedly reused
2021–2023 development history; neither annualized nor a combined portfolio or
independent test. Candidate selected improvement1.3278pp misses the2pp gate;
final change−12.8868pp fails. Candidate final stock/ETF4.4602/11.8533% versus
holding32.6265/19.4660%. Its final stock change−17.3073pp versus control fails
the nonworsening group gate; ETF improves11.4258pp. Mean drawdown worsening
0.0127pp is within risk tolerance but cannot compensate for failed return gates.
Both phases fail stock/ETF/median−2pp absolute targets. Selected stock/ETF excess
−10.3746/−5.6617pp, median−8.5500pp; final−28.1664/−7.6127pp, median−15.6457pp.
Reporting-only declining-stock joint positive total AND net short success1/3
selected→0/3 final; no robust profitable-short capability established.

## Actual training years and verification

Measured2020 execution share rises as intended: full stocks1.978%/ETFs3.005%,
recent stocks6.637%/ETFs8.263%. These realized finite samples differ from exact
complete-window expectations. Per-symbol/global/decision/execution/pair margins
reconcile; actual symbol draw counts match. Recent history changed composition
without providing stable development improvement. It does not prove all recency
methods fail, or establish a sole cause of deterioration. This fixed family is
closed; a future study needs a distinct preregistered hypothesis, not a year grid.

Full control reproduces the prior capacity control's exact selected/final tensor
hashes and symbol sampling, supporting instrumentation invariance. It was trained
fresh, not resumed or selected from old weights. Both arms restore identical
pinned untrained tensors/RNG. Strict source/data/accounting/holding checks and
actual selected/final models/Adam records restore. Final20,000 actual Adam steps,
1000 nominal epochs,61,513 parameters.156 screen frozen recurrent action/state/
repeat/reset checks pass;312 including smokes. Normalization remains immutable;
schema/feed/nonfinite rejection passes. Retired2024+ unopened; no local fitting,
new provider call or orders in reporting.

## Compute, artifacts and next work

All four smoke/screen sessions verified stopped. Official account349.27, zero
rate/assignments. Original project ledger38.68329334/150, remaining111.31670666;
study conservative increment0.08162810 versus observed account debit0.07,
including setup/idle/transfers/shutdown. Original baseline38.60166524433091/shared
2-unit allowance and project150/130/account150/170 safeguards retained.

Strict summary, compact10k learning curves with incremental changes, measured
year CSV, restored models, checkpoints, complete reports/source/datasets/ledger
retained. Interactive comparison is `artifacts/training-history/summary.html`.
Verified GitHub publication and duplicate-only cleanup follow this outcome commit.
Next diagnose accepted saved paths before choosing a distinct experiment; no
automatic training restart or retired-test evaluation.

[Evidence](../results/training-history-2026-10-23/README.md).


Training-history publication verified:17 GitHub asset sizes/digests match, four
frozen bundles and interactive reports public. Only two duplicate publication
archives removed (35,931,249 bytes),148 retained hashes unchanged. Original
transports/models/source/both datasets/ledger/plan.html retained. All four sessions
stopped, project38.68329334 unchanged. Fixed family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/training-history-2026-10-23).


## Saved history-study actions — 2026-10-23

Four accepted phases/104 episodes audited without dataset/model fitting/provider/
Colab access. Recent stock return22.25→4.46%, short contribution−3.58→−10.54pp,
closing short exposure11.82→19.56%; cost drag declines7.89→7.26pp. Observed
accounting, not a causal explanation or short-ban justification.22 focused
audit/action tests pass; source hashes unchanged. Fixed history family closed.
Next preregister read-only saved-policy replay on a common eligible2018–2020
training window before new training; not yet executed. No prospective success
or assumption that every bar was sampled. Project38.68329334 unchanged.
[Evidence](../results/history-actions-2026-10-23/README.md).


## Actual saved-policy replay completed

All four original26-symbol replays complete:19,136 execution rows per model,
76,544 total. Both immutable datasets produce identical common eligible
2018-01-02–2020-12-02 execution bars/features/ATR and original normalization,
anchored2017-12-29. All four descriptive training stock/ETF/median−2pp flags fail.

| Saved model | Training-window stocks | Training-window ETFs | Development mean |
| --- | ---: | ---: | ---: |
| Full selected |40.53%|50.22%|19.62%|
| Full final |33.32%|36.13%|18.48%|
| Recent selected |32.36%|28.54%|20.95%|
| Recent final |46.66%|31.18%|5.60%|
| Matched training-window holding |58.15%|47.15%|—|

Separate-symbol mean net total returns, not annualized or a combined portfolio.
Training and development have different dates/regimes/durations; raw cross-period
returns cannot identify a causal generalization effect. Recent training-window
mean improves31.77→44.28% while development20.95→5.60% declines, consistent with
a generalization tradeoff but not proof of its sole cause. Full training mean
42.02→33.76% and development19.62→18.48% both decline. Final recent median stock
excess−4.88pp still misses−2pp; all fitted stock means underperform holding.

This does not establish an accepted winner, justify more width/steps, reopen
history/capacity families or select another checkpoint. Actual visitation year
counts cannot prove every replay row was sampled. All phases are preexisting
saved weights and flags are reporting-only. No fitting, optimizer step,
normalization change, provider request, Colab allocation or retired-test access.
Complete weights/Adam moments and source hashes remain unchanged. Original
project ledger38.68329334/150 and last official account349.27 zero usage remain;
all prior owned sessions stopped.


[Evidence](../results/history-fit-2026-10-23/README.md).


History-model fitted-period charts published:7 server assets verified,104 equity
plots on four pages, no model duplication or raw bars. Duplicate chart archive
removed (3,373,270 bytes),19 retained hashes unchanged; full paths retained. No
additional Colab units; project38.68329334 unchanged.
[Release](https://github.com/jason490/RL-trading/releases/tag/history-fit-2026-10-23).


Saved fitted/development action comparison:8 accepted paths/208 episodes, source/
model/report hashes verified with no fitting/dataset/provider/Colab access. Final
recent stock closing shorts19.36% training versus19.56% development, but short
contribution−4.07 versus−10.54pp. Training cost drag8.95pp exceeds development
7.26pp despite better return; aggregate frequency/cost does not establish cause.
Explicit fixed training-window reporting scope added; default development-only
checks preserved.29 focused regressions pass. Next saved optimizer/parameter
audit before a distinct learning hypothesis; closed families remain closed.
[Evidence](../results/history-fit-actions-2026-10-23/README.md).


History saved optimizer/parameter audit: four accepted models finite/exact; final
20,000 actual Adam updates. Recent selected/final norms29.00→29.89, displacement
5.88 over10,200 updates; full29.28→29.71, displacement3.74 over4,200. Intervals
differ, and KL growth in both arms does not prove cause or a useful remedy.
37 focused checks pass, no fitting/dataset/provider/Colab. Direct recurrent PPO
L2 study not found in configs; any fixed experiment must be distinct, preregistered
and gated, with no coefficient or closed-family grid. Project38.68329334 unchanged.
[Evidence](../results/history-optimizer-2026-10-23/README.md).

Recurrent L2 version29 integration/construction acceptance:138 focused checks
pass, including coefficient/epsilon/coupling/type rejection, actual saved snapshot/
checkpoint/year contracts and prior capacity/history/guard/account-floor paths.
Both actual datasets/checkers/untrained full64 forwards and two immutable tasks
validate. Source compiles, no local fitting/Colab allocation. Actual two512 CPU
smokes and bound runtime approval remain before100k. Project38.68329334 unchanged.
[Evidence](../results/recurrent-regularization-2026-10-23/README.md).

Actual recurrent L2 smokes accepted: two512 CPU/defaultPython3.13.16 runs,
156 exact frozen recurrent checks, correct saved coupled decay/epsilon/moments,
initialization/sampling/year/checkpoint proof. Both owned sessions stopped.
Projection3426.6427s passes7200s; actual acceptance bytes bind100k plans. Project
38.70365471, study increment0.02036137, account349.26 zero rate/assignments.
[Evidence](../results/recurrent-regularization-2026-10-23/README.md).


## Completed coupled-L2 screen — 2026-10-23

Status: fixed recurrent PPO L2 comparison failed; family closed. No coefficient,
optimizer/model/reward/step grid, extension, confirmation or promotion. Earlier
sections retain preparation/smoke history. Both100k seed11 CPU runs completed in
Google Colab/defaultPython3.13.16 from clean frozen
1d640eebf0ad604461d0d510f584477efee75844.

| Arm | Selected return | Selected step | Final return |
| --- | ---: | ---: | ---: |
| Control decay0 |19.6245%|80,000|18.4844%|
| Coupled L2 decay0.0001 |30.4487%|50,000|18.4155%|
| Matched holding |30.6019%|—|30.6019%|

Separate26-symbol mean net total returns on repeatedly reused2021–2023
development history; neither annualized, combined portfolio nor independent test.
Selected relative gain+10.8241pp passes2pp, final−0.0689pp fails. Final stock
change−2.4975pp fails nonworsening; ETF+13.2886pp improves. Mean symbol drawdown
improves1.3741pp but cannot compensate for failed return gates.

Candidate selected stocks33.9981% versus holding32.6265%, but ETFs10.9265%
versus19.4660%; selected median stock excess−3.1679pp. Thus the near-matched
selected overall mean does not pass separate stock/ETF/median−2pp gates. Final
stocks19.2699%/ETFs13.7160%, excess−13.3566/−5.7500pp and median−6.7186pp,
all fail. Reporting-only declining-stock joint positive total AND net short P&L
1/3 selected→0/3 final. No robust profitable-short capability established.

## Actual optimizer/data/frozen evidence

Both actual selected/final models, moments, checkpoint IDs and default accounting/
holding paths restore and reconcile. Actual coupled Adam defaults and group
flagsFalse, coefficient0/0.0001 and explicit eps1e-5 verified. Final20,000 actual
Adam updates/1000 nominal epochs,61,513 parameters each. Exact same pinned
untrained tensors/CPU RNG, actual symbol sampling and measured year margins pass.
Fresh control reproduces prior full-history control's exact selected/final
tensor hashes and sampling, supporting preserved epsilon/default behavior; model
ZIP bytes need not match across saved provenance. No trained weights reused.

156 exact screen frozen action/hidden/cell/repeat/reset predictions,312 including
smokes; immutable normalization/schema/feed/nonfinite rejection pass. Source/data/
reward/action/model architecture/steps remain fixed. No retired2024+ access,
local fitting, new provider calls or broker orders in report generation.

## Compute, retained artifacts and next work

All four smoke/screen sessions verified stopped; official account349.23, zero
rate/assignments. Original project ledger38.74617914/150, remaining111.25382086;
study conservative increment0.06288579 versus observed rounded account debit0.04,
including setup/idle/transfers/shutdown. Original baseline38.68329334472413/shared
2-unit allowance and project150/130/account150/170 safeguards retained.

Strict summary, compact10k curves with incremental changes, actual optimizer/
initialization/year/checkpoint records, full models/reports/source/dataset/ledger
retained. Interactive comparison: `artifacts/recurrent-l2/summary.html`. Verified
GitHub publication and duplicate-only cleanup follow this outcome commit, then
saved-path diagnosis before any distinct hypothesis. No automatic restart or
relaxed gates. The full research objective remains unachieved.

[Evidence](../results/recurrent-regularization-2026-10-23/README.md).


Recurrent L2 publication complete:16 server assets verified, four model bundles
and interactive reports public. Only two duplicate archives removed (35,744,305
bytes),145 retained hashes unchanged; original source/data/models/checkpoints/
ledger/plan.html preserved. Strict summary reproduces byte for byte from retained
transports with actual models/Adam/coupling restored. All four sessions stopped;
project38.74617914 unchanged. Fixed L2 family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/recurrent-regularization-2026-10-23).


L2 saved-action diagnosis:4 phases/104 episodes, accepted hashes/panel unchanged,
no model fitting/dataset/provider/Colab access. Candidate stock return34.00→19.27%,
closing longs91.36→91.31%, cost drag4.33→3.74pp; aggregate participation/cost
does not explain deterioration.30 focused regressions pass. Next preregister a
read-only2×2 selected/final direction/protection crossover with independent
recurrent states, native-control reproduction and unchanged execution. Not yet
implemented/executed, no hybrid selection/promotion or failed-family rescue.
Project38.74617914 unchanged. [Evidence](../results/l2-actions-2026-10-23/README.md).


L2 crossover native control rejected fixed tolerance before any mixed case.
Bounded diagnostic retry retained exact failure; direct unwrapped replay equals
wrapper result exactly. Both differ from original Colab by at most$0.00467059
daily equity, with no structural/count/quantity/date mismatch. Runtime rounding
is plausible, not established cause. No tolerance widening, mixed-policy returns,
hybrid selection/promotion or family rescue.41 tests pass; source/model/optimizer/
normalizer unchanged, no training/provider/Colab/test access or new credits.
[Evidence](../results/l2-crossover-2026-10-23/README.md).

Feature-scale audit:7 exact-transform/strict-threshold/schema/nonfinite/immutability
checks pass. Actual immutable111/26 source distributions and9 top source-arithmetic
witnesses computed without fitting/model/provider/Colab/test access. Raw source
and normalizer unchanged. Input tails are observed, not proof of cause/benefit;
no clipping or altered trading data applied. Project38.74617914 unchanged.
[Evidence](../results/feature-scale-2026-10-23/README.md).

Input clipping preparation:83 focused tests pass, actual immutable candidate
c895d6af… has identical raw298,946 training/19,058 development rows and means/
scales, bounded transformed market values. Old default generationc2e6c58c… and
sourcecdb93a33… unchanged. Untrained recurrent bundle action/state parity,
schema rejection, frozen normalization and fixed-action accounting pass. Legacy
three-field Normalizer unchanged, explicit optional factory only. No candidate
preprocessing fit or model training/Colab/provider/test access; default-generation
identity verification recomputes training-only legacy statistics. Integration/
actual study smokes remain. [Evidence](../results/input-clipping-2026-10-23/README.md).

- 2026-10-23 input clipping version30 integration:156 focused protocol/artifact/
  report/preprocessing/legacy/account-floor tests pass. Both actual immutable
  datasets pass Gymnasium/SB3 checkers and full64 untrained recurrent forwards;
  exact shared initial tensors/RNG,61,513 parameters. No local fitting. Actual
  Colab512 smokes/runtime acceptance remain before100k. Evidence:
  `results/input-clipping-2026-10-23/construction.json`.
- Additional shared fleet/budget/job/artifact/control regression suite:41 pass
  (197 checks across the two focused invocations). Original `plan.html` preserved.
- Actual input-clipping Colab smokes: both512 CPU/defaultPython3.13.16 complete;
  156 exact frozen sequence/reset predictions, actual saved optimizer/model records,
  unchanged normalization and matched initial tensors/RNG/sampling pass. Both owned
  sessions retrieved and stopped; project38.76724093, account349.22 zero assignments.
  Runtime projection3970.4465s passes7200s; fixed100k plans prepared, not allocated.


### 2026-10-23 — fixed market-input clipping screen failed

Two100k Colab CPU runs completed; control selected/final19.6245/18.4844%,
clip5 selected/final24.7028/18.5720%, matched holding30.6019%. Selectedrelative
+5.0783pp but final+0.0877pp fails2pp; stock nonworsening and all absolute
stock/ETF/median gates fail. Family closed, no cap grid/extension/confirmation.
312 frozen checks including smokes pass; actual models/Adam/initialization/year
sampling restore; fresh control exactly reproduces prior tensors/sampling.
Control CLI disconnect recovered via advancing checkpoints/final export without
restarting training. All four owned sessions retrieved/stopped. Project38.81646887,
account349.17 zero rate/assignments. Full artifacts retained; publication and
saved-path diagnosis next, robust-return objective unachieved.

[Detailed evidence](../results/input-clipping-2026-10-23/README.md).

Clipping publication:16 GitHub server-size/SHA-verified assets, four complete
bundles/reports; only two duplicate archives35,719,386bytes removed,146 retained
hashes unchanged. Strict summary byte-identical after actual model/Adam restoration.
[Receipts](../results/input-clipping-2026-10-23/release-verification.json).


Clipping saved-action audit: four accepted phases/104 episodes,31 focused tests.
Selected→final closing short exposure stock3.47→6.87%/ETF1.15→10.59%; net short
contribution−0.99→−4.04pp/+1.73→−5.98pp, incremental cost drag5.11→5.92pp/
4.90→6.50pp. Observational relationships, not causal attribution or justification
for disabling shorts/reward edits. Exact source hashes preserved; no fitting/data
access/Colab allocation. Family remains closed; distinct hypothesis review before
new training. [Evidence](../results/clipping-actions-2026-10-23/README.md).

- Gaussian action-boundary diagnostic implementation:54 focused tests pass,
  including analytic tails, numerical extremes and actual untrained recurrent
  carry/reset/model/Adam/Torch-RNG invariants. Fixed source/model/data contracts;
  no actual saved-policy diagnostic or training result yet. See[action boundaries](action-boundaries.md).


### Completed action-boundary diagnostic

Three fixed Gaussian policies ×111 training prefixes ×32 observations audited
without fitting/policy-action execution/validation-return scoring. Conditional
any-component mass beyond[-1,1]:untrained78.28%, selected79.65%, final79.38%.
Final marginals exposure36.23%, stop32.44%, near37.67%, width24.06%. Full model/
Adam/RNG/normalizer/source invariants pass;54 focused tests. These are conditional
probabilities, not historical rollout frequencies or a causal return diagnosis.
Next separately preregister a same-network squashed-diagonal-Gaussian comparison;
do not conflate with failed market-input clipping or add gSDE simultaneously.
No allocation or extra units. [Evidence](../results/action-boundaries-2026-10-23/README.md).
- Squashed recurrent policy:70 focused checks pass. Same61,513 parameters/pinned
  initial tensors, correct transformed sample likelihood/Jacobian/entropy fallback,
  actual128-transition synthetic recurrent rollout likelihood parity and frozen
  save/load/state parity. No local learning/optimizer updates or allocation.
  Native/hybrid legacy tests pass; dedicated study wiring/smokes still required.
- Version31 squashed-action integration:125 focused policy/protocol/artifact/
  report/legacy/account-floor checks pass;62 shared-controller/frozen/hybrid
  checks pass separately with overlap. Both immutable-dataset Gymnasium/SB3
  checkers/full64 forwards restore identical initial tensors/RNG. No local
  learning/optimizer updates. Colab512 smokes/runtime approval precede100k.


### Actual squashed-action Colab smokes accepted

Both512 CPU/defaultPython3.13.16 runs from cleanfa63d6c… complete;156 exact frozen
recurrent sequence/reset predictions, immutable normalization, actual saved models/
Adam, distribution flags/Jacobian and identical initial tensors/RNG/symbol/year
sampling pass.80 actual updates/40 nominal epochs,61,513 parameters each. Both
owned sessions retrieved/stopped. Runtime129.7219s control/159.4628s candidate gives
fixed3472.1638s projection including1200s reserve, below7200s. Exact accepted smoke
bytes bind prepared100k plans, not yet allocated. Project38.83857572/account349.16
zero rate/assignments, shared2-unit allowance/original project/account limits
unchanged. No smoke-return selection, promotion or retired-test access.
[Evidence](../results/squashed-actions-2026-10-23/README.md).


### 2026-10-23 — fixed squashed-action screen failed

Two100k Colab CPU runs completed; native control selected/final19.6245/18.4844%,
squashed15.4740/5.1381%, matched holding30.6019%. Relative and absolute stock/ETF/
median gates fail; final stock6.4716%/ETF−2.1964%. Fixed family closed with no
transform/temperature/log_std grid/extension/confirmation.312 frozen checks
including smokes, actual model/Adam/distribution/initialization/year proofs pass;
fresh control exactly reproduces prior tensors/sampling. All four owned sessions
retrieved/stopped. Project38.90354117/account349.10 zero assignments. Large
conditional boundary mass did not demonstrate a causal cure or policy improvement.
Full artifacts retained; publication and saved-path diagnosis next. Robust-return
objective unachieved. [Evidence](../results/squashed-actions-2026-10-23/README.md).

Squashed-action publication:16 GitHub server-size/SHA-verified assets, four complete
bundles/reports; only two duplicate archives35,834,973bytes removed,144 retained
hashes unchanged. Strict summary byte-identical after actual model/Adam restoration.
[Receipts](../results/squashed-actions-2026-10-23/release-verification.json).


Squashed saved-path diagnosis: four accepted phases/104 episodes,32 focused tests.
Selected→final closing short exposure stock4.59→10.14%/ETF4.34→9.26%; net short
contribution−5.38→−11.18pp/−1.24→−5.41pp; long contributions also deteriorate.
Incremental cost drag3.88→5.05pp/3.54→4.56pp is not the whole gap. Observational
relationships, not a cause or justification for disabling shorts/reward edits.
Exact source hashes preserved; no fitting/data access/Colab allocation. Fixed
family closed; distinct hypothesis review before new training.
[Evidence](../results/squashed-actions-diagnosis-2026-10-23/README.md).

- Compact market context:60 focused feature/core checks pass. Actual cached
  source audit validates all298,946 train/19,058 development rows with zero
  unavailable contexts, exact raw/source bars and unchanged source/protocol/
  implementation hashes. No normalization fit, new dataset/model, inference,
  performance scoring, provider requests, Colab allocation or retired-test access.
  See[coverage evidence](../results/market-context-2026-10-23/README.md).
- Market-context immutable payloads: maskeda392ba93…/fullf439b8f4… reconstructed
  exactly after content-addressed readback. Original298,946 train/19,058 development
  rows/bars/ATR/features/first-seven stats exact. Extra fit uses train only; masked
  zero/identity without fitting.48 focused contracts pass, including untrained
  full/masked recurrent frozen carry/reset parity and nonzero-mask rejection.
  156 actual legacy native/squashed frozen predictions pass on current host.
  No policy learning/updates, provider/test access or allocation. See[preflight](../results/market-context-2026-10-23/payload-preflight.json).
- Market-context initialization/final protocol:54 focused checks pass. Three actual
  mapped24-input snapshots for11/23/37 retain original common tensors/input/account
  columns, native new columns and exact original per-seed RNG; full/masked share
  complete tensors. Legacy61,513/augmented63,049 parameters. All original/generator
  hashes verified. No dataset opened by generator, fitting/optimizer updates or
  allocation. Final protocolf9045fe8… binds actual payload/initialization identities
  without changing preregistered settings. [Evidence](../results/market-context-2026-10-23/initialization.json).
- Market-context version32: all three actual immutable environments pass
  Gymnasium/SB3 checkers/full64 untrained forwards and fixed model/Adam/mapped
  initialization validation.53 new contract,156 shared/legacy and updated121
  focused checks pass (overlap). Actual frozen mask rejection now mandatory in
  three-smoke acceptance. No local policy fitting/updates; official349.10 zero
  assignments/project38.90354117. [Construction](../results/market-context-2026-10-23/construction.json).


### Actual market-context Colab smokes accepted — 2026-10-24 UTC

Three512 CPU/defaultPython3.13.16 runs from clean88fb98c… complete.234 exact frozen
sequence/reset predictions and actual masked nonzero-input rejection pass; saved
models/Adam, mapped initial tensors/RNG and actual symbol/year sampling restore.
80 actual updates/40 nominal epochs each, legacy61,513/augmented63,049 parameters.
All three sessions retrieved/stopped. Durations138.0922/138.7362/136.6898s give
fixed3406.0216s projection including1200s reserve, below7200s. Exact accepted bytes
bind prepared three100k seed11 plans, not allocated. Project38.93466769/account
349.08 zero rate/assignments; original2-unit allowance and project/account limits
unchanged. No smoke-return tuning or retired-test access. [Evidence](../results/market-context-2026-10-23/README.md).


### Compact market-context screen failed — 2026-10-24 UTC

Three100k Colab CPU runs complete: legacy selected/final19.6245/18.4844%, masked
18.7314/18.7314%, full7.6978/−3.7731%, holding30.6019%. Full fails both-control
relative/stock/ETF/risk and all absolute/median gates. Final stock−3.4809%/ETF−5.3803%;
drawdown worsens4.7250/4.9084pp versus controls. Family closed, no benchmark/window/
subset/model/reward/step grid or extension/confirmation.468 frozen checks including
smokes, actual models/Adam/mapped tensors/RNG/year sampling restore; legacy control
reproduces prior native tensors/sampling. All six sessions retrieved/stopped.
Project39.01437789/account349.02 zero assignments. Coverage did not imply alpha;
full artifacts retained, publication/diagnosis next, objective unachieved.
[Evidence](../results/market-context-2026-10-23/README.md).

Market-context publication:16 GitHub server-size/SHA-verified assets, six complete
bundles/reports; only two duplicate archives53,422,663bytes removed,204 retained
hashes unchanged. All three immutable dataset hashes reverified. Strict summary
byte-identical after actual model/Adam restoration. [Receipts](../results/market-context-2026-10-23/release-verification.json).


Market-context saved-action audit: six accepted phases/156 episodes,33 focused
checks. Full selected→final short exposure decreases stock15.12→8.55%/ETF4.65→1.53%,
but net short contribution remains negative/slightly worsens; net long contribution
falls19.74→8.53pp/5.69→−1.64pp. Cost drag4.96→5.96pp/3.20→4.63pp is not the whole
gap. Observational accounting, not a sole cause or reason to ban shorts/change
rewards. Source hashes unchanged; no fitting/data/provider access or new units.
Family closed; distinct hypothesis review before new allocation.
[Evidence](../results/market-actions-2026-10-24/README.md).


Completed fixed critic50k screen failed: nativeSAC selected/final7.9281/7.9281%,
untruncatedTQC1.9991/1.9991%, truncatedTQC9.7018/7.6975%, holding30.6019%.
Candidate fails both-control/absolute/risk gates; no confirmation, grid or
extension. NativeSAC reproduces every checkpoint and selected/final symbol
exactly.384 smoke/full frozen comparisons and actual learned optimizer/replay
checks pass; six owned sessions stopped. Conservative project39.154379/150,
study0.140001/2, official account348.89 with zero rate/assignments. Retired-test
isolation and immutable normalization preserved; objective remains unmet. See
results/distributional-critics-2026-10-24/screen.json and compute.json. Publication
and scoped duplicate cleanup remain next; no promoted model.


Distributional-critic publication/cleanup verified:12 GitHub release asset hashes
and sizes match;18 duplicate transports/upload payloads removed (520,262,328
bytes),191 retained files rehashed at cleanup. All models/replay/checkpoints,
source/data, curves and owned recovery records retained. Subsequent README
publication update is explicitly recorded in cleanup evidence. Six owned sessions
stopped, zero assignments; no additional units or training. Fixed critic study
failed and remains closed; next work is bounded saved-policy/execution sensitivity
diagnosis, not extension or promotion. Release:
https://github.com/jason490/RL-trading/releases/tag/distributional-critics-2026-10-24

VIX actual Colab smoke acceptance (2026-10-25): all three fixed512-transition CPU
runs pass shared-start, learned Adam/recurrent restoration,234 exact frozen
sequence predictions, immutable preprocessing/files and development accounting.
All kernels report3.13.16; all owned sessions are stopped. Cumulative conservative
project39.182288 units; official post-cleanup348.88 balance, zero rate/assignments.
Fixed conservative screen projection3471.70s fits7200s. See
[retained evidence](../results/vix-context-2026-10-25/README.md). Fixed100k screens,
historical legacy reproduction and performance gates remain; no VIX win claimed.

VIX fixed100k screen completed and failed: full selected/final24.0870/14.4391%,
legacy19.6245/18.4844%, masked24.3625/9.2004%, holding30.6019%. Historical legacy
reproduces exactly at all checkpoints/per-symbol.468 combined smoke/full frozen
sequence checks pass, actual learned Adam state and report accounting validate;
independent NumPy group error5.56e−17. All six owned sessions stopped. Cumulative
project39.272458; official348.80 balance, zero rate/assignments. Fixed family
closed without confirmation/extension/promotion; no retired-test access.

VIX publication/cleanup verified:14 GitHub server asset hashes/sizes match;
13 exact duplicates removed (419.93MiB),489 retained hashes unchanged. All six
differing checkpoint snapshots preserved. Complete models/source/data/ledger and
reports remain. All six sessions stopped; no cleanup recovery needed. Release:
https://github.com/jason490/RL-trading/releases/tag/vix-context-2026-10-25

Saved recurrent value-target diagnostic completed:52 fixed training-only stochastic
replays/26,624 rows, worker terminal0 in40.44s. Independent discount-matrix error
4.45e−16, SB3 float32 GAE error9.39e−8, metric error0.0; actual model/Adam/data/
ledger unchanged.16 focused tests pass. No new training, development replay,
retired-test access or Colab units. Descriptive target consistency is not unbiased
calibration or causal diagnosis. See results/recurrent-values-2026-10-25/README.md.

Benchmark subtraction audit completed failed:3/8 fixed conditions pass; both ETF
phases worsen at reward and discounted-target levels. Independent benchmark
reward/equity error0.0, covariance/discount/ratio error1.12e−15; source models,
prior traces, data and usage ledger unchanged. Eight focused tests pass; worker
terminal0 in5.22s. No replay/optimization/Colab units/retired data. Fixed subtraction
hypothesis closed; no economic improvement or coefficient search authorized.

GAE implementation acceptance:61 focused original/shared initialization, full64
seed11/23/37 ownership/RNG, lambda parameter, real serialization and actual SB3
buffer-fixture checks pass. Standalone callback reconstructs timeout corrections,
GAE and segmented lambda1 returns before updates, preserving model/state/RNG.
New modules/tests pass Ruff; no `.learn`, optimizer step or Colab allocation.
Runner/fleet/archive/physical learned acceptance remains pending. See
results/gae-traces-2026-10-25/README.md; fixtures are not learned model evidence.

GAE runner integration:78 focused tests pass, including early request rejection,
real buffer/initializer/serialization fixtures, offending-array retention and prior
shared/VIX regressions. Registered runner now attaches shared starts, source NPZ,
model/Adam records and actual pre-update rollout checks. New modules/tests pass
Ruff. No local optimization or Colab allocation; version35 task/budget/archive and
actual learned smoke acceptance remain before study execution.

GAE version35 runtime/archive integration passes113 focused GAE/VIX/fleet/budget
regressions plus5 smoke-gate rejection cases. Exact rollout arrays/file set/math/
counters are recomputed; changed checksums cannot hide corruption. Fixed CPU
plans require actual smoke restoration/arrays/parity/cleanup and runtime projection
before full runs. Dedicated persistent two-unit baseline preserves original ledger
and150/130 project/150/170 account safeguards. No training or session; physical
learned acceptance remains pending. Static fixtures are not Colab results.

GAE physical restoration/smoke acceptance tooling ready:86 focused tests verify
actual serializer/model/Adam records, rollout arrays,78 sequence predictions per
arm, immutable data/files/state, failure RNG restoration and prior VIX behavior.
Fixtures are non-optimizing and not learned smoke evidence. Diagnostic-statistic
roundoff tolerance1e−12/1e−9 leaves actual target/financial gates and hashes exact.
Clean-source account/kernel preflight and actual512-transition CPU smokes remain.
