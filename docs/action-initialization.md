# PPO action-initialization experiment

## Question and fixed scope

The existing constrained long baseline earns more than the learned models on
the development panel, while a simple causal EMA long/short rule loses money.
This motivates testing PPO's initial exploration and protective-action means;
it does not prove either causes weak returns.

The [declared protocol](../configs/experiments/action_initialization.json) is a
2×2 factorial screen using seed 11 and 50,000 transitions for each of four arms:

| Arm | Initial protective mean bias | Initial Gaussian log standard deviation |
| --- | ---: | ---: |
| `standard` | Native 0 | Native 0 (std 1) |
| `low_noise` | 0 | -1 (std about 0.368) |
| `wide` | 0.75 | 0 |
| `wide_low_noise` | 0.75 | -1 |

Only initialization changes. Seeded action-head weights, the optimizer, basic
features, training normalization, reward, costs, splits and four-action contract
are unchanged. Exposure mean bias stays zero: there is no long-only constraint
or direction label. Every action component remains trainable, and the Gaussian
distribution can explore both exposure signs. Wider initial means do not freeze
target zones or substitute a fixed rule for learned outputs.

Use the original 22-stock/four-ETF complete Yahoo dataset, train through 2020,
validate on 2021–2023 with the existing 20-session purge. The retired test is not
loaded. The larger newly acquired stock universe remains a separate experiment.

## Gate and bounded budget

Run a 512-transition CPU Colab smoke for each initialization before research.
The research screen uses exact 50k PPO steps, n_steps 1000, batch 50 and unchanged
learning settings. Evaluate every 10k and retain the separate final post-update
endpoint. The standard arm must be freshly rerun with the same source as the
candidates. Compare it against the earlier control as a reproducibility check.

Promote at most one candidate only if selected **and** final equal-weight returns
improve at least 2 percentage points, final stock and ETF groups each improve or
tie, and final mean drawdown worsens by no more than 2 points. Rank eligible arms
by final return, using protocol order for ties. Declining-stock labels are only
reported; they never select checkpoints or promote a candidate.

If the screen fails, stop this family. If it passes, add seeds 23/37 for only the
promoted arm and standard control: at most eight research cells total, 400k
transitions. Confirmation requirements and absolute stock/ETF comparability
targets are fixed in the protocol. Passing a relative improvement gate alone
does not satisfy the research objective or establish independent profitability.

Use CPU with initial concurrency two and the existing aggregate fleet meter.
The study's incremental ceiling is 2 conservative units from its recorded
starting ledger value; check that allowance before every wave and against
minute readings/projected operations. Stop pending work if the allowance is
insufficient. The shared 150-unit project ceiling, 130-unit teardown, reliable
metering and each allocation's two-hour deadline remain in force. Retrieve and
verify artifacts, stop every owned runtime and verify shutdown on every path.

## Operation and provenance

```bash
PYTHONPATH=src .venv/bin/python scripts/build_initialization_jobs.py \
  --phase smoke --output .runtime/initialization-smoke-jobs.json
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.fleet \
  --jobs .runtime/initialization-smoke-jobs.json --concurrency 2 \
  --cli-python /path/to/existing-colab-cli/python \
  --state-output .runtime/initialization-smoke-state.json
```

Use `--phase screen` for the four 50k cells only after smoke succeeds and owned
cleanup is verified. Research requires a clean committed source tree; actual
kernel Python is inspected by the existing bootstrap before dependency setup.
Do not reinstall the CLI or supply an older Python interpreter.

After a complete screen and verified cleanup, generate its gated summary:

```bash
PYTHONPATH=src .venv/bin/python scripts/report_initialization.py \
  --state .runtime/initialization-screen-state.json \
  --output results/action-initialization-2026-10-19/screen.json
```

Confirmation task preparation requires `--phase confirm --candidate ARM
--screen-summary PATH`; it rejects candidates without a matching passing screen
and protocol hash. Do not allocate confirmation when the screen fails.
The report also writes a standalone interactive HTML comparison beside its JSON,
showing validation return, incremental percentage-point change and drawdown.
At duplicate step counts, the final post-update observation supersedes the
pre-update callback point. The compact JSON preserves both group metrics and
the small per-checkpoint curve needed to reproduce this view.

`scripts/report_initialization_confirmation.py --screen SCREEN.json --state
STATE.json --output CONFIRMATION.json` verifies all three paired seeds against
the original screen's exact source/data/configuration and baseline-panel hash.
It reports relative improvement separately from absolute stock/ETF targets.
Comparable/outperforming targets require both selected and final group means,
at least two seeds passing both phases, and stock median excess no worse than
-2 points. The aggregate stock median first averages each stock's excess across
seeds; drawdown must not worsen by more than 2 points against holding. Positive
development targets still require prospective confirmation before a success claim.

If unrelated engineering advances the main branch during this study, use an
export of the original committed source for confirmation. Verify its package
source hash matches the screen and share the **original** `.runtime` ledger and
allocation lock, data and artifacts. Do not create a fresh budget in the export.
Record the source archive/revision and shared-ledger verification in results.

Version-7 remote plans strictly validate arm, CPU, seed, smoke/50k budget,
reward scale, evaluation interval and final milestone. Run metadata and frozen
bundle manifests record the initialization. Archive validation rejects incorrect
initialization, missing curve points and a missing post-update endpoint.
Unit checks preserve native control weights/RNG exactly, verify that only the
declared distribution state changes, retain both exposure signs and trainable
weights, and verify save/load action parity without local training.

Persist phase states and the study's starting ledger under `.runtime`; publish
compact metrics, protocol, learning curves and verified frozen artifacts under
the standard results/release workflow. Quiet progress output must not disable
the fleet's once-per-minute account metering.
