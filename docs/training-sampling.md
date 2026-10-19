# Causal training starts and downside coverage

## Purpose and contracts

The preceding PPO initialization study improved overall 50k returns but worsened
the declining-stock reporting cohort. The original training panel contained
2.31% persistent trailing-decline observations; the verified 107-stock expansion
contained 5.15%. This motivates testing sampling and universe size separately.
It does not establish that downtrends can be traded profitably.

The [declared protocol](../configs/experiments/causal_sampling.json) uses the
same seven market features, five portfolio-state values, signed four-component
action contract, costs, whole shares, exposure limits and net-log-equity reward.
All arms use PPO with the previously tested `wide` initialization; every action
component remains learnable, and exposure has no imposed sign bias. No reward
bonus or penalty uses stock identity or future recovery.

## Sampling modes

| Mode | Behavior |
| --- | --- |
| `legacy_uniform` | Existing episode/start selection; default behavior and random draws remain unchanged |
| `ready_uniform` | Uniform selection across all eligible training episode/start pairs |
| `downtrend_balanced` | Choose downtrend/other with probability 0.5 each, then uniformly select an eligible pair in that class |

Both new modes use **the identical eligible pool**: feature-row index at least
252 and below `len(bars)-21`, retaining the existing maximum-holding headroom.
Every training episode must contain eligible starts. Balanced mode additionally
requires both classes; it never silently drops a symbol or invents a class.
The uniform reference is deliberately readiness-matched and differs from the
older legacy-uniform experiment, so a fresh control is required.

A start is in the downtrend class when its currently available close is below
the close 252 observed sessions earlier by more than 20%, and below the close
63 sessions earlier. Only current/past prices enter the label. The implementation
requires verified split-adjusted prices with unit split factors, just like the
training-regime audit. Dividends are excluded from this price-trend label and
included in simulated equity and short liabilities.

The sampler rejects validation/test episodes, training dates after 2020 and
unsupported adjustment contracts. Labels do not enter observations, reward or
checkpoint selection. Full-period negative-validation-return cohorts remain
reporting-only. Research episodes remain 512 sessions (smoke 128), with existing
cutoff/truncation and bootstrapping behavior. Half the **starts** in a class does
not imply half the transitions; long episodes can cross regimes.

## Factorial study and gates

| Arm | Training universe | Start mode |
| --- | --- | --- |
| `small_uniform` | 22 stocks + four broad ETFs | Readiness-matched uniform reference |
| `small_balanced` | Same small panel | Balanced downtrend starts |
| `large_uniform` | 107 stocks + four broad ETFs | Readiness-matched uniform |
| `large_balanced` | Same expanded panel | Balanced downtrend starts |

All arms score the exact original 26-symbol 2021–2023 development panel. Each
universe retains its own frozen training-only normalizer. Raw validation bars,
features and ATR match exactly; new normalizers must not be substituted into
old models. The expanded sample remains retrospective survivor research, with
EA/AVB/EQR explicitly excluded and historical membership/delistings unresolved.

The screen uses seed 11 and exact 50k transitions per arm. Promote at most one
candidate with selected and final gains of at least 2 points, no deterioration
in either final stock/ETF group and no more than 2 points worse mean final
drawdown versus `small_uniform`. Rank passing candidates by final return using
protocol order for ties. Failure stops this family. A passing candidate and
fresh reference receive seeds 23/37: at most eight research cells total.

Confirmation separates relative improvement from absolute comparable/holding
targets, requires paired multi-seed evidence and retains unsuccessful arms.
Selected and final weights stay separate. The retired 2024–September 2026 test
is neither loaded nor used; development gates do not prove prospective returns.

## Evidence and operations

`preflight_sampling.py` verifies dataset/validation identities and fingerprints
all four causal pools before allocation. Tests check prefix causality, training
isolation, adjustment/empty-class rejection, class frequency, identical eligibility,
legacy RNG parity and actual Gymnasium/SB3 checkers. Evaluation/inference continue
to start at the beginning of an episode, with frozen preprocessing.

Training metadata stores pool counts and a `research_variant`. Lifetime counters
record sampled starts and actual transitions in each class. Checker-generated
steps are excluded by resetting counters before learning. Validation snapshots
write `training-sampling.json`; final metadata captures counts before the extra
save/load verification reset. Version-8 archive validation checks mode, exact
transition count, counter consistency, intervals and final post-update curves.

```bash
PYTHONPATH=src .venv/bin/python scripts/preflight_sampling.py \
  --output results/causal-sampling-2026-10-19/preflight.json
PYTHONPATH=src .venv/bin/python scripts/build_sampling_jobs.py \
  --phase smoke --output .runtime/causal-smoke-jobs.json
```

Commit verified code before provisioning. `freeze_research_source.py --output
.runtime/causal-source` exports the commit and shares the **original** ledger,
allocation lock, data, artifacts and Python environment. Its source hash and
shared-path proof prevent a new export from resetting the project allowance.
Run every phase from this source with the existing centrally metered Colab fleet.
Each bootstrap checks the actual default kernel before dependency setup; do not
install an older Python or another CLI.

Use CPU, initially two concurrent sessions, with an incremental **2-unit** study
allowance and the same starting ledger value in every phase. Pass fleet options
`--study-unit-limit 2 --study-start-units START`. The guard projects the next
operation and 360-second cleanup window, including estimated unmetered time.
Study/project budget stops persist and block automatic restart. The shared total
150-unit ceiling, 130-unit teardown and each allocation's two-hour deadline remain
in force. Stop only owned sessions, verify retrieval and teardown, verify published
checksums before removing duplicate archives, and preserve the ledger.

After smoke acceptance use `--phase screen`; only a matching passing screen may
prepare confirmation with `--phase confirm --screen SCREEN.json`. Generate
screen/confirmation JSON and interactive curves with `report_causal_sampling.py`,
passing `--preflight`, `--state` and optionally `--screen`. Reports verify pool,
source/configuration and baseline hashes and show start/transition balance alongside
stock/ETF and short-side diagnostics. No broker orders are part of this work.
