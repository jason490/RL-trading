# Matched expanded-universe experiment

The declared protocol is [sector_study.json](../configs/experiments/sector_study.json).
It compares four already verified datasets: expanded baseline, supply/demand,
sector rotation, and both families. All use the same 26 tradable instruments,
episode dates, costs, action contract, net-equity reward and train-only fitting.
Current issuer-sector snapshots are metadata; historical rotation is derived
from available sector-ETF returns. The final-test interval starts in 2024.

## Training and selection

Each dataset receives PPO, SAC and TD3 at 50,000 transitions for seeds 11/23/37:
36 research runs in total. Each run selects its checkpoint using equal-weight
mean net validation return. The family (dataset plus algorithm) is selected by
the highest mean score over its three seeds. Exact family ties use dataset name
then algorithm name; seed ties use the lowest seed. These rules are committed
before full training, not chosen from the eventual outcomes.

The highest-validation seed in that family supplies the candidate bundle.
Freeze the selected feature/algorithm family and publish its decision before
auditing the three family seeds once on held-out data. Report test outcomes
without selecting another seed or retuning against them. Multi-seed variation,
cash/buy-and-hold comparisons, costs and drawdowns remain part of the evidence.
Validation winners are exploratory, especially after multiple comparisons;
they are not profitability or live-execution acceptance.

## Runtime and recovery

First complete the lower-credit CPU expanded-input smoke. For each dataset:

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.runner \
  data/datasets/DATASET_HASH --study --cli-python /path/to/CLI_PYTHON
```

`--study` uses A100, runs CPU/CUDA startup smokes, then the nine matched research
jobs sequentially in one owned session. The worker validates the full job plan,
including each seed. The execution operation is bounded to 90 minutes; the
overall owned allocation remains bounded to two hours including setup/retrieval.
Final downloads allow up to three 120-second attempts. The existing cumulative
150-unit ledger, 130-unit working limit and minute metering apply unchanged.
Budget-stop or unreliable metering triggers teardown and never automatic restart.

Checkpoint archives retain all completed jobs alongside the current job.
Selected-checkpoint reports are saved separately from the latest-validation
report, together with per-symbol metrics and the selected transition count.
Frozen bundles still verify checksums, schemas and deterministic save/load
prediction parity. Artifacts are downloaded before verified backend termination.

## GitHub delivery

Code, configuration, protocol, compact result summaries and documentation are
committed and pushed to the existing branch. Large model bundles and derived
performance reports belong in GitHub release assets rather than Git history.
Keep original provider bar datasets, replay contents and private `.runtime/`
authentication/session logs out of publication. The local retrieved archives
retain the full research outputs needed for continuation and audit.

See [verification](verification.md) for actual execution, artifacts, measured/
estimated compute and cleanup status; this protocol alone does not claim that
all runs have completed.
