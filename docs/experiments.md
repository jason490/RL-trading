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

`--study --study-algorithm ppo` uses A100 for one three-seed research batch
after accepted lower-credit preflight. Repeat for SAC/TD3 on each dataset.
The original unsliced `--study` plan runs startup smokes plus nine research jobs;
profiling on 26 symbols showed that this larger group risks the execution limit.
Completed cells from that initial group are retained and independently verified;
remaining cells use bounded algorithm batches. The worker validates the full job plan,
including each seed. Hardware and environment checks still run for every batch. The execution operation is bounded to 90 minutes; the
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

## Verifying scheduling changes

Full uploaded package hashes remain recorded. Comparison verifies each hash
against the exact clean Git tree in its metadata, then requires byte-identical
learning files (data, features, environment, risk, algorithms, training, inference
and evaluation). Runtime/reporting-only changes may differ after this proof;
unknown hashes, dirty uploads or changed learning code are rejected. Model
weights, dataset identities, costs, seeds and selection criteria are unchanged
by splitting a batch. No final-test outcomes inform this adjustment.

## Committed-selection audit command

After the complete matrix is verified, commit the validation study JSON containing
the selected family and candidate. `scripts/audit_selected.py DECISION_JSON
ARTIFACT_ROOT... --output artifacts/audits/UNIQUE_NAME` requires the decision bytes
to match Git HEAD, all predeclared family seeds and the complete research matrix.
It evaluates those frozen bundles once on test, saves reports/metrics, and checks
that normalization and bundle files are unchanged. It refuses an existing output
directory rather than silently overwriting an earlier audit. No training or
checkpoint selection occurs in this command. Audit results are separate from
validation selection.
