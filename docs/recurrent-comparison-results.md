# Recurrent comparison acceptance and reports

## Purpose and current status

This offline command verifies completed research cells, applies the fixed
multi-seed gates and writes a compact acceptance receipt with a standalone
interactive dashboard. It never trains, provisions Colab, resets the ledger,
selects another checkpoint or evaluates retired data. No actual100k comparison
exists yet; implementation tests use labelled synthetic orchestration fixtures
and separately tested physical checkpoint verification.

The [research protocol](recurrent-sac-comparison.md) owns model/data/gate settings.
The [execution review](recurrent-execution-review.md) owns the pending permission
for a corrected engineering profile. This reporting command does not bypass that
permission or admit performance training.

## Input manifest

Provide a JSON object with exactly these fields:

| Field | Required value |
| --- | --- |
| `schema` | `recurrent-comparison-matrix-v1` |
| `protocol_sha256` | The fixed research protocol digest in the comparison guide |
| `stage` | `screen` or `confirmation` |
| `cells` | All three roles for seed11, or all three roles for11/23/37 |

Each cell has exactly `arm`, `seed`, `folder`, `source_hash` and `source_revision`.
Use the registered roles `native_sac`, `recurrent_ppo_reference` and `gru_sac`.
Seeds are integers. Folders resolve relative to the matrix file; absolute paths
also work. Duplicate physical folders and incomplete matrices reject.

Source identity must come from the externally admitted job completion/provenance:
`source_hash` is its64-character package hash, and `source_revision` its40-character
clean Git revision. Do not guess these values, use current HEAD for an older run,
or take a claimed source identity solely from an unverified result folder.
There is no valid real manifest to publish before the actual cells complete.

## Physical acceptance and receipt

The command calls the full-cell verifier for every entry before using its metrics.
It checks immutable data, portable starts, complete artifact inventory, eleven
snapshots, actual counters/moments/settings, strict selected/final pointers,
recorded-action economics, compact curves and CSV consistency. The receipt
retains an inventory digest covering every scientific file in each cell, compact
checkpoint curves and selected/final stock/ETF/median/short-side summaries.
It does not duplicate full transaction paths already retained in the cell folders.

Seed11 always verifies first. A failed screen produces a failed receipt and closes
the fixed family. Confirmation requires the original passing screen receipt,
whose checksum and protocol/data/isolation flags must agree. Seed11's complete
artifact inventory must remain identical; it is reverified, not retrained or
reselected. If the receipt or seed11 artifacts changed, confirmation aborts before
reading seeds23/37. Only then does the fixed three-seed reducer run.

The receipt includes source-cell provenance, manifest digest and its own checksum.
Neither a screen pass nor a confirmation pass sets `objective_achieved`: independent
prospective evidence remains required. Receipt checksums provide file-integrity
checks; the original admitted-source records remain necessary provenance.

## Commands and publication

After real seed11 cells have completed and been retrieved:

```bash
MPLCONFIGDIR="$PWD/.runtime/matplotlib" PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/accept_recurrent_comparison.py \
  --matrix artifacts/recurrent-screen-matrix.json \
  --dataset data/datasets/cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b/dataset.json \
  --starts .runtime/recurrent-comparison-starts-v1 \
  --output artifacts/recurrent-screen-acceptance-v1
```

For confirmation, supply a nine-cell matrix with `stage=confirmation` and add
`--screen-receipt artifacts/recurrent-screen-acceptance-v1/acceptance.json`.
Do not repeat or overwrite the original seed11 runs.

Output must be fresh and outside all datasets, starts and input-cell directories.
The command stages `acceptance.json` and `comparison.html` together and publishes
only after verification and rendering succeed. A rendering failure removes the
temporary directory and leaves input files intact. Existing outputs reject;
use a new named output to retain a separate review record.

## Dashboard interpretation

The standalone Plotly report shows selected and final stock/ETF mean returns by
seed beside matched holding, every model's10k-step development curve, and final
individual-stock excess returns averaged across the included seeds. Solid/dashed/
dotted curves distinguish seeds11/23/37. Returns are cumulative episode returns;
stock excess is measured in percentage points. These averages are neither an
executed diversified portfolio nor a model ensemble.

The gate result and reused2021–2023 development interval are explicit in the title.
Individual transaction paths and protective-order overlays remain in each cell's
selected/final reports. Declining-stock labels and short P&L stay reporting-only
and cannot affect the fixed gate or checkpoint selection.
