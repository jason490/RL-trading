# Numerical execution provenance

## Purpose and limits

Package versions, seeds and initial tensors describe part of a training run.
They do not establish identical numerical execution on different machines.
PyTorch2.8 documents that reproducibility and bitwise floating-point equality
are not guaranteed across platforms, and mathematically equivalent operations
can round differently. See its official [reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html)
and [numerical accuracy](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)
guides. This is a general limitation, not proof of the cause of any saved run's
discrepancy. A failed preregistered gate stays failed; backend documentation does
not authorize widening tolerances or restarting a stopped study.

## Recorded fields

`core/execution_provenance.py` reads the actual selected model device immediately
before initial run metadata is written. `metadata.execution_backend` records:

- Torch version, machine architecture, CPU capability, visible logical CPU count
  and process affinity count.
- Intra-op/inter-op Torch threads and the allowlisted OMP/MKL/OpenBLAS/NumExpr
  thread environment. Only numeric counts are retained; arbitrary values are
  replaced with `non-numeric`. No general environment dump is collected.
- Default dtype, float32 matmul precision, deterministic-algorithm and warn-only
  settings, MKLDNN enablement/determinism, CUDA matmul TF32 and cuDNN flags.
- GPU name only when the selected model device is CUDA. CPU capture does not
  query GPU names.

The reader never sets threads, backend flags, precision or seeds. It allocates
no Colab session and performs no learning. Values describe capture time; they
do not prove settings stayed unchanged throughout a run. Selected device and
version checks elsewhere remain in force. New metadata accompanies the existing
source/dataset/configuration identities and is included in saved/frozen bundles.

Historical runs without this field remain readable. Do not populate their
missing backend fields from the current local machine or an unrelated runtime.
Matching captured fields still does not establish mathematical or bitwise
reproducibility; use actual saved-state, replay and preregistered result checks.

## CrossQ integrity investigation

[Expanded read-only diagnosis](../results/crossq-integrity-2026-10-27/diagnosis-v2.json)
compares actual serialized actor/critic/entropy Adam parameter groups. Beta
tuples versus lists and integer versus float zero weight decay are represented
separately from effective numerical values. All three saved groups have equal
effective values; no optimizer step was used in this audit. Initial tensor/RNG
identity and replay divergence evidence remain as originally recorded.

The original Colab backend capability/thread/precision record is unavailable,
so this investigation cannot establish backend equality or attribute the first
four-ULP action drift to a particular kernel. The CrossQ persistent integrity
stop remains enforced. No training retry, return tolerance change, retired-test
use or budget reset follows from this provenance addition.

## Verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python -m pytest -q \
  tests/test_execution_provenance.py tests/test_replay_divergence.py
```

Tests capture the installed backend without changing RNG or settings, forbid
thread/determinism setters and CPU GPU-name queries, check safe environment
filtering, distinguish numerical settings from containers, reject changed beta
values, and verify exact-prefix/four-ULP replay diagnosis. No fitting or remote
allocation is involved.


## Scoped historical-source audit

[Immutable-commit comparison](../results/crossq-integrity-2026-10-27/source-comparison.json) finds65 byte-identical common files
in the selected core/environment/features/risk/evaluation/SAC/dataset/checkpoint
scope. The dependency lock is unchanged. Only registry distribution/factory and
training runner change among common scoped files; added optional modules are
listed separately. Factory `create`, callback `_on_step`, seed calls and the
single environment-construction call are structurally identical. Both sources
explicitly call `torch.set_num_threads(1)`.

This narrows the investigation: different configured intra-op counts are not
established by missing runtime metadata. Actual inter-op settings, processor
capability and numerical kernel execution remain unrecorded historically.
Runner/registry additions, initialization/provenance paths and installed backend
execution are not proven semantically identical by a scoped byte/AST audit.
Root cause stays unproven; no tolerance change or training retry is authorized.
The audit imports/trains neither source, changes no retained model/data artifact
and allocates no runtime.
