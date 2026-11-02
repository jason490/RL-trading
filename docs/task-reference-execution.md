# Task conditioning with reference performance execution

The original stock/ETF-conditioning hypothesis is incomplete: masked shared and
tagged shared controls trained100k, while the routed candidate has not trained
its full cell. Tagged native replay failed exact numerical acceptance; that
failure remains retained. A separate reference actor and simulator now reproduce
one fixed19,136-decision training replay across local/Colab machines. See
[reference inference](canonical-inference.md) for engineering evidence and limits.
No robust winner or trading promotion has been demonstrated.

## Fixed contract and interpretation

`configs/experiments/task_reference_execution.json`, canonical SHA
`8c8f7bb00b1ba4c38e01d6dbbee401033d2b4b4d895ad9dbab2cbaf0b37e79a8`,
is declared before any new reference-control final results. It retains the
[original scientific protocol](task-conditioning-research.md): data, normalization,
coarse-type evidence, sampler, reward/costs, initialization, seed11, network shapes,
SAC settings and100k transitions per cell. It changes the explicit performance
execution definition to the independently verified reference arithmetic.

Checkpoint selection stays on the original native **post-update training-window**
curves at10k increments, taking the earliest strict maximum. Existing controls
retain their already selected50k snapshots. The future candidate must use the
same native selection rule. Reference inference evaluates the selected and final
snapshots for **every role**; it does not select checkpoints. Native selection
curves and reference performance reports must carry distinct execution labels.
Development never selects a checkpoint, and retired2024+ remains closed.

The prior tagged-shared selected50k engineering result50.26779262694228% on the
training window is already known. It is disclosed in the declaration and has not
changed a gate, selection rule or hyperparameter. The final snapshot, other
controls and routed candidate still require their own verified reference results.
An exact reference run does not accept old native prediction results or imply a
model improvement. This is completion of the original declared comparison under
a new numerical execution contract, not a reward/indicator/width rescue grid.

## Reusing immutable control assets

The declaration pins original archive hashes, source revisions and source
identities for `masked_shared` and `tagged_shared`. Reuse requires independent
physical consistency checks rather than another control training run:

- Every extracted file must match bytes in its hash-pinned original archive;
  original completion, dataset, source and run metadata must agree.
- Initial plus ten saved research checkpoints must have the right role, full22
  observations/four actions,64-wide trunks, finite binary32 weights, seed and
  native hyperparameters. Transition/update counts, complete Adam ownership,
  step counters and finite moments must agree, including99000 final updates.
- Full100k replay must have the original cursor, finite aligned arrays and valid
  role tags. Actual sampling metadata must cover all111 training symbols.
- Each original26-symbol training report must have complete aligned rows. Metrics,
  matched baselines and directional accounting must reconstruct from raw paths;
  ten compact curve points, native selection and final cell must agree.
- Selected/final frozen metadata, checkpoints, original normalization, execution
  rule and historical type evidence must match. No native prediction is performed.

`training/task_reference_assets.py` implements these checks. Guards reject native
SAC prediction, learning, training, Adam steps and backward calls; RNG state is
preserved around trusted checkpoint loading. It handles only task-owned trusted
pickle archives. Original source files and archives are rehashed after checking.
This verifies physical consistency and native selection records, not historical
optimizer execution independently of the trusted source logs, native inference
parity or new model performance.

Run once from clean committed source:

```bash
MPLCONFIGDIR=$PWD/.runtime/matplotlib PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/verify_task_reference_assets.py
```

The fresh output directory is `artifacts/task-reference-assets-v1`; a1200-second
local deadline bounds checking. This is file/state/record verification, not
training. No Colab allocation or provider requests occur. Tests cover all three
untrained research-role save/load states, counters versus missing moments,
nonfinite weights, role/hyperparameter drift, hook cleanup, archive substitution,
extra files and inconsistent economic metrics/baselines/attribution.

## Gates and remaining implementation

All original numerical performance thresholds remain authoritative. Complete
reference-final fitted results must pass the original+2pp-versus-each-control and
stock/ETF/median-stock floors before development may be read. Selected/final
reference development results then face the original gains, group excess and
maximum-drawdown gates. Only a passing seed11 screen permits seeds23/37 and the
unchanged multi-seed confirmation gate. A frozen prospective comparison remains
required. Failure closes the comparison; no extra steps or rescue search.

The original two-unit task-study allowance is shared from baseline
39.92585919127071, including all prior attempts and the engineering work. It is
not reset. Project150/teardown130 and account150/teardown170 remain in force.
CPU is the default; only Colab may perform training. This declaration authorizes
**no allocation**. Before the routed cell can run, the system still must:

1. Complete control physical admission and selected/final reference evaluations.
2. Review unchanged native learning source versus the measured profile and
   explicitly include reference evaluation/verification time in projection.
3. Integrate a candidate worker, durable artifacts, independent acceptance and a
   one-time claim preserving the original study baseline and failures.
4. Retrieve, checksum and stop/verify owned sessions before applying gates.

Declining-stock results remain reporting-only; future decline labels never enter
features, rewards, sampling, selection or promotion. Passing fitted gates alone
would not establish generalization, historical membership completeness or trading
readiness.

### Local verifier correction

The first physical-check attempt reached replay hashing after checking the masked
control's saved checkpoints and native economic curves, then failed because a
tensor-only fingerprint helper did not accept NumPy replay arrays. The failed
local attempt is retained under `artifacts/task-reference-assets-v1/failure.json`;
this is a verifier defect, not a model/performance failure. All protected source
files passed the final unchanged-hash check. No Colab allocation occurred.

Replay hashing now binds each array's shape, dtype and exact contiguous bytes.
The regression test rejects shape, dtype and value changes without optimizer
calls. Fifteen focused tests pass; a fresh output directory is required for the
corrected full control check. Original native acceptance and gates are unchanged.

### Actual control physical verification

The [control asset receipt](../results/task-conditioning-2026-10-31/reference-control-assets.json)
passes both actual saved controls with verifier source acfa618. It checks22
checkpoints,80 original extracted files, two complete100k replay buffers,20 native
training curves and382,720 reported decision rows including economic metrics,
matched baselines and directional attribution. Both original training-selected
checkpoints remain50k. Sampling records show actual111-symbol coverage in each
run. All source archive/file hashes remain unchanged. Original backend records
are absent and remain explicitly absent; none were inferred or backfilled.

This is physical consistency admission only. The original tagged native replay
remains rejected, prior failure records and claims stay retained, and no model
was trained or native prediction executed. No Colab allocation/credit use, new
reference-control performance result, development access or promotion occurred.
Next reference-evaluate selected/final controls under the fixed declaration,
reusing the already accepted tagged-selected engineering point only if identities
match. Candidate runtime projection, worker acceptance and its one-time claim
remain pending; no candidate allocation is authorized yet.

## Selected/final reference control evaluation

`evaluation/task_reference_controls.py` pins the complete physical-admission
receipt SHA b12c1796… and its producer artifact. It reads only original native
training-selected/final frozen snapshots and the fixed verified dataset. It
exports a separate reference bundle for each snapshot and evaluates the fixed
original26 fitted windows under the declared arithmetic. Native checkpoint
selection stays unchanged. Each point records both selection and performance
execution labels, original/reference bundle identities, complete result/holding
hashes, stock/ETF metrics, drawdown and reporting-only short attribution.

The already accepted tagged-selected50k point is reused only when its newly
exported reference bundle matches the accepted actor ID and both retrieved
metadata/gzip hashes validate. Its full paths, not a rounded scalar, supply the
point. Other three snapshots run fresh reference inference. Reuse is disclosed
and does not count as independent confirmation or a new model outcome.

```bash
MPLCONFIGDIR=$PWD/.runtime/matplotlib PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluate_task_reference_controls.py
```

The fresh output directory `artifacts/task-reference-controls-v1` has a1200-second
local deadline. Each new point keeps a standalone interactive HTML report and
lossless canonical JSON gzip. The temporary duplicate pretty-JSON sidecar is
removed only after its contents and the retained gzip match exactly. Original
native reports, models and receipts remain retained. The reused point links to
its existing accepted full paths/report. A failure preserves completed points
and a failure record; no directory is overwritten or rerun automatically.

The complete matrix requires both phases for both controls, selected steps from
physical admission,100k endpoints,26 episodes/19,136 decisions per point, matching
holding paths and explicit reference performance. It cannot authorize candidate
allocation, open development, select another checkpoint or promote a policy.
Tests reject mixed execution, reselection, changed periods/baselines/source
bundles, physical artifact substitution, mismatched reused actors/paths and loss
of report-sidecar data. Native predict/learning/backward remain forbidden.
