# Standard SAC reference-control preparation

A future TRPO performance comparison must include a strong relevant SAC control,
not only PPO. The completed native SAC comparison used21 full execution inputs,
64x64 ReLU actor layers, unrestricted training exposure and unit-magnitude
evaluation exposure. The execution rule and checkpoint-selection differences
must remain explicit in any future matched comparison.

## Separate immutable actor contract

`inference/canonical_sac_policy.py` implements
`canonical-standard-sac-policy-v1`:21 inputs, two64-unit ReLU layers, four mean
outputs, certified tanh and the existing explicit binary32 Box-unscale operation.
No task tag or input padding is introduced. Geometry/schema/checksum/source and
finite binary32 fields are strict. State is immutable and manifest access returns
a copy. Loading/prediction needs only the standard library.

`inference/canonical_sac_export.py` exports only trusted project native SACPolicy/
Actor checkpoints with exact21/4 float32 Box geometry,64x64 ReLU layers, standard
squashed diagonal Gaussian and no gSDE/custom feature extractor. It validates the
stochastic-head geometry too; deterministic reference uses the mean head.
Python/NumPy/Torch RNG and source bytes are preserved; output is create-only.
Native pickle checkpoints are not an untrusted-file interchange format.

The existing task-SAC and PPO/TRPO actor formats, reference arithmetic primitives
and running Colab full-path job are unchanged. All114 numerical source files for
that job still match its original revision.

## Actual read-only checkpoint export

The originally selected10k and final100k native SAC checkpoint pointers from the
completed October31 screen are reused without reselection. Physical checkpoint
identity, counters, finite optimizer moments and lineage are revalidated first.
Both export and reproduce128 fixed synthetic reference predictions exactly on
repeat. Source checkpoints, learned tensors, actor/critic/entropy optimizer states
and original acceptance/pointer files remain unchanged.

| Phase | Reference seconds for128 points | Equal native-batch coordinates /512 | Maximum native-batch difference |
| --- | ---: | ---: | ---: |
| Selected10k |0.386|436|1.1921e-7|
| Final100k |0.381|362|6.5565e-7|

Native/reference bit parity is **not** established. These are different explicit
arithmetic paths; no tolerance was widened and previous native failures/results
remain unchanged. The source checkpoints' old development returns are not
relabelled as reference returns. No reference market replay or performance
selection has occurred. Full frozen preprocessing, cross-machine/action/path
acceptance and a separately fixed comparison protocol are still required.

Four no-fit tests cover architecture/input/schema rejection, immutable state,
stdlib-only loading, duplicate fields, RNG/source preservation, nonfinite weights
and create-only export. Tests forbid native learning/training/Adam steps.
The first actual diagnostic harness attempted a nonexistent policy.optimizer;
it stopped before export and was corrected to SAC's separate actor/critic/entropy
optimizer fingerprints. That failed harness attempt touched no source model and
is recorded in the acceptance receipt.

[Actual receipt](../results/standard-sac-reference-2026-11-08/acceptance.json)
binds the original accepted source and both exported actors/probes. Artifacts
remain under `artifacts/standard-sac-reference-v1`; no staging copies, local
fitting, development scoring, retired-test access or new Colab allocation occurred.
This preparation does not reopen the failed GRU/SAC candidate family or admit
performance training. The overall robust-return objective remains unachieved.

Regression verification:40 focused standard-SAC, existing task-SAC, PPO/TRPO and
reference-math checks pass. Existing numerical calculation source inventory remains
unchanged, so this preparation does not alter the active Colab comparison.
