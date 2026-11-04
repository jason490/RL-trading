# TRPO numerical engineering — 2026-11-04

Local untrained engineering and the fixed Colab reference arithmetic probe pass.
No TRPO learning, market evaluation or economic promotion occurred.

[Local receipt](local-reference.json) binds a saved untrained native checkpoint,
checksummed reference actor and128 synthetic21-field input/output vectors. The
reference is21→64tanh→64tanh→4linear with direct−1…1 clipping. Each affine dot
rounds once to binary32 after exact summation; each hidden tanh is certified.
This is an explicit inference definition, not native Torch bit-parity.

The128 reference predictions took1.4174s (0.01107s each) locally. Native/reference
maximum action difference is5.58794e−9;74/512 coordinates are bit-identical.
These differences are preserved, not rounded away or presented as identical
execution. No trading-path tolerance or gate was changed. Cross-machine and full
simulator parity remain unverified. Timing on untrained synthetic inputs does
not admit a performance allocation or replace actual Colab profiling.

Twenty-six focused reference/export/TRPO/existing-SAC tests pass, including
closed-form hidden activations and clipping, immutable/schema/source binding,
wrong checkpoint-type rejection, RNG-preserving export, isolated stdlib-only
prediction, common PPO/TRPO reference outputs and unchanged SAC acceptance.
Existing accepted task-SAC arithmetic modules were not edited.

Artifacts remain in `artifacts/trpo-reference-engineering-v1`: native initial ZIP,
reference JSON and complete probe vectors. They are intentional engineering
evidence, not trained models. All prior owned sessions remain stopped; project
usage40.213170920/150 unchanged.

Next: an owned default-kernel cross-machine arithmetic probe and actual Colab
smoke/update-counter acceptance, plus bundle/preprocessing/full-path validation
and research-geometry timing. No performance study or prospective winner exists.


## Exact Colab reference arithmetic accepted

[Independent acceptance](cross-machine-acceptance.json) rechecks all128 points/
512 output coordinates against the retained host reference. Every output matches
exactly, with original math/actor/export source hashes unchanged. Actual default
ColabPython3.13.18, Linux CPU; isolated worker imports neither Torch nor NumPy.
Reference execution took3.4792 seconds. No dependencies/interpreters installed,
model fitted, market episode evaluated or normalization changed. The source is
committedddfb4c2; protocol/actor/input/output hashes are retained.

This establishes the fixed untrained synthetic reference contract across the
two machines. It does not establish native Torch bit parity, learned-policy
parity, complete simulation parity or economic performance. Actual Colab learning
smoke, update/optimizer counters, frozen preprocessing/bundle/full-path acceptance
and research-geometry cost profiling remain before performance registration.

Owned session`rl-stocks-cpu-6224d90805b4` is stopped and shutdown verified.
Measured debit0, estimated/conservative charge0.000526086 units; project
40.213697006/150, remaining109.786302994. Fresh account548.04, rate0, assignments0.
The one-time claim remains. No cleanup recovery is required.

[Cleanup proof](probe-cleanup.json) removes one staging input duplicate (94,699 bytes) only
after canonical equality to the retained prepared-input field; all protected
files rehash unchanged. Native initial ZIP, reference actor/vectors, generated
worker source, retrieved result/summary, ledger/claim andplan.html remain.
