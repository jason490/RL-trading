# TRPO numerical engineering — 2026-11-04

Local-only untrained engineering passes; no TRPO learning, market evaluation,
Colab allocation or economic promotion occurred.

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
