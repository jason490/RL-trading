# Fixed untrained initialization inputs

These four small NPZ files contain newly constructed policy tensors and CPU Torch
RNG state, generated without datasets, predictions, learning or optimizer steps.
They were not chosen using returns. Smoke uses seed11/32-unit networks; screen and
conditional confirmation use seeds11/23/37/64-unit networks. `manifest.json`
records each file SHA-256, tensor fingerprint, tensor schema and RNG identity.
Experiment protocol version2 pins the manifest checksum.

This realizes the original requirement of identical initial parameters across
runtime hosts after version1's seeded constructions produced different hashes.
No equality tolerance or performance gate changes. Both fresh arms receive the
same input for their seed/size. Framework/backend training arithmetic can still
differ; this is not a claim of bitwise identical learning across hardware.

The NPZ format is loaded with `allow_pickle=False`. Loading rejects incompatible
Torch versions, tensor keys/shapes/dtypes, invalid values/RNG, trained models,
nonempty optimizers and repeated study initialization. Parameters are copied in
place so the optimizer retains ownership. The source artifact is copied into
each run and published with model evidence. Initial RNG refers to the start of
learning; exact continued-learning RNG/environment restoration is not promised.

Generator: `scripts/build_guard_initializations.py`. It refuses an existing
output directory. Do not regenerate these committed files in place or use a
new seed/initialization based on results. A deliberate compatibility migration
needs new artifacts, checksums and preregistration before training.
