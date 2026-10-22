# Session-context shared untrained initialization

Generated once without fitting, datasets, provider calls or performance selection.
The manifest pins tensors, CPU RNG, shape/dtype, Python-independent NPZ checksum
and Torch2.8.0 compatibility. Do not regenerate these files in place.

Masked/full share the same25-observation artifact per seed/size. Legacy reuses
the original pinned21-observation untrained starts in `recurrent_guard_v2`,
without reusing trained weights or reopening that failed family. Legacy capacity
is different; only masked/full are exactly parameter and tensor matched.

Smoke11 usesLSTM32/head32,512 transitions later in Colab only. Screen11/23/37
usesLSTM64/head64,100k fixed transitions later if gates permit. Creating these
files performed no learning or optimizer updates. The protocol binds the
manifest hash before allocation; validation verifies every checksum/shape/RNG
and restores weights in place before fitting, preserving empty Adam ownership.
Actual saved models retain initialization provenance but this is not exact
training-resume or a guarantee of cross-hardware training determinism.
