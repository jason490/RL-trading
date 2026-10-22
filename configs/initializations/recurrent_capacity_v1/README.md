# Recurrent-capacity untrained starts

These are once-generated initialization arrays and CPU RNG, not trained models.
Do not regenerate in place. The fixed protocol binds the manifest hash.

Control uses the pinned21-observation/LSTM64/head64 seed11/23/37 artifacts from
`recurrent_guard_v2`. Larger uses21-observation/LSTM128/head128 arrays generated
without data or learning. After larger weight generation, the corresponding
control CPU RNG is restored, then snapshot weights/RNG are checksummed.

Smoke11 and screen11 reference the same full-architecture snapshot for each arm;
only rollout/batch/episode/transition budgets differ. This deliberately avoids
testing a smaller proxy network. Manifest smoke labels describe the run phase,
not a different hidden width. Shapes differ between arms; exact cross-width
weight equality is neither possible nor claimed. Within each arm/seed/phase,
all tensor/schema/RNG hashes must match, and optimizer state must be empty before
loading. Saved run metadata retains the loaded proof and input snapshot.

Generation performed no learn/optimizer step, data/provider access or performance
selection. This is start provenance, not exact resume or deterministic training
across all hardware. See `docs/recurrent-capacity.md` for fixed gates and budget.
