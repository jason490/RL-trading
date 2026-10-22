# Recurrent policy-update guard

Status: version2 smokes accepted with exact shared initialization; version1 remains rejected. Fixed100k screen is next. First attempt: initial tensor fingerprints differed. Both512-transition runs completed, were retrieved and passed156 exact frozen prediction checks individually; both sessions verified stopped. No100k screen allocated.

[Initialization failure](initialization-failure.json) records actual source, versions, fingerprints, Adam counters and charges. Same seed/settings/package versions do not establish exact tensor equality across these runtimes. Local untrained control/guard constructions agree with each other, but their fingerprint differs from both remote fingerprints. The cause and magnitude of differences have not been established; initial arrays were not retained in this first attempt.

The equality requirement is preserved. Next prepare checksum-pinned common untrained initialization tensors and RNG state per seed, generated without learning or performance selection, and load them before fitting in both runtimes. Preserve this failed protocol/source and original cumulative ledger before revised smokes. No tolerance relaxation, hyperparameter sweep or cash-family rescue.

Both runs used80 actual Adam steps/40 nominal epochs, so the short guard smoke did not activate early stopping, as allowed by the protocol. That is separate from the initialization failure.

Project38.32607251/150; attempt conservatively0.01706648 including setup/idle/retrieval. Last official account349.56, zero rate/assignments. Full original artifacts and failure evidence retained. No retired-test access or broker orders.

Original [protocol v1](protocol-v1.json) and [exact smoke plans](smoke-plans-v1.json) are retained for source84e97b1. The rejected attempt is not relabeled as a passing comparison.

## Version2 actual smokes accepted

Two512 CPU runs from clean workered9d834, Python3.13.16 accepted after durable
retrieval and verified shutdown.156 exact frozen action/hidden/cell/repeat/reset
predictions, immutable normalization/input rejection, restored selected/final
settings/Adam evidence and current checkpoint identities pass. Both load the
same pinned tensors and CPU RNG, with exact initial fingerprint equality and
matching actual sampling. Input snapshots are retained in both runs.

Both smokes record40 nominal epochs and80 actual Adam steps. The short guard
does not activate early stopping, explicitly allowed; actual reduction is still
required at the fixed100k screen. This is wiring acceptance, not performance
evidence or a threshold-selection exercise. Original version1 rejection remains.

All four owned smoke sessions stopped. Project38.34607251/150, remaining
111.65392749; study conservative increment0.03706648 includes rejected attempt.
Official account349.54, zero assignments/rate. Original study baseline38.309006
and2-unit allowance, project150/130 and account150/170 remain. Same-source two100k
seed11 screen next; no tuning from smoke returns.

[Acceptance](smoke-v2-summary.json), [restoration](smoke-v2-restoration.json), [usage](smoke-v2-compute.json).
