# Recurrent policy-update guard

Status: first smoke comparison rejected because initial tensor fingerprints differ. Both512-transition runs completed, were retrieved and passed156 exact frozen prediction checks individually; both sessions verified stopped. No100k screen allocated.

[Initialization failure](initialization-failure.json) records actual source, versions, fingerprints, Adam counters and charges. Same seed/settings/package versions do not establish exact tensor equality across these runtimes. Local untrained control/guard constructions agree with each other, but their fingerprint differs from both remote fingerprints. The cause and magnitude of differences have not been established; initial arrays were not retained in this first attempt.

The equality requirement is preserved. Next prepare checksum-pinned common untrained initialization tensors and RNG state per seed, generated without learning or performance selection, and load them before fitting in both runtimes. Preserve this failed protocol/source and original cumulative ledger before revised smokes. No tolerance relaxation, hyperparameter sweep or cash-family rescue.

Both runs used80 actual Adam steps/40 nominal epochs, so the short guard smoke did not activate early stopping, as allowed by the protocol. That is separate from the initialization failure.

Project38.32607251/150; attempt conservatively0.01706648 including setup/idle/retrieval. Last official account349.56, zero rate/assignments. Full original artifacts and failure evidence retained. No retired-test access or broker orders.

Original [protocol v1](protocol-v1.json) and [exact smoke plans](smoke-plans-v1.json) are retained for source84e97b1. The rejected attempt is not relabeled as a passing comparison.
