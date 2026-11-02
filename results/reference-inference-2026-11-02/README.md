# Reference actor engineering profile

The [host profile](host-profile.json) measures only deterministic prediction on128
fixed synthetic observations, using the retained tagged shared50k selected actor.
The original native run remains unaccepted; this is a different declared arithmetic
path and does not revise its reports, checkpoint choice or performance gates.

The new reference bundle retains source manifest, checksums, preprocessing and
historical type provenance. Its source native archive and extracted bundle remain
unchanged. The locally generated reference bundle is retained under the path in
the receipt; models are excluded from Git. See [the category guide](../../docs/canonical-inference.md)
for exact synthetic inputs, schema contracts, test scope and remaining integration.

Measured localPython3.13.16 latency is3.06ms per prediction after four warmups.
This does not prove Colab timing, cross-machine equality, entire simulator
reproducibility or trading profitability. No market episode returns were evaluated,
no optimization occurred, no Colab session was allocated and no credits were used.
Retired2024+ remains closed; new development outcomes and promotion remain gated.

## Colab actor cross-machine result

The [acceptance receipt](cross-machine-acceptance.json) verifies640 synthetic
predictions bit for bit against the isolated host reference:128 masked-shared,
256 tagged-shared (both tags) and256 routed (both heads). The owned CPU session
`rl-stocks-cpu-f04787394fc0` used committed source378a217 and Colab's default
Python3.13.18; the host used3.13.16. No Torch/NumPy import, dependency installation,
market episode or optimization occurred. This is reference-to-reference parity,
not native Torch parity or acceptance of the original tagged native results.

Remote actor timings were1.0465s masked/128,1.5940s tagged-shared/256 and1.6077s
routed/256: approximately6–8ms per prediction. The raw report was retrieved and
its checksum is pinned in the receipt. All source weights and original native
failure records remain retained. The new format has not yet proved entire
simulator-path parity across machines.

Owned termination was verified; a separate official idle reading reported
balance548.11, rate0 and zero assignments. Measured debit is0.00 at the CLI's
reported resolution; conservative estimated/charged usage is0.000553813 units,
not a claim of zero consumption. Cumulative project40.121717184/150, remaining
109.878282816. Original ledger/floor150/teardown170 are preserved. No retry or
new training followed the probe. Next declare a numerical execution variant
before saved-market outcomes; development and retired-test gates remain closed.

## Fixed selected50k saved-market engineering replay

The [market receipt](market-replay.json) completes the separately declared protocol
f5fb168a… on source5750a7a. All26 original training episodes/19,136 decisions pass
alignment and completeness checks. Replay took66.58s; total including export and
reporting70.36s. Protected dataset, original frozen metadata and checkpoint hashes
are unchanged. Complete per-session directional attribution reconciles within
3.21e-10 dollars. Full46.04MB JSON paths and10.45MB interactive HTML remain at
`artifacts/canonical-market-replay-v1/replay.json` and `replay.html`; receipt hashes
were independently checked after execution. The report includes equity/holding,
drawdown, fills, learned stops/target zones, exposure and exit-fill diagnostics.

This is the **2018–2020 training window**, not the2021–2023 development comparison:

| Fixed saved actor | Reference replay | Matched holding |
| --- | ---: | ---: |
| Overall mean26 |50.27%|56.45%|
| Stocks22 |54.08%|58.15%|
| ETFs4 |29.29%|47.15%|

Mean maximum drawdown is−35.33%; median stock excess is−11.72pp. It remains
below holding overall and in both groups. The source was already selected50k;
this diagnostic does not select a new checkpoint, improve the fitted weights,
accept the original native replay or promote a trading model.

The reporting-only declining-stock cohort has two symbols. CVX total+0.14%
versus holding−17.07%, with+8.78pp short contribution and−8.64pp long contribution.
XOM total−33.51% versus holding−40.25%, with+10.58pp short contribution outweighed
by−44.09pp long contribution. Only1/2 has both positive total return and positive
short P&L. Decliner labels are not inputs or selection criteria.

Thirty-eight focused arithmetic/actor/bundle/diagnostic checks pass. Actual
execution guards forbid native SAC predict, training and backward; no optimization,
normalization fitting, provider calls, Colab allocation, new development outcome
or retired-test access occurred. No credits were charged by this local replay.
Full simulator cross-machine verification and a new declared performance protocol
are still pending; the broader objective is incomplete.

## Full simulator Colab probe — setup failure

The [failed attempt receipt](market-probe-failure.json) preserves source d849084,
protocol b9bcb419…, local preflight and original one-time claim. The new minimal
worker first passed the **complete local**26-episode/19,136-decision replay:
64.48s, exact canonical result b2c464c4…, no Torch/SB3 imports. The inspected
upload was2.92MB/397 members and contained only training episodes and public
source/frozen metadata; no credentials or native model files.

Owned CPU session `rl-stocks-cpu-3cdf89d21cd1` then failed at
`/usr/bin/python3 -m venv <owned-root>/venv`, exit1, before dependency installation
or remote market replay. The original wrapper did not capture child stdout/stderr,
so the underlying cause is **unknown**. Missing distribution-packaged ensurepip
is a possible explanation, not a verified diagnosis. The CLI returned despite
the remote error; the missing independently required artifact correctly prevented
acceptance. Bounded retrieval exhausted; no remote report was retrieved. The
local CLI execution log is retained and its hash pinned in the failure receipt.

Cleanup ran and owned termination was verified. A separate official idle reading
reported548.11 balance, rate0 and zero assignments. Conservative charge0.000525920
units; measured0.00 at reported CLI resolution. Project40.122243104/150, remaining
109.877756896. The original claim remains consumed, no retry occurred, and full
simulator cross-machine equality remains unproven. No training, new development
outcome, retired-test access or original-native acceptance change occurred.

## Complete Colab/local simulator parity — independently accepted

The [reviewed acceptance](market-cross-machine-acceptance.json) proves exact
canonical equality of **all26 episodes/19,136 decisions**, including every row,
fill, metric, matched baseline and directional attribution. Default Colab
Python3.13.18 used the same five pinned inference dependencies, no Torch/SB3
imports, and the same fixed source actor/window. Colab replay took150.62s versus
64.48s for the minimal local worker. Corrected pip-free same-interpreter setup
therefore worked on the actual runtime. This remains one fixed training-window
checkpoint, not training determinism or generalization evidence.

The CLI `exec` call failed with `Timeout waiting for reply`, while the remote
worker had completed and durably written both required artifacts. The controller
entered cleanup, retrieved the compact report and complete gzip paths, stopped
and verified its owned session `rl-stocks-cpu-f4a2bb8115c6`. Its original failed
summary/reference_verified=false is retained unchanged. Independent post-cleanup
verification checked the report metadata, archive/claim/source identities,
compressed SHA and full decompressed canonical-result SHA against the fixed
local target; all pass. This acceptance records artifact evidence separately
from the failed CLI transport status. No rerun was used to obtain parity.

Installed CLI0.7.4 help documents a default30-second `exec --timeout`; the old
probe supplied only the local process timeout. The controller now explicitly
passes its bounded execution window to the CLI as well. This fixes the timeout
mismatch for future probes; the original transport failure is not relabeled.

Fresh official usage after verified shutdown shows548.11 balance, rate0,
assignments0. Conservative reviewed charge0.004493171 units, measured0.00 at CLI
resolution. Shared original-plus-reviewed engineering charge0.005019091/0.1,
with original baseline and both consumed claims preserved. Project
40.126736274/150, remaining109.873263726. All artifacts remain retrieved and
checksummed. No training, new development/test outcome, native acceptance change
or trading promotion occurred. Numerical execution is now demonstrated for this
fixed replay; the next research method must be declared separately before new
model outcomes.

The [cleanup receipt](preflight-cleanup.json) removes470 owned temporary files,
19,851,406 bytes. Every extracted source/training-input file was byte-identical
to the retained original ZIP. The generated gzip paths were an exact duplicate
of the retained accepted Colab paths. Small host worker metadata was retained
separately; original local-preflight acceptance, both claims/failures, payload
archives, frozen models, complete market reports and hashes remain intact.
