# MOVE and VIX context catalog

[ICE's MOVE description](https://developer.ice.com/fixed-income-data-services/catalog/ice-data-indices-move-index)
identifies MOVE as fixed-income implied volatility. VIX describes equity option
implied volatility; see the existing [VIX guide](implied-volatility.md). Neither
index is an executable equity in this system.

The separate [MOVE config](../configs/context/move.json) and
[VIX config](../configs/context/vix.json) declare context-only, nontradable records.
Yahoo's current-vintage historical observations are retained immutably and
checksummed. Acquisition is not a claim of official historical publication
vintages, free commercial licensing, or model usefulness. Never substitute
realized Treasury volatility for missing MOVE observations without a new schema.

The November8 audit downloaded3,693 MOVE observations (2009–2023), and reused
all3,774 retained VIX observations after checking the original snapshot checksum.
Against2010–2023 XNYS dates, MOVE lacks79 observations; VIX lacks none. This raw
same-date comparison is not an availability-aware feature coverage gate. Bond
and equity holidays differ. Do not forward-fill gaps or treat them as zero.

MOVE availability is conservatively assumed at the next calendar day23:59:59UTC;
VIX retains its prior following-XNYS-close-plus15minutes assumption. Neither is
verified original publication timing. Any combined feature builder must use
backward availability joins, an explicit staleness limit and lagged coverage.

Both inputs remain disabled in the current model. The earlier VIX experiment
failed and remains closed; catalog reuse does not reopen it. A distinct MOVE or
cross-asset hypothesis requires coverage/preprocessing checks and a bounded
matched protocol before any training. [Acquisition and coverage](../results/stock-context-expansion-2026-11-08/README.md).
