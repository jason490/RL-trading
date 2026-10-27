# SAC first-update diagnostic: inputs registered and verified

Status: protocol/input preparation only. No native update, local fitting, Colab
allocation or trading-performance evaluation has occurred.

[Fixed protocol](../../configs/experiments/sac_first_update_diagnostic.json)
registers two SAC default representations, one native gradient update each on
one CPU. At most two model updates/six Adam steps; no environment transitions,
CrossQ learner, evaluation, selection, grid or automatic retry. Ceiling0.1units,
deadline900s including setup/retrieval/cleanup, unchanged original project/account
guards. Worker/controller and artifact checks remain before allocation.

[Actual input proof](inputs.json) confirms all1001 copied rows in six replay fields
agree bitwise between the retained original and stopped-screen SAC replays,
with matching follow-up observation. Original source hashes remain unchanged.
Input identity7f50968c0272bda3d5b5a75861e5ea25d2f4e7d328a1b014af7de219a0d5a8a7;
raw normalized replay input stays local under ignored data storage. Original
full64 seed11 tensors and original RNG streams are pinned; parameter values
are fixed and checked before any model construction.11 input tests pass.

This is a separate numerical diagnosis, not a CrossQ performance-screen retry
or a change to its failed historical gate. Equality or difference cannot prove
the old host's cause by itself or authorize promotion. The old integrity marker
and failure evidence remain. All owned sessions stay stopped; project usage
39.518547649665/150, last official account348.60/idle.

[Operating guide](../../docs/sac-first-update.md).
