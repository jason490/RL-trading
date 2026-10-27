# SAC first-update diagnostic: inputs registered and verified

Status: worker core and untrained real-input preparation verified. No native
update, local fitting, Colab allocation or performance evaluation has occurred.

[Fixed protocol](../../configs/experiments/sac_first_update_diagnostic.json)
registers two SAC default representations, one native gradient update each on
one CPU. At most two model updates/six Adam steps; no environment transitions,
CrossQ learner, evaluation, selection, grid or automatic retry. Ceiling0.1units,
deadline900s including setup/retrieval/cleanup, unchanged original project/account
guards. Controller and independent physical artifact checks remain before allocation.

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


## Actual untrained worker preparation

[Preparation proof](preparation.json) uses the bound real input with native
learn/train/Adam.step forbidden. Both conditions load exact original tensors/RNG
and produce equal pre-update actions and sampled batch hashes. Policy/entropy
state and input remain unchanged; caller RNG restores and observers detach.
Original betas remain tuples; explicit serialized betas remain lists.

The worker copies nested configuration because SB3 adds policy kwargs during
construction. Tests reuse a shared input to reject that regression. Pre/post
hooks distinguish attempted from completed optimizer steps, and the sampler
accepts native one-step `discounts=None` without object-valued NPZ data. Native
updates remain guarded for Colab CPU only.35 related tests pass, including
manual-state serialization and local-execution rejection. These fixtures never
fit and do not establish learned-state/native-update acceptance.

Independent saved-state/sample/RNG restoration is now implemented; next integrate
the bounded controller before allocation. CrossQ stop and all original budget history stay intact.


## Physical verifier implementation

Four raw Torch RNG snapshots now accompany each condition. Independent read-only
verification reconstructs native replay sampling and two actor forwards, restores
saved model/target/raw-entropy/Adam tensors exactly, replays action RNG, checks
local repeatability and recomputes the two-condition comparison. Equal and
different valid outcomes are accepted as diagnostic evidence; neither grants
promotion, proves the historical cause or clears the old stop. Remote action
arithmetic equality is reported separately from exact saved-state restoration.

14 tests cover manual native serialization fixtures, corrupted batch/state/RNG/
action/count/moment evidence, differing valid numerical outcomes and forbidden
summary claims. No native gradient update or Colab allocation has occurred.
Budget/deadline/controller integration remains before actual execution. The
earlier preparation proof retains its original source hash; RNG artifact
recording was added before any actual numerical update.


## Verified temporary cleanup

[Cleanup inventory](temporary-cleanup.json) records three removed staging
duplicates/88,340,415 logical bytes (84.25MiB). Two source tar files were checked
against retained immutable source directories; duplicate comparison output
matched the committed report.3,342 protected files rehashed unchanged. No
scientific model/replay/dataset/checkpoint evidence deleted. Original ledger
and CrossQ stop preserved; all owned sessions stopped.


## Runtime integration ready

The fixed version38 CPU job now includes preflight input/assets, clean-source
provenance, one persistent allocation claim,0.1-unit shared diagnostic allowance,
900-second deadline with420-second teardown reserve and physical artifact
verification.58 related tests pass using manual fixtures/no local optimization.
Next clean source/account preflight and actual native Colab updates; no prior
performance study is resumed automatically.
