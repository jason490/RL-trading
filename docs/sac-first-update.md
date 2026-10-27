# SAC first-update numerical diagnostic

## Question and boundary

The stopped CrossQ comparison diverges from its historical SAC control after
the first native update. Initial policy/RNG/data/execution contracts agree;
first1001 stored actions agree, then the near-target action differs by four
float32 ULPs. Historical CPU/backend provenance is incomplete. This separate
diagnostic asks whether the original SAC default representations and the newer
explicit serialized defaults produce equal first-update states on one fixed CPU.

It does not train CrossQ, retry the50k screen, compare trading returns, select
checkpoints, change the old tolerance or clear the integrity stop. An equality
result rules out a representation effect only for this input/backend. A
difference localizes a discrepancy for this input/backend; it does not by itself
prove the old host's cause. No performance promotion follows either result.

## Fixed protocol

[Preregistration](../configs/experiments/sac_first_update_diagnostic.json) fixes
seed11 and two native SAC conditions: original defaults and explicit defaults
after JSON serialization. Both keep21 observations, four actions, full64×64
networks, batch256, original unfitted tensors and original Torch/NumPy/Python
streams. Effective numerical settings must agree. The actor/critic/entropy
container differences remain visible rather than being rewritten to match.

Each condition permits exactly one native gradient update: at most two model
updates/six Adam steps total. No environment rollout is collected. The fixed
1001-row replay prefix is training-only retained evidence. One stochastic
pre-update actor call at original row1000 precedes the native update; the
follow-up observation is original row1001. Record actual sampled batch, RNG,
policy/target/entropy tensors, all Adam states, actions and numerical backend.
Any result must distinguish copied input equality from actual update equality.

## Immutable input and local checks

`training/sac_update_input.py` validates the fixed protocol, source identities,
original full64 initializer, parameter representations, six finite float32 replay
fields, action bounds, timeout masks and the shared follow-up observation.
`scripts/build_sac_update_inputs.py` reads only the pinned original metadata/
replay and pinned owned diagnostic checkpoint. All1001 copied rows of each field
must agree bitwise between the two retained training replays. Original files are
hashed before/after; no provider request, new label or model construction occurs.
The raw normalized replay-derived input stays in ignored local data storage.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_sac_update_inputs.py \
  --output data/datasets \
  --proof results/sac-first-update-2026-10-27/inputs.json
```

Outputs are immutable; existing proof paths are not overwritten.
[Actual input proof](../results/sac-first-update-2026-10-27/inputs.json) pins
input7f50968c0272bda3d5b5a75861e5ea25d2f4e7d328a1b014af7de219a0d5a8a7,
source and implementation hashes.11 pure-input tests pass without fitting or
allocation. These checks are not native-update or learned-state acceptance.

## Colab-only execution still pending

The worker and budget-controlled runtime/artifact acceptance must be implemented
before allocation. Local learning/optimizer steps remain prohibited. Use one
owned CPU session, at most0.1 incremental units and900 seconds including setup,
allocated idle, execution, retrieval and cleanup. Preserve a diagnostic baseline
across failures and the original project ledger;150/130 project and150/170
account guards remain. Match the actual default Colab Python before dependency
setup and reuse the installed official CLI. Stop on unreliable metering, retrieve
bounded artifacts and verify owned shutdown on every outcome. No automatic
retry after failure. The original CrossQ integrity-stop record stays untouched.

This diagnostic can inform a later reviewed research decision. It does not
authorize returning to the failed performance protocol automatically.
