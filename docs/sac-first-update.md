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

The guarded worker core is implemented; budget-controlled runtime dispatch and
independent saved-artifact restoration is implemented and tested; runtime dispatch
and budget/deadline integration remain before allocation. Local learning/optimizer steps remain prohibited. Use one
owned CPU session, at most0.1 incremental units and900 seconds including setup,
allocated idle, execution, retrieval and cleanup. Preserve a diagnostic baseline
across failures and the original project ledger;150/130 project and150/170
account guards remain. Match the actual default Colab Python before dependency
setup and reuse the installed official CLI. Stop on unreliable metering, retrieve
bounded artifacts and verify owned shutdown on every outcome. No automatic
retry after failure. The original CrossQ integrity-stop record stays untouched.

This diagnostic can inform a later reviewed research decision. It does not
authorize returning to the failed performance protocol automatically.


## Worker preparation and update guards

`training/sac_update_worker.py` constructs the two native SAC conditions with
separate copies of nested configuration. SB3 mutates its policy kwargs during
setup, so passing shared input dictionaries would invalidate the second
condition. A regression test reuses one input object and verifies immutability.

Preparation loads pinned full64 initial tensors and original streams, fills
exactly1001 replay rows, and sets original first-update progress without an
environment reset or rollout. The worker uses the historical source's explicit
single intra-op thread setting. Native updates require default-kernel Python3.13,
Colab release context, `/content/rl-stocks` working directory and CPU hardware.
The local-execution guard runs before model/output construction.

The observer wraps one native replay sample and attaches optimizer pre/post
hooks. A second sample or second optimizer attempt is rejected before extra
updates. Attempted and completed Adam steps are recorded separately; an
exception after a pre-hook is not counted as a completed step. Native one-step
replay must supply `discounts=None` and the five finite float32 sample fields.
All hook handles and the sample override are removed, including failure paths.

Successful execution must prove one native update and one completed step of
each actor/critic/entropy optimizer per condition, with actual per-parameter
Adam counters. It retains post-update native archives, numerical tensor/moment
NPZs, sampled batch NPZs and action/RNG records. Partial failures retain observer
evidence; caller RNG is restored. There is no trading performance evaluation.

[Actual untrained preparation](../results/sac-first-update-2026-10-27/preparation.json)
verifies real-input pre-update action/sample parity, immutable tensors/input,
original RNG, correct tuple/list representations and removed observers while
learn/train/Adam.step are forbidden. These local checks are not actual native
update acceptance.35 related tests pass. Budget/controller and independent
physical artifact checks remain before any Colab allocation.


## Independent saved-artifact verification

The worker now retains four raw CPU Torch snapshots in `torch-rng.npz`: initial,
before update, after update and after follow-up action. Each must match its
recorded RNG hash and valid native CPU generator geometry.

`training/sac_update_verify.py` restores original inputs/states, reconstructs the
actual replay batch, and advances Torch RNG through the two native stochastic
actor forwards under `no_grad`. No backward pass, optimizer step or learner call
is used by verification. It independently loads the saved native SAC archive
and compares policy/target/raw-entropy/Adam tensors with the complete NPZ state.
Actual saved step counters must be1 for every optimizer parameter; sample and
attempt/completion counts must agree.

Action records require exact four-component float32 hashes. The verifier replays
saved stochastic RNG states and checks exact repeatability on the verifier host,
with unchanged numerical tensors and restored caller RNG. Remote action
recomputation equality is reported separately: different CPU arithmetic may
change its value, and that observation neither changes a return tolerance nor
authorizes another fit. Raw tensors/moments/entropy and stored RNG remain exact.

The directory verifier recomputes the reported comparison from both physical
condition artifacts. It accepts either equal or different numerical outcomes
with valid matched inputs; it rejects altered counts, extra transitions,
promotion/root-cause claims and any clearing of the CrossQ stop. All source
files are rehashed unchanged.14 manual-fixture tests pass; these use forward
calls and manually populated Adam state, never fitting. Actual Colab native
update acceptance remains pending budget/runtime integration. Earlier preparation
evidence remains versioned to its original worker hash.


## Budget-controlled Colab dispatch

Version38 is a separate numerical job route, never a generic trading learner.
It validates the exact input and portable initial assets before allocation, runs
only the guarded diagnostic, and writes clean-source/kernel/hardware provenance.
The final archive is allowlisted and bounded to64MiB; physical verification
reconstructs both models, batches and RNG before the controller accepts success.
Native per-condition checkpoints are exported between the two updates.

One dedicated CPU allocation is enforced, with a persistent0.1-unit baseline
and immutable `sac-update-allocation.json` claim. Any later controller refuses
another allocation, including after failure. The original CrossQ stop remains
independent and unchanged. Original project150/130 and account150/170 guards
apply. The900-second inclusive child deadline reserves420 seconds for bounded
retrieval/cleanup, so normal work must yield by480 seconds; no unstarted-worker
execution retry is allowed for this route. Other studies retain their existing
concurrency and deadline defaults.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_sac_update_jobs.py \
  --output .runtime/sac-update-jobs-v1.json
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.fleet \
  --jobs .runtime/sac-update-jobs-v1.json --concurrency 1 \
  --cli-python /path/to/existing-colab-cli/python \
  --state-output .runtime/sac-update-state-v1.json
```

Use a clean immutable source export with the original whole-runtime ledger
symlink and exact dataset directory. Read account usage before allocation and
verify terminal cleanup afterward.58 integration/related regressions pass with
manual serialization fixtures and remote execution stubbed; actual native
Colab update evidence is still pending.
