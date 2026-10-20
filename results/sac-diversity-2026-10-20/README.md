# SAC training diversity — smoke passed, screen next

Two1024-transition Colab CPU smokes pass on actual default Python3.13.16.
Worker source `5cad3fa2055e4262ea130410868a80400010e57a` is unchanged across
smoke and planned screen. Actual actor/critic/target fingerprints and initial
alpha1 match exactly; original frozen normalization matches in both datasets.
Sampler per-symbol/group counts reconcile.156 restored selected-bundle prefix
predictions match exactly, normalization remains immutable, and incompatible
inputs are rejected. Full111-symbol coverage is required only at screen length;
it is not inferred from these short smokes.

[Smoke records](smoke-summary.json), [restoration](smoke-restoration.json),
[protocol](../../configs/experiments/sac_diversity.json),
[guide](../../docs/sac-diversity.md).

Both owned sessions stopped; cleanup verified. Original CLI failed before
bootstrap; authenticated absence checks permitted one same-session retry,
without new allocation or reset deadlines. Conservative smoke increment
0.02075881 units; cumulative36.16986557/150. Separate account350.40, zero
reported rate/assignments after shutdown. Original study start36.14910675 and
2-unit allowance include failures/retries/setup/idle/retrieval and remain fixed.

The screen will use frozen controllerc6f69d4 (bounded delayed-absence recovery)
with worker payloads still from5cad3fa. Their revisions are recorded separately:
controller changes cannot silently change the worker source, data or parameters.
Two fresh100k CPU cells compare original and expanded under identical stock/ETF
sampling probabilities, full-window cycles, original preprocessing and exact
26-symbol development panel. No performance verdict, extension or promotion.
Retired data stays isolated. Artifacts retained until verified publication and
duplicate-only cleanup; the full objective remains unmet.
