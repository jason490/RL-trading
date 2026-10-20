# SAC initial-temperature comparison — smoke passed, screen launched

Two1024-transition Colab CPU smokes pass on actual Python3.13.16. Initial alpha
is1 versus0.00099999993 (float32 representation of0.001), target entropy−4 in
both. Actual actor/critic/target-network initialization fingerprints match
exactly. All156 restored selected-bundle prefix predictions are exact;
normalization is immutable and incompatible inputs rejected.
[Smoke evidence](smoke-summary.json), [restoration](smoke-restoration.json),
[protocol](../../configs/experiments/sac_temperature.json),
[guide](../../docs/sac-temperature.md).

Both owned smoke sessions stopped, cleanup verified. Conservative smoke
increment0.01244813 units; cumulative36.02214035/150. Separate account350.51,
zero reported assignments/rate verified after smoke. Models/checkpoints retained
until final publication and verified duplicate-only cleanup.

The fixed two-arm50k seed11 CPU screen is now launched from the same source
`0f04e6b3ad96a4afb30243dc8f9066fcf8dfb4f9`. Original study start36.00969223
and2-unit allowance include smoke/setup/idle/retrieval/failures and are never
reset. Only initial learned temperature differs; reward scale remains1. Fresh
default model tensors and numerical checkpoint scores must reproduce the prior
full-state control before interpreting effects. No performance gate verdict,
promotion or prospective success is available. Retired test stays excluded.
