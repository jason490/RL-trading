# Recurrent sizing-alignment smoke acceptance

Three512-transition Google Colab CPU smokes complete on actual default Python
3.13.16, clean workercaddfcb3bbccc855c769e194fe10a075ce596a96. All use identical
masked21 inputs, immutable expanded111 data, normalization and sampling.

| Arm | Actual training mode | Changed sizes | Processed non-unit sizes | Changed protections |
| --- | --- | ---: | ---: | ---: |
| adaptive | unrestricted | 0 | 340 | 0 |
| posthoc_unit | unrestricted | 0 | 340 | 0 |
| aligned_unit | unit magnitude | 343 | 0 | 0 |

Each counter covers exactly512 transitions after checker resets. Actual model
settings/steps/tensor fingerprints and current checkpoint identities restore.
All initial tensors match; adaptive/posthoc final tensors match exactly despite
different validation execution and selection. All actual sampling counts agree.
Unit selected/final validation traces preserve all three protective components.

234 exact frozen prefix predictions cover three observations on each26-symbol
original development episode per model. Mapped actions, carried hidden/cell state,
deterministic repeats, independent resets and caller-state immutability pass.
Normalization/data match immutable source; schema/feed/nonfinite rejection passes.
These are wiring checks, not financial-performance evidence; smoke returns do
not tune the fixed experiment. [Acceptance](smoke-summary.json),
[frozen checks](smoke-restoration.json),
[protocol](../../configs/experiments/recurrent_alignment.json),
[guide](../../docs/recurrent-sizing-alignment.md).

All three owned sessions stopped and cleanup verified. Project38.04785547/150,
study increment0.03344750 including setup/idle/retrieval, remaining111.95214453.
Official post-shutdown account349.78, zero rate/assignments. Existing CLI reused;
ledger not reset; project150/130 and account150/170 safeguards remain. No provider
calls, retired-test evaluation or orders.

Next fixed fresh100k seed11 adaptive/posthoc/aligned screen on the same worker,
within the existing3-unit total study allowance. Both controls, absolute stock/
ETF selected/final targets and drawdown gates remain; no promotion or prospective
success claimed. Original failed state/sizing-replay evidence remains unchanged.
