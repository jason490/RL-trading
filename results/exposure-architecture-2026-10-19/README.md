# Exposure architecture study — preflight complete

The committed three-arm protocol compares native full-state PPO, an unmasked
market/full-observation branch layout, and the identical branch layout with
exposure's account connections masked. All arms use seven existing market
features plus 14 execution/account values, wide initialization, unchanged costs,
actions and net-equity reward. Source:
`4b05a6aba75a16d55c111c8165b83518a8c71ad7`.

Offline regression passed 237 tests. Preflight verifies immutable development-only
data, 21-value inputs and identical initial branch tensors except the fixed
mask. Research parameter counts are 11,465 native and 16,393 in each branch arm.
The treatment deliberately has fewer active exposure connections. No local
research training was performed.

Three 512-transition CPU Colab smokes precede the declared three-arm 50k seed-11
screen. Only treatment can be promoted, after passing selected/final, stock/ETF
and drawdown requirements against both controls. No performance result is claimed
at preflight. All phases share one frozen source and the original ledger; the
two-unit study allowance starts at 35.687822780613104 project units. Total ceiling
150, teardown threshold 130 and per-allocation two-hour deadlines remain active.

See [protocol](../../configs/experiments/exposure_architecture.json),
[methodology](../../docs/policy-architecture.md), and [preflight](preflight.json).
Retired test data and future declining-stock selection labels are excluded.

All three Colab CPU smokes now pass on actual Python 3.13.16. Three frozen
bundles pass 234 exact prefix predictions, immutable normalization checks and
input rejection. Six selected/final trained-policy contracts pass, with exact
market-only exposure invariance to account perturbations. Native smoke tensors
and every checkpoint score exactly reproduce the prior full-state control. All
three owned sessions stopped; conservative project usage 35.711819/150. The
50k screen is next; no research performance is claimed yet. See [smokes](smoke.json),
[frozen checks](smoke-frozen-checks.json), [trained contracts](trained-input-contract-smoke.json)
and [native parity](native-smoke-parity.json).
