# Frozen inference bundles

A bundle is a content-addressed directory under `artifacts/frozen/BUNDLE_HASH`. It contains selected model weights, canonical manifest, dependency lock when present (otherwise clearly recorded direct pins), and checksums for every file. The manifest includes feature/action schemas, ordered names, immutable preprocessing, exact source/split assumptions, resolved model/simulation settings, versions, seeds, selection history and model checksum.

The bundle identity hashes the manifest; the manifest hashes the weights. Loading validates the file set, each checksum, directory identity, expected feature/action schema and expected feed. Extra files, corrupt weights or mismatched feeds/features fail before prediction. Checksums provide integrity, not cryptographic publisher authentication.

## API

```python
from rl_stocks.inference.bundle import FrozenPolicy
policy = FrozenPolicy(bundle_path, feature_schema='daily-indicators-v1', feed='yahoo', device='cpu')
action = policy.predict(normalized_market_features_plus_portfolio_state)
```

`predict` requires exactly feature_count+5 finite float32-compatible values and requests deterministic SB3 prediction. The model policy is placed in evaluation mode. Loading never attaches a replay buffer; inference has no learning or normalizer update path.

`decide(features, names, portfolio_state, symbol, decision_time, expiry_time, reference, atr, equity, quantity, available_time, now=None, max_age_seconds=86400)` transforms raw market features with the frozen normalizer, appends the same five portfolio-state ratios as the environment, predicts and decodes a typed research decision.

It rejects an unknown symbol, mismatched name/order, malformed observation, nonpositive equity/reference/ATR, unavailable features, future decision time, expired signal or source age beyond the configured maximum. Required ordering is availability <= decision <= current < expiry. UTC timestamps are mandatory. Set an explicit maximum age consistent with your session/holiday scheduling; the default 24 hours is conservative and may reject stale weekend data.

Returned fields include immutable bundle ID plus action schema, decision/expiry/reference, raw and clipped action, desired exposure/whole-share quantity, notional, direction/side, stop and ordered target zone, and reconciliation intent. These are research proposals. A future broker adapter must separately reconcile actual holdings and enforce account/feed/quantity/shortability/order restrictions.

## Parity and limitations

Training compares selected pre-export deterministic action with the reloaded frozen action on the same observation using exact array equality. Integration tests also check repeated predictions, normalizer immutability, schema/feed rejection and checksum corruption. These checks pass with the reviewed Python 3.13 lock. All five downloaded A100-session bundles also passed local CPU save/load parity, repeated predictions and unchanged normalization/file checks. This does not claim bit-for-bit CPU/CUDA parity. See [verification](verification.md) for artifacts.

All current candidates, including the initial 50,000-transition PPO model, have `paper_eligible=false`. No inference call places an order, updates replay or retrains. Shared batch feature functions support causal historical recomputation; a durable live-data scheduler, incremental feature state, drift monitoring and restart reconciliation remain the shadow-inference milestone. Alpaca paper execution remains later still.

## Recurrent bundles

A `recurrent_ppo` bundle uses `predict_recurrent(observation, state=..., episode_start=...)`
and returns `(action, new_state)`. Begin with `state=None, episode_start=True`;
carry the returned hidden/cell tuple into subsequent predictions with
`episode_start=False`. Keep one state per ordered symbol/episode stream. Both
arrays must have shape `(LSTM layers, 1, hidden size)` and finite values. Input
state arrays are copied; model inference cannot modify a caller's stored state.
The normalizer remains immutable.

Reset at a new episode/symbol and after a restart unless validated matching state
or causal replay restores context. Do not reset merely because a position closes.
The stateless `predict()` and its `decide()` convenience path reject recurrent
bundles: use the explicit recurrent prediction API and the shared risk decoder.
Live stream persistence and recurrent decision orchestration are later work.
Multi-step action/state parity, reset parity, invalid state rejection and stock
order independence are covered by local acceptance. See
[algorithm stability](algorithm-stability.md) for the experimental protocol.

## Persistent forward tracking

Use the [shadow runtime](shadow-inference.md) for frozen daily inference with a
transactional position/decision journal, explicit recurrent-state recovery and
next-session simulated execution. It preserves per-stream memory across process
restarts, enforces observed-before-open signals and rejects gaps or revisions.
The registered campaign and its forward reports are separate from validation
selection and the retired test audit. It does not connect to a broker.

TQC bundles require `metadata.critic_geometry` matching actual quantile critics,
target heads, dropped/retained quantiles and algorithm/policy identity. The loader
rejects missing or internally checksummed inconsistent declarations, including a
TQC checkpoint mislabeled as SAC. Untrained deterministic prediction parity and
normalization immutability pass; learned Colab restoration remains pending in the
[distributional-critic study](distributional-critics.md).
