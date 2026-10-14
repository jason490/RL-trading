# Gymnasium environment API

`environment/trading/env.py` defines `TradingEnv(gymnasium.Env)`. It consumes already generated, local episodes plus a frozen normalizer. Reset/step contain no provider retrieval, authentication, provisioning or broker calls.

## Constructor

`TradingEnv(episodes, normalizer, names, config=SimulationConfig(), random_start=False, episode_steps=None)` requires at least one episode, a matching ordered feature schema and a single chronological split. Training can choose random symbols/starts within training only. Evaluation creates one deterministic environment for each validation/test episode.

Action space: `Box(-1, 1, shape=(4,), dtype=float32)`. Observation space: `Box(-inf, inf, shape=(feature_count + 5,), dtype=float32)` with explicit finite-value checks. Unbounded declared observation limits avoid pretending that a normalizer imposes hard economic bounds.

The portfolio state appended after normalized market features is:

1. Signed marked position notional / marked equity.
2. Cash / marked equity.
3. Holding age / maximum holding sessions.
4. Unrealized price P&L / marked equity.
5. Marked equity / initial equity minus one.

Cash ratio can exceed one for a short because short-sale proceeds are included; it is not unrestricted buying power. The action gate still constrains notional exposure. Bankruptcy observations use a small denominator floor to remain finite; true terminal state prevents subsequent steps.

## Reset

`reset(*, seed=None, options=None)` calls `super().reset(seed=seed)` and uses `self.np_random` for episode/start selection. `options={"episode": index}` selects a specific episode; its state is recreated with initial cash, no holdings or protection. Return value is `(observation, info)`; reset info names symbol/session. Training also seeds action/observation spaces, Python, NumPy and torch.

## Step

`step(action)` uses the current completed decision bar/ATR and advances to the next local bar. Availability is checked against that execution open. It returns `(observation, reward, terminated, truncated, info)` with float32 observation, scalar finite reward and Python booleans.

Reward is net log marked-equity change after fees, adverse fills, corporate-action cash flows and short borrow. Additional turnover/downside/holding rewards are disabled. Info includes accounting state, cumulative costs/turnover, step borrow, raw return, raw/clipped decision, approved target quantity, risk clipping/rejections, active protection and this step's fill events.

Equity <=0 is true termination with finite reward -20. Dataset end or configured episode-step limit is truncation. The returned terminal marked observation permits correct SB3 timeout bootstrapping. No forced liquidation is added at those cutoffs; costs/returns are documented as mark-to-market. Calling `step` after a boundary raises until reset.

## SB3 adaptation and acceptance

The raw environment remains a Gymnasium interface. SB3 wraps it using its supported vector adapter; the raw five-return step/reset tuple must not be used as if it were an SB3 VecEnv API. Gymnasium and SB3 checkers run in the training entry point and integration suite.

Dependency-free tests verify the underlying simulator and boundaries conceptually, but do not replace those actual checker runs. The checker/seeded-reset/time-limit integration test is currently skipped until real dependencies are installed. Its status is explicit in [verification.md](verification.md).
