# TD3 execution-state comparison — smoke passed, screen next

Three 1,024-transition Colab CPU smokes pass on actual default Python 3.13.16,
from frozen source `76151d145250519aacedc05a6314436e7868a15c`. Actual masked/full
initial actor, target actor, critic and target critic hashes match exactly.
All arms verify Gaussian exploration mean 0/sigma 0.1 on four actions, policy
delay 2, target noise 0.2/clipping 0.5. All 234 restored selected-bundle prefix
predictions match exactly; normalization is immutable, incompatible inputs reject.

[Smoke evidence](smoke-summary.json), [restoration](smoke-restoration.json),
[fixed protocol](../../configs/experiments/td3_execution_state.json),
[guide](../../docs/td3-execution-state.md).

All three owned sessions stopped, cleanup verified. Conservative smoke increment
0.02168043 units, cumulative 36.09087843/150. Separate account balance 350.46,
zero reported rate and assignments verified after shutdown. Original study start
36.06919800 and 2-unit allowance include setup/idle/retrieval/retries and remain
unchanged for the fixed three-arm 50k seed-11 screen. No smoke return selects
parameters. No performance gate verdict or prospective success exists yet.
Retired data remains isolated; retain models/checkpoints pending verified final
publication and duplicate-only cleanup. The overall objective remains unmet.
