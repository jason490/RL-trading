# Forest signal feasibility — dependency/data preflight accepted

No forest has been fitted and no new Colab session allocated. This stage adds
an optional, hash-locked tree dependency group and portable numerical tree
storage/prediction. All existing package pins are preserved.

[Preflight](preflight.json) records Python 3.13.16, scikit-learn 1.9.1, SciPy
1.17.1, joblib 1.6.0 and threadpoolctl 3.7.0. Core environment and package checks
pass. The immutable input remains 45,266 fit and 18,616 probe examples; no
project-validation or retired-test labels are added.

[Protocol](../../configs/experiments/forest_signal.json) fixes six 128-tree forests,
two feature arms, seeds 11/23/37, depth/leaf/sampling limits and unchanged numeric
predictive gates. Two prefix trees must pass dependency/input/portable-parity
acceptance in Colab before full fitting. There is no hyperparameter sweep.

264 non-training regression tests and three focused forest checks pass without
local model fitting. Tests verify numerical threshold/averaging behavior,
serialization and malformed/nonfinite rejection. Actual trained sklearn-to-NPZ
prediction parity and model-quality results remain pending.

The CPU fitting worker, complete frozen forecast bundle, strict version14
fleet/artifact contracts and post-download verification are now implemented.
42 focused tests pass without local fitting. Next use the existing
metered Colab fleet under the one-unit study cap. No trading policy, PPO warm
start, order interface or profitability claim is introduced. Project usage stays
35.861240/150; no cleanup of an owned runtime was needed at this stage.

[Detailed setup, definitions and sources](../../docs/forest-signal.md).
