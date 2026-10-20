# SAC regularization — smoke acceptance

Fixed actor/critic Adam decay0 versus1e-4, unchanged learned entropy optimizer,
expanded111-symbol training and original26 development. Final performance has
not been screened yet. [Protocol](../../configs/experiments/sac_regularization.json).

The first smoke attempt failed strict plan validation because SB3 mutated nested
caller settings. Factory deep-copy fix6792d2b preserves requested metadata.
[Failure evidence](smoke-attempt-1.json) remains; both original sessions stopped.

Two replacement Colab CPU smokes pass1024 transitions each on Python3.13.16.
Initial actor/critic/target hashes and alpha match; actual sampling counts match.
Actual selected/final saved optimizer records restore exactly, actor/critic decay
matches each arm, entropy decay stays0.156 selected-bundle causal prefix predictions
are exact; normalization remains immutable and incompatible inputs are rejected.
[Smoke summary](smoke-summary.json), [frozen restoration](smoke-restoration.json).

All4 smoke-attempt sessions stopped with verified cleanup. Cumulative conservative
project36.29759750/150, remaining113.70240250; includes failed setup/training,
replacement setup, idle and retrieval. Account350.29, zero rate/assignments after
shutdown. Original study starting units36.26389527 and2-unit ceiling remain.

Fresh paired100k CPU screening is next from the same frozen6792d2b worker;
retain10k curves and selected/final paths, actual sampling equality and fixed
absolute/relative gates. No retired2024–September2026 data, future-label selection,
broker orders, confirmation or objective-completion claim.
