# Recurrent cash-choice study

Status: initial smoke attempt failed; no accepted training result or performance claim.

[Failure and cleanup record](smoke-failure.json) preserves four owned session IDs, source identity, retained archive checksum and charges. Binary policy likelihood rejected the library's zero-padded recurrent minibatch rows. The controller retrieved the failure archive and verified shutdown of all four sessions. Other cells had no retrievable snapshot when stopped.

The implementation fix gives only all-zero binary evaluation placeholders a valid dummy direction before the unchanged library mask excludes them. Valid sampled actions and binary action support remain unchanged. A regression builds the actual recurrent rollout buffer with unequal episode lengths, checks padded likelihoods and masked gradients, and confirms no input mutation or local optimizer updates.

Project charge is38.14814485 units; this attempt conservatively charged0.01148649 from rate estimates despite zero observed rounded account debit. Last official usage349.70 balance, zero assignments/rate. No budget stop; original cumulative ledger and four-unit study allowance remain. No retired-test data or orders.

[Protocol and next acceptance steps](../../docs/recurrent-cash-choice.md).
