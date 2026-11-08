# Standard21-input SAC reference preparation — 2026-11-08

Exported the existing selected10k/final100k standard SAC64x64 actors under a
separate immutable21-input reference schema. Original pointers were not reselected;
physical source checkpoints/counters/moments pass restoration. Both128-point
synthetic reference predictions repeat exactly; source weights, optimizer states,
checkpoint bytes and original acceptance evidence remain unchanged.

Native batch bit parity is not claimed:436/512 selected and362/512 final output
coordinates match, with maximum differences1.1921e-7/6.5565e-7. No tolerance change
or transfer of native returns into reference results. [Contract and limits](../../docs/standard-sac-reference.md).

[Acceptance](acceptance.json) binds source/export/probe hashes and the corrected
initial diagnostic harness error. Full local artifacts remain in
`artifacts/standard-sac-reference-v1`. No fitting, market evaluation, new Colab
allocation, retired-test access or performance promotion. This prepares a strong
control for a separately preregistered future comparison; it does not reopen any
failed family. Existing PPO/TRPO Colab path calculations remain unchanged.
