"""Allowlisted, independently rechecked GAE evidence for public model archives."""

from rl_stocks.runtime.colab.gae_rollout_artifacts import validate_rollout_evidence
from rl_stocks.training.gae_spec import protocol


def public_evidence(run, metadata):
    spec = protocol()
    arm, smoke = metadata.get("gae_arm"), metadata.get("smoke")
    if arm not in spec["arms"] or type(smoke) is not bool:
        raise ValueError("registered GAE arm/phase required for publication")
    steps = spec["smoke_steps" if smoke else "steps_per_cell"]
    if type(metadata.get("actual_steps")) is not int or metadata["actual_steps"] != steps:
        raise ValueError("complete registered GAE steps required for publication")
    files = [run / "gae-initialization.npz", run / "gae-rollouts.jsonl"]
    directory = run / "gae-rollouts"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("owned GAE rollout directory required")
    files += sorted(directory.iterdir())
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("regular owned GAE evidence files required")
    names = {str(p.relative_to(run)) for p in files}
    if (run / "gae-rollout-failure.json").exists():
        names.add("gae-rollout-failure.json")
    validate_rollout_evidence(
        lambda name: (run / name).read_bytes(),
        "",
        metadata,
        {"gae_study": dict(arm=arm, smoke=smoke), "jobs": [dict(steps=steps)]},
        names,
    )
    return files
