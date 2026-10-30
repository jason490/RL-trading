"""Measured execution-only CPU screen contract; original research settings stay frozen."""
import hashlib
import json
from pathlib import Path
import tarfile

from rl_stocks.core.contracts import digest
from rl_stocks.training.recurrent_comparison import PROTOCOL_SHA256,protocol,profile_admission

EXECUTION_SHA256='b6186a119bda37509d4f661536b0aa663b3f4641f2998f05b3a6dc12c388b318'
ROOT=Path(__file__).resolve().parents[3]


def specification():
    value=json.loads((ROOT/'configs/experiments/recurrent_sac_cpu_screen.json').read_text())
    if digest(value)!=EXECUTION_SHA256 or value['research_protocol_sha256']!=PROTOCOL_SHA256:
        raise ValueError('fixed measured CPU screen contract required')
    return value


def verify_compute_sources(root):
    for name,expected in specification()['compute_files'].items():
        if hashlib.sha256((Path(root)/'src/rl_stocks'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('accepted-profile compute implementation changed: '+name)


def verify_profile(root):
    from rl_stocks.runtime.colab.artifacts import validate_archive
    from rl_stocks.runtime.colab.jobs import recurrent_reviewed_profile_plan
    spec=specification()
    verify_compute_sources(root)
    archive=Path(root)/spec['profile_archive']
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=spec['profile_archive_sha256']:
            raise ValueError('exact accepted corrected profile archive required')
    proof=validate_archive(archive,recurrent_reviewed_profile_plan(),protocol()['dataset'])
    if not proof[0]['saved_artifacts_verified']:
        raise ValueError('physical profile acceptance required')
    with tarfile.open(archive) as stream:
        timings=json.load(stream.extractfile('artifacts/comparison-profile/profile.json'))
    projection=profile_admission(timings)
    passes={arm:projection['required_allocation_seconds'][arm]<=seconds
            for arm,seconds in spec['allocation_seconds'].items()}
    if not all(passes.values()):
        raise ValueError('measured complete100k comparison does not fit declared CPU deadlines')
    return dict(physical_profile_accepted=True,execution_sha256=EXECUTION_SHA256,
                profile_archive_sha256=spec['profile_archive_sha256'],arm_passes=passes,
                required_allocation_seconds=projection['required_allocation_seconds'],
                original_two_hour_projection_passed=projection['profile_projection_passed'],
                research_settings_unchanged=True)
