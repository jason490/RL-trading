import io
import json
import tarfile

import pytest

from rl_stocks.training.task_screen_acceptance import bind_verified_cell


def test_retained_cell_must_match_independent_replay_proof(tmp_path):
    cell = {'steps': 100000, 'metrics': {'score': .1}}
    proof = dict(task_role='tagged_routed', replayed_checkpoints=10, performance_promotion=False,
                 training_cell=cell, selected_step=50000,
                 frozen_bundles={'selected': 'a' * 64, 'final': 'b' * 64})
    run = dict(status='complete', task_role='tagged_routed', actual_transitions=100000,
               development_evaluated=False, test_evaluated=False,
               selection={'selected': {'steps': 50000}})
    path = tmp_path / 'cell.tar'
    with tarfile.open(path, 'w') as archive:
        items = {'training-cell.json': cell, 'run.json': run}
        items.update({f'frozen-{phase}/{identity}/manifest.json': {} for phase, identity in proof['frozen_bundles'].items()})
        for name, value in items.items():
            data = json.dumps(value).encode()
            member = tarfile.TarInfo('artifacts/task-screen/' + name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    assert bind_verified_cell(path, proof, 'tagged_routed') == cell
    proof['training_cell'] = {'steps': 100000, 'metrics': {'score': .9}}
    with pytest.raises(ValueError, match='gate cell mismatch'):
        bind_verified_cell(path, proof, 'tagged_routed')
    proof['replayed_checkpoints'] = 1
    with pytest.raises(ValueError, match='ten-checkpoint'):
        bind_verified_cell(path, proof, 'tagged_routed')
