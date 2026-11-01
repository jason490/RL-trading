import json

import pytest

from rl_stocks.core.contracts import digest
from rl_stocks.training.task_conditioning import PROTOCOL_SHA256
from rl_stocks.training.task_selection import select_training_checkpoint
from test_task_bundle import bundle


def test_saved_success_cannot_replace_actual_checkpoint_updates(tmp_path, monkeypatch):
    import rl_stocks.training.task_screen_verify as verifier
    data = {'manifest': {'evaluation_symbols': ['A']}}
    monkeypatch.setattr(verifier, 'protocol', lambda: {
        'dataset': digest(data), 'selection': {'checkpoints': list(range(10000, 100001, 10000))}})
    monkeypatch.setattr(verifier, 'validate_type_manifest', lambda *args:
        '7c4a078fcec05be000d720e2d9625c6cd5d7c03b5d1d6502e46357f526ba1590')
    history = [dict(steps=s, phase='training-window', score=.1, holding_path_sha256='f' * 64)
               for s in range(10000, 100001, 10000)]
    run = dict(status='complete', task_role='tagged_routed', seed=11,
        protocol_sha256=PROTOCOL_SHA256, source='source', repository={'revision': 'revision'},
        actual_transitions=100000, development_evaluated=False, test_evaluated=False,
        dataset=digest(data), optimizer_steps=dict.fromkeys(('actor', 'critic', 'entropy'), 99000),
        selection=select_training_checkpoint(history))
    (tmp_path / 'run.json').write_text(json.dumps(run))
    (tmp_path / 'curve.json').write_text(json.dumps(history))
    model, _, _ = bundle(tmp_path)
    model.batch_size = 256
    model.learning_starts = 1000
    model.save(tmp_path / 'initial.zip')
    model.save(tmp_path / 'checkpoint-10000.zip')
    with pytest.raises(ValueError, match='optimizer counters'):
        verifier.verify_cell(tmp_path, data, {}, role='tagged_routed',
                             source='source', git={'revision': 'revision'})
