import io
import json
import tarfile

import pytest

from rl_stocks.runtime.colab.status import checkpoint, summarize


def test_only_retained_checkpoint_state_is_read_without_extraction(tmp_path):
    path = tmp_path / 'checkpoint.tar.gz'
    record = dict(latest=dict(steps=20000, phase='on_step'), selected=dict(steps=10000), score=0.1)
    with tarfile.open(path, 'w:gz') as archive:
        for name, content in [('artifacts/runs/a/checkpoint-state.json', json.dumps(record).encode()),
                              ('../../outside', b'no extraction')]:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    before = path.read_bytes()
    result = checkpoint(path)
    assert result['latest_steps'] == 20000 and result['latest_score'] == 0.1
    assert result['selected_steps'] == 10000 and 'selected_score' not in result
    assert result['provisional'] and before == path.read_bytes()
    assert not (tmp_path.parent / 'outside').exists()
    path.write_bytes(b'incomplete download')
    assert checkpoint(path)['status'] == 'unreadable_or_being_replaced'
    assert checkpoint(tmp_path / 'missing')['status'] == 'not_retrieved'


def test_recorded_status_does_not_prove_liveness_or_write_ledger(tmp_path):
    state, ledger = tmp_path / 'state.json', tmp_path / 'ledger.json'
    state.write_text(json.dumps(dict(status='running', sessions={'owned': dict(task='full', phase='exec')})))
    ledger.write_text(json.dumps(dict(consumed=39, ceiling=150, budget_stopped=False)))
    before = ledger.read_bytes()
    result = summarize(state, tmp_path, ledger)
    assert result['recorded_status'] == 'running' and not result['remote_liveness_verified']
    assert result['remaining_project_units'] == 111 and ledger.read_bytes() == before
    state.write_text(json.dumps(dict(sessions={'../escape': dict(task='full', phase='exec')})))
    with pytest.raises(ValueError):
        summarize(state, tmp_path, ledger)
