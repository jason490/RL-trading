"""Read-only status fixtures; no model construction, fitting or cloud calls."""
import io
import json
import tarfile

import pytest

from rl_stocks.runtime.colab.recurrent_screen_status import observe, snapshot


def fixture(tmp_path):
    runtime = tmp_path/'.runtime'
    runtime.mkdir()
    ledger = dict(consumed=40., budget_stopped=False,
                  attempts=[dict(session='fleet-example', rate=.16, last_time=100.)])
    (runtime/'usage-ledger.json').write_text(json.dumps(ledger))
    state = runtime/'state.json'
    state.write_text(json.dumps(dict(fleet='fleet-example', status='running',
                                    sessions={'rl-stocks-cpu-example':
                                              dict(task='example', phase='exec')})))
    path = tmp_path/'artifacts/colab/rl-stocks-cpu-example/checkpoint.tar.gz'
    path.parent.mkdir(parents=True)
    return state, path


def test_observation_is_read_only_and_cannot_establish_liveness(tmp_path):
    state, path = fixture(tmp_path)
    before = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = observe(tmp_path, state, now=110.)
    assert result['meter_age_seconds'] == 10
    assert result['sessions'][0]['snapshot'] == 'not_downloaded'
    assert not result['process_liveness_verified'] and not result['full_artifact_acceptance']
    assert before == {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    path.write_bytes(b'incomplete archive')
    assert observe(tmp_path, state)['sessions'][0]['snapshot'] == 'download_incomplete'


def test_partial_curve_and_optional_exact_chart_copy(tmp_path):
    state, path = fixture(tmp_path)
    files = {'learning-curve.jsonl': json.dumps(dict(steps=10000, return_pct=3.,
                                                    excess_return_pp=-2.)).encode(),
             'selected.json': b'{"steps":10000}', 'learning-curve.html': b'<html>partial</html>'}
    with tarfile.open(path, 'w:gz') as archive:
        for name, value in files.items():
            member = tarfile.TarInfo('artifacts/comparison-cell/'+name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    charts = tmp_path/'charts'
    result = observe(tmp_path, state, charts=charts)
    assert result['sessions'][0]['provisional_curve'][0]['return_pct'] == 3.
    assert (charts/'rl-stocks-cpu-example-provisional-curve.html').read_bytes() == files['learning-curve.html']
    assert result['partial'] and not result['full_artifact_acceptance']
    (charts/'rl-stocks-cpu-example-provisional-curve.html').unlink()
    (charts/'rl-stocks-cpu-example-provisional-curve.html').symlink_to(state)
    with pytest.raises(ValueError, match='symlink'):
        observe(tmp_path, state, charts=charts)


def test_duplicate_members_rejected(tmp_path):
    path = tmp_path/'duplicate.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        for _ in range(2):
            member = tarfile.TarInfo('artifacts/comparison-cell/selected.json')
            member.size = 2
            archive.addfile(member, io.BytesIO(b'{}'))
    with pytest.raises(ValueError, match='unique'):
        snapshot(path)
