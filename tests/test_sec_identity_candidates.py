"""Identity discovery must never become historical verification through a join."""
import importlib.util
import json
from pathlib import Path

import pytest


def run(tmp_path, monkeypatch, discovery, records):
    import sys
    spec = importlib.util.spec_from_file_location(
        'sec_candidates', Path(__file__).parents[1] / 'scripts/audit_sec_identity_candidates.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, structural, output = [tmp_path / n for n in ['discovery.json', 'archive.json', 'out.json']]
    source.write_text(json.dumps(discovery))
    structural.write_text(json.dumps(dict(schema='sec-archive-structural-audit-v3',
                                         causal_ready=False, quarter='2020q1',
                                         archive_sha256='source-hash', submissions=records)))
    monkeypatch.setattr(sys, 'argv', ['audit', '--discovery', str(source),
                                    '--structural', str(structural), '--output', str(output)])
    module.main()
    return json.loads(output.read_text())


def discovery(candidates):
    return dict(schema='sec-current-identity-discovery-v1', discovery_only=True,
                historical_mapping_verified=False, source_sha256='current-source', candidates=candidates)


def test_present_absent_and_shared_ciks_stay_unverified(tmp_path, monkeypatch):
    candidates = {s: [{'cik': c}] for s, c in [('A', '1'), ('B', '1'), ('C', '2')]}
    records = [dict(cik='1', form='10-K', period='20191231', adsh='original')]
    result = run(tmp_path, monkeypatch, discovery(candidates), records)
    assert result['candidate_present_count'] == 2
    assert result['candidate_absent'] == ['C']
    assert result['per_symbol']['A']['shared_cik']
    assert not result['historical_mapping_verified']
    assert not result['model_inputs_enabled']
    assert all(not r['historical_identity_verified'] for r in result['per_symbol'].values())


def test_other_forms_do_not_fill_required_financial_coverage(tmp_path, monkeypatch):
    records = [dict(cik='1', form='8-K', period='20191231', adsh='event')]
    result = run(tmp_path, monkeypatch, discovery({'A': [{'cik': '1'}], 'B': []}), records)
    assert result['candidate_present_count'] == 0
    assert result['per_symbol']['B']['ambiguous_current_candidates']


def test_current_snapshot_cannot_claim_historical_verification(tmp_path, monkeypatch):
    source = discovery({'A': [{'cik': '1'}]})
    source['historical_mapping_verified'] = True
    with pytest.raises(ValueError, match='unverified current-identity'):
        run(tmp_path, monkeypatch, source, [])
