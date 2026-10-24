import importlib.util
from pathlib import Path

import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        'original_pilot_report', Path(__file__).parents[1] / 'scripts/report_original_flow_pilot.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fixture(ready):
    selected = [dict(cik='1', accession=str(i), stratum='early' if i < 10 else 'late')
                for i in range(20)]
    completed = [dict(r, status='inspected', arithmetic_ready=i in ready,
                      snapshot=dict(arithmetic_ready=i in ready, reason=None if i in ready else 'missing'))
                 for i, r in enumerate(selected)]
    return (dict(status='complete', completed=completed),
            dict(selected=selected, candidate_ciks=['1', '2'],
                 absent_issuer_strata=[dict(cik='2', stratum='early')]))


def test_per_era_failure_cannot_hide_in_passing_aggregate():
    state, manifest = fixture(set(range(6)) | set(range(10, 20)))
    result = module().summarize(state, manifest)
    assert result['readiness_fraction'] == 0.8
    assert not result['fixed_source_feasibility_gate_passed']
    assert not result['per_stratum']['early']['threshold_pass']


def test_exact_overall_and_stratum_boundaries_pass_source_gate_only():
    state, manifest = fixture(set(range(7)) | set(range(10, 19)))
    result = module().summarize(state, manifest)
    assert result['fixed_source_feasibility_gate_passed']
    assert result['absent_issuer_strata'] == 1 and result['selected_originals'] == 20


def test_incomplete_failure_is_not_a_completed_gate():
    state, manifest = fixture(set(range(20)))
    state.update(status='failed', completed=state['completed'][:2])
    result = module().summarize(state, manifest)
    assert not result['gate_evaluated'] and not result['fixed_source_feasibility_gate_passed']
    assert result['incomplete_selected_sources'] == 18


def test_missing_document_remains_in_denominator():
    state, manifest = fixture(set(range(20)))
    state['completed'][0].update(status='source_missing_404', arithmetic_ready=False)
    result = module().summarize(state, manifest)
    assert result['selected_originals'] == 20 and result['arithmetic_ready'] == 19


@pytest.mark.parametrize('change', ['live', 'reorder', 'flag'])
def test_live_reordered_or_false_ready_state_rejected(change):
    state, manifest = fixture(set(range(20)))
    if change == 'live':
        state['status'] = 'running'
    elif change == 'reorder':
        state['completed'].reverse()
    else:
        state['completed'][0]['snapshot']['arithmetic_ready'] = False
    with pytest.raises(ValueError):
        module().summarize(state, manifest)


def test_oversized_document_remains_failure_in_fixed_denominator():
    state, manifest = fixture(set(range(20)))
    state['completed'][0].update(status='source_exceeds_limit', arithmetic_ready=False)
    result = module().summarize(state, manifest)
    assert result['selected_originals'] == 20 and result['arithmetic_ready'] == 19
    assert result['status_counts']['source_exceeds_limit'] == 1
