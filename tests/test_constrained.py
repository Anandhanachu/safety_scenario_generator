"""Safety boundary tests use synthetic adapters and explicit test-only expert metadata."""
import copy
import json
import threading

import pytest
from pydantic import ValidationError
from backend.configuration import ROOT
from backend.schemas import ScenarioInput, Generation
from backend.constrained import (SelectionOutput, ModelDetails, execute, baseline, stored_risk,
                                 verify_selection, output_schema, scope)
from evaluation.labelled import LabelledDataset, evaluate, precision_recall
from knowledge_base.core import parse_upload, retrieve_with_diagnostics
from models.adapters import BackendError, Cancelled


@pytest.fixture
def kb():
    return parse_upload((ROOT / 'data/constrained_knowledge_base.json').read_bytes())


@pytest.fixture
def evidence(kb):
    return retrieve_with_diagnostics(kb, ScenarioInput(faults=['camera_failure']).model_dump(), 12, 0)[0]


class Responses:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def generate_scenarios(self, model, messages, settings, cancel):
        self.calls.append((copy.deepcopy(messages), settings))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def selected(evidence):
    return baseline(evidence).model_dump_json()


@pytest.mark.parametrize('mutation', ['missing_source', 'missing_tags', 'broken_hazard', 'wrong_type', 'bad_range', 'missing_reviewer', 'missing_backlink'])
def test_upload_specific_failures(kb, mutation):
    raw = kb.model_dump()
    if mutation == 'missing_source': raw['records'][0]['source'] = 'website'
    if mutation == 'missing_tags': raw['records'][0]['tags'] = None
    if mutation == 'broken_hazard': raw['records'][2]['hazard_ids'] = ['M-1']
    if mutation == 'wrong_type': raw['records'][1]['test_ids'] = ['M-1']
    if mutation == 'bad_range': raw['records'][1]['severity'] = 'S4'
    if mutation == 'missing_reviewer': raw['records'][0]['review_status'] = 'approved'
    if mutation == 'missing_backlink': raw['records'][1]['mitigation_ids'] = []
    with pytest.raises(ValidationError):
        parse_upload(json.dumps(raw).encode())


def test_scope_filter_before_ranking(kb):
    inputs = ScenarioInput(faults=['communication_loss'], notes='camera stale perception')
    hits, diagnostics = retrieve_with_diagnostics(kb, inputs.model_dump(), 12, 0)
    assert {e['record']['id'] for e in hits} == {'SRC-DEMO', 'H-2', 'M-2', 'T-2'}
    assert diagnostics['eligible_records'] == 4
    assert all(e['score'] is not None for e in hits)
    assert retrieve_with_diagnostics(kb, ScenarioInput(faults=['lidar_failure']).model_dump())[0] == [] or all(e['record']['id'] == 'SRC-DEMO' for e in retrieve_with_diagnostics(kb, ScenarioInput(faults=['lidar_failure']).model_dump())[0])
    assert scope(ScenarioInput(faults=['not_supported'])) is None
    assert scope(ScenarioInput(faults=['camera_failure'], weather='wet')) is None


def test_id_enums_and_prose_forbidden(evidence):
    schema = output_schema(evidence)
    assert schema['$defs']['Choice']['properties']['hazard_id']['enum'] == ['H-1']
    data = json.loads(selected(evidence))
    data['confidence'] = 1.0
    with pytest.raises(ValidationError): SelectionOutput.model_validate(data)


def test_retry_once_then_success(evidence, matrix):
    adapter = Responses('{broken', selected(evidence))
    result = execute(adapter, ModelDetails(backend='test', name='mock'), ScenarioInput(faults=['camera_failure']), evidence, matrix)
    assert result.output and len(result.attempts) == 2
    assert result.confidence == 1.0 and result.needs_human_review and not result.accepted
    assert adapter.calls[0][1] == {'temperature': 0.0, 'seed': 42, 'max_tokens': 1800}
    assert 'Validation errors' in adapter.calls[1][0][-1]['content']
    assert adapter.calls[0][0] == adapter.calls[1][0][:-1]


def test_final_invalid_never_accepted(evidence, matrix):
    raw = json.loads(selected(evidence))
    raw['selections'][0]['mitigation_ids'] = ['INVENTED']
    adapter = Responses(json.dumps(raw), json.dumps(raw))
    result = execute(adapter, ModelDetails(backend='test', name='mock'), ScenarioInput(), evidence, matrix)
    assert result.output is None and len(adapter.calls) == 2
    assert result.confidence == pytest.approx(2 / 3)
    assert result.needs_human_review and not result.accepted


def test_wrong_hazard_link_counts_as_unsupported(kb, matrix):
    evidence = [{'record': r.model_dump(), 'score': 1.0} for r in kb.records]
    data = SelectionOutput(schema_version='2.0', outcome='selected', selections=[{'hazard_id': 'H-1', 'mitigation_ids': ['M-2'], 'acceptance_test_ids': ['T-1']}])
    check = verify_selection(data, evidence, matrix)
    assert not check['valid'] and check['confidence'] == pytest.approx(2 / 3)


def test_transport_failure_not_retried(evidence, matrix):
    adapter = Responses(BackendError('unavailable'))
    with pytest.raises(BackendError): execute(adapter, ModelDetails(backend='test', name='mock'), ScenarioInput(), evidence, matrix)
    assert len(adapter.calls) == 1


def test_cancel_never_sends(evidence, matrix):
    cancel = threading.Event()
    cancel.set()
    adapter = Responses('unused')
    with pytest.raises(Cancelled): execute(adapter, ModelDetails(backend='test', name='mock'), ScenarioInput(), evidence, matrix, cancel=cancel)
    assert adapter.calls == []


def test_expert_risk_only_and_conflicts(kb, matrix):
    hazard = kb.records[1].model_dump()
    assert stored_risk(matrix, hazard)['risk_score'] is None
    hazard.update(reviewed_by='TEST-ONLY', review_status='approved', illustrative=False)
    matrix = matrix.model_copy(update={'illustrative': False})
    risk = stored_risk(matrix, hazard)
    assert risk['risk_score'] == 6 and risk['risk_level'] == 'medium'
    assert 'not a calibrated probability' in risk['explanation']
    assert stored_risk(matrix, {**hazard, 'risk_conflict': True})['risk_score'] is None
    assert stored_risk(matrix, {**hazard, 'likelihood': None})['risk_score'] is None


def test_baseline_and_all_labels_measured(kb, matrix):
    labels = LabelledDataset.model_validate_json((ROOT / 'data/labelled_cases_24.json').read_bytes())
    assert 20 <= len(labels.cases) <= 30
    assert sum(c.expected_outcome == 'insufficient_evidence' for c in labels.cases) >= 3
    result = evaluate(labels, kb, matrix, repeats=3)
    assert len(result['rows']) == 72
    summary = result['summary'][0]
    assert summary['hazards_recall']['mean'] == 1.0
    assert summary['mitigations_precision']['mean'] == 1.0
    assert summary['acceptance_tests_recall']['mean'] == 1.0
    assert summary['insufficient_evidence_correct']['mean'] == 1.0
    assert summary['risk_correct']['mean'] == 0.0  # Draft expert inputs cannot produce approved risk.
    assert not labels.reviewed
    with pytest.raises(ValueError): evaluate(labels, kb, matrix, repeats=1)


def test_precision_denominators_and_generation_settings():
    assert precision_recall(['a', 'b'], ['b', 'c']) == {'tp': 1, 'fp': 1, 'fn': 1, 'precision': 0.5, 'recall': 0.5}
    assert precision_recall([], [])['precision'] is None
    with pytest.raises(ValidationError): Generation(temperature=0.2)
    with pytest.raises(ValidationError): Generation(seed=43)


def test_fair_prompts_and_repetitions(kb, evidence, matrix):
    labels = LabelledDataset.model_validate_json((ROOT / 'data/sample_cases_5.json').read_bytes())
    assert len(labels.cases) == 5 and sum(c.expected_outcome == 'insufficient_evidence' for c in labels.cases) == 1
    labels = labels.model_copy(update={'cases': labels.cases[:1]})
    a, b = Responses(*([selected(evidence)] * 3)), Responses(*([selected(evidence)] * 3))
    result = evaluate(labels, kb, matrix, [(ModelDetails(backend='laptop_local', name='test-a'), a), (ModelDetails(backend='uno_q', name='test-b'), b)], repeats=3)
    assert len(result['rows']) == 9
    assert a.calls == b.calls
    assert all(row['model']['parameter_size'] is None for row in result['rows'])


def test_review_gate_and_report_fields(store, evidence, matrix):
    from backend.constrained import scenario_views
    from evaluation.labelled import baseline_result
    from reporting.exports import comparison_rows, csv_bytes, report
    result = baseline_result(evidence, matrix)
    scenarios = scenario_views(result, ScenarioInput(faults=['camera_failure']))
    run = {'results': {'laptop_local': {'model_id': 'TEST-ONLY', 'status': 'completed', 'scenarios': scenarios,
                                      'constrained': result.model_dump()}},
           'snapshot': {'baseline_comparison': result.model_dump(), 'evidence': evidence}}
    identity = store.save('run', run)
    with pytest.raises(ValueError, match='Resolve expert'):
        store.review(identity, 'laptop_local:SELECTION-1', 'accepted', 'TEST-ONLY', '')
    rows = comparison_rows(run)
    assert len(rows) == 2
    for row in rows:
        for key in ('retrieved_evidence', 'verification', 'risk_calculation_steps', 'human_review_flags', 'settings', 'confidence', 'baseline_comparison', 'limitations'):
            assert key in row
    assert 'not measured' in csv_bytes(rows).decode('utf-8-sig')
    assert report(run)['limitations']


def test_v2_top_k_never_expands_past_limit(kb):
    hits, diagnostics = retrieve_with_diagnostics(kb, ScenarioInput(faults=['camera_failure']).model_dump(), limit=1, threshold=0)
    assert len(hits) == 1 and diagnostics['relationship_matches'] == 0
