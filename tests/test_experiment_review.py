"""Review regression checks; all adapter outputs and expert approvals here are test fixtures."""
import copy
import io
import json

import httpx
import pytest
from pydantic import ValidationError
from pypdf import PdfReader

from backend.configuration import ROOT, defaults
from backend.constrained import ModelDetails, SelectionOutput, baseline, verify_selection
from backend.schemas import ScenarioInput
from evaluation.labelled import LabelledDataset, baseline_result, case_evidence, evaluate_predictions, run_experiment
from knowledge_base.core import parse_upload, retrieve_with_diagnostics
from models.adapters import ChatAdapter, QualcommAdapter, UnoAdapter, BackendError
from reporting.exports import experiment_report, experiment_csv, json_bytes
from reporting.pdf_export import experiment_pdf_bytes
from risk_engine.calculator import RiskMatrix


@pytest.fixture
def reviewed():
    kb = parse_upload((ROOT / 'data/constrained_knowledge_base.json').read_bytes())
    for record in kb.records:
        record.reviewed_by, record.review_status, record.illustrative = 'TEST-ONLY', 'approved', False
        record.expert_support_review = 'supported'
    matrix = RiskMatrix.model_validate_json((ROOT / 'configs/risk_matrix.example.json').read_bytes())
    matrix.illustrative = False
    labels = LabelledDataset.model_validate_json((ROOT / 'data/labelled_cases_25.json').read_bytes())
    return kb, matrix, labels


def first_result(reviewed):
    kb, matrix, labels = reviewed
    evidence = retrieve_with_diagnostics(kb, labels.cases[0].inputs().model_dump(), 12, 0)[0]
    return baseline_result(evidence, matrix), evidence


def test_25_cases_explicit_issues_and_review_dates(reviewed):
    _, _, labels = reviewed
    assert len(labels.cases) == 25
    assert sum(c.expected_outcome == 'insufficient_evidence' for c in labels.cases) >= 4
    assert sum(c.risk_input_issue.startswith('missing_') for c in labels.cases) == 2
    assert sum(c.risk_input_issue == 'conflicting_evidence' for c in labels.cases) == 2
    assert all(c.expected_risk_category is None for c in labels.cases if c.expected_outcome == 'needs_review')
    assert not labels.reviewed and labels.reviewer is None and labels.reviewed_date is None
    invalid = labels.model_dump()
    invalid.update(reviewed=True, reviewer='TEST-ONLY', reviewed_date='2026-10-18')
    with pytest.raises(ValidationError, match='every case'):
        LabelledDataset.model_validate(invalid)


@pytest.mark.parametrize('issue', ['missing_severity', 'missing_likelihood', 'conflicting_evidence'])
def test_missing_and_conflicting_inputs_force_review_without_category(reviewed, issue):
    kb, matrix, labels = reviewed
    _, evidence = first_result(reviewed)
    changed = case_evidence(evidence, issue)
    result = baseline_result(changed, matrix)
    case = next(c for c in labels.cases if c.risk_input_issue == issue)
    prediction = evaluate_predictions(case, result)
    assert prediction['human_review_flag']
    assert prediction['predicted_risk_category'] is None
    assert prediction['predicted_decision'] == 'needs_review'
    assert prediction['metrics']['needs_review_correct'] == 1
    assert all(r['risk_level'] is None for r in result.risk.values())
    assert not any(e['record'].get('risk_conflict') for e in evidence)


@pytest.mark.parametrize('actual,expected_distance', [('low', 1), ('medium', 0), ('high', 1)])
def test_risk_exact_match_and_ordinal_distance(reviewed, actual, expected_distance):
    result, _ = first_result(reviewed)
    result.risk['H-1']['risk_level'] = actual
    prediction = evaluate_predictions(reviewed[2].cases[0], result)
    assert prediction['metrics']['risk_category_exact_match_rate'] == int(actual == 'medium')
    assert prediction['metrics']['ordinal_distance_error'] == expected_distance
    assert prediction['predicted_decision'] == actual


def test_citation_validity_separate_from_expert_support(reviewed):
    kb, matrix, _ = reviewed
    _, evidence = first_result(reviewed)
    for e in evidence:
        if e['record']['id'] == 'M-1': e['record']['expert_support_review'] = 'unsupported'
        if e['record']['id'] == 'T-1': e['record']['expert_support_review'] = 'not_reviewed'
    check = verify_selection(baseline(evidence), evidence, matrix)
    assert check['valid'] and check['citation_link_validity'] == 1.0
    assert check['reviewed_support_rate'] == 0.5 and check['reviewed_support_count'] == 2
    assert any('unsupported' in flag for flag in check['human_review_flags'])
    malicious = baseline(evidence).model_dump()
    malicious['expert_support_review'] = 'supported'
    with pytest.raises(ValidationError): SelectionOutput.model_validate(malicious)
    malicious = baseline(evidence).model_dump()
    malicious['selections'][0]['description'] = 'Invented model prose'
    with pytest.raises(ValidationError): SelectionOutput.model_validate(malicious)


def test_expert_support_requires_attribution():
    raw = json.loads((ROOT / 'data/constrained_knowledge_base.json').read_bytes())
    raw['records'][1]['expert_support_review'] = 'supported'
    with pytest.raises(ValidationError, match='requires reviewed_by'):
        parse_upload(json.dumps(raw).encode())


class TestAdapter:
    __test__ = False

    def __init__(self, backend):
        self.backend, self.calls = backend, []

    def generate_with_settings(self, model, messages, settings, cancel):
        self.calls.append(copy.deepcopy((messages, settings)))
        if self.backend == 'qualcomm':
            raise TimeoutError('TEST-ONLY timeout')
        evidence = json.loads(messages[1]['content'])['evidence']
        return {'output': baseline(evidence).model_dump_json(), 'requested_settings': settings,
                'applied_settings': {**settings, 'seed': 'not supported'} if self.backend == 'uno_q' else dict(settings)}


def test_experiment_all_adapters_failures_settings_counts_and_confusion(reviewed):
    kb, matrix, labels = reviewed
    settings = defaults()
    for backend in ('laptop_local', 'uno_q', 'qualcomm'): settings[backend]['model_id'] = 'TEST-ONLY'
    adapters = {b: TestAdapter(b) for b in ('laptop_local', 'uno_q', 'qualcomm')}
    experiment = run_experiment(labels, kb, matrix, settings, 3, adapter_factory=lambda b, s: adapters[b])
    assert len(experiment['rows']) == 25 * 4 * 3
    summary = {r['backend']: r for r in experiment['summary']}
    assert summary['qualcomm']['failures'] == 75
    assert summary['qualcomm']['risk_category_exact_match_rate']['mean'] == 0
    assert summary['qualcomm']['needs_review_correct']['mean'] == 0
    assert summary['baseline']['risk_category_exact_match_rate']['mean'] == 1
    assert summary['baseline']['ordinal_distance_error']['mean'] == 0
    assert summary['baseline']['confusion_matrix']['needs_review']['needs_review'] == 12
    assert summary['baseline']['confusion_matrix']['insufficient_evidence']['insufficient_evidence'] == 21
    assert sum(sum(r.values()) for r in summary['qualcomm']['confusion_matrix'].values()) == 75
    assert summary['baseline']['confusion_matrix_spread']['needs_review']['needs_review']['mean'] == 4
    assert all(s['case_count'] == 25 and s['label'] == 'preliminary' for s in summary.values())
    assert summary['baseline']['risk_category_exact_match_rate']['case_count'] == 14
    assert summary['qualcomm']['ordinal_distance_error']['mean'] is None
    assert summary['qualcomm']['ordinal_distance_error']['eligible_case_count'] == 14
    assert adapters['laptop_local'].calls == adapters['uno_q'].calls == adapters['qualcomm'].calls
    uno = next(r for r in experiment['rows'] if r['model']['backend'] == 'uno_q')
    assert uno['requested_settings']['seed'] == 42 and uno['applied_settings']['seed'] == 'not supported'
    for s in summary.values():
        for m in s.values():
            if isinstance(m, dict) and 'mean' in m:
                assert {'case_count', 'label', 'stddev', 'min', 'max'} <= m.keys()


def test_unconfigured_backends_and_factory_errors_are_not_dropped(reviewed):
    kb, matrix, labels = reviewed
    labels = labels.model_copy(update={'cases': labels.cases[:1]})
    def broken(backend, settings): raise BackendError('TEST-ONLY factory error')
    result = run_experiment(labels, kb, matrix, defaults(), 3, adapter_factory=broken)
    assert len(result['rows']) == 12 and sum(r['status'] == 'failed' for r in result['rows']) == 9
    assert all(set(r['applied_settings'].values()) == {'not measured'} for r in result['rows'] if r['status'] == 'failed')


@pytest.mark.parametrize('adapter_cls,backend,path', [(ChatAdapter, 'laptop_local', '/v1/chat/completions'), (QualcommAdapter, 'qualcomm', '/v2/chat/completions')])
def test_actual_adapter_settings_reports(adapter_cls, backend, path):
    requested = {'temperature': 0.0, 'seed': 42, 'max_tokens': 1800}
    def handler(request):
        assert request.url.path == path
        return httpx.Response(200, json={'model': 'TEST', 'choices': [{'message': {'content': '{}'}}], 'applied_settings': {'temperature': 0.0, 'max_tokens': 1800}, 'unsupported_settings': ['seed']})
    url = 'http://127.0.0.1:8080' if backend == 'laptop_local' else 'https://aisuite.cirrascale.com'
    adapter = adapter_cls(url, backend, 'TEST-ONLY', transport=httpx.MockTransport(handler))
    response = adapter.generate_with_settings('TEST', [], requested)
    assert response['requested_settings'] == requested
    assert response['applied_settings'] == {'temperature': 0.0, 'seed': 'not supported', 'max_tokens': 1800}


def test_missing_runtime_report_is_not_confirmation():
    adapter = ChatAdapter('http://127.0.0.1:8080', 'laptop_local', transport=httpx.MockTransport(lambda r: httpx.Response(200, json={'model': 'TEST', 'choices': [{'message': {'content': '{}'}}]})))
    response = adapter.generate_with_settings('TEST', [], {'temperature': 0.0, 'seed': 42, 'max_tokens': 1800})
    assert set(response['applied_settings'].values()) == {'not measured'}


def test_uno_passes_only_explicit_remote_settings():
    requested = {'temperature': 0.0, 'seed': 42, 'max_tokens': 1800}
    def handler(request):
        if request.method == 'POST':
            return httpx.Response(202, json={'id': 'test123'})
        return httpx.Response(200, json={'status': 'completed', 'output': '{}',
                                        'applied_settings': {'temperature': 0.0, 'seed': 'not supported', 'max_tokens': 1800}})
    adapter = UnoAdapter('http://127.0.0.1:8090', 'uno_q', 'TEST-ONLY', transport=httpx.MockTransport(handler))
    response = adapter.generate_with_settings('TEST', [], requested)
    assert response['requested_settings'] == requested
    assert response['applied_settings'] == {'temperature': 0.0, 'seed': 'not supported', 'max_tokens': 1800}


def test_explicit_unsupported_setting_retained_after_http_failure():
    adapter = ChatAdapter('http://127.0.0.1:8080', 'laptop_local', transport=httpx.MockTransport(lambda r: httpx.Response(400, json={'unsupported_settings': ['seed']})))
    with pytest.raises(BackendError):
        adapter.generate_with_settings('TEST', [], {'temperature': 0.0, 'seed': 42, 'max_tokens': 1800})
    assert adapter.applied_settings == {'temperature': 'not measured', 'seed': 'not supported', 'max_tokens': 'not measured'}


def test_unresolved_risk_without_review_flag_is_incorrect(reviewed):
    result, evidence = first_result(reviewed)
    case = next(c for c in reviewed[2].cases if c.risk_input_issue == 'missing_severity')
    result = baseline_result(case_evidence(evidence, 'missing_severity'), reviewed[1])
    result.needs_human_review = False
    prediction = evaluate_predictions(case, result)
    assert prediction['metrics']['needs_review_correct'] == 0
    assert prediction['metrics']['decision_accuracy'] == 0
    assert prediction['predicted_decision'] == 'failed'


def test_experiment_reports_keep_failures_and_case_counts(reviewed):
    kb, matrix, labels = reviewed
    labels = labels.model_copy(update={'cases': labels.cases[:1]})
    result = run_experiment(labels, kb, matrix, defaults(), 3)
    exported = experiment_report(result)
    assert len(exported['rows']) == 12
    assert 'preliminary' in json_bytes(exported).decode()
    csv = experiment_csv(result).decode('utf-8-sig')
    assert 'requested_settings' in csv and 'applied_settings' in csv and 'not measured' in csv
    pdf = experiment_pdf_bytes(result)
    text = '\n'.join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
    for token in ('n=1', 'preliminary', 'citation_link_validity', 'reviewed_support_rate', 'failed', 'not measured'):
        assert token in text
    target = ROOT / 'runtime/pdf-qa/experiment-review.pdf'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pdf)
