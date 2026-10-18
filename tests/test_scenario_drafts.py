"""Batch screening and engineer exclusions must not leak rejected claims into risk."""
import copy
import json
import time

import pytest
from streamlit.testing.v1 import AppTest

from backend.configuration import ROOT
from backend.schemas import ScenarioInput, RunRequest, Selection
from backend.scenario_generation import screen_drafts, generate_drafts, recalculate
from backend.services import Services
from backend.orchestration import RunManager
from reporting.exports import report, comparison_rows, csv_bytes


@pytest.fixture
def draft_evidence():
    evidence = []
    for number, severity, likelihood in [(1, 'S3', 'L3'), (2, 'S1', 'L1')]:
        hazard, mitigation, test = f'H{number}', f'M{number}', f'T{number}'
        common = {'illustrative': False, 'review_status': 'approved', 'reviewed_by': 'TEST ENGINEER',
                  'expert_support_review': 'supported'}
        for identity, kind, text in [(hazard, 'hazard', f'Hazard description {number}.'),
                                     (mitigation, 'mitigation', f'Mitigation description {number}.'),
                                     (test, 'acceptance_test', f'Test description {number}.')]:
            record = {**common, 'id': identity, 'title': identity, 'description': text, 'type': kind,
                      'hazard_ids': [] if kind == 'hazard' else [hazard],
                      'mitigation_ids': [mitigation] if kind == 'hazard' else [],
                      'test_ids': [test] if kind == 'hazard' else []}
            if kind == 'hazard':
                record.update(severity=severity, likelihood=likelihood, risk_conflict=False)
            evidence.append({'record': record, 'score': .5, 'retrieval': 'tfidf'})
    return evidence


def draft(evidence, number=1):
    refs = {f'H{number}', f'M{number}', f'T{number}'}
    return {'scenario_id': 'MODEL-ID', 'title': 'AI scenario label',
            'statements': [{'record_id': item['record']['id'], 'text': item['record']['description']}
                           for item in evidence if item['record']['id'] in refs]}


def test_independent_screening_retains_bad_peers_and_rejects_duplicates(draft_evidence, matrix):
    inputs = ScenarioInput(faults=['Camera failure'])
    bad = draft(draft_evidence)
    bad['statements'][0]['text'] = 'Invented guarantee: the robot is always safe.'
    scenarios = screen_drafts([bad, draft(draft_evidence), draft(draft_evidence), {'bad': 'format'}], inputs, draft_evidence, matrix)
    assert [s['system_decision'] for s in scenarios] == ['rejected', 'accepted', 'rejected', 'rejected']
    assert len({s['scenario_id'] for s in scenarios}) == 4
    assert not scenarios[0]['risk_assessments']
    assert 'differs from KB' in scenarios[0]['screening_reasons'][0]


@pytest.mark.parametrize('change', ['unknown_id', 'negation', 'unlinked', 'unsupported', 'out_of_scope', 'repeated_reference'])
def test_rejects_unverified_claims(draft_evidence, matrix, change):
    candidate = draft(draft_evidence)
    inputs = ScenarioInput(faults=['Camera failure'])
    if change == 'unknown_id':
        candidate['statements'][0]['record_id'] = 'invented'
    elif change == 'negation':
        candidate['statements'][0]['text'] = 'Not ' + candidate['statements'][0]['text']
    elif change == 'unlinked':
        candidate['statements'][1] = draft(draft_evidence, 2)['statements'][1]
    elif change == 'unsupported':
        draft_evidence[0]['record']['expert_support_review'] = 'unsupported'
    elif change == 'repeated_reference':
        candidate['statements'].append(copy.deepcopy(candidate['statements'][0]))
    else:
        inputs = ScenarioInput(faults=['unknown fault'])
    assert screen_drafts([candidate], inputs, draft_evidence, matrix)[0]['system_decision'] == 'rejected'


def run_fixture(evidence, matrix):
    matrix = matrix.model_copy(update={'illustrative': False})
    inputs = ScenarioInput(faults=['Camera failure'])
    scenarios = screen_drafts([draft(evidence), draft(evidence, 2)], inputs, evidence, matrix)
    return {'snapshot': {'risk_matrix': matrix.model_dump(), 'evidence': evidence},
            'results': {'laptop_local': {'backend': 'laptop_local', 'model_id': 'TEST', 'status': 'completed',
                                       'generation_mode': 'scenario_drafts', 'scenarios': scenarios}}}


def test_exclusion_recalculates_frozen_risk_and_preserves_original(draft_evidence, matrix, store):
    run = run_fixture(draft_evidence, matrix)
    original = copy.deepcopy(run)
    identity = store.save('run', run)
    initial = recalculate(run)['laptop_local']
    assert initial['risk_level'] == 'high'
    store.review(identity, 'laptop_local:DRAFT-1', 'rejected', 'TEST REVIEWER', 'Exclude first scenario')
    after = recalculate(run, store.reviews(identity))['laptop_local']
    assert after['hazard_ids'] == ['H2'] and after['risk_level'] == 'low'
    assert after['claim_count'] == 3 and after['reviewed_support_rate'] == 1.
    store.review(identity, 'laptop_local:DRAFT-2', 'rejected', 'TEST REVIEWER', 'Exclude remaining scenario')
    empty = recalculate(run, store.reviews(identity))['laptop_local']
    assert empty['risk_level'] is None and empty['risk_score'] is None
    assert empty['status'] == 'no_included_scenarios'
    assert empty['citation_link_validity'] is None and empty['reviewed_support_rate'] is None
    assert run == original and store.get(identity)['payload'] == original
    store.review(identity, 'laptop_local:DRAFT-1', 'accepted', 'TEST REVIEWER', 'Reconsidered exclusion')
    assert recalculate(run, store.reviews(identity))['laptop_local']['risk_level'] == 'high'


def test_missing_inputs_keep_risk_unresolved(draft_evidence, matrix):
    draft_evidence[0]['record']['severity'] = None
    run = run_fixture(draft_evidence, matrix)
    result = recalculate(run)['laptop_local']
    assert result['risk_level'] is None and result['unresolved_hazards'] == ['H1']


def test_reviews_are_scoped_by_backend_and_shared_hazards_remain(draft_evidence, matrix):
    run = run_fixture(draft_evidence, matrix)
    first = run['results']['laptop_local']['scenarios'][0]
    run['results']['laptop_local']['scenarios'].append({**copy.deepcopy(first), 'scenario_id': 'DRAFT-3'})
    run['results']['qualcomm'] = {**copy.deepcopy(run['results']['laptop_local']), 'backend': 'qualcomm'}
    reviews = [{'scenario_id': 'laptop_local:DRAFT-1', 'payload': {'decision': 'rejected'}}]
    result = recalculate(run, reviews)
    assert result['laptop_local']['risk_level'] == 'high'  # Same hazard remains in DRAFT-3.
    assert len(result['laptop_local']['included']) == 2
    assert len(result['qualcomm']['included']) == 3


def test_system_rejection_cannot_be_overridden(draft_evidence, matrix, store):
    run = run_fixture(draft_evidence, matrix)
    run['results']['laptop_local']['scenarios'][0]['system_decision'] = 'rejected'
    identity = store.save('run', run)
    with pytest.raises(ValueError, match='System-rejected'):
        store.review(identity, 'laptop_local:DRAFT-1', 'accepted', 'TEST', 'attempt override')


def test_reports_include_effective_risk_and_rejected_drafts(draft_evidence, matrix):
    import io
    from pypdf import PdfReader
    from reporting.pdf_export import pdf_bytes
    run = run_fixture(draft_evidence, matrix)
    run['id'] = 'DEVELOPMENT-DRAFT-REVIEW-QA-ONLY'
    reviews = [{'scenario_id': 'laptop_local:DRAFT-1', 'payload': {'decision': 'rejected'}}]
    exported = report(run, reviews)
    assert exported['risk_after_engineer_review']['laptop_local']['risk_level'] == 'low'
    assert json.loads(comparison_rows(run, reviews)[0]['effective_risk_after_review'])['included'] == ['laptop_local:DRAFT-2']
    assert b'engineer_excluded' in csv_bytes(comparison_rows(run, reviews))
    pdf = pdf_bytes(exported)
    text = '\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    assert 'Included scenarios and recalculated risk' in text
    assert 'laptop_local:DRAFT-1' in text
    qa_path = ROOT / 'runtime/pdf-qa/scenario-draft-review.pdf'
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_bytes(pdf)


def test_batch_retry_and_partial_success(draft_evidence, matrix):
    class TestAdapter:
        calls = 0
        def generate_scenarios(self, model, messages, settings, cancel):
            self.calls += 1
            if self.calls == 1:
                return 'malformed'
            return json.dumps({'schema_version': '3.0', 'scenarios': [draft(draft_evidence), {'bad': 'format'}]})
    adapter = TestAdapter()
    result = generate_drafts(adapter, 'TEST', ScenarioInput(faults=['Camera failure']), draft_evidence,
                             matrix, {'temperature': 0., 'seed': 42, 'max_tokens': 2000}, 10)
    assert result['status'] == 'completed' and result['valid_scenarios'] == 1
    assert len(result['scenarios']) == 2 and adapter.calls == 2
    assert len(result['attempts']) == 2


def test_incomplete_ai_drafts_recover_complete_kb_group(draft_evidence, matrix):
    class PartialAdapter:
        def generate_scenarios(self, model, messages, settings, cancel):
            evidence = json.loads(messages[1]['content'])['evidence']
            # Simulate Qwen omitting the mitigation and acceptance-test IDs.
            item = evidence[0]
            return json.dumps({'schema_version': '3.0', 'scenarios': [{
                'scenario_id': 'PARTIAL-1', 'title': 'Incomplete draft',
                'statements': [{'record_id': item['record']['id'], 'text': item['record']['description']}],
            }]})

    result = generate_drafts(PartialAdapter(), 'TEST', ScenarioInput(faults=['Camera failure']),
                             draft_evidence, matrix,
                             {'temperature': 0., 'seed': 42, 'max_tokens': 2000}, 10)
    assert result['valid_scenarios'] == 2
    assert result['scenarios'][0]['system_decision'] == 'rejected'
    recovery = result['scenarios'][1]
    assert recovery['origin'] == 'deterministic_kb_recovery'
    assert recovery['system_decision'] == 'accepted'
    assert len({item['scenario_id'] for item in result['scenarios']}) == len(result['scenarios'])


def test_full_rejected_batch_does_not_block_kb_recovery(draft_evidence, matrix):
    class BadAdapter:
        def generate_scenarios(self, *args):
            return json.dumps({'schema_version': '3.0', 'scenarios': [
                {'scenario_id': str(i), 'title': 'Unsupported', 'statements': [
                    {'record_id': 'invented', 'text': 'Invented claim'}]} for i in range(10)]})
    result = generate_drafts(BadAdapter(), 'TEST', ScenarioInput(faults=['Camera failure'], weather='wet'),
                             draft_evidence, matrix, {'temperature': 0., 'seed': 42, 'max_tokens': 4096}, 10)
    assert result['ai_candidate_count'] == 10
    assert result['recovered_candidate_count'] == result['valid_scenarios'] == 2
    assert all(s['system_decision'] == 'rejected' for s in result['scenarios'][:10])
    assert len({s['scenario_id'] for s in result['scenarios']}) == 12


def test_no_evidence_never_calls_model(matrix):
    class NeverAdapter:
        def generate_scenarios(self, *args):
            raise AssertionError('No model request is allowed without evidence')
    result = generate_drafts(NeverAdapter(), 'TEST', ScenarioInput(faults=['Camera failure']), [],
                             matrix, {'temperature': 0., 'seed': 42, 'max_tokens': 4096}, 10)
    assert result['status'] == 'failed' and result['scenarios'] == []
    assert result['generation_latency_ms'] is None


def test_new_pipeline_and_engineer_interface(tmp_path, monkeypatch):
    import ui.dashboard as dashboard
    service = Services(tmp_path)
    from knowledge_base.core import parse_upload
    service.activate_dataset(parse_upload((ROOT / 'data/constrained_knowledge_base.json').read_bytes()))
    class TestAdapter:
        def validate_configuration(self, model):
            return True
        def generate_scenarios(self, model, messages, settings, cancel):
            evidence = json.loads(messages[1]['content'])['evidence']
            refs = {'H-1', 'M-1', 'T-1'}
            statements = [{'record_id': item['record']['id'], 'text': item['record']['description']}
                          for item in evidence if item['record']['id'] in refs]
            return json.dumps({'schema_version': '3.0', 'scenarios': [
                {'scenario_id': 'AI-1', 'title': 'Draft for review', 'statements': statements}, {'invalid': True}]})
    service.manager = RunManager(service.store, lambda backend, settings: TestAdapter())
    identity = service.manager.start(RunRequest(inputs=ScenarioInput(faults=['Camera failure']),
                                    selections=[Selection(backend='laptop_local', model_id='TEST')],
                                    generation_mode='scenario_drafts'), service.dataset(), service.matrix(), service.settings())
    for _ in range(200):
        run = service.manager.get(identity)
        if run['status'] != 'running':
            break
        time.sleep(.01)
    assert run['results']['laptop_local']['valid_scenarios'] == 1
    assert len(run['results']['laptop_local']['scenarios']) == 2
    monkeypatch.setattr(dashboard, 'services', lambda: service)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
    for page in ['Scenario Register', 'Reports and History', 'Risk Analysis', 'Model Comparison']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, (page, app.exception)
    app.sidebar.radio[0].set_value('Scenario Register').run()
    next(w for w in app.text_input if w.label == 'Engineer / reviewer name').set_value('TEST REVIEWER')
    next(w for w in app.selectbox if w.label == 'Engineering decision').set_value('rejected')
    next(w for w in app.button if w.label == 'Save engineering decision').click().run()
    assert not app.exception
    assert recalculate(run, service.store.reviews(identity))['laptop_local']['status'] == 'no_included_scenarios'
    service.manager.pool.shutdown()
