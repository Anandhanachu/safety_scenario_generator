import json

import httpx
import pytest

from models.adapters import make_adapter, BackendError
from backend.scenario_generation import draft_messages
from backend.schemas import ScenarioInput


def test_fast_options_and_measured_usage_are_recorded():
    def handler(request):
        payload = json.loads(request.content)
        assert payload['reasoning_effort'] == 'none'
        assert payload['response_format'] == {'type': 'json_object'}
        assert payload['seed'] == 42 and payload['max_tokens'] == 1024
        return httpx.Response(200, json={'model': 'TEST', 'usage': {'prompt_tokens': 200, 'completion_tokens': 50,
                                    'total_tokens': 250, 'completion_tokens_details': {'reasoning_tokens': 0}},
                                    'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]})
    adapter = make_adapter('laptop_local', {'laptop_local': {'base_url': 'http://127.0.0.1:11434'}}, httpx.MockTransport(handler))
    result = adapter.generate_with_settings('TEST', [], {'temperature': 0., 'seed': 42, 'max_tokens': 1024})
    assert result['requested_runtime_options']['reasoning_effort'] == 'none'
    assert result['runtime_metrics']['reasoning_tokens'] == 0
    assert result['applied_settings']['seed'] == 'not measured'


@pytest.mark.parametrize('backend,port,flag', [('laptop_local', 8080, None), ('laptop_local', 11434, False), ('qualcomm', 443, None)])
def test_other_servers_and_explicit_opt_out_do_not_get_ollama_options(backend, port, flag, monkeypatch):
    monkeypatch.setenv('QUALCOMM_API_KEY', 'TEST')
    cfg = {'base_url': f'http://127.0.0.1:{port}' if backend == 'laptop_local' else 'https://aisuite.cirrascale.com/apis'}
    if flag is not None:
        cfg['ollama_fast_mode'] = flag
    def handler(request):
        payload = json.loads(request.content)
        assert 'reasoning_effort' not in payload and 'response_format' not in payload
        return httpx.Response(200, json={'model': 'TEST', 'choices': [{'message': {'content': '{}'}}]})
    adapter = make_adapter(backend, {backend: cfg}, httpx.MockTransport(handler))
    adapter.generate_scenarios('TEST', [], {})


def test_unsupported_options_fail_without_silent_removal_or_retry():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={'error': 'TEST unsupported option'})
    adapter = make_adapter('laptop_local', {'laptop_local': {'base_url': 'http://127.0.0.1:11434'}}, httpx.MockTransport(handler))
    with pytest.raises(BackendError, match='HTTP 400'):
        adapter.generate_scenarios('TEST', [], {})
    assert len(calls) == 1


def test_pipeline_schema_is_sent_as_a_decoder_constraint():
    schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}
    def handler(request):
        payload = json.loads(request.content)
        assert payload['response_format']['type'] == 'json_schema'
        assert payload['response_format']['json_schema']['schema'] == schema
        return httpx.Response(200, json={'model': 'TEST', 'choices': [{'message': {'content': '{"ok":true}'}}]})
    adapter = make_adapter('laptop_local', {'laptop_local': {'base_url': 'http://127.0.0.1:11434'}}, httpx.MockTransport(handler))
    adapter.generate_scenarios('TEST', [{'role': 'user', 'content': json.dumps({'schema': schema})}], {})


def test_compact_prompt_preserves_descriptions_and_links():
    record = {'id': 'H1', 'type': 'hazard', 'title': 'TEST', 'description': 'TEST description',
              'mitigation_ids': ['M1'], 'test_ids': ['T1'], 'severity': 'S3', 'reviewed_by': 'TEST REVIEWER',
              'illustrative': False}
    prompt = draft_messages(ScenarioInput(), [{'record': record, 'score': .9}], 3)
    supplied = json.loads(prompt[1]['content'])
    compact = supplied['evidence'][0]['record']
    assert compact['description'] == record['description']
    assert compact['mitigation_ids'] == ['M1'] and compact['test_ids'] == ['T1']
    assert 'reviewed_by' not in compact and 'severity' not in compact
    assert supplied['max_scenarios'] == 3
    complete = [{'record': record},
                {'record': {'id': 'M1', 'type': 'mitigation', 'description': 'TEST mitigation', 'hazard_ids': ['H1']}},
                {'record': {'id': 'T1', 'type': 'acceptance_test', 'description': 'TEST test', 'hazard_ids': ['H1']}}]
    schema = json.loads(draft_messages(ScenarioInput(faults=['Camera failure']), complete, 1)[1]['content'])['schema']
    assert schema['properties']['scenarios']['minItems'] == 1
    assert set(schema['$defs']['CitedStatement']['properties']['record_id']['enum']) == {'H1', 'M1', 'T1'}
    assert set(schema['$defs']['CitedStatement']['properties']['text']['enum']) == {'TEST description', 'TEST mitigation', 'TEST test'}


def test_repeated_local_runs_are_not_queued_in_the_manager(tmp_path):
    import threading
    from backend.services import Services
    from backend.orchestration import RunManager
    from backend.schemas import RunRequest, Selection
    service = Services(tmp_path)
    gate, started = threading.Event(), threading.Event()
    class TestAdapter:
        def validate_configuration(self, model):
            return True
        def generate_scenarios(self, *args):
            started.set()
            gate.wait(5)
            return '{"schema_version":"2.0","outcome":"insufficient_evidence","selections":[]}'
    manager = RunManager(service.store, lambda backend, settings: TestAdapter())
    request = RunRequest(inputs=ScenarioInput(faults=['Camera failure']), selections=[Selection(backend='laptop_local', model_id='TEST')])
    try:
        manager.start(request, service.dataset(), service.matrix(), service.settings())
        assert started.wait(2)
        with pytest.raises(ValueError, match='already has an active request'):
            manager.start(request, service.dataset(), service.matrix(), service.settings())
    finally:
        gate.set()
        manager.pool.shutdown()
