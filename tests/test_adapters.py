import json
import threading

import httpx
import pytest

from models.adapters import BackendError, Cancelled, ChatAdapter, QualcommAdapter, UnoAdapter, content_from_response, discover_files


def mock_transport(handler):
    # HTTP fixtures only; no mock results appear in the shipped app.
    return httpx.MockTransport(handler)


@pytest.mark.parametrize("code,message", [(401, "Authentication"), (403, "Authentication"), (429, "Rate limited"), (500, "HTTP 500"), (302, "HTTP 302")])
def test_cloud_errors(code, message):
    adapter = QualcommAdapter("https://aisuite.cirrascale.com/apis", "qualcomm", "test-only", transport=mock_transport(lambda r: httpx.Response(code)))
    with pytest.raises(BackendError, match=message):
        adapter.list_models()


def test_imagine_catalogue_and_documented_chat():
    requests = []
    def handler(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-only"
        if request.method == "GET":
            return httpx.Response(200, json={"chat": ["mock-model"], "embedding": ["mock-embedding"]})
        payload = json.loads(request.content)
        assert payload["model"] == "mock-model" and payload["stream"] is False and payload['do_sample'] is False
        assert payload['messages'] == [{'role': 'user', 'content': 'test'}]
        return httpx.Response(200, json={"model": "mock-model", "choices": [{"delta": {"content": "{}"}, "finish_reason": "stop"}]})
    adapter = QualcommAdapter("https://aisuite.cirrascale.com/apis", "qualcomm", "test-only", transport=mock_transport(handler))
    assert len(adapter.list_models()) == 2
    assert adapter.generate_scenarios("mock-model", [{"role": "user", "content": "test"}], {"max_tokens": 64, "temperature": 0.2}) == "{}"
    assert requests[0].url.path == "/apis/v2/models"
    assert requests[1].url.path == "/apis/v2/chat/completions"


def test_empty_message_uses_documented_delta_text():
    response = {'model': 'test', 'choices': [{'message': {}, 'delta': {'content': '{"ok":true}'}, 'finish_reason': 'stop'}]}
    assert content_from_response(response, 'test') == '{"ok":true}'


@pytest.mark.parametrize('content', [{'String': '{"ok":true}'}, {'Vec': ['{"ok":', 'true}']}])
def test_imagine_tagged_content_is_decoded(content):
    response = {'model': 'test', 'choices': [{'delta': {'content': content}, 'finish_reason': 'stop'}]}
    assert content_from_response(response, 'test') == '{"ok":true}'


def test_imagine_non_text_union_remains_an_error():
    response = {'model': 'test', 'choices': [{'delta': {'content': {'MultiModal': []}}}]}
    with pytest.raises(BackendError, match="MultiModal.*length.*0"):
        content_from_response(response, 'test')


@pytest.mark.parametrize('choice', [
    {'delta': {'String': '{"ok":true}'}},
    {'message': {'content': {'MultiModal': [{'Text': {'text': '{"ok":true}'}}]}}},
    {'text': '{"ok":true}'},
])
def test_imagine_documented_response_variants(choice):
    response = {'model': 'test', 'choices': [{**choice, 'finish_reason': 'stop'}]}
    assert content_from_response(response, 'test') == '{"ok":true}'


def test_qualcomm_does_not_send_undocumented_seed():
    def handler(request):
        payload = json.loads(request.content)
        assert 'seed' not in payload
        return httpx.Response(200, json={'model': 'test', 'choices': [{'delta': {'content': '{}'}, 'finish_reason': 'stop'}]})
    adapter = QualcommAdapter('https://aisuite.cirrascale.com/apis', 'qualcomm', 'token', transport=mock_transport(handler))
    measured = adapter.generate_with_settings('test', [], {'temperature': 0, 'seed': 42, 'max_tokens': 64})
    assert measured['applied_settings']['seed'] == 'not supported'


@pytest.mark.parametrize("data", [{"data": {}}, {"chat": "bad"}, [], {"chat": [4]}])
def test_malformed_cloud_catalogue(data):
    adapter = QualcommAdapter("https://aisuite.cirrascale.com/apis", "qualcomm", "test-only", transport=mock_transport(lambda r: httpx.Response(200, json=data)))
    with pytest.raises(BackendError):
        adapter.list_models()


def test_missing_auth_fails_before_network():
    adapter = QualcommAdapter("https://aisuite.cirrascale.com/apis", "qualcomm", "")
    with pytest.raises(BackendError, match="authentication token"):
        adapter.list_models()


@pytest.mark.parametrize("adapter_class,backend", [(ChatAdapter, "laptop_local"), (UnoAdapter, "uno_q")])
def test_disconnection(adapter_class, backend):
    def fail(request):
        raise httpx.ConnectError("test-only failure")
    adapter = adapter_class("http://127.0.0.1:8080", backend, "test-only", transport=mock_transport(fail))
    with pytest.raises(BackendError, match="unavailable"):
        adapter.list_models()


def test_timeout():
    def fail(request):
        raise httpx.ReadTimeout("test timeout")
    adapter = ChatAdapter("http://127.0.0.1:8080", "laptop_local", transport=mock_transport(fail))
    with pytest.raises(BackendError, match="timed out"):
        adapter.list_models()


def test_make_adapter_uses_configured_backend_timeout(monkeypatch):
    monkeypatch.setenv("SAFESCENARIO_BACKEND_TIMEOUT_SECONDS", "480")
    from models.adapters import make_adapter

    adapter = make_adapter("laptop_local", {"laptop_local": {"base_url": "http://127.0.0.1:8080"}})
    assert adapter.timeout == 480


def test_backend_timeout_is_bounded(monkeypatch):
    from models.adapters import backend_timeout_seconds

    monkeypatch.setenv("SAFESCENARIO_BACKEND_TIMEOUT_SECONDS", "9999")
    assert backend_timeout_seconds() == 900


def test_unavailable_model():
    adapter = ChatAdapter("http://127.0.0.1:8080", "laptop_local", transport=mock_transport(lambda r: httpx.Response(200, json={"data": []})))
    with pytest.raises(BackendError, match="not available"):
        adapter.validate_configuration("not-installed")


def test_local_cancelled_before_request():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        ChatAdapter("http://127.0.0.1:8080", "laptop_local").generate_scenarios("model", [], {}, cancel)


def test_uno_cancels_remote_job():
    requests = []
    cancel = threading.Event()
    def handler(request):
        requests.append(str(request.url))
        if request.url.path == "/v1/jobs":
            cancel.set()
        return httpx.Response(200, json={"id": "abc123", "status": "waiting"})
    adapter = UnoAdapter("http://127.0.0.1:8090", "uno_q", "test-only", transport=mock_transport(handler))
    with pytest.raises(Cancelled):
        adapter.generate_scenarios("model", [], {}, cancel)
    assert requests[-1].endswith("/v1/jobs/abc123/cancel")


@pytest.mark.parametrize("data", [{}, {"model": "other"}, {"model": "m", "choices": []}, {"model": "m", "choices": [None]}, {"model": "m", "choices": [{"message": {"content": 2}}]}, {"model": "m", "choices": [{"finish_reason": "length"}]}])
def test_inconsistent_response(data):
    with pytest.raises(BackendError):
        content_from_response(data, "m")


def test_file_inventory_is_not_inference_readiness(tmp_path):
    (tmp_path / "test.gguf").write_bytes(b"test-only-file")
    inventory = discover_files([str(tmp_path)])
    assert inventory[0]["status"] == "file_present_runtime_unverified"
    assert inventory[0]["size_bytes"] == 14


def test_reasoning_model_thinking_tags_are_stripped():
    response = {'model': 'test', 'choices': [{'message': {'content': '<think>\ninternal reasoning\n</think>\n{"ok":true}'}, 'finish_reason': 'stop'}]}
    assert content_from_response(response, 'test') == '{"ok":true}'


def test_truncated_thinking_fails_content_validation():
    response = {'model': 'test', 'choices': [{'message': {'content': '<think>\nincomplete reasoning without end tag'}, 'finish_reason': 'stop'}]}
    with pytest.raises(BackendError, match="missing text content"):
        content_from_response(response, 'test')


def test_qualcomm_captures_token_usage_metrics():
    def handler(request):
        return httpx.Response(200, json={
            'model': 'test',
            'choices': [{'message': {'content': '{"ok":true}'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 15, 'completion_tokens': 25, 'total_tokens': 40}
        })
    adapter = QualcommAdapter('https://aisuite.cirrascale.com/apis', 'qualcomm', 'token', transport=mock_transport(handler))
    res = adapter.generate_with_settings('test', [], {'temperature': 0, 'max_tokens': 64})
    assert res['runtime_metrics']['prompt_tokens'] == 15
    assert res['runtime_metrics']['completion_tokens'] == 25
    assert res['runtime_metrics']['total_tokens'] == 40

