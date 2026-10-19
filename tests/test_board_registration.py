import pytest
from uno_q_agent import model_manager
from backend.persistence import Store
from models.adapters import BackendError, UnoAdapter
import httpx


def test_board_health_check_does_not_run_inference(monkeypatch):
    import requests
    calls = []

    class Response:
        status_code = 200
        def json(self):
            return {'status': 'ok'}

    def health(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(requests, 'get', health)
    monkeypatch.setattr(requests, 'post', lambda *a, **k: pytest.fail('Health must not run inference'))
    assert UnoAdapter('http://10.85.3.137:5000', 'uno_q').ping_check()
    assert calls == ['http://10.85.3.137:5000/health']


def test_board_prediction_uses_configured_timeout(monkeypatch):
    import requests

    class Response:
        status_code = 200
        def json(self):
            return {'model_reply': 'OFF', 'output': 0}

    def predict(url, **kwargs):
        assert url == 'http://10.85.3.137:5000/predict'
        assert kwargs['timeout'] == (5.0, 120)
        return Response()

    monkeypatch.setattr(requests, 'post', predict)
    assert UnoAdapter('http://10.85.3.137:5000', 'uno_q', timeout=120).predict('turn off')['output'] == 0


def test_board_file_registration_requires_permitted_path(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "defaults", lambda: {"model_dirs": [str(tmp_path)]})
    registry = Store(tmp_path / "test-agent.sqlite")
    monkeypatch.setattr(model_manager, "store", lambda: registry)
    file = tmp_path / "example.gguf"
    file.write_bytes(b"test-only")
    assert model_manager.register_file(str(file))["inference_ready"] is False
    assert len(model_manager.registered_files()) == 1
    with pytest.raises(ValueError):
        model_manager.register_file(str(tmp_path.parent / "escape.gguf"))


def test_board_registered_model_cannot_become_ready_from_file_alone():
    adapter = UnoAdapter("http://127.0.0.1:8090", "uno_q", "test-only", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"models": [{"id": "/models/test.gguf", "inference_ready": False}]})))
    with pytest.raises(BackendError, match="not confirmed"):
        adapter.validate_configuration("/models/test.gguf")
