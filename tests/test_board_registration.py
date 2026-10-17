import pytest
from uno_q_agent import model_manager
from backend.persistence import Store
from models.adapters import BackendError, UnoAdapter
import httpx


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
