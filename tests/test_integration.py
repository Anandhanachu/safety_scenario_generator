import json
import threading
import time

from fastapi.testclient import TestClient
import pytest
from streamlit.testing.v1 import AppTest

from backend.configuration import ROOT
from backend.orchestration import RunManager
from backend.persistence import Store
from backend.schemas import RunRequest, ScenarioInput, Selection
from backend.services import Services
from evaluation.benchmark import BenchmarkRunner, draft_dataset
from models.adapters import BackendError


def wait_for(manager, identity, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = manager.get(identity)
        if result["status"] != "running":
            return result
        time.sleep(0.01)
    raise AssertionError("Test run did not finish in time")


def factory(model_output, failed_backend=None, malformed=False, block=None):
    class MockAdapter:
        def validate_configuration(self, model_id):
            return True
        def generate_scenarios(self, model_id, messages, generation, cancel):
            if block:
                block.wait(2)
            if model_id == "fail":
                raise BackendError("Mock backend failure")
            return "not-json" if malformed else json.dumps(model_output)
    return lambda backend, settings: MockAdapter()


def test_full_comparison_partial_failure_frozen_snapshot(store, dataset, matrix, settings, model_output):
    manager = RunManager(store, factory(model_output))
    request = RunRequest(inputs=ScenarioInput(), selections=[Selection(backend="laptop_local", model_id="good"), Selection(backend="qualcomm", model_id="fail")])
    identity = manager.start(request, dataset, matrix, settings)
    result = wait_for(manager, identity)
    assert result["results"]["laptop_local"]["status"] == "completed"
    assert result["results"]["qualcomm"]["status"] == "failed"
    assert result["snapshot"]["knowledge_base_version"] == dataset.version
    assert result["snapshot"]["risk_matrix"]["version"] == matrix.version
    assert result["results"]["laptop_local"]["scenarios"][0]["review"]["status"] == "pending_review"
    restored = RunManager(Store(store.path)).get(identity)
    assert restored == result
    manager.pool.shutdown()


def test_invalid_output_preserves_diagnostics_without_raw(store, dataset, matrix, settings, model_output, monkeypatch):
    monkeypatch.setenv("SAFESCENARIO_RETAIN_RAW", "false")
    manager = RunManager(store, factory(model_output, malformed=True))
    identity = manager.start(RunRequest(inputs=ScenarioInput(), selections=[Selection(backend="laptop_local", model_id="test")]), dataset, matrix, settings)
    result = wait_for(manager, identity)["results"]["laptop_local"]
    assert result["status"] == "failed"
    assert "Invalid model JSON" in result["error"]
    assert result["response_bytes"] == 8
    assert "raw_response" not in result
    manager.pool.shutdown()


def test_cooperative_cancel_discards_late_result(store, dataset, matrix, settings, model_output):
    block = threading.Event()
    manager = RunManager(store, factory(model_output, block=block))
    identity = manager.start(RunRequest(inputs=ScenarioInput(), selections=[Selection(backend="laptop_local", model_id="test")]), dataset, matrix, settings)
    assert manager.cancel(identity)["status"] == "cancellation_requested"
    block.set()
    result = wait_for(manager, identity)
    assert result["status"] == "cancelled"
    assert result["results"]["laptop_local"]["status"] == "cancelled"
    assert result["results"]["laptop_local"]["scenarios"] == []
    manager.pool.shutdown()


def test_llm_only_uses_no_prompt_evidence(store, dataset, matrix, settings, model_output):
    seen = []
    class MockAdapter:
        def validate_configuration(self, model_id): pass
        def generate_scenarios(self, model_id, messages, generation, cancel):
            seen.append(json.loads(messages[1]["content"]))
            return json.dumps(model_output)
    manager = RunManager(store, lambda b, s: MockAdapter())
    identity = manager.start(RunRequest(inputs=ScenarioInput(), selections=[Selection(backend="laptop_local", model_id="test")], method="llm_only"), dataset, matrix, settings)
    wait_for(manager, identity)
    assert seen[0]["evidence_snapshot"] == []
    manager.pool.shutdown()


def test_invalid_kb_does_not_replace_active(tmp_path):
    services = Services(tmp_path)
    original = services.dataset().version
    from knowledge_base.core import parse_upload
    with pytest.raises(ValueError):
        services.activate_dataset(parse_upload(b"{}"))
    assert services.dataset().version == original


def test_rules_only_benchmark_actual_empty_reviewed_candidates(store, dataset, matrix, settings):
    manager = RunManager(store)
    runner = BenchmarkRunner(manager, store)
    draft = draft_dataset()
    draft = draft.model_copy(update={"cases": draft.cases[:2]})
    identity = runner.start(draft, [], ["rules_only"], dataset, matrix, settings)
    result = wait_for(runner, identity)
    assert result["status"] == "completed"
    assert len(result["rows"]) == 2
    assert result["rows"][0]["candidate_ids"] == []
    assert store.get(identity)["kind"] == "evaluation"
    runner.pool.shutdown()
    manager.pool.shutdown()


def test_app_api_auth_and_health(monkeypatch):
    from backend.api import app
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    monkeypatch.delenv("SAFESCENARIO_API_TOKEN", raising=False)
    assert client.get("/v1/runs/test").status_code == 503
    monkeypatch.setenv("SAFESCENARIO_API_TOKEN", "test-only")
    assert client.get("/v1/runs/test").status_code == 401


def test_board_auth_body_limits_and_command_allowlist(monkeypatch):
    from uno_q_agent.app import app
    client = TestClient(app)
    monkeypatch.delenv("UNO_Q_AGENT_TOKEN", raising=False)
    assert client.get("/v1/health").status_code == 503
    monkeypatch.setenv("UNO_Q_AGENT_TOKEN", "test-only")
    assert client.get("/v1/health").status_code == 401
    auth = {"Authorization": "Bearer test-only"}
    assert client.post("/v1/hardware-tests", headers=auth, json={"name": "shell", "confirmed": True}).status_code == 422
    assert client.post("/v1/hardware-tests", headers=auth, json={"name": "ping", "confirmed": False}).status_code == 400
    assert client.post("/v1/jobs", headers=auth, content=b"x" * (1024 * 1024 + 1)).status_code == 413
    assert client.get("/v1/jobs/missing", headers=auth).status_code == 404


def test_dashboard_all_pages_and_theme(tmp_path, monkeypatch):
    import ui.dashboard as dashboard
    test_services = Services(tmp_path)
    monkeypatch.setattr(dashboard, "services", lambda: test_services)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    for page in dashboard.PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"{page}: {app.exception}"
    app.sidebar.toggle[0].set_value(True).run()
    assert not app.exception
    assert app.session_state["dark"] is True


def test_dashboard_saved_result_review_and_risk_views(tmp_path, monkeypatch, dataset, matrix, settings, model_output):
    import ui.dashboard as dashboard
    test_services = Services(tmp_path)
    test_services.manager = RunManager(test_services.store, factory(model_output))
    identity = test_services.manager.start(RunRequest(inputs=ScenarioInput(), selections=[Selection(backend="laptop_local", model_id="test")]), dataset, matrix, settings)
    wait_for(test_services.manager, identity)
    monkeypatch.setattr(dashboard, "services", lambda: test_services)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    for page in ("Reports and History", "Risk Analysis", "Model Comparison"):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"{page}: {app.exception}"
    app.sidebar.radio[0].set_value("Reports and History").run()
    next(x for x in app.text_input if x.label == "Reviewer name").set_value("Test Engineer")
    next(x for x in app.button if x.label == "Save human review").click().run()
    assert not app.exception
    assert test_services.store.reviews(identity)[0]["payload"]["reviewer"] == "Test Engineer"
    app.sidebar.radio[0].set_value("Risk Analysis").run()
    next(x for x in app.button if x.label == "Calculate and record assessment").click().run()
    assert not app.exception
    assert test_services.store.list("risk_assessment")[0]["payload"]["assessment"]["risk_score"] is None
    test_services.manager.pool.shutdown()


def test_agent_job_lifecycle_uses_validated_runtime_boundary(monkeypatch, model_output):
    from uno_q_agent import app as agent
    class MockRuntime:
        def validate_configuration(self, model_id):
            assert model_id == "test-only-model"
        def generate_scenarios(self, model_id, messages, generation, cancel):
            return json.dumps(model_output)
    monkeypatch.setenv("UNO_Q_AGENT_TOKEN", "test-only")
    monkeypatch.setattr(agent, "runtime", lambda: MockRuntime())
    client = TestClient(agent.app)
    headers = {"Authorization": "Bearer test-only"}
    response = client.post("/v1/jobs", headers=headers, json={"model_id": "test-only-model", "messages": [{"role": "user", "content": "test-only"}]})
    assert response.status_code == 202
    identity = response.json()["id"]
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = client.get(f"/v1/jobs/{identity}", headers=headers).json()
        if result["status"] == "completed":
            break
        time.sleep(0.01)
    assert result["status"] == "completed"
    assert json.loads(result["output"]) == model_output


def test_register_page_reviews_and_counts_across_models(tmp_path, monkeypatch, dataset, matrix, settings, model_output):
    import ui.dashboard as dashboard
    from backend.analytics import scenario_register, review_counts
    service = Services(tmp_path)
    service.manager = RunManager(service.store, factory(model_output))
    identity = service.manager.start(RunRequest(inputs=ScenarioInput(), selections=[Selection(backend="laptop_local",model_id="test"), Selection(backend="qualcomm",model_id="test")]), dataset,matrix,settings)
    wait_for(service.manager, identity)
    monkeypatch.setattr(dashboard, "services", lambda:service)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    app.sidebar.radio[0].set_value("Scenario Register").run()
    assert not app.exception
    next(x for x in app.text_input if x.label == "Proposal reviewer").set_value("Engineer")
    next(x for x in app.selectbox if x.label == "Proposal decision").set_value("accepted")
    next(x for x in app.button if x.label == "Save proposal decision").click().run()
    assert not app.exception
    counts = review_counts(scenario_register(service.store))
    assert counts["accepted"] == 1 and counts["pending_review"] == 1
    next(x for x in app.selectbox if x.label == "Proposal decision").set_value("rejected")
    next(x for x in app.button if x.label == "Save proposal decision").click().run()
    counts = review_counts(scenario_register(service.store))
    assert counts["accepted"] == 0 and counts["rejected"] == 1
    app.sidebar.radio[0].set_value("Model Comparison").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) >= 6
    service.manager.pool.shutdown()


def test_schema_invalid_proposals_remain_in_register(store, dataset, matrix, settings, model_output):
    from backend.analytics import scenario_register
    invalid = json.loads(json.dumps(model_output))
    invalid["scenarios"][0]["title"] = "Missing mandatory narrative"
    del invalid["scenarios"][0]["scenario_narrative"]
    manager = RunManager(store, factory(invalid))
    identity = manager.start(RunRequest(inputs=ScenarioInput(), selections=[Selection(backend="laptop_local",model_id="test")]), dataset,matrix,settings)
    finished = wait_for(manager, identity)
    assert finished["results"]["laptop_local"]["status"] == "failed"
    rows = scenario_register(store)
    assert len(rows) == 1 and rows[0]["title"] == "Missing mandatory narrative"
    assert not rows[0]["schema_valid"] and rows[0]["decision"] == "pending_review"
    manager.pool.shutdown()


def test_knowledge_search_inspector_returns_actual_matches(tmp_path, monkeypatch):
    import ui.dashboard as dashboard
    service = Services(tmp_path)
    monkeypatch.setattr(dashboard, "services", lambda:service)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    app.sidebar.radio[0].set_value("Knowledge Base").run()
    next(x for x in app.button if x.label == "Search evidence").click().run()
    assert not app.exception
    result = app.session_state["search_inspection"]
    assert result["results"][0]["record"]["id"] == "FLT-LIDAR"
    assert result["diagnostics"]["direct_matches"] == 5
    assert any(m.label == "Measured search total" for m in app.metric)
