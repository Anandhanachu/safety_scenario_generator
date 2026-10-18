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
from knowledge_base.core import parse_upload


@pytest.fixture
def dataset():
    # Test-only expert attribution; production sample stays unreviewed.
    kb = parse_upload((ROOT / "data/constrained_knowledge_base.json").read_bytes())
    for record in kb.records:
        record.reviewed_by = "TEST-ONLY"
        record.review_status = "approved"
        record.illustrative = False
        record.expert_support_review = 'supported'
    return kb


@pytest.fixture
def model_output():
    return {"schema_version": "2.0", "outcome": "selected", "selections": [{"hazard_id": "H-1", "mitigation_ids": ["M-1"], "acceptance_test_ids": ["T-1"]}]}


@pytest.fixture
def matrix():
    from risk_engine.calculator import RiskMatrix
    value = RiskMatrix.model_validate_json((ROOT / "configs/risk_matrix.example.json").read_bytes())
    value.illustrative = False
    return value


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
    request = RunRequest(inputs=ScenarioInput(faults=["camera_failure"]), selections=[Selection(backend="laptop_local", model_id="good"), Selection(backend="qualcomm", model_id="fail")])
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
    identity = manager.start(RunRequest(inputs=ScenarioInput(faults=["camera_failure"]), selections=[Selection(backend="laptop_local", model_id="test")]), dataset, matrix, settings)
    result = wait_for(manager, identity)["results"]["laptop_local"]
    assert result["status"] == "failed"
    assert "needs human review" in result["error"]
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["response_bytes"] == 8
    assert "raw_response" not in result
    manager.pool.shutdown()


def test_cooperative_cancel_discards_late_result(store, dataset, matrix, settings, model_output):
    block = threading.Event()
    manager = RunManager(store, factory(model_output, block=block))
    identity = manager.start(RunRequest(inputs=ScenarioInput(faults=["camera_failure"]), selections=[Selection(backend="laptop_local", model_id="test")]), dataset, matrix, settings)
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
    with pytest.raises(ValueError, match="retired"):
        manager.start(RunRequest(inputs=ScenarioInput(faults=["camera_failure"]), selections=[Selection(backend="laptop_local", model_id="test")], method="llm_only"), dataset, matrix, settings)
    assert seen == []
    manager.pool.shutdown()


def test_invalid_kb_does_not_replace_active(tmp_path):
    services = Services(tmp_path)
    original = services.dataset().version
    from knowledge_base.core import parse_upload
    with pytest.raises(ValueError):
        services.activate_dataset(parse_upload(b"{}"))
    assert services.dataset().version == original


def test_historical_rules_benchmark_uses_reviewed_test_candidates(store, dataset, matrix, settings):
    manager = RunManager(store)
    runner = BenchmarkRunner(manager, store)
    draft = draft_dataset()
    draft = draft.model_copy(update={"cases": draft.cases[:2]})
    identity = runner.start(draft, [], ["rules_only"], dataset, matrix, settings)
    result = wait_for(runner, identity)
    assert result["status"] == "completed"
    assert len(result["rows"]) == 2
    assert result["rows"][0]["candidate_ids"] == ['H-1']
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
    identity = test_services.manager.start(RunRequest(inputs=ScenarioInput(faults=["camera_failure"]), selections=[Selection(backend="laptop_local", model_id="test")]), dataset, matrix, settings)
    wait_for(test_services.manager, identity)
    monkeypatch.setattr(dashboard, "services", lambda: test_services)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    for page in ("Reports and History", "Risk Analysis", "Model Comparison"):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"{page}: {app.exception}"
    app.sidebar.radio[0].set_value("Reports and History").run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Risk Analysis").run()
    assert not any(x.label == 'Calculate and record assessment' for x in app.button)
    assert not test_services.store.list('risk_assessment')
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
    from backend.scenario_generation import recalculate
    service = Services(tmp_path)
    service.manager = RunManager(service.store, factory(model_output))
    identity = service.manager.start(RunRequest(inputs=ScenarioInput(faults=["camera_failure"]), selections=[Selection(backend="laptop_local",model_id="test"), Selection(backend="qualcomm",model_id="test")]), dataset,matrix,settings)
    wait_for(service.manager, identity)
    monkeypatch.setattr(dashboard, "services", lambda:service)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    app.sidebar.radio[0].set_value("Scenario Register").run()
    assert not app.exception
    assert any("Qualcomm Cloud" in tab.label for tab in app.tabs)
    assert any("Laptop Local" in tab.label for tab in app.tabs)
    remove_btn = next((x for x in app.button if "Remove from Risk" in x.label), None)
    if remove_btn:
        remove_btn.click().run()
        assert not app.exception
        run = service.store.get(identity)["payload"]
        recalc = recalculate(run, service.store.reviews(identity))
        if "laptop_local" in recalc:
            assert recalc["laptop_local"]["status"] in ("calculated", "provisional_needs_engineer_review", "no_included_scenarios")
    app.sidebar.radio[0].set_value("Model Comparison").run()
    assert not app.exception
    service.manager.pool.shutdown()


def test_schema_invalid_proposals_remain_in_register(store, dataset, matrix, settings, model_output):
    from backend.analytics import scenario_register
    invalid = json.loads(json.dumps(model_output))
    del invalid["selections"][0]["mitigation_ids"]
    manager = RunManager(store, factory(invalid))
    identity = manager.start(RunRequest(inputs=ScenarioInput(faults=["camera_failure"]), selections=[Selection(backend="laptop_local",model_id="test")]), dataset,matrix,settings)
    finished = wait_for(manager, identity)
    assert finished["results"]["laptop_local"]["status"] == "failed"
    rows = scenario_register(store)
    assert len(rows) == 1 and rows[0]["title"] == "Unverified record selection"
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
    assert {r['record']['type'] for r in result['results']} == {'hazard', 'mitigation', 'acceptance_test'}
    assert 0 < result["diagnostics"]["direct_matches"] <= 5
    assert any(m.label == "Measured search total" for m in app.metric)


def test_labelled_evaluation_completed_and_saved_views(tmp_path, monkeypatch):
    import ui.dashboard as dashboard
    from evaluation.labelled import LabelledDataset
    service = Services(tmp_path)
    labels = LabelledDataset.model_validate_json((ROOT / 'data/sample_cases_5.json').read_bytes())
    from knowledge_base.core import parse_upload
    service.activate_dataset(parse_upload((ROOT / 'data/constrained_knowledge_base.json').read_bytes()))
    identity = service.benchmarks.start_labelled(labels, [], service.dataset(), service.matrix(), service.settings(), 3)
    result = wait_for(service.benchmarks, identity)
    assert result['status'] == 'completed' and len(result['rows']) == 60
    monkeypatch.setattr(dashboard, 'services', lambda: service)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
    app.session_state['active_benchmark'] = identity
    app.sidebar.radio[0].set_value('Evaluation and Benchmarks').run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 4
    app.sidebar.radio[0].set_value('Model Comparison').run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 2
    service.benchmarks.pool.shutdown()
    service.manager.pool.shutdown()


def test_archived_evaluation_does_not_require_new_metadata(tmp_path, monkeypatch):
    import ui.dashboard as dashboard
    service = Services(tmp_path)
    service.store.save('evaluation', {'status': 'completed', 'rows': [], 'summary': [{'backend': 'baseline', 'model': 'legacy', 'executions': 3, 'failures': 0, 'hazards_recall': {'mean': 1.0, 'stddev': 0.0, 'min': 1.0, 'max': 1.0}}]})
    monkeypatch.setattr(dashboard, 'services', lambda: service)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
    app.sidebar.radio[0].set_value('Evaluation and Benchmarks').run()
    assert not app.exception
    assert any(b.label == 'Download archived evaluation JSON' for b in app.get('download_button'))
