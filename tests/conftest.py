import json
import pytest

from backend.configuration import ROOT, defaults
from backend.persistence import Store
from backend.schemas import ScenarioInput
from knowledge_base.core import parse_upload
from risk_engine.calculator import RiskMatrix


@pytest.fixture
def dataset():
    return parse_upload((ROOT / "data/sample_knowledge_base.json").read_bytes())


@pytest.fixture
def matrix():
    return RiskMatrix.model_validate_json((ROOT / "configs/risk_matrix.example.json").read_text())


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.sqlite")


@pytest.fixture
def settings():
    return defaults()


@pytest.fixture
def model_output():
    # Synthetic output is used ONLY in automated tests.
    return {"schema_version": "1.0", "scenarios": [{"scenario_id": "SCN-1", "title": "Test sensing failure",
        "input_conditions": ScenarioInput().model_dump(), "scenario_narrative": "A camera fault may prevent obstacle recognition.",
        "hazards": [{"hazard_id": "HAZ-001", "description": "Possible pedestrian contact", "consequences": ["Injury"], "evidence_ids": ["SRC-001"], "hypothesis": True}],
        "mitigation_ids": ["MIT-001"], "acceptance_test_ids": ["TEST-001"]}]}
