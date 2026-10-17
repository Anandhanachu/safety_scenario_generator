import copy
import json

import pytest
from pydantic import ValidationError

from backend.configuration import validate_url
from backend.schemas import ModelOutput, ScenarioInput
from evaluation.benchmark import draft_dataset, metrics
from knowledge_base.core import Dataset, parse_upload, retrieve
from reporting.exports import csv_bytes, json_bytes, report
from risk_engine.calculator import Classification, calculate
from simulation.engine import SimulationInput, simulate
from verification.pipeline import verify


def test_valid_upload_and_version(dataset):
    assert len(dataset.records) == 9
    assert dataset.version == Dataset.model_validate(dataset.model_dump()).version


@pytest.mark.parametrize("raw", [b"no JSON", b'{"x":1,"x":2}', b'{}', b'[]', b'"text"'])
def test_invalid_upload(raw):
    with pytest.raises((ValueError, ValidationError)):
        parse_upload(raw)


def test_upload_size_limit():
    with pytest.raises(ValueError, match="5 MiB"):
        parse_upload(b" " * (5 * 1024 * 1024 + 1))


@pytest.mark.parametrize("mutation", ["duplicate", "broken", "wrong_type", "unknown_field"])
def test_kb_schema_relationships(dataset, mutation):
    raw = dataset.model_dump()
    if mutation == "duplicate":
        raw["records"].append(raw["records"][0])
    elif mutation == "broken":
        raw["records"][0]["source_ids"] = ["NOT-EXISTS"]
    elif mutation == "wrong_type":
        raw["records"][0]["mitigation_ids"] = ["SRC-001"]
    else:
        raw["records"][0]["exec"] = "command"
    with pytest.raises(ValidationError):
        Dataset.model_validate(raw)


def test_retrieval_includes_references(dataset):
    results = retrieve(dataset, "camera failure", limit=2)
    ids = {r["record"]["id"] for r in results}
    assert "FLT-CAMERA" in ids
    assert "HAZ-001" in ids
    assert results == retrieve(dataset, "camera failure", limit=2)


def test_retrieval_no_evidence(dataset):
    assert retrieve(dataset, "zzzzunrelatedxyz") == []


@pytest.mark.parametrize("kwargs", [{"speed_mps": -1}, {"speed_mps": float("nan")}, {"faults": ["a"] * 6}, {"faults": ["a", "a"]}, {"notes": "x" * 4001}])
def test_input_validation(kwargs):
    with pytest.raises(ValidationError):
        ScenarioInput(**kwargs)


def approved_evidence():
    return [{"record": {"id": "EV-REAL", "type": "evidence", "review_status": "approved", "illustrative": False}}]


def test_risk_is_ordinal_and_illustrative(matrix):
    c = Classification(severity="S3", likelihood="L3", evidence_ids=["EV-REAL"], reviewer="Engineer", rationale="Documented project exposure criteria")
    result = calculate(matrix, c, approved_evidence())
    assert result["risk_score"] == 9
    assert result["risk_level"] == "high"
    assert result["status"] == "requires_review"
    assert "Ordinal" in result["explanation"]


@pytest.mark.parametrize("c", [None, Classification(severity="S3"), Classification(severity="S3", likelihood="L9"), Classification(severity="S3", likelihood="L3", reviewer="A", rationale="B", evidence_ids=["missing"])])
def test_insufficient_risk_returns_null(matrix, c):
    result = calculate(matrix, c, approved_evidence())
    assert result["risk_score"] is None
    assert result["status"] == "requires_review"


def test_unreviewed_evidence_never_supports_risk(matrix, dataset):
    c = Classification(severity="S3", likelihood="L3", reviewer="A", rationale="B", evidence_ids=["SRC-001"])
    assert calculate(matrix, c, [{"record": r.model_dump()} for r in dataset.records])["risk_score"] is None


def test_verification_missing_and_invalid_references(model_output, matrix, dataset):
    candidate = model_output["scenarios"][0]
    candidate["hazards"][0]["evidence_ids"] = ["invented"]
    candidate["mitigation_ids"] = ["missing"]
    candidate["acceptance_test_ids"] = ["missing"]
    result = verify(ModelOutput.model_validate(model_output), ScenarioInput(), [{"record": r.model_dump()} for r in dataset.records], matrix)[0]
    assert not result["verification"]["valid"]
    assert result["verification"]["invalid_evidence"] == 1
    assert result["verification"]["invalid_mitigation_refs"] == 1
    assert result["verification"]["invalid_test_refs"] == 1
    assert result["risk_assessments"]["HAZ-001"]["risk_score"] is None


def test_missing_evidence_duplicate_and_input_conflict(model_output, matrix):
    candidate = model_output["scenarios"][0]
    candidate["hazards"][0].update(evidence_ids=[], hypothesis=False)
    candidate["input_conditions"]["weather"] = "rain"
    model_output["scenarios"].append(copy.deepcopy(candidate))
    result = verify(ModelOutput.model_validate(model_output), ScenarioInput(), [], matrix)
    assert result[0]["verification"]["missing_evidence"] == 1
    assert result[0]["verification"]["unsupported_claims_flagged"] == 1
    assert "contradict" in result[0]["verification"]["errors"][0]
    assert result[1]["verification"]["duplicate"]


def test_model_cannot_set_risk_or_approval(model_output):
    model_output["scenarios"][0]["risk_score"] = 2
    with pytest.raises(ValidationError):
        ModelOutput.model_validate(model_output)


@pytest.mark.parametrize("inputs,outcome", [
    (SimulationInput(sensor_age_ms=201), "pass"),
    (SimulationInput(sensor_age_ms=200), "fail"),
    (SimulationInput(sensor_age_ms=201, watchdog_enabled=False), "fail"),
    (SimulationInput(communication_ok=False), "pass"),
    (SimulationInput(rule="unsupported"), "inconclusive")])
def test_simulation(inputs, outcome):
    assert simulate(inputs)["outcome"] == outcome
    assert simulate(inputs) == simulate(inputs)


def test_persistence_and_secret_rejection(store):
    identity = store.save("run", {"results": {}})
    assert store.get(identity)["payload"] == {"results": {}}
    with pytest.raises(ValueError):
        store.set_setting("preferences", {"nested": {"api_key": "secret"}})
    store.set_setting("preferences", {"generation": {"max_tokens": 100}})
    assert store.setting("preferences")["generation"]["max_tokens"] == 100


def test_reviews_separate_and_scoped(store):
    identity = store.save("run", {"results": {"laptop_local": {"scenarios": [{"scenario_id": "S1"}]}}})
    store.review(identity, "laptop_local:S1", "accepted", "Engineer", "reviewed links")
    assert store.reviews(identity)[0]["payload"]["decision"] == "accepted"
    assert "review" not in store.get(identity)["payload"]["results"]["laptop_local"]["scenarios"][0]
    with pytest.raises(ValueError):
        store.review(identity, "qualcomm:S1", "accepted", "Engineer", "")


def test_deterministic_exports_and_formula_escape():
    run = {"results": {}}
    assert json_bytes(report(run)) == json_bytes(report(run))
    assert "'=CMD" in csv_bytes([{"text": "=CMD"}]).decode("utf-8-sig")


@pytest.mark.parametrize("url,kind", [("http://example.com", "qualcomm"), ("https://evil.example", "qualcomm"), ("http://user:pass@127.0.0.1", "laptop_local"), ("http://0.0.0.0", "uno_q"), ("http://169.254.169.254", "uno_q"), ("http://8.8.8.8", "laptop_local")])
def test_url_restrictions(url, kind):
    with pytest.raises(ValueError):
        validate_url(url, kind)


def test_evaluation_draft_and_null_metrics():
    dataset = draft_dataset()
    assert len(dataset.cases) == 24
    assert not dataset.reviewed
    result = metrics(dataset.cases[0], {"status": "failed", "scenarios": [], "schema_valid": False}, [])
    assert result["expected_hazard_coverage"]["value"] == 0
    assert result["valid_evidence_id_rate"]["value"] is None
    assert result["unsupported_claim_rate"] is None
