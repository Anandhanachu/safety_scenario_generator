import copy

import pytest

from backend.analytics import scenario_register, review_counts, performance_rows, proposal_entries
from reporting.exports import report
from ui.charts import latency_figure, outcome_figure, validation_figure


def run_payload(model_output, status="completed", model="test-model"):
    candidates = copy.deepcopy(model_output["scenarios"])
    for c in candidates:
        c["verification"] = {"valid": True}
    return {"snapshot": {"evidence": []}, "results": {"laptop_local": {"model_id": model, "status": status,
        "scenarios": candidates, "generation_latency_ms": 20 if status == "completed" else 1,
        "total_pipeline_latency_ms": 30 if status == "completed" else 2}}}


def test_latest_decision_counts_and_audit_trail(store, model_output):
    identity = store.save("run", run_payload(model_output))
    assert review_counts(scenario_register(store))["pending_review"] == 1
    store.review(identity, "laptop_local:SCN-1", "accepted", "A", "first")
    store.review(identity, "laptop_local:SCN-1", "rejected", "B", "reconsidered")
    counts = review_counts(scenario_register(store))
    assert counts["proposed"] == 1
    assert counts["accepted"] == 0
    assert counts["rejected"] == 1
    assert len(store.reviews(identity)) == 2


def test_all_history_exceeds_200_runs(store, model_output):
    payload = run_payload(model_output)
    for _ in range(205):
        store.save("run", payload)
    assert len(scenario_register(store)) == 205


def test_distinct_backend_and_run_scopes(store, model_output):
    payload = run_payload(model_output)
    payload["results"]["qualcomm"] = copy.deepcopy(payload["results"]["laptop_local"])
    one, two = store.save("run", payload), store.save("run", payload)
    store.review(one, "laptop_local:SCN-1", "accepted", "A", "")
    store.review(two, "qualcomm:SCN-1", "rejected", "A", "")
    counts = review_counts(scenario_register(store))
    assert (counts["proposed"], counts["accepted"], counts["rejected"], counts["pending_review"]) == (4,1,1,2)


def test_duplicate_ids_never_collapse_proposals(store, model_output):
    payload = run_payload(model_output)
    payload["results"]["laptop_local"]["scenarios"] *= 2
    identity = store.save("run", payload)
    rows = scenario_register(store)
    assert len(rows) == 2 and len({r["review_scope"] for r in rows}) == 2
    store.review(identity, rows[0]["review_scope"], "rejected", "A", "duplicate")
    assert review_counts(scenario_register(store))["rejected"] == 1


def test_invalid_candidates_are_visible_not_automatically_rejected(store):
    payload = {"results":{"laptop_local":{"model_id":"test", "status":"failed", "scenarios":[],
        "candidate_inventory":[{"scenario_id":"BAD-1","title":"Malformed AI proposal"}]}}}
    identity = store.save("run", payload)
    counts = review_counts(scenario_register(store))
    assert counts["proposed"] == 1 and counts["pending_review"] == 1 and counts["rejected"] == 0
    with pytest.raises(ValueError, match="revision"):
        store.review(identity, "laptop_local:BAD-1", "accepted", "A", "")
    store.review(identity, "laptop_local:BAD-1", "rejected", "A", "Malformed")
    assert review_counts(scenario_register(store))["rejected"] == 1


def test_performance_excludes_fast_failures_from_success_latency(store, model_output):
    store.save("run", run_payload(model_output))
    failed = run_payload(model_output, "failed")
    failed["results"]["laptop_local"]["scenarios"] = []
    store.save("run", failed)
    row = performance_rows(store)[0]
    assert row["median_generation_ms"] == 20
    assert row["failed"] == 1 and row["executions"] == 2
    assert row["accepted"] == 0
    figure = latency_figure([row], historical=True)
    assert list(figure.data[0].y) == [20]
    assert list(outcome_figure([row]).data[2].y) == [1]
    assert list(validation_figure([row]).data[0].y) == [1]
