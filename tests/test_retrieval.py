import copy
import pytest

from backend.configuration import ROOT
from backend.services import Services
from knowledge_base.core import parse_upload, retrieve, retrieve_with_diagnostics, clear_index_cache, query_text
from knowledge_base.benchmark import benchmark_retrieval


@pytest.fixture
def expanded():
    return parse_upload((ROOT / "data/expanded_knowledge_base.json").read_bytes())


def test_expanded_catalogue_coverage(expanded):
    assert len(expanded.records) == 181
    assert len({r.id for r in expanded.records}) == 181
    assert {r.type for r in expanded.records} == {"source","evidence","fault","hazard","consequence","operating_condition","mitigation","acceptance_test","risk_criteria"}
    assert all(r.illustrative and r.review_status == "pending_review" for r in expanded.records)


def test_cache_reuse_and_snapshot_mutation_isolation(expanded):
    clear_index_cache()
    first, d1 = retrieve_with_diagnostics(expanded, "lidar scan dropout")
    second, d2 = retrieve_with_diagnostics(expanded, "lidar scan dropout")
    assert not d1["cache_reused"] and d2["cache_reused"]
    assert first == second
    first[0]["record"]["title"] = "MUTATED"
    assert retrieve(expanded, "lidar scan dropout") == second


def test_cache_invalidates_on_content_change(expanded):
    clear_index_cache()
    retrieve(expanded, "lidar")
    changed = expanded.model_copy(deep=True)
    changed.records[0].description += " updated source assumption"
    _, diagnostics = retrieve_with_diagnostics(changed, "lidar")
    assert not diagnostics["cache_reused"]
    assert expanded.version != changed.version


def test_structured_query_avoids_json_key_pollution():
    assert query_text('{"faults":["lidar scan loss"],"weather":"rain","speed_mps":1}') == "lidar scan loss rain"


def test_no_overlap_and_threshold_boundaries(expanded):
    assert retrieve(expanded, "qqqunrelatedxyz") == []
    result, diagnostics = retrieve_with_diagnostics(expanded, "lidar", limit=1)
    assert diagnostics["direct_matches"] == 1
    assert len(result) >= 1
    with pytest.raises(ValueError):
        retrieve(expanded, "lidar", limit=0)


def test_lexical_retrieval_benchmark_is_measured(expanded):
    result = benchmark_retrieval(expanded, repeats=1)
    assert result["case_count"] == 16
    assert result["hit_at_5"] >= 0.9
    assert result["mean_reciprocal_rank"] >= 0.5
    assert result["warm_p95_total_ms"] >= result["warm_p50_total_ms"] >= 0
    assert all(len(r["direct_top5"]) == 5 for r in result["cases"])


def test_custom_kb_is_preserved_and_starter_is_upgraded(tmp_path, dataset):
    service = Services(tmp_path)
    assert len(service.dataset().records) == 216
    service.activate_dataset(dataset)
    assert len(Services(tmp_path).dataset().records) == 216
    custom = dataset.model_copy(deep=True)
    custom.dataset_name = "User edited dataset"
    service.activate_dataset(custom)
    assert Services(tmp_path).dataset().dataset_name == "User edited dataset"


def test_empty_vocabulary_and_deterministic_ties(dataset):
    empty = dataset.model_copy(deep=True)
    for record in empty.records:
        record.id = "a" + str(empty.records.index(record))
        record.title = "the"
        record.description = "and"
    # All query terms absent, even when IDs supply vocabulary.
    assert retrieve(empty, "qzzunrelated") == []
