"""Validated datasets and deterministic TF-IDF retrieval with relationship expansion."""
import hashlib
import json
from collections import OrderedDict
import threading
import time
from typing import Literal

from pydantic import Field, model_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from backend.schemas import Strict

TYPES = ("operating_condition", "fault", "hazard", "consequence", "evidence", "source", "mitigation", "acceptance_test", "risk_criteria")
REF_TYPES = {"source_ids": {"source", "evidence"}, "mitigation_ids": {"mitigation"}, "test_ids": {"acceptance_test"}}


class Record(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    type: Literal["operating_condition", "fault", "hazard", "consequence", "evidence", "source", "mitigation", "acceptance_test", "risk_criteria"]
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=12000)
    source_ids: list[str] = Field(default_factory=list, max_length=100)
    mitigation_ids: list[str] = Field(default_factory=list, max_length=100)
    test_ids: list[str] = Field(default_factory=list, max_length=100)
    related_ids: list[str] = Field(default_factory=list, max_length=100)
    review_status: Literal["pending_review", "approved", "rejected"] = "pending_review"
    source: str | None = Field(default=None, max_length=2000)
    illustrative: bool = True


class Dataset(Strict):
    schema_version: Literal["1.0"]
    dataset_id: str = Field(min_length=1, max_length=100)
    dataset_name: str = Field(min_length=1, max_length=200)
    records: list[Record] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def references(self):
        index = {r.id: r for r in self.records}
        if len(index) != len(self.records):
            raise ValueError("Duplicate record IDs")
        for r in self.records:
            for field in (*REF_TYPES, "related_ids"):
                for ref in getattr(r, field):
                    if ref not in index:
                        raise ValueError(f"{r.id}: broken reference {ref}")
                    if field in REF_TYPES and index[ref].type not in REF_TYPES[field]:
                        raise ValueError(f"{r.id}: wrong record type for {field}: {ref}")
        return self

    @property
    def version(self):
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:16]


def parse_upload(raw: bytes) -> Dataset:
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError("Knowledge base exceeds 5 MiB")
    # Reject duplicate JSON keys as well as duplicate record IDs.
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return Dataset.model_validate(json.loads(raw, object_pairs_hook=pairs))


RETRIEVAL_VERSION = "tfidf-word-bigram-v2"
_cache, _cache_lock = OrderedDict(), threading.RLock()


def clear_index_cache():
    with _cache_lock:
        _cache.clear()


def query_text(query):
    """Search values, not JSON field names such as `faults` or `weather`."""
    if isinstance(query, str):
        try:
            decoded = json.loads(query)
        except ValueError:
            return query[:16000]
        if not isinstance(decoded, dict):
            return query[:16000]
        query = decoded
    if hasattr(query, "model_dump"):
        query = query.model_dump()
    if isinstance(query, dict):
        def values(value):
            if isinstance(value, str):
                return [value]
            if isinstance(value, (list, tuple)):
                return [part for v in value for part in values(v)]
            if isinstance(value, dict):
                return [part for v in value.values() for part in values(v)]
            return []
        return " ".join(values(query))[:16000]
    raise ValueError("Query must be text or structured operating inputs")


class RetrievalIndex:
    def __init__(self, records):
        start = time.perf_counter()
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                                          strip_accents="unicode", dtype=np.float32, max_features=100000)
        corpus = [f"{r['id'].replace('-', ' ')} {r['title']} {r['description']}" for r in records]
        try:
            self.matrix = self.vectorizer.fit_transform(corpus)
        except ValueError:
            self.matrix = None
        self.build_ms = (time.perf_counter() - start) * 1000


def get_index(dataset):
    serialized = json.dumps(dataset.model_dump(), sort_keys=True, separators=(",", ":"))
    identity = hashlib.sha256(serialized.encode()).hexdigest()
    with _cache_lock:
        reused = identity in _cache
        if not reused:
            _cache[identity] = RetrievalIndex(json.loads(serialized)["records"])
            if len(_cache) > 4:
                _cache.popitem(last=False)
        _cache.move_to_end(identity)
        return _cache[identity], reused


def retrieve_with_diagnostics(dataset, query, limit=12, threshold=0.05):
    if not 1 <= limit <= 40 or not 0 <= threshold <= 1:
        raise ValueError("Retrieval limit must be 1-40 and threshold 0-1")
    start = time.perf_counter()
    index, reused = get_index(dataset)
    search = query_text(query)
    query_start = time.perf_counter()
    scores = [] if index.matrix is None else cosine_similarity(index.vectorizer.transform([search]), index.matrix)[0]
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], index.records[i]["id"]))
    selected = [i for i in ranked if scores[i] > 0 and scores[i] >= threshold][:limit]
    # Return copies: callers cannot mutate a shared cached index or another run's snapshot.
    result = [{"record": json.loads(json.dumps(index.records[i])), "score": round(float(scores[i]), 6), "retrieval": "tfidf"} for i in selected]
    known = {x["record"]["id"] for x in result}
    # References are included even if their lexical score is zero; cap total snapshot size.
    for entry in list(result):
        record = index.by_id[entry["record"]["id"]]
        for ref in record["source_ids"] + record["mitigation_ids"] + record["test_ids"] + record["related_ids"]:
            if ref not in known and len(result) < 100:
                result.append({"record": json.loads(json.dumps(index.by_id[ref])), "score": None, "retrieval": f"relationship:{record['id']}"})
                known.add(ref)
    return result, {"version": RETRIEVAL_VERSION, "cache_reused": reused, "indexed_records": len(index.records),
                    "index_build_ms": round(index.build_ms, 3), "query_ms": round((time.perf_counter()-query_start)*1000, 3),
                    "total_ms": round((time.perf_counter()-start)*1000, 3), "normalized_query": search,
                    "direct_matches": len(selected), "relationship_matches": len(result)-len(selected)}


def retrieve(dataset: Dataset, query, limit=12, threshold=0.05):
    return retrieve_with_diagnostics(dataset, query, limit, threshold)[0]
