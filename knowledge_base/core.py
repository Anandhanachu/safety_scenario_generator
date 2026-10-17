"""Validated datasets and deterministic TF-IDF retrieval with relationship expansion."""
import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def retrieve(dataset: Dataset, query: str, limit=12, threshold=0.05):
    corpus = [f"{r.title} {r.description}" for r in dataset.records]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    except ValueError:
        return []
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], dataset.records[i].id))
    selected = [i for i in ranked if scores[i] > 0 and scores[i] >= threshold][:limit]
    result = [{"record": dataset.records[i].model_dump(), "score": round(float(scores[i]), 6), "retrieval": "tfidf"} for i in selected]
    known = {x["record"]["id"] for x in result}
    index = {r.id: r for r in dataset.records}
    # References are included even if their lexical score is zero; cap total snapshot size.
    for entry in list(result):
        record = index[entry["record"]["id"]]
        for ref in record.source_ids + record.mitigation_ids + record.test_ids + record.related_ids:
            if ref not in known and len(result) < 100:
                result.append({"record": index[ref].model_dump(), "score": None, "retrieval": f"relationship:{record.id}"})
                known.add(ref)
    return result
