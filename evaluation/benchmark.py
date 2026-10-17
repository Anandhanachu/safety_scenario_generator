"""Explicit ID-based label matching. Draft cases are never represented as reviewed truth."""
from concurrent.futures import ThreadPoolExecutor
import copy
import threading
import time
from uuid import uuid4

from pydantic import Field, model_validator
from backend.schemas import RunRequest, ScenarioInput, Strict
from backend.persistence import now
from knowledge_base.core import retrieve


class Case(Strict):
    id: str = Field(min_length=1, max_length=100)
    inputs: ScenarioInput
    expected_hazard_ids: list[str]
    acceptable_evidence_ids: list[str]
    required_mitigation_ids: list[str]
    expected_review: str = "pending_review"


class BenchmarkDataset(Strict):
    version: str = Field(min_length=1, max_length=100)
    reviewed: bool = False
    reviewer: str = Field(default="", max_length=200)
    cases: list[Case] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def valid(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("Duplicate benchmark case IDs")
        if self.reviewed and not self.reviewer:
            raise ValueError("Reviewed datasets require reviewer attribution")
        return self


def draft_dataset():
    cases = []
    for time_of_day in ("day", "night"):
        for weather in ("dry", "wet", "rain"):
            for fault in ("Camera failure", "Communication loss", "Reduced traction", "Processing delay"):
                hazard = "HAZ-002" if fault == "Communication loss" else "HAZ-001"
                cases.append(Case(id=f"DRAFT-{len(cases)+1:02d}", inputs=ScenarioInput(time_of_day=time_of_day, weather=weather, faults=[fault]),
                                  expected_hazard_ids=[hazard], acceptable_evidence_ids=[hazard, "SRC-001"], required_mitigation_ids=["MIT-001"]))
    return BenchmarkDataset(version="illustrative-draft-24-v1", reviewed=False, cases=cases)


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator, "value": numerator / denominator if denominator else None}


def metrics(case, result, evidence):
    scenarios = result.get("scenarios", [])
    hazards = [h for s in scenarios for h in s["hazards"]]
    found = {h["hazard_id"] for h in hazards} | {i for h in hazards for i in h["evidence_ids"]}
    refs = [i for h in hazards for i in h["evidence_ids"]]
    mitigations = [i for s in scenarios for i in s["mitigation_ids"]]
    tests = [i for s in scenarios for i in s["acceptance_test_ids"]]
    index = {e["record"]["id"]: e["record"] for e in evidence}
    relevant = sum(bool(({h["hazard_id"]} | set(h["evidence_ids"])) & set(case.expected_hazard_ids)) for h in hazards)
    return {"expected_hazard_coverage": ratio(len(set(case.expected_hazard_ids) & found), len(set(case.expected_hazard_ids))),
            "relevant_hazard_count": relevant,
            "valid_evidence_id_rate": ratio(sum(i in index and index[i]["type"] in {"source", "evidence", "hazard", "risk_criteria"} for i in refs), len(refs)),
            "acceptable_evidence_rate": ratio(sum(i in case.acceptable_evidence_ids for i in refs), len(refs)),
            "required_mitigation_coverage": ratio(len(set(mitigations) & set(case.required_mitigation_ids)), len(set(case.required_mitigation_ids))),
            "valid_mitigation_reference_rate": ratio(sum(i in index and index[i]["type"] == "mitigation" for i in mitigations), len(mitigations)),
            "valid_test_reference_rate": ratio(sum(i in index and index[i]["type"] == "acceptance_test" for i in tests), len(tests)),
            "schema_valid_output_rate": ratio(int(result.get("schema_valid", False)), 1),
            "duplicate_scenario_rate": ratio(sum(s["verification"]["duplicate"] for s in scenarios), len(scenarios)),
            "backend_failure_rate": ratio(int(result["status"] == "failed"), 1),
            "unsupported_claim_rate": None, "unsupported_claim_note": "Requires human claim annotation; heuristic flags are not a measured claim rate.",
            "generation_latency_ms": result.get("generation_latency_ms"), "total_pipeline_latency_ms": result.get("total_pipeline_latency_ms"),
            "human_review_outcome": None}


class BenchmarkRunner:
    def __init__(self, manager, store):
        self.manager, self.store = manager, store
        self.jobs, self.lock = {}, threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)

    def start_labelled(self, labels, selections, kb, matrix, settings, repeats=3):
        from evaluation.labelled import evaluate
        from backend.constrained import ModelDetails
        from models.adapters import make_adapter
        labels.validate_kb(kb)
        with self.lock:
            if any(j['status'] == 'running' for j in self.jobs.values()):
                raise ValueError('A benchmark is already running')
            identity = uuid4().hex
            self.jobs[identity] = {'id': identity, 'status': 'running', 'rows': [], 'cancel': threading.Event()}
        def run():
            try:
                models = []
                for selection in selections:
                    cfg = settings[selection.backend]
                    adapter = make_adapter(selection.backend, settings)
                    models.append((ModelDetails(backend=selection.backend, name=selection.model_id,
                                   parameter_size=cfg.get('parameter_size'), quantization=cfg.get('quantization')), adapter))
                result = evaluate(labels, kb, matrix, models, repeats, settings['retrieval_limit'], settings['relevance_threshold'],
                                  settings.get('confidence_threshold', 1.0), self.jobs[identity]['cancel'], settings.get('max_tokens', 1800))
                with self.lock:
                    self.jobs[identity].update(result)
            except Exception:
                with self.lock:
                    self.jobs[identity].update(status='failed', error='Evaluation failed; verify labels, evidence and backend configuration')
            finally:
                with self.lock:
                    self.store.save('evaluation', {k: v for k, v in self.jobs[identity].items() if k != 'cancel'}, identity)
        self.pool.submit(run)
        return identity

    def start(self, dataset, selections, methods, kb, matrix, settings):
        if not methods or any(m not in {"rules_only", "llm_only", "full_pipeline"} for m in methods):
            raise ValueError("Choose a supported benchmark method")
        with self.lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise ValueError("A benchmark is already running")
            identity = uuid4().hex
            self.jobs[identity] = {"id": identity, "status": "running", "created": now(), "dataset": dataset.model_dump(), "rows": [], "cancel": threading.Event()}
            self.pool.submit(self._run, identity, dataset, selections, methods, kb, matrix, copy.deepcopy(settings))
        return identity

    def _run(self, identity, dataset, selections, methods, kb, matrix, settings):
        job = self.jobs[identity]
        try:
            for case in dataset.cases:
                for method in methods:
                    if job["cancel"].is_set():
                        break
                    if method == "rules_only":
                        evidence = retrieve(kb, case.inputs.model_dump_json(), settings["retrieval_limit"], settings["relevance_threshold"])
                        candidates = [e["record"] for e in evidence if e["record"]["type"] == "hazard" and e["record"]["review_status"] == "approved" and not e["record"]["illustrative"]]
                        # Deterministic candidate enumeration, not invented LLM output.
                        found = {c["id"] for c in candidates}
                        row = {"case_id": case.id, "method": method, "backend": "rules", "model_id": None,
                               "candidate_ids": sorted(found), "knowledge_base_version": kb.version,
                               "expected_hazard_coverage": ratio(len(found & set(case.expected_hazard_ids)), len(case.expected_hazard_ids)),
                               "note": "Reviewed non-illustrative hazard retrieval only; narrative/schema/latency metrics not comparable to LLM inference."}
                        with self.lock:
                            job["rows"].append(row)
                        continue
                    request = RunRequest(inputs=case.inputs, selections=selections, method=method, retrieval_limit=settings["retrieval_limit"], relevance_threshold=settings["relevance_threshold"])
                    run_id = self.manager.start(request, kb, matrix, settings)
                    while True:
                        if job["cancel"].is_set():
                            self.manager.cancel(run_id)
                        run = self.manager.get(run_id)
                        if run["status"] != "running":
                            break
                        time.sleep(0.1)
                    with self.lock:
                        for backend, result in run["results"].items():
                            job["rows"].append({"case_id": case.id, "method": method, "backend": backend, "model_id": result["model_id"],
                                                "run_id": run_id, "config_hash": run["snapshot"]["config_hash"],
                                                "metrics": metrics(case, result, run["snapshot"]["evidence"]), "error": result.get("error")})
                if job["cancel"].is_set():
                    break
            with self.lock:
                job["status"] = "cancelled" if job["cancel"].is_set() else "completed"
        except Exception:
            with self.lock:
                job.update(status="failed", error="Benchmark interrupted by a pipeline error; partial rows retained")
        finally:
            with self.lock:
                job["finished"] = now()
                self.store.save("evaluation", {k: v for k, v in job.items() if k != "cancel"}, identity)

    def get(self, identity):
        with self.lock:
            return copy.deepcopy({k: v for k, v in self.jobs[identity].items() if k != "cancel"})

    def cancel(self, identity):
        with self.lock:
            self.jobs[identity]["cancel"].set()
