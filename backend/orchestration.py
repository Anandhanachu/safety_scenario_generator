"""Frozen run snapshots and independent parallel executions with cooperative cancellation."""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import threading
import time
from uuid import uuid4

from pydantic import ValidationError

from backend.persistence import now
from knowledge_base.core import retrieve_with_diagnostics, RETRIEVAL_VERSION
from models.adapters import BackendError, Cancelled, make_adapter
from backend.constrained import execute, ModelDetails, scenario_views, messages, LIMITATIONS
from evaluation.labelled import baseline_result
from backend.scenario_generation import generate_drafts

PROMPT_VERSION = "2.0-record-selection"


def prompt_messages(inputs, evidence):
    return messages(inputs, evidence)


class RunManager:
    def __init__(self, store, adapter_factory=make_adapter):
        self.store, self.adapter_factory = store, adapter_factory
        self.jobs, self.lock = {}, threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="scenario")

    def start(self, request, dataset, matrix, settings):
        query = json.dumps(request.inputs.model_dump())
        evidence, retrieval_diagnostics = retrieve_with_diagnostics(dataset, query, request.retrieval_limit, request.relevance_threshold)
        if dataset.schema_version != '2.0':
            raise ValueError('Activate a validated v2 tagged knowledge base before constrained generation. Existing legacy history is retained.')
        if request.method != 'full_pipeline':
            raise ValueError('Constrained generation requires the shared retrieved evidence; llm_only is retired')
        deterministic = baseline_result(evidence, matrix, request.citation_link_validity_threshold, request.selections[0].generation.max_tokens).model_dump()
        deterministic['latency_ms'] += retrieval_diagnostics['total_ms']
        snapshot = {"schema_version": "1.0", "created": now(), "request": request.model_dump(),
                    "knowledge_base_version": dataset.version, "knowledge_base": dataset.model_dump(),
                    "risk_matrix": matrix.model_dump(), "prompt_version": '3.0-scenario-drafts' if request.generation_mode == 'scenario_drafts' else PROMPT_VERSION,
                    "retrieval": {"method": RETRIEVAL_VERSION, "limit": request.retrieval_limit, "threshold": request.relevance_threshold, "diagnostics": retrieval_diagnostics},
                    "evidence": evidence, 'baseline_comparison': deterministic, 'limitations': LIMITATIONS,
                    "backend_endpoints": {s.backend: settings[s.backend]["base_url"] for s in request.selections},
                    "rules_candidates": [e["record"]["id"] for e in evidence if e["record"]["type"] == "hazard" and e["record"]["review_status"] == "approved" and not e["record"]["illustrative"]]}
        snapshot["config_hash"] = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
        job_id = uuid4().hex
        job = {"id": job_id, "snapshot": snapshot, "results": {}, "progress": {}, "status": "running", "cancel": threading.Event()}
        with self.lock:
            if any(selection.backend == 'laptop_local' for selection in request.selections) and any(
                job['status'] == 'running' and 'laptop_local' in job['progress'] and 'laptop_local' not in job['results']
                for job in self.jobs.values()
            ):
                raise ValueError('The laptop backend already has an active request. Wait for it to finish before starting another; duplicate requests queue inside Ollama and increase waiting time.')
            if sum(j["status"] == "running" for j in self.jobs.values()) >= 4:
                raise ValueError("Four runs already active; wait or cancel one")
            if len(self.jobs) >= 100:
                finished = next((i for i, j in self.jobs.items() if j["status"] != "running"), None)
                if finished:
                    del self.jobs[finished]
            self.jobs[job_id] = job
            for selection in request.selections:
                job["progress"][selection.backend] = [{"stage": "waiting", "at": now()}]
            for selection in request.selections:
                self.pool.submit(self._execute, job_id, selection, request.inputs, matrix, copy.deepcopy(settings))
        return job_id

    def _stage(self, identity, backend, stage):
        with self.lock:
            self.jobs[identity]["progress"][backend].append({"stage": stage, "at": now()})

    def _execute(self, identity, selection, inputs, matrix, settings):
        start = time.perf_counter()
        job = self.jobs[identity]
        result = {"backend": selection.backend, "model_id": selection.model_id, "status": "failed", "scenarios": [],
                  "generation_latency_ms": None, "schema_valid": False, 'needs_human_review': True,
                  'model_details': {'backend': selection.backend, 'name': selection.model_id,
                                    'parameter_size': settings[selection.backend].get('parameter_size'),
                                    'quantization': settings[selection.backend].get('quantization')},
                  'run_settings': selection.generation.model_dump()}
        adapter = None
        try:
            if job["cancel"].is_set():
                raise Cancelled("Cancelled before readiness check")
            self._stage(identity, selection.backend, "checking_readiness")
            adapter = self.adapter_factory(selection.backend, settings)
            adapter.validate_configuration(selection.model_id)
            self._stage(identity, selection.backend, "running_inference")
            cfg = settings[selection.backend]
            if job['snapshot']['request'].get('generation_mode') == 'scenario_drafts':
                result.update(generate_drafts(adapter, selection.model_id, inputs, job['snapshot']['evidence'], matrix,
                                             selection.generation.model_dump(), job['snapshot']['request']['max_scenarios'],
                                             job['snapshot']['request']['citation_link_validity_threshold'], job['cancel']))
            else:
                self._execute_selection(result, adapter, selection, cfg, inputs, job, matrix)
            self._stage(identity, selection.backend, 'calculating_risk')
        except Cancelled as e:
            result.update(status="cancelled", error=str(e))
        except ValidationError as e:
            result["error"] = "Invalid model schema: " + json.dumps(e.errors(include_input=False, include_url=False))[:8000]
        except BackendError as e:
            result["error"] = str(e)
        except Exception:
            result["error"] = "Unexpected pipeline error; no successful result recorded"
        result['generation_mode'] = job['snapshot']['request'].get('generation_mode', 'record_selection')
        if adapter is not None:
            result.setdefault('requested_runtime_options', dict(getattr(adapter, 'requested_runtime_options', {})))
            result.setdefault('runtime_metrics', dict(getattr(adapter, 'runtime_metrics', {})))
        result["total_pipeline_latency_ms"] = round((time.perf_counter() - start) * 1000 + job['snapshot']['retrieval']['diagnostics']['total_ms'], 3)
        result["finished"] = now()
        with self.lock:
            self._stage(identity, selection.backend, result["status"])
            job["results"][selection.backend] = result
            if len(job["results"]) == len(job["snapshot"]["request"]["selections"]):
                job["status"] = "cancelled" if job["cancel"].is_set() else "completed"
                job["finished"] = now()
                payload = {k: v for k, v in job.items() if k != "cancel"}
                try:
                    self.store.save("run", payload, identity)
                except Exception:
                    job["persistence_error"] = "Run finished but could not be saved to SQLite"

    def _execute_selection(self, result, adapter, selection, cfg, inputs, job, matrix):
        measured = execute(adapter, ModelDetails(backend=selection.backend, name=selection.model_id,
                           parameter_size=cfg.get('parameter_size'), quantization=cfg.get('quantization')),
                           inputs, job['snapshot']['evidence'], matrix, selection.generation.max_tokens,
                           job['snapshot']['request']['citation_link_validity_threshold'], job['cancel'])
        result.update(constrained=measured.model_dump(), model_details=measured.model.model_dump(),
                      generation_latency_ms=measured.latency_ms, schema_valid=measured.output is not None,
                      scenarios=scenario_views(measured, inputs), needs_human_review=measured.needs_human_review,
                      citation_link_validity=measured.citation_link_validity, attempts=measured.attempts)
        result['reviewed_support_rate'] = measured.reviewed_support_rate
        result['requested_settings'] = measured.requested_settings
        result['applied_settings'] = measured.applied_settings
        result['requested_runtime_options'] = measured.requested_runtime_options
        result['runtime_metrics'] = measured.runtime_metrics
        result['candidate_inventory'] = measured.attempts[-1].get('candidate_inventory', [])
        result['declared_candidate_count'] = len(result['scenarios']) or len(result['candidate_inventory'])
        result['status'] = 'completed' if measured.output is not None else 'failed'
        result['valid_scenarios'] = sum(s['verification']['valid'] for s in result['scenarios'])
        if measured.output is None:
            result['error'] = 'needs human review: verification failed after one corrective retry; output not accepted'

    def get(self, identity):
        with self.lock:
            if identity in self.jobs:
                return copy.deepcopy({k: v for k, v in self.jobs[identity].items() if k != "cancel"})
        return self.store.get(identity)["payload"]

    def cancel(self, identity):
        with self.lock:
            job = self.jobs[identity]
            if job["status"] == "running":
                job["cancel"].set()
                return {"status": "cancellation_requested", "note": "In-flight requests can continue until timeout; late results are discarded."}
        return {"status": "already_finished"}
