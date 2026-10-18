"""Latest human outcomes and complete persisted proposal inventory, independent of UI."""
from collections import Counter, defaultdict
from statistics import median

DECISIONS = ("accepted", "rejected", "pending_review", "revision_requested", "evidence_insufficient", "additional_testing")


def proposal_entries(run):
    """Keep duplicate AI IDs distinct without breaking historical review keys."""
    entries = []
    for backend, result in run.get("results", {}).items():
        candidates = result.get("scenarios", []) or result.get("candidate_inventory", [])
        ids = [str(c.get("scenario_id", f"candidate-{i+1}")) for i, c in enumerate(candidates)]
        frequencies = Counter(ids)
        used_scopes = set()
        for i, candidate in enumerate(candidates):
            identity = ids[i]
            scope = f"{backend}:{identity}" + (f"#candidate-{i+1}" if frequencies[identity] > 1 else "")
            while scope in used_scopes:
                scope += f"#candidate-{i+1}"
            used_scopes.add(scope)
            entries.append({"scope": scope, "backend": backend, "model_id": result.get("model_id", "unrecorded"), "candidate": candidate,
                            "schema_valid": candidate.get('schema_valid', bool(result.get("scenarios"))), "candidate_index": i})
    return entries


def scenario_register(store, run_id=None):
    reviews = store.latest_reviews(run_id)
    runs = [store.get(run_id)] if run_id else store.iter_snapshots("run")
    rows = []
    for saved in runs:
        run = saved["payload"]
        for entry in proposal_entries(run):
            candidate = entry["candidate"]
            review = reviews.get((saved["id"], entry["scope"]))
            rows.append({"run_id": saved["id"], "review_scope": entry["scope"], "scenario_id": candidate.get("scenario_id", ""),
                         "title": candidate.get("title", "Untitled proposal"), "backend": entry["backend"], "model_id": entry["model_id"],
                         "narrative": candidate.get("scenario_narrative", "Unavailable: schema-invalid candidate inventory only"),
                         "created": saved["created"], "decision": review["payload"]["decision"] if review else "pending_review",
                         "reviewer": review["payload"]["reviewer"] if review else "", "reviewed_at": review["created"] if review else None,
                         "notes": review["payload"]["notes"] if review else "", "schema_valid": entry["schema_valid"],
                         "system_decision": candidate.get('system_decision', 'not recorded'),
                         "validation_passed": bool(candidate.get("verification", {}).get("valid", False))})
    return rows


def review_counts(rows):
    counts = Counter(r["decision"] for r in rows)
    return {"proposed": len(rows), **{d: counts[d] for d in DECISIONS}, "reviewed": sum(counts[d] for d in DECISIONS if d != "pending_review")}


def performance_rows(store):
    grouped, proposals = defaultdict(list), defaultdict(list)
    for row in scenario_register(store):
        proposals[(row["backend"], row["model_id"])].append(row)
    for saved in store.iter_snapshots("run"):
        for backend, result in saved["payload"].get("results", {}).items():
            grouped[(backend, result["model_id"])].append(result)
    rows = []
    for (backend, model), executions in sorted(grouped.items()):
        completed = [r for r in executions if r["status"] == "completed"]
        latencies = [r["generation_latency_ms"] for r in completed if r.get("generation_latency_ms") is not None]
        pipelines = [r["total_pipeline_latency_ms"] for r in completed if r.get("total_pipeline_latency_ms") is not None]
        counts = review_counts(proposals[(backend, model)])
        rows.append({"backend": backend, "model_id": model, "executions": len(executions), "completed": len(completed),
                     "failed": sum(r["status"] == "failed" for r in executions), "cancelled": sum(r["status"] == "cancelled" for r in executions),
                     "median_generation_ms": median(latencies) if latencies else None, "median_pipeline_ms": median(pipelines) if pipelines else None,
                     "validation_passed": sum(r["validation_passed"] for r in proposals[(backend, model)]), **counts})
    return rows
