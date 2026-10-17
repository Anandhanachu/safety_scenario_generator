import csv
import io
import json


def json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")


def comparison_rows(run):
    rows = []
    for backend, result in run.get("results", {}).items():
        scenarios = result.get("scenarios", [])
        checks = [s["verification"] for s in scenarios]
        rows.append({"backend": backend, "model_id": result["model_id"], "status": result["status"],
                     "schema_valid": result.get("schema_valid", False),
                     'parameter_size': result.get('model_details', {}).get('parameter_size') or 'not measured',
                     'quantization': result.get('model_details', {}).get('quantization') or 'not measured',
                     'confidence': result.get('confidence'), 'needs_human_review': result.get('needs_human_review'),
                     'settings': json.dumps(result.get('constrained', {}).get('settings', result.get('run_settings', {}))),
                     'human_review_flags': json.dumps(result.get('constrained', {}).get('verification', {}).get('human_review_flags', [])),
                     'retrieved_evidence': json.dumps(run.get('snapshot', {}).get('evidence', [])),
                     'verification': json.dumps(result.get('constrained', {}).get('verification', {})),
                     'risk_calculation_steps': json.dumps(result.get('constrained', {}).get('risk', {})),
                     'baseline_comparison': json.dumps(run.get('snapshot', {}).get('baseline_comparison')),
                     'limitations': json.dumps(run.get('snapshot', {}).get('limitations', [])),
                     "generation_latency_ms": result.get("generation_latency_ms"), "total_pipeline_latency_ms": result.get("total_pipeline_latency_ms"),
                     "candidate_scenarios": result.get("declared_candidate_count", len(scenarios)), "valid_scenarios": result.get("valid_scenarios", 0),
                     "invalid_evidence": sum(c["invalid_evidence"] for c in checks), "missing_evidence": sum(c["missing_evidence"] for c in checks),
                     "unsupported_claims_flagged": sum(c["unsupported_claims_flagged"] for c in checks),
                     "duplicates": sum(c["duplicate"] for c in checks), "invalid_mitigation_refs": sum(c["invalid_mitigation_refs"] for c in checks),
                     "invalid_test_refs": sum(c["invalid_test_refs"] for c in checks),
                     "risk_categories": json.dumps([r["risk_level"] for s in scenarios for r in s["risk_assessments"].values()]),
                     "warnings": json.dumps([w for c in checks for w in c["warnings"]]), "error": result.get("error", "")})
    baseline = run.get('snapshot', {}).get('baseline_comparison')
    if baseline:
        count = len((baseline.get('output') or {}).get('selections', []))
        rows.append({'backend': 'baseline', 'model_id': 'TF-IDF + stored risk matrix', 'status': 'completed', 'schema_valid': True,
                     'generation_latency_ms': None, 'total_pipeline_latency_ms': baseline['latency_ms'], 'candidate_scenarios': count,
                     'valid_scenarios': count if baseline['verification']['valid'] else 0, 'confidence': baseline['confidence'],
                     'needs_human_review': baseline['needs_human_review'], 'parameter_size': 'not applicable', 'quantization': 'not applicable',
                     'settings': json.dumps(baseline['settings']), 'retrieved_evidence': json.dumps(baseline['evidence']),
                     'verification': json.dumps(baseline['verification']), 'risk_calculation_steps': json.dumps(baseline['risk']),
                     'human_review_flags': json.dumps(baseline['verification']['human_review_flags']),
                     'baseline_comparison': json.dumps(baseline), 'limitations': json.dumps(baseline['limitations'])})
    return sorted(rows, key=lambda r: r["backend"])


def csv_bytes(rows):
    if not rows:
        return b""
    stream = io.StringIO(newline="")
    fields = sorted({k for r in rows for k in r})
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        row = {k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()}
        # Protect spreadsheets from formula injection in model/user text.
        writer.writerow({k: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else v for k, v in row.items()})
    return stream.getvalue().encode("utf-8-sig")


def report(run, reviews=(), risk_assessments=(), simulations=()):
    from backend.constrained import LIMITATIONS
    from backend.analytics import proposal_entries, review_counts
    reviews = list(reviews)
    latest = {r["scenario_id"]: r["payload"]["decision"] for r in reviews}
    outcomes = [{"decision": latest.get(e["scope"], "pending_review")} for e in proposal_entries(run)]
    return {"report_schema": "1.0", "title": "SafeScenario AI engineering review report",
            "notice": "Unreviewed AI proposals; illustrative records and software simulation are labelled. This report is not certification or formal safety approval.",
            "run": run, 'limitations': LIMITATIONS, 'baseline_comparison': run.get('snapshot', {}).get('baseline_comparison'), "comparison": comparison_rows(run), "human_reviews": reviews, "proposal_review_summary": review_counts(outcomes),
            "human_risk_assessments": list(risk_assessments), "software_simulations": list(simulations)}
