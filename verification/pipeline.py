from difflib import SequenceMatcher
import re

from risk_engine.calculator import calculate


def verify(output, inputs, evidence, matrix):
    index = {e["record"]["id"]: e["record"] for e in evidence}
    results, narratives, scenario_ids = [], [], set()
    for candidate in output.scenarios:
        errors, warnings = [], ["AI proposal pending engineering review; factual and contradiction checks are limited heuristics."]
        missing, invalid_evidence, unsupported = 0, 0, 0
        if candidate.input_conditions != inputs:
            errors.append("Generated operating conditions contradict the frozen request")
        if candidate.scenario_id in scenario_ids:
            errors.append("Duplicate scenario ID")
        scenario_ids.add(candidate.scenario_id)
        if len({h.hazard_id for h in candidate.hazards}) != len(candidate.hazards):
            errors.append("Duplicate hazard IDs")
        for h in candidate.hazards:
            if not h.evidence_ids:
                missing += 1
                warnings.append(f"{h.hazard_id}: missing evidence")
            for ref in h.evidence_ids:
                if ref not in index or index[ref]["type"] not in {"source", "evidence", "hazard", "risk_criteria"}:
                    errors.append(f"{h.hazard_id}: invalid evidence reference {ref}")
                    invalid_evidence += 1
                elif index[ref]["illustrative"] or index[ref]["review_status"] != "approved":
                    warnings.append(f"{ref}: illustrative or unapproved evidence")
            if not h.hypothesis and (not h.evidence_ids or any(i not in index or index[i]["illustrative"] or index[i]["review_status"] != "approved" for i in h.evidence_ids)):
                unsupported += 1
                warnings.append(f"{h.hazard_id}: factual claim lacks reviewed evidence")
        linked = {}
        invalid_refs = {}
        for field, kind in (("mitigation_ids", "mitigation"), ("acceptance_test_ids", "acceptance_test")):
            linked[field], invalid_refs[field] = [], 0
            for ref in getattr(candidate, field):
                if ref not in index or index[ref]["type"] != kind:
                    errors.append(f"Invalid {kind} reference: {ref}")
                    invalid_refs[field] += 1
                else:
                    linked[field].append(index[ref])
                    if index[ref]["review_status"] != "approved" or index[ref]["illustrative"]:
                        warnings.append(f"{ref}: proposed link is unapproved or illustrative")
        text = re.sub(r"\W+", " ", candidate.scenario_narrative.lower()).strip()
        duplicate = any(SequenceMatcher(None, text, previous).ratio() >= 0.9 for previous in narratives)
        narratives.append(text)
        if duplicate:
            warnings.append("Near-duplicate narrative (similarity >= 0.9)")
        if re.search(r"\b(guaranteed safe|certified safe|zero risk)\b", candidate.scenario_narrative, re.I):
            unsupported += 1
            warnings.append("Unsupported safety assurance claim detected")
        verification = {"schema_valid": True, "evidence_valid": invalid_evidence == 0 and missing == 0,
                        "references_valid": not any(invalid_refs.values()), "valid": not errors and not missing,
                        "invalid_evidence": invalid_evidence, "missing_evidence": missing,
                        "unsupported_claims_flagged": unsupported, "duplicate": duplicate,
                        "invalid_mitigation_refs": invalid_refs["mitigation_ids"],
                        "invalid_test_refs": invalid_refs["acceptance_test_ids"],
                        "warnings": warnings, "errors": errors}
        results.append({**candidate.model_dump(), "verification": verification,
                        "risk_assessments": {h.hazard_id: calculate(matrix, None, evidence) for h in candidate.hazards},
                        "linked_records": linked, "review": {"status": "pending_review"}})
    return results
