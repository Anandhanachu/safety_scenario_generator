"""Ordinal ranking only. Classifications must be recorded by a human with source evidence."""
from typing import Literal
from pydantic import Field, model_validator
from backend.schemas import Strict


class RiskMatrix(Strict):
    version: str = Field(min_length=1, max_length=100)
    illustrative: bool = True
    severity: dict[str, str]
    likelihood: dict[str, str]
    values: dict[str, dict[str, int]]
    medium_min: int = Field(default=4, ge=1)
    high_min: int = Field(default=9, ge=1)

    @model_validator(mode="after")
    def complete(self):
        if not self.severity or not self.likelihood or self.high_min <= self.medium_min:
            raise ValueError("Invalid matrix categories or thresholds")
        if set(self.values) != set(self.severity):
            raise ValueError("Matrix severity rows do not match criteria")
        if any(set(row) != set(self.likelihood) or any(v < 1 for v in row.values()) for row in self.values.values()):
            raise ValueError("Matrix likelihood columns or scores invalid")
        return self


class Classification(Strict):
    severity: str | None = None
    likelihood: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(default="", max_length=4000)
    reviewer: str = Field(default="", max_length=200)
    origin: Literal["human"] = "human"


def stored_risk(matrix: RiskMatrix, record: dict):
    """New runs use only frozen expert KB inputs; the model cannot classify risk."""
    base = {'severity': record.get('severity'), 'likelihood': record.get('likelihood'), 'risk_score': None,
            'risk_level': None, 'status': 'requires_review', 'matrix_version': matrix.version,
            'explanation': 'Missing, conflicting, or unreviewed expert risk inputs. Engineering ranking, not a calibrated probability.'}
    severity, likelihood = base['severity'], base['likelihood']
    if (severity not in matrix.severity or likelihood not in matrix.likelihood or record.get('risk_conflict')
            or not record.get('reviewed_by') or record.get('review_status') != 'approved' or record.get('illustrative', True)):
        return base
    score = matrix.values[severity][likelihood]
    level = 'high' if score >= matrix.high_min else 'medium' if score >= matrix.medium_min else 'low'
    return {**base, 'risk_score': score, 'risk_level': level, 'status': 'requires_review' if matrix.illustrative else 'calculated',
            'explanation': f"Expert {record['reviewed_by']} stored {severity}, {likelihood} in {record['id']}; matrix {matrix.version}[{severity}][{likelihood}] = {score}; high >= {matrix.high_min}, medium >= {matrix.medium_min}; category {level}. Engineering ranking, not a calibrated probability."}


def calculate(matrix: RiskMatrix, classification: Classification | None, evidence: list[dict]):
    base = {"severity": None, "likelihood": None, "risk_score": None, "risk_level": None,
            "matrix_version": matrix.version, "method": "configured ordinal matrix lookup",
            "status": "requires_review", "illustrative": matrix.illustrative}
    if classification is None:
        return {**base, "explanation": "No evidence-backed human severity and likelihood classifications supplied."}
    c = classification
    records = {e["record"]["id"]: e["record"] for e in evidence}
    if c.severity not in matrix.severity or c.likelihood not in matrix.likelihood:
        return {**base, "explanation": "Severity or likelihood is missing or outside the matrix."}
    if not c.reviewer or not c.rationale or not c.evidence_ids or any(
        i not in records or records[i]["review_status"] != "approved" or records[i]["illustrative"]
        or records[i]["type"] not in {"evidence", "source", "risk_criteria"} for i in c.evidence_ids
    ):
        return {**base, "explanation": "Approved, non-illustrative source/criteria evidence, reviewer and rationale are required."}
    score = matrix.values[c.severity][c.likelihood]
    return {**base, "severity": c.severity, "likelihood": c.likelihood, "risk_score": score,
            "risk_level": "high" if score >= matrix.high_min else "medium" if score >= matrix.medium_min else "low",
            "status": "requires_review" if matrix.illustrative else "calculated",
            "classification": c.model_dump(),
            "explanation": f"{c.severity} × {c.likelihood}: configured cell = {score}. Ordinal ranking, not a probability. " + ("Illustrative matrix requires policy review." if matrix.illustrative else "")}
