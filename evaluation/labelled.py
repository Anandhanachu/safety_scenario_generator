"""ID precision/recall and repeated baseline/AI evaluation; no inferred telemetry."""
import argparse
import json
import statistics
import time
import copy
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from backend.schemas import Strict, ScenarioInput, Generation
from backend.constrained import (Fault, Condition, ModelDetails, RunResult, SelectionOutput,
                                 baseline, execute, verify_selection, LIMITATIONS)
from knowledge_base.core import parse_upload, retrieve_with_diagnostics
from risk_engine.calculator import RiskMatrix
from models.adapters import BackendError, Cancelled, make_adapter

RISK_ORDER = {'low': 0, 'medium': 1, 'high': 2}
DECISIONS = ['low', 'medium', 'high', 'needs_review', 'insufficient_evidence', 'failed']


class LabelledCase(Strict):
    id: str = Field(min_length=1)
    fault_type: Fault
    operating_condition: Condition
    query: str = Field(min_length=1, max_length=4000)
    expected_hazard_ids: list[str]
    expected_mitigation_ids: list[str]
    expected_test_ids: list[str]
    expected_risk_category: Literal['low', 'medium', 'high'] | None
    expected_outcome: Literal['selected', 'insufficient_evidence', 'needs_review']
    risk_input_issue: Literal['none', 'missing_severity', 'missing_likelihood', 'conflicting_evidence'] = 'none'
    reviewer: str | None = None
    reviewed_date: str | None = None

    @model_validator(mode='after')
    def consistent(self):
        groups = [self.expected_hazard_ids, self.expected_mitigation_ids, self.expected_test_ids]
        if any(len(set(g)) != len(g) for g in groups):
            raise ValueError(f'{self.id}: duplicate expected IDs')
        if self.expected_outcome == 'insufficient_evidence' and (any(groups) or self.expected_risk_category):
            raise ValueError(f'{self.id}: insufficient evidence requires empty labels and null risk')
        if self.expected_outcome in {'selected', 'needs_review'} and not all(groups):
            raise ValueError(f'{self.id}: selected case requires hazard, mitigation and test labels')
        if self.expected_outcome == 'selected' and self.expected_risk_category is None:
            raise ValueError(f'{self.id}: selected requires a risk category; use needs_review for unresolved risk')
        if self.expected_outcome == 'needs_review' and self.expected_risk_category is not None:
            raise ValueError(f'{self.id}: needs_review requires no risk category')
        if self.risk_input_issue != 'none' and self.expected_outcome != 'needs_review':
            raise ValueError(f'{self.id}: missing/conflicting risk inputs must expect needs_review')
        if self.reviewed_date:
            date.fromisoformat(self.reviewed_date)
            if not self.reviewer:
                raise ValueError(f'{self.id}: reviewed_date requires reviewer')
        return self

    def inputs(self):
        return ScenarioInput(faults=[self.fault_type], time_of_day='night' if self.operating_condition == 'night_dry' else 'day',
                             weather='rain' if self.operating_condition == 'rain' else 'dry',
                             visibility='reduced' if self.operating_condition == 'reduced_visibility' else 'clear', notes=self.query)


class LabelledDataset(Strict):
    version: str
    knowledge_base_id: str
    reviewed: bool = False
    reviewed_by: str | None = None
    reviewer: str | None = None
    reviewed_date: str | None = None
    cases: list[LabelledCase] = Field(min_length=1, max_length=100)

    @model_validator(mode='after')
    def unique(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError('Duplicate labelled case IDs')
        if self.reviewed_date:
            date.fromisoformat(self.reviewed_date)
        if self.reviewed and (not (self.reviewer or self.reviewed_by) or not self.reviewed_date or any(not c.reviewer or not c.reviewed_date for c in self.cases)):
            raise ValueError('Reviewed labels require reviewer and reviewed_date on the dataset and every case')
        return self

    def validate_kb(self, kb):
        if self.knowledge_base_id != kb.dataset_id or kb.schema_version != '2.0':
            raise ValueError('Labels require the matching v2 knowledge base')
        index = {r.id: r for r in kb.records}
        for c in self.cases:
            for ids, kind in [(c.expected_hazard_ids, 'hazard'), (c.expected_mitigation_ids, 'mitigation'), (c.expected_test_ids, 'acceptance_test')]:
                for ref in ids:
                    if ref not in index or index[ref].type != kind:
                        raise ValueError(f'{c.id}: missing/wrong-type expected {kind} {ref}')
                    if kind != 'hazard' and not set(index[ref].hazard_ids) & set(c.expected_hazard_ids):
                        raise ValueError(f'{c.id}: {ref} not linked to an expected hazard')


def precision_recall(expected, predicted):
    expected, predicted = set(expected), set(predicted)
    tp = len(expected & predicted)
    return {'tp': tp, 'fp': len(predicted - expected), 'fn': len(expected - predicted),
            'precision': tp / len(predicted) if predicted else None,
            'recall': tp / len(expected) if expected else None}


def score(case, output):
    choices = output.selections if output else []
    groups = [('hazards', case.expected_hazard_ids, [c.hazard_id for c in choices]),
              ('mitigations', case.expected_mitigation_ids, [i for c in choices for i in c.mitigation_ids]),
              ('acceptance_tests', case.expected_test_ids, [i for c in choices for i in c.acceptance_test_ids])]
    return {**{name: precision_recall(e, p) for name, e, p in groups},
            'insufficient_evidence_correct': int(output is not None and output.outcome == 'insufficient_evidence') if case.expected_outcome == 'insufficient_evidence' else None}


def spread(values, case_count=0, eligible_case_count=None):
    values = [v for v in values if v is not None]
    return {'sample_count': len(values), 'case_count': case_count, 'eligible_case_count': case_count if eligible_case_count is None else eligible_case_count,
            'label': 'preliminary' if case_count < 30 else 'measured', 'mean': statistics.mean(values) if values else None,
            'stddev': statistics.pstdev(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def baseline_result(evidence, matrix, threshold=1.0, max_tokens=1800):
    start = time.perf_counter()
    output = baseline(evidence)
    check = verify_selection(output, evidence, matrix, threshold)
    return RunResult(model=ModelDetails(backend='baseline', name='TF-IDF + stored risk matrix'),
                     settings={'temperature': 0.0, 'seed': 42, 'max_tokens': max_tokens}, evidence=evidence,
                     requested_settings={'temperature': 0.0, 'seed': 42, 'max_tokens': max_tokens},
                     applied_settings={k: 'not supported' for k in ('temperature', 'seed', 'max_tokens')},
                     output=output, verification=check, risk=check['risk'], citation_link_validity=check['citation_link_validity'],
                     reviewed_support_rate=check['reviewed_support_rate'],
                     needs_human_review=bool(check['human_review_flags']), latency_ms=round((time.perf_counter()-start)*1000, 3))


def evaluate_predictions(case, result, failed=False):
    """Exact-match retains failures; ordinal distance requires two measured categories."""
    output = result.output if result and not failed else None
    categories = [r.get('risk_level') for r in result.risk.values()] if output else []
    category = max(categories, key=RISK_ORDER.get) if categories and all(c in RISK_ORDER for c in categories) else None
    review_flag = bool(result and result.needs_human_review)
    actual = ('failed' if failed or output is None else 'insufficient_evidence' if output.outcome == 'insufficient_evidence'
              else 'needs_review' if category is None and review_flag else 'failed' if category is None or output.outcome == 'needs_review' else category)
    expected = case.expected_risk_category if case.expected_outcome == 'selected' else case.expected_outcome
    review_correct = int(actual == 'needs_review' and review_flag and category is None) if case.expected_outcome == 'needs_review' else None
    metrics = score(case, output)
    metrics.update(risk_category_exact_match_rate=int(category == case.expected_risk_category and actual != 'failed') if case.expected_risk_category else None,
                   ordinal_distance_error=abs(RISK_ORDER[category] - RISK_ORDER[case.expected_risk_category]) if category in RISK_ORDER and case.expected_risk_category in RISK_ORDER else None,
                   decision_accuracy=int(actual == expected), needs_review_correct=review_correct,
                   missing_input_review_correct=review_correct if case.risk_input_issue.startswith('missing_') else None,
                   conflicting_input_review_correct=review_correct if case.risk_input_issue == 'conflicting_evidence' else None,
                   citation_link_validity=result.citation_link_validity if output else None,
                   reviewed_support_rate=result.reviewed_support_rate if output else None,
                   failure_rate=int(failed or output is None))
    return {'metrics': metrics, 'expected_decision': expected, 'predicted_decision': actual,
            'predicted_risk_category': category, 'human_review_flag': review_flag}


def case_evidence(evidence, issue):
    """Declared robustness fixtures alter only risk inputs, never expected output IDs."""
    copied = copy.deepcopy(evidence)
    for entry in copied:
        record = entry['record']
        if record['type'] == 'hazard':
            if issue == 'missing_severity': record['severity'] = None
            if issue == 'missing_likelihood': record['likelihood'] = None
            if issue == 'conflicting_evidence': record['risk_conflict'] = True
    return copied


def evaluate(labels, kb, matrix, models=(), repeats=3, limit=12, threshold=0.0, citation_link_validity_threshold=1.0, cancel=None, max_tokens=1800):
    if repeats not in range(3, 6):
        raise ValueError('Use 3-5 repetitions for fair comparison')
    labels.validate_kb(kb)
    requested = Generation(max_tokens=max_tokens).model_dump()
    rows = []
    for case in labels.cases:
        inputs = case.inputs()
        evidence, diagnostics = retrieve_with_diagnostics(kb, inputs.model_dump(), limit, threshold)
        evidence = case_evidence(evidence, case.risk_input_issue)
        for repeat in range(repeats):
            for details, adapter in [(ModelDetails(backend='baseline', name='TF-IDF + stored risk matrix'), None), *models]:
                if cancel and cancel.is_set():
                    return finish('cancelled', labels, kb, matrix, requested, rows, repeats, limit, threshold, citation_link_validity_threshold)
                error, result = None, None
                applied = {k: 'not measured' for k in requested}
                started = time.perf_counter()
                try:
                    if isinstance(adapter, Exception):
                        raise adapter
                    if adapter is not None and details.name == 'not configured':
                        raise BackendError('Model not configured; inference not attempted')
                    result = baseline_result(evidence, matrix, citation_link_validity_threshold, max_tokens) if adapter is None else execute(adapter, details, inputs, evidence, matrix, max_tokens=max_tokens, threshold=citation_link_validity_threshold, cancel=cancel)
                    applied = result.applied_settings
                    if result.output is None:
                        error = 'Verification failed after corrective retry'
                except Cancelled:
                    return finish('cancelled', labels, kb, matrix, requested, rows, repeats, limit, threshold, citation_link_validity_threshold)
                except Exception as exc:
                    # Keep timeouts and unexpected adapter failures in all relevant denominators.
                    error = str(exc) if isinstance(exc, (BackendError, TimeoutError)) else f'Adapter failed: {type(exc).__name__}'
                    if details.name != 'not configured':
                        applied = dict(getattr(adapter, 'applied_settings', applied))
                row = {'case_id': case.id, 'repeat': repeat + 1, 'model': details.model_dump(), 'status': 'failed' if error else 'completed',
                       'risk_input_issue': case.risk_input_issue, 'requested_settings': dict(requested), 'applied_settings': applied,
                       'retrieval': diagnostics, 'evidence': evidence, 'error': error,
                       'execution_latency_ms': round((time.perf_counter()-started)*1000, 3), 'result': result.model_dump() if result else None}
                row.update(evaluate_predictions(case, result, failed=bool(error)))
                row['metric_eligible_cases'] = {name: (case.expected_risk_category is not None if name in {'ordinal_distance_error', 'risk_category_exact_match_rate'}
                                                     else bool(case.expected_hazard_ids) if name in {'citation_link_validity', 'reviewed_support_rate'} else row['metrics'][name] is not None)
                                                for name, value in row['metrics'].items() if not isinstance(value, dict)}
                rows.append(row)
    return finish('completed', labels, kb, matrix, requested, rows, repeats, limit, threshold, citation_link_validity_threshold)


def finish(status, labels, kb, matrix, requested, rows, repeats, limit, threshold, citation_link_validity_threshold):
    return {'status': status, 'labels': labels.model_dump(), 'knowledge_base_version': kb.version, 'matrix': matrix.model_dump(),
            'settings': {**requested, 'repeats': repeats, 'top_k': limit, 'threshold': threshold, 'citation_link_validity_threshold': citation_link_validity_threshold}, 'rows': rows,
            'summary': summarize(rows), 'case_count': len(labels.cases), 'label': 'preliminary' if len(labels.cases) < 30 else 'measured',
            'limitations': LIMITATIONS}


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['model']['backend'], row['model']['name']), []).append(row)
    summary = []
    for (backend, model), group in groups.items():
        cases = {r['case_id'] for r in group}
        repeats = sorted({r['repeat'] for r in group})
        result = {'backend': backend, 'model': model, 'case_count': len(cases), 'label': 'preliminary' if len(cases) < 30 else 'measured',
                  'executions': len(group), 'failures': sum(r['status'] == 'failed' for r in group)}
        for name in ('hazards', 'mitigations', 'acceptance_tests'):
            for metric in ('precision', 'recall'):
                values, eligible = [], set()
                for repeat in repeats:
                    selected = [r for r in group if r['repeat'] == repeat]
                    counts = [r['metrics'][name] for r in selected]
                    tp = sum(c['tp'] for c in counts)
                    denominator = tp + sum(c['fp' if metric == 'precision' else 'fn'] for c in counts)
                    values.append(tp / denominator if denominator else None)
                    eligible.update(r['case_id'] for r in selected if r['metrics'][name][metric] is not None)
                result[f'{name}_{metric}'] = spread(values, len(eligible), len(cases))
        for name, metric in group[0]['metrics'].items():
            if isinstance(metric, dict):
                continue
            rates, eligible = [], {r['case_id'] for r in group if r['metrics'][name] is not None}
            for repeat in repeats:
                values = [r['metrics'][name] for r in group if r['repeat'] == repeat and r['metrics'][name] is not None]
                if name in {'citation_link_validity', 'reviewed_support_rate'}:
                    checks = [r['result']['verification'] for r in group if r['repeat'] == repeat and r['result'] and r['metrics'][name] is not None]
                    numerator, denominator = ('backed_claims', 'claims') if name == 'citation_link_validity' else ('supported_count', 'reviewed_support_count')
                    total = sum(c.get(denominator, 0) for c in checks)
                    rates.append(sum(c.get(numerator, 0) for c in checks) / total if total else None)
                else:
                    rates.append(statistics.mean(values) if values else None)
            eligible_cases = {r['case_id'] for r in group if r['metric_eligible_cases'][name]}
            result[name] = spread(rates, len(eligible), len(eligible_cases))
        result['execution_latency_ms'] = spread([r['execution_latency_ms'] for r in group], len(cases))
        result['generation_latency_ms'] = spread([r['result']['latency_ms'] for r in group if r['result'] and backend != 'baseline'],
                                                len({r['case_id'] for r in group if r['result'] and backend != 'baseline'}), len(cases))
        result['confusion_matrix'] = {e: {p: sum(r['expected_decision'] == e and r['predicted_decision'] == p for r in group) for p in DECISIONS} for e in DECISIONS}
        result['confusion_matrix_spread'] = {e: {p: spread([sum(r['expected_decision'] == e and r['predicted_decision'] == p and r['repeat'] == repeat for r in group) for repeat in repeats], len(cases)) for p in DECISIONS} for e in DECISIONS}
        summary.append(result)
    return summary


def run_experiment(labels, kb, matrix, settings, repeats=3, adapter_factory=make_adapter, **kwargs):
    """Account for baseline and every backend, including unconfigured models and failed runs."""
    models = []
    for backend in ('laptop_local', 'uno_q', 'qualcomm'):
        cfg = settings[backend]
        details = ModelDetails(backend=backend, name=cfg.get('model_id') or 'not configured',
                               parameter_size=cfg.get('parameter_size'), quantization=cfg.get('quantization'))
        try:
            adapter = adapter_factory(backend, settings)
        except Exception as exc:
            adapter = exc
        models.append((details, adapter))
    return evaluate(labels, kb, matrix, models, repeats, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kb', default='data/constrained_knowledge_base.json')
    parser.add_argument('--cases', default='data/labelled_cases_25.json')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--configured-models', action='store_true', help='Send real inference requests to configured backends; may incur cloud charges')
    args = parser.parse_args()
    kb = parse_upload(Path(args.kb).read_bytes())
    labels = LabelledDataset.model_validate_json(Path(args.cases).read_bytes())
    matrix = RiskMatrix.model_validate_json(Path('configs/risk_matrix.example.json').read_bytes())
    if args.configured_models:
        from backend.services import Services
        result = run_experiment(labels, kb, matrix, Services().settings(), args.repeats)
    else:
        result = evaluate(labels, kb, matrix, repeats=args.repeats)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
