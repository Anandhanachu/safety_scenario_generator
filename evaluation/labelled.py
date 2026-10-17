"""ID precision/recall and repeated baseline/AI evaluation; no inferred telemetry."""
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from backend.schemas import Strict, ScenarioInput, Generation
from backend.constrained import (Fault, Condition, ModelDetails, RunResult, SelectionOutput,
                                 baseline, execute, verify_selection, LIMITATIONS)
from knowledge_base.core import parse_upload, retrieve_with_diagnostics
from risk_engine.calculator import RiskMatrix


class LabelledCase(Strict):
    id: str = Field(min_length=1)
    fault_type: Fault
    operating_condition: Condition
    query: str = Field(min_length=1, max_length=4000)
    expected_hazard_ids: list[str]
    expected_mitigation_ids: list[str]
    expected_test_ids: list[str]
    expected_risk_category: Literal['low', 'medium', 'high'] | None
    expected_outcome: Literal['selected', 'insufficient_evidence']

    @model_validator(mode='after')
    def consistent(self):
        groups = [self.expected_hazard_ids, self.expected_mitigation_ids, self.expected_test_ids]
        if any(len(set(g)) != len(g) for g in groups):
            raise ValueError(f'{self.id}: duplicate expected IDs')
        if self.expected_outcome == 'insufficient_evidence' and (any(groups) or self.expected_risk_category):
            raise ValueError(f'{self.id}: insufficient evidence requires empty labels and null risk')
        if self.expected_outcome == 'selected' and not all(groups):
            raise ValueError(f'{self.id}: selected case requires hazard, mitigation and test labels')
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
    cases: list[LabelledCase] = Field(min_length=1, max_length=100)

    @model_validator(mode='after')
    def unique(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError('Duplicate labelled case IDs')
        if self.reviewed and not self.reviewed_by:
            raise ValueError('Expert-labelled datasets require reviewed_by')
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


def spread(values):
    values = [v for v in values if v is not None]
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'stddev': statistics.pstdev(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def baseline_result(evidence, matrix, threshold=1.0, max_tokens=1800):
    start = time.perf_counter()
    output = baseline(evidence)
    check = verify_selection(output, evidence, matrix, threshold)
    return RunResult(model=ModelDetails(backend='baseline', name='TF-IDF + stored risk matrix'),
                     settings={'temperature': 0.0, 'seed': 42, 'max_tokens': max_tokens}, evidence=evidence,
                     output=output, verification=check, risk=check['risk'], confidence=check['confidence'],
                     needs_human_review=bool(check['human_review_flags']), latency_ms=round((time.perf_counter()-start)*1000, 3))


def evaluate(labels, kb, matrix, models=(), repeats=3, limit=12, threshold=0.0, confidence_threshold=1.0, cancel=None, max_tokens=1800):
    if repeats not in range(3, 6):
        raise ValueError('Use 3-5 repetitions for fair comparison')
    labels.validate_kb(kb)
    Generation(max_tokens=max_tokens)
    rows = []
    # Retrieval once per case freezes identical evidence for every backend/repetition.
    for case in labels.cases:
        inputs = case.inputs()
        evidence, diagnostics = retrieve_with_diagnostics(kb, inputs.model_dump(), limit, threshold)
        for repeat in range(repeats):
            if cancel and cancel.is_set():
                return {'status': 'cancelled', 'rows': rows, 'summary': summarize(rows), 'limitations': LIMITATIONS}
            for details, adapter in [(ModelDetails(backend='baseline', name='TF-IDF + stored risk matrix'), None), *models]:
                error = None
                try:
                    result = baseline_result(evidence, matrix, confidence_threshold, max_tokens) if adapter is None else execute(adapter, details, inputs, evidence, matrix, max_tokens=max_tokens, threshold=confidence_threshold, cancel=cancel)
                    output = result.output
                except Exception as exc:
                    from models.adapters import Cancelled, BackendError
                    if isinstance(exc, Cancelled):
                        raise
                    if not isinstance(exc, BackendError):
                        raise
                    error, output, result = str(exc), None, None
                row = {'case_id': case.id, 'repeat': repeat + 1, 'model': details.model_dump(),
                       'metrics': score(case, output), 'retrieval': diagnostics, 'error': error,
                       'result': result.model_dump() if result else None}
                row['risk_correct'] = (int(result is not None and bool(result.risk) and all(r['risk_level'] == case.expected_risk_category for r in result.risk.values()))
                                       if case.expected_outcome == 'selected' else None)
                rows.append(row)
    return {'status': 'completed', 'labels': labels.model_dump(), 'knowledge_base_version': kb.version,
            'matrix': matrix.model_dump(), 'settings': {'repeats': repeats, 'top_k': limit, 'threshold': threshold, 'confidence_threshold': confidence_threshold, 'temperature': 0.0, 'seed': 42, 'max_tokens': max_tokens},
            'rows': rows, 'summary': summarize(rows), 'limitations': LIMITATIONS}


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row['model']['backend'], row['model']['name'])
        groups.setdefault(key, []).append(row)
    summary = []
    for (backend, model), group in groups.items():
        result = {'backend': backend, 'model': model, 'executions': len(group), 'failures': sum(r['error'] is not None for r in group)}
        for name in ('hazards', 'mitigations', 'acceptance_tests'):
            # Sum per repetition first; all repetitions include the same cases.
            for metric in ('precision', 'recall'):
                values = []
                for repeat in sorted({r['repeat'] for r in group}):
                    counts = [r['metrics'][name] for r in group if r['repeat'] == repeat]
                    tp = sum(c['tp'] for c in counts)
                    denom = tp + sum(c['fp' if metric == 'precision' else 'fn'] for c in counts)
                    values.append(tp / denom if denom else None)
                result[f'{name}_{metric}'] = spread(values)
        for name in ('insufficient_evidence_correct', 'risk_correct'):
            rates = []
            for repeat in sorted({r['repeat'] for r in group}):
                values = [(r['metrics'][name] if name == 'insufficient_evidence_correct' else r[name]) for r in group if r['repeat'] == repeat]
                values = [v for v in values if v is not None]
                rates.append(statistics.mean(values) if values else None)
            result[name] = spread(rates)
        result['generation_latency_ms'] = spread([r['result']['latency_ms'] for r in group if r['result'] and backend != 'baseline'])
        summary.append(result)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kb', default='data/constrained_knowledge_base.json')
    parser.add_argument('--cases', default='data/labelled_cases_24.json')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--configured-models', action='store_true', help='Send real inference requests to configured backends; may incur cloud charges')
    args = parser.parse_args()
    kb = parse_upload(Path(args.kb).read_bytes())
    labels = LabelledDataset.model_validate_json(Path(args.cases).read_bytes())
    matrix = RiskMatrix.model_validate_json(Path('configs/risk_matrix.example.json').read_bytes())
    models = []
    if args.configured_models:
        from backend.services import Services
        from models.adapters import make_adapter
        settings = Services().settings()
        for backend in ('laptop_local', 'uno_q', 'qualcomm'):
            cfg = settings[backend]
            if cfg.get('model_id'):
                adapter = make_adapter(backend, settings)
                adapter.validate_configuration(cfg['model_id'])
                models.append((ModelDetails(backend=backend, name=cfg['model_id'], parameter_size=cfg.get('parameter_size'), quantization=cfg.get('quantization')), adapter))
    print(json.dumps(evaluate(labels, kb, matrix, models, args.repeats), indent=2))


if __name__ == '__main__':
    main()
