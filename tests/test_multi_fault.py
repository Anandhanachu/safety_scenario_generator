"""Every supported fault combination must retain complete, risk-bearing evidence."""
from itertools import combinations

from backend.configuration import ROOT
from backend.constrained import FAULT_LABELS, baseline, scope
from backend.schemas import ScenarioInput
from backend.scenario_generation import screen_drafts, recalculate
from knowledge_base.core import parse_upload, retrieve_with_diagnostics


def test_all_fault_combinations_and_conditions(matrix):
    kb = parse_upload((ROOT / 'data/multi_fault_knowledge_base.json').read_bytes())
    assert len(kb.records) == 180
    cases = 0
    for count in (1, 2, 3):
        for faults in combinations(FAULT_LABELS.values(), count):
            for context in ({}, {'time_of_day': 'night'}, {'weather': 'rain'}, {'visibility': 'reduced'}):
                inputs = ScenarioInput(faults=list(faults), **context)
                evidence, diagnostics = retrieve_with_diagnostics(kb, inputs, 12, .05)
                output = baseline(evidence)
                index = {item['record']['id']: item['record'] for item in evidence}
                covered = {fault for choice in output.selections for fault in index[choice.hazard_id]['tags']['fault_type']}
                assert covered == set(faults)
                drafts = [{'scenario_id': str(n), 'title': 'Test', 'statements': [
                    {'record_id': ref, 'text': index[ref]['description']}
                    for ref in [choice.hazard_id, *choice.mitigation_ids, *choice.acceptance_test_ids]]}
                    for n, choice in enumerate(output.selections)]
                scenarios = screen_drafts(drafts, inputs, evidence, matrix)
                assert all(s['system_decision'] == 'accepted' for s in scenarios)
                run = {'snapshot': {'evidence': evidence, 'risk_matrix': matrix.model_dump(),
                                   'request': {'inputs': inputs.model_dump()}},
                       'results': {'test': {'generation_mode': 'scenario_drafts', 'scenarios': scenarios}}}
                result = recalculate(run)['test']
                assert result['risk_score'] == max(s['risk_assessments'][h['hazard_id']]['risk_score']
                                                  for s in scenarios for h in s['hazards'])
                assert result['uncovered_faults'] == []
                if context in ({'weather': 'rain'}, {'visibility': 'reduced'}):
                    assert result['risk_score'] == 9 and result['risk_level'] == 'high'
                assert len(evidence) <= 12
                cases += 1
    assert cases == 100


def test_scope_rejects_four_faults():
    assert scope(ScenarioInput(faults=list(FAULT_LABELS)[:4])) is None
