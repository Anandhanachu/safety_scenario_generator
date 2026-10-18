"""Engineering workspace backed by shared services; no synthetic activity or model results."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from html import escape

import plotly.express as px
import streamlit as st
from pydantic import ValidationError

from backend.configuration import ROOT, validate_url
from backend.schemas import Generation, RunRequest, ScenarioInput, Selection
from backend.services import Services
from backend.orchestration import IMPLEMENTATION_VERSION
from backend.analytics import scenario_register, review_counts, performance_rows, proposal_entries, DECISIONS
from evaluation.benchmark import BenchmarkDataset, draft_dataset
from evaluation.labelled import LabelledDataset, baseline_result
from backend.constrained import FAULT_LABELS, LIMITATIONS
from backend.scenario_generation import recalculate
from knowledge_base.core import Dataset, ConstrainedDataset, parse_upload, retrieve, retrieve_with_diagnostics
from knowledge_base.benchmark import benchmark_retrieval
from ui.charts import review_figure, latency_figure, validation_figure, outcome_figure
from models.adapters import BackendError, discover_files, make_adapter
from reporting.exports import comparison_rows, csv_bytes, json_bytes, report
from reporting.exports import experiment_report, experiment_csv
from reporting.pdf_export import pdf_bytes
from reporting.pdf_export import experiment_pdf_bytes
from risk_engine.calculator import Classification, RiskMatrix, calculate
from simulation.engine import SimulationInput, simulate

LABELS = {"laptop_local": "Laptop local", "qualcomm": "Qualcomm cloud", "uno_q": "Arduino UNO Q"}
ALL_BACKENDS = "all_backends"
PAGES = ["Overview", "Scenario Builder", "Model Studio", "Model Comparison", "Scenario Register", "Knowledge Base", "Risk Analysis", "Test Simulation", "Evaluation and Benchmarks", "Reports and History", "Device Control", "Settings"]


@st.cache_resource
def services():
    """Shared services v5: complete multi-fault evidence and preflight validation."""
    return Services()


def theme(dark):
    bg, surface, text, muted, border = ("#101827", "#182235", "#e5edf8", "#adbdd3", "#34445d") if dark else ("#f5f7fb", "#ffffff", "#17233b", "#52627a", "#dbe2ee")
    sidebar_surface, sidebar_text, sidebar_hover, sidebar_selected = (("#111c32", "#d8e2f3", "#213352", "#33467b") if dark
                                                                  else ("#ffffff", "#17233b", "#eef2ff", "#e0e7ff"))
    st.markdown(f"""<style>
    .stApp, [data-testid="stAppViewContainer"] {{ background:{bg}; color:{text}; }}
    [data-testid="stHeader"], [data-testid="stSidebar"] {{ background:{surface}; color:{text}; }}
    [data-testid="stSidebar"] {{ border-right:1px solid {border}; }}
    .block-container {{ padding-top:2rem; max-width:1600px; }}
    [data-testid="stSidebar"] {{background:{sidebar_surface}!important;}}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{color:{sidebar_text}!important;}}
    [data-testid="stSidebar"] [role="radiogroup"] label {{padding:6px 10px;border-radius:8px;transition:background .15s;}}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{background:{sidebar_hover};}}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{background:{sidebar_selected};}}
    .backend-status {{display:block;font-size:12px;margin:7px 0 3px;line-height:1.45;}}
    .status-dot {{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;}}
    .status-good {{color:#15803d!important;}} .status-good .status-dot {{background:#22c55e;box-shadow:0 0 0 3px rgba(34,197,94,.16);}}
    .status-warn {{color:#a16207!important;}} .status-warn .status-dot {{background:#f59e0b;}}
    .status-neutral {{color:{muted}!important;}} .status-neutral .status-dot {{background:#94a3b8;}}
    .workspace-hero {{border:1px solid {border};border-left:5px solid #6366f1;background:{surface};border-radius:16px;padding:25px 28px;margin-bottom:24px;}}
    .workspace-hero h1 {{margin:6px 0 8px!important;font-size:2.15rem!important;}}
    .workspace-hero p {{color:{muted};margin:0;max-width:850px;line-height:1.6;}}
    .eyebrow {{font-size:11px;letter-spacing:2px;font-weight:700;color:#818cf8;text-transform:uppercase;}}
    .workflow-band {{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 24px;}}
    .workflow-chip {{border:1px solid {border};border-radius:24px;background:{surface};padding:8px 15px;font-size:12px;color:{muted};}}
    .empty-panel {{background:{surface};border:1px dashed {border};border-radius:14px;padding:32px 26px;margin:12px 0;min-height:200px;}}
    .empty-panel strong {{font-size:17px;color:{text};display:block;margin-bottom:10px;}}
    .empty-panel p {{color:{muted};font-size:14px;line-height:1.7;}}
    [data-testid="stMetric"] {{border-top:3px solid #6366f1!important;box-shadow:0 5px 20px rgba(15,23,42,.035);}}
    [data-testid="stMetricValue"] {{font-weight:650;letter-spacing:-.04em;}}
    [data-testid="stTabs"] [role="tablist"] {{gap:16px;border-bottom:1px solid {border};}}
    h1 {{ font-size:2rem!important; letter-spacing:-.03em; }}
    h2 {{ font-size:1.35rem!important; }}
    h1,h2,h3,p,label, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {{ color:{text}; }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{color:{muted}!important;opacity:1!important;}}
    [data-testid="stMetric"], [data-testid="stExpander"], [data-testid="stForm"] {{background:{surface}; border:1px solid {border}; border-radius:12px; padding:14px;}}
    [data-testid="stMetricValue"] {{font-size:1.8rem; color:{text};}}
    [data-testid="stMetricLabel"] {{height:auto;min-height:2.5rem;}}
    [data-testid="stMetricLabel"] * {{white-space:normal!important;text-overflow:clip!important;overflow:visible!important;}}
    @media (min-width:641px) and (max-width:1100px) {{
      [data-testid="stHorizontalBlock"] {{flex-wrap:wrap;}}
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{flex:1 1 220px!important;min-width:220px!important;width:auto!important;}}
    }}
    @media (max-width:640px) {{
      .workspace-hero {{padding:18px;margin-bottom:18px;}}
      .workspace-hero h1 {{font-size:1.7rem!important;}}
      [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {{flex-direction:row!important;flex-wrap:wrap!important;}}
      [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {{flex:1 1 42%!important;width:42%!important;min-width:120px!important;}}
      [data-testid="stMetricLabel"] {{min-height:1.5rem;}}
      [data-testid="stMetricValue"] {{font-size:1.5rem;}}
    }}
    [data-baseweb="select"]>div, [data-baseweb="input"], [data-baseweb="textarea"], textarea, input {{background:{surface}!important;color:{text}!important;}}
    button[kind="secondary"] {{background:{surface};color:{text};border-color:{border};}}
    [data-testid="stAlert"] p {{ color:{text}; }}
    </style>""", unsafe_allow_html=True)


def backend_status_badge(backend, settings, store):
    """Show runtime reachability without claiming that inference is verified."""
    model_id = settings[backend].get("model_id")
    if not model_id:
        return '<span class="backend-status status-neutral"><span class="status-dot"></span>No model selected</span>'
    if st.session_state.get(f"reachable_{backend}"):
        completed = any(
            result.get("status") == "completed" and result.get("model_id") == model_id
            for saved in store.list("run", 50)
            for result in saved["payload"].get("results", {}).values()
            if result.get("backend") == backend
        )
        label = "Runtime reachable · inference completed" if completed else "Runtime reachable · inference unverified"
        return f'<span class="backend-status status-good"><span class="status-dot"></span>{label}</span>'
    return '<span class="backend-status status-warn"><span class="status-dot"></span>Configured · refresh to check</span>'


def error(exc):
    if isinstance(exc, ValidationError):
        st.error(json.dumps(exc.errors(include_input=False, include_url=False), indent=2))
    else:
        st.error(str(exc))


def scenario_detail(scenario, evidence, decision="pending_review"):
    if 'system_decision' in scenario:
        st.write('Automatic KB screening:', scenario['system_decision'])
        st.caption('AI title is a display label. Statement support uses exact KB description matching; engineer approval is recorded separately.')
        if scenario.get('recovery_note'):
            st.info(scenario['recovery_note'])
        for reason in scenario.get('screening_reasons', []):
            st.error(reason)
        if not scenario.get('schema_valid'):
            st.json(scenario)
            return
    if 'citation_link_validity' in scenario:
        st.caption(f"citation_link_validity: {scenario['citation_link_validity'] if scenario['citation_link_validity'] is not None else 'not measured'} · reviewed support rate: {scenario.get('reviewed_support_rate') if scenario.get('reviewed_support_rate') is not None else 'not measured'} · needs human review: {scenario.get('needs_human_review', True)}")
    st.subheader(scenario["title"])
    st.caption(f"{scenario['scenario_id']} · AI proposal · latest human status: {decision.replace('_', ' ')}")
    st.write(scenario["scenario_narrative"])
    conditions, hazards, sources, links, checks = st.tabs(["Conditions", "Hazards & risk", "Evidence", "Mitigations & tests", "Verification"])
    with conditions:
        st.json(scenario["input_conditions"])
    with hazards:
        risk_assessments = scenario.get("risk_assessments") or {}
        for hazard in scenario["hazards"]:
            st.write(f"{hazard['hazard_id']} · {hazard['description']}")
            st.write("Potential consequences", hazard["consequences"])
            st.caption("Hypothesis" if hazard["hypothesis"] else "Source-backed claim proposed by model; review needed")
            assessment = risk_assessments.get(hazard["hazard_id"])
            if assessment is None:
                st.warning("Risk assessment is not available for this hazard. The scenario remains eligible for human review.")
                st.json({
                    "hazard_id": hazard["hazard_id"],
                    "status": "requires_review",
                    "risk_score": None,
                    "risk_level": None,
                    "explanation": "No matching stored risk assessment was found in this run.",
                })
            else:
                st.json(assessment)
    with sources:
        cited = {ref for h in scenario["hazards"] for ref in h["evidence_ids"]}
        for item in evidence:
            record = item["record"]
            if record["id"] in cited:
                st.write(f"{record['id']} · {record['title']}")
                st.write(record["description"])
                st.caption(f"Type: {record['type']} · review: {record['review_status']} · illustrative: {record['illustrative']} · relevance: {item.get('score')} · retrieval: {item.get('retrieval', 'snapshot')}")
                if record.get("source"):
                    st.text(record["source"])
        missing = cited - {e["record"]["id"] for e in evidence}
        if missing:
            st.error("Unresolved evidence IDs: " + ", ".join(sorted(missing)))
        if not cited:
            st.warning("No evidence cited")
    with links:
        st.write("Proposed mitigation records")
        st.json(scenario["linked_records"]["mitigation_ids"])
        st.write("Proposed acceptance-test records")
        st.json(scenario["linked_records"]["acceptance_test_ids"])
        st.caption("Record linkage is not proof of effectiveness or an executed test. Original IDs remain in the exported proposal, including invalid IDs.")
    with checks:
        verification = scenario["verification"]
        st.write("Schema valid", verification["schema_valid"])
        st.write("Evidence IDs valid", verification["evidence_valid"])
        st.write("Mitigation/test references valid", verification["references_valid"])
        for message in verification["errors"]:
            st.error(message)
        for message in verification["warnings"]:
            st.warning(message)


def choose_models(settings, prefix, selectable=True):
    chosen = []
    run_all = st.session_state.get("active_backend") == ALL_BACKENDS
    for backend, label in LABELS.items():
        cfg = settings[backend]
        inventory = st.session_state.get(f"catalogue_{backend}", [])
        ids = list(dict.fromkeys(m["id"] for m in inventory))
        persisted = cfg.get("model_id", "")
        if persisted and persisted not in ids:
            ids.insert(0, persisted)
        with st.container(border=True):
            st.markdown(f"**{label}**")
            if not ids:
                st.caption("Setup required · Refresh the real catalogue in Model Studio.")
                continue
            selected = st.selectbox("Model", ids, index=ids.index(persisted) if persisted in ids else 0, key=f"{prefix}_model_{backend}")
            st.caption("Catalogue listing does not confirm successful inference; readiness is checked on execution.")
            use_key = f"{prefix}_use_{backend}"
            if run_all:
                st.session_state[use_key] = True
            elif use_key not in st.session_state:
                st.session_state[use_key] = backend == 'laptop_local'
            use = st.checkbox("Include in run", key=use_key, disabled=run_all) if selectable else True
            if use:
                params = {'temperature': 0.0, 'seed': 42, 'max_tokens': settings.get('max_tokens', 1800)}
                chosen.append(Selection(backend=backend, model_id=selected, generation=Generation.model_validate(params)))
    return chosen


def overview(s):
    runs = s.store.list("run")
    rows = scenario_register(s.store)
    counts = review_counts(rows)
    cols = st.columns(5)
    follow_up = sum(counts[d] for d in ("revision_requested", "evidence_insufficient", "additional_testing"))
    for col, label, value in zip(cols, ["AI scenarios proposed", "Accepted", "Rejected", "Pending review", "Needs follow-up"], [counts["proposed"], counts["accepted"], counts["rejected"], counts["pending_review"], follow_up]):
        col.metric(label, value)
    cols = st.columns(4)
    cols[0].metric("KB records", len(s.dataset().records))
    cols[1].metric("Validation passed", sum(r["validation_passed"] for r in rows))
    cols[2].metric("Validation flagged / unparsed", sum(not r["validation_passed"] for r in rows))
    cols[3].metric("Backends reachable at last check", sum(st.session_state.get(f"reachable_{b}", False) for b in LABELS))
    st.caption("Scenario totals cover all persisted runs. Each proposal contributes once using its latest human decision. Validation checks do not accept or reject a scenario.")
    st.markdown('<div class="workflow-band"><span class="workflow-chip">01 · Configure models</span><span class="workflow-chip">02 · Build a scenario</span><span class="workflow-chip">03 · Compare evidence</span><span class="workflow-chip">04 · Record a decision</span></div>', unsafe_allow_html=True)
    decisions, performance, activity = st.tabs(["Review outcomes", "Model performance", "Recent activity"])
    with decisions:
        a, b = st.columns(2)
        with a:
            st.subheader("Engineering review pipeline")
            if rows:
                st.plotly_chart(review_figure(rows, st.session_state.dark), use_container_width=True)
            else:
                st.markdown('<div class="empty-panel"><strong>Your scenario review pipeline starts here</strong><p>Generated proposals will appear with accepted, rejected and pending decisions. No scenarios have been generated yet.</p></div>', unsafe_allow_html=True)
        with b:
            st.subheader("Knowledge coverage")
            categories = Counter(r.type.replace("_", " ") for r in s.dataset().records)
            figure = px.bar(x=list(categories.values()), y=list(categories), orientation="h", labels={"x":"Records", "y":""}, color_discrete_sequence=["#6366f1"])
            from ui.charts import polish
            st.plotly_chart(polish(figure, st.session_state.dark), use_container_width=True)
    with performance:
        performance_panel(s)
    with activity:
        if runs:
            st.dataframe([{"created": r["created"], "run_id": r["id"], "status": r["payload"]["status"], "backends": ", ".join(r["payload"]["results"])} for r in runs[:15]], use_container_width=True)
        else:
            st.info("No executions yet. Configure a runtime in Settings, refresh models in Model Studio, then use Scenario Builder.")


def performance_panel(s):
    rows = performance_rows(s.store)
    if not rows:
        st.markdown('<div class="empty-panel"><strong>Model performance, backed by actual runs</strong><p>Generation latency, validation results and human decisions will appear after inference. No performance measurements are available yet.</p></div>', unsafe_allow_html=True)
        return
    st.caption("Historical latency medians include completed executions with measurements. Workloads and generation settings may differ; use a single comparison run for matched inputs. Counts and speed are not quality rankings.")
    a, b = st.columns(2)
    a.plotly_chart(latency_figure(rows, st.session_state.dark, historical=True), use_container_width=True, key="historic_latency")
    b.plotly_chart(outcome_figure(rows, st.session_state.dark), use_container_width=True, key="historic_outcomes")
    st.plotly_chart(validation_figure(rows, st.session_state.dark), use_container_width=True, key="historic_validation")
    st.dataframe(rows, use_container_width=True)
    st.download_button("Export measured model performance", csv_bytes(rows), "model-performance.csv", key="performance_csv")


def draft_controls(prefix):
    mode = st.radio('Generation workflow', ['scenario_drafts', 'record_selection'],
                    format_func=lambda value: 'AI writes scenarios → KB screening → engineer review' if value == 'scenario_drafts' else 'Record selection (earlier workflow)',
                    key=f'{prefix}_generation_mode')
    limit = st.number_input('Maximum distinct scenarios per model', 1, 20, 10, key=f'{prefix}_scenario_limit', disabled=mode != 'scenario_drafts')
    if mode == 'scenario_drafts':
        st.caption('AI combines cited hazard, mitigation and test statements. Automatic acceptance requires exact KB description matches and valid links. Unsupported wording and duplicate record combinations are rejected. Coverage, evidence and the token limit determine how many scenarios can be produced.')
    return {'generation_mode': mode, 'max_scenarios': limit}


def risk_selection_panel(run, reviews, backend=None):
    results = recalculate(run, reviews)
    if backend:
        results = {backend: results[backend]} if backend in results else {}
    if not results:
        return
    st.subheader('Risk calculated from accepted scenarios')
    st.caption('The rule-based system calculates risk immediately from severity and likelihood stored in the knowledge base. AI does not assign risk. Engineer review happens afterward: rejecting or excluding a scenario removes it and recalculates the result.')
    for key, value in results.items():
        with st.container(border=True):
            st.markdown(f"**{LABELS.get(key, key)}**")
            a, b, c, d = st.columns(4)
            a.metric('Included scenarios', len(value['included']))
            b.metric('Engineer excluded', len(value['engineer_excluded']))
            c.metric('Overall risk', (value['risk_level'] or 'not available').upper())
            d.metric('Risk score', value['risk_score'] if value['risk_score'] is not None else 'not available')
            if value['status'] == 'provisional_needs_engineer_review':
                st.warning('Provisional risk is available now. Complete the engineer review below and remove any unwanted scenarios; the result will recalculate automatically.')
            elif value['status'] == 'risk_input_missing':
                st.error('Risk cannot be calculated because severity or likelihood is missing or conflicting in the knowledge base.')
            elif value['status'] == 'no_included_scenarios':
                st.info('No scenarios remain in the calculation. This means no risk result, not low risk.')
            else:
                st.success('Risk calculated from approved knowledge-base inputs.')
            for step in value.get('calculation_steps', []):
                st.write(step)
            if value.get('uncovered_faults'):
                st.warning(value['coverage_note'] + '. The displayed risk covers the remaining included scenarios only.')
            st.caption('Combined faults use the highest included hazard score. Undocumented interactions do not receive an invented risk multiplier.')
    st.dataframe([{'backend': LABELS.get(key, key), 'proposed': value['proposed'],
                   'system accepted': value['system_accepted'], 'system rejected': value['system_rejected'],
                   'engineer excluded': len(value['engineer_excluded']), 'included scenarios': len(value['included']),
                   'unique hazards': len(value['hazard_ids']), 'risk': value['risk_level'] or 'unassigned',
                   'citation validity': value['citation_link_validity'], 'reviewed support': value['reviewed_support_rate'],
                   'score': value['risk_score'], 'status': value['status'].replace('_', ' ')} for key, value in results.items()],
                 use_container_width=True, hide_index=True)
    with st.expander('Technical details: included IDs, exclusions and KB risk records'):
        st.json(results)


def register_page(s):
    rows = scenario_register(s.store)
    counts = review_counts(rows)
    columns = st.columns(4)
    for column, label, value in zip(columns, ["All proposed", "Accepted", "Rejected", "Pending review"], [counts["proposed"],counts["accepted"],counts["rejected"],counts["pending_review"]]):
        column.metric(label, value)
    st.caption("Review one frozen run from evidence to model output to engineering decision. Failed and truncated model runs remain visible even when they produced no proposal.")

    saved_runs = s.store.list("run")
    if saved_runs:
        run_options = {f"{item['created']} · {item['id']}": item for item in saved_runs}
        run_label = st.selectbox("Run to review", list(run_options), key="register_run")
        saved = run_options[run_label]
        run = saved["payload"]
        evidence = run.get("snapshot", {}).get("evidence", [])
        if not evidence:
            st.warning('This saved run retrieved no KB evidence. Its rejected scenarios remain in history. Start a new run using the current build and active knowledge base.')
        results = run.get("results", {})
        input_data = run.get("snapshot", {}).get("request", {}).get("inputs", {})

        st.markdown('<div class="workflow-band"><span class="workflow-chip">1 · Inspect TF-IDF evidence</span><span class="workflow-chip">2 · Compare model output</span><span class="workflow-chip">3 · Record engineering decision</span></div>', unsafe_allow_html=True)
        evidence_tab, output_tab, decision_tab = st.tabs(["1 · TF-IDF evidence", "2 · Model output", "3 · Engineer decision"])

        with evidence_tab:
            st.caption("This is the exact filtered and ranked evidence snapshot shared by every model in this run. Relevance is not engineering approval.")
            if input_data:
                with st.expander("Scenario input", expanded=False):
                    st.json(input_data)
            if evidence:
                evidence_rows = []
                for item in evidence:
                    record = item.get("record", {})
                    evidence_rows.append({
                        "rank": len(evidence_rows) + 1,
                        "record_id": record.get("id"),
                        "type": record.get("type"),
                        "title": record.get("title"),
                        "tfidf_score": item.get("score"),
                        "severity": record.get("severity") or "not applicable",
                        "likelihood": record.get("likelihood") or "not applicable",
                        "record_review": record.get("review_status"),
                        "expert_support": record.get("expert_support_review", "not_reviewed"),
                        "reviewed_by": record.get("reviewed_by") or "not reviewed",
                        "illustrative": record.get("illustrative", False),
                    })
                st.dataframe(evidence_rows, use_container_width=True, hide_index=True)
                record_options = {f"{r['record'].get('id')} · {r['record'].get('title')}": r for r in evidence}
                evidence_choice = st.selectbox("Preview retrieved record", list(record_options), key=f"evidence_{saved['id']}")
                chosen_evidence = record_options[evidence_choice]
                a, b, c = st.columns(3)
                a.metric("TF-IDF rank", list(record_options).index(evidence_choice) + 1)
                chosen_score = chosen_evidence.get("score")
                b.metric("Relevance score", f"{chosen_score:.4f}" if isinstance(chosen_score, (int, float)) else "not measured")
                c.metric("Record status", chosen_evidence["record"].get("review_status", "not recorded").replace("_", " "))
                st.write(chosen_evidence["record"].get("description", "No description"))
                with st.expander("Full record and retrieval details"):
                    st.json(chosen_evidence)
                st.info("Approve or correct reusable evidence in Knowledge Base. Proposal decisions below apply only to a model's use of this evidence in this run.")
            else:
                st.warning("This saved run contains no retrievable evidence snapshot.")

        backend = None
        result = None
        entry = None
        scoped_row = None
        if results:
            backend = st.session_state.get(f"register_backend_{saved['id']}")
            if backend not in results:
                backend = next(iter(results))

        with output_tab:
            if not results:
                st.warning("No backend results were recorded for this run.")
            else:
                backend = st.selectbox("Model result", list(results), format_func=lambda value: f"{LABELS.get(value, value)} · {results[value].get('model_id', 'unrecorded')}", key=f"register_backend_{saved['id']}")
                result = results[backend]
                a, b, c, d = st.columns(4)
                a.metric("Execution", result.get("status", "unknown"))
                b.metric("Schema valid", "Yes" if result.get("schema_valid") else "No")
                c.metric("Proposals", len(result.get("scenarios", [])))
                if result.get('recovered_candidate_count'):
                    st.caption(f"AI drafts: {result['ai_candidate_count']}. Additional KB recovery scenarios: {result['recovered_candidate_count']}. Recovery is deterministic and is not AI output.")
                latency = result.get("generation_latency_ms")
                d.metric("Generation time", f"{latency / 1000:.1f} s" if latency is not None else "not measured")
                if result.get("error"):
                    st.error(result["error"])
                st.caption(f"Requested settings: {result.get('run_settings') or result.get('requested_settings') or 'not recorded'} · applied settings: {result.get('applied_settings', 'not measured')}")
                if result.get('requested_runtime_options'):
                    st.caption(f"Runtime options requested: {result['requested_runtime_options']} · reported token usage: {result.get('runtime_metrics') or 'not measured'}")
                model_entries = [item for item in proposal_entries(run) if item["backend"] == backend]
                if model_entries:
                    entry_options = {f"{item['scope']} · {item['candidate'].get('title', 'Untitled proposal')}": item for item in model_entries}
                    entry_label = st.selectbox("Proposal preview", list(entry_options), key=f"register_proposal_{saved['id']}_{backend}")
                    entry = entry_options[entry_label]
                    scoped_row = next((row for row in rows if row["run_id"] == saved["id"] and row["review_scope"] == entry["scope"]), None)
                    if entry["schema_valid"]:
                        scenario_detail(entry["candidate"], evidence, scoped_row["decision"] if scoped_row else "pending_review")
                    else:
                        st.warning("The model returned an identifiable candidate, but it failed schema or evidence verification.")
                        st.json(entry["candidate"])
                else:
                    st.warning("This model produced no reviewable proposal. Inspect the error and attempt diagnostics, then rerun after correcting the model settings.")
                with st.expander("Model diagnostics and complete normalized result"):
                    st.json(result)

        with decision_tab:
            if not results:
                st.info("There is no model output to review in this run.")
            else:
                selected_backend = st.session_state.get(f"register_backend_{saved['id']}", backend)
                selected_result = results.get(selected_backend, result or {})
                model_entries = [item for item in proposal_entries(run) if item["backend"] == selected_backend]
                if not model_entries:
                    st.info("No proposal can be approved or rejected because this model produced no identifiable scenario. The failed execution remains visible in Model Output.")
                else:
                    selected_scope = st.session_state.get(f"register_proposal_{saved['id']}_{selected_backend}")
                    entry = next((item for item in model_entries if selected_scope == f"{item['scope']} · {item['candidate'].get('title', 'Untitled proposal')}"), model_entries[0])
                    scoped_row = next((row for row in rows if row["run_id"] == saved["id"] and row["review_scope"] == entry["scope"]), None)
                    current = scoped_row["decision"] if scoped_row else "pending_review"
                    st.markdown(f"**Reviewing:** {LABELS.get(selected_backend, selected_backend)} · {selected_result.get('model_id', 'unrecorded')} · {entry['scope']}")
                    st.caption(f"Current decision: {current.replace('_', ' ')}. The provisional risk below is calculated before this review. Reject or request follow-up to remove this scenario from the calculation; the result updates immediately.")
                    reviewer = st.text_input("Engineer / reviewer name", key=f"register_reviewer_{saved['id']}_{entry['scope']}")
                    decisions = [d for d in DECISIONS if d != "pending_review" and (d != "accepted" or (entry["schema_valid"] and entry['candidate'].get('system_decision') != 'rejected' and not entry["candidate"].get("needs_human_review")))]
                    decision = st.selectbox("Engineering decision", decisions, format_func=lambda value: value.replace("_", " ").title(), key=f"register_decision_{saved['id']}_{entry['scope']}")
                    notes = st.text_area("Review rationale / required follow-up", key=f"register_notes_{saved['id']}_{entry['scope']}")
                    if st.button("Save engineering decision", disabled=not reviewer.strip(), type="primary", key=f"save_review_{saved['id']}_{entry['scope']}"):
                        s.store.review(saved["id"], entry["scope"], decision, reviewer, notes)
                        st.rerun()
                    with st.expander("Human review audit trail"):
                        st.json([review for review in s.store.reviews(saved["id"]) if review["scenario_id"] == entry["scope"]])
                risk_selection_panel(run, s.store.reviews(saved['id']), selected_backend)
    else:
        st.info("No saved runs yet. Generate a scenario or comparison first; both successful and failed model executions will appear here.")

    st.divider()
    st.subheader("All proposal decisions")
    st.caption("Compact register across all runs. Use the review workspace above for evidence and model-level inspection.")
    search = st.text_input("Search all proposed scenarios", placeholder="Scenario ID, title, model, backend or reviewer")
    statuses = st.multiselect("Human decision filter", list(DECISIONS))
    backends = st.multiselect("Backend filter", list(LABELS), format_func=LABELS.get)
    selected = [r for r in rows if (not statuses or r["decision"] in statuses) and (not backends or r["backend"] in backends)
                and search.lower() in " ".join(str(v) for v in r.values()).lower()]
    st.caption(f"{len(selected)} of {len(rows)} proposals match these filters")
    st.download_button("Export all proposed scenarios CSV", csv_bytes(rows), "all-proposed-scenarios.csv")
    st.download_button("Export filtered proposals JSON", json_bytes(selected), "filtered-proposals.json")
    if not selected:
        st.info("No proposals match. Generate scenarios or adjust the filters.")
        return
    page = st.number_input("Register page (50 proposals per page)", 1, max(1, (len(selected)+49)//50), 1)
    st.dataframe(selected[(page-1)*50:page*50], use_container_width=True, hide_index=True)


def scenario_builder(s):
    settings, dataset = s.settings(), s.dataset()
    saved = s.store.setting("scenario_input", ScenarioInput().model_dump())
    st.caption("Define operating conditions, inspect the shared evidence snapshot, then select the models to run. Choose All backends in the sidebar to preselect every configured backend.")
    with st.form("scenario_form"):
        a, b, c = st.columns(3)
        time_of_day = a.selectbox("Time of day", ["day", "night"], index=["day", "night"].index(saved["time_of_day"]))
        weather = b.selectbox("Weather", ["dry", "wet", "rain"], index=["dry", "wet", "rain"].index(saved["weather"]))
        visibility = c.selectbox("Visibility", ["clear", "reduced"], index=["clear", "reduced"].index(saved["visibility"]))
        surface = a.text_input("Surface", saved["surface"])
        density = b.selectbox("Pedestrian density", ["low", "medium", "high"], index=["low", "medium", "high"].index(saved["pedestrian_density"]))
        speed = c.number_input("Speed (m/s)", 0.0, 5.0, float(saved["speed_mps"]), 0.1)
        location = a.text_input("Operating context", saved["location"])
        navigation = b.text_input("Navigation mode", saved["navigation_mode"])
        sensors = c.text_input("Sensors (comma separated)", ", ".join(saved["sensors"]))
        faults_available = list(FAULT_LABELS)
        faults = st.multiselect("Fault types (select 1 to 3)", faults_available, default=[f for f in saved["faults"] if f in faults_available][:3], max_selections=3)
        capabilities = st.text_area("Documented robot capabilities (include source identifiers)", saved["documented_capabilities"])
        notes = st.text_area("Additional context / unsupported faults", saved["notes"])
        save = st.form_submit_button("Validate and save input", type="primary")
    if save:
        try:
            inputs = ScenarioInput(time_of_day=time_of_day, weather=weather, visibility=visibility, surface=surface,
                                   pedestrian_density=density, speed_mps=speed, location=location, navigation_mode=navigation,
                                   sensors=[v.strip() for v in sensors.split(",") if v.strip()], faults=faults, documented_capabilities=capabilities, notes=notes)
            if not 1 <= len(inputs.faults) <= 3:
                raise ValueError('Select between one and three fault types before saving.')
            s.store.set_setting("scenario_input", inputs.model_dump())
            st.success("Input validated and saved")
        except (ValidationError, ValueError) as exc:
            error(exc)
    if st.button("Reset saved input"):
        s.store.set_setting("scenario_input", ScenarioInput().model_dump())
        st.rerun()
    inputs = ScenarioInput.model_validate(s.store.setting("scenario_input", ScenarioInput().model_dump()))
    with st.expander("Validated request and retrieved evidence"):
        st.json(inputs.model_dump())
        st.json(retrieve(dataset, inputs.model_dump_json(), settings["retrieval_limit"], settings["relevance_threshold"]))
    selections = choose_models(settings, "builder")
    generation_options = draft_controls('builder')
    if st.button('Run no-AI baseline'):
        if dataset.schema_version != '2.0':
            st.error('Activate a tagged v2 knowledge base first')
        else:
            evidence = retrieve(dataset, inputs.model_dump(), settings['retrieval_limit'], settings['relevance_threshold'])
            st.json(baseline_result(evidence, s.matrix(), settings.get('citation_link_validity_threshold', 1.0)).model_dump())
    if st.button("Run selected models", type="primary", disabled=not selections):
        try:
            request = RunRequest(inputs=inputs, selections=selections, retrieval_limit=settings["retrieval_limit"], relevance_threshold=settings["relevance_threshold"], citation_link_validity_threshold=settings.get('citation_link_validity_threshold', 1.0), **generation_options)
            st.session_state.active_run = s.manager.start(request, dataset, s.matrix(), settings)
            st.success("Run started. Progress and results appear below and in Model Comparison.")
        except (ValidationError, ValueError) as exc:
            error(exc)
    run_progress(s)


def model_studio(s):
    settings = s.settings()
    for backend, label in LABELS.items():
        with st.container(border=True):
            st.subheader(label)
            st.caption(settings[backend]["base_url"] or "Address not configured")
            if st.button("Refresh models", key=f"refresh_{backend}"):
                try:
                    adapter = make_adapter(backend, settings)
                    inventory = adapter.list_models(refresh=True)
                    st.session_state[f"catalogue_{backend}"] = inventory
                    st.session_state[f"reachable_{backend}"] = True
                    st.success(f"Service reached; {len(inventory)} models listed. Inference remains unverified.")
                except BackendError as exc:
                    st.session_state[f"catalogue_{backend}"] = []
                    st.session_state[f"reachable_{backend}"] = False
                    error(exc)
            inventory = st.session_state.get(f"catalogue_{backend}", [])
            if inventory:
                ids = list(dict.fromkeys(x["id"] for x in inventory))
                persisted = settings[backend].get("model_id")
                selected = st.selectbox("Select model", ids, index=ids.index(persisted) if persisted in ids else 0, key=f"studio_{backend}")
                with st.expander("Model details / runtime capabilities"):
                    st.json(next(x for x in inventory if x["id"] == selected))
                    st.json(make_adapter(backend, settings).capabilities)
                a, b = st.columns(2)
                if a.button("Save selection", key=f"save_{backend}"):
                    settings[backend]["model_id"] = selected
                    s.store.set_setting("preferences", settings)
                    st.success("Selection saved independently for this backend")
                if b.button("Test real scenario inference", key=f"test_{backend}"):
                    request = RunRequest(inputs=ScenarioInput.model_validate(s.store.setting('scenario_input', ScenarioInput().model_dump())), selections=[Selection(backend=backend, model_id=selected, generation=Generation(max_tokens=settings.get('max_tokens', 1800)))], generation_mode='scenario_drafts')
                    st.session_state.active_run = s.manager.start(request, s.dataset(), s.matrix(), settings)
                st.button("Load model (runtime unsupported)", key=f"load_{backend}", disabled=True)
                st.button("Unload model (runtime unsupported)", key=f"unload_{backend}", disabled=True)
            else:
                st.info("No verified catalogue loaded. Configure the endpoint in Settings, then refresh.")
            successes = [r for run in s.store.list("run", 50) for r in run["payload"]["results"].values() if r["backend"] == backend and r["status"] == "completed"]
            if successes:
                st.caption(f"Last completed inference: {successes[0]['finished']} · {successes[0]['model_id']}")
    st.subheader("Local GGUF inventory")
    st.caption("Files are separate from models advertised by the running llama.cpp server. File presence does not verify compatibility.")
    if st.button("Scan permitted model directories"):
        st.session_state.local_files = discover_files(settings.get("model_dirs", []))
    st.dataframe(st.session_state.get("local_files", []), use_container_width=True)
    manual = st.text_input("Register installed GGUF file inside a permitted model directory")
    if st.button("Register file", disabled=not manual):
        p = Path(manual).expanduser().resolve()
        roots = [Path(d).expanduser().resolve() for d in settings.get("model_dirs", [])]
        if p.suffix.lower() != ".gguf" or not p.is_file() or not any(p.is_relative_to(root) for root in roots):
            st.error("File must exist inside a configured permitted directory and have a .gguf extension")
        else:
            files = s.store.setting("registered_files", [])
            if str(p) not in files:
                s.store.set_setting("registered_files", files + [str(p)])
            st.success("File registered, runtime unverified. Start llama-server with this path, then refresh its catalogue.")
    st.write("Registered files", s.store.setting("registered_files", []))
    run_progress(s)


@st.fragment(run_every=1)
def run_progress(s):
    identity = st.session_state.get("active_run")
    if not identity:
        return
    try:
        run = s.manager.get(identity)
    except KeyError:
        st.warning("Run unavailable in this process; reopen a saved run from history")
        return
    st.subheader("Execution")
    st.caption(identity)
    for backend, events in run["progress"].items():
        st.write(f"{LABELS[backend]} · {events[-1]['stage']} · {events[-1]['at']}")
    if run["status"] == "running":
        if st.button("Cancel running comparison"):
            st.info(s.manager.cancel(identity)["note"])
    if run.get("persistence_error"):
        st.error(run["persistence_error"])
    if run["results"]:
        st.dataframe(comparison_rows(run), use_container_width=True)
        columns = st.columns(len(run["results"]))
        for column, (backend, result) in zip(columns, run["results"].items()):
            with column:
                st.markdown(f"**{LABELS[backend]} · {result['model_id']}**")
                if result.get("error"):
                    st.error(result["error"])
                for scenario in result.get("scenarios", []):
                    with st.expander(scenario["title"]):
                        scenario_detail(scenario, run["snapshot"]["evidence"])
                with st.expander("Raw normalized result"):
                    st.json(result)
        risk_selection_panel(run, s.store.reviews(identity) if run['status'] != 'running' else [])
    if run["status"] != "running":
        st.download_button("Export comparison JSON", json_bytes(report(run)), f"comparison-{identity}.json", "application/json")
        st.download_button("Export comparison CSV", csv_bytes(comparison_rows(run)), f"comparison-{identity}.csv", "text/csv")
        with st.expander("Frozen configuration, evidence and stage timestamps"):
            st.json(run["snapshot"])


def comparison(s):
    st.caption(LIMITATIONS[-1])
    latest_evaluations = [r['payload'] for r in s.store.list('evaluation', 10) if r['payload'].get('summary')]
    if latest_evaluations:
        st.subheader('Labelled evaluation: baseline and measured models')
        evaluation_summary(latest_evaluations[0], 'comparison_eval')
    st.caption("Each model receives the same saved input and frozen evidence. Results are measured independently; no automatic quality ranking is applied.")
    selections = choose_models(s.settings(), "comparison")
    generation_options = draft_controls('comparison')
    if st.button("Run comparison", disabled=not selections, type="primary"):
        try:
            settings = s.settings()
            inputs = ScenarioInput.model_validate(s.store.setting("scenario_input", ScenarioInput().model_dump()))
            st.session_state.active_run = s.manager.start(RunRequest(inputs=inputs, selections=selections, retrieval_limit=settings["retrieval_limit"], relevance_threshold=settings["relevance_threshold"], citation_link_validity_threshold=settings.get('citation_link_validity_threshold', 1.0), **generation_options), s.dataset(), s.matrix(), settings)
        except ValueError as exc:
            error(exc)
    if st.button("Clear displayed results"):
        st.session_state.pop("active_run", None)
    matched, historical = st.tabs(["Matched comparison charts", "Historical model performance"])
    with matched:
        saved_runs = s.store.list("run")
        if saved_runs:
            identity = st.selectbox("Saved run for graphical comparison", [r["id"] for r in saved_runs])
            run = s.store.get(identity)["payload"]
            measured = comparison_rows(run)
            st.caption("Matched inputs, evidence and risk policy within this run. Milliseconds are actual measurements; a fast failure is not a successful generation.")
            a, b = st.columns(2)
            a.plotly_chart(latency_figure(measured, st.session_state.dark), use_container_width=True, key="matched_latency")
            b.plotly_chart(validation_figure(measured, st.session_state.dark), use_container_width=True, key="matched_validation")
            scoped_rows = scenario_register(s.store, identity)
            model_rows = [{**r, **review_counts([p for p in scoped_rows if p["backend"] == r["backend"]])} for r in measured]
            st.plotly_chart(outcome_figure(model_rows, st.session_state.dark), use_container_width=True, key="matched_decisions")
            st.dataframe(measured, use_container_width=True)
        else:
            st.info("Run a model comparison to see measured latency, proposal validation and human-decision charts.")
    with historical:
        performance_panel(s)
    run_progress(s)


def knowledge(s):
    dataset = s.dataset()
    if dataset.schema_version != '2.0':
        st.warning('Legacy catalogue retained for history. Constrained generation requires reviewed, tagged v2 records.')
    st.download_button('Download 10-record v2 draft', (ROOT / 'data/constrained_knowledge_base.json').read_bytes(), 'constrained_knowledge_base.json')
    if st.button('Activate 10-record constrained draft'):
        s.activate_dataset(parse_upload((ROOT / 'data/constrained_knowledge_base.json').read_bytes()))
        st.rerun()
    st.caption(f"Active: {dataset.dataset_name} · version {dataset.version} · {len(dataset.records)} records")
    with st.expander("TF-IDF search inspector and measured retrieval benchmark", expanded=True):
        query = st.text_input("Search engineering evidence", "camera failure stale perception controlled stop")
        a, b = st.columns(2)
        search_limit = a.slider("Direct result limit", 1, 40, 5)
        search_threshold = b.slider("Relevance threshold", 0.0, 1.0, 0.05, 0.01)
        if st.button("Search evidence", type="primary"):
            inputs = ScenarioInput.model_validate(s.store.setting('scenario_input', ScenarioInput(faults=['Camera failure']).model_dump()))
            if dataset.schema_version == '2.0':
                inputs = inputs.model_copy(update={'notes': query})
            hits, diagnostics = retrieve_with_diagnostics(dataset, inputs.model_dump() if dataset.schema_version == '2.0' else query, search_limit, search_threshold)
            st.session_state.search_inspection = {"dataset_version": dataset.version, "results": hits, "diagnostics": diagnostics}
        inspection = st.session_state.get("search_inspection")
        if inspection and inspection["dataset_version"] == dataset.version:
            diagnostics = inspection["diagnostics"]
            cols = st.columns(3)
            cols[0].metric("Measured search total", f"{diagnostics['total_ms']:.2f} ms")
            cols[1].metric("Direct matches", diagnostics["direct_matches"])
            cols[2].metric("Index", 'No eligible records' if not diagnostics['indexed_records'] else "Reused" if diagnostics["cache_reused"] else "Built")
            st.caption("Timing includes content-key calculation and retrieval. First builds additionally fit the sparse word/bigram index. Related records have no invented lexical score.")
            direct = [r for r in inspection["results"] if r["retrieval"] == "tfidf"]
            if direct:
                st.plotly_chart(px.bar(x=[r["score"] for r in direct], y=[r["record"]["title"] for r in direct], orientation="h", labels={"x":"TF-IDF cosine similarity", "y":"Record"}, template="plotly_dark" if st.session_state.dark else "plotly_white"), use_container_width=True)
                st.dataframe([{"id":r["record"]["id"],"title":r["record"]["title"],"type":r["record"]["type"],"score":r["score"],"retrieval":r["retrieval"]} for r in inspection["results"]], use_container_width=True)
            else:
                st.info("No positive lexical matches met this threshold. Evidence is insufficient.")
            with st.expander("Search diagnostics and full evidence"):
                st.json(inspection)
        if st.button("Run lexical relevance and speed benchmark"):
            try:
                measured = benchmark_retrieval(dataset)
                s.store.save("retrieval_benchmark", measured)
                st.session_state.retrieval_benchmark = measured
            except ValueError as exc:
                error(exc)
        saved_benchmarks = s.store.list("retrieval_benchmark", 1)
        benchmark = st.session_state.get("retrieval_benchmark", saved_benchmarks[0]["payload"] if saved_benchmarks else None)
        if benchmark and benchmark["dataset_version"] == dataset.version:
            cols = st.columns(3)
            cols[0].metric("Lexical hit@5", f"{benchmark['hit_at_5']:.1%}")
            cols[1].metric("Warm median total", f"{benchmark['warm_p50_total_ms']:.2f} ms")
            cols[2].metric("Warm p95 total", f"{benchmark['warm_p95_total_ms']:.2f} ms")
            st.caption(benchmark["limitations"])
            st.dataframe(benchmark["cases"], use_container_width=True)
            if benchmark.get("synthetic_load_scaling"):
                rows = [{"Records":r["record_count"], "Duration ms":r[field], "Measurement":label} for r in benchmark["synthetic_load_scaling"]
                        for field,label in (("warm_p50_total_ms","Warm median"),("warm_p95_total_ms","Warm p95"),("first_request_total_ms","First request"))]
                st.plotly_chart(px.line(rows,x="Records",y="Duration ms",color="Measurement",markers=True,
                                      title="Measured synthetic index-load scaling (not engineering data)",
                                      template="plotly_dark" if st.session_state.dark else "plotly_white"),use_container_width=True)
            st.download_button("Export measured retrieval benchmark", json_bytes(benchmark), "retrieval-benchmark.json")
    expanded_data = parse_upload((ROOT / "data/expanded_knowledge_base.json").read_bytes())
    st.download_button("Download expanded 181-record catalogue", json_bytes(expanded_data.model_dump()), "expanded-knowledge-base.json")
    if dataset.version != expanded_data.version:
        if st.button("Activate expanded bundled catalogue"):
            s.activate_dataset(expanded_data)
            st.rerun()
    upload = st.file_uploader("Upload knowledge base JSON (maximum 5 MiB)", type=["json"], key="kb_upload")
    if upload:
        try:
            candidate = parse_upload(upload.getvalue())
            if candidate.schema_version == '1.0':
                st.warning('Legacy file: missing v2 scope/provenance contract. It can be inspected but cannot run constrained inference.')
            unresolved = [r.id for r in candidate.records if r.type == 'hazard' and (not r.severity or not r.likelihood or r.risk_conflict or not r.reviewed_by or r.review_status != 'approved' or r.illustrative)]
            if unresolved:
                st.warning('Risk requires expert review for: ' + ', '.join(unresolved))
            st.success(f"Schema and references valid: {candidate.dataset_name} · {len(candidate.records)} records · version {candidate.version}")
            st.json(dict(Counter(r.type for r in candidate.records)))
            st.dataframe([r.model_dump() for r in candidate.records[:50]], use_container_width=True)
            st.download_button("Download validation report", json_bytes({"valid": True, "version": candidate.version, "count": len(candidate.records), "note": "Structural validity is not approval of safety content."}), "kb-validation.json")
            confirmed = st.checkbox("Activate this validated dataset", key="activate_upload_confirm")
            if st.button("Activate uploaded dataset", disabled=not confirmed):
                s.activate_dataset(candidate)
                st.rerun()
        except (ValueError, ValidationError) as exc:
            error(exc)
            st.download_button("Download validation failure", json_bytes({"valid": False, "errors": str(exc)[:12000]}), "kb-validation-errors.json")
    a, b = st.columns(2)
    a.download_button("Download JSON schema", json_bytes(ConstrainedDataset.model_json_schema()), "knowledge-base.schema.json", "application/json")
    b.download_button("Download illustrative sample", (ROOT / "data/sample_knowledge_base.json").read_bytes(), "sample-knowledge-base.json", "application/json")
    query = st.text_input("Search records")
    kind = st.selectbox("Record type", ["all"] + sorted({r.type for r in dataset.records}))
    records = [r for r in dataset.records if (kind == "all" or r.type == kind) and query.lower() in f"{r.id} {r.title} {r.description}".lower()]
    st.dataframe([r.model_dump() for r in sorted(records, key=lambda r: r.id)], use_container_width=True)
    with st.expander("Add, edit or delete records as a new dataset version"):
        st.caption("Edit the JSON, including record additions or deletions. Full validation checks all references before activation. Deleting referenced records requires updating their relationships.")
        edited = st.text_area("Dataset JSON", dataset.model_dump_json(indent=2), height=320, key=f"edit_kb_{dataset.version}")
        confirm = st.checkbox("Confirm record changes, including any deletions")
        if st.button("Validate and activate edited dataset", disabled=not confirm):
            try:
                s.activate_dataset(parse_upload(edited.encode()))
                st.rerun()
            except (ValidationError, ValueError) as exc:
                error(exc)
    versions = s.store.list("knowledge_base")
    with st.expander("Version history and restoration"):
        version = st.selectbox("Saved dataset", [v["id"] for v in versions], format_func=lambda i: next(f"{v['created']} · {v['payload']['dataset_name']}" for v in versions if v["id"] == i))
        if st.button("Restore selected version"):
            s.store.set_setting("active_kb", version)
            st.rerun()
    st.download_button("Export active dataset", json_bytes(dataset.model_dump()), f"knowledge-base-{dataset.version}.json", "application/json")


def select_run(s, prefix):
    runs = s.store.list("run")
    if not runs:
        st.info("No saved runs yet")
        return None
    identity = st.selectbox("Saved run", [r["id"] for r in runs], format_func=lambda i: next(f"{r['created']} · {i[:10]}" for r in runs if r["id"] == i), key=f"{prefix}_run")
    return s.store.get(identity)["payload"]


def risk_analysis(s):
    run = select_run(s, "risk")
    if not run:
        return
    if run['snapshot'].get('prompt_version') == '3.0-scenario-drafts':
        risk_selection_panel(run, s.store.reviews(run['id']))
        st.info('Exclude or reject scenarios in Scenario Register → Engineer decision. These results recalculate from the latest decisions.')
        return
    entries = {f"{b}:{scenario['scenario_id']}:{h['hazard_id']}": (scenario, h) for b, result in run["results"].items() for scenario in result.get("scenarios", []) for h in scenario["hazards"]}
    if not entries:
        st.info("This run has no parsed hazards")
        return
    key = st.selectbox("Hazard", list(entries))
    scenario, hazard = entries[key]
    st.write(hazard["description"])
    if run['snapshot'].get('prompt_version') == '2.0-record-selection':
        st.caption('Severity and likelihood come from the frozen expert record. Correct the KB and rerun to change them. Risk categories are engineering rankings, not calibrated probabilities.')
        assessment = (scenario.get('risk_assessments') or {}).get(hazard['hazard_id'])
        if assessment is None:
            st.warning("Risk assessment is not available for this hazard in the saved run; review the KB inputs and rerun.")
            st.json({
                "hazard_id": hazard["hazard_id"],
                "status": "requires_review",
                "risk_score": None,
                "risk_level": None,
                "explanation": "No matching stored risk assessment was found in this run.",
            })
        else:
            st.json(assessment)
        return
    matrix = RiskMatrix.model_validate(run["snapshot"]["risk_matrix"])
    st.json(matrix.model_dump())
    st.caption("Use the matrix frozen for this run. Severity and likelihood require a named reviewer, rationale, and approved non-illustrative evidence.")
    severity = st.selectbox("Severity", [None] + list(matrix.severity), format_func=lambda x: "Unassigned" if x is None else f"{x} · {matrix.severity[x]}")
    likelihood = st.selectbox("Likelihood", [None] + list(matrix.likelihood), format_func=lambda x: "Unassigned" if x is None else f"{x} · {matrix.likelihood[x]}")
    evidence = st.multiselect("Supporting evidence IDs", [e["record"]["id"] for e in run["snapshot"]["evidence"]])
    reviewer = st.text_input("Risk reviewer")
    rationale = st.text_area("Classification rationale and project criteria")
    if st.button("Calculate and record assessment"):
        classification = Classification(severity=severity, likelihood=likelihood, evidence_ids=evidence, reviewer=reviewer, rationale=rationale)
        assessment = calculate(matrix, classification, run["snapshot"]["evidence"])
        s.store.save("risk_assessment", {"run_id": run["id"], "hazard_key": key, "classification": classification.model_dump(), "assessment": assessment})
        st.json(assessment)
    st.subheader("Recorded assessments")
    st.json([r["payload"] for r in s.store.list("risk_assessment") if r["payload"]["run_id"] == run["id"]])


def simulation_page(s):
    st.caption("A deterministic watchdog state machine. Every output is a software simulation; no physical robot is controlled.")
    rule = st.selectbox("Rule", ["watchdog", "unsupported"])
    age = st.number_input("Sensor data age (ms)", 0, 60000, 300)
    threshold = st.number_input("Stale-data threshold (ms)", 1, 60000, 200)
    communication_ok = st.checkbox("Communication available", value=True)
    enabled = st.checkbox("Watchdog enabled", value=True)
    expected = st.selectbox("Acceptance criterion: expected logical state", ["STOP", "MOVE"])
    if st.button("Execute software simulation", type="primary"):
        result = simulate(SimulationInput(rule=rule, sensor_age_ms=age, stale_threshold_ms=threshold, communication_ok=communication_ok, watchdog_enabled=enabled, expected_state=expected))
        s.store.save("simulation", result)
        st.session_state.simulation_result = result
    if st.session_state.get("simulation_result"):
        result = st.session_state.simulation_result
        st.json(result)
        st.download_button("Export simulation JSON", json_bytes(result), "software-simulation.json")
        st.download_button("Export simulation CSV", csv_bytes([{k: json.dumps(v) if isinstance(v, dict) else v for k, v in result.items()}]), "software-simulation.csv")
    with st.expander("Simulation history"):
        st.json(s.store.list("simulation", 20))


@st.fragment(run_every=2)
def benchmark_progress(s):
    identity = st.session_state.get("active_benchmark")
    if not identity:
        return
    job = s.benchmarks.get(identity)
    st.write(f"Benchmark {job['status']} · {len(job['rows'])} results")
    if job["status"] == "running" and st.button("Cancel benchmark"):
        s.benchmarks.cancel(identity)
    if job.get('error'):
        st.error(job['error'])
    if job["status"] != "running":
        evaluation_summary(job, f'active_{identity}')
        experiment_downloads(job, identity, 'active')
        if job['rows']:
            with st.expander('Inspect a measured case execution'):
                selected = st.selectbox('Case execution', range(len(job['rows'])), format_func=lambda i: f"{job['rows'][i]['case_id']} · {job['rows'][i].get('model', {}).get('backend', job['rows'][i].get('backend'))} · repeat {job['rows'][i].get('repeat', 1)}")
                st.json(job['rows'][selected])


def evaluation_page(s):
    label_file = 'multi_fault_cases_25.json' if s.dataset().dataset_id == 'multi-fault-draft-120' else 'labelled_cases_25.json'
    dataset = LabelledDataset.model_validate_json((ROOT / 'data' / label_file).read_bytes())
    st.warning('The 25 labelled cases are preliminary drafts awaiting expert review and dates. Repeated executions do not increase the case count.')
    st.download_button("Download 25-case draft dataset", json_bytes(dataset.model_dump()), "benchmark-draft-25.json")
    st.download_button("Download benchmark schema", json_bytes(LabelledDataset.model_json_schema()), "benchmark.schema.json")
    upload = st.file_uploader("Reviewed benchmark JSON (maximum 5 MiB)", type=["json"], key="benchmark_upload")
    valid_dataset = True
    if upload:
        try:
            if upload.size > 5 * 1024 * 1024:
                raise ValueError("Dataset exceeds 5 MiB")
            dataset = LabelledDataset.model_validate_json(upload.getvalue())
        except (ValueError, ValidationError) as exc:
            valid_dataset = False
            error(exc)
    repeats = st.number_input('Repetitions per case', 3, 5, 3)
    st.caption('Every experiment includes baseline, laptop, UNO Q and Qualcomm. Unconfigured adapters and timeouts count as failed runs.')
    selections = []
    confirm = st.checkbox("Execute selected cases and models; cloud requests may incur account charges")
    if st.button("Run benchmark", disabled=not confirm or not valid_dataset):
        try:
            st.session_state.active_benchmark = s.benchmarks.start_labelled(dataset, selections, s.dataset(), s.matrix(), s.settings(), repeats)
        except ValueError as exc:
            error(exc)
    st.caption('Precision and recall use exact IDs; empty denominators are not measured. Citation link validity measures valid record linkage. ' + LIMITATIONS[-1])
    benchmark_progress(s)
    with st.expander("Saved evaluations"):
        saved = s.store.list('evaluation', 10)
        if saved:
            identity = st.selectbox('Saved evaluation', [r['id'] for r in saved])
            payload = s.store.get(identity)['payload']
            evaluation_summary(payload, f'saved_{identity}')
            experiment_downloads(payload, identity, 'saved')


def experiment_downloads(job, identity, prefix):
    if 'case_count' not in job:
        st.caption('Archived evaluation format: new risk metrics and case-count metadata were not measured. Rerun the current cases to generate revised reports.')
        st.download_button('Download archived evaluation JSON', json_bytes(job), f'archived-{identity}.json', key=f'{prefix}_archived_{identity}')
        return
    st.download_button('Export experiment JSON', json_bytes(experiment_report(job)), f'experiment-{identity}.json', key=f'{prefix}_json_{identity}')
    st.download_button('Export experiment CSV', experiment_csv(job), f'experiment-{identity}.csv', key=f'{prefix}_csv_{identity}')
    if st.button('Prepare experiment PDF', key=f'{prefix}_pdf_prepare_{identity}'):
        st.session_state[f'experiment_pdf_{identity}'] = experiment_pdf_bytes(job)
    prepared = st.session_state.get(f'experiment_pdf_{identity}')
    if prepared:
        st.download_button('Download experiment PDF', prepared, f'experiment-{identity}.pdf', 'application/pdf', key=f'{prefix}_pdf_{identity}')


def evaluation_summary(job, key='summary'):
    labels = job.get('labels', {})
    st.caption(f"Label set: {labels.get('version', 'not recorded')} · reviewer: {labels.get('reviewer') or labels.get('reviewed_by') or 'not reviewed'} · reviewed date: {labels.get('reviewed_date') or 'not reviewed'}")
    st.caption('citation_link_validity counts valid ID links only. Reviewed support rate counts supported claims among expert-reviewed claims. Applied settings require a runtime report; unknown values are not measured.')
    rows, chart_rows = [], []
    for summary in job.get('summary', []):
        row = {k: summary.get(k) for k in ('backend', 'model', 'case_count', 'label', 'executions', 'failures')}
        for name, metric in summary.items():
            if not isinstance(metric, dict) or 'mean' not in metric:
                continue
            value = lambda x: 'not measured' if x is None else f'{x:.4f}'
            n = metric.get('case_count', summary.get('case_count', 'not measured'))
            row[name] = f"{value(metric['mean'])} ± {value(metric['stddev'])} [{value(metric['min'])}, {value(metric['max'])}] | n={n} cases | {metric.get('label', 'preliminary')}"
            if metric['mean'] is not None and (name.endswith(('_precision', '_recall', '_correct')) or name in {'risk_category_exact_match_rate', 'decision_accuracy'}):
                chart_rows.append({'Model': f"{summary['backend']} / {summary['model']}", 'Metric': f"{name} (n={n} cases; {metric.get('label', 'preliminary')})", 'Mean': metric['mean'], 'Stddev': metric['stddev']})
        rows.append(row)
    for backend in LABELS:
        if not any(r['backend'] == backend for r in rows):
            fields = {field for row in rows for field in row} | {'model', 'executions'}
            rows.append({**{field: 'not measured' for field in fields}, 'backend': backend})
    st.dataframe(rows, use_container_width=True, key=f'evaluation_table_{key}')
    if chart_rows:
        st.plotly_chart(px.bar(chart_rows, x='Model', y='Mean', color='Metric', barmode='group', error_y='Stddev', range_y=[0, 1], title=f"{job.get('label', 'archived').title()} accuracy metrics (unique case counts in legend)"), use_container_width=True, key=f'evaluation_chart_{key}')
    latencies = [{'Backend': r['model']['backend'], 'Status': r.get('status', 'not measured'), 'Milliseconds': r['execution_latency_ms'], 'Case': r['case_id']} for r in job.get('rows', []) if r.get('execution_latency_ms') is not None]
    if latencies:
        st.plotly_chart(px.box(latencies, x='Backend', y='Milliseconds', color='Status', points='all', hover_data=['Case'], title=f"{job.get('label', 'archived').title()} run duration, including failures (n={job.get('case_count', 'not measured')} cases)"), use_container_width=True, key=f'latency_distribution_{key}')
    with st.expander('Risk confusion matrix and measured ranges'):
        st.json(job.get('summary', []))


def history(s):
    run = select_run(s, "history")
    if not run:
        return
    st.dataframe(comparison_rows(run), use_container_width=True)
    with st.expander("Complete run detail"):
        st.json(run)
    entries = {entry["scope"]: entry["candidate"] for entry in proposal_entries(run) if entry["schema_valid"]}
    latest = s.store.latest_reviews(run["id"])
    counts = review_counts(scenario_register(s.store, run["id"]))
    st.write("Latest human decisions", counts)
    if entries:
        key = st.selectbox("Scenario to inspect and review", list(entries))
        current_review = latest.get((run["id"], key))
        scenario_detail(entries[key], run["snapshot"]["evidence"], current_review["payload"]["decision"] if current_review else "pending_review")
        st.download_button("Export individual scenario", json_bytes({"scenario": entries[key], "configuration": run["snapshot"], "review_scope": key, "human_reviews": s.store.reviews(run["id"])}), f"scenario-{run['id']}.json")
        st.download_button("Export scenario CSV", csv_bytes([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in entries[key].items()}]), f"scenario-{run['id']}.csv")
        reviewer = st.text_input("Reviewer name")
        choices = ['revision_requested', 'rejected', 'evidence_insufficient', 'additional_testing'] if entries[key].get('needs_human_review') else ['accepted', 'revision_requested', 'rejected', 'evidence_insufficient', 'additional_testing']
        decision = st.selectbox('Engineering decision', choices)
        notes = st.text_area("Review notes / mitigation and test link assessment")
        if st.button("Save human review", disabled=not reviewer.strip()):
            try:
                s.store.review(run["id"], key, decision, reviewer, notes)
                st.success("Human review saved separately from AI content")
            except ValueError as exc:
                error(exc)
    reviews = s.store.reviews(run["id"])
    risk_selection_panel(run, reviews)
    assessments = [r for r in s.store.list("risk_assessment") if r["payload"]["run_id"] == run["id"]]
    exported = report(run, reviews, assessments)
    report_signature = hashlib.sha256(json_bytes(exported)).hexdigest()
    st.download_button("Export full engineering report JSON", json_bytes(exported), f"report-{run['id']}.json", "application/json")
    if st.button("Prepare engineering PDF"):
        with st.spinner("Preparing paginated engineering report"):
            st.session_state.pdf_report = (report_signature, pdf_bytes(exported))
    prepared_pdf = st.session_state.get("pdf_report")
    if prepared_pdf and prepared_pdf[0] == report_signature:
        st.download_button("Download engineering PDF", prepared_pdf[1], f"report-{run['id']}.pdf", "application/pdf")
    st.download_button("Export comparison CSV", csv_bytes(comparison_rows(run, reviews)), f"report-{run['id']}.csv", "text/csv")
    st.download_button("Export reviewed scenario collection", json_bytes({"run": run, "reviews": reviews}), f"reviewed-collection-{run['id']}.json")
    st.caption("JSON retains the complete source dataset and exact Unicode content. PDF provides a paginated engineering report with visible character escapes where needed.")


def device(s):
    settings = s.settings()
    st.caption("Connect to the authenticated Linux agent over local Wi-Fi. Initial board Wi-Fi setup and agent installation are required.")
    adapter = make_adapter("uno_q", settings)
    st.write("Configured address", settings["uno_q"]["base_url"] or "Not configured")
    a, b, c = st.columns(3)
    if a.button("Connect / refresh status"):
        try:
            st.session_state.board_status = adapter.get_runtime_information()
            st.session_state.reachable_uno_q = True
        except BackendError as exc:
            st.session_state.pop("board_status", None)
            st.session_state.reachable_uno_q = False
            error(exc)
    if b.button("Disconnect dashboard"):
        st.session_state.pop("board_status", None)
        st.session_state.reachable_uno_q = False
    if c.button("Refresh board models"):
        try:
            st.session_state.catalogue_uno_q = adapter.list_models(refresh=True)
            st.success("Board catalogue refreshed; see Model Studio")
        except BackendError as exc:
            error(exc)
    status = st.session_state.get("board_status")
    if status:
        st.caption("Last explicitly requested health response; refresh for current status.")
        st.json(status)
        if st.button("Retrieve agent logs"):
            try:
                st.json(adapter.request("GET", "/v1/logs"))
            except BackendError as exc:
                error(exc)
        tests = status.get("mcu", {}).get("tests", [])
        if tests:
            name = st.selectbox("Firmware test", tests)
            confirmed = st.checkbox("Confirm this read-only MCU RPC test")
            if st.button("Run selected hardware test", disabled=not confirmed):
                try:
                    result = adapter.request("POST", "/v1/hardware-tests", json={"name": name, "confirmed": confirmed})
                    s.store.save("hardware_test", result)
                    st.json(result)
                except BackendError as exc:
                    error(exc)
        else:
            st.info("No verified MCU commands. Install the optional matching read-only probe sketch and configure Bridge before enabling tests.")
    else:
        st.info("Board not connected in this session. Configure its hostname/IP and UNO_Q_TOKEN, then connect.")
    with st.expander("Register installed board model file"):
        st.caption("The board agent restricts paths to its configured LOCAL_MODEL_DIRS. Registration checks file existence, not runtime compatibility or successful inference.")
        board_path = st.text_input("Absolute GGUF path on board Linux")
        if st.button("Register board file", disabled=not status or not board_path):
            try:
                st.json(adapter.request("POST", "/v1/models/register", json={"path": board_path}))
            except BackendError as exc:
                error(exc)
    st.caption("Model inference and model selection are in Model Studio. Automatic mDNS scanning, Wi-Fi changes, firmware upload and system-service management are not implemented.")


def settings_page(s):
    settings = s.settings()
    with st.form("settings"):
        endpoints, selected, params = {}, {}, {}
        for backend, label in LABELS.items():
            st.markdown(f"**{label}**")
            endpoints[backend] = st.text_input("Base URL", settings[backend]["base_url"], key=f"endpoint_{backend}")
            if backend == 'laptop_local':
                ollama_fast = st.checkbox('Ollama fast response: disable thinking and request JSON',
                                          value=settings[backend].get('ollama_fast_mode', urlsplit(settings[backend]['base_url']).port == 11434),
                                          help='For Ollama only. Requests reasoning_effort=none and JSON output. Turn off for other local servers. Runtime options are included in reports.')
            selected[backend] = st.text_input("Selected model ID (revalidated on execution)", settings[backend].get("model_id", ""), key=f"configured_{backend}")
            a, b = st.columns(2)
            current = settings[backend].get("generation", {})
            a.caption('Temperature 0 · seed 42 · fixed for every backend; runtime support unverified until tested')
            params[backend] = {'temperature': 0.0, 'seed': 42, 'max_tokens': settings.get('max_tokens', 1800)}
            settings[backend]['parameter_size'] = a.text_input('Parameter size (verified metadata only; leave blank if unavailable)', settings[backend].get('parameter_size') or '', key=f'parameters_{backend}') or None
            settings[backend]['quantization'] = b.text_input('Quantization (verified metadata only; leave blank if unavailable)', settings[backend].get('quantization') or '', key=f'quantization_{backend}') or None
        max_tokens = st.number_input('Shared maximum generated tokens', 64, 8192, settings.get('max_tokens', 1800))
        backend_timeout_seconds = st.number_input('Backend request timeout (seconds)', 30, 900, int(settings.get('backend_timeout_seconds', 900)), 30, help='Applies to laptop, UNO Q and cloud requests. Slow local reasoning models may need the full 900 seconds.')
        citation_link_validity_threshold = st.number_input('citation_link_validity threshold for human review', 0.0, 1.0, float(settings.get('citation_link_validity_threshold', 1.0)), 0.05)
        directories = st.text_area("Permitted local model directories (one absolute path per line)", "\n".join(settings.get("model_dirs", [])))
        limit = st.number_input("TF-IDF result limit", 1, 40, settings["retrieval_limit"])
        threshold = st.number_input("Minimum relevance", 0.0, 1.0, float(settings["relevance_threshold"]), 0.01)
        submit = st.form_submit_button("Validate and save settings", type="primary")
    if submit:
        try:
            for backend, value in endpoints.items():
                if value:
                    validate_url(value, backend)
            roots = [str(Path(d.strip()).expanduser().resolve()) for d in directories.splitlines() if d.strip()]
            if any(not Path(d).is_dir() for d in roots):
                raise ValueError("Each permitted model directory must exist")
            updated = {b: {"base_url": endpoints[b], "model_id": selected[b], "generation": {**params[b], 'max_tokens': max_tokens}, 'parameter_size': settings[b].get('parameter_size'), 'quantization': settings[b].get('quantization')} for b in LABELS}
            updated['laptop_local']['ollama_fast_mode'] = ollama_fast
            updated.update(model_dirs=roots, retrieval_limit=limit, relevance_threshold=threshold, max_tokens=max_tokens, backend_timeout_seconds=backend_timeout_seconds, citation_link_validity_threshold=citation_link_validity_threshold)
            s.store.set_setting("preferences", updated)
            for backend in LABELS:
                st.session_state.pop(f"catalogue_{backend}", None)
                st.session_state[f"reachable_{backend}"] = False
            st.success("Settings saved. Refresh catalogues to check availability.")
        except ValueError as exc:
            error(exc)
    st.subheader("Connect Qualcomm cloud")
    st.caption('Paste your Cirrascale API token here to use it in this running app. It is held in server memory until restart and is never saved in reports or preferences.')
    def apply_cloud_token():
        token = st.session_state.get('cloud_token_input', '').strip()
        if token:
            os.environ['QUALCOMM_API_KEY'] = token
            st.session_state['cloud_token_input'] = ''
            st.session_state.pop('catalogue_qualcomm', None)
            st.session_state['reachable_qualcomm'] = False
    st.text_input('Qualcomm API token', type='password', key='cloud_token_input')
    st.button('Use Qualcomm token', on_click=apply_cloud_token)
    if st.button('Check Qualcomm connection and models'):
        try:
            catalogue = make_adapter('qualcomm', s.settings()).list_models(refresh=True)
            st.session_state.catalogue_qualcomm = catalogue
            st.session_state.reachable_qualcomm = True
            st.success(f'Connected. Retrieved {len(catalogue)} models. Select a chat model in Model Studio, then test scenario inference.')
        except BackendError as exc:
            error(exc)
    st.subheader("Credential configuration status")
    for name in ("QUALCOMM_API_KEY", "UNO_Q_TOKEN", "SAFESCENARIO_API_TOKEN"):
        st.write(f"{name}: {'configured' if os.getenv(name) else 'setup required'}")
    st.caption("Set credentials in environment variables or the ignored .env file and restart. Values are never displayed or saved as preferences. QUALCOMM_ALLOWED_HOSTS restricts alternate HTTPS cloud endpoints.")
    with st.expander("Risk matrix editor"):
        matrix_json = st.text_area("Versioned risk matrix JSON", s.matrix().model_dump_json(indent=2), height=300)
        if st.button("Validate and activate matrix"):
            try:
                matrix = RiskMatrix.model_validate_json(matrix_json)
                old = s.matrix()
                if matrix.version == old.version and matrix != old:
                    raise ValueError("Change the matrix version when changing its contents")
                s.activate_matrix(matrix)
                st.success("Matrix activated; previous runs retain their frozen matrix")
            except (ValidationError, ValueError) as exc:
                error(exc)
    st.subheader("History and diagnostics")
    st.caption(f"Raw model response retention: {os.getenv('SAFESCENARIO_RETAIN_RAW', 'false')}. Enable explicitly through the environment when debugging. Default diagnostics retain hashes, sizes and validation errors.")
    confirmed = st.checkbox("Permanently clear saved run, review, risk-assessment, simulation, evaluation and retrieval-benchmark history")
    if st.button("Clear history", disabled=not confirmed):
        active = any(j["status"] == "running" for j in s.manager.jobs.values()) or any(j["status"] == "running" for j in s.benchmarks.jobs.values())
        if active:
            st.error("Wait for active runs and benchmarks to finish before clearing history")
        else:
            s.store.clear_history()
            s.manager.jobs.clear()
            s.benchmarks.jobs.clear()
            st.session_state.pop("active_run", None)
            st.session_state.pop("active_benchmark", None)
            st.session_state.pop("retrieval_benchmark", None)
            st.session_state.pop("pdf_report", None)
            for key in list(st.session_state):
                if key.startswith('experiment_pdf_'):
                    st.session_state.pop(key, None)
            st.success("Saved history deleted. This cannot be undone; datasets and settings retained.")


def main():
    st.set_page_config(page_title="SafeScenario AI", page_icon="🛡️", layout="wide")
    s = services()
    with st.sidebar:
        st.markdown("### SafeScenario AI")
        st.caption("AUTONOMOUS PRODUCT SAFETY")
        st.caption(f'Build {IMPLEMENTATION_VERSION}')
        dark = st.toggle("Dark theme", key="dark")
        page = st.radio("Workspace", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption(f"KB {s.dataset().version} · {len(s.dataset().records)} records")
        backend_options = [ALL_BACKENDS, *LABELS]
        active_backend = st.selectbox("Active backend", backend_options, format_func=lambda value: "All backends" if value == ALL_BACKENDS else LABELS[value], key="active_backend")
        if active_backend == ALL_BACKENDS:
            configured = [f"{LABELS[b]}: {s.settings()[b].get('model_id') or 'not selected'}" for b in LABELS]
            st.caption("Run together: " + " · ".join(configured))
            ready = sum(bool(s.settings()[b].get("model_id")) for b in LABELS)
            st.caption(f"Configured backends: {ready}/{len(LABELS)} · Scenario Builder and Model Comparison will preselect all available models")
            for backend in LABELS:
                st.markdown(f"{LABELS[backend]} {backend_status_badge(backend, s.settings(), s.store)}", unsafe_allow_html=True)
        else:
            selected = s.settings()[active_backend].get("model_id")
            st.caption(f"Model: {selected or 'not selected'}")
            st.markdown(backend_status_badge(active_backend, s.settings(), s.store), unsafe_allow_html=True)
            st.caption("Green means the runtime catalogue was reached. It does not by itself prove successful structured inference.")
    theme(dark)
    with st.expander('Out of scope / limitations'):
        for limitation in LIMITATIONS:
            st.write(limitation)
    subtitle = {"Overview":"A complete view of AI proposals, engineering decisions and measured model performance.",
                "Model Comparison":"Compare measured latency, validation outcomes and human review across your inference backends.",
                "Scenario Register":"Every identifiable AI proposal, its latest decision and its full review audit trail.",
                "Knowledge Base":"Traceable engineering knowledge with inspectable search relevance and measured retrieval speed."}.get(page, "Evidence-linked proposals. Transparent risk calculations. Accountable engineering decisions.")
    st.markdown(f'<div class="workspace-hero"><div class="eyebrow">SafeScenario AI / Engineering workspace</div><h1>{escape(page)}</h1><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True)
    handlers = {"Overview":overview,"Scenario Builder":scenario_builder,"Model Studio":model_studio,"Model Comparison":comparison,
                "Scenario Register":register_page,"Knowledge Base":knowledge,"Risk Analysis":risk_analysis,"Test Simulation":simulation_page,
                "Evaluation and Benchmarks":evaluation_page,"Reports and History":history,"Device Control":device,"Settings":settings_page}
    handlers[page](s)
