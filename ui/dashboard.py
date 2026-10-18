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
from backend.analytics import scenario_register, performance_rows, proposal_entries, DECISIONS
from evaluation.benchmark import BenchmarkDataset, draft_dataset
from evaluation.labelled import LabelledDataset, baseline_result
from backend.constrained import FAULT_LABELS, LIMITATIONS
from backend.scenario_generation import recalculate
from knowledge_base.core import Dataset, ConstrainedDataset, parse_upload, retrieve, retrieve_with_diagnostics
from knowledge_base.benchmark import benchmark_retrieval
from ui.charts import review_figure, latency_figure, validation_figure, outcome_figure, unified_benchmark_figure
from models.adapters import BackendError, discover_files, make_adapter
from reporting.exports import comparison_rows, csv_bytes, json_bytes, report
from reporting.exports import experiment_report, experiment_csv
from reporting.pdf_export import pdf_bytes
from reporting.pdf_export import experiment_pdf_bytes
from risk_engine.calculator import Classification, RiskMatrix, calculate
from simulation.engine import SimulationInput, simulate

LABELS = {"laptop_local": "Laptop local", "qualcomm": "Qualcomm cloud", "uno_q": "Arduino UNO Q"}
ALL_BACKENDS = "all_backends"
PAGES = ["Overview", "Scenario Builder", "Model Studio", "Model Comparison", "Scenario Register", "Knowledge Base", "Risk Analysis", "Evaluation and Benchmarks", "Reports and History", "Device Control", "Settings"]



@st.cache_resource
def services():
    """Shared services v5: complete multi-fault evidence and preflight validation."""
    return Services()


def theme(dark):
    bg, surface, text, muted, border = ("#0f172a", "#1e293b", "#f1f5f9", "#94a3b8", "#334155") if dark else ("#f6f8fc", "#ffffff", "#0f172a", "#64748b", "#e2e8f0")
    sidebar_surface, sidebar_text, sidebar_hover, sidebar_selected = (("#0b1120", "#cbd5e1", "#1e293b", "#334155") if dark
                                                                  else ("#ffffff", "#0f172a", "#f1f5f9", "#e2e8f0"))
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp {{ font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif; }}
    .stApp, [data-testid="stAppViewContainer"] {{ background:{bg}; color:{text}; }}
    [data-testid="stHeader"], [data-testid="stSidebar"] {{ background:{sidebar_surface}; color:{sidebar_text}; }}
    [data-testid="stSidebar"] {{ border-right:1px solid {border}; }}
    .block-container {{ padding-top:1.5rem; max-width:1500px; }}
    [data-testid="stSidebar"] {{background:{sidebar_surface}!important;}}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{color:{sidebar_text}!important;}}
    [data-testid="stSidebar"] [role="radiogroup"] label {{padding:6px 12px;border-radius:12px;font-size:13px;font-weight:500;transition:all .15s;}}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{background:{sidebar_hover};}}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{background:{sidebar_selected};font-weight:600;}}
    .backend-status {{display:block;font-size:11px;margin:5px 0 2px;line-height:1.4;}}
    .status-dot {{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;}}
    .status-good {{color:#16a34a!important;font-weight:600;}} .status-good .status-dot {{background:#22c55e;box-shadow:0 0 0 3px rgba(34,197,94,.16);}}
    .status-warn {{color:#d97706!important;font-weight:600;}} .status-warn .status-dot {{background:#f59e0b;}}
    .status-neutral {{color:{muted}!important;}} .status-neutral .status-dot {{background:#94a3b8;}}
    .workspace-hero {{border:1px solid {border};background:{surface};border-radius:20px;padding:18px 24px;margin-bottom:20px;box-shadow:0 2px 12px rgba(15,23,42,0.03);}}
    .workspace-hero h1 {{margin:4px 0 6px!important;font-size:1.45rem!important;font-weight:700!important;letter-spacing:-0.02em;}}
    .workspace-hero p {{color:{muted};margin:0;font-size:13px;max-width:850px;line-height:1.5;}}
    .eyebrow {{font-size:10px;letter-spacing:1.5px;font-weight:700;color:#6366f1;text-transform:uppercase;}}
    .workflow-band {{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 20px;}}
    .workflow-chip {{border:1px solid {border};border-radius:9999px;background:{surface};padding:6px 14px;font-size:11.5px;color:{muted};font-weight:500;}}
    .empty-panel {{background:{surface};border:1px dashed {border};border-radius:16px;padding:28px 22px;margin:10px 0;min-height:180px;}}
    .empty-panel strong {{font-size:15px;color:{text};display:block;margin-bottom:8px;}}
    .empty-panel p {{color:{muted};font-size:13px;line-height:1.6;}}
    [data-testid="stMetric"] {{background:{surface}!important;border:1px solid {border}!important;border-radius:16px!important;padding:12px 16px!important;box-shadow:0 2px 8px rgba(15,23,42,0.025);}}
    [data-testid="stMetricValue"] {{font-weight:700!important;font-size:1.35rem!important;color:{text}!important;letter-spacing:-0.02em;}}
    [data-testid="stMetricLabel"] {{height:auto;min-height:1.3rem;font-size:0.75rem!important;font-weight:600!important;text-transform:uppercase;letter-spacing:0.04em;color:{muted}!important;}}
    [data-testid="stMetricLabel"] * {{white-space:normal!important;text-overflow:clip!important;overflow:visible!important;}}
    h1 {{ font-size:1.45rem!important; letter-spacing:-0.02em; font-weight:700!important; }}
    h2 {{ font-size:1.2rem!important; font-weight:650!important; letter-spacing:-0.01em; }}
    h3 {{ font-size:1.05rem!important; font-weight:600!important; }}
    h4 {{ font-size:0.95rem!important; font-weight:600!important; }}
    h1,h2,h3,h4,p,label, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {{ color:{text}; }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{color:{muted}!important;opacity:1!important;font-size:12px;}}
    [data-testid="stExpander"], [data-testid="stForm"] {{background:{surface}; border:1px solid {border}; border-radius:16px; padding:14px;}}
    [data-testid="stTabs"] [role="tablist"] {{gap:10px;border-bottom:1px solid {border};padding-bottom:6px;}}
    [data-testid="stTabs"] [role="tab"] {{border-radius:9999px;padding:6px 14px;font-size:13px;font-weight:500;}}
    .eye-mode-btn {{display:block;width:100%;text-align:center;padding:14px;border-radius:18px;background:{surface};border:1.5px solid {border};cursor:pointer;transition:all .2s ease;}}
    .eye-mode-btn:hover {{border-color:#6366f1;box-shadow:0 4px 16px rgba(99,102,241,0.08);}}
    .eye-mode-active {{border:2px solid #6366f1!important;background:{surface}!important;box-shadow:0 4px 18px rgba(99,102,241,0.12)!important;}}
    .pill-badge {{display:inline-flex;align-items:center;border-radius:9999px;padding:3px 10px;font-size:11px;font-weight:600;}}
    .pill-cloud {{background:#eef2ff;color:#4338ca;border:1px solid #c7d2fe;}}
    .pill-uno {{background:#fff7ed;color:#c2410c;border:1px solid #fed7aa;}}
    .pill-local {{background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;}}
    .pill-risk-high {{background:#fef2f2;color:#b91c1c;border:1px solid #fecaca;}}
    .pill-risk-med {{background:#fffbeb;color:#b45309;border:1px solid #fde68a;}}
    .pill-risk-low {{background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;}}
    .pill-stmt-hazard {{background:#fef2f2;color:#b91c1c;border:1px solid #fecaca;padding:2px 8px;border-radius:6px;font-size:10.5px;font-weight:700;letter-spacing:0.4px;text-transform:uppercase;}}
    .pill-stmt-mit {{background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;padding:2px 8px;border-radius:6px;font-size:10.5px;font-weight:700;letter-spacing:0.4px;text-transform:uppercase;}}
    .pill-stmt-test {{background:#f5f3ff;color:#6d28d9;border:1px solid #ddd6fe;padding:2px 8px;border-radius:6px;font-size:10.5px;font-weight:700;letter-spacing:0.4px;text-transform:uppercase;}}
    .light-card-cloud {{border:1.5px solid #c7d2fe!important;background:rgba(99,102,241,0.03)!important;border-radius:16px;padding:14px;}}
    .light-card-uno {{border:1.5px solid #fed7aa!important;background:rgba(245,158,11,0.03)!important;border-radius:16px;padding:14px;}}
    .light-card-local {{border:1.5px solid #bbf7d0!important;background:rgba(16,185,129,0.03)!important;border-radius:16px;padding:14px;}}
    .risk-meter-bg {{width:100%;height:8px;background:#e2e8f0;border-radius:9999px;overflow:hidden;margin:6px 0;}}
    .risk-meter-fill-high {{height:100%;background:linear-gradient(90deg, #f87171, #ef4444);border-radius:9999px;}}
    .risk-meter-fill-med {{height:100%;background:linear-gradient(90deg, #fbbf24, #f59e0b);border-radius:9999px;}}
    .risk-meter-fill-low {{height:100%;background:linear-gradient(90deg, #4ade80, #10b981);border-radius:9999px;}}
    [data-baseweb="select"]>div, [data-baseweb="input"], [data-baseweb="textarea"], textarea, input {{background:{surface}!important;color:{text}!important;border-radius:12px!important;}}
    button[kind="secondary"], button[kind="primary"] {{border-radius:12px!important;font-size:13px!important;font-weight:600!important;}}
    @media (max-width:640px) {{
      .workspace-hero {{padding:14px;margin-bottom:14px;}}
      .workspace-hero h1 {{font-size:1.25rem!important;}}
      [data-testid="stMetricValue"] {{font-size:1.2rem!important;}}
    }}
    </style>""", unsafe_allow_html=True)


def render_odometer_gauge(score, level, width=280, height=160):
    """Render a sleek circular speedometer/odometer SVG risk gauge."""
    import math
    if score is None:
        score_val = 0
        display_score = "—"
        level_str = "NOT ASSIGNED"
        color = "#94a3b8"
        bg_pill = "#f1f5f9"
        border_pill = "#cbd5e1"
    else:
        try:
            score_val = max(0, min(100, int(round(float(score)))))
        except (ValueError, TypeError):
            score_val = 0
        display_score = str(score_val)
        level_str = str(level).upper() if level else "CALCULATING"
        if score_val >= 66 or "HIGH" in level_str:
            color = "#ef4444"
            bg_pill = "#fef2f2"
            border_pill = "#fecaca"
        elif score_val >= 36 or "MED" in level_str:
            color = "#f59e0b"
            bg_pill = "#fffbeb"
            border_pill = "#fde68a"
        else:
            color = "#10b981"
            bg_pill = "#f0fdf4"
            border_pill = "#bbf7d0"

    cx, cy, r = 140, 115, 82
    angle = math.pi * (1.0 - score_val / 100.0)

    # Needle tip coordinates:
    nx = cx - 64 * math.cos(math.pi - angle)
    ny = cy - 64 * math.sin(math.pi - angle)

    # Needle base:
    bx1 = cx - 6 * math.sin(math.pi - angle)
    by1 = cy + 6 * math.cos(math.pi - angle)
    bx2 = cx + 6 * math.sin(math.pi - angle)
    by2 = cy - 6 * math.cos(math.pi - angle)

    ax = cx - r * math.cos(math.pi - angle)
    ay = cy - r * math.sin(math.pi - angle)
    large_arc = 1 if score_val > 50 else 0

    val_path = f"M 58 115 A {r} {r} 0 {large_arc} 1 {ax:.1f} {ay:.1f}" if score_val > 0 else ""

    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:6px 0;">
      <svg width="{width}" height="{height}" viewBox="0 0 280 160" style="overflow:visible;max-width:100%;">
        <defs>
          <linearGradient id="meterTrackGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#10b981" />
            <stop offset="50%" stop-color="#f59e0b" />
            <stop offset="100%" stop-color="#ef4444" />
          </linearGradient>
          <filter id="needleGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.25"/>
          </filter>
        </defs>
        <!-- Background Track -->
        <path d="M 58 115 A 82 82 0 0 1 222 115" fill="none" stroke="#e2e8f0" stroke-width="14" stroke-linecap="round"/>
        <!-- Value Arc -->
        {f'<path d="{val_path}" fill="none" stroke="url(#meterTrackGrad)" stroke-width="14" stroke-linecap="round"/>' if val_path else ''}
        <!-- Ticks -->
        <text x="44" y="132" font-size="10" font-weight="600" fill="#94a3b8" text-anchor="middle">0</text>
        <text x="140" y="22" font-size="10" font-weight="600" fill="#94a3b8" text-anchor="middle">50</text>
        <text x="236" y="132" font-size="10" font-weight="600" fill="#94a3b8" text-anchor="middle">100</text>
        <!-- Needle -->
        <polygon points="{bx1:.1f},{by1:.1f} {nx:.1f},{ny:.1f} {bx2:.1f},{by2:.1f}" fill="{color}" filter="url(#needleGlow)"/>
        <circle cx="140" cy="115" r="8" fill="#1e293b"/>
        <circle cx="140" cy="115" r="4" fill="#ffffff"/>
        <!-- Center Odometer Score Display -->
        <text x="140" y="92" font-size="28" font-weight="800" fill="{color}" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif">{display_score}</text>
        <text x="140" y="106" font-size="9" font-weight="700" fill="#64748b" text-anchor="middle" letter-spacing="1.2">RISK SCORE</text>
      </svg>
      <div style="margin-top:-8px;">
        <span class="pill-badge" style="background:{bg_pill};color:{color};border:1.5px solid {border_pill};font-size:11px;font-weight:700;padding:3px 12px;text-transform:uppercase;letter-spacing:0.5px;">
          {level_str} RISK
        </span>
      </div>
    </div>
    """


def backend_status_badge(backend, settings, store):
    """Show runtime reachability without falsely claiming that inference is verified."""
    model_id = settings[backend].get("model_id")
    if not model_id:
        return '<span class="backend-status status-neutral"><span class="status-dot"></span>No model selected</span>'
    if st.session_state.get(f"reachable_{backend}"):
        runs = store.list("run", 20)
        latest_res = None
        for saved in runs:
            res = saved.get("payload", {}).get("results", {}).get(backend)
            if res:
                latest_res = res
                break
        if latest_res:
            st_val = latest_res.get("status")
            if st_val == "completed":
                label = "Runtime reachable · inference completed"
                cls = "status-good"
            elif st_val == "failed":
                label = "Runtime reachable · last run failed"
                cls = "status-warn"
            elif st_val == "cancelled":
                label = "Runtime reachable · cancelled"
                cls = "status-neutral"
            else:
                label = "Runtime reachable · inference unverified"
                cls = "status-good"
        else:
            label = "Runtime reachable · inference unverified"
            cls = "status-good"
        return f'<span class="backend-status {cls}"><span class="status-dot"></span>{label}</span>'
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
    latest = runs[0]['payload'] if runs else None
    rows = scenario_register(s.store, latest['id']) if latest else []
    calculated = recalculate(latest, s.store.reviews(latest['id'])) if latest else {}
    risk_values = [value for value in calculated.values() if value.get('risk_score') is not None]
    highest = max(risk_values, key=lambda value: value['risk_score']) if risk_values else None
    cols = st.columns(5)
    system_accepted = sum(value['system_accepted'] for value in calculated.values())
    system_rejected = sum(value['system_rejected'] for value in calculated.values())
    included = sum(len(value['included']) for value in calculated.values())
    for col, label, value in zip(cols,
        ["Latest run proposals", "Automatic KB accepted", "Automatic KB rejected", "Included in risk", "Current risk"],
        [len(rows), system_accepted, system_rejected, included,
         (highest['risk_level'].upper() + f" · {highest['risk_score']}") if highest else "NOT AVAILABLE"]):
        col.metric(label, value)
    st.caption("These figures describe the newest saved run, not mixed totals from older implementations. Risk is calculated before engineer review from the included, automatically screened scenarios.")
    st.markdown('<div class="workflow-band"><span class="workflow-chip">01 · Define operating conditions</span><span class="workflow-chip">02 · Retrieve TF-IDF evidence</span><span class="workflow-chip">03 · AI proposes + KB screens</span><span class="workflow-chip">04 · Rule-based risk</span><span class="workflow-chip">05 · Engineer removes + recalculates</span></div>', unsafe_allow_html=True)
    current, backends, history = st.tabs(["Current run", "Backend status", "History and performance"])
    with current:
        if latest:
            st.caption(f"Run {latest['id']} · build {latest.get('snapshot', {}).get('implementation_version') or 'historical'} · {latest.get('finished') or latest.get('snapshot', {}).get('created')}")
            risk_selection_panel(latest, s.store.reviews(latest['id']))
        else:
            st.markdown('<div class="empty-panel"><strong>No run yet</strong><p>Build an operating scenario, select a model and run it. TF-IDF evidence, automatic screening and risk will then appear here.</p></div>', unsafe_allow_html=True)
    with backends:
        status_rows = []
        for backend, label in LABELS.items():
            latest_result = next((run['payload']['results'][backend] for run in runs if backend in run['payload'].get('results', {})), None)
            successful = next((run['payload']['results'][backend] for run in runs
                               if backend in run['payload'].get('results', {}) and run['payload']['results'][backend].get('status') == 'completed'), None)
            status_rows.append({
                'backend': label,
                'token / endpoint': 'catalogue connected' if st.session_state.get(f'reachable_{backend}', False) else 'not checked in this session',
                'inference': 'verified by saved run' if successful else 'not verified',
                'latest model': latest_result.get('model_id') if latest_result else 'not run',
                'latest result': latest_result.get('status') if latest_result else 'not run',
                'latest error': latest_result.get('error', '') if latest_result else '',
            })
        st.dataframe(status_rows, use_container_width=True, hide_index=True)
        st.caption("A configured token or a successful model catalogue request does not prove inference. Only a completed saved run is marked inference verified.")
    with history:
        a, b = st.columns(2)
        with a:
            st.subheader("All saved engineer decisions")
            historical_rows = scenario_register(s.store)
            if historical_rows:
                st.plotly_chart(review_figure(historical_rows, st.session_state.dark), use_container_width=True)
        with b:
            st.subheader("Knowledge coverage")
            categories = Counter(r.type.replace("_", " ") for r in s.dataset().records)
            figure = px.bar(x=list(categories.values()), y=list(categories), orientation="h", labels={"x":"Records", "y":""}, color_discrete_sequence=["#6366f1"])
            from ui.charts import polish
            st.plotly_chart(polish(figure, st.session_state.dark), use_container_width=True)
        performance_panel(s)
        if runs:
            st.dataframe([{"created": r["created"], "run_id": r["id"], "status": r["payload"]["status"], "backends": ", ".join(r["payload"]["results"])} for r in runs[:15]], use_container_width=True)


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
    st.markdown('**Current workflow: AI proposals → automatic KB screening → rule-based risk → engineer review**')
    limit = st.number_input('Maximum distinct AI scenarios per model', 1, 20, 10, key=f'{prefix}_scenario_limit')
    st.caption('AI drafts scenarios from retrieved records. Exact citations and links are screened automatically; risk is calculated from accepted scenarios before the engineer removes any unwanted scenarios.')
    return {'generation_mode': 'scenario_drafts', 'max_scenarios': limit}


def risk_selection_panel(run, reviews, backend=None):
    results = recalculate(run, reviews)
    if backend:
        results = {backend: results[backend]} if backend in results else {}
    if not results:
        return
    st.subheader('Risk calculated from accepted scenarios')
    st.caption('The rule-based system calculates risk immediately from severity and likelihood stored in the knowledge base. AI does not assign risk. Engineer review happens afterward: rejecting or excluding a scenario removes it and recalculates the result.')
    for key, value in results.items():
        score_val = value.get('risk_score')
        score_100 = min(100, round((float(score_val) / 9.0) * 100)) if score_val is not None else None
        score_display = f"{score_100} / 100" if score_100 is not None else 'N/A'
        risk_lvl = (value['risk_level'] or 'not available').upper()
        fill_cls = "risk-meter-fill-high" if score_100 and score_100 >= 66 else ("risk-meter-fill-med" if score_100 and score_100 >= 36 else "risk-meter-fill-low")
        meter_pct = score_100 if score_100 is not None else 0

        with st.container(border=True):
            st.markdown(f"**{LABELS.get(key, key)}**")
            col_info, col_gauge = st.columns([3, 2])
            with col_info:
                a, b = st.columns(2)
                a.metric('Included scenarios', len(value['included']))
                b.metric('Engineer excluded', len(value['engineer_excluded']))
                c, d = st.columns(2)
                c.metric('Overall risk', risk_lvl)
                d.metric('Risk Score (/100)', score_display)
            with col_gauge:
                st.markdown(render_odometer_gauge(score_100, risk_lvl, width=220, height=125), unsafe_allow_html=True)

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
                suffix = (' No risk is shown because no accepted scenario covers the selected fault.'
                          if not value['included'] else ' The displayed risk covers the other included scenarios only.')
                st.warning(value['coverage_note'] + '.' + suffix)
            st.caption('Combined faults use the highest included hazard score. Undocumented interactions do not receive an invented risk multiplier.')
    st.dataframe([{'backend': LABELS.get(key, key), 'proposed': value['proposed'],
                   'system accepted': value['system_accepted'], 'system rejected': value['system_rejected'],
                   'engineer excluded': len(value['engineer_excluded']), 'included scenarios': len(value['included']),
                   'unique hazards': len(value['hazard_ids']), 'risk': value['risk_level'] or 'unassigned',
                   'citation validity': value['citation_link_validity'], 'reviewed support': value['reviewed_support_rate'],
                   'score (/100)': f"{min(100, round((float(value['risk_score'])/9.0)*100))}/100" if value['risk_score'] is not None else 'N/A',
                   'status': value['status'].replace('_', ' ')} for key, value in results.items()],
                 use_container_width=True, hide_index=True)
    with st.expander('Technical details: included IDs, exclusions and KB risk records'):
        st.json(results)


def register_page(s):
    saved_runs = s.store.list("run", limit=1)
    if not saved_runs:
        st.info("No saved runs found yet. Generate scenarios in Scenario Builder first.")
        return

    latest_saved = saved_runs[0]
    run = latest_saved["payload"]
    run_id = latest_saved["id"]
    snapshot = run.get("snapshot", {})
    req = snapshot.get("request", {})
    inputs = req.get("inputs", {})
    results = run.get("results", {})

    st.markdown("### Latest Run Engineer Review")
    st.caption("Review proposed safety scenarios from the last run for each model mode. Exclude or re-include any scenario to dynamically recalculate the overall risk.")

    # High-level run metadata
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Run ID", run_id[:12])
    m2.metric("Operating Context", f"{inputs.get('time_of_day', 'day')}, {inputs.get('weather', 'dry')}")
    m3.metric("Selected Faults", ", ".join(inputs.get("faults", [])) or "None")
    m4.metric("Generation Mode", req.get("generation_mode", "scenario_drafts").replace("_", " ").title())

    # The three modes
    modes = [
        ("qualcomm", "Qualcomm Cloud"),
        ("laptop_local", "Laptop Local"),
        ("uno_q", "Arduino UNO Q"),
    ]

    # Current reviews and recalculated risks
    reviews = s.store.reviews(run_id)
    calculated_risk = recalculate(run, reviews)

    # 1. Eye-Catching Summary Cards Across All 3 Modes
    st.write("")
    c1, c2, c3 = st.columns(3)
    card_cols = {"qualcomm": c1, "laptop_local": c2, "uno_q": c3}

    for backend, label in modes:
        col = card_cols[backend]
        res = results.get(backend)
        b_scenarios = res.get("scenarios", []) if res else []
        b_eval = calculated_risk.get(backend, {})
        b_risk = (b_eval.get("risk_level") or ("NO DATA" if not res else "EXCLUDED")).upper()
        b_score = b_eval.get("risk_score")
        b_score_100 = min(100, round((float(b_score) / 9.0) * 100)) if b_score is not None else None

        with col:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                if res:
                    st.caption(f"{res.get('status', 'unknown').title()} · {len(b_scenarios)} proposed")
                    risk_badge_cls = "pill-risk-high" if b_score_100 and b_score_100 >= 66 else ("pill-risk-med" if b_score_100 and b_score_100 >= 36 else "pill-risk-low")
                    score_txt = f"{b_score_100}/100" if b_score_100 is not None else "N/A"
                    st.markdown(f'<span class="pill-badge {risk_badge_cls}">Risk: {b_risk} ({score_txt})</span>', unsafe_allow_html=True)
                else:
                    st.caption("Not executed in this run")
                    st.markdown('<span class="pill-badge pill-slate">Not Run</span>', unsafe_allow_html=True)

    st.write("")

    # 2. Eye-Catching Styled Tabs for Each Mode
    mode_tabs = st.tabs([label for _, label in modes])
    all_export_scenarios = []

    for tab, (backend, label) in zip(mode_tabs, modes):
        with tab:
            result = results.get(backend)
            if not result:
                st.info(f"**{label}** was not executed in this run. Run inference in Scenario Builder to inspect scenarios.")
                continue

            status = result.get("status", "unknown")
            model_name = result.get("model_id") or result.get("model_details", {}).get("name") or "unrecorded"
            scenarios = result.get("scenarios", [])

            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Model", model_name)
            h2.metric("Execution", status.title())
            latency = result.get("generation_latency_ms")
            h3.metric("Generation Time", f"{latency / 1000:.1f} s" if latency is not None else "N/A")
            tokens = result.get("runtime_metrics", {}).get("total_tokens")
            h4.metric("Tokens Used", f"{tokens:,}" if tokens else "N/A")

            if status == "failed":
                st.error(f"Inference failed: {result.get('error', 'Unknown error')}")
                continue

            if not scenarios:
                st.warning(f"No scenarios were proposed by {label} in this run.")
                continue

            # Mode risk evaluation
            mode_eval = calculated_risk.get(backend, {})
            included_scopes = set(mode_eval.get("included", []))
            excluded_scopes = set(mode_eval.get("engineer_excluded", []))

            risk_level = (mode_eval.get("risk_level") or "NO INCLUDED SCENARIOS").upper()
            risk_score = mode_eval.get("risk_score")
            risk_score_100 = min(100, round((float(risk_score) / 9.0) * 100)) if risk_score is not None else None
            fill_cls = "risk-meter-fill-high" if risk_score_100 and risk_score_100 >= 66 else ("risk-meter-fill-med" if risk_score_100 and risk_score_100 >= 36 else "risk-meter-fill-low")
            meter_pct = risk_score_100 if risk_score_100 is not None else 0

            # Risk Summary Card with Circular Odometer Gauge
            with st.container(border=True):
                g_c1, g_c2 = st.columns([1.1, 2.9])
                with g_c1:
                    st.markdown(render_odometer_gauge(risk_score_100, risk_level), unsafe_allow_html=True)
                with g_c2:
                    st.markdown(f"**{label} — Recalculated Safety Risk Evaluation**")
                    rk1, rk2, rk3 = st.columns(3)
                    rk1.metric("Overall Risk", risk_level)
                    rk2.metric("Included Scenarios", len(included_scopes))
                    rk3.metric("Excluded Scenarios", len(excluded_scopes))
                    for step in mode_eval.get("calculation_steps", []):
                        st.caption(f"• {step}")

            st.markdown(f"#### Proposed Scenarios ({len(scenarios)})")

            for i, sc in enumerate(scenarios, 1):
                scenario_id = sc.get("scenario_id", f"DRAFT-{i}")
                scope = f"{backend}:{scenario_id}"
                is_excluded = scope in excluded_scopes
                for rev in reviews:
                    if rev.get("scenario_id") == scope and rev.get("payload", {}).get("decision") == "rejected":
                        is_excluded = True
                        break

                hazards = sc.get("hazards", [])
                hazard_id = hazards[0].get("hazard_id") if hazards else None
                risk_assess = sc.get("risk_assessments", {})
                sc_risk = risk_assess.get(hazard_id, {}) if hazard_id else {}
                sc_level = (sc_risk.get("risk_level") or "unknown").upper()
                raw_sc_score = sc_risk.get("risk_score")
                sc_score_100 = min(100, round((float(raw_sc_score) / 9.0) * 100)) if isinstance(raw_sc_score, (int, float)) else "—"
                sc_sev = sc_risk.get("severity", "—")
                sc_lik = sc_risk.get("likelihood", "—")

                all_export_scenarios.append({
                    "backend": backend,
                    "mode": label,
                    "scenario_id": scenario_id,
                    "title": sc.get("title", "Untitled"),
                    "status": "Excluded" if is_excluded else "Included",
                    "hazard_id": hazard_id or "",
                    "severity": sc_sev,
                    "likelihood": sc_lik,
                    "risk_score_100": sc_score_100,
                    "risk_level": sc_level,
                    "narrative": sc.get("scenario_narrative", ""),
                })

                with st.container(border=True):
                    col_title, col_action = st.columns([3.2, 1.2])
                    with col_title:
                        title_text = sc.get("title", "Untitled scenario")
                        status_badge = '<span class="pill-badge pill-risk-high">EXCLUDED</span>' if is_excluded else '<span class="pill-badge pill-local">INCLUDED IN RISK</span>'
                        st.markdown(f"{status_badge} &nbsp; **{scenario_id}: {title_text}**", unsafe_allow_html=True)
                        st.caption(f"Risk: **{sc_level}** (Score: {sc_score_100}/100 · Severity: {sc_sev} · Likelihood: {sc_lik})")

                    with col_action:
                        if is_excluded:
                            if st.button("Re-include in Risk", key=f"reinc_{backend}_{scenario_id}", use_container_width=True):
                                s.store.delete_reviews_for_scenario(run_id, scope)
                                st.rerun()
                        else:
                            if st.button("Remove from Risk", key=f"rem_{backend}_{scenario_id}", use_container_width=True):
                                s.store.review(run_id, scope, "rejected", "Engineer Review", "Excluded from risk evaluation")
                                st.rerun()

                    stmts = sc.get("statements", [])
                    if stmts:
                        for stmt in stmts:
                            rec_id = stmt.get("record_id", "")
                            text = stmt.get("text", "")
                            if rec_id.startswith("H-") or rec_id.startswith("HAZ-"):
                                st.markdown(f'<span class="pill-stmt-hazard">HAZARD</span> &nbsp; **{rec_id}:** {text}', unsafe_allow_html=True)
                            elif rec_id.startswith("M-") or rec_id.startswith("MIT-"):
                                st.markdown(f'<span class="pill-stmt-mit">MITIGATION</span> &nbsp; **{rec_id}:** {text}', unsafe_allow_html=True)
                            elif rec_id.startswith("T-") or rec_id.startswith("TEST-"):
                                st.markdown(f'<span class="pill-stmt-test">VERIFICATION TEST</span> &nbsp; **{rec_id}:** {text}', unsafe_allow_html=True)
                            else:
                                st.markdown(f"• **{rec_id}:** {text}")
                    elif sc.get("scenario_narrative"):
                        st.write(sc.get("scenario_narrative"))

                    if sc_risk.get("explanation"):
                        with st.expander("Knowledge Base Risk Details", expanded=False):
                            st.write(sc_risk.get("explanation"))

    # Result Export Option at the bottom
    st.divider()
    st.subheader("Export Results")
    st.caption("Download the last run review decisions and recalculated risk evaluation.")

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        export_payload = {
            "run_id": run_id,
            "created": latest_saved.get("created"),
            "inputs": inputs,
            "recalculated_risk": calculated_risk,
            "scenarios": all_export_scenarios,
        }
        st.download_button(
            "Export Full Review & Risk (JSON)",
            json_bytes(export_payload),
            f"review-results-{run_id[:8]}.json",
            "application/json",
            use_container_width=True
        )

    with exp_col2:
        if all_export_scenarios:
            st.download_button(
                "Export Scenarios & Decisions (CSV)",
                csv_bytes(all_export_scenarios),
                f"scenarios-review-{run_id[:8]}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.button("Export Scenarios & Decisions (CSV)", disabled=True, use_container_width=True)




def scenario_builder(s):
    settings, dataset = s.settings(), s.dataset()
    saved = s.store.setting("scenario_input", ScenarioInput().model_dump())

    st.caption("Configure operating conditions and generate safety scenarios across Cloud, Uno Q, or Local. Selecting any option displays its controls, execution, proposed scenarios, and recalculating risk evaluation under there.")

    # Eye-catching banner for Operating Context & Fault Selection
    inputs_curr = ScenarioInput.model_validate(saved)
    active_fault_names = ", ".join(inputs_curr.faults) if inputs_curr.faults else "None"
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(245,158,11,0.05) 50%, rgba(16,185,129,0.05) 100%);
                border:1.5px solid #c7d2fe;border-radius:18px;padding:16px 20px;margin-bottom:12px;
                box-shadow:0 4px 16px rgba(99,102,241,0.04);">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
        <div>
          <span class="eyebrow" style="color:#4f46e5;">Target Operational Design Domain</span>
          <h3 style="margin:2px 0 0 0;font-size:1.15rem;font-weight:700;">Operating Context & Fault Selection</h3>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
          <span class="pill-badge pill-cloud">{inputs_curr.time_of_day.upper()} · {inputs_curr.weather.upper()} · {inputs_curr.visibility.upper()}</span>
          <span class="pill-badge pill-uno">{len(inputs_curr.faults)} FAULT{'S' if len(inputs_curr.faults) != 1 else ''}: {active_fault_names}</span>
          <span class="pill-badge pill-local">{inputs_curr.speed_mps} m/s</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Customize Operating Context & Fault Types (1–3 Faults)", expanded=False):
        with st.form("scenario_form"):
            a, b, c = st.columns(3)
            time_of_day = a.selectbox("Time of day", ["day", "night"], index=["day", "night"].index(saved.get("time_of_day", "day")))
            weather = b.selectbox("Weather", ["dry", "wet", "rain"], index=["dry", "wet", "rain"].index(saved.get("weather", "dry")))
            visibility = c.selectbox("Visibility", ["clear", "reduced"], index=["clear", "reduced"].index(saved.get("visibility", "clear")))
            surface = a.text_input("Surface", saved.get("surface", "paved sidewalk"))
            density = b.selectbox("Pedestrian density", ["low", "medium", "high"], index=["low", "medium", "high"].index(saved.get("pedestrian_density", "medium")))
            speed = c.number_input("Speed (m/s)", 0.0, 5.0, float(saved.get("speed_mps", 1.0)), 0.1)
            location = a.text_input("Operating context", saved.get("location", "sidewalk"))
            navigation = b.text_input("Navigation mode", saved.get("navigation_mode", "autonomous"))
            sensors = c.text_input("Sensors (comma separated)", ", ".join(saved.get("sensors", [])))

            # Fault mapping and safe default calculation:
            code_to_label = {v: k for k, v in FAULT_LABELS.items()}
            faults_available = list(FAULT_LABELS.keys())
            default_faults = []
            for f in saved.get("faults", []):
                if f in faults_available:
                    default_faults.append(f)
                elif f in code_to_label:
                    default_faults.append(code_to_label[f])
                else:
                    norm = f.lower().replace("_", " ")
                    match = next((k for k in faults_available if k.lower() == norm), None)
                    if match:
                        default_faults.append(match)
            default_faults = list(dict.fromkeys(default_faults))[:3]
            if not default_faults and faults_available:
                default_faults = [faults_available[0]]

            faults = st.multiselect(
                "Fault types (select 1 to 3 supported faults)",
                faults_available,
                default=default_faults,
                max_selections=3,
                help="Choose 1 to 3 robot faults. Matched hazard, mitigation, and acceptance test evidence will be retrieved."
            )
            capabilities = st.text_area("Documented robot capabilities", saved.get("documented_capabilities", ""))
            notes = st.text_area("Additional context / notes", saved.get("notes", ""))
            save = st.form_submit_button("Validate and save input", type="primary", use_container_width=True)

        if save:
            try:
                selected_faults = faults if faults else [faults_available[0]]
                new_inputs = ScenarioInput(
                    time_of_day=time_of_day, weather=weather, visibility=visibility, surface=surface,
                    pedestrian_density=density, speed_mps=speed, location=location, navigation_mode=navigation,
                    sensors=[v.strip() for v in sensors.split(",") if v.strip()], faults=selected_faults,
                    documented_capabilities=capabilities, notes=notes
                )
                s.store.set_setting("scenario_input", new_inputs.model_dump())
                st.success("Operating context and active faults updated.")
                st.rerun()
            except (ValidationError, ValueError) as exc:
                error(exc)
        if st.button("Reset to default conditions"):
            s.store.set_setting("scenario_input", ScenarioInput().model_dump())
            st.rerun()

    inputs = ScenarioInput.model_validate(s.store.setting("scenario_input", ScenarioInput().model_dump()))

    # Context Summary Badge
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Context", f"{inputs.time_of_day.title()}, {inputs.weather.title()}")
    c2.metric("Active Faults", ", ".join(inputs.faults) or "None")
    c3.metric("Pedestrians", inputs.pedestrian_density.title())
    c4.metric("Mode", inputs.navigation_mode.title())

    st.markdown("#### Select Inference Option")
    mode_options = [
        ("qualcomm", "Cloud (Qualcomm)"),
        ("uno_q", "Uno Q (Arduino)"),
        ("laptop_local", "Local (Laptop)"),
    ]
    mode_labels = [label for _, label in mode_options]
    selected_label = st.radio("Inference Target", mode_labels, horizontal=True, label_visibility="collapsed", key="builder_selected_target")

    # Determine chosen backend
    selected_backend, active_label = next(
        ((b, l) for b, l in mode_options if l == selected_label),
        mode_options[0]
    )

    # Content under the selected option
    with st.container(border=True):
        st.markdown(f"### {active_label} Generator & Risk Evaluation")

        cfg = settings.get(selected_backend, {})
        inventory = st.session_state.get(f"catalogue_{selected_backend}", [])
        if not inventory:
            try:
                adapter = make_adapter(selected_backend, settings)
                inventory = adapter.list_models(refresh=False)
                if inventory:
                    st.session_state[f"catalogue_{selected_backend}"] = inventory
            except Exception:
                pass
        model_ids = list(dict.fromkeys(m["id"] for m in inventory))
        persisted = cfg.get("model_id", "")
        if persisted and persisted not in model_ids:
            model_ids.insert(0, persisted)
        if not model_ids and selected_backend == "qualcomm":
            model_ids = ["DeepSeek-R1-Distill-Llama-70B"]
        elif not model_ids and selected_backend == "uno_q":
            model_ids = ["qwen3:0.6b"]
        elif not model_ids and selected_backend == "laptop_local":
            model_ids = ["qwen3:1.7b"]
        elif not model_ids:
            model_ids = [persisted or "Default Model"]

        col_m, col_l, col_btn1, col_btn2 = st.columns([2, 1, 1.5, 1.2])
        with col_m:
            chosen_model = st.selectbox("Model", model_ids, index=0, key=f"sel_m_{selected_backend}")
        with col_l:
            scenario_limit = st.number_input("Max scenarios", 1, 10, 3, key=f"lim_{selected_backend}")
        with col_btn1:
            st.write("")
            st.write("")
            run_btn = st.button(f"Generate ({active_label.split()[0]})", type="primary", key=f"gen_{selected_backend}", use_container_width=True)
        with col_btn2:
            st.write("")
            st.write("")
            run_all_btn = st.button("Run All 3", key=f"gen_all_{selected_backend}", use_container_width=True)

        if run_btn:
            try:
                gen_opts = Generation(max_tokens=settings.get("max_tokens", 1800), temperature=0.0, seed=42)
                selection = Selection(backend=selected_backend, model_id=chosen_model, generation=gen_opts)
                request = RunRequest(
                    inputs=inputs,
                    selections=[selection],
                    retrieval_limit=settings["retrieval_limit"],
                    relevance_threshold=settings["relevance_threshold"],
                    citation_link_validity_threshold=settings.get('citation_link_validity_threshold', 1.0),
                    generation_mode='scenario_drafts',
                    max_scenarios=scenario_limit,
                )
                st.session_state.active_run = s.manager.start(request, dataset, s.matrix(), settings)
                st.rerun()
            except (ValidationError, ValueError) as exc:
                error(exc)

        if run_all_btn:
            try:
                selections = []
                for b, _, _ in mode_options:
                    m_id = settings[b].get("model_id") or ("DeepSeek-R1-Distill-Llama-70B" if b == "qualcomm" else ("qwen3:0.6b" if b == "uno_q" else "qwen3:1.7b"))
                    selections.append(Selection(backend=b, model_id=m_id, generation=Generation(max_tokens=settings.get('max_tokens', 1800), temperature=0.0, seed=42)))
                request = RunRequest(
                    inputs=inputs,
                    selections=selections,
                    retrieval_limit=settings["retrieval_limit"],
                    relevance_threshold=settings["relevance_threshold"],
                    citation_link_validity_threshold=settings.get('citation_link_validity_threshold', 1.0),
                    generation_mode='scenario_drafts',
                    max_scenarios=scenario_limit,
                )
                st.session_state.active_run = s.manager.start(request, dataset, s.matrix(), settings)
                st.rerun()
            except (ValidationError, ValueError) as exc:
                error(exc)

        # Check Active or Latest Run
        active_id = st.session_state.get("active_run")
        run_data = None
        run_identity = None

        if active_id:
            try:
                active_job = s.manager.get(active_id)
                run_identity = active_id
                if active_job.get("status") == "running":
                    with st.spinner(f"Inference in progress on {active_label}… results will appear automatically"):
                        import time as _time
                        _time.sleep(2)
                    st.rerun()
                run_data = active_job
            except KeyError:
                st.session_state.pop("active_run", None)

        if not run_data:
            saved_runs = s.store.list("run", limit=1)
            if saved_runs:
                run_data = saved_runs[0]["payload"]
                run_identity = saved_runs[0]["id"]

        st.divider()

        # Display results under the selected mode
        if run_data and selected_backend in run_data.get("results", {}):
            result = run_data["results"][selected_backend]
            status = result.get("status", "unknown")
            scenarios = result.get("scenarios", [])
            latency = result.get("generation_latency_ms")
            tokens = result.get("runtime_metrics", {}).get("total_tokens")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Model", result.get("model_id") or chosen_model)
            s2.metric("Execution", status.title())
            s3.metric("Latency", f"{latency / 1000:.1f} s" if latency is not None else "N/A")
            s4.metric("Tokens Used", f"{tokens:,}" if tokens else "N/A")

            if status == "failed":
                st.error(f"Inference error: {result.get('error', 'Execution failed')}")
            elif not scenarios:
                st.warning(f"No scenarios were generated for {active_label} in this run.")
            else:
                # Recalculate Risk Evaluation dynamically
                reviews = s.store.reviews(run_identity) if run_identity else []
                calculated_risk = recalculate(run_data, reviews)
                mode_eval = calculated_risk.get(selected_backend, {})
                included_scopes = set(mode_eval.get("included", []))
                excluded_scopes = set(mode_eval.get("engineer_excluded", []))

                risk_level = (mode_eval.get("risk_level") or ("NO INCLUDED SCENARIOS" if excluded_scopes else "CALCULATING")).upper()
                risk_score = mode_eval.get("risk_score")
                risk_score_100 = min(100, round((float(risk_score) / 9.0) * 100)) if risk_score is not None else None
                fill_cls = "risk-meter-fill-high" if risk_score_100 and risk_score_100 >= 66 else ("risk-meter-fill-med" if risk_score_100 and risk_score_100 >= 36 else "risk-meter-fill-low")
                meter_pct = risk_score_100 if risk_score_100 is not None else 0

                # Risk Summary Box with Circular Odometer Gauge
                with st.container(border=True):
                    g_c1, g_c2 = st.columns([1.1, 2.9])
                    with g_c1:
                        st.markdown(render_odometer_gauge(risk_score_100, risk_level), unsafe_allow_html=True)
                    with g_c2:
                        st.markdown("**Safety Risk Evaluation (Recalculated)**")
                        rk1, rk2, rk3 = st.columns(3)
                        rk1.metric("Overall Risk", risk_level)
                        rk2.metric("Included Scenarios", len(included_scopes))
                        rk3.metric("Excluded Scenarios", len(excluded_scopes))
                        for step in mode_eval.get("calculation_steps", []):
                            st.caption(f"• {step}")

                st.markdown(f"#### Proposed Scenarios ({len(scenarios)})")
                st.caption("Remove any scenario below to immediately recalculate the risk evaluation above.")

                export_list = []
                for i, sc in enumerate(scenarios, 1):
                    scenario_id = sc.get("scenario_id", f"DRAFT-{i}")
                    scope = f"{selected_backend}:{scenario_id}"
                    is_excluded = scope in excluded_scopes
                    for rev in reviews:
                        if rev.get("scenario_id") == scope and rev.get("payload", {}).get("decision") == "rejected":
                            is_excluded = True
                            break

                    hazards = sc.get("hazards", [])
                    hazard_id = hazards[0].get("hazard_id") if hazards else None
                    risk_assess = sc.get("risk_assessments", {})
                    sc_risk = risk_assess.get(hazard_id, {}) if hazard_id else {}
                    sc_level = (sc_risk.get("risk_level") or "unknown").upper()
                    raw_sc_score = sc_risk.get("risk_score")
                    sc_score_100 = min(100, round((float(raw_sc_score) / 9.0) * 100)) if isinstance(raw_sc_score, (int, float)) else "—"
                    sc_sev = sc_risk.get("severity", "—")
                    sc_lik = sc_risk.get("likelihood", "—")

                    export_list.append({
                        "mode": active_label,
                        "scenario_id": scenario_id,
                        "title": sc.get("title", "Untitled"),
                        "status": "Excluded" if is_excluded else "Included",
                        "severity": sc_sev,
                        "likelihood": sc_lik,
                        "risk_score_100": sc_score_100,
                        "risk_level": sc_level,
                    })

                    with st.container(border=True):
                        top_left, top_right = st.columns([3.2, 1.2])
                        with top_left:
                            title_text = sc.get("title", "Untitled scenario")
                            status_badge = '<span class="pill-badge pill-risk-high">EXCLUDED</span>' if is_excluded else '<span class="pill-badge pill-local">INCLUDED IN RISK</span>'
                            st.markdown(f"{status_badge} &nbsp; **{scenario_id}: {title_text}**", unsafe_allow_html=True)
                            st.caption(f"Risk: **{sc_level}** (Score: {sc_score_100}/100 · Severity: {sc_sev} · Likelihood: {sc_lik})")
                        with top_right:
                            if is_excluded:
                                if st.button("Re-include in Risk", key=f"gen_reinc_{selected_backend}_{scenario_id}_{i}", use_container_width=True):
                                    if run_identity:
                                        s.store.delete_reviews_for_scenario(run_identity, scope)
                                    st.rerun()
                            else:
                                if st.button("Remove from Risk", key=f"gen_rem_{selected_backend}_{scenario_id}_{i}", use_container_width=True):
                                    if run_identity:
                                        s.store.review(run_identity, scope, "rejected", "Engineer Review", "Excluded from risk evaluation")
                                    st.rerun()

                        stmts = sc.get("statements", [])
                        if stmts:
                            for stmt in stmts:
                                rec_id = stmt.get("record_id", "")
                                text = stmt.get("text", "")
                                if rec_id.startswith("H-") or rec_id.startswith("HAZ-"):
                                    st.markdown(f'<span class="pill-stmt-hazard">HAZARD</span> &nbsp; **{rec_id}:** {text}', unsafe_allow_html=True)
                                elif rec_id.startswith("M-") or rec_id.startswith("MIT-"):
                                    st.markdown(f'<span class="pill-stmt-mit">MITIGATION</span> &nbsp; **{rec_id}:** {text}', unsafe_allow_html=True)
                                elif rec_id.startswith("T-") or rec_id.startswith("TEST-"):
                                    st.markdown(f'<span class="pill-stmt-test">VERIFICATION TEST</span> &nbsp; **{rec_id}:** {text}', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'• **{rec_id}:** {text}')
                        elif sc.get("scenario_narrative"):
                            st.write(sc.get("scenario_narrative"))

                        if sc_risk.get("explanation"):
                            with st.expander("Knowledge Base Risk Details", expanded=False):
                                st.write(sc_risk.get("explanation"))

                # Download options for this mode
                st.write("")
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        f"Export {active_label} Scenarios (JSON)",
                        json_bytes({"mode": active_label, "recalculated_risk": mode_eval, "scenarios": scenarios}),
                        f"{selected_backend}-scenarios.json",
                        "application/json",
                        use_container_width=True,
                        key=f"dl_json_{selected_backend}"
                    )
                with dl2:
                    st.download_button(
                        f"Export {active_label} Scenarios (CSV)",
                        csv_bytes(export_list),
                        f"{selected_backend}-scenarios.csv",
                        "text/csv",
                        use_container_width=True,
                        key=f"dl_csv_{selected_backend}"
                    )
        else:
            st.info(f"No scenario generation results yet for {active_label}. Click 'Generate ({active_label.split()[0]})' above to run inference and view proposed scenarios.")


def model_studio(s):
    settings = s.settings()
    st.caption("Inspect runtime catalogues, choose active models, and test isolated inference for each environment without vertical scrolling.")

    # 3-column top status strip
    top_c1, top_c2, top_c3 = st.columns(3)
    with top_c1:
        st.markdown(f'<div class="light-card-cloud"><span class="pill-badge pill-cloud">Cloud</span> &nbsp; <strong>Qualcomm Cirrascale</strong><br><small style="color:#64748b;">Target: {settings["qualcomm"].get("model_id") or "DeepSeek-R1-Distill-Llama-70B"}</small></div>', unsafe_allow_html=True)
    with top_c2:
        st.markdown(f'<div class="light-card-uno"><span class="pill-badge pill-uno">Edge</span> &nbsp; <strong>Arduino UNO Q</strong><br><small style="color:#64748b;">Target: {settings["uno_q"].get("model_id") or "qwen3:0.6b"}</small></div>', unsafe_allow_html=True)
    with top_c3:
        st.markdown(f'<div class="light-card-local"><span class="pill-badge pill-local">Local</span> &nbsp; <strong>Laptop Ollama</strong><br><small style="color:#64748b;">Target: {settings["laptop_local"].get("model_id") or "qwen3:1.7b"}</small></div>', unsafe_allow_html=True)

    st.write("")

    ms_tabs = st.tabs(["Qualcomm Cloud", "Arduino UNO Q", "Laptop Local"])
    tab_list = [("qualcomm", ms_tabs[0]), ("uno_q", ms_tabs[1]), ("laptop_local", ms_tabs[2])]

    for backend, current_tab in tab_list:
        label = LABELS[backend]
        with current_tab:
            st.subheader(label)
            st.caption(settings[backend]["base_url"] or "Address not configured")
            if f"catalogue_{backend}" not in st.session_state:
                if st.session_state.get(f"reachable_{backend}", False):
                    try:
                        adapter = make_adapter(backend, settings)
                        inv = adapter.list_models(refresh=False)
                        if inv:
                            st.session_state[f"catalogue_{backend}"] = inv
                    except Exception:
                        st.session_state[f"catalogue_{backend}"] = []

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
                if a.button("Save selection", key=f"save_{backend}", use_container_width=True):
                    settings[backend]["model_id"] = selected
                    s.store.set_setting("preferences", settings)
                    st.success("Selection saved independently for this backend")
                if b.button("Test real scenario inference", key=f"test_{backend}", use_container_width=True):
                    request = RunRequest(inputs=ScenarioInput.model_validate(s.store.setting('scenario_input', ScenarioInput().model_dump())), selections=[Selection(backend=backend, model_id=selected, generation=Generation(max_tokens=settings.get('max_tokens', 1800)))], generation_mode='scenario_drafts')
                    st.session_state.active_run = s.manager.start(request, s.dataset(), s.matrix(), settings)
                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    st.button("Load model (runtime unsupported)", key=f"load_{backend}", disabled=True, use_container_width=True)
                with btn_c2:
                    st.button("Unload model (runtime unsupported)", key=f"unload_{backend}", disabled=True, use_container_width=True)
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
    else:
        # Inference is still running in a background thread; auto-refresh every 3 s so
        # results appear automatically when the thread completes, without a manual reload.
        import time as _time
        with st.spinner("Inference running — results will appear automatically…"):
            _time.sleep(3)
        st.rerun()
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
            st.caption("Matched inputs, evidence and risk policy within this run. Compare all 3 modes across latency, safety risk, and validity.")
            st.plotly_chart(unified_benchmark_figure(measured, st.session_state.dark), use_container_width=True, key="matched_unified_benchmark")
            a, b = st.columns(2)
            a.plotly_chart(latency_figure(measured, st.session_state.dark), use_container_width=True, key="matched_latency")
            b.plotly_chart(validation_figure(measured, st.session_state.dark), use_container_width=True, key="matched_validation")
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

    reviews = s.store.reviews(run['id'])
    results = recalculate(run, reviews)

    st.markdown("### Safety Risk Analysis (Scale: 0 – 100)")
    st.caption("Deterministic, rule-based risk calculated from frozen knowledge-base severity and likelihood. Standardized to a 100-point scale for comparative engineering audits.")

    # 1. Comparative 3-Mode Risk Cards
    modes_meta = [
        ("qualcomm", "Qualcomm Cloud"),
        ("laptop_local", "Laptop Local"),
        ("uno_q", "Arduino UNO Q"),
    ]

    rc1, rc2, rc3 = st.columns(3)
    cols_map = {"qualcomm": rc1, "laptop_local": rc2, "uno_q": rc3}

    for b, label in modes_meta:
        col = cols_map[b]
        b_res = results.get(b)
        with col:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                if b_res and b_res.get('included'):
                    raw_sc = b_res.get('risk_score')
                    sc_100 = min(100, round((float(raw_sc) / 9.0) * 100)) if raw_sc is not None else 0
                    lvl = (b_res.get('risk_level') or "unassigned").upper()
                    st.markdown(render_odometer_gauge(sc_100, lvl, width=200, height=120), unsafe_allow_html=True)

                    c_a, c_b = st.columns(2)
                    c_a.caption(f"Included: **{len(b_res['included'])}**")
                    c_b.caption(f"Excluded: **{len(b_res['engineer_excluded'])}**")
                elif b_res and b_res.get('engineer_excluded'):
                    st.caption("All proposed scenarios excluded")
                    st.markdown('<span class="pill-badge pill-slate">No Active Scenarios</span>', unsafe_allow_html=True)
                else:
                    st.caption("Mode not evaluated in this run")
                    st.markdown('<span class="pill-badge pill-slate">Not Run</span>', unsafe_allow_html=True)

    st.write("")

    # 2. Detailed Recalculated Risk Panel
    risk_selection_panel(run, reviews)

    # 3. Comprehensive Hazard Risk Traceability Table
    all_hazards = []
    for b_key, b_val in results.items():
        mode_label = LABELS.get(b_key, b_key)
        risk_map = b_val.get('risk', {})
        for hid in b_val.get('hazard_ids', []):
            h_info = risk_map.get(hid, {})
            raw_h_score = h_info.get('risk_score')
            h_score_100 = min(100, round((float(raw_h_score) / 9.0) * 100)) if raw_h_score is not None else None
            all_hazards.append({
                "mode": mode_label,
                "hazard_id": hid,
                "severity": h_info.get('severity', '—'),
                "likelihood": h_info.get('likelihood', '—'),
                "raw_matrix_score": raw_h_score if raw_h_score is not None else "—",
                "risk_score_100": f"{h_score_100} / 100" if h_score_100 is not None else "—",
                "risk_level": (h_info.get('risk_level') or 'unknown').upper(),
                "status": "Included in Calculation",
            })

    if all_hazards:
        st.write("")
        st.subheader("Traceable Hazard Risk Matrix (0 – 100 Scale)")
        st.caption("Individual hazards identified from approved knowledge-base inputs, mapped through the active risk matrix and scaled out of 100.")
        st.dataframe(all_hazards, use_container_width=True, hide_index=True)

    st.info("**Dynamic Risk Control**: Rejecting or excluding any scenario in **Scenario Register** or **Scenario Builder** immediately recalculates the 0–100 safety risk score in real time.")
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
    # Unified 3-Mode Performance Benchmark from Latest Run
    saved_runs = s.store.list("run", 10)
    if saved_runs:
        st.subheader("Unified 3-Mode Benchmark Comparison")
        st.caption("Direct visual comparison of Qualcomm Cloud, Arduino UNO Q, and Laptop Local across Latency, Safety Risk (out of 100), and Valid Scenarios in a single graph.")
        sel_run_id = st.selectbox(
            "Select Run for 3-Mode Graphical Benchmark",
            [r["id"] for r in saved_runs],
            format_func=lambda i: next(f"{r['created']} · {i[:10]}" for r in saved_runs if r["id"] == i),
            key="eval_run_selector"
        )
        run_payload = s.store.get(sel_run_id)["payload"]
        measured_rows = comparison_rows(run_payload)
        st.plotly_chart(unified_benchmark_figure(measured_rows, st.session_state.dark), use_container_width=True, key="unified_bench_eval_chart")
        st.divider()

    label_file = 'multi_fault_cases_25.json' if s.dataset().dataset_id in {'multi-fault-draft-120', 'multi-fault-draft-180'} else 'labelled_cases_25.json'
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
    if entries:
        key = st.selectbox("Scenario to inspect", list(entries))
        scenario_detail(entries[key], run["snapshot"]["evidence"])
        st.download_button("Export individual scenario", json_bytes({"scenario": entries[key], "configuration": run["snapshot"], "review_scope": key}), f"scenario-{run['id']}.json")
        st.download_button("Export scenario CSV", csv_bytes([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in entries[key].items()}]), f"scenario-{run['id']}.csv")
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
    st.caption("Central Device & Inference Control: Manage connectivity, runtime telemetry, model catalogues, and diagnostics for Cloud, Edge, and Local environments.")

    # 1. Device Fleet Overview Status Cards
    c1, c2, c3 = st.columns(3)

    # Cloud Status
    q_url = settings["qualcomm"]["base_url"]
    q_key = bool(os.getenv("QUALCOMM_API_KEY"))
    q_reach = st.session_state.get("reachable_qualcomm", False)
    with c1:
        with st.container(border=True):
            st.markdown("### Qualcomm Cloud")
            st.caption(f"Host: `{urlsplit(q_url).netloc or 'cirrascale.com'}`")
            status_cls = "pill-cloud" if q_reach else ("pill-uno" if q_key else "pill-risk-high")
            status_txt = "Connected" if q_reach else ("API Key Set" if q_key else "Key Required")
            st.markdown(f'<span class="pill-badge {status_cls}">{status_txt}</span>', unsafe_allow_html=True)
            st.write("")
            if st.button("Ping Cloud API", key="dev_ping_cloud", use_container_width=True):
                try:
                    cat = make_adapter("qualcomm", settings).list_models(refresh=True)
                    st.session_state.catalogue_qualcomm = cat
                    st.session_state.reachable_qualcomm = True
                    st.success(f"Online: {len(cat)} models available")
                except BackendError as exc:
                    st.session_state.reachable_qualcomm = False
                    error(exc)

    # Uno Q Status
    u_url = settings["uno_q"]["base_url"] or "Not configured"
    u_reach = st.session_state.get("reachable_uno_q", False)
    with c2:
        with st.container(border=True):
            st.markdown("### Arduino UNO Q")
            st.caption(f"Address: `{u_url}`")
            status_cls = "pill-uno" if u_reach else "pill-slate"
            status_txt = "Online" if u_reach else "Disconnected"
            st.markdown(f'<span class="pill-badge {status_cls}">{status_txt}</span>', unsafe_allow_html=True)
            st.write("")
            if st.button("Connect Board", key="dev_ping_uno", use_container_width=True):
                try:
                    adapter = make_adapter("uno_q", settings)
                    st.session_state.board_status = adapter.get_runtime_information()
                    st.session_state.reachable_uno_q = True
                    st.success("Board reached successfully")
                except BackendError as exc:
                    st.session_state.pop("board_status", None)
                    st.session_state.reachable_uno_q = False
                    error(exc)

    # Local Laptop Status
    l_url = settings["laptop_local"]["base_url"]
    l_reach = st.session_state.get("reachable_laptop_local", False)
    with c3:
        with st.container(border=True):
            st.markdown("### Laptop Local")
            st.caption(f"Endpoint: `{urlsplit(l_url).netloc or '127.0.0.1:11434'}`")
            status_cls = "pill-local" if l_reach else "pill-slate"
            status_txt = "Available" if l_reach else "Unverified"
            st.markdown(f'<span class="pill-badge {status_cls}">{status_txt}</span>', unsafe_allow_html=True)
            st.write("")
            if st.button("Check Local Service", key="dev_ping_local", use_container_width=True):
                try:
                    cat = make_adapter("laptop_local", settings).list_models(refresh=True)
                    st.session_state.catalogue_laptop_local = cat
                    st.session_state.reachable_laptop_local = True
                    st.success(f"Local Server Ready: {len(cat)} models")
                except BackendError as exc:
                    st.session_state.reachable_laptop_local = False
                    error(exc)

    st.write("")

    # 2. Detailed Controls Per Environment
    tabs = st.tabs(["Qualcomm Cloud Hub", "Arduino UNO Q Workstation", "Local Inference Station"])

    with tabs[0]:
        st.subheader("Qualcomm AI Suite Cloud Service")
        st.caption("Manages Cirrascale cloud inference connectivity, token management, and live model catalog inspection.")
        q_col1, q_col2 = st.columns([2, 1])
        with q_col1:
            st.text_input("Cirrascale Base Endpoint", value=settings["qualcomm"]["base_url"], disabled=True, help="Update in Settings if using alternate allowed hosts.")
            cat = st.session_state.get("catalogue_qualcomm", [])
            if cat:
                model_ids = [m["id"] for m in cat]
                selected_q = st.selectbox("Active Cloud Model", model_ids, index=0 if model_ids else 0, key="dev_q_model_sel")
                st.caption(f"Selected: `{selected_q}` · Configured to use temperature=0, seed=42.")
            else:
                st.info("Catalog not loaded in memory. Click 'Refresh Cloud Catalogue' to query Cirrascale API.")
        with q_col2:
            st.write("**Quick Actions**")
            if st.button("Refresh Cloud Catalogue", key="dev_q_refresh_cat", use_container_width=True):
                try:
                    inventory = make_adapter("qualcomm", settings).list_models(refresh=True)
                    st.session_state.catalogue_qualcomm = inventory
                    st.session_state.reachable_qualcomm = True
                    st.success(f"Retrieved {len(inventory)} models.")
                except BackendError as exc:
                    error(exc)
            token_status = "Configured in Environment" if os.getenv("QUALCOMM_API_KEY") else "Missing Token"
            st.caption(f"Auth Token: **{token_status}**")

    with tabs[1]:
        st.subheader("Arduino UNO Q Board Control & Hardware RPC")
        st.caption("Interact with the authenticated Linux daemon on the board, inspect hardware telemetry, and run firmware tests.")
        u_adapter = make_adapter("uno_q", settings)
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            if st.button("Refresh Board Status", key="dev_uno_refresh", use_container_width=True):
                try:
                    st.session_state.board_status = u_adapter.get_runtime_information()
                    st.session_state.reachable_uno_q = True
                    st.success("Board status updated.")
                except BackendError as exc:
                    st.session_state.pop("board_status", None)
                    st.session_state.reachable_uno_q = False
                    error(exc)
        with u_col2:
            if st.button("Disconnect Board Session", key="dev_uno_disconnect", use_container_width=True):
                st.session_state.pop("board_status", None)
                st.session_state.reachable_uno_q = False
                st.info("Board session disconnected.")

        status = st.session_state.get("board_status")
        if status:
            with st.container(border=True):
                st.markdown("**Board Telemetry & Runtime Info**")
                st.json(status)

        st.markdown("---")
        st.markdown("#### Real-Time Edge Predict (`/predict`)")
        st.caption("Executes requests directly against the Arduino Uno Q Python App: `POST /predict` (`{'input': text}`). Controls the physical LED via `Bridge.call('set_led')`.")
        pred_c1, pred_c2 = st.columns([3, 1])
        with pred_c1:
            test_prompt = st.text_input("Predict Input", value="turn on", key="dev_uno_predict_input", help="e.g. 'turn on', 'turn off', or safety instructions")
        with pred_c2:
            st.write("")
            if st.button("Send to Uno Q", key="dev_uno_predict_btn", use_container_width=True):
                try:
                    import time
                    t0 = time.time()
                    p_res = u_adapter.predict(test_prompt)
                    latency = (time.time() - t0) * 1000
                    st.session_state.board_predict_result = {
                        "input": test_prompt,
                        "response": p_res,
                        "latency_ms": f"{latency:.1f} ms",
                        "status": "success"
                    }
                    st.session_state.reachable_uno_q = True
                    st.success(f"Uno Q Replied in {latency:.1f} ms!")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
        if "board_predict_result" in st.session_state:
            with st.container(border=True):
                st.markdown("**Last Predict Response from Uno Q**")
                st.json(st.session_state.board_predict_result)

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Retrieve Board Agent Logs", key="dev_uno_logs", use_container_width=True):
                    try:
                        st.json(u_adapter.request("GET", "/v1/logs"))
                    except BackendError as exc:
                        error(exc)
            with b2:
                tests = status.get("mcu", {}).get("tests", [])
                if tests:
                    name = st.selectbox("MCU Firmware Test", tests, key="dev_uno_test_sel")
                    confirmed = st.checkbox("Confirm read-only probe RPC", key="dev_uno_conf")
                    if st.button("Execute Hardware Test", disabled=not confirmed, key="dev_uno_run_test", use_container_width=True):
                        try:
                            result = u_adapter.request("POST", "/v1/hardware-tests", json={"name": name, "confirmed": confirmed})
                            s.store.save("hardware_test", result)
                            st.json(result)
                        except BackendError as exc:
                            error(exc)
                else:
                    st.caption("No MCU commands verified. Read-only probe sketch required.")

            with st.expander("Register GGUF Model on Board Linux"):
                board_path = st.text_input("Absolute path to .gguf on board Linux", key="dev_uno_gguf_path")
                if st.button("Register Board GGUF", disabled=not board_path, key="dev_uno_reg_gguf"):
                    try:
                        st.json(u_adapter.request("POST", "/v1/models/register", json={"path": board_path}))
                        st.success("Model registered with board Linux agent.")
                    except BackendError as exc:
                        error(exc)
        else:
            st.info("Board not currently connected. Ensure its IP address is configured in Settings, then click 'Connect Board'.")

    with tabs[2]:
        st.subheader("Local Inference Engine (Laptop / Ollama / Llama.cpp)")
        st.caption("Manage local runtime options, directory scan for GGUF model files, and fast inference toggles.")
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            st.text_input("Local Endpoint URL", value=settings["laptop_local"]["base_url"], disabled=True)
            fast_mode = st.checkbox(
                "Ollama Fast Mode (Disable Thinking / Request Clean JSON)",
                value=settings["laptop_local"].get("ollama_fast_mode", True),
                help="Suppresses reasoning chain tokens to deliver fast, deterministic output."
            )
            if fast_mode != settings["laptop_local"].get("ollama_fast_mode"):
                settings["laptop_local"]["ollama_fast_mode"] = fast_mode
                s.store.set_setting("preferences", settings)
                st.success("Ollama fast mode setting updated.")
        with l_col2:
            st.write("**Model Files & Directory Scanner**")
            if st.button("Scan Permitted Model Directories", key="dev_local_scan", use_container_width=True):
                st.session_state.local_files = discover_files(settings.get("model_dirs", []))
                st.success(f"Discovered {len(st.session_state.local_files)} files.")

        files = st.session_state.get("local_files", [])
        if files:
            st.markdown("##### Discovered Local Model Files")
            st.dataframe(files, use_container_width=True)
        else:
            st.caption(f"Permitted directories: {', '.join(settings.get('model_dirs', [])) or 'None configured'}. Add paths in Settings.")


def settings_page(s):
    settings = s.settings()
    with st.form("settings"):
        st.markdown("#### 3-Tier Model Execution Settings")
        st.caption("Configure endpoints, model targets, and hardware metadata side-by-side to minimize scrolling.")

        c_cloud, c_uno, c_local = st.columns(3)
        cols_map = {
            "qualcomm": (c_cloud, "pill-cloud", "Qualcomm Cloud", "light-card-cloud"),
            "uno_q": (c_uno, "pill-uno", "Arduino UNO Q", "light-card-uno"),
            "laptop_local": (c_local, "pill-local", "Laptop Local", "light-card-local")
        }

        endpoints, selected, params = {}, {}, {}
        ollama_fast = settings["laptop_local"].get("ollama_fast_mode", True)

        for backend, (col, pill_cls, title, card_cls) in cols_map.items():
            with col:
                with st.container(border=True):
                    st.markdown(f'<span class="pill-badge {pill_cls}">{title}</span>', unsafe_allow_html=True)
                    st.write("")
                    endpoints[backend] = st.text_input("Base URL", settings[backend]["base_url"], key=f"endpoint_{backend}")
                    if backend == 'laptop_local':
                        ollama_fast = st.checkbox(
                            'Ollama fast response (no-think JSON)',
                            value=settings[backend].get('ollama_fast_mode', urlsplit(settings[backend]['base_url']).port == 11434),
                            help='Requests reasoning_effort=none and JSON output.'
                        )
                    selected[backend] = st.text_input("Selected model ID", settings[backend].get("model_id", ""), key=f"configured_{backend}")
                    params[backend] = {'temperature': 0.0, 'seed': 42, 'max_tokens': settings.get('max_tokens', 1800)}
                    settings[backend]['parameter_size'] = st.text_input('Parameter size', settings[backend].get('parameter_size') or '', key=f'parameters_{backend}', help='e.g. 70B, 0.6B') or None
                    settings[backend]['quantization'] = st.text_input('Quantization', settings[backend].get('quantization') or '', key=f'quantization_{backend}', help='e.g. Q4_K_M, INT8') or None

        st.markdown("---")
        st.markdown("#### Global Policy & Pipeline Parameters")
        g1, g2 = st.columns(2)
        with g1:
            max_tokens = st.number_input('Shared maximum generated tokens', 64, 8192, settings.get('max_tokens', 1800))
            backend_timeout_seconds = st.number_input('Backend request timeout (seconds)', 30, 900, int(settings.get('backend_timeout_seconds', 900)), 30, help='Applies to laptop, UNO Q and cloud requests.')
            citation_link_validity_threshold = st.number_input('citation_link_validity threshold for human review', 0.0, 1.0, float(settings.get('citation_link_validity_threshold', 1.0)), 0.05)
        with g2:
            limit = st.number_input("TF-IDF result limit", 1, 40, settings["retrieval_limit"])
            threshold = st.number_input("Minimum relevance", 0.0, 1.0, float(settings["relevance_threshold"]), 0.01)
            directories = st.text_area("Permitted local model directories (one absolute path per line)", "\n".join(settings.get("model_dirs", [])), height=100)
        submit = st.form_submit_button("Validate and save settings", type="primary", use_container_width=True)
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
    st.set_page_config(page_title="SafeScenario AI", layout="wide")
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
                "Scenario Register":register_page,"Knowledge Base":knowledge,"Risk Analysis":risk_analysis,
                "Evaluation and Benchmarks":evaluation_page,"Reports and History":history,"Device Control":device,"Settings":settings_page}
    handlers[page](s)

