"""Engineering workspace backed by shared services; no synthetic activity or model results."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import plotly.express as px
import streamlit as st
from pydantic import ValidationError

from backend.configuration import ROOT, validate_url
from backend.schemas import Generation, RunRequest, ScenarioInput, Selection
from backend.services import Services
from evaluation.benchmark import BenchmarkDataset, draft_dataset
from knowledge_base.core import Dataset, parse_upload, retrieve
from models.adapters import BackendError, discover_files, make_adapter
from reporting.exports import comparison_rows, csv_bytes, json_bytes, report
from reporting.pdf_export import pdf_bytes
from risk_engine.calculator import Classification, RiskMatrix, calculate
from simulation.engine import SimulationInput, simulate

LABELS = {"laptop_local": "Laptop local", "qualcomm": "Qualcomm cloud", "uno_q": "Arduino UNO Q"}
PAGES = ["Overview", "Scenario Builder", "Model Studio", "Model Comparison", "Knowledge Base", "Risk Analysis", "Test Simulation", "Evaluation and Benchmarks", "Reports and History", "Device Control", "Settings"]


@st.cache_resource
def services():
    return Services()


def theme(dark):
    bg, surface, text, muted, border = ("#101827", "#182235", "#e5edf8", "#adbdd3", "#34445d") if dark else ("#f5f7fb", "#ffffff", "#17233b", "#52627a", "#dbe2ee")
    st.markdown(f"""<style>
    .stApp, [data-testid="stAppViewContainer"] {{ background:{bg}; color:{text}; }}
    [data-testid="stHeader"], [data-testid="stSidebar"] {{ background:{surface}; color:{text}; }}
    [data-testid="stSidebar"] {{ border-right:1px solid {border}; }}
    .block-container {{ padding-top:2.5rem; max-width:1400px; }}
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
    [data-baseweb="select"]>div, [data-baseweb="input"], [data-baseweb="textarea"], textarea, input {{background:{surface}!important;color:{text}!important;}}
    button[kind="secondary"] {{background:{surface};color:{text};border-color:{border};}}
    [data-testid="stAlert"] p {{ color:{text}; }}
    </style>""", unsafe_allow_html=True)


def error(exc):
    if isinstance(exc, ValidationError):
        st.error(json.dumps(exc.errors(include_input=False, include_url=False), indent=2))
    else:
        st.error(str(exc))


def scenario_detail(scenario, evidence):
    st.subheader(scenario["title"])
    st.caption(f"{scenario['scenario_id']} · AI proposal · pending engineering review")
    st.write(scenario["scenario_narrative"])
    conditions, hazards, sources, links, checks = st.tabs(["Conditions", "Hazards & risk", "Evidence", "Mitigations & tests", "Verification"])
    with conditions:
        st.json(scenario["input_conditions"])
    with hazards:
        for hazard in scenario["hazards"]:
            st.write(f"{hazard['hazard_id']} · {hazard['description']}")
            st.write("Potential consequences", hazard["consequences"])
            st.caption("Hypothesis" if hazard["hypothesis"] else "Source-backed claim proposed by model; review needed")
            st.json(scenario["risk_assessments"][hazard["hazard_id"]])
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
            use = st.checkbox("Include in run", value=backend == "laptop_local", key=f"{prefix}_use_{backend}") if selectable else True
            if use:
                params = cfg.get("generation", {})
                chosen.append(Selection(backend=backend, model_id=selected, generation=Generation.model_validate(params)))
    return chosen


def overview(s):
    runs = s.store.list("run")
    results = [r for run in runs for r in run["payload"]["results"].values()]
    scenarios = [scenario for result in results for scenario in result.get("scenarios", [])]
    review_map = {}
    for run in runs:
        for review in s.store.reviews(run["id"]):
            review_map[(run["id"], review["scenario_id"])] = review["payload"]["decision"]
    awaiting = sum((run["id"], f"{backend}:{scenario['scenario_id']}") not in review_map for run in runs for backend, result in run["payload"]["results"].items() for scenario in result.get("scenarios", []))
    risks = [risk for scenario in scenarios for risk in scenario["risk_assessments"].values()]
    assessed = [row["payload"]["assessment"] for row in s.store.list("risk_assessment")]
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Scenarios generated", "Awaiting review", "High risk assessments", "KB records"], [len(scenarios), awaiting, sum(r["risk_level"] == "high" for r in risks + assessed), len(s.dataset().records)]):
        col.metric(label, value)
    cols = st.columns(4)
    cols[0].metric("Validation passed", sum(x["verification"]["valid"] for x in scenarios))
    cols[1].metric("Validation flagged", sum(not x["verification"]["valid"] for x in scenarios))
    cols[2].metric("Models listed this session", sum(len(st.session_state.get(f"catalogue_{b}", [])) for b in LABELS))
    cols[3].metric("Backends reachable at last check", sum(st.session_state.get(f"reachable_{b}", False) for b in LABELS))
    st.caption("Activity totals cover the latest 200 saved runs. Connection indicators describe the last explicit check, not continuous monitoring.")
    if not runs:
        st.info("No scenario runs yet. Configure a runtime in Settings, refresh its models in Model Studio, then open Scenario Builder.")
    else:
        st.subheader("Recent executions")
        st.dataframe([{"created": r["created"], "run_id": r["id"], "status": r["payload"]["status"], "backends": ", ".join(r["payload"]["results"])} for r in runs[:15]], use_container_width=True)
        counts = Counter(r["risk_level"] or "requires_review" for r in risks + assessed)
        if counts:
            st.plotly_chart(px.bar(x=list(counts), y=list(counts.values()), labels={"x": "Risk classification", "y": "Assessments"}, template="plotly_dark" if st.session_state.dark else "plotly_white"), use_container_width=True)
        failures = [{"backend": r["backend"], "error": r["error"]} for r in results if r.get("error")]
        if failures:
            st.subheader("Recent errors")
            st.dataframe(failures[:10], use_container_width=True)
        comparisons = [r for r in runs if len(r["payload"]["results"]) > 1]
        if comparisons:
            st.subheader("Latest comparison")
            st.dataframe(comparison_rows(comparisons[0]["payload"]), use_container_width=True)


def scenario_builder(s):
    settings, dataset = s.settings(), s.dataset()
    saved = s.store.setting("scenario_input", ScenarioInput().model_dump())
    st.caption("Define operating conditions, inspect the shared evidence snapshot, then select the models to run.")
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
        faults_available = [r.title for r in dataset.records if r.type == "fault"]
        faults = st.multiselect("Faults represented in active KB (maximum 5)", faults_available, default=[f for f in saved["faults"] if f in faults_available], max_selections=5)
        capabilities = st.text_area("Documented robot capabilities (include source identifiers)", saved["documented_capabilities"])
        notes = st.text_area("Additional context / unsupported faults", saved["notes"])
        save = st.form_submit_button("Validate and save input", type="primary")
    if save:
        try:
            inputs = ScenarioInput(time_of_day=time_of_day, weather=weather, visibility=visibility, surface=surface,
                                   pedestrian_density=density, speed_mps=speed, location=location, navigation_mode=navigation,
                                   sensors=[v.strip() for v in sensors.split(",") if v.strip()], faults=faults, documented_capabilities=capabilities, notes=notes)
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
    if st.button("Run selected models", type="primary", disabled=not selections):
        try:
            request = RunRequest(inputs=inputs, selections=selections, retrieval_limit=settings["retrieval_limit"], relevance_threshold=settings["relevance_threshold"])
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
                    request = RunRequest(inputs=ScenarioInput(), selections=[Selection(backend=backend, model_id=selected)])
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
    if run["status"] != "running":
        st.download_button("Export comparison JSON", json_bytes(report(run)), f"comparison-{identity}.json", "application/json")
        st.download_button("Export comparison CSV", csv_bytes(comparison_rows(run)), f"comparison-{identity}.csv", "text/csv")
        with st.expander("Frozen configuration, evidence and stage timestamps"):
            st.json(run["snapshot"])


def comparison(s):
    st.caption("Each model receives the same saved input and frozen evidence. Results are measured independently; no automatic quality ranking is applied.")
    selections = choose_models(s.settings(), "comparison")
    if st.button("Run comparison", disabled=not selections, type="primary"):
        try:
            settings = s.settings()
            inputs = ScenarioInput.model_validate(s.store.setting("scenario_input", ScenarioInput().model_dump()))
            st.session_state.active_run = s.manager.start(RunRequest(inputs=inputs, selections=selections, retrieval_limit=settings["retrieval_limit"], relevance_threshold=settings["relevance_threshold"]), s.dataset(), s.matrix(), settings)
        except ValueError as exc:
            error(exc)
    if st.button("Clear displayed results"):
        st.session_state.pop("active_run", None)
    run_progress(s)


def knowledge(s):
    dataset = s.dataset()
    st.caption(f"Active: {dataset.dataset_name} · version {dataset.version} · {len(dataset.records)} records")
    upload = st.file_uploader("Upload knowledge base JSON (maximum 5 MiB)", type=["json"], key="kb_upload")
    if upload:
        try:
            candidate = parse_upload(upload.getvalue())
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
    a.download_button("Download JSON schema", json_bytes(Dataset.model_json_schema()), "knowledge-base.schema.json", "application/json")
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
    entries = {f"{b}:{scenario['scenario_id']}:{h['hazard_id']}": (scenario, h) for b, result in run["results"].items() for scenario in result.get("scenarios", []) for h in scenario["hazards"]}
    if not entries:
        st.info("This run has no parsed hazards")
        return
    key = st.selectbox("Hazard", list(entries))
    scenario, hazard = entries[key]
    st.write(hazard["description"])
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
    st.json(job)
    if job["status"] != "running":
        st.download_button("Export benchmark JSON", json_bytes(job), f"benchmark-{identity}.json")
        st.download_button("Export benchmark CSV", csv_bytes([{**{k: v for k, v in row.items() if k != "metrics"}, "metrics": json.dumps(row.get("metrics"))} for row in job["rows"]]), f"benchmark-{identity}.csv")


def evaluation_page(s):
    dataset = draft_dataset()
    st.warning("The included 24 cases are illustrative drafts, not carefully reviewed labels. Import and attribute a reviewed dataset for defensible evaluation.")
    st.download_button("Download 24-case draft dataset", json_bytes(dataset.model_dump()), "benchmark-draft-24.json")
    st.download_button("Download benchmark schema", json_bytes(BenchmarkDataset.model_json_schema()), "benchmark.schema.json")
    upload = st.file_uploader("Reviewed benchmark JSON (maximum 5 MiB)", type=["json"], key="benchmark_upload")
    valid_dataset = True
    if upload:
        try:
            if upload.size > 5 * 1024 * 1024:
                raise ValueError("Dataset exceeds 5 MiB")
            dataset = BenchmarkDataset.model_validate_json(upload.getvalue())
        except (ValueError, ValidationError) as exc:
            valid_dataset = False
            error(exc)
    count = st.number_input("Cases to execute", 1, len(dataset.cases), min(3, len(dataset.cases)))
    methods = st.multiselect("Methods", ["rules_only", "llm_only", "full_pipeline"], default=["rules_only"])
    selections = choose_models(s.settings(), "benchmark") if any(m != "rules_only" for m in methods) else []
    confirm = st.checkbox("Execute selected cases and models; cloud requests may incur account charges")
    if st.button("Run benchmark", disabled=not confirm or not valid_dataset or not methods or (any(m != "rules_only" for m in methods) and not selections)):
        subset = dataset.model_copy(update={"cases": dataset.cases[:count]})
        try:
            st.session_state.active_benchmark = s.benchmarks.start(subset, selections, methods, s.dataset(), s.matrix(), s.settings())
        except ValueError as exc:
            error(exc)
    st.caption("Coverage uses exact expected hazard IDs or cited hazard record IDs. Counts are not quality rankings. Undefined denominators yield null. Unsupported-claim rates require human annotation and remain unavailable.")
    benchmark_progress(s)
    with st.expander("Saved evaluations"):
        st.json(s.store.list("evaluation", 10))


def history(s):
    run = select_run(s, "history")
    if not run:
        return
    st.dataframe(comparison_rows(run), use_container_width=True)
    with st.expander("Complete run detail"):
        st.json(run)
    entries = {f"{b}:{scenario['scenario_id']}": scenario for b, result in run["results"].items() for scenario in result.get("scenarios", [])}
    if entries:
        key = st.selectbox("Scenario to inspect and review", list(entries))
        scenario_detail(entries[key], run["snapshot"]["evidence"])
        st.download_button("Export individual scenario", json_bytes({"scenario": entries[key], "configuration": run["snapshot"], "review_scope": key, "human_reviews": s.store.reviews(run["id"])}), f"scenario-{run['id']}.json")
        st.download_button("Export scenario CSV", csv_bytes([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in entries[key].items()}]), f"scenario-{run['id']}.csv")
        reviewer = st.text_input("Reviewer name")
        decision = st.selectbox("Engineering decision", ["accepted", "revision_requested", "rejected", "evidence_insufficient", "additional_testing"])
        notes = st.text_area("Review notes / mitigation and test link assessment")
        if st.button("Save human review", disabled=not reviewer.strip()):
            try:
                s.store.review(run["id"], key, decision, reviewer, notes)
                st.success("Human review saved separately from AI content")
            except ValueError as exc:
                error(exc)
    reviews = s.store.reviews(run["id"])
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
    st.download_button("Export comparison CSV", csv_bytes(comparison_rows(run)), f"report-{run['id']}.csv", "text/csv")
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
            selected[backend] = st.text_input("Selected model ID (revalidated on execution)", settings[backend].get("model_id", ""), key=f"configured_{backend}")
            a, b = st.columns(2)
            current = settings[backend].get("generation", {})
            params[backend] = {"temperature": a.number_input("Temperature", 0.0, 2.0, float(current.get("temperature", 0.2)), key=f"temp_{backend}"), "max_tokens": b.number_input("Maximum generated tokens", 64, 8192, int(current.get("max_tokens", 1800)), key=f"tokens_{backend}")}
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
            updated = {b: {"base_url": endpoints[b], "model_id": selected[b], "generation": params[b]} for b in LABELS}
            updated.update(model_dirs=roots, retrieval_limit=limit, relevance_threshold=threshold)
            s.store.set_setting("preferences", updated)
            for backend in LABELS:
                st.session_state.pop(f"catalogue_{backend}", None)
                st.session_state[f"reachable_{backend}"] = False
            st.success("Settings saved. Refresh catalogues to check availability.")
        except ValueError as exc:
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
    confirmed = st.checkbox("Permanently clear saved run, review, risk-assessment, simulation and evaluation history")
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
            st.success("Saved history deleted. This cannot be undone; datasets and settings retained.")


def main():
    st.set_page_config(page_title="SafeScenario AI", page_icon="🛡️", layout="wide")
    s = services()
    with st.sidebar:
        st.markdown("### SafeScenario AI")
        st.caption("AUTONOMOUS PRODUCT SAFETY")
        dark = st.toggle("Dark theme", key="dark")
        page = st.radio("Workspace", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption(f"KB {s.dataset().version} · {len(s.dataset().records)} records")
        active_backend = st.selectbox("Active backend", list(LABELS), format_func=LABELS.get)
        selected = s.settings()[active_backend].get("model_id")
        st.caption(f"Model: {selected or 'not selected'}")
        st.caption("Availability: " + ("reachable at last check" if st.session_state.get(f"reachable_{active_backend}") else "not checked / setup required"))
    theme(dark)
    st.title(page)
    st.caption("SafeScenario AI · Evidence-linked proposals, transparent risk calculations, engineering review")
    handlers = dict(zip(PAGES, [overview, scenario_builder, model_studio, comparison, knowledge, risk_analysis, simulation_page, evaluation_page, history, device, settings_page]))
    handlers[page](s)
