# SafeScenario AI: scenario generation and KB screening

## Current interactive workflow

Scenario Builder, Model Comparison and Model Studio inference tests now use AI-written scenario batches by default. The UI requests as many distinct supported scenarios as possible up to a configurable cap of 1-20 per model (default 10). Available evidence and token budget determine the actual count; no promise of maximum coverage is made. Models compose written scenarios as a list of cited statements with a display title. Scenario context is attached from the validated user input. Risk values remain entirely server-calculated.

`backend/scenario_generation.py` contains `CitedStatement`, `ScenarioDraft`, `ScenarioBatch`, batch generation, independent scenario screening and review-based recalculation. Automatic acceptance requires all cited descriptions to match retrieved KB descriptions exactly, complete hazard/mitigation/test coverage, valid reciprocal links, supported scope, and no explicitly rejected/unsupported evidence. Unverified paraphrases, extra claims, invented IDs, invalid individual schemas and duplicate record combinations are retained as rejected drafts. AI titles are display labels, not evidence. This conservative check does not prove semantic support for arbitrary new prose or real-world validity of a KB record. Pending/illustrative records retain review flags and cannot support approved risk inputs.

A well-formed batch keeps every draft, even if some are rejected. Invalid batch JSON/envelopes receive one corrective retry; transport failures are recorded without automatic retries. Original responses have hashes/byte counts and per-attempt settings. Runtime failures remain visible.

System-accepted drafts initially contribute to a model's risk analysis, separately from engineer approval. A named engineer can reject a draft or request follow-up in Scenario Register; that excludes it from the effective calculation. The latest scoped decision controls inclusion. A subsequent human acceptance can restore a previously excluded scenario only when all existing acceptance checks pass. System-rejected drafts cannot be overridden as accepted. Reviews stay append-only and original run snapshots are immutable.

`recalculate()` reads the run's frozen evidence/matrix and current decisions. It recomputes included/excluded counts, unique hazards, per-hazard risk, worst included risk category/score, citation_link_validity and reviewed_support_rate. Risk ranks are never averaged or treated as probabilities. Missing/conflicting/unreviewed inputs leave the overall risk unresolved. Excluding the last scenario gives `no_included_scenarios` and null risk, never low risk. Exclusion changes analysis coverage; it does not demonstrate reduced physical risk.

Scenario Register, execution results, Risk Analysis, and JSON/CSV/PDF engineering reports expose the effective calculation. Reports retain original proposals, system decisions, rejection reasons and human audit history alongside recalculated results. The earlier record-selection mode and immutable historical runs remain supported. REST requests default to `record_selection` for compatibility; request `generation_mode: scenario_drafts` and `max_scenarios` to use the new path.

The 25-case repeated evaluator remains an evaluation of **record selection**, as its labels name record IDs rather than scenario prose. Its metrics must not be presented as measured draft-writing accuracy. Real model compatibility and new generation latency must be established with actual runs.

## Earlier record-selection workflow (still available)

Streamlit and the authenticated FastAPI API share Pydantic contracts, TF-IDF retrieval, deterministic record verification and SQLite/JSON snapshots. The inference boundaries remain llama.cpp on the laptop, the UNO Q Linux agent and the Qualcomm cloud API.

## What changed and why

1. A model selects retrieved hazard, mitigation and acceptance-test IDs in a fixed JSON contract. The prompt schema enumerates the retrieved IDs. The server rejects invented IDs, new prose, model risk inputs and incorrectly linked selections. The UI renders canonical record text with citations, which removes free-form factual claims from generation.
2. V2 records require source category, reviewer attribution (nullable until reviewed), fault/condition tags and typed hazard links. Upload validation rejects duplicates, invalid ranges and broken or non-reciprocal links before activation.
3. The initial scope is five fault types and four conditions, with one fault per case. Exact tags filter the dataset before TF-IDF is fitted and ranked. V2 evidence is the top-k scored records only; relationship expansion cannot bypass the filter or top-k.
4. Expert severity and likelihood are stored on hazard records. Missing inputs, explicit conflicts, unreviewed/illustrative records or an illustrative matrix require human review. The model never supplies these inputs.
5. All models receive the same prompt, frozen evidence, schema, temperature 0, seed 42 and token limit. One corrective retry appends the validation errors. A second failure retains diagnostics and candidate metadata, requires human review and yields no accepted output. Transport/authentication errors are not retried.
6. citation_link_validity is correctly linked record IDs / selected record IDs. It measures existence, type and linkage only. Optional expert_support_review on KB records is supported / unsupported / not_reviewed; reviewed_support_rate is supported / (supported + unsupported). Unreviewed claims are excluded from that denominator. Unsupported claims remain visible and eligible for human review. Models supply neither metric nor expert judgments. The app fills all descriptions from records.
7. run_experiment runs every case 3-5 times on baseline, laptop, UNO Q and Qualcomm using shared evidence/settings. Unconfigured adapters, exceptions, validation failures and timeouts remain failed rows. evaluate_predictions computes exact risk-category match, ordinal distance (low=0, medium=1, high=2), an outcome/risk confusion matrix and explicit review-required correctness. Missing/conflicting inputs must produce a review flag and no risk category. Precision/recall and abstention correctness remain separate.
8. JSON, CSV and PDF include evidence, model metadata, run settings, verification, risk steps, citation_link_validity, human-review flags, baseline comparison and limitations. Missing model size, quantization and measurements remain null in machine data and display as “not measured”.

## Contracts and implementation

| Concern | Implementation |
|---|---|
| KB upload models | `knowledge_base/core.py`: ConstrainedRecord, ConstrainedDataset |
| Fixed selection, run result and model metadata | `backend/constrained.py`: Choice, SelectionOutput, ModelDetails, RunResult |
| Verification, retry and canonical rendering | `backend/constrained.py`: verify_selection, execute, scenario_views |
| Rule-based risk lookup and explanation | `risk_engine/calculator.py`: stored_risk calculates from KB severity and likelihood before scenario review; pending inputs are labelled provisional |
| Filtered TF-IDF with scores and timings | `knowledge_base/core.py`: retrieve_with_diagnostics |
| Labels, baseline and repeated evaluation | `evaluation/labelled.py` |
| Background jobs and immutable snapshots | `backend/orchestration.py`, `evaluation/benchmark.py` |
| Human decisions | `backend/persistence.py`; acceptance blocked until current review flags are resolved |

V1 schemas and historical verification/evaluation utilities remain readable for saved history. New inference requires a validated v2 KB; the LLM-only experiment is retired. The untouched bundled legacy catalogue is upgraded by saving a new v2 snapshot; custom datasets are preserved and require explicit v2 conversion. Earlier catalogue versions and human review rows remain stored.

The seed is requested identically from all providers; honoring it and deterministic runtime behavior have not been verified. Unsupported API parameters must fail visibly, with no silent parameter removal or backend fallback. Size and quantization may be recorded from verified metadata, and must not be inferred from a model name.

## Out of scope / limitations

Demo fault types: camera_failure, communication_loss, reduced_traction, processing_delay, lidar_failure, ultrasonic_failure, motor_control_failure, battery_fault.
Initial conditions: day_dry, night_dry, rain, reduced_visibility. One to three of the eight supported faults can be selected; other faults and undocumented fault interactions remain outside scope. Wet input conservatively uses the rain evidence bucket.

Risk categories are engineering rankings, not calibrated probabilities. Citation link validity measures record linkage, not truth or real-world safety. Expert attribution is an audit field, not an identity-verification service; KB content and labels still require domain review.

The current label file is data/labelled_cases_25.json: 25 preliminary draft cases, including seven insufficient-evidence cases, two missing-severity/likelihood cases and two conflicting-evidence cases. The latter four are explicit synthetic risk-input fixtures: copies of the retrieved hazard records have one input removed or the expert conflict flag set. Original KB records are unchanged. Reviewer and reviewed_date remain null until actual expert review; no expert approval is invented. The earlier 24-case file is retained as historical data. The separate five-case sample remains downloadable.

Metrics include unique case counts, eligible/measured counts and preliminary labels when n < 30. Three repetitions of 25 cases still mean n=25, not n=75. Scores are aggregated per repetition, with mean/population-SD/min/max. Citation/support fractions are weighted by claim counts. Risk exact-match includes failed predictions as incorrect; ordinal distance requires two categories and reports missing coverage explicitly. Confusion matrices include failed predictions and provide per-cell repetition statistics. Descriptive accuracy counts use exact record IDs, not semantic equivalence.

Each adapter returns requested_settings separately from applied_settings. Only explicit runtime applied_settings / unsupported_settings reports populate the latter; otherwise values are not measured. Unsupported settings are not supported. The UNO Q agent forwards the runtime report. A requested seed does not establish it was applied; no silent fallback removes parameters. Run-duration distributions include measured failure/preflight time and are labelled accordingly, separately from successful generation timings.

Edge, cloud and laptop models differ in size and are not like-for-like comparisons. Hardware performance, API access, seed support and model compatibility remain unverified. Unconfigured backends show “not measured”. No physical test or certification claim is made.

Active jobs remain process-local and cancellation is cooperative. Durable job resumption, identity management, semantic inference beyond record selection and physical robot validation are outside scope.
