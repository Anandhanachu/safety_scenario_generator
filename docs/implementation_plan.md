# Prioritized validity upgrade

1. Labelled test set: define exact hazard, mitigation, acceptance-test, risk-category and insufficient-evidence labels with reviewer attribution. Delivered 24 draft cases and a five-case sample. A domain expert must correct and approve these before reporting safety accuracy.
2. Expert-assigned severity/likelihood: store S1-S3 and L1-L3 on hazard records, with reviewed_by, explicit conflict flags and provenance. Calculate only from the frozen records and matrix; explain every lookup. Missing or conflicting inputs remain unresolved.
3. No-AI baseline: run scope filtering, top-k TF-IDF and the same matrix; evaluate on the same labels before comparing AI models. Delivered the baseline and repeatable CLI/UI evaluator.
4. Validate KB uploads: require v2 source categories and tags, unique IDs, allowed ranges and reciprocal hazard links. Preserve old/custom datasets and reject new invalid activations.
5. Constrain generation: enumerated record IDs only, canonical record text, shared temperature 0/seed 42/schema/evidence/token limit, one corrective retry and a hard stop on invalid output.
6. Add measured evaluation: 3-5 repetitions, separate precision/recall for three record types, insufficient-evidence correctness, risk correctness, measured latency, mean/stddev/range, backend/model/size/quantization metadata.
7. Surface review decisions: confidence, unresolved risk and unreviewed evidence prevent acceptance; publish baseline comparisons, full evidence and limitations in all report formats.
8. Verify real integrations: independently test each configured runtime with the shared request contract and fixed seed, record returned model identity and measured timings, and keep unsupported capabilities explicit. Credentials, hardware and expert sign-off are external prerequisites.

## Run the reproducible draft baseline

```powershell
.\.venv\Scripts\python.exe -m evaluation.labelled --repeats 3
```

The CLI prints complete JSON to stdout. It sends no model requests by default. Add --configured-models only to execute actual configured inference endpoints; cloud calls may incur charges. The Evaluation and Benchmarks UI runs the same evaluator in a background job and saves JSON/CSV results in SQLite. The Model Comparison page displays the latest labelled evaluation next to the baseline.

## Files for expert review

- data/constrained_knowledge_base.json: 10 v2 records; all illustrative and pending review. Example severity/likelihood values must be checked and attributed by a domain expert.
- data/labelled_cases_24.json: 24 draft labelled cases, including seven insufficient-evidence cases. Review expected IDs, context and risk targets independently before expert approval.
- data/sample_cases_5.json: five compact examples, including one insufficient-evidence case.
- configs/risk_matrix.example.json: illustrative ordinal matrix requiring project-policy approval.

Approve records only after checking provenance, scope tags, linkage, severity/likelihood and risk_conflict. Set reviewed_by to the actual reviewer, review_status to approved, and illustrative to false only for reviewed real project evidence. Approve labels separately with reviewed=true and an actual reviewed_by. Changing an existing run is prohibited: activate a new KB/matrix version and rerun.
