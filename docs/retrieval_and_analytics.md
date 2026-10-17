# Retrieval and proposal analytics upgrade

Historical note: this page describes the earlier 181-record catalogue and lexical measurements. Current constrained v2 generation filters tags before ranking and uses the 10-record draft starter. See architecture.md and implementation_plan.md for current behavior; these historical timings are not current v2 measurements.

Implemented 18 October 2026.

## Expanded engineering catalogue

`data/expanded_knowledge_base.json` contains 181 structurally validated records across all nine supported categories. It preserves the nine original starter records and adds 24 fault-topic bundles, two contextual sources and two risk-review checklists. Topics include lidar, proximity sensors, sensor disagreement, glare, night/rain degradation, localization/GNSS, mapping/navigation, crossing boundaries, curbs, braking/steering, encoders, power, thermal/timestamp faults, network/replay faults, protective signals, payload retention, crowds and ice.

Every added engineering record remains illustrative and pending review. The source records point to [NHTSA automated vehicle safety](https://www.nhtsa.gov/vehicle-safety/automated-vehicle-safety) and [OSHA robotics](https://www.osha.gov/robotics) as contextual reading only; neither is represented as approval of these hypotheses or proof that a mitigation works on a sidewalk robot.

Fresh installations use the expanded catalogue. An existing untouched nine-record starter upgrades automatically, preserving the previous snapshot. Custom/edited KBs stay active. Download or explicitly activate the bundled expansion from Knowledge Base when using custom data.

## TF-IDF performance and relevance

The previous search refit a vectorizer on every query. Retrieval v2 caches up to four immutable indexes by a full SHA-256 content key. Updated datasets get a new index; callers receive copies so one run cannot mutate another's evidence. Search uses sparse float32 word/unigram-bigram TF-IDF, sublinear term frequency, cosine similarity, deterministic tie ordering and the existing positive-score threshold/relationship expansion. Structured input queries search values rather than noisy JSON field names.

The search inspector displays direct similarity scores, relationships, cold-build time, warm/cold cache state and measured query/total durations. Historical run snapshots record the exact retrieval version and diagnostics. An empty vocabulary or no matching terms yields no evidence, never invented results.

Reproduce measured checks against the active expanded KB:

```powershell
.\.venv\Scripts\python.exe -m knowledge_base.benchmark --repeats 5 --scale --save
```

The UI can also run the 16-query lexical regression suite. Only direct top-five matches contribute to metrics; relationship expansion cannot inflate them. Labels are curated document-relevance expectations, not independently reviewed semantic or safety ground truth. Custom datasets without matching labels show a clear unsupported-benchmark message.

Observed on the development laptop (Windows / Python 3.10.10; 181 records, 16 queries, 80 warm timing samples):

| Metric | Observed result |
| --- | --- |
| At least one expected record in top five | 16/16 (100%) |
| Mean labelled recall at five | 85.42% |
| Mean reciprocal rank | 0.96875 |
| Warm median total retrieval | 1.463 ms |
| Warm p95 total retrieval | 2.077 ms |
| Initial sparse index build | 7.002 ms |

Separate **synthetic index-load** measurements (not engineering records or relevance results; ten warm samples per size):

| Synthetic record count | First request | Warm median total | Warm p95 total |
| --- | --- | --- | --- |
| 181 | 9.437 ms | 1.277 ms | 1.743 ms |
| 1,000 | 36.809 ms | 5.152 ms | 35.176 ms |
| 5,000 | 217.287 ms | 31.489 ms | 99.125 ms |

Totals include dataset serialization/content hashing, scoring, sorting, copying and relationship expansion. Timings vary with hardware/load; these are observations, not an SLA. Synthetic workloads are never activated as knowledge. Saved benchmark JSON contains individual query results and measurement provenance.

## All proposals and human outcomes

Scenario Register scans all saved runs, without the former 200-run counting limit. Review scope includes run, backend and proposal; duplicate model-assigned scenario IDs receive distinct stable scopes. For each proposal, only the latest human decision contributes to current counts, while all prior decisions stay in the audit trail.

Proposed = accepted + rejected + pending review + revision requested + evidence insufficient + additional testing. The overview groups the latter three as Needs follow-up. Automated schema/evidence validation never constitutes a human accept/reject decision. Schema-invalid but identifiable AI candidates are listed as pending until a reviewer rejects or requests revision; they cannot be accepted before valid regeneration. Their metadata and diagnostics are retained even with raw-response retention disabled. Unreadable JSON has no trustworthy candidate count. The original proposal content remains separate from human decisions.

Filter by decision/backend/text, inspect narratives, save review decisions, and export all or filtered proposals. Reports and History now shows the current human status rather than permanently displaying pending review. JSON engineering reports include a proposal review summary.

## Measured model charts

Model Comparison displays grouped generation/pipeline latency bars, stacked passed/flagged proposal counts and accepted/rejected/pending/follow-up outcomes. A selected saved run provides matched inputs/evidence/configuration. Historical charts group by exact backend and model ID; successful-execution latency medians exclude failed/cancelled work, and failure/cancellation counts remain visible in the data table. Historical workloads may differ, so aggregate speed is not a fair model-quality ranking. All outcomes use current human decisions and all saved history. Empty history shows a descriptive empty state instead of fabricated model measurements.
