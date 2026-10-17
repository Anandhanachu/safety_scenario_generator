# Verification methodology

Run `python -m pytest -q` from the repository in the installed virtual environment. Fixtures construct synthetic outputs exclusively inside `tests/`; HTTP adapters use `httpx.MockTransport`, so no automated test consumes cloud quota or claims a physical board connection.

Coverage includes valid/invalid/oversized knowledge uploads, duplicate keys and IDs, typed/broken references, lexical retrieval and empty evidence, strict input/output schemas, missing and unapproved risk evidence, deterministic risk cells, missing severity/likelihood, model-assigned risk rejection, input conflicts, duplicates, invalid mitigation/test links, persistence restoration, scoped separate human reviews, deterministic exports and formula escaping, simulation pass/fail/inconclusive outcomes, authentication failures, rate limits, malformed catalogues, local/board disconnections, unavailable models, timeouts, response model consistency and cancellation. Integration tests preserve successful peer results during a failed comparison and verify LLM-only prompt differences. Streamlit AppTest opens all eleven pages and toggles the theme.

Application startup is checked with actual local HTTP health responses and a browser inspection of the empty dashboard. These checks prove local startup and surrounding pipeline behavior, not real model compatibility, live provider success, robot safety or firmware correctness. See `docs/status.md` for recorded results and remaining external tests.

## Evaluation methods

The included 24 cases cover 2 times of day × 3 weather states × 4 faults. They are **draft illustrative labels**, not reviewed ground truth. Download the JSON, review expected hazard IDs, acceptable evidence, required mitigations and expected review outcomes, attribute the reviewer and import the reviewed dataset. Exact ID matching is deliberately transparent: a generated hazard is counted as relevant if its `hazard_id` or a cited hazard-record ID matches an expected ID. This can miss paraphrases or be gamed by irrelevant citation; it is not a semantic accuracy score.

- Rules-only: enumerate reviewed non-illustrative hazard records from the same deterministic retrieval. This baseline does not generate narratives and has no comparable LLM/schema timing metrics. With the bundled unreviewed sample, this legitimately yields no reviewed candidates.
- LLM-only: send identical operating conditions and output schema with an empty evidence context. Keep a frozen reference snapshot for downstream checks; the method difference is recorded.
- Full pipeline: send the shared retrieved evidence and run the same verification/risk pipeline.

Each inference-backed evaluation row links its immutable run containing the full configuration, model IDs, versions, timings and errors. Rules rows record the retrieved candidate IDs and KB version. Dataset version, review status and timestamps are retained at the benchmark level. Failures preserve partial rows.

## Metric definitions

| Metric | Numerator / denominator |
| --- | --- |
| Expected hazard coverage | Distinct expected IDs matched / distinct expected IDs |
| Relevant hazard count | Count of generated hazards with an exact expected-ID match (count only) |
| Valid evidence-ID rate | References resolving to allowed evidence types in frozen snapshot / all generated evidence references |
| Acceptable evidence rate | References on the case's acceptable list / all generated evidence references |
| Required mitigation coverage | Distinct required mitigations returned / distinct required mitigations |
| Valid mitigation-reference rate | References resolving to mitigation records / all mitigation references |
| Valid acceptance-test-reference rate | References resolving to test records / all test references |
| Schema-valid output rate | Valid structured outputs / attempted output for each row |
| Duplicate scenario rate | Near-duplicate scenarios flagged / parsed scenarios |
| Backend failure rate | Failed executions / attempted execution for each row |
| Generation/pipeline latency | Actual measured duration in milliseconds, not an accuracy metric |

Every defined rate includes numerator, denominator and value; an empty denominator yields null. The automatic unsupported-assurance heuristics produce flags, **not** an unsupported-claim rate. A defensible rate requires human claim-level annotations; annotation tooling and rate calculation are not implemented. Human review outcomes remain separate; expected-versus-observed review agreement is not automatically scored. No model ranking is based on speed or quantity alone.
