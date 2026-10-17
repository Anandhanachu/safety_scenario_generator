# Architecture and data provenance

`backend/schemas.py` defines strict normalized operating inputs, selections, generation parameters and AI proposal schemas. Unknown model output fields, including model-assigned risk scores or review decisions, fail validation.

`knowledge_base/core.py` enforces a versioned, bounded JSON dataset with unique IDs, typed references and content-hash versions. TF-IDF runs once for a comparison, using stable tie ordering and relationship expansion. Every model receives the same JSON input and evidence except in the explicitly labelled LLM-only benchmark method.

`backend/orchestration.py` creates an immutable configuration snapshot before work begins. The snapshot includes the entire KB, retrieved records and relevance scores, risk matrix, model IDs, generation settings, backend endpoints, retrieval settings, UTC timestamp, prompt version and SHA-256 configuration hash. Independent worker threads check model catalogues, call real adapters, parse strict output, verify references and calculate risk. Stage events and wall-clock durations are recorded from actual execution. A failed worker does not discard a peer result.

`models/adapters.py` separates Imagine's v2 protocol from llama.cpp's v1 OpenAI-compatible API and the UNO Q agent's job API. Optional operations report unsupported status. No mock inventory, automatic model download or inference fallback exists in application code.

`verification/pipeline.py` checks evidence existence/type, proposed references, operating-input consistency and duplicate IDs/narratives. A small set of unsupported assurance statements is flagged heuristically. These checks cannot establish that a source entails a claim or detect every contradiction. Human review remains necessary even for schema-valid output.

`risk_engine/calculator.py` only calculates a configured ordinal cell after human classifications, rationale, attribution and reviewed non-illustrative source/criteria evidence. AI proposals cannot supply these fields. The sample matrix always carries an illustrative label. A matrix is frozen in every run; editing policy changes future runs only.

SQLite schema v1 uses append-only snapshots, settings and separate review rows. Parameterized statements and transactional writes protect consistency. A complete snapshot reconstructs finished runs after restart. Active jobs are bounded and process-local; remote agent tasks are ephemeral. Queue saturation is explicit. Cooperative cancellation discards late results but cannot terminate a provider's underlying request. A process killed during work can lose its active run; durable active-job resumption is not implemented.

`uno_q_agent/app.py` runs only on a configured Linux host. The dashboard reports that host's identity without asserting it has independently verified the hardware model. The MPU runs inference; the optional MCU probe returns capability and ping strings. No motor/actuator interface exists.

`simulation/engine.py` is an independent deterministic software state machine. `evaluation/benchmark.py` records actual rules-only/LLM-only/full runs with explicit label matching; `reporting/exports.py` produces stable JSON and formula-escaped CSV. Human decisions, ordinal assessments, AI text, retrieved evidence, software simulation and MCU observations remain separate data products.
