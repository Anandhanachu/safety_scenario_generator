# Knowledge-base format and activation

Download the complete Pydantic-generated JSON schema from Knowledge Base. The canonical schema lives in `knowledge_base/core.py`; a valid sample is in `data/sample_knowledge_base.json` and is downloadable in the UI.

The root requires `schema_version: "1.0"`, `dataset_id`, `dataset_name` and 1–5000 records. Unknown fields are rejected. Files are limited to 5 MiB. Duplicate JSON keys and duplicate record IDs are rejected. Invalid imports never replace active data.

Each record requires `id` (1–100 ASCII letters/digits/underscore/dot/hyphen, beginning alphanumeric), `type`, nonempty `title` (up to 200 characters) and `description` (up to 12000). Supported types: `operating_condition`, `fault`, `hazard`, `consequence`, `evidence`, `source`, `mitigation`, `acceptance_test`, `risk_criteria`.

Optional fields:

| Field | Meaning |
| --- | --- |
| `source_ids` | Existing source/evidence IDs only |
| `mitigation_ids` | Existing mitigation IDs only |
| `test_ids` | Existing acceptance-test IDs only |
| `related_ids` | Any existing record IDs |
| `source` | Source citation/metadata text, up to 2000 characters; never fetched/executed automatically |
| `review_status` | `pending_review` (default), `approved`, `rejected` |
| `illustrative` | Defaults to true; false should only describe actual project/source evidence |

All reference lists are bounded to 100 entries. `approved` in an uploaded record is a declaration by its author, not an independently authenticated approval. Uploaded classifications must be reviewed through your engineering data-governance process; the application does not certify provenance.

Upload, inspect preview and counts, explicitly confirm activation. An edit/add/delete is a complete JSON dataset edit with validation before activation. Broken references must be corrected in the same dataset change. Each activation creates a SQLite snapshot with timestamp and content-derived version; restore from the version-history expander. Runs retain the entire source dataset and evidence snapshot, so later KB changes cannot silently alter an old report.

TF-IDF indexes title and description with English stop words. It returns only positive similarity scores meeting the threshold, with deterministic tie ordering and configurable top-k. Direct relationships from the initially retrieved records are added to the evidence snapshot, up to 100 total records, with a relationship label instead of an invented relevance score. This is lexical retrieval, not semantic proof. An empty match reports insufficient evidence; it does not create source records.

The bundled records, source assumption, mitigations and tests are illustrative and unreviewed. Replace them with actual reviewed project records before using the tool to inform real engineering decisions.
