# SafeScenario AI: constrained selection architecture

Streamlit and the authenticated FastAPI API share Pydantic contracts, TF-IDF retrieval, deterministic record verification and SQLite/JSON snapshots. The inference boundaries remain llama.cpp on the laptop, the UNO Q Linux agent and the Qualcomm cloud API.

## What changed and why

1. A model selects retrieved hazard, mitigation and acceptance-test IDs in a fixed JSON contract. The prompt schema enumerates the retrieved IDs. The server rejects invented IDs, new prose, model risk inputs and incorrectly linked selections. The UI renders canonical record text with citations, which removes free-form factual claims from generation.
2. V2 records require source category, reviewer attribution (nullable until reviewed), fault/condition tags and typed hazard links. Upload validation rejects duplicates, invalid ranges and broken or non-reciprocal links before activation.
3. The initial scope is five fault types and four conditions, with one fault per case. Exact tags filter the dataset before TF-IDF is fitted and ranked. V2 evidence is the top-k scored records only; relationship expansion cannot bypass the filter or top-k.
4. Expert severity and likelihood are stored on hazard records. Missing inputs, explicit conflicts, unreviewed/illustrative records or an illustrative matrix require human review. The model never supplies these inputs.
5. All models receive the same prompt, frozen evidence, schema, temperature 0, seed 42 and token limit. One corrective retry appends the validation errors. A second failure retains diagnostics and candidate metadata, requires human review and yields no accepted output. Transport/authentication errors are not retried.
6. Confidence is backed claims / all claims, with correctly linked IDs as the claim units. Zero claims yields null. Unreviewed content remains flagged even at confidence 1.0. No output is automatically accepted.
7. A no-AI baseline selects all complete retrieved hazard/mitigation/test groups and uses the same stored risk matrix. Labelled evaluation measures precision/recall separately for all three ID types and the correct abstention rate. Each case runs 3-5 times with mean, standard deviation and range.
8. JSON, CSV and PDF include evidence, model metadata, run settings, verification, risk steps, confidence, human-review flags, baseline comparison and limitations. Missing model size, quantization and measurements remain null in machine data and display as “not measured”.

## Contracts and implementation

| Concern | Implementation |
|---|---|
| KB upload models | `knowledge_base/core.py`: ConstrainedRecord, ConstrainedDataset |
| Fixed selection, run result and model metadata | `backend/constrained.py`: Choice, SelectionOutput, ModelDetails, RunResult |
| Verification, retry and canonical rendering | `backend/constrained.py`: verify_selection, execute, scenario_views |
| Expert risk lookup and explanation | `risk_engine/calculator.py`: stored_risk |
| Filtered TF-IDF with scores and timings | `knowledge_base/core.py`: retrieve_with_diagnostics |
| Labels, baseline and repeated evaluation | `evaluation/labelled.py` |
| Background jobs and immutable snapshots | `backend/orchestration.py`, `evaluation/benchmark.py` |
| Human decisions | `backend/persistence.py`; acceptance blocked until current review flags are resolved |

V1 schemas and historical verification/evaluation utilities remain readable for saved history. New inference requires a validated v2 KB; the LLM-only experiment is retired. The untouched bundled legacy catalogue is upgraded by saving a new v2 snapshot; custom datasets are preserved and require explicit v2 conversion. Earlier catalogue versions and human review rows remain stored.

The seed is requested identically from all providers; honoring it and deterministic runtime behavior have not been verified. Unsupported API parameters must fail visibly, with no silent parameter removal or backend fallback. Size and quantization may be recorded from verified metadata, and must not be inferred from a model name.

## Out of scope / limitations

Initial fault types: camera_failure, communication_loss, reduced_traction, processing_delay, lidar_failure.
Initial conditions: day_dry, night_dry, rain, reduced_visibility. Multiple simultaneous faults, other faults and wet-without-rain are outside scope and produce no eligible evidence.

Risk categories are engineering rankings, not calibrated probabilities. Confidence measures record linkage, not truth or real-world safety. Expert attribution is an audit field, not an identity-verification service; KB content and labels still require domain review.

The supplied 10-record KB and 24-case labels are illustrative drafts, with no invented expert approval. Seven cases require insufficient_evidence. The separate five-case sample includes one insufficient-evidence case. Expected risk categories are draft targets; the unreviewed sample intentionally leaves calculated risk unresolved. There is no defensible safety-accuracy result until experts approve the evidence, labels, risk inputs and matrix.

The baseline fixtures are deliberately simple and share their vocabulary with the labels; draft benchmark scores do not demonstrate generalization or added AI value. Expert evaluation should use independently reviewed cases. The final four draft cases add camera-failure contexts within the same scope categories.

Edge, cloud and laptop models differ in size and are not like-for-like comparisons. Hardware performance, API access, seed support and model compatibility remain unverified. Unconfigured backends show “not measured”. No physical test or certification claim is made.

Active jobs remain process-local and cancellation is cooperative. Durable job resumption, identity management, semantic inference beyond record selection and physical robot validation are outside scope.
