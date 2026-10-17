# Testing and evaluation

Run `python -m pytest -q`. The review revision contains 138 tests. These use synthetic adapters and explicitly test-only expert approvals; no successful hardware/cloud inference is claimed.

The current label file is `data/labelled_cases_25.json`. It contains 25 preliminary, unreviewed cases: seven insufficient-evidence cases, two missing severity/likelihood cases and two conflicting-evidence cases. Risk fault fixtures copy retrieved records and remove one input or set the expert conflict flag; source KB records are not mutated. Reviewer and reviewed_date stay null until real review.

`evaluate_predictions()` measures exact risk-category match, ordinal distance (low=0, medium=1, high=2), review-required correctness and a risk/outcome confusion matrix. Missing/conflicting cases pass only with a human-review flag and no risk category. Tests cover each independently and reject unresolved risk without a review flag.

`run_experiment()` runs baseline, laptop, UNO Q and Qualcomm 3-5 times per case with shared prompts, evidence, schema and requested settings. All failures, timeouts, unconfigured backends and invalid outputs remain rows. Exact-match/review/abstention denominators include failures as incorrect. Undefined precision/ordinal distance/support rates remain not measured; eligible and measured case counts show coverage. No maximum ordinal error is invented for missing predictions.

Every statistic has unique case_count, eligible_case_count, statistic sample_count, mean, population standard deviation, minimum and maximum. Accuracy statistics average per-repetition scores; latency statistics use actual run-duration samples. Repeated executions do not create independent cases. Each metric is preliminary when its measured case count is below 30. Confusion matrices include failed decisions, raw execution counts and per-cell repeat statistics.

Adapter mock-transport tests distinguish requested settings from explicit runtime applied/unsupported reports. A successful response without settings telemetry does not establish settings were applied. UNO Q forwards its runtime report. Explicit unsupported fields become not supported; unavailable telemetry is not measured. These tests validate software behavior, not target compatibility.

citation_link_validity tests cover correct and incorrect record links. Separate expert_support_review tests cover supported, unsupported and not_reviewed records, including the reviewed-support denominator. Unsupported records stay visible for human review; no model output can set expert judgments or descriptive prose.

JSON/CSV/PDF export tests retain failures, settings, evidence, review flags, counts, uncertainty of unavailable measurements and preliminary labels. Streamlit tests cover all pages, the minimal comparison table, grouped accuracy chart, latency distribution, and archived evaluations missing new metadata. Generated PDF layout is rendered and inspected.

Earlier 181-record lexical timing results and the 24-case format are historical fixtures. They do not establish current semantic safety accuracy. Review the KB, risk policy and labels with a domain expert before making substantive accuracy claims.
