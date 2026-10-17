import io
import pytest
from pypdf import PdfReader
from reporting.exports import report
from reporting.pdf_export import pdf_bytes


def test_pdf_is_deterministic_parseable_and_paginates_long_untrusted_content():
    evidence = {"record": {"id": "QA-ONLY", "title": "Synthetic test fixture only", "description": "<tag> & untrusted evidence " * 200,
                            "type": "evidence", "review_status": "pending_review", "illustrative": True}, "score": 0.1}
    run = {"id": "PDF-QA-ONLY", "status": "fixture", "snapshot": {"evidence": [evidence]}, "results": {}}
    payload = report(run)
    output = pdf_bytes(payload)
    assert output == pdf_bytes(payload)
    reader = PdfReader(io.BytesIO(output))
    assert len(reader.pages) >= 2
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Synthetic test fixture only" in text
    assert "<tag> & untrusted evidence" in text
    assert "not certification" in text
    assert 'Out of scope / limitations' in text
    assert 'not calibrated probabilities' in text
    assert 'Baseline comparison' in text


def test_generate_pdf_qa_artifact(tmp_path):
    # Development fixture, not a model run or a real engineering report.
    from backend.configuration import ROOT
    payload = report({"id": "DEVELOPMENT-QA-ONLY", "status": "no_inference_executed", "snapshot": {
        "created": "2026-10-18", "knowledge_base_version": "illustrative-qa", "request": {"notes": "Layout verification only. No model output, measurements or safety conclusions."},
        "evidence": []}, "results": {}})
    target = ROOT / "runtime" / "pdf-qa" / "layout-verification.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pdf_bytes(payload))
    assert len(PdfReader(target).pages) >= 1
