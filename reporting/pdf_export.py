"""Portable paginated engineering report; JSON remains the exact machine-readable record."""
import io
import json
import unicodedata
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether


def printable(value):
    value = str(value).replace("\u2014", "-").replace("\u2013", "-").replace("\u00d7", "x")
    # Preserve unsupported characters as visible escapes, rather than missing-glyph squares.
    return unicodedata.normalize("NFKD", value).encode("ascii", "backslashreplace").decode("ascii")


def pdf_bytes(payload):
    stream = io.BytesIO()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportText", fontName="Helvetica", fontSize=9, leading=13, spaceAfter=6, alignment=TA_LEFT, splitLongWords=True))
    styles.add(ParagraphStyle(name="ReportLabel", fontName="Helvetica-Bold", fontSize=9, leading=13, textColor=colors.HexColor("#34445d"), spaceBefore=7))
    styles["Title"].fontSize, styles["Title"].leading = 24, 30
    styles["Title"].textColor = colors.HexColor("#17233b")
    styles["Heading2"].textColor = colors.HexColor("#4338ca")
    story = []

    def p(text, style="ReportText"):
        return Paragraph(escape(printable(text)).replace("\n", "<br/>"), styles[style])

    def field(label, value):
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2)
        else:
            text = "Unavailable" if value is None else str(value)
        story.append(p(label, "ReportLabel"))
        # Paragraph chunks prevent oversized unsplittable table cells and keep long logs flowing.
        for start in range(0, max(1, len(text)), 3000):
            story.append(p(text[start:start+3000]))

    def section(title):
        story.append(Spacer(1, 4 * mm))
        story.append(p(title, "Heading2"))

    story.extend([p("SafeScenario AI", "Title"), p("Engineering scenario review report", "Heading2"),
                  p(payload.get("notice", "AI proposals require engineering review. This report is not certification.")),
                  p("Portable PDF uses visible escapes for unsupported characters. Export JSON for exact Unicode text and complete source-dataset reconstruction.")])
    run = payload.get("run", {})
    snapshot = run.get("snapshot", {})
    field("Run identity and timestamps", {k: run.get(k) for k in ("id", "status", "finished")})
    section("1. Frozen project configuration")
    for name in ("created", "config_hash", "request", "knowledge_base_version", "prompt_version", "retrieval", "backend_endpoints", "risk_matrix", "rules_candidates"):
        field(name.replace("_", " ").title(), snapshot.get(name))
    section("2. Model comparison")
    for row in payload.get("comparison", []):
        field(f"{row['backend']} / {row['model_id']}", row)
    if not payload.get("comparison"):
        story.append(p("No completed backend results are available."))
    section("3. Frozen retrieved evidence")
    for evidence in snapshot.get("evidence", []):
        record = evidence["record"]
        story.append(KeepTogether([p(f"{record['id']} - {record['title']}", "Heading3"), p(record["description"])]))
        field("Source and review metadata", {k: v for k, v in record.items() if k not in {"title", "description"}})
        field("Retrieval", {k: v for k, v in evidence.items() if k != "record"})
    if not snapshot.get("evidence"):
        story.append(p("No relevant evidence retrieved. Evidence is insufficient."))
    section("4. Scenario proposals and verification")
    for backend, result in run.get("results", {}).items():
        field(f"Backend {backend}", {k: v for k, v in result.items() if k not in {"scenarios", "raw_response"}})
        for candidate in result.get("scenarios", []):
            story.append(p(f"{candidate['scenario_id']} - {candidate['title']}", "Heading3"))
            for name, value in candidate.items():
                if name not in {"scenario_id", "title"}:
                    field(name.replace("_", " ").title(), value)
    section("5. Human review and ordinal assessments")
    field("Human reviews (separate from AI content)", payload.get("human_reviews", []))
    field("Human risk assessments", payload.get("human_risk_assessments", []))
    section("6. Simulation and execution trace")
    field("Software simulations", payload.get("software_simulations", []))
    story.append(p("No simulation results attached means no simulated test was associated with this report. A proposed acceptance test is not an executed test. Physical robot performance and MCU-only timing are unavailable unless separately measured and recorded."))
    field("Timestamped pipeline stages", run.get("progress", {}))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#dbe2ee"))
        canvas.line(18*mm, 16*mm, 192*mm, 16*mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#52627a"))
        canvas.drawString(18*mm, 11*mm, "SafeScenario AI | Engineering review - not certification")
        canvas.drawRightString(192*mm, 11*mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(stream, pagesize=(210*mm, 297*mm), rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=18*mm, bottomMargin=22*mm, title="SafeScenario AI engineering review", invariant=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return stream.getvalue()
