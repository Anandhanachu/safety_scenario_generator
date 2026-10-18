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
            text = "not measured" if value is None else str(value)
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
    field('Included scenarios and recalculated risk after engineer decisions', payload.get('risk_after_engineer_review', {}))
    section("6. Simulation and execution trace")
    field("Software simulations", payload.get("software_simulations", []))
    story.append(p("No simulation results attached means no simulated test was associated with this report. A proposed acceptance test is not an executed test. Physical robot performance and MCU-only timing are unavailable unless separately measured and recorded."))
    field("Timestamped pipeline stages", run.get("progress", {}))
    section('7. Baseline comparison')
    field('No-AI baseline on identical evidence', payload.get('baseline_comparison'))
    section('8. Out of scope / limitations')
    field('Limitations', payload.get('limitations', []))

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


def experiment_pdf_bytes(experiment):
    """Compact summaries plus complete per-trial audit; descriptions come only from KB data."""
    stream = io.BytesIO()
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize, styles['BodyText'].leading = 8, 11
    story = []

    def text(value, style='BodyText'):
        story.append(Paragraph(escape(printable(value)), styles[style]))

    def detail(label, value):
        from reporting.exports import display_measurements
        value = display_measurements(value)
        text(label, 'Heading3')
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        for start in range(0, max(1, len(encoded)), 1800):
            text(encoded[start:start+1800])

    text('SafeScenario AI experiment', 'Title')
    text(f"{experiment.get('label', 'preliminary')} | unique cases n={experiment.get('case_count', 'not measured')}", 'Heading2')
    text('Repeated runs do not increase the independent case count. Unavailable measurements are not measured. Runtime-reported applied settings do not independently verify hardware or model compatibility.')
    detail('Label reviewer and date', {k: experiment.get('labels', {}).get(k) for k in ('version', 'reviewer', 'reviewed_date', 'reviewed')})
    detail('Requested experiment settings', experiment.get('settings', {}))
    for summary in experiment.get('summary', []):
        text(f"{summary['backend']} / {summary['model']} | {summary['label']} | n={summary['case_count']}", 'Heading2')
        text(f"Executions {summary['executions']}; failed runs {summary['failures']}")
        for name, metric in summary.items():
            if isinstance(metric, dict) and 'mean' in metric:
                show = lambda value: 'not measured' if value is None else str(round(value, 6))
                text(f"{name}: mean {show(metric['mean'])}; SD {show(metric['stddev'])}; min-max {show(metric['min'])} to {show(metric['max'])}; n={metric['case_count']} cases ({metric['label']}); eligible cases={metric['eligible_case_count']}")
        detail(f"Risk/outcome confusion matrix | n={summary['case_count']} cases", summary.get('confusion_matrix', {}))
    text('Per-case evidence (shared by adapters and repetitions)', 'Heading2')
    seen = set()
    for row in experiment.get('rows', []):
        if row['case_id'] not in seen:
            detail(row['case_id'], row['evidence'])
            seen.add(row['case_id'])
    text('Measured run audit (failed runs retained)', 'Heading2')
    for row in experiment.get('rows', []):
        result = row.get('result') or {}
        detail(f"{row['case_id']} / {row['model']['backend']} / repeat {row['repeat']}",
               {k: row.get(k) for k in ('model', 'status', 'requested_settings', 'applied_settings', 'execution_latency_ms', 'error', 'metrics', 'expected_decision', 'predicted_decision', 'human_review_flag')})
        detail('Verification, expert support and risk calculation', {k: result.get(k, 'not measured') for k in ('verification', 'risk', 'citation_link_validity', 'reviewed_support_rate')})
    limitations = [Paragraph('Out of scope / limitations', styles['Heading2'])]
    limitations.extend(Paragraph(escape(printable(item)), styles['BodyText']) for item in experiment.get('limitations', []))
    story.append(KeepTogether(limitations))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.drawString(18*mm, 10*mm, 'SafeScenario AI | Experiment report')
        canvas.drawRightString(192*mm, 10*mm, f'Page {doc.page}')
        canvas.restoreState()

    SimpleDocTemplate(stream, pagesize=(210*mm, 297*mm), rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm, invariant=1).build(story, onFirstPage=footer, onLaterPages=footer)
    return stream.getvalue()
