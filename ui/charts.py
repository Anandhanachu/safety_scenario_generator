"""Plotly figures consume measured persisted data only; zero activity stays empty."""
import plotly.graph_objects as go
from backend.analytics import review_counts

COLORS = {"accepted": "#15a884", "rejected": "#ef6672", "pending_review": "#8b9bb5",
          "revision_requested": "#e9b44c", "evidence_insufficient": "#a78bfa", "additional_testing": "#38bdf8"}


def polish(figure, dark=False, height=330):
    figure.update_layout(template="plotly_dark" if dark else "plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", height=height, margin=dict(l=12,r=12,t=40,b=25),
                         font=dict(family="Arial", size=12), legend=dict(orientation="h", y=-0.2),
                         hovermode="closest", colorway=["#6366f1", "#22b8cf", "#15a884", "#e9b44c"])
    return figure


def review_figure(rows, dark=False):
    counts = review_counts(rows)
    labels = [d for d in COLORS if counts[d]]
    figure = go.Figure(go.Pie(labels=[s.replace("_", " ").title() for s in labels], values=[counts[s] for s in labels],
                               hole=0.72, marker_colors=[COLORS[s] for s in labels], textinfo="label+value"))
    figure.add_annotation(text=f"<b>{counts['proposed']}</b><br>proposed", showarrow=False, font_size=18)
    return polish(figure, dark)


def latency_figure(rows, dark=False, historical=False):
    figure = go.Figure()
    labels = [f"{r['backend']} / {r['model_id']}" for r in rows]
    fields = [("median_generation_ms", "Median generation") , ("median_pipeline_ms", "Median pipeline")] if historical else [("generation_latency_ms", "Generation"), ("total_pipeline_latency_ms", "Full pipeline")]
    for field, label in fields:
        figure.add_bar(name=label, x=labels, y=[r.get(field) for r in rows], hovertemplate="%{x}<br>%{y:.2f} ms<extra>%{fullData.name}</extra>")
    figure.update_layout(barmode="group", yaxis_title="Milliseconds", title="Measured latency")
    return polish(figure, dark)


def validation_figure(rows, dark=False):
    figure = go.Figure()
    labels = [f"{r['backend']} / {r['model_id']}" for r in rows]
    figure.add_bar(name="Validation passed", x=labels, y=[r.get("valid_scenarios", r.get("validation_passed", 0)) for r in rows], marker_color="#15a884")
    figure.add_bar(name="Flagged / unparsed", x=labels, y=[max(0, r.get("candidate_scenarios", r.get("proposed", 0))-r.get("valid_scenarios", r.get("validation_passed", 0))) for r in rows], marker_color="#e9b44c")
    figure.update_layout(barmode="stack", title="Proposal validation (not human approval)", yaxis_title="Scenarios")
    return polish(figure, dark)


def outcome_figure(rows, dark=False):
    figure = go.Figure()
    labels = [f"{r['backend']} / {r['model_id']}" for r in rows]
    for decision, color in COLORS.items():
        figure.add_bar(name=decision.replace("_", " ").title(), x=labels, y=[r.get(decision, 0) for r in rows], marker_color=color)
    figure.update_layout(barmode="stack", title="Latest human decisions by model", yaxis_title="Scenarios")
    return polish(figure, dark)
