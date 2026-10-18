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


def unified_benchmark_figure(rows, dark=False):
    """Compare all 3 inference modes (Cloud, Uno Q, Local) in a single unified graph."""
    from plotly.subplots import make_subplots
    import json

    BACKEND_COLORS = {
        "qualcomm": "#6366f1",
        "laptop_local": "#10b981",
        "uno_q": "#f97316",
        "baseline": "#94a3b8"
    }
    BACKEND_LABELS = {
        "qualcomm": "Cloud (Qualcomm)",
        "laptop_local": "Local (Laptop)",
        "uno_q": "Uno Q (Arduino)",
        "baseline": "Baseline (TF-IDF)"
    }

    # Ensure backends are ordered consistently
    order = ["qualcomm", "uno_q", "laptop_local", "baseline"]
    sorted_rows = sorted(rows, key=lambda r: order.index(r.get("backend", "baseline")) if r.get("backend") in order else 99)

    modes = []
    latencies_sec = []
    risk_scores_100 = []
    valid_counts = []
    bar_colors = []

    for r in sorted_rows:
        b = r.get("backend", "unknown")
        modes.append(BACKEND_LABELS.get(b, b))
        bar_colors.append(BACKEND_COLORS.get(b, "#6366f1"))

        # Latency in seconds
        lat = r.get("generation_latency_ms") or r.get("total_pipeline_latency_ms") or 0
        latencies_sec.append(round(lat / 1000.0, 2))

        # Risk score out of 100 (assuming matrix max 9, scaled to 100)
        eff_risk_raw = r.get("effective_risk_after_review")
        eff_risk = {}
        if eff_risk_raw:
            try:
                eff_risk = json.loads(eff_risk_raw) if isinstance(eff_risk_raw, str) else eff_risk_raw
            except Exception:
                eff_risk = {}
        sc = eff_risk.get("risk_score") if isinstance(eff_risk, dict) else None
        if sc is not None:
            risk_scores_100.append(min(100, round((float(sc) / 9.0) * 100)))
        else:
            risk_scores_100.append(0)

        # Valid scenarios
        valid_counts.append(r.get("valid_scenarios", 0))

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("⚡ Latency (Seconds)", "🛡️ Safety Risk Index (0-100)", "✅ Valid Scenarios"),
        horizontal_spacing=0.08
    )

    # 1. Latency Bar
    fig.add_trace(
        go.Bar(
            x=modes, y=latencies_sec,
            marker_color=bar_colors,
            name="Latency (s)",
            text=[f"{v}s" for v in latencies_sec],
            textposition="auto",
            showlegend=False
        ),
        row=1, col=1
    )

    # 2. Risk Score out of 100
    fig.add_trace(
        go.Bar(
            x=modes, y=risk_scores_100,
            marker_color=bar_colors,
            name="Risk / 100",
            text=[f"{v}/100" if v > 0 else "N/A" for v in risk_scores_100],
            textposition="auto",
            showlegend=False
        ),
        row=1, col=2
    )

    # 3. Valid Scenarios
    fig.add_trace(
        go.Bar(
            x=modes, y=valid_counts,
            marker_color=bar_colors,
            name="Valid Scenarios",
            text=[str(v) for v in valid_counts],
            textposition="auto",
            showlegend=False
        ),
        row=1, col=3
    )

    fig.update_yaxes(title_text="Seconds", row=1, col=1)
    fig.update_yaxes(title_text="Risk Score (/100)", range=[0, 105], row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=3)

    fig.update_layout(
        title="<b>Unified 3-Mode Performance Benchmark</b> (Cloud vs Uno Q vs Local)",
        title_font_size=15,
        height=320,
        margin=dict(l=20, r=20, t=50, b=30),
        font=dict(family="Inter, system-ui, sans-serif", size=11)
    )

    return polish(fig, dark, height=330)

