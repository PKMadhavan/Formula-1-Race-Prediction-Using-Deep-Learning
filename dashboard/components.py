"""
Reusable Streamlit UI components shared across dashboard pages.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_api_status(health: dict) -> None:
    """Top-bar showing API + model availability."""
    status = health.get("status", "unreachable")
    models = health.get("models_loaded", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = "✅" if status == "ok" else "🔴"
        st.metric("API", f"{color} {status.upper()}")
    for col, (name, loaded) in zip([col2, col3, col4], models.items()):
        with col:
            icon = "✅" if loaded else "⚠️"
            label = name.replace("_", " ").title()
            st.metric(label, f"{icon} {'Ready' if loaded else 'Not trained'}")


def lap_time_gauge(predicted_sec: float, typical_min: float = 60, typical_max: float = 120) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_sec,
        number={"suffix": " s", "valueformat": ".2f"},
        title={"text": "Predicted Lap Time"},
        gauge={
            "axis": {"range": [typical_min, typical_max]},
            "bar":  {"color": "#E8002D"},
            "steps": [
                {"range": [typical_min, typical_min + (typical_max - typical_min) * 0.33], "color": "#1e7a34"},
                {"range": [typical_min + (typical_max - typical_min) * 0.33,
                           typical_min + (typical_max - typical_min) * 0.66], "color": "#f5a623"},
                {"range": [typical_min + (typical_max - typical_min) * 0.66, typical_max], "color": "#d0021b"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def pit_probability_bar(probability: float, threshold: float) -> go.Figure:
    color = "#E8002D" if probability >= threshold else "#15151E"
    fig = go.Figure(go.Bar(
        x=["Pit Probability"],
        y=[probability * 100],
        marker_color=color,
        text=[f"{probability * 100:.1f}%"],
        textposition="outside",
    ))
    fig.add_hline(
        y=threshold * 100,
        line_dash="dash", line_color="orange",
        annotation_text=f"Threshold {threshold*100:.0f}%",
        annotation_position="top right",
    )
    fig.update_layout(
        yaxis=dict(range=[0, 110], title="Probability (%)"),
        height=300,
        margin=dict(t=30, b=20, l=20, r=20),
    )
    return fig


def position_podium(position: int) -> go.Figure:
    colors = {1: "gold", 2: "silver", 3: "#cd7f32"}
    color  = colors.get(position, "#15151E")
    fig = go.Figure(go.Indicator(
        mode="number",
        value=position,
        number={"prefix": "P", "font": {"size": 80, "color": color}},
        title={"text": "Predicted Finish Position"},
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=0))
    return fig


def lap_history_chart(lap_history: list[dict]) -> go.Figure:
    """Line chart of lap times over race laps."""
    laps  = [r["lap"] for r in lap_history]
    times = [r["lap_time_sec"] for r in lap_history]
    preds = [r.get("predicted") for r in lap_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=laps, y=times,
        mode="lines+markers", name="Actual",
        line=dict(color="#15151E", width=2),
        marker=dict(size=6),
    ))
    if any(p is not None for p in preds):
        fig.add_trace(go.Scatter(
            x=laps, y=preds,
            mode="lines+markers", name="Predicted",
            line=dict(color="#E8002D", width=2, dash="dot"),
            marker=dict(size=6),
        ))
    fig.update_layout(
        xaxis_title="Lap", yaxis_title="Lap Time (s)",
        legend=dict(orientation="h", y=1.1),
        height=320, margin=dict(t=20, b=30, l=40, r=20),
    )
    return fig
