from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def action_space_18() -> go.Figure:
    coords = [(dx, dy) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]

    def dir_label(dx: int, dy: int) -> str:
        if dx == 0 and dy == 0:
            return "stay"
        parts = []
        if dy == -1:
            parts.append("upper")
        if dy == 1:
            parts.append("lower")
        if dx == -1:
            parts.append("left")
        if dx == 1:
            parts.append("right")
        if len(parts) == 2:
            return f"{parts[0]}-{parts[1]}"
        return parts[0]

    base_x = [dx + 1 for dx, _ in coords]
    base_y = [(-dy) + 1 for _, dy in coords]

    xs_now = base_x
    ys_now = base_y

    scale_outer = 1.32
    xs_next, ys_next = [], []
    for cx, cy, (dx, dy) in zip(base_x, base_y, coords):
        if dx == 0 and dy == 0:
            xs_next.append(cx + 0.28)
            ys_next.append(cy)
            continue
        vx = cx - 1
        vy = cy - 1
        xs_next.append(1 + vx * scale_outer)
        ys_next.append(1 + vy * scale_outer)

    hover_now, hover_next = [], []
    for dx, dy in coords:
        label = dir_label(dx, dy)
        if label == "stay":
            hover_now.append("Stay at current region (current time window)")
            hover_next.append("Stay at current region (next time window)")
        else:
            hover_now.append(f"Move to {label} neighbor (current time window)")
            hover_next.append(f"Move to {label} neighbor (next time window)")

    fig = go.Figure()
    heat = go.Heatmap(
        z=np.zeros((3, 3)),
        x=[0, 1, 2],
        y=[0, 1, 2],
        showscale=False,
        zmin=0,
        zmax=1,
        colorscale=[[0, "#ffffff"], [1, "#ffffff"]],
        xgap=2,
        ygap=2,
        hoverinfo="skip",
    )
    fig.add_trace(heat)

    fig.add_trace(
        go.Scatter(
            x=xs_now,
            y=ys_now,
            mode="markers+text",
            marker=dict(size=20, color="#1f77b4", line=dict(color="#0f3d66", width=1)),
            text=["S" if (dx == 0 and dy == 0) else "" for dx, dy in coords],
            textposition="middle center",
            name="Current window",
            hovertext=hover_now,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xs_next,
            y=ys_next,
            mode="markers",
            marker=dict(size=16, symbol="circle-open", line=dict(color="#1f77b4", width=2)),
            name="Next window",
            hovertext=hover_next,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )

    for (dx, dy), x_cell, y_cell in zip(coords, base_x, base_y):
        if dx == 0 and dy == 0:
            continue
        fig.add_annotation(
            x=x_cell,
            y=y_cell,
            ax=1,
            ay=1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=0,
            arrowsize=2.0,
            arrowwidth=2.0,
            arrowcolor="#1f77b4",
            opacity=0.9,
        )

    fig.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=50, b=5),
        title="18 actions: 3×3 local grid (filled = current window, hollow = advance time)",
        plot_bgcolor="#22262a",
        xaxis=dict(
            range=[-0.5, 2.5],
            dtick=1,
            showgrid=False,
            zeroline=False,
            constrain="domain",
            showticklabels=False,
            ticks="",
        ),
        yaxis=dict(
            range=[-0.5, 2.5],
            dtick=1,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            autorange="reversed",
            showticklabels=False,
            ticks="",
        ),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.add_annotation(
        x=1,
        y=3.05,
        text="Filled = current time window · Hollow = advance time · Cells align with other grid visuals",
        showarrow=False,
        font=dict(size=12, color="#bbbbbb"),
    )
    return fig


def render_action_space_section() -> None:
    st.header("Action space (semantic reference)")
    st.plotly_chart(action_space_18(), use_container_width=True)
    st.caption("Hover any marker for semantics. Center S = stay; neighbors = move to adjacent region. Each has both 'now' and 'next window' variants (18 total).")
