from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data_access import get_trajectory
from sidebar import SidebarSelection


def infer_position_dims(
    traj: List[List[float]],
    grid_h: int,
    grid_w: int,
    sample_cap: int = 400,
) -> Tuple[Optional[int], Optional[int], Optional[int], str]:
    L = min(len(traj), sample_cap)
    arr = np.asarray(traj[:L], dtype=float)
    if arr.size == 0:
        return None, None, None, "no data"
    D = arr.shape[1]

    def int_like(values: np.ndarray) -> bool:
        return np.allclose(values, np.round(values), atol=1e-6)

    cand_x, cand_y = [], []
    for d in range(D):
        col = arr[:, d]
        if int_like(col):
            if np.all((col >= 0) & (col < grid_w)) and (col.max() - col.min() >= 2):
                cand_x.append(d)
            if np.all((col >= 0) & (col < grid_h)) and (col.max() - col.min() >= 2):
                cand_y.append(d)

    for xd in cand_x:
        for yd in cand_y:
            if xd != yd:
                return xd, yd, None, f"inferred separate x={xd}, y={yd}"

    for d in range(D):
        col = arr[:, d]
        if int_like(col) and np.all((col >= 0) & (col < grid_h * grid_w)) and (col.max() - col.min() >= grid_w):
            return None, None, d, f"inferred flat region id dim={d}"

    return None, None, 0, "fallback: used first feature (may be incorrect)"


@st.fragment
def draw_path_for_trajectory(
    selection: SidebarSelection,
    grid_h: int,
    grid_w: int,
    x_idx: int,
    y_idx: int,
) -> None:
    traj_arr = get_trajectory(selection.expert, selection.traj_idx)
    if traj_arr.size == 0:
        st.info("Trajectory not available or empty.")
        return

    traj = traj_arr.tolist()
    if x_idx >= 0 and y_idx >= 0:
        xd, yd, fd, decode_note = x_idx, y_idx, None, f"explicit x={x_idx}, y={y_idx}"
    else:
        xd, yd, fd, decode_note = infer_position_dims(traj, grid_h, grid_w)

    xs, ys = [], []
    for state in traj:
        v = np.asarray(state, dtype=float)
        if xd is not None and yd is not None:
            xx, yy = int(round(v[xd])), int(round(v[yd]))
        else:
            rid = int(round(v[fd])) if fd is not None and fd < len(v) else -1
            if 0 <= rid < grid_h * grid_w:
                yy, xx = divmod(rid, grid_w)
            else:
                continue
        if 0 <= xx < grid_w and 0 <= yy < grid_h:
            xs.append(xx)
            ys.append(yy)

    if not xs:
        st.info("Could not decode any (x,y) positions for this trajectory.")
        return

    visits = np.zeros((grid_h, grid_w), dtype=int)
    for yy, xx in zip(ys, xs):
        visits[yy, xx] += 1

    vmax = max(1, visits.max())
    colorscale = [
        [0.0, "#f0f2f5"],
        [0.00001, "#e2e6ea"],
        [1.0, "#0d4d92"],
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=visits,
            x=list(range(grid_w)),
            y=list(range(grid_h)),
            colorscale=colorscale,
            zmin=0,
            zmax=vmax,
            xgap=1,
            ygap=1,
            colorbar=dict(title="Visits"),
            hovertemplate="(x=%{x}, y=%{y})<br>visits=%{z}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line=dict(width=2, color="#26456e"),
            marker=dict(size=6, color="#17324f"),
            name="Path",
            hovertemplate="Step %{pointNumber}<br>(x=%{x}, y=%{y})<extra></extra>",
        )
    )

    if 0 <= selection.state_idx < len(xs):
        fig.add_trace(
            go.Scatter(
                x=[xs[selection.state_idx]],
                y=[ys[selection.state_idx]],
                mode="markers",
                marker=dict(size=13, color="#ff4136", line=dict(width=2, color="white")),
                name="Current state",
                hovertemplate=f"Current state idx={selection.state_idx}<br>(x=%{{x}}, y=%{{y}})<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[xs[0]],
            y=[ys[0]],
            mode="markers+text",
            text=["Start"],
            textposition="top center",
            marker=dict(size=11, color="#2ecc40"),
            name="Start",
            hovertemplate="Start<br>(x=%{x}, y=%{y})<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[xs[-1]],
            y=[ys[-1]],
            mode="markers+text",
            text=["End"],
            textposition="top center",
            marker=dict(size=11, color="#ff851b"),
            name="End",
            hovertemplate="End<br>(x=%{x}, y=%{y})<extra></extra>",
        )
    )

    for i in range(min(len(xs) - 1, 200)):
        fig.add_annotation(
            x=xs[i + 1],
            y=ys[i + 1],
            ax=xs[i],
            ay=ys[i],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=4,
            arrowsize=1.8,
            arrowwidth=1.5,
            opacity=0.85,
            arrowcolor="#17324f",
        )

    target_width_px = 1100
    aspect = grid_h / grid_w
    fig_height = int(target_width_px * aspect)

    fig.update_layout(
        height=fig_height,
        margin=dict(l=0, r=0, t=70, b=0),
        title=f"Expert {selection.expert} — Trajectory {selection.traj_idx} (len={len(xs)})",
        plot_bgcolor="#22262a",
        xaxis=dict(
            title="x (col)",
            range=[-0.5, grid_w - 0.5],
            dtick=1,
            showgrid=False,
            zeroline=False,
            constrain="domain",
        ),
        yaxis=dict(
            title="y (row)",
            range=[-0.5, grid_h - 0.5],
            autorange="reversed",
            dtick=1,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(orientation="h", y=-0.08, x=0, bgcolor="rgba(0,0,0,0)"),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Visited regions (heatmap) during trajectory. Green=start, Orange=end, Red=current.")


def render_trajectory_section(
    selection: SidebarSelection,
    grid_h: int,
    grid_w: int,
    x_idx: int,
    y_idx: int,
) -> None:
    st.header(f"Trajectory path on grid ({grid_h}×{grid_w})")
    draw_path_for_trajectory(selection, grid_h, grid_w, x_idx, y_idx)
