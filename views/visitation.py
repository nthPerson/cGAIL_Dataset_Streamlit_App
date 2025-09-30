from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import HELPER_PATHS
from data_access import get_lookup, get_states_matrix, load_paths, load_visit_npz
from sidebar import SidebarSelection
from .trajectory import infer_position_dims


def _aggregate_from_paths(df: pd.DataFrame, grid_h: int, grid_w: int) -> np.ndarray:
    mat = np.zeros((grid_h, grid_w), dtype=np.int64)
    if df.empty:
        return mat
    grouped = df.groupby(["y", "x"]).size().reset_index(name="count")
    mat[grouped["y"].to_numpy(), grouped["x"].to_numpy()] = grouped["count"].to_numpy()
    return mat


def _aggregate_dynamic(
    grid_h: int,
    grid_w: int,
    expert_subset: Optional[List[str]] = None,
) -> Tuple[np.ndarray, str]:
    mat = np.zeros((grid_h, grid_w), dtype=np.int64)
    note_parts: List[str] = []
    lookup = get_lookup()
    states_matrix = get_states_matrix()

    experts_iter = expert_subset if expert_subset is not None else list(lookup.keys())
    for expert in experts_iter:
        traj_list = lookup.get(expert, [])
        xd = yd = fd = None
        for start, length in traj_list:
            if length > 0:
                arr = states_matrix[start:start + length]
                xd, yd, fd, inf_note = infer_position_dims(arr, grid_h, grid_w)
                note_parts.append(f"{expert}:{inf_note}")
                break
        if xd is None and yd is None and fd is None:
            continue
        for start, length in traj_list:
            if length == 0:
                continue
            arr = states_matrix[start:start + length]
            for v in arr:
                if xd is not None and yd is not None:
                    xx, yy = int(round(v[xd])), int(round(v[yd]))
                else:
                    rid = int(round(v[fd])) if fd is not None and fd < len(v) else -1
                    if 0 <= rid < grid_h * grid_w:
                        yy, xx = divmod(rid, grid_w)
                    else:
                        continue
                if 0 <= xx < grid_w and 0 <= yy < grid_h:
                    mat[yy, xx] += 1
    note = "; ".join(note_parts) if note_parts else "no positions decoded"
    return mat, note


@st.fragment
def visitation_panel(
    selection: SidebarSelection,
    grid_h: int,
    grid_w: int,
    x_idx: int,
    y_idx: int,
) -> None:
    scope = st.radio("Scope", ["All experts", "Selected expert"], horizontal=True, key="visit_scope")

    scale_mode = st.radio(
        "Intensity scaling",
        ["Linear", "Log (ln(1+v))", "Percentile clip"],
        horizontal=True,
        key="visit_scale",
    )
    clip_p = None
    if scale_mode == "Percentile clip":
        clip_p = st.slider("Upper percentile (clip)", 90, 100, 99, 1, key="visit_clip")

    def make_heatmap(mat: np.ndarray, title_suffix: str, note: str) -> None:
        raw = mat.astype(float)

        if scale_mode.startswith("Log"):
            mat_disp = np.log1p(raw)
            raw_ticks = np.array([0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, raw.max()]).astype(int)
            raw_ticks = raw_ticks[raw_ticks <= raw.max()]
            tickvals = np.log1p(raw_ticks)
            ticktext = [str(t) for t in raw_ticks]
            colorbar = dict(title="Visits", tickvals=tickvals, ticktext=ticktext)
            hover_tmpl = "(x=%{x}, y=%{y})<br>visits=%{customdata}<br>log1p=%{z:.2f}<extra></extra>"
            custom_data = raw
        elif scale_mode.startswith("Percentile"):
            upper = np.percentile(raw, clip_p) if raw.max() > 0 else 1
            mat_disp = np.clip(raw, 0, upper)
            colorbar = dict(title=f"Visits (≤p{clip_p} clipped)")
            hover_tmpl = "(x=%{x}, y=%{y})<br>visits=%{customdata}<extra></extra>"
            custom_data = raw
        else:
            mat_disp = raw
            colorbar = dict(title="Visits")
            hover_tmpl = "(x=%{x}, y=%{y})<br>visits=%{z}<extra></extra>"
            custom_data = None

        vmax = max(1e-9, mat_disp.max())
        colorscale = [
            [0.0, "#f0f2f5"],
            [0.00001, "#e2e6ea"],
            [1.0, "#0d4d92"],
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=mat_disp,
                x=list(range(grid_w)),
                y=list(range(grid_h)),
                colorscale=colorscale,
                zmin=0,
                zmax=vmax,
                xgap=1,
                ygap=1,
                colorbar=colorbar,
                customdata=custom_data,
                hovertemplate=hover_tmpl,
            )
        )

        target_width_px = 1100
        aspect = grid_h / grid_w
        fig_height = int(target_width_px * aspect)
        fig.update_layout(
            height=fig_height,
            margin=dict(l=0, r=0, t=60, b=0),
            title=f"State visitation counts — {title_suffix}",
            plot_bgcolor="#22262a",
            xaxis=dict(range=[-0.5, grid_w - 0.5], dtick=1, showgrid=False, zeroline=False, constrain="domain"),
            yaxis=dict(range=[-0.5, grid_h - 0.5], autorange="reversed", dtick=1, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        extra = f"{note} | scale={scale_mode}"
        if scale_mode.startswith("Percentile"):
            extra += f" (clip @ p{clip_p}={int(np.percentile(raw, clip_p))})"
        st.caption(extra)

    if scope == "All experts":
        if x_idx >= 0 and y_idx >= 0:
            grid3d = load_visit_npz(HELPER_PATHS["visit_npz"])
        else:
            grid3d = None
        if grid3d is not None:
            mat = grid3d.sum(axis=2)
            note = "precomputed overall visitation_overall.npz"
        else:
            paths_file = load_paths(HELPER_PATHS["paths_parquet"])
            if x_idx >= 0 and y_idx >= 0 and not paths_file.empty:
                mat = _aggregate_from_paths(paths_file, grid_h, grid_w)
                note = "aggregated from paths.parquet"
            else:
                with st.spinner("Decoding all expert trajectories…"):
                    mat, note = _aggregate_dynamic(grid_h, grid_w)
        make_heatmap(mat, "All experts (aggregated)", note)
    else:
        sel = selection.expert
        if x_idx >= 0 and y_idx >= 0:
            df_expert = load_paths(HELPER_PATHS["paths_parquet"], sel)
            if not df_expert.empty:
                mat = _aggregate_from_paths(df_expert, grid_h, grid_w)
                note = "paths.parquet"
            else:
                with st.spinner("Decoding selected expert trajectory positions…"):
                    mat, note = _aggregate_dynamic(grid_h, grid_w, [sel])
        else:
            with st.spinner("Decoding selected expert trajectory positions (no x/y indices)…"):
                mat, note = _aggregate_dynamic(grid_h, grid_w, [sel])
        make_heatmap(mat, f"Expert {sel}", note)


def render_visitation_section(
    selection: SidebarSelection,
    grid_h: int,
    grid_w: int,
    x_idx: int,
    y_idx: int,
) -> None:
    st.header("State visitation counts")
    visitation_panel(selection, grid_h, grid_w, x_idx, y_idx)
