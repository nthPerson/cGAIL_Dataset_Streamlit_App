from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    HELPER_PATHS,
    TRAFFIC_COUNT_DEFAULT,
    TRAFFIC_FEATURES,
    TRAFFIC_NEIGHBORHOOD,
)
from data_access import (
    get_state_vector,
    load_lengths,
    load_or_build_norm_stats,
)
from sidebar import SidebarSelection


def reshape_traffic_maps(
    vec: np.ndarray,
    start: int,
    traffic_len: int,
    n: int,
    feature_names: List[str],
) -> Dict[str, np.ndarray]:
    if vec is None or len(vec) < start + traffic_len:
        return {name: np.full((n, n), np.nan, dtype=float) for name in feature_names}

    block = vec[start:start + traffic_len]
    out: Dict[str, np.ndarray] = {}
    for i, name in enumerate(feature_names):
        sub = block[i * n * n:(i + 1) * n * n]
        out[name] = np.asarray(sub, dtype=float).reshape(n, n)
    return out


def render_state_level_section(selection: SidebarSelection) -> None:
    st.header("State-Level visualizations")
    sel_expert = selection.expert
    lens = load_lengths(HELPER_PATHS["lengths_dir"], sel_expert)
    traj_len = int(lens[selection.traj_idx]) if lens.size and selection.traj_idx < lens.size else 0

    metric_cols = st.columns(3)
    metric_cols[0].metric("Expert", f"{sel_expert}")
    metric_cols[1].metric(
        f"Trajectories recorded for expert {sel_expert}",
        f"{lens.size:,}",
    )
    metric_cols[2].metric(
        f"States visited in trajectory {selection.traj_idx}",
        f"{traj_len:,}",
    )

    vec = get_state_vector(sel_expert, selection.traj_idx, selection.state_idx) if traj_len else np.array([])

    st.markdown(
        f"#### **State vector (trajectory {selection.traj_idx}, state {selection.state_idx}) — Raw (unnormalized)**"
    )
    feat_df = pd.DataFrame({"dim": [f"f{i}" for i in range(len(vec))], "value": vec})
    bar_raw = px.bar(feat_df, x="dim", y="value")
    bar_raw.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=10, b=20),
        xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(bar_raw, use_container_width=True)

    show_norm = st.checkbox(
        "Show normalized view",
        value=False,
        help="Toggle to compute & display normalized state vector (cached stats).",
    )
    if show_norm and vec.size:
        stats = load_or_build_norm_stats()
        if stats["mean"].shape[0] == vec.shape[0]:
            scale_mode = st.radio(
                "Normalization scaling",
                ["Z-score (mean/std)", "Min-Max [0,1]", "Robust (median/MAD)"],
                horizontal=True,
                key="norm_scale_mode",
            )
            mean_all = stats["mean"]
            std_all = stats["std"]
            min_all = stats["min"]
            max_all = stats["max"]
            median_all = stats["median"]
            mad_all = stats["mad"]
            with np.errstate(divide="ignore", invalid="ignore"):
                if scale_mode.startswith("Z-score"):
                    values = np.where(std_all > 0, (vec - mean_all) / std_all, 0.0)
                    y_title = "Z-score"
                elif scale_mode.startswith("Min-Max"):
                    denom = (max_all - min_all)
                    values = np.where(denom > 0, (vec - min_all) / denom, 0.0)
                    y_title = "Min-Max scaled"
                else:
                    values = np.where(mad_all > 0, (vec - median_all) / mad_all, 0.0)
                    y_title = "(value - median)/MAD"
            norm_df = pd.DataFrame({"dim": [f"f{i}" for i in range(len(values))], "value": values})
            bar_norm = px.bar(norm_df, x="dim", y="value")
            bar_norm.update_layout(
                height=260,
                margin=dict(l=0, r=0, t=10, b=20),
                xaxis=dict(showticklabels=False),
                yaxis=dict(title=y_title),
            )
            st.markdown(f"**State vector (normalized — {y_title})**")
            st.plotly_chart(bar_norm, use_container_width=True)
            st.caption("Stats persisted to norm_stats.npz (reused across runs). Constant features get 0 in all modes.")
        else:
            st.warning("Normalization skipped: feature count mismatch with stored stats.")
    elif show_norm and not vec.size:
        st.info("No state vector available for normalization.")

    st.markdown("#### **Traffic neighborhood for current state (5×5 × 4 features)**")
    traffic_maps = reshape_traffic_maps(
        vec,
        start=0,
        traffic_len=TRAFFIC_COUNT_DEFAULT,
        n=TRAFFIC_NEIGHBORHOOD,
        feature_names=list(TRAFFIC_FEATURES),
    )
    cols_tm = st.columns(len(TRAFFIC_FEATURES))
    for col, name in zip(cols_tm, TRAFFIC_FEATURES):
        Z = traffic_maps[name]
        hm = go.Figure(
            data=[
                go.Heatmap(
                    z=Z,
                    x=list(range(TRAFFIC_NEIGHBORHOOD)),
                    y=list(range(TRAFFIC_NEIGHBORHOOD)),
                    xgap=1,
                    ygap=1,
                    colorscale="Blues",
                    zmin=np.nanmin(Z) if np.isfinite(Z).any() else 0,
                    zmax=np.nanmax(Z) if np.isfinite(Z).any() else 1,
                    hovertemplate="row=%{y}, col=%{x}<br>value=%{z:.3f}<extra></extra>",
                )
            ]
        )
        hm.update_layout(
            title=name,
            height=230,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(
                range=[-0.5, TRAFFIC_NEIGHBORHOOD - 0.5],
                tickmode="array",
                tickvals=list(range(TRAFFIC_NEIGHBORHOOD)),
                ticktext=[str(j + 1) for j in range(TRAFFIC_NEIGHBORHOOD)],
                showgrid=False,
                zeroline=False,
                constrain="domain",
            ),
            yaxis=dict(
                range=[-0.5, TRAFFIC_NEIGHBORHOOD - 0.5],
                tickmode="array",
                tickvals=list(range(TRAFFIC_NEIGHBORHOOD)),
                ticktext=[str(j + 1) for j in range(TRAFFIC_NEIGHBORHOOD)],
                autorange="reversed",
                showgrid=False,
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
            ),
            plot_bgcolor="#22262a",
        )
        col.plotly_chart(hm, use_container_width=True)
