from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import HELPER_PATHS
from data_access import load_lengths
from sidebar import SidebarSelection


@st.fragment
def experts_treemap_panel(summary: pd.DataFrame) -> None:
    color_choice = st.radio(
        "Treemap color",
        ["Average length", "Total states"],
        horizontal=True,
        key="color_treemap",
    )
    color_col = "states_avg" if color_choice.startswith("Average") else "states_total"

    treemap = px.treemap(
        summary,
        path=["expert"],
        values="trajectories",
        color=color_col,
        color_continuous_scale="Blues",
        hover_data={
            "trajectories":":,",
            "states_total":":,",
            "states_avg":":.2f",
            "states_min":":.0f",
            "states_med":":.0f",
            "states_max":":.0f",
            "states_p90":":.0f",
        },
    )
    treemap.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=430,
        coloraxis_colorbar=dict(
            title=dict(
                text="Avg length" if color_col == "states_avg" else "Total states",
                side="top",
            )
        ),
    )
    st.subheader("Experts (larger area indicates higher trajectory count)")
    st.plotly_chart(treemap, use_container_width=True)


def render_hierarchy_section(summary: pd.DataFrame, selection: SidebarSelection) -> None:
    if summary.empty:
        st.info("No summary available yet. Build helper artifacts to populate the hierarchy.")
        return

    experts_treemap_panel(summary)

    sel_expert = selection.expert
    st.subheader(f"Trajectory length distribution for expert {sel_expert} (selected in sidebar)")

    lens = load_lengths(HELPER_PATHS["lengths_dir"], sel_expert)
    nbins = min(60, max(10, int(math.sqrt(max(1, lens.size)))))
    hist = go.Figure(data=[go.Histogram(x=lens, nbinsx=nbins)])
    hist.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title=f"Trajectory length (# states) — Expert {sel_expert}",
        yaxis_title="Count",
    )
    st.plotly_chart(hist, use_container_width=True)

    search_fraction_label = st.radio(
        "Assumed search percentage (portion of states that advance time window)",
        ["10%", "25%", "35%", "50%"],
        index=3,
        horizontal=True,
        key="search_fraction_choice",
    )
    fraction_lookup = {"10%": 0.10, "25%": 0.25, "35%": 0.35, "50%": 0.50}
    search_fraction = fraction_lookup[search_fraction_label]
    avg_search_minutes = (lens.mean() * search_fraction * 5) if lens.size else float("nan")

    cA, cB, cC, cD = st.columns(4)
    if lens.size:
        cA.metric("# Traj", f"{lens.size:,}")
        cB.metric("Mean length", f"{lens.mean():.1f}")
        cC.metric("Min/Max", f"{lens.min():.0f}/{lens.max():.0f}")
        cD.metric("Avg search time (min)", f"{avg_search_minutes:.1f}")
    else:
        cA.metric("# Traj", "0")
        cB.metric("Mean length", "-")
        cC.metric("Min/Max", "-")
        cD.metric("Avg search time (min)", "-")

    st.caption("Avg search time ≈ mean_length × chosen_fraction × 5 minutes per time-window state.")
