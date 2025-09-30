"""
Implement a system to allow users to add comments and suggest edits!!! --Steven's idea


Expert Explorer — fast, hierarchical visualization for taxi IL trajectories.

Key optimizations:
- Builds compact helper files (Parquet/NPY/NPZ) on first run and reuses them.
- Uses @st.fragment to avoid rerunning the whole script.
- Plotly figures kept light (no per-point dataframes when not needed).

Run:
  pip install streamlit plotly pandas numpy pyarrow
  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config import (
    GRID_H_DEFAULT,
    GRID_W_DEFAULT,
    POI_COUNT,
    STATES_NPY,
    TRAFFIC_FEATURES,
    T_SLOTS_DEFAULT,
    TRAJ_INDEX_PARQUET,
)
from data_access import ensure_hf_artifacts, get_states_matrix, load_summary
from sidebar import build_sidebar
from views import (
    render_action_space_section,
    render_hierarchy_section,
    render_overview_section,
    render_state_level_section,
    render_trajectory_section,
    render_visitation_section,
)


def main() -> None:
    st.set_page_config(page_title="Shenzhen Taxi Dataset Explorer", layout="wide")

    ensure_hf_artifacts()
    if not (STATES_NPY.exists() and TRAJ_INDEX_PARQUET.exists()):
        st.error("Missing helper artifacts (states_all.npy + traj_index.parquet).")
        st.stop()

    summary = load_summary()
    states_matrix = get_states_matrix()
    state_dim = int(states_matrix.shape[1]) if states_matrix.size else 0

    selection = build_sidebar(summary)

    st.title("Shenzhen Taxi Dataset Explorer")

    render_overview_section(
        summary=summary,
        state_dim=state_dim,
        grid_h=GRID_H_DEFAULT,
        grid_w=GRID_W_DEFAULT,
        t_slots=T_SLOTS_DEFAULT,
        traffic_features=TRAFFIC_FEATURES,
        poi_count=POI_COUNT,
    )

    st.divider()
    render_hierarchy_section(summary, selection)

    st.divider()
    render_state_level_section(selection)

    st.divider()
    # Indices remain unavailable in helper-only deployment
    x_idx = -1
    y_idx = -1
    render_trajectory_section(selection, GRID_H_DEFAULT, GRID_W_DEFAULT, x_idx, y_idx)

    st.divider()
    render_visitation_section(selection, GRID_H_DEFAULT, GRID_W_DEFAULT, x_idx, y_idx)

    st.divider()
    render_action_space_section()

    st.markdown("Created by Dr. Xin Zhang and Robert Ashe at San Diego State University")


if __name__ == "__main__":
    main()