from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import streamlit as st

from config import HELPER_PATHS
from data_access import load_lengths


@dataclass
class SidebarSelection:
    expert: str
    traj_idx: int
    state_idx: int


def build_sidebar(summary: pd.DataFrame) -> SidebarSelection:
    st.sidebar.title("Data Selector")

    if summary.empty:
        st.sidebar.info("Building helpers… selectors will populate shortly.")
        selection = SidebarSelection(expert="0", traj_idx=0, state_idx=0)
    else:
        expert_list: List[str] = summary["expert"].astype(str).tolist()
        expert_default = expert_list[0] if expert_list else "0"
        sel_expert = st.sidebar.selectbox("Expert", expert_list, key="sidebar_expert", index=expert_list.index(expert_default) if expert_default in expert_list else 0)

        lens_e_sidebar = load_lengths(HELPER_PATHS["lengths_dir"], sel_expert)
        max_traj_idx = int(lens_e_sidebar.size - 1) if lens_e_sidebar.size else 0
        current_traj_idx = min(st.session_state.get("sidebar_traj_idx", 0), max_traj_idx)
        sel_traj_idx = st.sidebar.number_input(
            "Trajectory index",
            min_value=0,
            max_value=max_traj_idx,
            value=current_traj_idx,
            step=1,
            key="sidebar_traj_idx",
        )

        traj_len = int(lens_e_sidebar[sel_traj_idx]) if lens_e_sidebar.size else 0
        max_state_idx = max(0, traj_len - 1)
        current_state_idx = min(st.session_state.get("sidebar_state_idx", 0), max_state_idx)
        sel_state_idx = st.sidebar.number_input(
            "State index",
            min_value=0,
            max_value=max_state_idx,
            value=current_state_idx,
            step=1,
            key="sidebar_state_idx",
        )
        selection = SidebarSelection(
            expert=str(sel_expert),
            traj_idx=int(sel_traj_idx),
            state_idx=int(sel_state_idx),
        )

    st.sidebar.markdown(
        """## **Selectors drive:**
- **State details** (state vector and traffic features)
- **Trajectory path**
- **State visitation counts** (Selected expert only)."""
    )

    return selection
