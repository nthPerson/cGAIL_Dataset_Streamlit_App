from __future__ import annotations

from typing import Sequence

import pandas as pd
import streamlit as st

from config import ACTION_MODE


def render_overview_section(
    summary: pd.DataFrame,
    state_dim: int,
    grid_h: int,
    grid_w: int,
    t_slots: int,
    traffic_features: Sequence[str],
    poi_count: int,
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total experts", f"{len(summary):,}")
    c2.metric(
        "Total trajectories",
        f"{int(summary['trajectories'].sum()):,}" if not summary.empty else "0",
    )
    c3.metric(
        "Total states visited in dataset",
        f"{int(summary['states_total'].sum()):,}" if not summary.empty else "0",
    )
    c4.metric("Features per state", f"{state_dim}")

    with st.expander("Model summary & data flow", expanded=False):
        st.markdown(
            f"""
- **State space**: **{grid_h} × {grid_w}** spatial grid × **{t_slots}** five-minute time slots per day.
- **State features** (example schema used here):
  - Traffic (5×5 neighborhood × 4 features = 100 dims): {', '.join(traffic_features)}
  - POI distances: {poi_count} dims
  - Temporal: remaining dims (time of day, day of week, etc.)
- **Condition features**: Home location distance, working schedule, familiarity (avg visitations to current state), and loction identifier.
- **Action space**: **{ '18 — 9 spatial with/without time advance' if ACTION_MODE.startswith('18') else '9 (paper) — stay or move to 8 neighbors' }**.
- **Raw → model flow**:
  1. Raw CSV → (plate_id, lat, lon, timestamp, passenger flag)
  2. Segment into **vacant trajectories** (drop-off → next pick-up)
  3. Map GPS to **grid & time slots**, build per-state feature vectors
  4. Train IL (e.g., cGAIL) conditioned on driver/location
            """
        )
