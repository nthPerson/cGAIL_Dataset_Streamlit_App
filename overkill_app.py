# -*- coding: utf-8 -*-
"""
Imitation Learning Dataset Explorer (Expert Explorer)
Author: ChatGPT (Robert's collaborator)

Run:
  pip install streamlit plotly pandas numpy streamlit-plotly-events
  streamlit run app.py

Notes:
- Designed for a large dataset (~380 MB pickle).
- Caches heavy computations; avoids rendering thousands of leaves at once.
- Treemap click -> drill to distributions, representatives, maps, and states.

Expected dataset structure (Python pickle):
{
  expert_id: [  # list of trajectories
      [ [f1, f2, ..., fD], [f1, ..., fD], ... ],   # trajectory 0 (list of states)
      [ ... ],                                     # trajectory 1
      ...
  ],
  ...
}
Where each state is a feature vector (default D=126).
"""

from __future__ import annotations
import os, pickle, math, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events


# ------------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------------
DEFAULT_DATA_PATH = "/home/robert/FAMAIL/data/Processed_Data/all_trajs.pkl"

st.set_page_config(
    page_title="Expert Explorer — Imitation Learning Trajectories",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Sidebar: dataset + schema -----------------------------------------------------
st.sidebar.title("Dataset & Schema")

data_path = st.sidebar.text_input("Pickle path", value=DEFAULT_DATA_PATH)
grid_h = st.sidebar.number_input("Grid height (rows)", min_value=1, value=40, help="Number of rows in the city grid.")
grid_w = st.sidebar.number_input("Grid width (cols)", min_value=1, value=50, help="Number of columns in the city grid.")

st.sidebar.markdown("#### Feature schema (per state)")
st.sidebar.caption(
    "Default assumes the first 100 dims are a 5×5 neighborhood × 4 traffic features, "
    "followed by POI distances, then temporal features. Adjust if your layout differs."
)

# Traffic block: 5x5 × 4 features = 100
TRAFFIC_NEIGHBORHOOD = 5  # 5x5
TRAFFIC_FEATURES = ["Speed", "Volume", "Demand", "Waiting"]
traffic_total = (TRAFFIC_NEIGHBORHOOD ** 2) * len(TRAFFIC_FEATURES)
traffic_start = st.sidebar.number_input("Traffic block start idx", min_value=0, value=0, step=1)
traffic_count = st.sidebar.number_input("Traffic block length", min_value=0, value=traffic_total, step=1)

# POIs block
default_pois = 25
poi_start = st.sidebar.number_input("POI block start idx", min_value=0, value=traffic_start + traffic_count, step=1)
poi_count = st.sidebar.number_input("POI count", min_value=0, value=default_pois, step=1)

# Temporal block (everything after POIs)
temporal_start = st.sidebar.number_input(
    "Temporal block start idx", min_value=0, value=poi_start + poi_count, step=1
)

# Optional: indices for (x, y) if embedded in the vector (0-indexed). -1 means "not present".
st.sidebar.markdown("#### (Optional) Cell decoder from state vector")
cell_x_idx = st.sidebar.number_input("X index in vector (0..D-1, -1=none)", value=-1, step=1)
cell_y_idx = st.sidebar.number_input("Y index in vector (0..D-1, -1=none)", value=-1, step=1)

# Action space toggle
action_mode = st.sidebar.selectbox("Action space", options=["9 actions (paper)", "18 actions (extended)"], index=1)

# ------------------------------------------------------------------------------------
# IO + CACHING
# ------------------------------------------------------------------------------------

def _file_sig(p: Path) -> str:
    """Create a cache-busting signature for the dataset file."""
    try:
        stat = p.stat()
        return f"{stat.st_size}-{int(stat.st_mtime)}"
    except Exception:
        return "no-file"

@st.cache_resource(show_spinner=True)
def load_dataset(path: str) -> Dict[str, List[List[List[float]]]]:
    p = Path(path)
    if not p.exists():
        st.error(f"File not found: {path}")
        st.stop()
    sig = _file_sig(p)  # triggers rerun if file changes
    with p.open("rb") as f:
        data = pickle.load(f)
    # normalize expert keys to str for plotting
    norm = {str(k): v for k, v in data.items()}
    return norm

@st.cache_data(show_spinner=False)
def compute_long_df(dataset: Dict[str, List[List[List[float]]]]) -> pd.DataFrame:
    rows = []
    for e, trajs in dataset.items():
        for t_idx, t in enumerate(trajs):
            rows.append({"expert": str(e), "traj_idx": t_idx, "states": len(t)})
    df = pd.DataFrame(rows)
    return df

@st.cache_data(show_spinner=False)
def compute_expert_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    g = long_df.groupby("expert")
    summary = pd.DataFrame({
        "expert": g.size().index,
        "trajectories": g.size().values.astype(int),
        "states_total": g["states"].sum().values.astype(int),
        "states_avg": g["states"].mean().values,
        "states_min": g["states"].min().values.astype(int),
        "states_med": g["states"].median().values,
        "states_max": g["states"].max().values.astype(int),
        "states_p90": g["states"].quantile(0.90).values
    }).sort_values("trajectories", ascending=False).reset_index(drop=True)
    return summary

@st.cache_data(show_spinner=False)
def get_expert_lengths(long_df: pd.DataFrame, expert: str) -> np.ndarray:
    return long_df.loc[long_df["expert"] == expert, "states"].to_numpy()

@st.cache_data(show_spinner=False)
def get_expert_traj_df(long_df: pd.DataFrame, expert: str) -> pd.DataFrame:
    return long_df.loc[long_df["expert"] == expert, ["traj_idx", "states"]].copy()

# ------------------------------------------------------------------------------------
# FEATURE HELPERS
# ------------------------------------------------------------------------------------

def reshape_traffic_maps(vec: np.ndarray,
                         start: int, count: int,
                         neighborhood: int = 5,
                         feature_names: List[str] = TRAFFIC_FEATURES) -> Dict[str, np.ndarray]:
    """
    Convert a flat block of length (n*n*F) into F separate n×n maps.
    Assumes order: for each of 25 cells, consecutive F features.
    """
    F = len(feature_names)
    n = neighborhood
    total = n * n * F
    if count < total or start + total > len(vec):
        # return empty maps with NaN
        return {name: np.full((n, n), np.nan) for name in feature_names}

    block = vec[start:start + total]
    maps = {name: np.zeros((n, n), dtype=float) for name in feature_names}
    for cell in range(n * n):
        base = cell * F
        r, c = divmod(cell, n)
        for f_idx, name in enumerate(feature_names):
            maps[name][r, c] = float(block[base + f_idx])
    return maps

def slice_poi(vec: np.ndarray, start: int, count: int) -> np.ndarray:
    if start < 0 or count <= 0 or start + count > len(vec):
        return np.array([])
    return np.array(vec[start:start + count], dtype=float)

def slice_temporal(vec: np.ndarray, start: int) -> np.ndarray:
    if start < 0 or start >= len(vec):
        return np.array([])
    return np.array(vec[start:], dtype=float)

def try_decode_cell(vec: np.ndarray,
                    grid_h: int, grid_w: int,
                    x_idx: int, y_idx: int) -> Optional[Tuple[int, int]]:
    """Return (row, col) if indices are configured; else None."""
    if 0 <= x_idx < len(vec) and 0 <= y_idx < len(vec):
        x = int(round(vec[x_idx])); y = int(round(vec[y_idx]))
        if 0 <= y < grid_h and 0 <= x < grid_w:
            return (y, x)  # (row, col)
    return None

@st.cache_data(show_spinner=False)
def compute_visitation_for_expert(dataset: Dict[str, Any],
                                  expert: str,
                                  grid_h: int, grid_w: int,
                                  x_idx: int, y_idx: int) -> Optional[np.ndarray]:
    """Count state visits per (row, col) for one expert. Requires x/y indices configured."""
    if x_idx < 0 or y_idx < 0:
        return None
    grid = np.zeros((grid_h, grid_w), dtype=np.int64)
    for traj in dataset[expert]:
        for state in traj:
            cell = try_decode_cell(np.asarray(state), grid_h, grid_w, x_idx, y_idx)
            if cell is not None:
                r, c = cell
                grid[r, c] += 1
    return grid

# ------------------------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------------------------
st.title("Imitation Learning — Expert Explorer")

with st.spinner("Loading dataset..."):
    dataset = load_dataset(data_path)

long_df = compute_long_df(dataset)
summary = compute_expert_summary(long_df)

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Experts", value=f"{len(dataset):,}")
k2.metric("Total trajectories", value=f"{int(summary['trajectories'].sum()):,}")
k3.metric("Total states", value=f"{int(summary['states_total'].sum()):,}")
# infer feature dimension from first state
_sample_expert = next(iter(dataset.keys()))
_sample_state = np.asarray(dataset[_sample_expert][0][0], dtype=float)
k4.metric("Features per state", value=len(_sample_state))

st.divider()

# ------------------------------------------------------------------------------------
# LEVEL 1: EXPERTS TREEMAP (clickable)
# ------------------------------------------------------------------------------------
st.subheader("Experts at a glance")

color_metric = st.radio(
    "Treemap color",
    ["Average trajectory length", "Total states"],
    horizontal=True
)

if color_metric.startswith("Average"):
    color_col = "states_avg"
else:
    color_col = "states_total"

treemap = px.treemap(
    summary,
    path=["expert"],
    values="trajectories",
    color=color_col,
    color_continuous_scale="Blues",
    hover_data={
        "trajectories":":,",
        "states_total":":,",
        "states_avg":":.1f",
        "states_min":":.0f",
        "states_med":":.0f",
        "states_max":":.0f",
        "states_p90":":.0f",
    }
)
treemap.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420)

cA, cB = st.columns([1.1, 1.9], gap="large")
with cA:
    sel = plotly_events(treemap, click_event=True, hover_event=False, select_event=False,
                        override_height=420, override_width="100%")
    if sel:
        sel_expert = sel[0].get("label") or sel[0].get("text") or sel[0].get("customdata")
    else:
        sel_expert = st.selectbox("Pick an expert", summary["expert"].tolist())

    topN = st.slider("Show top-N experts in treemap (by # trajectories)", 5, len(summary), min(20, len(summary)))
    # filter view only (not required for selection)
    if topN < len(summary):
        show_experts = summary.iloc[:topN]["expert"].tolist()
        if sel and sel_expert not in show_experts:
            st.info("Selected expert not in top-N filter; details still shown on the right.")
# else:
#     sel_expert = summary.iloc[0]["expert"]  # fallback

with cB:
    st.markdown(f"### Drill-in: **{sel_expert}**")
    exp_df = get_expert_traj_df(long_df, sel_expert)
    exp_lengths = exp_df["states"].to_numpy()
    n_traj = len(exp_lengths)

    tabs = st.tabs(["Distribution", "Representatives", "Quantiles", "Visitation map", "All trajectories"])

    # ---------------- Distribution tab ----------------
    with tabs[0]:
        bins = max(10, int(np.sqrt(max(1, n_traj))))
        hist = px.histogram(exp_df, x="states", nbins=bins, opacity=0.9,
                            labels={"states": "Trajectory length (# states)"},
                            template="plotly_white")
        hist.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(hist, use_container_width=True)

        # quick stats
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trajectories", f"{n_traj:,}")
        c2.metric("Mean", f"{exp_lengths.mean():.1f}")
        c3.metric("Median", f"{np.median(exp_lengths):.0f}")
        c4.metric("IQR", f"{np.percentile(exp_lengths,75)-np.percentile(exp_lengths,25):.0f}")
        c5.metric("p90", f"{np.percentile(exp_lengths,90):.0f}")
        c6.metric("Min / Max", f"{exp_lengths.min():.0f} / {exp_lengths.max():.0f}")

        # quantile chips
        q_vals = np.quantile(exp_lengths, [0.25, 0.5, 0.75])
        st.caption(f"Quantiles → Q1 ≤ {q_vals[0]:.0f}, Q2 ≤ {q_vals[1]:.0f}, Q3 ≤ {q_vals[2]:.0f}, Q4 > {q_vals[2]:.0f}")

    # ---------------- Representatives tab ----------------
    with tabs[1]:
        st.markdown("**Samples: shortest • typical • longest**")
        k = st.slider("How many from each group?", 2, min(8, n_traj), min(3, n_traj), key=f"k_{sel_expert}")
        sort_idx = np.argsort(exp_lengths)
        shortest_idx = sort_idx[:k]
        longest_idx = sort_idx[-k:][::-1]
        med = np.median(exp_lengths)
        typical_idx = np.argsort(np.abs(exp_lengths - med))[:k]

        def strip_bar(indices, title):
            df = exp_df.iloc[indices][["traj_idx", "states"]].sort_values("states")
            fig = px.bar(df, x="states", y="traj_idx", orientation="h",
                         labels={"states": "states", "traj_idx": "traj"},
                         title=title)
            fig.update_layout(height=180, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            return fig, df

        c1, c2, c3 = st.columns(3)
        with c1:
            fig_s, df_s = strip_bar(shortest_idx, "Shortest")
            st.plotly_chart(fig_s, use_container_width=True)
        with c2:
            fig_t, df_t = strip_bar(typical_idx, "Typical (≈ median)")
            st.plotly_chart(fig_t, use_container_width=True)
        with c3:
            fig_l, df_l = strip_bar(longest_idx, "Longest")
            st.plotly_chart(fig_l, use_container_width=True)

        st.caption("Tip: copy a trajectory index below to open it in the State Inspector.")

    # ---------------- Quantiles tab ----------------
    with tabs[2]:
        qlabels = ["Q1 (shortest 25%)", "Q2 (25–50%)", "Q3 (50–75%)", "Q4 (longest 25%)"]
        q = pd.qcut(exp_lengths, q=4, duplicates="drop",
                    labels=qlabels[:len(pd.qcut(exp_lengths, 4, duplicates='drop').categories)])
        exp_df["quantile"] = q.astype(str)
        stats = (exp_df.groupby("quantile")["states"]
                 .agg(count="count", total="sum", avg="mean", min="min", med="median", max="max")
                 .reset_index().sort_values("quantile"))

        cards = st.columns(len(stats))
        for col, (_, r) in zip(cards, stats.iterrows()):
            with col:
                st.metric(f"{r['quantile']}", f"{int(r['count']):,} traj")
                st.caption(f"min/med/max: {int(r['min'])} / {r['med']:.0f} / {int(r['max'])}")
                mini = px.histogram(exp_df[exp_df["quantile"] == r["quantile"]], x="states", nbins=15)
                mini.update_layout(height=120, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None)
                st.plotly_chart(mini, use_container_width=True)

    # ---------------- Visitation map tab ----------------
    with tabs[3]:
        grid = compute_visitation_for_expert(dataset, sel_expert, grid_h, grid_w, cell_x_idx, cell_y_idx)
        if grid is None:
            st.info("Cell decoder not configured. Set X/Y indices in the sidebar to enable the visitation map.")
        else:
            heat = go.Figure(data=go.Heatmap(
                z=grid, coloraxis="coloraxis",
                hovertemplate="row=%{y}, col=%{x}<br>visits=%{z}<extra></extra>"
            ))
            heat.update_layout(
                height=450, margin=dict(l=0, r=0, t=10, b=0),
                coloraxis=dict(colorscale="Blues"),
                xaxis_title="Column (x)", yaxis_title="Row (y)", yaxis_autorange="reversed"
            )
            st.plotly_chart(heat, use_container_width=True)
            st.caption("Counts of state visits per grid cell for the selected expert.")

    # ---------------- All trajectories heatmap tab ----------------
    with tabs[4]:
        max_traj_idx = int(exp_df["traj_idx"].max()) if n_traj > 0 else 0
        z = [[np.nan] * (max_traj_idx + 1)]
        for _, r in exp_df.iterrows():
            z[0][int(r["traj_idx"])] = int(r["states"])
        heat = go.Figure(data=go.Heatmap(
            z=z, x=list(range(max_traj_idx + 1)), y=[sel_expert],
            coloraxis="coloraxis",
            hovertemplate="expert=%{y}<br>traj=%{x}<br>#states=%{z}<extra></extra>"
        ))
        heat.update_layout(height=160, margin=dict(l=0, r=0, t=0, b=0), coloraxis=dict(colorscale="Blues"))
        chosen = plotly_events(heat, click_event=True, select_event=False)
        if chosen:
            chosen_traj = int(chosen[0]["x"])
            st.session_state["chosen_traj_idx"] = chosen_traj
            st.success(f"Selected trajectory {chosen_traj}. Scroll to State Inspector below.")

st.divider()

# ------------------------------------------------------------------------------------
# STATE INSPECTOR
# ------------------------------------------------------------------------------------
st.header("State Inspector")

colSI = st.columns([1, 1])
with colSI[0]:
    sel_expert_input = st.selectbox("Expert", summary["expert"].tolist(), index=summary.index[summary["expert"] == sel_expert].tolist()[0] if sel_expert in summary["expert"].tolist() else 0)
with colSI[1]:
    max_traj_for_expert = len(dataset[sel_expert_input]) - 1
    default_traj = min(st.session_state.get("chosen_traj_idx", 0), max_traj_for_expert)
    sel_traj_idx = st.number_input("Trajectory index", min_value=0, max_value=max_traj_for_expert, value=default_traj, step=1)

trajectory = dataset[sel_expert_input][sel_traj_idx]
state_count = len(trajectory)
st.caption(f"Trajectory has **{state_count}** states.")

# State runway (hover shows it's a 126-d vector)
X = np.array(trajectory, dtype=float)  # [n_states, D]
preview_dims = min(6, X.shape[1])
preview = np.round(X[:, :preview_dims], 3).tolist()
runway = go.Figure(data=go.Scatter(
    x=list(range(state_count)), y=[0]*state_count, mode="markers",
    marker=dict(size=6),
    customdata=preview,
    hovertemplate=("state=%{x}<br>first dims=%{customdata}<br>(vector length = " + str(X.shape[1]) + ")<extra></extra>")
))
runway.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(visible=False))
st.plotly_chart(runway, use_container_width=True)

sel_state = st.slider("Inspect state index", 0, max(0, state_count-1), 0)
vec = X[sel_state]

# ---- Traffic maps (5×5×4) ----
traffic_maps = reshape_traffic_maps(vec, start=traffic_start, count=traffic_count,
                                    neighborhood=TRAFFIC_NEIGHBORHOOD, feature_names=TRAFFIC_FEATURES)

st.markdown("#### Traffic neighborhood (5×5 around current cell)")
grid_cols = st.columns(4)
for i, name in enumerate(TRAFFIC_FEATURES):
    with grid_cols[i % 4]:
        m = traffic_maps.get(name, np.full((TRAFFIC_NEIGHBORHOOD, TRAFFIC_NEIGHBORHOOD), np.nan))
        fig = go.Figure(data=go.Heatmap(z=m, coloraxis="coloraxis",
                                        hovertemplate="r=%{y}, c=%{x}<br>value=%{z:.3f}<extra></extra>"))
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0),
                          coloraxis=dict(colorscale="Blues"),
                          title=name, yaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

# ---- POI bar chart ----
poi_vals = slice_poi(vec, poi_start, poi_count)
st.markdown("#### POI distances")
if poi_vals.size > 0:
    poi_df = pd.DataFrame({"POI": [f"POI_{i+1}" for i in range(len(poi_vals))], "distance": poi_vals})
    bar = px.bar(poi_df, x="POI", y="distance")
    bar.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=20), xaxis_tickangle=-45)
    st.plotly_chart(bar, use_container_width=True)
else:
    st.info("POI block not configured or not present at these indices.")

# ---- Temporal chip ----
temporal_vals = slice_temporal(vec, temporal_start)
st.markdown("#### Temporal features")
if temporal_vals.size > 0:
    tf = pd.DataFrame({"feature": [f"t{i}" for i in range(len(temporal_vals))], "value": temporal_vals})
    tfig = px.bar(tf, x="feature", y="value")
    tfig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=20), xaxis=dict(showticklabels=True))
    st.plotly_chart(tfig, use_container_width=True)
else:
    st.info("Temporal block not configured or not present at these indices.")

st.divider()

# ------------------------------------------------------------------------------------
# ACTION SPACE VIEWER (semantics)
# ------------------------------------------------------------------------------------
st.header("Action space viewer")

def action_glyph(advance_time: bool) -> go.Figure:
    """
    Draw a 3x3 grid centered on (0,0). If advance_time=True, hollow markers.
    """
    xs, ys = [], []
    arrows_x, arrows_y, arrows_dx, arrows_dy = [], [], [], []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            xs.append(dx); ys.append(dy)
            if dx == 0 and dy == 0:
                continue
            arrows_x.append(0); arrows_y.append(0)
            arrows_dx.append(dx); arrows_dy.append(dy)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text",
                             marker=dict(size=18, color="#1f77b4", symbol="diamond"),
                             text=["S"], textposition="bottom center", name="state"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers",
                             marker=dict(size=10, color="rgba(0,0,0,0)" if advance_time else "#1f77b4",
                                         line=dict(width=1, color="#1f77b4")),
                             showlegend=False))
    for x, y, dx, dy in zip(arrows_x, arrows_y, arrows_dx, arrows_dy):
        fig.add_annotation(x=x+dx*0.8, y=y+dy*0.8, ax=x, ay=y,
                           axref="x", ayref="y", xref="x", yref="y",
                           arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#1f77b4")
    title = "Advance time" if advance_time else "No time advance"
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0),
                      xaxis=dict(range=[-1.5, 1.5], zeroline=False, showgrid=False, showticklabels=False),
                      yaxis=dict(range=[-1.5, 1.5], zeroline=False, showgrid=False, showticklabels=False),
                      title=title)
    return fig

if action_mode.startswith("18"):
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(action_glyph(False), use_container_width=True)
    with c2: st.plotly_chart(action_glyph(True), use_container_width=True)
else:
    st.plotly_chart(action_glyph(False), use_container_width=True)
    st.caption("9 actions: stay or move to one of 8 neighbors (no time advancement).")

st.divider()

# ------------------------------------------------------------------------------------
# ABOUT / PIPELINE (context)
# ------------------------------------------------------------------------------------
with st.expander("About this dataset & pipeline", expanded=False):
    st.markdown("""
**Hierarchy:** 50 experts (drivers) → trajectories (drop-off to next pick-up) → states.

**State representation (default schema here):**
- Traffic block: 5×5 neighborhood × 4 features = 100 dims (Speed, Volume, Demand, Waiting)
- POI distances: configurable (default 25)
- Temporal features: remainder of the vector (e.g., time-of-day, day-of-week)
- Total vector length commonly ~126 (may include/omit an action label depending on preprocessing)

**Actions:**
- 9 actions (paper): stay or move to 8 neighboring cells.
- 18 actions (extended in this app): the 9 actions above, plus versions that also advance the time slot.

**Performance tips:**
- All heavy summaries are cached. Changing the feature schema or dataset path will invalidate caches.
- Visitation map requires knowing each state's (x, y) cell. If your vector includes them, set indices in the sidebar.
- For huge experts, use representatives / quantile tabs rather than drawing all trajectories at once.
    """)

st.caption("Built with ❤️ for Robert’s IL project.")
