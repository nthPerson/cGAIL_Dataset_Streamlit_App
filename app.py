# -*- coding: utf-8 -*-
"""
Expert Explorer — fast, hierarchical visualization for taxi IL trajectories.

Key optimizations:
- Builds compact helper files (Parquet/NPY/NPZ) on first run and reuses them.
- Avoids hashing the huge pickle in cache keys (uses a tiny file signature string).
- Lazy per-expert loads, but precomputes the small things so sub-views are snappy.
- Uses @st.fragment to avoid rerunning the whole script.
- Replaces full sorts with np.argpartition for O(n) representative picks.
- Plotly figures kept light (no per-point dataframes when not needed).

Run:
  pip install streamlit plotly pandas numpy pyarrow streamlit-plotly-events
  streamlit run app.py
"""

from __future__ import annotations
import json, os, pickle, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# ----------------------------- EARLY FAST LOADER NEEDED BY SIDEBAR ---------------
@st.cache_data(show_spinner=False)
def load_lengths(lengths_dir: str, expert: str) -> np.ndarray:
    """
    Load (and cache) the per-trajectory lengths array for a given expert.

    Placed early because the sidebar needs it before the later bulk loader
    section is reached during script execution.
    """
    path = Path(lengths_dir) / f"{expert}.npy"
    if not path.exists():
        return np.array([], dtype=np.int32)
    try:
        return np.load(path)
    except Exception:
        return np.array([], dtype=np.int32)

# ----------------------------- CONFIG -------------------------------------------
DEFAULT_PKL = "/home/robert/FAMAIL/data/Processed_Data/all_trajs.pkl"
DERIVED_DIR = Path("./derived")  # helper files live here
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

GRID_H_DEFAULT, GRID_W_DEFAULT, T_SLOTS_DEFAULT = 40, 50, 288  # 40x50 grid, 288 five-minute slots/day

TRAFFIC_NEIGHBORHOOD = 5
TRAFFIC_FEATURES = ["Traffic Speed", "Traffic Volume", "Traffic Demand", "Traffic Waiting"]
TRAFFIC_COUNT_DEFAULT = (TRAFFIC_NEIGHBORHOOD ** 2) * len(TRAFFIC_FEATURES)  # 100

st.set_page_config(page_title="Expert Explorer — fast IL viz", layout="wide")

# ----------------------------- SIDEBAR (REVAMPED) --------------------------------
st.sidebar.title("Data Selector")

# (Dataset + static grid params — keep hard‑coded for now)
pkl_path = DEFAULT_PKL
grid_h = GRID_H_DEFAULT
grid_w = GRID_W_DEFAULT
t_slots = T_SLOTS_DEFAULT

# Feature block (kept static)
traffic_start = 0
traffic_len = TRAFFIC_COUNT_DEFAULT
poi_start = traffic_start + traffic_len
poi_count = 23
temporal_start = poi_start + poi_count

# Removed: X/Y/T index & action space selectors.
# Set them to -1 so downstream code knows they are unavailable.
x_idx = -1
y_idx = -1
t_idx = -1
action_mode = "18 actions (extended)"

# Load summary early so we can drive sidebar selectors
# (helper_paths is created a bit later; we need a safe path first)
# We temporarily derive path; after ensure_helpers runs we reload summary (cheap).
temp_summary_path = DERIVED_DIR / "experts_summary.parquet"
if temp_summary_path.exists():
    try:
        _summary_sidebar = pd.read_parquet(temp_summary_path)
    except Exception:
        _summary_sidebar = pd.DataFrame(columns=["expert"])
else:
    _summary_sidebar = pd.DataFrame(columns=["expert"])

expert_list_sidebar = _summary_sidebar["expert"].tolist()
if not expert_list_sidebar:
    # Fallback placeholder until helpers build; user can rerun selectors after warmup
    st.sidebar.info("Building helpers… selectors will populate shortly.")
    sel_expert = "0"
    sel_traj_idx = 0
    sel_state_idx = 0
else:
    sel_expert = st.sidebar.selectbox("Expert", expert_list_sidebar, key="sidebar_expert")
    lens_e_sidebar = load_lengths(str(DERIVED_DIR / "lengths_by_expert"), sel_expert)
    max_traj_idx_sb = int(lens_e_sidebar.size-1) if lens_e_sidebar.size else 0
    sel_traj_idx = st.sidebar.number_input(
        "Trajectory index", min_value=0, max_value=max_traj_idx_sb,
        value=min(st.session_state.get("sidebar_traj_idx", 0), max_traj_idx_sb),
        step=1, key="sidebar_traj_idx"
    )
    traj_len_sb = int(lens_e_sidebar[sel_traj_idx]) if lens_e_sidebar.size else 0
    sel_state_idx = st.sidebar.number_input(
        "State index", min_value=0, max_value=max(0, traj_len_sb-1),
        value=min(st.session_state.get("sidebar_state_idx", 0), max(0, traj_len_sb-1)),
        step=1, key="sidebar_state_idx"
    )

st.sidebar.markdown("Selectors drive: State details, Trajectory path, Visitation (Selected expert).")

# ----------------------------- UTILS --------------------------------------------
def file_sig(path: str) -> str:
    """Tiny string used as cache key; avoids hashing the giant dataset."""
    p = Path(path)
    if not p.exists(): return "missing"
    stt = p.stat()
    return f"{stt.st_size}-{int(stt.st_mtime)}"

SIG = file_sig(pkl_path)

def derived_meta_path() -> Path:
    return DERIVED_DIR / "derived_meta.json"

def read_meta() -> dict:
    if derived_meta_path().exists():
        return json.loads(derived_meta_path().read_text())
    return {}

def write_meta(meta: dict) -> None:
    derived_meta_path().write_text(json.dumps(meta, indent=2))

@st.cache_resource(show_spinner=True)
def load_pkl(path: str) -> Dict[str, List[List[List[float]]]]:
    """Load the original pickle once per session; keep it out of cache keys elsewhere."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    # normalize expert keys to strings
    return {str(k): v for k, v in raw.items()}


# ----------------------- FAST LOADERS (cached by tiny keys) -----------------------
@st.cache_data(show_spinner=False)
def load_summary(p: str) -> pd.DataFrame:
    return pd.read_parquet(p)

@st.cache_data(show_spinner=False)
def load_paths(paths_parquet: str, expert: Optional[str]=None) -> pd.DataFrame:
    p = Path(paths_parquet)
    if not p.exists(): return pd.DataFrame()
    df = pd.read_parquet(p)
    if expert is not None:
        df = df[df["expert"] == str(expert)]
    return df

@st.cache_data(show_spinner=False)
def load_visit_npz(npz_path: str) -> Optional[np.ndarray]:
    p = Path(npz_path)
    if not p.exists(): return None
    return np.load(p)["counts"]

# ----------------------- HELPER FILE BUILDER -------------------------------------
def ensure_helpers(pkl_sig: str,
                   grid_h: int, grid_w: int, t_slots: int,
                   x_idx: int, y_idx: int, t_idx: int) -> dict:
    """
    Build helper files if missing or if schema/file signature changed.

    Writes:
      - experts_summary.parquet : per-expert counts
      - lengths_by_expert/*.npy : trajectory lengths per expert
      - paths.parquet (optional) : expert, traj_idx, state_idx, y, x, t (if x/y present)
      - visitation_overall.npz (optional) : (grid_h, grid_w, t_slots) counts across all experts
    """
    meta = read_meta()
    wanted = dict(
        pkl_sig=pkl_sig, grid_h=grid_h, grid_w=grid_w, t_slots=t_slots,
        x_idx=x_idx, y_idx=y_idx, t_idx=t_idx, version=2
    )
    rebuild = (meta != wanted)

    summary_parquet = DERIVED_DIR / "experts_summary.parquet"
    lengths_dir = DERIVED_DIR / "lengths_by_expert"
    lengths_dir.mkdir(exist_ok=True, parents=True)
    paths_parquet = DERIVED_DIR / "paths.parquet"
    visit_npz = DERIVED_DIR / "visitation_overall.npz"

    if rebuild or not summary_parquet.exists():
        st.info("Building helper files (first run or schema changed)…")
        DATA = load_pkl(pkl_path)

        # --- summary + lengths
        rows = []
        for e, trajs in DATA.items():
            lens = np.fromiter((len(t) for t in trajs), dtype=np.int32)
            np.save(lengths_dir / f"{e}.npy", lens)
            rows.append(dict(
                expert=str(e),
                trajectories=int(lens.size),
                states_total=int(lens.sum()),
                states_avg=float(lens.mean()) if lens.size else 0.0,
                states_min=int(lens.min()) if lens.size else 0,
                states_med=float(np.median(lens)) if lens.size else 0.0,
                states_max=int(lens.max()) if lens.size else 0,
                states_p90=float(np.percentile(lens, 90)) if lens.size else 0.0,
            ))
        pd.DataFrame(rows).to_parquet(summary_parquet, index=False)

        # --- optional path/time extraction
        if x_idx >= 0 and y_idx >= 0:
            # write compact (expert, traj, state, y, x, maybe t)
            # we keep small dtypes to shrink disk size
            recs_e, recs_t, recs_s, recs_y, recs_x, recs_tt = [], [], [], [], [], []
            for e, trajs in DATA.items():
                for ti, traj in enumerate(trajs):
                    for si, state in enumerate(traj):
                        v = np.asarray(state)
                        xx, yy = int(round(v[x_idx])), int(round(v[y_idx]))
                        recs_e.append(str(e)); recs_t.append(np.int32(ti)); recs_s.append(np.int32(si))
                        recs_x.append(np.int16(max(0, min(grid_w-1, xx))))
                        recs_y.append(np.int16(max(0, min(grid_h-1, yy))))
                        if t_idx >= 0:
                            tt = int(round(v[t_idx])); recs_tt.append(np.int16(max(0, min(t_slots-1, tt))))
                        else:
                            recs_tt.append(np.int16(-1))
            df_paths = pd.DataFrame(dict(
                expert=recs_e, traj_idx=recs_t, state_idx=recs_s,
                y=recs_y, x=recs_x, t=recs_tt
            ))
            df_paths.to_parquet(paths_parquet, index=False)

            # overall visitation (3D) if time available
            if t_idx >= 0:
                grid = np.zeros((grid_h, grid_w, t_slots), dtype=np.uint32)
                # vectorized bincount via flat indices
                flat = (df_paths["y"].astype(np.int64) * grid_w + df_paths["x"].astype(np.int64)) * t_slots + df_paths["t"].astype(np.int64)
                valid = flat >= 0
                bc = np.bincount(flat[valid], minlength=grid_h*grid_w*t_slots)
                grid = bc.reshape(grid_h, grid_w, t_slots)
                np.savez_compressed(visit_npz, counts=grid)

        write_meta(wanted)
    else:
        # If grid indices are no longer available, remove stale spatial helper files
        if (x_idx < 0 or y_idx < 0):
            if paths_parquet.exists():
                try: paths_parquet.unlink()
                except Exception: pass
            if visit_npz.exists():
                try: visit_npz.unlink()
                except Exception: pass

    return dict(
        summary_parquet=str(summary_parquet),
        lengths_dir=str(lengths_dir),
        paths_parquet=str(paths_parquet),
        visit_npz=str(visit_npz),
        meta=read_meta()
    )

# (helper_paths created here as before)
helper_paths = ensure_helpers(SIG, grid_h, grid_w, t_slots, x_idx, y_idx, t_idx)

# Reload summary now that helpers may have been (re)built
summary = load_summary(helper_paths["summary_parquet"])
if sel_expert not in summary["expert"].astype(str).tolist():
    # If sidebar was shown before helper build, reset to first expert now
    sel_expert = summary.iloc[0]["expert"]

# ---------------------------- SMALL HELPERS ---------------------------------------
def state_vec_from_pkl(dataset: Dict[str, Any], expert: str, traj_idx: int, state_idx: int) -> np.ndarray:
    return np.asarray(dataset[expert][traj_idx][state_idx], dtype=float)

def reshape_traffic_maps(vec: np.ndarray, start: int, count: int,
                         n: int = TRAFFIC_NEIGHBORHOOD,
                         feature_names: List[str] = TRAFFIC_FEATURES) -> Dict[str, np.ndarray]:
    F = len(feature_names)
    needed = n*n*F
    if start < 0 or count < needed or start+needed > len(vec):
        return {name: np.full((n, n), np.nan) for name in feature_names}
    block = vec[start:start+needed]
    out = {name: np.zeros((n, n), float) for name in feature_names}
    for cell in range(n*n):
        r, c = divmod(cell, n)
        base = cell*F
        for fi, name in enumerate(feature_names):
            out[name][r, c] = block[base+fi]
    return out

def argpartition_reps(lens: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lens.size == 0: return np.array([], int), np.array([], int), np.array([], int)
    k = min(k, lens.size)
    shortest = np.argpartition(lens, k-1)[:k]
    longest = np.argpartition(-lens, k-1)[:k]
    med = np.median(lens)
    typical = np.argpartition(np.abs(lens - med), k-1)[:k]
    return shortest, typical, longest

# ================================================================================
#                                APP BODY
# ================================================================================

st.title("Expert Explorer — cGAIL Taxi Dataset")

# ---------------------- DATA SUMMARY (1) & MODEL SUMMARY (2) ----------------------
summary = load_summary(helper_paths["summary_parquet"])

# Infer features per state from the original PKL (load once)
with st.spinner("Warming up (loading original pickle once…)"):
    DATA = load_pkl(pkl_path)
first_expert = next(iter(DATA.keys()))
first_state_vec = np.asarray(DATA[first_expert][0][0], dtype=float)

# Want to add a description of the model here: 
#   see cGAIL paper for visual (maybe include the vis)
# 

c1, c2, c3, c4 = st.columns(4)
c1.metric("Experts", f"{len(summary):,}")
c2.metric("Total trajectories", f"{int(summary['trajectories'].sum()):,}")
c3.metric("Total states visited by experts", f"{int(summary['states_total'].sum()):,}")
c4.metric("Features per state", f"{len(first_state_vec)}")

with st.expander("Model summary & data flow", expanded=False):
    st.markdown(f"""
- **State space**: **{grid_h} × {grid_w}** spatial grid × **{t_slots}** five-minute time slots per day.
- **State features** (example schema used here):
  - Traffic (5×5 neighborhood × 4 features = 100 dims): {', '.join(TRAFFIC_FEATURES)}
  - POI distances: {poi_count} dims
  - Temporal: remaining dims (time of day, day of week, etc.)
- **Condition features**: driver, location (region) and optionally familiarity/home indicators.
- **Action space**: **{ '18 (extended) — 9 spatial with/without time advance' if action_mode.startswith('18') else '9 (paper) — stay or move to 8 neighbors' }**.
- **Raw → model flow**:
  1. Raw CSV → (plate_id, lat, lon, timestamp, passenger flag)
  2. Segment into **vacant trajectories** (drop-off → next pick-up)
  3. Map GPS to **grid & time slots**, build per-state feature vectors
  4. Train IL (e.g., cGAIL) conditioned on driver/location
""")

st.divider()

# ---------------------- HIERARCHY: EXPERTS → DISTRIBUTION (4) ---------------------
st.header("Hierarchy")

@st.fragment
def experts_treemap_panel():
    # Treemap full-width; color toggle preserved
    color_choice = st.radio(
        "Treemap color",
        ["Average length", "Total states"],
        horizontal=True,
        key="color_treemap"
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
        }
    )
    treemap.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=430,
        coloraxis_colorbar=dict(
            title=dict(
                text="Avg length" if color_col == "states_avg" else "Total states",
                side="top"
            )
        )
    )
    st.subheader("Experts (larger area indicates higher number of trajectories)")
    # Render treemap
    st.plotly_chart(treemap, use_container_width=True)

    # NOTE: Without plotly_events we cannot capture click → automatic syncing
    # of selected expert from treemap to the distribution panel is not possible.
    # Users select expert via the selectbox below for now.
    # (If you later allow plotly_events again, wrap it in a fragment to sync session_state.)
experts_treemap_panel()

st.subheader("Trajectory length distribution by expert")
expert_list = summary["expert"].tolist()
selected_expert = st.session_state.get("expert_selected", expert_list[0])
matching_indices = summary.index[summary["expert"] == selected_expert].tolist()
index = matching_indices[0] if matching_indices else 0
expert = st.selectbox(
    "Expert",
    options=expert_list,
    index=index,
    key="expert_selectbox_right"
)
lens = load_lengths(helper_paths["lengths_dir"], expert)

nbins = min(60, max(10, int(math.sqrt(max(1, lens.size)))))
hist = go.Figure(data=[go.Histogram(x=lens, nbinsx=nbins)])
hist.update_layout(
    height=300,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis_title="Trajectory length (# states)",
    yaxis_title="Count"
)
st.plotly_chart(hist, use_container_width=True)

cA, cB, cC = st.columns(3)
if lens.size:
    cA.metric("# Traj", f"{lens.size:,}")
    cB.metric("Mean", f"{lens.mean():.1f}")
    # cC.metric("Median", f"{np.median(lens):.0f}")    
    # cD.metric("IQR", f"{np.percentile(lens,75)-np.percentile(lens,25):.0f}")
    # cE.metric("p90", f"{np.percentile(lens,90):.0f}")
    cC.metric("Min/Max", f"{lens.min():.0f}/{lens.max():.0f}")
else:
    cA.metric("# Traj", "0"); cB.metric("Mean","-"); cC.metric("Min/Max","-")

st.divider()

# ---------------------- SELECT → TRAJ → STATE (5) ---------------------------------
st.header("State (select expert → trajectory → state)")

# We now use sidebar selections (sel_expert, sel_traj_idx, sel_state_idx)
lens_e = load_lengths(helper_paths["lengths_dir"], sel_expert)
traj_len = int(lens_e[sel_traj_idx]) if lens_e.size else 0

metric_cols = st.columns(3)
metric_cols[0].metric(f"Expert", f"{sel_expert}")
metric_cols[1].metric(f"Trajectories recorded for expert {sel_expert}", f"{lens_e.size:,}")
metric_cols[2].metric(f"States visited in trajectory {sel_traj_idx}", f"{traj_len:,}")

# fetch vector directly
if traj_len == 0:
    st.info("Selected trajectory is empty.")
    vec = np.array([])
else:
    vec = state_vec_from_pkl(DATA, sel_expert, sel_traj_idx, sel_state_idx)

st.markdown(f"**State vector (trajectory {sel_traj_idx}, state {sel_state_idx})**")
feat_df = pd.DataFrame({"dim": [f"f{i}" for i in range(len(vec))], "value": vec})
bar = px.bar(feat_df, x="dim", y="value")
bar.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=20), xaxis=dict(showticklabels=False))
st.plotly_chart(bar, use_container_width=True)

# Traffic neighborhood (semantic peek)
st.header("**Traffic neighborhood for current state (5×5 × 4 features)**")
# st.markdown("**Traffic neighborhood for current state (5×5 × 4 features)**")
traffic_maps = reshape_traffic_maps(vec, traffic_start, traffic_len, n=TRAFFIC_NEIGHBORHOOD, feature_names=TRAFFIC_FEATURES)
cols_tm = st.columns(4)
for i, name in enumerate(TRAFFIC_FEATURES):
    Z = traffic_maps[name]
    hm = go.Figure(data=[go.Heatmap(
        z=Z,
        x=list(range(5)), y=list(range(5)),
        xgap=1, ygap=1,                        # optional: visual cell borders (remove if undesired)
        colorscale="Blues",
        zmin=np.nanmin(Z) if np.isfinite(Z).any() else 0,
        zmax=np.nanmax(Z) if np.isfinite(Z).any() else 1,
        hovertemplate="row=%{y}, col=%{x}<br>value=%{z:.3f}<extra></extra>"
    )])
    hm.update_layout(
        title=name,
        height=230,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            range=[-0.5, 4.5],
            tickmode="array",
            tickvals=list(range(5)),
            ticktext=[str(j+1) for j in range(5)],
            showgrid=False,
            zeroline=False,
            constrain="domain"
        ),
        yaxis=dict(
            range=[-0.5, 4.5],
            tickmode="array",
            tickvals=list(range(5)),
            ticktext=[str(j+1) for j in range(5)],
            autorange="reversed",
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor="#22262a"
    )
    cols_tm[i].plotly_chart(hm, use_container_width=True)

st.divider()

# ---------------------- TRAJECTORY ON GRID (6) ------------------------------------
st.header("Trajectory path on grid (40×50)")

def infer_position_dims(traj: List[List[float]], grid_h: int, grid_w: int,
                        sample_cap: int = 400) -> Tuple[Optional[int], Optional[int], Optional[int], str]:
    """
    Try to infer (x,y) feature indices OR a flattened region-id index.
    Returns (x_dim, y_dim, flat_dim, strategy_msg).
    Priority: explicit sidebar indices > inferred (x,y) pair > inferred flat > fallback first feature.
    """
    L = min(len(traj), sample_cap)
    arr = np.asarray(traj[:L], dtype=float)
    D = arr.shape[1]

    # Helper: integer-like
    def int_like(v: np.ndarray) -> np.ndarray:
        return np.allclose(v, np.round(v), atol=1e-6)

    # Collect candidate x and y dims
    cand_x, cand_y = [], []
    for d in range(D):
        col = arr[:, d]
        if int_like(col):
            if np.all((col >= 0) & (col < grid_w)) and (col.max() - col.min() >= 2):
                cand_x.append(d)
            if np.all((col >= 0) & (col < grid_h)) and (col.max() - col.min() >= 2):
                cand_y.append(d)

    # Try to pair (distinct dims)
    for xd in cand_x:
        for yd in cand_y:
            if xd != yd:
                return xd, yd, None, f"inferred separate x={xd}, y={yd}"

    # Look for a flattened id dim
    for d in range(D):
        col = arr[:, d]
        if int_like(col) and np.all((col >= 0) & (col < grid_h * grid_w)) and (col.max() - col.min() >= grid_w):
            return None, None, d, f"inferred flat region id dim={d}"

    # Fallback: first feature
    return None, None, 0, "fallback: used first feature (may be incorrect)"

@st.fragment
def draw_path_for_trajectory(expert: str, traj_idx: int, state_idx: int):
    if expert not in DATA or traj_idx >= len(DATA[expert]):
        st.info("Trajectory not available.")
        return
    traj = DATA[expert][traj_idx]
    if not traj:
        st.info("Empty trajectory.")
        return

    # Decide decoding strategy
    decode_note = ""
    if x_idx >= 0 and y_idx >= 0:
        xd, yd, fd = x_idx, y_idx, None
        decode_note = f"explicit x={xd}, y={yd}"
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
            xs.append(xx); ys.append(yy)

    if not xs:
        st.info("Could not decode any (x,y) positions for this trajectory.")
        return

    # Build visitation matrix
    visits = np.zeros((grid_h, grid_w), dtype=int)
    for yy, xx in zip(ys, xs):
        visits[yy, xx] += 1

    # Custom colorscale: zero = light gray
    vmax = max(1, visits.max())
    colorscale = [
        [0.0, "#f0f2f5"],
        [0.00001, "#e2e6ea"],
        [1.0, "#0d4d92"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=visits,
        x=list(range(grid_w)),          # explicit coordinates
        y=list(range(grid_h)),
        colorscale=colorscale,
        zmin=0,
        zmax=vmax,
        xgap=1,                         # <— adds vertical cell borders (gap width in px)
        ygap=1,                         # <— adds horizontal cell borders
        colorbar=dict(title="Visits"),
        hovertemplate="(x=%{x}, y=%{y})<br>visits=%{z}<extra></extra>"
    ))

    # Path polyline
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="lines+markers",
        line=dict(width=2, color="#26456e"),
        marker=dict(size=6, color="#17324f"),
        name="Path",
        hovertemplate="Step %{pointNumber}<br>(x=%{x}, y=%{y})<extra></extra>"
    ))

    # Highlight current state
    if 0 <= state_idx < len(xs):
        fig.add_trace(go.Scatter(
            x=[xs[state_idx]],
            y=[ys[state_idx]],
            mode="markers",
            marker=dict(size=13, color="#ff4136", line=dict(width=2, color="white")),
            name="Current state",
            hovertemplate=f"Current state idx={state_idx}<br>(x=%{{x}}, y=%{{y}})<extra></extra>"
        ))

    # Start / End
    fig.add_trace(go.Scatter(
        x=[xs[0]], y=[ys[0]],
        mode="markers+text",
        text=["Start"],
        textposition="top center",
        marker=dict(size=11, color="#2ecc40"),
        name="Start",
        hovertemplate="Start<br>(x=%{x}, y=%{y})<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]],
        mode="markers+text",
        text=["End"],
        textposition="top center",
        marker=dict(size=11, color="#ff851b"),
        name="End",
        hovertemplate="End<br>(x=%{x}, y=%{y})<extra></extra>"
    ))

    # Optional arrows (first 200 transitions)
    for i in range(min(len(xs)-1, 200)):
        fig.add_annotation(
            x=xs[i+1], y=ys[i+1],
            ax=xs[i], ay=ys[i],
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1, arrowwidth=1.3,
            opacity=0.55, arrowcolor="#26456e"
        )

    # Compute a height that preserves aspect ratio (rows/cols) for typical wide container
    target_width_px = 1100  # heuristic; Streamlit will still scale width to container
    aspect = grid_h / grid_w
    fig_height = int(target_width_px * aspect)  # ensures square cells

    fig.update_layout(
        height=fig_height,
        margin=dict(l=0, r=0, t=70, b=0),
        title=f"Expert {expert} — Trajectory {traj_idx} (len={len(xs)})",
        plot_bgcolor="#22262a",
        xaxis=dict(
            title="x (col)",
            range=[-0.5, grid_w-0.5],
            dtick=1,
            showgrid=False,
            zeroline=False,
            constrain="domain"          # use full width for x-domain
        ),
        yaxis=dict(
            title="y (row)",
            range=[-0.5, grid_h-0.5],
            autorange="reversed",
            dtick=1,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",            # lock y-units to x-units → square cells
            scaleratio=1
        ),
        legend=dict(
            orientation="h",
            y=-0.08,
            x=0,
            bgcolor="rgba(0,0,0,0)"
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Visited regions (heatmap) during trajectory. Green=start, Orange=end, Red=current.")

draw_path_for_trajectory(sel_expert, sel_traj_idx, sel_state_idx)

st.divider()

# ---------------------- STATE VISITATION COUNTS (7) -------------------------------
st.header("State visitation counts")

@st.fragment
def visitation_panel():
    scope = st.radio("Scope", ["All experts", "Selected expert"], horizontal=True, key="visit_scope")

    # --- Intensity scaling controls (apply to subsequent plots) ---
    scale_mode = st.radio(
        "Intensity scaling",
        ["Linear", "Log (ln(1+v))", "Percentile clip"],
        horizontal=True,
        key="visit_scale"
    )
    clip_p = None
    if scale_mode == "Percentile clip":
        clip_p = st.slider("Upper percentile (clip)", 90, 100, 99, 1, key="visit_clip")

    def make_heatmap(mat: np.ndarray, title_suffix: str, note: str):
        raw = mat.astype(float)

        # --- Transform for display ---
        if scale_mode.startswith("Log"):
            mat_disp = np.log1p(raw)
            # Choose nice raw tick marks
            raw_ticks = np.array([0, 1, 5, 10, 25, 50, 100, 250, 500, 1000,
                                  2500, 5000, 10000, raw.max()]).astype(int)
            raw_ticks = raw_ticks[raw_ticks <= raw.max()]
            tickvals = np.log1p(raw_ticks)
            ticktext = [str(t) for t in raw_ticks]
            colorbar = dict(
                title="Visits",
                tickvals=tickvals,
                ticktext=ticktext
            )
            hover_tmpl = "(x=%{x}, y=%{y})<br>visits=%{customdata}<br>log1p=%{z:.2f}<extra></extra>"
            custom_data = raw
        elif scale_mode.startswith("Percentile"):
            upper = np.percentile(raw, clip_p) if raw.max() > 0 else 1
            mat_disp = np.clip(raw, 0, upper)
            colorbar = dict(title=f"Visits (≤p{clip_p} clipped)")
            hover_tmpl = "(x=%{x}, y=%{y})<br>visits=%{customdata}<extra></extra>"
            custom_data = raw
        else:  # Linear
            mat_disp = raw
            colorbar = dict(title="Visits")
            hover_tmpl = "(x=%{x}, y=%{y})<br>visits=%{z}<extra></extra>"
            custom_data = None  # z already raw

        vmax = max(1e-9, mat_disp.max())
        colorscale = [
            [0.0, "#f0f2f5"],
            [0.00001, "#e2e6ea"],
            [1.0, "#0d4d92"]
        ]

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
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
            hovertemplate=hover_tmpl
        ))

        # Maintain square cells
        aspect = grid_h / grid_w
        target_width_px = 1100
        fig_height = int(target_width_px * aspect)
        fig.update_layout(
            height=fig_height,
            margin=dict(l=0, r=0, t=60, b=0),
            title=f"State visitation counts — {title_suffix}",
            plot_bgcolor="#22262a",
            xaxis=dict(range=[-0.5, grid_w-0.5], dtick=1, showgrid=False, zeroline=False, constrain="domain"),
            yaxis=dict(range=[-0.5, grid_h-0.5], autorange="reversed", dtick=1, showgrid=False,
                       zeroline=False, scaleanchor="x", scaleratio=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        extra = f"{note} | scale={scale_mode}"
        if scale_mode.startswith('Percentile'):
            extra += f" (clip @ p{clip_p}={int(np.percentile(raw, clip_p))})"
        st.caption(extra)

    # ---- Helper: aggregate using paths.parquet (fast) ----
    def aggregate_from_paths(df: pd.DataFrame) -> np.ndarray:
        mat = np.zeros((grid_h, grid_w), dtype=np.int64)
        if df.empty: return mat
        g = df.groupby(["y", "x"]).size().reset_index(name="count")
        mat[g["y"].to_numpy(), g["x"].to_numpy()] = g["count"].to_numpy()
        return mat

    # ---- Helper: dynamic decode (slow fallback) ----
    def aggregate_dynamic(expert_subset: Optional[List[str]] = None) -> Tuple[np.ndarray, str]:
        note_parts = []
        mat = np.zeros((grid_h, grid_w), dtype=np.int64)
        experts_iter = expert_subset if expert_subset is not None else list(DATA.keys())
        for e in experts_iter:
            trajs = DATA[e]
            # find first non-empty trajectory to infer dims
            xd = yd = fd = None
            inf_note = ""
            for t in trajs:
                if t:
                    _, _, _, inf_note = infer_position_dims(t, grid_h, grid_w)
                    xd_i, yd_i, fd_i, _ = infer_position_dims(t, grid_h, grid_w)
                    xd, yd, fd = xd_i, yd_i, fd_i
                    break
            if xd is None and yd is None and fd is None:
                continue
            for t in trajs:
                for state in t:
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
                        mat[yy, xx] += 1
            note_parts.append(f"{e}: {inf_note}")
        return mat, "; ".join(note_parts) if note_parts else "no positions decoded"

    # ---------------- Scope logic ----------------
    if scope == "All experts":
        used = ""
        grid3d = None
        # Only trust precomputed NPZ if we actually have x/y indices now
        if x_idx >= 0 and y_idx >= 0:
            grid3d = load_visit_npz(helper_paths["visit_npz"])
        if grid3d is not None:
            mat = grid3d.sum(axis=2)
            used = "precomputed overall visitation_overall.npz"
        else:
            paths_file = Path(helper_paths["paths_parquet"])
            if x_idx >= 0 and y_idx >= 0 and paths_file.exists():
                df_all = load_paths(helper_paths["paths_parquet"])
                mat = aggregate_from_paths(df_all)
                used = "aggregated from paths.parquet"
            else:
                with st.spinner("Decoding all expert trajectories (no spatial helper files)…"):
                    mat, inf = aggregate_dynamic()
                used = f"dynamic decode ({inf})"
        make_heatmap(mat, "All experts (aggregated)", f"")
        # make_heatmap(mat, "All experts (aggregated)", f"Source: {used}")
    else:
        # Selected expert
        sel = sel_expert  # from earlier selection
        used = ""
        if x_idx >= 0 and y_idx >= 0:
            # Try helper paths first
            df_e = load_paths(helper_paths["paths_parquet"], sel)
            if not df_e.empty:
                mat = aggregate_from_paths(df_e)
                used = "paths.parquet"
            else:
                with st.spinner("Decoding selected expert trajectory positions…"):
                    mat, inf = aggregate_dynamic([sel])
                used = f"dynamic decode ({inf})"
        else:
            # No explicit indices -> dynamic
            with st.spinner("Decoding selected expert trajectory positions (no x/y indices)…"):
                mat, inf = aggregate_dynamic([sel])
            used = f"dynamic decode ({inf})"
        make_heatmap(mat, f"Expert {sel}", f"")
        # make_heatmap(mat, f"Expert {sel}", f"Source: {used}")

visitation_panel()

st.divider()

# ---------------------- ACTION SPACE (reference) ----------------------------------
st.header("Action space (semantic reference)")

def action_space_18() -> go.Figure:
    """
    Single figure illustrating 18 actions:
      - 9 spatial in current time window (filled markers)
      - Same 9 with time advance (hollow markers)
    Improvements:
      - Correct vertical orientation (up appears at top).
      - Current / next markers for the SAME spatial move share the same y (no stagger).
      - Only a slight horizontal offset differentiates current vs next window.
    """
    coords = [(dx, dy) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]  # logical coords (dy=-1 => up)

    def dir_label(dx, dy):
        if dx == 0 and dy == 0:
            return "stay"
        parts = []
        if dy == -1: parts.append("upper")
        if dy == 1:  parts.append("lower")
        if dx == -1: parts.append("left")
        if dx == 1:  parts.append("right")
        if len(parts) == 2:
            return f"{parts[0]}-{parts[1]}"
        return parts[0]

    # Flip vertical axis so dy=-1 (up) is rendered higher (positive y on screen)
    base_x = [dx for dx, _ in coords]
    base_y = [-dy for _, dy in coords]  # visual y

    # Current window (no time advance)
    xs_now = base_x
    ys_now = base_y

    # Next window (time advance) – horizontal offset only so pairs align vertically
    OFFSET = 0.08
    # OFFSET = 0.18
    xs_next = [x + (OFFSET if x <= 0 else -OFFSET) for x in base_x]
    ys_next = base_y  # no vertical offset

    # Hover texts
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

    # Current time actions (filled)
    fig.add_trace(go.Scatter(
        x=xs_now, y=ys_now,
        mode="markers+text",
        marker=dict(size=20, color="#1f77b4", line=dict(color="#0f3d66", width=1)),
        text=["S" if (dx == 0 and dy == 0) else "" for dx, dy in coords],
        textposition="middle center",
        name="Current window",
        hovertext=hover_now,
        hovertemplate="%{hovertext}<extra></extra>"
    ))

    # Next time actions (hollow, horizontally shifted)
    fig.add_trace(go.Scatter(
        x=xs_next, y=ys_next,
        mode="markers",
        marker=dict(size=16, symbol="circle-open", line=dict(color="#1f77b4", width=2)),
        name="Next window",
        hovertext=hover_next,
        hovertemplate="%{hovertext}<extra></extra>"
    ))

    # Direction arrows from center
    for dx, dy in coords:
        if dx == 0 and dy == 0:
            continue
        fig.add_annotation(
            x=dx * 0.7,
            y=(-dy) * 0.7,
            ax=0,
            ay=0,
            showarrow=True,             # <— required
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1.6,
            arrowcolor="#1f77b4",
            opacity=0.9
        )

    fig.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=50, b=0),
        title="18 actions: 9 spatial in current time window + same 9 with time advance",
        xaxis=dict(range=[-1.5, 1.5], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[-1.5, 1.5], showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.add_annotation(
        x=0, y=-1.35,
        text="Filled = current window (no time advance) · Hollow = next window (advance time) · Total = 18 actions",
        showarrow=False,
        font=dict(size=12, color="#bbbbbb")
    )
    return fig

st.plotly_chart(action_space_18(), use_container_width=True)
st.caption("Hover any marker for semantics. Center S = stay; neighbors = move to adjacent region. Each has both 'now' and 'next window' variants (18 total).")

st.caption("Helper files cached in ./derived/. Change schema or PKL to trigger a rebuild.")
