"""
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
import json, os, pickle, math
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shutil

# Set page config FIRST (before any spinner / error / sidebar usage)
st.set_page_config(page_title="Shenzhen Taxi Dataset Explorer", layout="wide")

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
DATA_DIR = Path(__file__).parent / "data"
STATES_NPY = DATA_DIR / "states_all.npy"
TRAJ_INDEX_PARQUET = DATA_DIR / "traj_index.parquet"
DATA_DIR.mkdir(exist_ok=True)

# Helper‑only deployment: ignore PKL (env left for future)
DEFAULT_PKL = os.environ.get("DATA_PKL", str(DATA_DIR / "all_trajs.pkl"))

def ensure_hf_artifacts():
    """
    Ensure helper artifacts exist locally; download from Hugging Face if missing.
    Avoid cross-device rename by:
      - Requesting download directly into DATA_DIR (local_dir_use_symlinks=False)
      - Falling back to copy if the hub still returns a cache path.
    """
    if STATES_NPY.exists() and TRAJ_INDEX_PARQUET.exists():
        return
    if hf_hub_download is None:
        st.error("huggingface_hub not installed and helper artifacts are missing.")
        st.stop()

    repo_id = os.environ.get("HF_REPO", "nthPerson/cGAIL-taxi-helper")
    revision = os.environ.get("HF_REVISION", "main")
    fname_states = os.environ.get("HF_FILE_STATES", "states_all.npy")
    fname_index  = os.environ.get("HF_FILE_INDEX", "traj_index.parquet")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _fetch(fname: str, target: Path):
        if target.exists():
            return
        # Try to have HF copy the file directly into DATA_DIR (no symlink)
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
            revision=revision,
            local_dir=str(DATA_DIR),
            local_dir_use_symlinks=False
        )
        lp = Path(local_path)
        if lp == target or target.exists():
            return
        # Fallback: copy (NOT rename) to avoid cross-device link errors
        try:
            shutil.copy2(lp, target)
        except Exception as e:
            # Final fallback: streamed copy
            try:
                with open(lp, "rb") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            except Exception as e2:
                raise RuntimeError(f"Failed to place {fname}: {e} / {e2}")

    with st.spinner(f"Downloading helper artifacts from {repo_id} …"):
        try:
            _fetch(fname_states, STATES_NPY)
            _fetch(fname_index, TRAJ_INDEX_PARQUET)
        except Exception as e:
            st.error(f"Download failed: {e.__class__.__name__}: {e}")
            st.stop()

# Call before checking existence:
ensure_hf_artifacts()

# Decide mode AFTER potential download
HELPER_CORE_EXISTS = STATES_NPY.exists() and TRAJ_INDEX_PARQUET.exists()
DATA_MODE = "HELPER" if HELPER_CORE_EXISTS else "MISSING"
if DATA_MODE == "MISSING":
    st.error("Missing helper artifacts (states_all.npy + traj_index.parquet).")
    st.stop()

DERIVED_DIR = Path("./derived")  # helper files live here
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

GRID_H_DEFAULT, GRID_W_DEFAULT, T_SLOTS_DEFAULT = 40, 50, 288  # 40x50 grid, 288 five-minute slots/day

TRAFFIC_NEIGHBORHOOD = 5
TRAFFIC_FEATURES = ["Traffic Speed", "Traffic Volume", "Traffic Demand", "Traffic Waiting"]
TRAFFIC_COUNT_DEFAULT = (TRAFFIC_NEIGHBORHOOD ** 2) * len(TRAFFIC_FEATURES)  # 100

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

st.sidebar.markdown(f"""## **Selectors drive:**
- **State details** (state vector and traffic features)
- **Trajectory path**
- **State visitation counts** (Selected expert only).""")

# ----------------------------- UTILS --------------------------------------------
def file_sig(path: str) -> str:
    # In helper-only mode this is unused; retained for compatibility.
    p = Path(path)
    if not p.exists(): return "missing"
    stt = p.stat()
    return f"{stt.st_size}-{int(stt.st_mtime)}"

SIG = "helper-only"

# (meta helpers unchanged)
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


# ----------------------- HELPER FILE BUILDER -------------------------------------
# We skip ensure_helpers building from PKL (no PKL). Reconstruct helper_paths directly.
summary_parquet = DERIVED_DIR / "experts_summary.parquet"
helper_paths = dict(
    summary_parquet=str(summary_parquet),
    lengths_dir=str(DERIVED_DIR / "lengths_by_expert"),
    paths_parquet=str(DERIVED_DIR / "paths.parquet"),
    visit_npz=str(DERIVED_DIR / "visitation_overall.npz"),
    # meta=read_meta()
)

@st.cache_data(show_spinner=False)
def load_summary(path: str) -> pd.DataFrame:
    """
    Load experts summary parquet (built offline) with graceful fallback.
    Ensures required columns exist even if file missing or partly corrupt.
    """
    cols = [
        "expert",
        "trajectories",
        "states_total",
        "states_avg",
        "states_min",
        "states_med",
        "states_max",
        "states_p90"
    ]
    if not Path(path).exists():
        return pd.DataFrame({c: [] for c in cols})
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame({c: [] for c in cols})

    # Add any missing columns with defaults
    for c in cols:
        if c not in df.columns:
            if c == "expert":
                df[c] = []
            else:
                df[c] = pd.Series(dtype="float64")

    # Normalize dtypes
    df["expert"] = df["expert"].astype(str)
    numeric = [c for c in cols if c != "expert"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[cols].sort_values("expert").reset_index(drop=True)

# Load summary
summary = load_summary(helper_paths["summary_parquet"])

# ----------------------- LOAD CORE MATRICES --------------------------------------
@st.cache_resource(show_spinner=False)
def load_states_matrix(path: str):
    return np.load(path, mmap_mode="r")

@st.cache_data(show_spinner=False)
def load_traj_index_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

STATES_MATRIX = load_states_matrix(str(STATES_NPY))
TRAJ_INDEX = load_traj_index_df(str(TRAJ_INDEX_PARQUET))

# Build lookup: expert -> list[(start,length)]
lookup: Dict[str, List[Tuple[int,int]]] = {}
for e, sub in TRAJ_INDEX.groupby("expert"):
    max_ti = int(sub["traj_idx"].max()) if len(sub) else -1
    arr = [(0,0)]*(max_ti+1)
    for _, row in sub.iterrows():
        arr[int(row.traj_idx)] = (int(row.start), int(row.length))
    lookup[str(e)] = arr

def state_vec_from_pkl(DATA, expert: str, traj_idx: int, state_idx: int) -> np.ndarray:
    """Legacy stub retained for compatibility."""
    raise RuntimeError("PKL mode disabled (helper-only deployment).")

# ---------------------- OPTIONAL HELPER LOADERS (SAFE STUBS) --------------------
@st.cache_data(show_spinner=False)
def load_visit_npz(path: str):
    """Load visitation_overall.npz if present; else return None.
    Returned array shape expected: (grid_h, grid_w, t_slots).
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = np.load(p)
        # Accept either direct array or stored under a key
        if isinstance(data, np.lib.npyio.NpzFile):
            # Prefer a common key name if present
            for k in ("visits", "arr_0"):
                if k in data:
                    return data[k]
            # Fallback: first item
            first_key = list(data.keys())[0]
            return data[first_key]
        return data
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_paths(path: str, expert: Optional[str] = None) -> pd.DataFrame:
    """Load paths.parquet if present. Optionally filter by expert.
    Expected columns: expert, traj_idx, y, x, (optional t)
    Returns empty DataFrame if unavailable or corrupt.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["expert", "traj_idx", "y", "x", "t"])  # schema placeholder
    try:
        df = pd.read_parquet(p)
    except Exception:
        return pd.DataFrame(columns=["expert", "traj_idx", "y", "x", "t"])  # safe fallback
    # Ensure required cols
    for c in ["expert", "traj_idx", "y", "x"]:
        if c not in df.columns:
            return pd.DataFrame(columns=["expert", "traj_idx", "y", "x", "t"])
    if expert is not None:
        df = df[df["expert"].astype(str) == str(expert)]
    return df

def get_state_vector(expert: str, traj_idx: int, state_idx: int) -> np.ndarray:
    if expert not in lookup: return np.array([])
    traj_list = lookup[expert]
    if traj_idx >= len(traj_list): return np.array([])
    start, length = traj_list[traj_idx]
    if state_idx < 0 or state_idx >= length: return np.array([])
    return STATES_MATRIX[start + state_idx]

def get_trajectory(expert: str, traj_idx: int) -> np.ndarray:
    if expert not in lookup: return np.empty((0,0))
    traj_list = lookup[expert]
    if traj_idx >= len(traj_list): return np.empty((0,0))
    start, length = traj_list[traj_idx]
    if length == 0: return np.empty((0, STATES_MATRIX.shape[1]))
    return STATES_MATRIX[start:start+length]

def fetch_trajectory_for_infer(expert: str, traj_idx: int) -> List[List[float]]:
    arr = get_trajectory(expert, traj_idx)
    return arr.tolist()

def reshape_traffic_maps(vec: np.ndarray,
                         start: int,
                         traffic_len: int,
                         n: int = 5,
                         feature_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Slice traffic neighborhood features from a state vector and reshape into
    per-feature (n x n) grids.
    """
    feature_names = feature_names or [f"feat{i}" for i in range(traffic_len // (n*n))]
    needed = n * n * len(feature_names)
    out: Dict[str, np.ndarray] = {}
    if vec is None or len(vec) < start + needed:
        for name in feature_names:
            out[name] = np.full((n, n), np.nan, dtype=float)
        return out
    block = vec[start:start+needed]
    for i, name in enumerate(feature_names):
        sub = block[i*n*n:(i+1)*n*n]
        out[name] = np.asarray(sub, dtype=float).reshape(n, n)
    return out

# ================================================================================
#                                APP BODY
# ================================================================================

st.title("cGAIL Dataset Explorer — Shenzhen Taxi Dataset")

# ---------------------- DATA SUMMARY METRICS -------------------------------------
first_state_vec = STATES_MATRIX[0] if STATES_MATRIX.shape[0] else np.array([])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total experts", f"{len(summary):,}")
c2.metric("Total trajectories", f"{int(summary['trajectories'].sum()):,}")
c3.metric("Total states visited in dataset", f"{int(summary['states_total'].sum()):,}")
c4.metric("Features per state", f"{len(first_state_vec)}")
# c5.metric("Total data points in datasaet", f"{int(summary['states_total'].sum() * len(first_state_vec)):,}")

with st.expander("Model summary & data flow", expanded=False):
    st.markdown(f"""
- **State space**: **{grid_h} × {grid_w}** spatial grid × **{t_slots}** five-minute time slots per day.
- **State features** (example schema used here):
  - Traffic (5×5 neighborhood × 4 features = 100 dims): {', '.join(TRAFFIC_FEATURES)}
  - POI distances: {poi_count} dims
  - Temporal: remaining dims (time of day, day of week, etc.)
- **Condition features**: Home location distance, working schedule, familiarity (avg visitations to current state), and loction identifier.
- **Action space**: **{ '18 — 9 spatial with/without time advance' if action_mode.startswith('18') else '9 (paper) — stay or move to 8 neighbors' }**.
- **Raw → model flow**:
  1. Raw CSV → (plate_id, lat, lon, timestamp, passenger flag)
  2. Segment into **vacant trajectories** (drop-off → next pick-up)
  3. Map GPS to **grid & time slots**, build per-state feature vectors
  4. Train IL (e.g., cGAIL) conditioned on driver/location
""")

st.divider()

# ---------------------- HIERARCHY: EXPERTS → DISTRIBUTION (4) ---------------------
st.header("Dataset Hierarchy: Experts → Trajetories → States")

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
    st.subheader("Experts (larger area indicates higher trajectory count)")
    # Render treemap
    st.plotly_chart(treemap, use_container_width=True)

experts_treemap_panel()

st.subheader(f"Trajectory length distribution for expert {sel_expert} (selected in sidebar)")

# Use sidebar-selected expert (sel_expert) instead of local selectbox
lens = load_lengths(helper_paths["lengths_dir"], sel_expert)

nbins = min(60, max(10, int(math.sqrt(max(1, lens.size)))))
hist = go.Figure(data=[go.Histogram(x=lens, nbinsx=nbins)])
hist.update_layout(
    height=300,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis_title=f"Trajectory length (# states) — Expert {sel_expert}",
    yaxis_title="Count"
)
st.plotly_chart(hist, use_container_width=True)

cA, cB, cC = st.columns(3)
if lens.size:
    cA.metric("# Traj", f"{lens.size:,}")
    cB.metric("Mean", f"{lens.mean():.1f}")
    cC.metric("Min/Max", f"{lens.min():.0f}/{lens.max():.0f}")
else:
    cA.metric("# Traj", "0"); cB.metric("Mean","-"); cC.metric("Min/Max","-")

st.divider()

# ---------------------- STATE VECTOR (5) ---------------------------------
st.header("State-Level visualizations")

# We now use sidebar selections (sel_expert, sel_traj_idx, sel_state_idx)
lens_e = load_lengths(helper_paths["lengths_dir"], sel_expert)
traj_len = int(lens_e[sel_traj_idx]) if lens_e.size else 0

metric_cols = st.columns(3)
metric_cols[0].metric(f"Expert", f"{sel_expert}")
metric_cols[1].metric(f"Trajectories recorded for expert {sel_expert}", f"{lens_e.size:,}")
metric_cols[2].metric(f"States visited in trajectory {sel_traj_idx}", f"{traj_len:,}")

# fetch vector directly
if traj_len == 0:
    vec = np.array([])
else:
    vec = get_state_vector(sel_expert, sel_traj_idx, sel_state_idx)

st.markdown(f"#### **State vector (trajectory {sel_traj_idx}, state {sel_state_idx}) — Raw (unnormalized)**")
feat_df = pd.DataFrame({"dim": [f"f{i}" for i in range(len(vec))], "value": vec})
bar_raw = px.bar(feat_df, x="dim", y="value")
bar_raw.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=20), xaxis=dict(showticklabels=False))
st.plotly_chart(bar_raw, use_container_width=True)

# --- Persistent normalization stats (mean, std, min, max, median, mad) ---------
NORM_STATS_PATH = DERIVED_DIR / "norm_stats.npz"

@st.cache_resource(show_spinner=False)
def load_or_build_norm_stats(matrix_path: str, out_path: Path):
    """Load normalization stats from disk or compute & persist them.
    Stats:
      mean, std (population), min, max, median, mad (median absolute deviation, scaled ~1.4826 for normal consistency)
    Returns dict of numpy arrays.
    """
    if out_path.exists():
        try:
            data = np.load(out_path)
            required = ["mean", "std", "min", "max", "median", "mad"]
            if all(k in data for k in required):
                return {k: data[k] for k in required}
        except Exception:
            pass  # fall through to rebuild
    mat = load_states_matrix(matrix_path)
    # Use float64 for stability
    mean = np.asarray(mat.mean(axis=0), dtype=float)
    var = np.asarray(((mat - mean) ** 2).mean(axis=0), dtype=float)
    std = np.sqrt(var)
    min_v = np.asarray(mat.min(axis=0), dtype=float)
    max_v = np.asarray(mat.max(axis=0), dtype=float)
    median = np.asarray(np.median(mat, axis=0), dtype=float)
    mad = np.asarray(np.median(np.abs(mat - median), axis=0), dtype=float)
    # Scale MAD to approximate std under normality
    mad_scaled = mad * 1.4826
    try:
        np.savez_compressed(out_path, mean=mean, std=std, min=min_v, max=max_v, median=median, mad=mad_scaled)
    except Exception:
        pass  # non-fatal
    return {"mean": mean, "std": std, "min": min_v, "max": max_v, "median": median, "mad": mad_scaled}

show_norm = st.checkbox("Show normalized view", value=False, help="Toggle to compute & display normalized state vector (cached stats).")
if show_norm and vec.size:
    stats = load_or_build_norm_stats(str(STATES_NPY), NORM_STATS_PATH)
    if stats["mean"].shape[0] == vec.shape[0]:
        scale_mode = st.radio(
            "Normalization scaling",
            ["Z-score (mean/std)", "Min-Max [0,1]", "Robust (median/MAD)"],
            horizontal=True,
            key="norm_scale_mode"
        )
        mean_all = stats["mean"]
        std_all = stats["std"]
        min_all = stats["min"]
        max_all = stats["max"]
        median_all = stats["median"]
        mad_all = stats["mad"]
        with np.errstate(divide='ignore', invalid='ignore'):
            if scale_mode.startswith("Z-score"):
                values = np.where(std_all > 0, (vec - mean_all) / std_all, 0.0)
                y_title = "Z-score"
            elif scale_mode.startswith("Min-Max"):
                denom = (max_all - min_all)
                values = np.where(denom > 0, (vec - min_all) / denom, 0.0)
                y_title = "Min-Max scaled"
            else:  # Robust
                values = np.where(mad_all > 0, (vec - median_all) / mad_all, 0.0)
                y_title = "(value - median)/MAD"
        norm_df = pd.DataFrame({"dim": [f"f{i}" for i in range(len(values))], "value": values})
        bar_norm = px.bar(norm_df, x="dim", y="value")
        bar_norm.update_layout(
            height=260,
            margin=dict(l=0,r=0,t=10,b=20),
            xaxis=dict(showticklabels=False),
            yaxis=dict(title=y_title)
        )
        st.markdown(f"**State vector (normalized — {y_title})**")
        st.plotly_chart(bar_norm, use_container_width=True)
        st.caption("Stats persisted to norm_stats.npz (reused across runs). Constant features get 0 in all modes.")
    else:
        st.warning("Normalization skipped: feature count mismatch with stored stats.")
elif show_norm and not vec.size:
    st.info("No state vector available for normalization.")

# ---------------------- TRAFFIC NEIGHBORHOOD (5×5 × 4 features) ---------------------------------

st.markdown("#### **Traffic neighborhood for current state (5×5 × 4 features)**")

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

# ---------------------- TRAJECTORY PATH SECTION ----------------------------------
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
    traj_arr = get_trajectory(expert, traj_idx)
    if traj_arr.size == 0:
        st.info("Trajectory not available or empty.")
        return
    traj = traj_arr.tolist()

    # Decide decoding strategy
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
            arrowhead=4,                # style of head
            arrowsize=1.8,              # (was 1) scale factor for head size
            arrowwidth=1.5,             # (was 1.3) shaft width
            opacity=0.85,               # increase visibility
            arrowcolor="#17324f"       # slightly darker for contrast
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
        mat = np.zeros((grid_h, grid_w), dtype=np.int64)
        note_parts = []
        experts_iter = expert_subset if expert_subset is not None else list(lookup.keys())
        for e in experts_iter:
            traj_list = lookup[e]
            # infer dims from first non-empty trajectory
            xd = yd = fd = None
            for start, length in traj_list:
                if length > 0:
                    arr = STATES_MATRIX[start:start+length]
                    xd, yd, fd, inf_note = infer_position_dims(arr, grid_h, grid_w)
                    note_parts.append(f"{e}:{inf_note}")
                    break
            if xd is None and yd is None and fd is None:
                continue
            # accumulate
            for start, length in traj_list:
                if length == 0: continue
                arr = STATES_MATRIX[start:start+length]
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
                with st.spinner("Decoding all expert trajectories…"):
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
    # Logical coordinate offsets relative to center (0,0) then mapped to grid indices (col,row) in a 3x3 block.
    coords = [(dx, dy) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]  # dy=-1 => up

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

    # Map to grid indices in 0..2 (center cell (1,1)) ; row index increases downward
    base_x = [dx + 1 for dx, _ in coords]        # 0,1,2 columns
    base_y = [(-dy) + 1 for _, dy in coords]     # invert dy then shift to 0..2

    # Current window (no time advance) marker positions
    xs_now = base_x
    ys_now = base_y

    # Next window (time advance) – place markers OUTSIDE and tangent to current markers along the ray from center (1,1)
    # geometry: center (1,1); target cell center (cx,cy). Current marker radius r_now ~ (marker size 20) ~ visual ~0.18 cell units.
    # We push next marker just beyond current marker: displacement factor slightly > 1.
    r_now = 0.0  # we will use multiplicative scaling instead of fixed radius due to pixel-to-data mapping variability
    scale_outer = 1.32  # >1 moves outside cell center along ray; tuned so marker edge sits just beyond current filled circle
    xs_next, ys_next = [], []
    for cx, cy, (dx, dy) in zip(base_x, base_y, coords):
        if dx == 0 and dy == 0:
            # Stay action: keep hollow marker slightly to the right of center for pair clarity
            xs_next.append(cx + 0.28)
            ys_next.append(cy)
            continue
        # direction vector from center to this cell center
        vx = cx - 1
        vy = cy - 1
        xs_next.append(1 + vx * scale_outer)
        ys_next.append(1 + vy * scale_outer)

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

    # Add a 3x3 heatmap as white cells with subtle border coloring to visually match other grid-based panels
    heat = go.Heatmap(
        z=np.zeros((3,3)),
        x=[0,1,2], y=[0,1,2],
        showscale=False,
        zmin=0, zmax=1,
        colorscale=[[0,"#ffffff"],[1,"#ffffff"]],
        xgap=2, ygap=2,
        hoverinfo="skip"
    )
    fig.add_trace(heat)

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

    # Next time actions (hollow, offset outward)
    fig.add_trace(go.Scatter(
        x=xs_next, y=ys_next,
        mode="markers",
        marker=dict(size=16, symbol="circle-open", line=dict(color="#1f77b4", width=2)),
        name="Next window",
        hovertext=hover_next,
        hovertemplate="%{hovertext}<extra></extra>"
    ))

    # Direction arrows from center cell (1,1) to neighbors
    for (dx, dy), x_cell, y_cell in zip(coords, base_x, base_y):
        if dx == 0 and dy == 0:
            continue
        fig.add_annotation(
            x=x_cell, y=y_cell,
            ax=1, ay=1,                # center (1,1)
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=0,
            arrowsize=2.0,
            arrowwidth=2.0,
            arrowcolor="#1f77b4",
            opacity=0.9
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
            ticks=""
        ),
        yaxis=dict(
            range=[-0.5, 2.5],
            dtick=1,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            autorange="reversed",  # align with other heatmaps (row 0 at top visually)
            showticklabels=False,
            ticks=""
        ),
        legend=dict(orientation="h", y=1.02, x=0)
    )
    fig.add_annotation(
        x=1, y=3.05,
        text="Filled = current time window · Hollow = advance time · Cells align with other grid visuals",
        showarrow=False,
        font=dict(size=12, color="#bbbbbb")
    )
    return fig

st.plotly_chart(action_space_18(), use_container_width=True)
st.caption("Hover any marker for semantics. Center S = stay; neighbors = move to adjacent region. Each has both 'now' and 'next window' variants (18 total).")

st.markdown(f'Created by Dr. Xin Zhang and Robert Ashe at San Diego State University')