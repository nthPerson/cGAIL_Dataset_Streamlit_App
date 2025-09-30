from __future__ import annotations

import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DATA_DIR,
    DEFAULT_PKL,
    DERIVED_DIR,
    HELPER_PATHS,
    NORM_STATS_PATH,
    STATES_NPY,
    TRAJ_INDEX_PARQUET,
)

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - runtime fallback
    hf_hub_download = None


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_lengths(lengths_dir: str, expert: str) -> np.ndarray:
    """Load and cache per-trajectory lengths for the given expert."""
    path = Path(lengths_dir) / f"{expert}.npy"
    if not path.exists():
        return np.array([], dtype=np.int32)
    try:
        return np.load(path)
    except Exception:
        return np.array([], dtype=np.int32)


def ensure_hf_artifacts() -> None:
    """Ensure helper artifacts exist locally; download from Hugging Face if missing."""
    if STATES_NPY.exists() and TRAJ_INDEX_PARQUET.exists():
        return

    if hf_hub_download is None:
        st.error("huggingface_hub not installed and helper artifacts are missing.")
        st.stop()

    repo_id = os.environ.get("HF_REPO", "nthPerson/cGAIL-taxi-helper")
    revision = os.environ.get("HF_REVISION", "main")
    fname_states = os.environ.get("HF_FILE_STATES", "states_all.npy")
    fname_index = os.environ.get("HF_FILE_INDEX", "traj_index.parquet")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _fetch(fname: str, target: Path) -> None:
        if target.exists():
            return
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
            revision=revision,
            local_dir=str(DATA_DIR),
            local_dir_use_symlinks=False,
        )
        lp = Path(local_path)
        if lp == target or target.exists():
            return
        try:
            shutil.copy2(lp, target)
        except Exception as exc:
            try:
                with open(lp, "rb") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            except Exception as exc2:  # pragma: no cover - defensive
                raise RuntimeError(f"Failed to place {fname}: {exc} / {exc2}") from exc

    with st.spinner(f"Downloading helper artifacts from {repo_id} …"):
        try:
            _fetch(fname_states, STATES_NPY)
            _fetch(fname_index, TRAJ_INDEX_PARQUET)
        except Exception as exc:
            st.error(f"Download failed: {exc.__class__.__name__}: {exc}")
            st.stop()


def file_sig(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    stt = p.stat()
    return f"{stt.st_size}-{int(stt.st_mtime)}"


SIG = "helper-only"


def derived_meta_path() -> Path:
    return DERIVED_DIR / "derived_meta.json"


def read_meta() -> dict:
    if derived_meta_path().exists():
        return json.loads(derived_meta_path().read_text())
    return {}


def write_meta(meta: dict) -> None:
    derived_meta_path().write_text(json.dumps(meta, indent=2))


@st.cache_resource(show_spinner=True)
def load_pkl(path: str = DEFAULT_PKL) -> Dict[str, List[List[List[float]]]]:
    """Load the original pickle once per session; keep it out of cache keys elsewhere."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {str(k): v for k, v in raw.items()}


@st.cache_data(show_spinner=False)
def load_summary(path: str = HELPER_PATHS["summary_parquet"]) -> pd.DataFrame:
    cols = [
        "expert",
        "trajectories",
        "states_total",
        "states_avg",
        "states_min",
        "states_med",
        "states_max",
        "states_p90",
    ]
    summary_path = Path(path)
    if not summary_path.exists():
        return pd.DataFrame({c: [] for c in cols})
    try:
        df = pd.read_parquet(summary_path)
    except Exception:
        return pd.DataFrame({c: [] for c in cols})

    for c in cols:
        if c not in df.columns:
            if c == "expert":
                df[c] = []
            else:
                df[c] = pd.Series(dtype="float64")

    df["expert"] = df["expert"].astype(str)
    numeric = [c for c in cols if c != "expert"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[cols].sort_values("expert").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def get_states_matrix() -> np.ndarray:
    return np.load(str(STATES_NPY), mmap_mode="r")


@st.cache_data(show_spinner=False)
def get_traj_index_df() -> pd.DataFrame:
    return pd.read_parquet(str(TRAJ_INDEX_PARQUET))


@st.cache_resource(show_spinner=False)
def get_lookup() -> Dict[str, List[Tuple[int, int]]]:
    df = get_traj_index_df()
    lookup: Dict[str, List[Tuple[int, int]]] = {}
    for expert, sub in df.groupby("expert"):
        max_ti = int(sub["traj_idx"].max()) if len(sub) else -1
        arr: List[Tuple[int, int]] = [(0, 0)] * (max_ti + 1)
        for _, row in sub.iterrows():
            arr[int(row.traj_idx)] = (int(row.start), int(row.length))
        lookup[str(expert)] = arr
    return lookup


def get_state_vector(expert: str, traj_idx: int, state_idx: int) -> np.ndarray:
    lookup = get_lookup()
    if expert not in lookup:
        return np.array([])
    traj_list = lookup[expert]
    if traj_idx >= len(traj_list):
        return np.array([])
    start, length = traj_list[traj_idx]
    if state_idx < 0 or state_idx >= length:
        return np.array([])
    mat = get_states_matrix()
    return mat[start + state_idx]


def get_trajectory(expert: str, traj_idx: int) -> np.ndarray:
    lookup = get_lookup()
    if expert not in lookup:
        return np.empty((0, 0))
    traj_list = lookup[expert]
    if traj_idx >= len(traj_list):
        return np.empty((0, 0))
    start, length = traj_list[traj_idx]
    if length == 0:
        return np.empty((0, get_states_matrix().shape[1]))
    return get_states_matrix()[start:start + length]


def fetch_trajectory_for_infer(expert: str, traj_idx: int) -> List[List[float]]:
    arr = get_trajectory(expert, traj_idx)
    return arr.tolist()


@st.cache_data(show_spinner=False)
def load_visit_npz(path: str = HELPER_PATHS["visit_npz"]):
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = np.load(p)
        if isinstance(data, np.lib.npyio.NpzFile):
            for key in ("visits", "arr_0"):
                if key in data:
                    return data[key]
            first_key = list(data.keys())[0]
            return data[first_key]
        return data
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_paths(path: str = HELPER_PATHS["paths_parquet"], expert: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["expert", "traj_idx", "y", "x", "t"])
    try:
        df = pd.read_parquet(p)
    except Exception:
        return pd.DataFrame(columns=["expert", "traj_idx", "y", "x", "t"])

    for col in ["expert", "traj_idx", "y", "x"]:
        if col not in df.columns:
            return pd.DataFrame(columns=["expert", "traj_idx", "y", "x", "t"])

    if expert is not None:
        df = df[df["expert"].astype(str) == str(expert)]
    return df


def state_vec_from_pkl(data: Dict[str, Any], expert: str, traj_idx: int, state_idx: int) -> np.ndarray:
    raise RuntimeError("PKL mode disabled (helper-only deployment).")


@st.cache_resource(show_spinner=False)
def load_or_build_norm_stats() -> Dict[str, np.ndarray]:
    out_path = NORM_STATS_PATH
    if out_path.exists():
        try:
            data = np.load(out_path)
            required = ["mean", "std", "min", "max", "median", "mad"]
            if all(key in data for key in required):
                return {key: data[key] for key in required}
        except Exception:
            pass

    mat = get_states_matrix()
    mean = np.asarray(mat.mean(axis=0), dtype=float)
    var = np.asarray(((mat - mean) ** 2).mean(axis=0), dtype=float)
    std = np.sqrt(var)
    min_v = np.asarray(mat.min(axis=0), dtype=float)
    max_v = np.asarray(mat.max(axis=0), dtype=float)
    median = np.asarray(np.median(mat, axis=0), dtype=float)
    mad = np.asarray(np.median(np.abs(mat - median), axis=0), dtype=float)
    mad_scaled = mad * 1.4826
    try:
        np.savez_compressed(
            out_path,
            mean=mean,
            std=std,
            min=min_v,
            max=max_v,
            median=median,
            mad=mad_scaled,
        )
    except Exception:
        pass
    return {
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "median": median,
        "mad": mad_scaled,
    }
