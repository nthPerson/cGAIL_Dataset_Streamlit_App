from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATES_NPY = DATA_DIR / "states_all.npy"
TRAJ_INDEX_PARQUET = DATA_DIR / "traj_index.parquet"
DEFAULT_PKL = os.environ.get("DATA_PKL", str(DATA_DIR / "all_trajs.pkl"))

DERIVED_DIR = BASE_DIR / "derived"
LENGTHS_DIR = DERIVED_DIR / "lengths_by_expert"
PATHS_PARQUET = DERIVED_DIR / "paths.parquet"
VISITATION_NPZ = DERIVED_DIR / "visitation_overall.npz"
SUMMARY_PARQUET = DERIVED_DIR / "experts_summary.parquet"
NORM_STATS_PATH = DERIVED_DIR / "norm_stats.npz"

DATA_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR.mkdir(parents=True, exist_ok=True)
LENGTHS_DIR.mkdir(parents=True, exist_ok=True)

HELPER_PATHS: Dict[str, str] = {
    "summary_parquet": str(SUMMARY_PARQUET),
    "lengths_dir": str(LENGTHS_DIR),
    "paths_parquet": str(PATHS_PARQUET),
    "visit_npz": str(VISITATION_NPZ),
}

# ---------------------------------------------------------------------------
# Grid / feature configuration
# ---------------------------------------------------------------------------
GRID_H_DEFAULT = 40
GRID_W_DEFAULT = 50
T_SLOTS_DEFAULT = 288

TRAFFIC_NEIGHBORHOOD = 5
TRAFFIC_FEATURES = [
    "Traffic Speed",
    "Traffic Volume",
    "Traffic Demand",
    "Traffic Waiting",
]
TRAFFIC_COUNT_DEFAULT = (TRAFFIC_NEIGHBORHOOD ** 2) * len(TRAFFIC_FEATURES)

TRAFFIC_START = 0
POI_COUNT = 23
POI_START = TRAFFIC_START + TRAFFIC_COUNT_DEFAULT
TEMPORAL_START = POI_START + POI_COUNT

ACTION_MODE = "18 actions (extended)"
