from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "app_data"
# DATA_DIR = BASE_DIR / "data"
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
# Grid / feature configuration — UPDATED 2025-10-08
# ---------------------------------------------------------------------------
# Grid dimensions confirmed by team discussion
GRID_H_DEFAULT = 50  # Height (rows) - Y dimension (updated from 40)
GRID_W_DEFAULT = 90  # Width (columns) - X dimension (updated from 50)
T_SLOTS_DEFAULT = 288  # Time slots per day (5-minute intervals)

# DEPRECATED: Traffic neighborhood features no longer used
# These settings preserved for reference but visualization is disabled
TRAFFIC_NEIGHBORHOOD = 5
TRAFFIC_FEATURES = [
    "Traffic Speed",
    "Traffic Volume",
    "Traffic Demand",
    "Traffic Waiting",
]
TRAFFIC_COUNT_DEFAULT = (TRAFFIC_NEIGHBORHOOD ** 2) * len(TRAFFIC_FEATURES)

# DEPRECATED: Old feature indices (preserved for reference)
TRAFFIC_START = 0
POI_COUNT = 23
POI_START = TRAFFIC_START + TRAFFIC_COUNT_DEFAULT
TEMPORAL_START = POI_START + POI_COUNT

# DEPRECATED: Action space visualization no longer used
ACTION_MODE = "18 actions (extended)"

# ---------------------------------------------------------------------------
# Feature indices — UPDATED 2025-10-08 based on team confirmation
# ---------------------------------------------------------------------------
# CONFIRMED: Only dimensions 0-3 are used from state vector (126 total dims)
# - Dimension 0: Y coordinate (grid row) [0, 49] — CORRECTED 2025-10-08
# - Dimension 1: X coordinate (grid column) [0, 89] — CORRECTED 2025-10-08
# - Dimension 2: Temporal feature 1 (time slot or time-of-day)
# - Dimension 3: Temporal feature 2 (day-of-week or secondary temporal)
# - Dimensions 4-124: NOT USED (alternate data sources replace these)
# - Dimension 125: Action label [0, 18] (expert's chosen action)

# IMPORTANT: Data analysis revealed dimensions were swapped from initial assumption
# Empirical ranges: dim0=[3,48] (50-row span), dim1=[1,81] (90-column span)
EXPECTED_Y_IDX = 0  # Grid row (0-based, range [0, 49]) — dimension 0
EXPECTED_X_IDX = 1  # Grid column (0-based, range [0, 89]) — dimension 1
EXPECTED_T_IDX = 2  # Primary temporal feature
EXPECTED_T2_IDX = 3  # Secondary temporal feature

# Action dimension (expert label, not a state feature)
ACTION_DIM_IDX = 125  # Range: [0, 18] - discrete action label

# ---------------------------------------------------------------------------
# Alternative data sources (replace dims 4-124)
# ---------------------------------------------------------------------------
# Traffic and demand features now sourced from separate pickle files:
# 
# 1. latest_volume_pickups.pkl
#    Structure: (x, y, time-of-day, day-of-week) -> traffic volume & pickup counts
#    Usage: Real-time demand visualization, pickup hotspot analysis
#
# 2. latest_traffic.pkl  
#    Structure: (x, y, time-of-day, day-of-week) -> traffic characteristics
#    Usage: Traffic pattern visualization (exact features TBD)
#
# 3. District mapping CSV (50x90 grid -> Shenzhen districts)
#    Usage: Apply district-level demographic data to grid cells

# Paths to alternate data sources
LATEST_VOLUME_PICKUPS_PKL = DATA_DIR / "latest_volume_pickups.pkl"
LATEST_TRAFFIC_PKL = DATA_DIR / "latest_traffic.pkl"
DISTRICT_MAPPING_CSV = DATA_DIR / "grid_to_district_mapping.csv"  # Expected path

# DEPRECATED: Legacy feature indices (preserved for reference only)
# These were based on incorrect interpretation of state vector structure
TRAFFIC_START_EMPIRICAL = 4  # No longer used
TRAFFIC_LEN_EMPIRICAL = 96   # No longer used
VOL_PICKUP_START = 104       # No longer used
VOL_PICKUP_LEN = 2           # No longer used
DEMAND_WAITING_START = 106   # No longer used
DEMAND_WAITING_LEN = 20      # No longer used

