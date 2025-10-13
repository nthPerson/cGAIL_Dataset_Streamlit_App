# Shenzhen Taxi Expert Explorer — Streamlit Application

An interactive Streamlit app that lets you explore Shenzhen taxi imitation-learning data in a modular, cache-friendly way. The UI highlights experts (drivers), their trajectories, and per-state feature vectors derived from the cGAIL training pipeline.

---

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the app from this directory:
   ```bash
   streamlit run app.py
   ```
3. On first run the app will download lightweight helper artifacts from Hugging Face (`nthPerson/cGAIL-taxi-helper`). Subsequent runs reuse the local cache located under `data/` and `derived/`.

Optional: regenerate derived helper files (length histograms, normalization stats) without launching the UI by running:
```bash
python build_helper_artifacts.py
```

---

## Modular Layout

```
streamlit/
├─ app.py                      # Orchestrates boot sequence and view ordering
├─ config.py                   # Grid/feature constants, paths, HF settings
├─ data_access.py              # Cached loaders, HF downloads, helper builders
├─ sidebar.py                  # Streamlit sidebar and selection dataclass
├─ views/                      # Independent Streamlit fragments for each visual
│  ├─ overview.py             # Dataset KPIs and summary
│  ├─ hierarchy.py            # Expert treemap and trajectory histograms
│  ├─ state_level.py          # Individual state feature analysis
│  ├─ trajectory.py           # Path visualization on grid
│  ├─ visitation.py           # Heatmaps of state visitation
│  ├─ diagnostics.py          # Dimension analysis tools
│  └─ action_space.py         # (Deprecated) Action glyph reference
├─ build_helper_artifacts.py  # Helper file regeneration script
├─ app_data/                   # Core data files (states, indices, CSVs)
│  └─ README_DATA_SOURCES.md  # Complete data catalog
├─ derived/                    # Generated helpers (summaries, stats, caches)
└─ ai_context_docs/           # Documentation for AI agents
   ├─ PROJECT_OVERVIEW.md     # Architecture and quick start
   ├─ DATA_MODEL.md           # State vector and grid specification
   ├─ VISUALS_SPEC.md         # Per-panel acceptance criteria
   ├─ DEV_GUIDE.md            # Caching, fragments, performance
   └─ archive/                # Historical docs and fixes
```

Key ideas:

- `app.py` executes the startup checklist (artifact availability, derived helper sync) and then calls each view module in sequence.
- Each file in `views/` exposes a `render(selection, cache, ...)` function wrapped in `@st.fragment`, isolating reruns to the panel the user interacts with.
- `sidebar.py` centralizes control widgets so all panels consume a single `SidebarSelection` object.
- `data_access.py` provides cached readers for NumPy/Parquet artifacts, visitation aggregation, and normalization stats used across multiple panels.

---

## Visual Panels

| Panel | Module | What it shows |
|-------|--------|---------------|
| KPIs & narrative summary | `views/overview.py` | Total experts, trajectories, states, feature dimensionality, and a raw→model explainer. |
| Expert hierarchy | `views/hierarchy.py` | Treemap of experts (area by trajectory count) plus a trajectory-length histogram with derived metrics. |
| State vector & traffic maps | `views/state_level.py` | Raw and normalized feature magnitudes for the selected state, alongside 5×5×4 traffic heatmaps. |
| Trajectory path | `views/trajectory.py` | Per-trajectory grid visualization with arrows, start/end/current markers, and dynamic `(x, y)` inference when needed. |
| Visitation heatmaps | `views/visitation.py` | Aggregate cell visitation for all experts and the selected expert, with linear/log/percentile scaling controls. |
| Action space glyph | `views/action_space.py` | Unified paired-marker glyph describing the 18 discrete actions (stay/move × advance time). |

All panels react to the shared sidebar controls (expert, trajectory index, state index) without forcing unrelated panels to rerun.

---

## Data Sources

### Core Helper Artifacts (`app_data/`)

| File | Description |
|------|-------------|
| `states_all.npy` | Concatenated state feature matrix (666,729 states × 126 dims). |
| `traj_index.parquet` | Mapping from `(expert, traj_idx)` to `(start, length)` offsets into `states_all.npy`. |
| `grid_to_district_ArcGIS_table.csv` | Maps 4,500 grid cells to Shenzhen districts with overlap metrics. |
| `district_id_mapping.csv` | District name to integer ID lookup. |
| Various socioeconomic CSVs | Housing prices, GDP, population, employment by district. |

**Note**: `states_all.npy` and `traj_index.parquet` are pulled from Hugging Face if missing. Downloads use `local_dir_use_symlinks=False` to avoid cross-device rename issues.

See `app_data/README_DATA_SOURCES.md` for complete data catalog.

### Derived Helpers (`derived/`)

| File | Description |
|------|-------------|
| `experts_summary.parquet` | Per-expert aggregates (counts, state metrics) feeding KPI and hierarchy views. |
| `lengths_by_expert/{expert}.npy` | Cached trajectory lengths for histogram and metrics. |
| `norm_stats.npz` | Persisted per-feature statistics (mean, std, min, max, median, MAD) powering the normalized state view. |
| `derived_meta.json` | Dataset signature, grid dimensions, and helper build metadata. |
| *(optional)* `paths.parquet`, `visitation_overall.npz` | Only present when indices are known; the app falls back to dynamic inference otherwise. |

Derived helpers are rebuilt automatically when signatures drift, or manually via `build_helper_artifacts.py`.

---

## Documentation

All documentation is organized in `ai_context_docs/`:

- **PROJECT_OVERVIEW.md**: Architecture, quick start, recent updates, future plans
- **DATA_MODEL.md**: Grid system, state vector schema, data files, integration patterns
- **VISUALS_SPEC.md**: Per-panel inputs, interactions, and acceptance criteria
- **DEV_GUIDE.md**: Caching rules, fragments, performance targets, debugging tips
- **archive/**: Historical documentation (coordinate fixes, dimension analysis, update logs)

For complete data source details, see `app_data/README_DATA_SOURCES.md`.

---

## Recent Updates

### 2025-10-12: Documentation Consolidation
- Moved historical docs to `ai_context_docs/archive/`
- Created `PROJECT_OVERVIEW.md` for comprehensive project context
- Renamed data README to `README_DATA_SOURCES.md`
- Added grid-to-district mapping files
- Streamlined AI agent documentation

### 2025-10-08: Grid Dimension & Coordinate Fix
- Updated grid size from 40×50 to 50×90
- Fixed coordinate swap (dim 0=Y, dim 1=X)
- Deprecated traffic neighborhood and action space visualizations
- Clarified only dimensions 0-3 are active

---

## Troubleshooting Tips

- If first-run downloads fail, delete `data/` and retry after verifying network access.
- When adding new helper artifacts, update `config.py` paths and register new cached loaders inside `data_access.py`.
- Use `python -m compileall .` or `streamlit docs` `--server.runOnSave true` while iterating to catch syntax/import errors quickly.

---

Happy exploring!
