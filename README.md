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
├─ app.py            # Orchestrates boot sequence and view ordering
├─ config.py         # Dataset signature, feature constants, Hugging Face settings
├─ data_access.py    # Cached loaders, HF downloads, derived helper builders
├─ sidebar.py        # Streamlit sidebar and selection dataclass
├─ views/            # Independent Streamlit fragments for each visual
│  ├─ overview.py
│  ├─ hierarchy.py
│  ├─ state_level.py
│  ├─ trajectory.py
│  ├─ visitation.py
│  └─ action_space.py
├─ build_helper_artifacts.py
├─ data/             # Core helper artifacts (downloaded if missing)
└─ derived/          # Locally generated helper files
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

### Core Helper Artifacts (`data/`)

| File | Description |
|------|-------------|
| `states_all.npy` | Concatenated state feature matrix (≈126 dims each). |
| `traj_index.parquet` | Mapping from `(expert, traj_idx)` to `(start, length)` offsets into `states_all.npy`. |

These files are pulled from Hugging Face if missing. Downloads use `local_dir_use_symlinks=False` to avoid cross-device rename issues.

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

## Data Model & Visual Specs

- **Data schema:** See `ai_context_docs/DATA_MODEL.md` for detailed feature indices, grid/time configuration, and helper pipeline notes.
- **Visual contracts:** See `ai_context_docs/VISUALS_SPEC.md` for per-panel inputs, interactions, and acceptance criteria.
- **Developer details:** `ai_context_docs/DEV_GUIDE.md` covers caching rules, fragments, and performance targets.

---

## Troubleshooting Tips

- If first-run downloads fail, delete `data/` and retry after verifying network access.
- When adding new helper artifacts, update `config.py` paths and register new cached loaders inside `data_access.py`.
- Use `python -m compileall .` or `streamlit docs` `--server.runOnSave true` while iterating to catch syntax/import errors quickly.

---

Happy exploring!
