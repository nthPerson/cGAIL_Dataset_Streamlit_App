# Documentation Quick Reference Card

**For AI Agents** — Essential information at a glance

---

## 📖 Reading Order (10 minutes total)

1. **ai_context_docs/README.md** (2 min) — Start here for navigation
2. **ai_context_docs/PROJECT_OVERVIEW.md** (5 min) — Complete context
3. **ai_context_docs/DATA_MODEL.md** (3 min) — Data structure reference

**As needed**:
- **VISUALS_SPEC.md** — When working with panels
- **DEV_GUIDE.md** — When optimizing performance

---

## 🎯 Critical Information (Memorize)

### Grid System
```
50 rows × 90 columns = 4,500 cells
Origin: top-left (0, 0)
Dim 0 = Y (row) [0, 49]
Dim 1 = X (column) [0, 89]
```

### Coordinate Code
```python
y_idx, x_idx = 0, 1  # ALWAYS
y = state[0]         # Row
x = state[1]         # Column
```

### State Vector
```
Dims 0-3: Active (X, Y, T1, T2)
Dims 4-124: Inactive (not used)
Dim 125: Action label (not state)
```

### Plotly Requirement
```python
yaxis=dict(autorange="reversed")  # REQUIRED
```

---

## 📂 File Locations

### Documentation
```
ai_context_docs/
├── README.md              # Index
├── PROJECT_OVERVIEW.md    # Architecture
├── DATA_MODEL.md          # Data reference
├── VISUALS_SPEC.md        # Panel specs
├── DEV_GUIDE.md           # Development
└── archive/               # Historical
```

### Data Files
```
app_data/
├── states_all.npy         # 666K states × 126 dims
├── traj_index.parquet     # Trajectory index
├── grid_to_district_*.csv # Grid mapping
└── *.csv                  # Demographics

See: app_data/README_DATA_SOURCES.md
```

### Code Structure
```
views/
├── overview.py        # KPIs
├── hierarchy.py       # Treemap
├── state_level.py     # Features
├── trajectory.py      # Paths
├── visitation.py      # Heatmaps
└── diagnostics.py     # Analysis
```

---

## 🔧 Common Tasks

### Load Trajectory
```python
idx = traj_index[(traj_index.expert == expert_id) & 
                 (traj_index.traj_idx == traj_idx)]
start, length = idx.iloc[0][['start', 'length']]
states = states_all[start:start+length]
```

### Get Coordinates
```python
y = states[:, 0]  # Rows
x = states[:, 1]  # Columns
```

### Create Heatmap
```python
vis = np.zeros((50, 90))
for s in states:
    y, x = int(s[0]), int(s[1])
    if 0 <= x < 90 and 0 <= y < 50:
        vis[y, x] += 1
```

### Grid to District
```python
grid_map = pd.read_csv('grid_to_district_ArcGIS_table.csv')
district = grid_map[(grid_map.row == y) & 
                    (grid_map.col == x)].district.iloc[0]
```

---

## ⚠️ Common Pitfalls

❌ `dim 0 = X, dim 1 = Y` → ✅ `dim 0 = Y, dim 1 = X`  
❌ Include dim 125 in viz → ✅ Exclude (it's action label)  
❌ Forget `autorange="reversed"` → ✅ Always reverse Y-axis  
❌ Assume all 4500 cells visited → ✅ Only 38.6% coverage  
❌ Use dims 4-124 → ✅ Use pickle files instead

---

## 🔍 Troubleshooting

### Wrong visitation pattern?
→ Check coordinate indices (y_idx=0, x_idx=1)

### Heatmap upside down?
→ Add `yaxis=dict(autorange="reversed")`

### Missing data?
→ Check `derived/` for helper artifacts

### Slow performance?
→ Review caching in DEV_GUIDE.md

### Coordinate confusion?
→ See archive/COORDINATE_FIX_2025-10-08.md

---

## 📊 Data Quick Stats

- **42 experts** (drivers)
- **~30K trajectories** total
- **666,729 states** total
- **1,735 cells** visited (38.6%)
- **Western region**: 96.9% activity
- **Eastern region**: 3.1% activity

---

## 🔗 Key References

- **Full data catalog**: app_data/README_DATA_SOURCES.md
- **Coordinate fix**: archive/COORDINATE_FIX_2025-10-08.md
- **Update log**: archive/UPDATES_2025-10-08.md
- **Main README**: streamlit/README.md

---

## 🎓 Learning Path

**New to project?**
1. Read PROJECT_OVERVIEW.md
2. Scan DATA_MODEL.md
3. Run `streamlit run app.py`
4. Explore panels in browser

**Need to modify code?**
1. Review relevant panel in VISUALS_SPEC.md
2. Check caching in DEV_GUIDE.md
3. Review integration patterns in DATA_MODEL.md
4. Make changes with confidence

**Debugging issue?**
1. Check Common Pitfalls in DATA_MODEL.md
2. Search archive/ for similar issues
3. Review code examples
4. Test with diagnostics panel

---

**Last Updated**: 2025-10-12  
**Print this card** for quick reference during development!
