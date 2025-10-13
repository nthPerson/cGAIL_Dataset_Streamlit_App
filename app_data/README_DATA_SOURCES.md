# Data Sources Reference — Shenzhen Taxi Expert Explorer

This document catalogs **all data sources** used in the Streamlit application, including core trajectory data, helper artifacts, and supplemental socioeconomic datasets.

---

## Core Trajectory Data

### Primary Sources (Required)

| File | Location | Format | Description |
|------|----------|--------|-------------|
| `states_all.npy` | `app_data/` | NumPy array | Concatenated state feature matrix (666,729 states × 126 dimensions). Downloaded from Hugging Face if missing. |
| `traj_index.parquet` | `app_data/` | Parquet | Trajectory index mapping `(expert, traj_idx)` to `(start, length)` offsets into `states_all.npy`. |

### Helper Artifacts (Generated)

| File | Location | Format | Description |
|------|----------|--------|-------------|
| `experts_summary.parquet` | `derived/` | Parquet | Per-expert aggregates (trajectory counts, state statistics) for KPI and hierarchy views. |
| `lengths_by_expert/{expert}.npy` | `derived/lengths_by_expert/` | NumPy array | Cached trajectory lengths for each expert, used in histograms. |
| `norm_stats.npz` | `derived/` | NPZ archive | Per-feature normalization statistics (mean, std, min, max, median, MAD). |
| `derived_meta.json` | `derived/` | JSON | Dataset signature, grid dimensions, and helper build metadata. |
| `paths.parquet` | `derived/` (optional) | Parquet | Extracted (x, y) paths for all trajectories when coordinate indices are known. |
| `visitation_overall.npz` | `derived/` (optional) | NPZ archive | Precomputed visitation counts across all experts. |

---

## Spatial Mapping Data

### Grid-to-District Mapping (New)

| File | Location | Columns | Description |
|------|----------|---------|-------------|
| `grid_to_district_ArcGIS_table.csv` | `app_data/` | `OID_`, `row`, `col`, `cell_id`, `Shape_Length`, `Shape_Area`, `cell_area`, `district`, `overlap_m2`, `overlap_pct`, `district_id` | Maps each grid cell (50×90 grid) to its corresponding Shenzhen district. Includes overlap area and percentage for cells spanning multiple districts. |
| `district_id_mapping.csv` | `app_data/` | `district`, `district_id` | Simple lookup table mapping district names to integer IDs for efficient joins. |

**Key Notes:**
- **Grid structure**: 50 rows × 90 columns = 4,500 cells
- **Coordinate system**: 
  - `row` corresponds to Y (dimension 0, range [0, 49])
  - `col` corresponds to X (dimension 1, range [0, 89])
  - Origin at top-left (0, 0)
- **District assignment**: Use `overlap_pct` to determine primary district if cell spans multiple districts (e.g., assign to district with highest overlap percentage)

---

## Socioeconomic & Demographic Data

All CSV files below are district-level supplemental datasets for enriching trajectory analysis with contextual socioeconomic metrics.

### Common Dimensions
- **District / Region**: Administrative unit (Futian, Nanshan, Luohu, Yantian, Bao'an, Longgang, Longhua, Pingshan, Guangming, Dapeng)
- **Time**: Only housing price file is time-series (monthly). Others are snapshot aggregates.

---

## Supplemental Dataset Inventory

### 1. `avg_housing_price_per_sq_meter_by_district.csv`
Monthly average **residential housing prices** (currency units per square meter) by district.

| Column | Type | Description |
|--------|------|-------------|
| `Month-Year` | string (MMM-YY) | Month label (e.g., `Jul-16`). Consider parsing to a proper datetime (assume day=1) for time series work. |
| `Luohu`, `Futian`, `Nanshan`, `Yantian`, `Bao'an`, `Longgang` | string numeric w/ thousands separators | Average price per square meter. Values contain embedded commas and are quoted (e.g., `"63,833"`). Convert via `price = int(value.replace(',',''))` (units appear to be RMB). |

**Row count:** 3 sample months provided (likely truncated sample).
**Cleaning Notes:**
- Cast district columns to integers or floats after stripping commas.
- If future months added, enforce monotonic time index and handle missing districts with `NaN`.

### 2. `economy_sector_proportions_by_district_by_unit.csv`
Counts of **surveyed economic units** by high‑level sector per district.

| Column | Type | Description |
|--------|------|-------------|
| `District` | string | District name or `Total`. Note: one value appears as `Bao�an` (encoding artifact for `Bao'an`). Normalize via UTF‑8 decode + replacement. |
| `Survey Units` | int | Total counted units. |
| `Industry`, `Construction`, `Wholesale and Retail`, `Hotel and Catering`, `Real Estate`, `Leasing and Business Services` | int | Sectoral counts. |

**Consistency Checks:** For each district (non‑Total), sectoral sum may be less than `Survey Units` if other sectors exist but are unlisted. Don't force equality without verifying source.
**Cleaning Notes:**
- Fix mojibake: replace `�` with apostrophe or remove (canonical: `Bao'an`).
- Enforce non‑negative integers; treat blanks as zero if they appear in future expansions.

### 3. `enterprise_financial_indicators_by_district.csv`
Financial & labor indicators for enterprises.

| Column | Type | Description |
|--------|------|-------------|
| `Region` | string | District or `Total`. Some rows show spacing anomalies (e.g., multi‑digit grouping in numeric columns). |
| `Number of Enterprises (unit)` | int | Enterprise count. |
| `Total Assets (100 million yuan)` | float | Aggregate assets (×1e8 yuan). |
| `Business Revenue (100 million yuan)` | float | Revenue (×1e8 yuan); one value has a space (`2 875.63`) that should be normalized (`2875.63`). |
| `Employee Compensation Payable (100 million yuan)` | float | Labor cost obligations (×1e8 yuan). |
| `Average Number of Employed Persons (person)` | int | Average employment (headcount). |

**Cleaning Notes:**
- Remove embedded spaces inside numeric tokens before casting.
- Validate that `Average Number of Employed Persons` correlates with enterprise count (outliers may warrant domain review).

### 4. `household_population_with_gender_by_district.csv`
Household registration population and gender split.

| Column | Type | Description |
|--------|------|-------------|
| `District` | string | District. Note: here `Baoan` lacks apostrophe; standardize to `Bao'an`. |
| `Household Registration Population (10,000 persons)` | float | Registered (hukou) population (×1e4 persons). |
| `Men (10,000 persons)` / `Women (10,000 persons)` | float | Gender disaggregation (×1e4). |
| `Sex Ratio (?=100)` | float | Presumably male:female * 100 ( >100 indicates more males). Rename downstream to `sex_ratio_m_per_100_f` for clarity. |

**Derived Validation:** `abs((Men + Women) - Household) < tolerance` (small rounding drift acceptable).

### 5. `number_of_fully_employed_workers_at_year_end_by_district.csv`
Sectoral distribution of fully employed workers at year end.

This table has a very wide schema; many cells are blank (interpretable as zero or missing). Numeric values include embedded spaces which must be stripped.

| Column (subset) | Type | Notes |
|-----------------|------|-------|
| `District` | string | District. Encoding artifact again for `Bao�an`. |
| `Total` | int | Total fully employed workers. Contains grouped digits with spaces (e.g., `977 584` → `977584`). |
| Remaining sector columns | int or null | Each corresponds to an industry classification (manufacturing, information tech, education, etc.). Blanks should be parsed as `NaN` then optionally filled with 0 if semantics = absence. |

**Cleaning Steps:**
1. Strip spaces from numeric tokens.
2. Convert blanks (`''`) to `NaN`; decide fill policy per downstream need.
3. Optionally compute row sum over sector columns and compare to `Total` for QA (allow discrepancies if aggregation definitions differ).
4. Normalize long header names to snake_case (e.g., `information_transmission_software_it`).

### 6. `GDP_by_district_10k_yuan.csv`
Gross Domestic Product by district (units: 10,000 yuan).

| Column | Type | Description |
|--------|------|-------------|
| `District` | string | Includes historical transitional labels (`Original_Bao'an`, `Original_Longgang`) plus `Total`. |
| `GDP` | int | GDP in 10,000 yuan. Multiply by 10,000 for yuan. |

**Integration Idea:** Join with land area & population to compute GDP per capita or GDP density.

### 7. `total_land_area_population_and_density_by_district.csv`
Geospatial and population density metrics.

| Column | Type | Description |
|--------|------|-------------|
| `Region` | string | District or `Total`. Should align with `District` in other files (rename to `District` when merging). |
| `Land Area (km�)` | float | Land area; header contains a Unicode replacement char for superscript 2 (`km²`). Normalize header to `land_area_km2`. |
| `Year-end Permanent Population (10,000 persons)` | float | Resident population (×1e4). |
| `Permanent Registered Population (10,000)` | float | Registered subset (×1e4). |
| `Permanent Non-registered Population (10,000)` | float | Non‑registered (×1e4). |
| `Population Density (persons/km�)` | int | Density; header has same encoding issue; normalize to `population_density_per_km2`. |

**Derived Metrics:**
- `registered_share = registered / permanent`
- `non_registered_ratio = non_registered / registered`
- Cross‑check: `(registered + non_registered) ≈ permanent`.

### 8. `grid_to_district_ArcGIS_table.csv` ⭐ NEW
**Critical spatial mapping file** connecting the 50×90 grid to Shenzhen districts.

| Column | Type | Description |
|--------|------|-------------|
| `OID_` | int | Object ID from ArcGIS export (sequential, not used for joins). |
| `row` | int | Grid row index [0, 49]. Maps to Y coordinate (dim 0 in state vector). |
| `col` | int | Grid column index [0, 89]. Maps to X coordinate (dim 1 in state vector). |
| `cell_id` | int | Unique cell identifier (typically `row * 90 + col`). |
| `Shape_Length` | float | Perimeter of grid cell polygon (meters). |
| `Shape_Area` | float | Area of grid cell polygon (square meters). |
| `cell_area` | float | Cell area (may differ slightly from `Shape_Area` due to projection). |
| `district` | string | District name this cell intersects (primary assignment). |
| `overlap_m2` | float | Overlap area between cell and district (square meters). |
| `overlap_pct` | float | Percentage of cell covered by this district (0-100). |
| `district_id` | int | Integer ID for district (foreign key to `district_id_mapping.csv`). |

**Key Notes:**
- **4,500 total cells** (50 rows × 90 columns)
- **Multi-district cells**: Some cells span multiple districts; use `overlap_pct` to determine primary district
- **Coordinate mapping**: `row` → Y (dim 0), `col` → X (dim 1)
- **Join key**: Use `(row, col)` or `cell_id` to link grid cells to trajectories

**Usage Examples:**
```python
# Load mapping
grid_map = pd.read_csv('grid_to_district_ArcGIS_table.csv')

# Get district for cell (y=20, x=35)
district = grid_map[(grid_map.row == 20) & (grid_map.col == 35)].district.iloc[0]

# Find primary district for multi-district cells
primary = grid_map.sort_values('overlap_pct', ascending=False).groupby(['row', 'col']).first()
```

### 9. `district_id_mapping.csv` ⭐ NEW
Simple lookup table for district names to integer IDs.

| Column | Type | Description |
|--------|------|-------------|
| `district` | string | Official district name (canonical form). |
| `district_id` | int | Unique integer identifier for joins and efficient storage. |

**Districts included** (10 total):
- Futian, Nanshan, Luohu, Yantian, Bao'an, Longgang
- Longhua, Pingshan, Guangming, Dapeng

**Usage**: Join with `grid_to_district_ArcGIS_table.csv` or socioeconomic CSVs for consistent district identification.

---
## Cross‑File Harmonization Recommendations
1. **District Name Canonicalization**: Map variants to a single form:
   - `Bao�an`, `Baoan` → `Bao'an`
   - `Region` → `District` column name alignment.
2. **Encoding / Unicode Repairs**: Replace `�` (U+FFFD) in headers and values; convert `km�` to `km²`; create ASCII fallbacks (`km2`).
3. **Numeric Normalization**:
   - Remove thousands separators (commas) and embedded spaces.
   - Cast empty strings to `NaN`, then decide fill vs drop.
4. **Schema Consistency**: Convert all column names to lower snake_case after ingestion for internal use (retain originals for export if needed).
5. **Units Metadata**: Maintain a dict mapping each numeric column to its base unit or scaling factor (e.g., `*_10k_persons` → multiplier 1e4, `_100_million_yuan` → multiplier 1e8) to enable on‑the‑fly unit toggles in the UI.

---
## Potential Integration with Streamlit Explorer
- **Augmented KPIs**: Display district population, density, or GDP next to trajectory / expert stats for contextual narrative (e.g., visitation vs. population density correlation).
- **Overlay Layers**: If spatial grid cells can be mapped to districts (via polygon lookup or majority assignment), compute per‑district visitation intensity and join socioeconomic metrics.
- **Feature Enrichment**: Add district‑level features (GDP per capita, housing price index) to state feature vectors as auxiliary conditioning inputs (requires consistent district mapping for each `(y, x)` cell).
- **Time Series Extensions**: Incorporate monthly housing price trends into temporal analysis (align month to time slot groups or broad seasonal categories).

---
## Example Cleaning Snippet (Pandas)
```python
import pandas as pd
import re

# Canonical district name mapping
DISTRICT_FIX = {
    "Bao�an": "Bao'an",
    "Baoan": "Bao'an",
    "Region": "District",  # header-level harmonization
}

def load_housing(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns[1:]:
        df[col] = (df[col]
                   .str.replace(',', '', regex=False)
                   .astype('Int64'))
    df['date'] = pd.to_datetime(df['Month-Year'], format='%b-%y', errors='coerce')
    return df

def normalize_numeric_token(token):
    if pd.isna(token):
        return token
    t = re.sub(r"[ ,]", "", str(token))  # remove spaces & commas
    return pd.to_numeric(t, errors='coerce')

# Generic cleaner for wide numeric tables
def clean_table(path, district_col='District'):
    df = pd.read_csv(path)
    if district_col not in df.columns and 'Region' in df.columns:
        df.rename(columns={'Region': 'District'}, inplace=True)
    df['District'] = df['District'].replace(DISTRICT_FIX)
    for col in df.columns:
        if col == 'District':
            continue
        df[col] = df[col].apply(normalize_numeric_token)
    return df
```

---
## Data Quality / Validation Checklist
- [ ] No unresolved U+FFFD replacement characters in values or headers.
- [ ] All numeric columns successfully cast to numeric (nullable dtypes allowed).
- [ ] District names harmonized; set of districts is consistent across files (allowing historical labels).
- [ ] Derived sanity: population component sums and sector totals within expected tolerance.
- [ ] Time series (housing prices) sorted and free of duplicate months.

---
## Versioning & Provenance
Original extraction summary is stored in `README_Extracted_Shenzhen_Demographic_Data.pdf`. If future updates add year specificity or additional sectors, bump a simple semantic version tag inside this README (e.g., `Data README v1.1`) and document new columns in an appended changelog section.

---
## Summary
These supplemental datasets provide rich contextual signals (population structure, economic scale, housing market dynamics) that can be joined to taxi trajectory analytics for exploratory correlations or model feature augmentation. With light normalization (encoding fixes, numeric parsing) they are ready for integration into Streamlit panels or backend feature engineering pipelines.
