# Shenzhen Socioeconomic & Demographic Data (Supplemental Datasets)

This directory contains supplemental, low‑volume tabular datasets (CSV) describing **district‑level socioeconomic, demographic, and economic structure metrics for Shenzhen**. They are intended to enrich or contextualize the taxi imitation‑learning explorer (e.g., for per‑district overlays, narrative KPIs, or conditioning experiments).

All CSV files are *wide* or *tall‑wide hybrids* with a small number of rows (districts or months) and modest column counts. File sizes are tiny (< ~10 KB each), so eager loading and in‑memory joins are practical.

---
## Common Dimensions / Keys
- **District / Region**: Administrative unit (e.g., Futian, Nanshan, Luohu, Yantian, Bao'an, Longgang, Longhua, Pingshan, Guangming, Dapeng). Some historical / transitional rows include `Original_Bao'an`, `Original_Longgang`, or an aggregate `Total` row.
- **Time**: Only the housing price file is time‑series (monthly). Others are single snapshot aggregates (year not explicitly encoded—assumed contemporaneous extraction; add a `year` column upstream if multi‑year series are introduced later).

---
## Dataset Inventory & Schemas

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
