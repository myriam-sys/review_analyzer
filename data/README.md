# Data Folder Structure

This folder follows a clean data engineering architecture with clear separation of concerns.

## 📂 Directory Overview

### `00_config/` - Static Configuration (Version Controlled)
- **Purpose:** Reference data and templates that rarely change
- **Contents:**
  - `cities/aliases.json` - City name normalization map
  - `cities/coordinates.ndjson` - OSM city data (one JSON per line)
  - `cities/regions.geojson` - Morocco region polygons (optional)
  - `templates/banks_template.csv` - Input template for users

### `01_raw/` - Append-Only Raw Data (Never Edited)
- **Purpose:** Original data as received from APIs/sources
- **Contents:**
  - `discovery/YYYYMMDD/` - SerpAPI discovery outputs (CSV)
  - `reviews/GoogleMaps/YYYYMMDD/` - JSONL files per place_id
  - `external/` - Third-party data drops

**Key Rule:** Files here are immutable - never edit, only append new runs

### `02_interim/` - Normalized & Deduped Data (Recomputable)
- **Purpose:** Cleaned, unified datasets ready for processing
- **Contents:**
  - `discovery/places.parquet` - Unified branches table
  - `discovery/cache/place_id_cache.json` - Place ID resolution cache
  - `collection/reviews.parquet` - Unified raw reviews (deduped)
  - `collection/checkpoints/review_fetch_log.parquet` - Fetch status tracking
  - `transform/reviews_normalized.parquet` - Dates/coordinates parsed
  - `transform/reviews_with_region.parquet` - Region-enriched reviews

**Key Rule:** All files here can be regenerated from `01_raw/`

### `03_processed/` - Final Outputs (Versioned)
- **Purpose:** Production-ready datasets for analysis/UI
- **Contents:**
  - `discovery/YYYYMMDD/discovered.csv`
  - `discovery/latest` → symlink to most recent
  - `collection/YYYYMMDD/summary.json`
  - `collection/latest` → symlink to most recent
  - `classification/YYYYMMDD/reviews_labeled.parquet`
  - `classification/YYYYMMDD/aggregates/` - Pre-computed rollups
  - `classification/latest` → symlink to most recent

**Key Rule:** Date-stamped runs with `latest` symlink for easy access

### `04_analysis/` - Notebook Outputs
- **Purpose:** Exports from analysis notebooks
- **Contents:**
  - `reports/` - Generated markdown/PDF reports
  - `figures/` - Charts and visualizations
  - `dashboards/` - Interactive HTML dashboards

### `99_archive/` - Deprecated Data
- **Purpose:** Old test runs and deprecated analyses
- **Contents:** Anything you want to keep but not in active use

## 🔄 Data Flow

```
00_config/ (lookup) → discover.py → 01_raw/discovery/YYYYMMDD/
                                   ↓
                           02_interim/discovery/places.parquet
                                   ↓
collect.py → 01_raw/reviews/GoogleMaps/YYYYMMDD/
           ↓
  02_interim/collection/reviews.parquet
           ↓
transform.py → 02_interim/transform/reviews_normalized.parquet
             ↓
classify.py → 03_processed/classification/YYYYMMDD/reviews_labeled.parquet
            ↓
    04_analysis/ (notebooks generate reports)
```

## 📋 File Formats

- **CSV:** Simple outputs, Excel compatibility
- **JSONL (NDJSON):** Streaming data, one JSON per line
- **Parquet:** Columnar format, fast analytics, compression
- **GeoJSON:** Geographic polygons

## 🚀 Usage

1. **Starting a discovery run:**
   - Place input in `01_raw/discovery/YYYYMMDD/`
   - Outputs go to `03_processed/discovery/YYYYMMDD/`

2. **Accessing latest results:**
   - Use `03_processed/*/latest/` symlinks
   - Always points to most recent run

3. **Reprocessing:**
   - Delete files in `02_interim/`
   - Re-run pipeline - it rebuilds from `01_raw/`

4. **Cleaning up:**
   - Move old test files to `99_archive/`
   - Never delete files from `01_raw/` (history!)

## ⚠️ Important Rules

1. ✅ **DO:** Add new date-stamped folders for each run
2. ✅ **DO:** Use Parquet for large datasets (>1MB)
3. ✅ **DO:** Update `latest` symlinks after successful runs
4. ❌ **DON'T:** Edit files in `01_raw/` (append-only)
5. ❌ **DON'T:** Mix test/production data in `03_processed/`
6. ❌ **DON'T:** Commit large data files to git

## 📝 Notes

- All date formats use `YYYYMMDD` (e.g., `20241125`)
- Symlinks may need Windows-specific handling (copy instead)
- Parquet files require pandas/pyarrow to read
- JSONL files: one complete JSON object per line

---

**Created:** 2024-11-25  
**Last Updated:** 2024-11-25
