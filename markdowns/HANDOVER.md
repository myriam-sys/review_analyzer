# Review Analyzer - Technical Handover

This document provides everything a developer needs to understand, maintain, and extend the Review Analyzer pipeline.

**Last Updated:** November 2025

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Data Flow](#data-flow)
4. [Module Reference](#module-reference)
5. [Data Folder Architecture](#data-folder-architecture)
6. [Configuration System](#configuration-system)
7. [Common Development Tasks](#common-development-tasks)
8. [Code Patterns](#code-patterns)
9. [Debugging Guide](#debugging-guide)
10. [Testing](#testing)
11. [Deployment](#deployment)

---

## Architecture Overview

The Review Analyzer is a 5-step pipeline for analyzing Google Maps reviews:

```
Discovery --> Collection --> Transform --> Classification --> Database/UI
   |              |             |               |                |
SerpAPI       SerpAPI      pandas/geo       OpenAI          DuckDB/Streamlit
```

**Key Design Principles:**
- **Modular:** Each step is independent and can run standalone
- **Resumable:** Long operations save checkpoints for recovery
- **Generalized:** Works with any business type (banks, hotels, restaurants)
- **Observable:** All operations are logged with progress tracking

---

## Project Structure

```
review_analyzer/
├── src/review_analyzer/           # Core Python package
│   ├── __init__.py               # Package exports
│   ├── config.py                 # Centralized configuration
│   ├── utils.py                  # Shared utilities (API client, CSV helpers)
│   ├── discover.py               # Step 1: Place discovery
│   ├── collect.py                # Step 2: Review collection
│   ├── transformers/             # Step 3: Data transformation
│   │   ├── __init__.py
│   │   ├── normalize_reviews.py  # Date parsing, field cleaning
│   │   ├── geocode.py            # Region assignment (Morocco)
│   │   └── aggregates.py         # Pre-computed rollups
│   ├── classify.py               # Step 4: AI classification
│   ├── database/                 # Step 5: Storage & analytics
│   │   ├── __init__.py
│   │   ├── models.py             # Data models (Place, Review, etc.)
│   │   ├── manager.py            # DatabaseManager with CRUD
│   │   ├── queries.py            # Pre-built analytics queries
│   │   └── migrate.py            # CSV to database migration
│   ├── geo.py                    # Geographic utilities
│   └── main.py                   # CLI orchestrator
│
├── data/                         # Data storage (see detailed section below)
│   ├── 00_config/                # Static configuration
│   ├── 01_raw/                   # Immutable raw data
│   ├── 02_interim/               # Processed intermediate data
│   ├── 03_processed/             # Final outputs
│   ├── 04_analysis/              # EDA exports
│   ├── 99_archive/               # Deprecated files
│   └── review_analyzer.duckdb    # DuckDB database file
│
├── notebooks/                    # Jupyter notebooks
│   ├── discover_placeids.ipynb   # Discovery walkthrough
│   ├── collect_reviews.ipynb     # Collection walkthrough
│   ├── transform_reviews.ipynb   # Transformation walkthrough
│   ├── classify_reviews.ipynb    # Classification walkthrough
│   └── EDA.ipynb                 # Exploratory data analysis
│
├── app.py                        # Streamlit dashboard (1800 lines)
├── db_cli.py                     # Database CLI tool
├── tests/                        # Test suite
├── logs/                         # Runtime logs
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image
├── docker-compose.yml            # Docker orchestration
└── Makefile                      # Common commands
```

---

## Data Flow

### End-to-End Pipeline

```
User Input (business names, cities)
        |
        v
[1. DISCOVERY] discover.py
        |-- Uses: SerpAPI Google Maps Search
        |-- Input: Business names + city list
        |-- Output: data/01_raw/discovery/YYYYMMDD/discovered.csv
        |-- Key fields: place_id, business, city, lat, lng, rating
        |
        v
[2. COLLECTION] collect.py
        |-- Uses: SerpAPI Reviews API
        |-- Input: CSV with place_ids
        |-- Output: data/02_interim/collection/bank_reviews.csv
        |-- Key fields: place_id, text, rating, date, reviewer_name
        |
        v
[3. TRANSFORM] transformers/
        |-- normalize_reviews.py: Parse French dates, clean fields
        |-- geocode.py: Assign Moroccan regions
        |-- aggregates.py: Build rollups by business/city/month
        |-- Output: data/02_interim/transform/reviews_normalized.csv
        |
        v
[4. CLASSIFICATION] classify.py
        |-- Uses: OpenAI GPT-4
        |-- Input: Normalized reviews
        |-- Output: data/03_processed/classification/latest/
        |-- Key fields: sentiment, categories_json, rationale
        |
        v
[5. STORAGE & UI]
        |-- database/: Store in DuckDB with analytics queries
        |-- app.py: Streamlit dashboard with 5 views
```

### Data Format Between Steps

**Discovery Output:**
```csv
place_id,business,city,title,address,lat,lng,rating,reviews_count
ChIJ...,CIH Bank,Casablanca,CIH Bank Maarif,123 Rue...,33.58,-7.60,3.8,245
```

**Collection Output:**
```csv
place_id,business,city,rating,date,text,reviewer_name
ChIJ...,CIH Bank,Casablanca,5,il y a 2 mois,Excellent service,Ahmed M.
```

**Transform Output (adds):**
```csv
...,created_at,month,region
...,2024-09-15,2024-09-01,Casablanca-Settat
```

**Classification Output (adds):**
```csv
...,sentiment,categories_json,language,rationale
...,Positif,[{"label":"Accueil chaleureux...","confidence":0.92}],French,Positive tone...
```

---

## Module Reference

### 1. config.py - Configuration Hub

**Purpose:** Single source of truth for all settings.

**Key Exports:**

| Variable | Type | Description |
|----------|------|-------------|
| `PROJECT_ROOT` | Path | Root directory of the project |
| `DATA_DIR` | Path | Data folder root |
| `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR` | Path | Data stage directories |
| `SERPAPI_CONFIG` | dict | SerpAPI settings (key, delay, retries) |
| `OPENAI_CONFIG` | dict | OpenAI settings (model, temperature) |
| `CATEGORIES` | list | 17 classification categories |
| `CONF_THRESHOLD` | float | Minimum confidence for category (0.55) |
| `PROCESSING_CONFIG` | dict | Batch size, checkpoint interval |

**Usage:**
```python
from review_analyzer import config

# Access paths
output_path = config.PROCESSED_DIR / "my_output.csv"

# Check API config
print(config.SERPAPI_CONFIG["delay_seconds"])  # 1.2

# Get timestamped output path
path = config.get_timestamped_path(config.RAW_DISCOVERY_DIR)
```

---

### 2. utils.py - Shared Utilities

**Purpose:** Eliminate code duplication across modules.

**A. SerpAPIClient**

The core API client for all Google Maps interactions:

```python
from review_analyzer.utils import SerpAPIClient

client = SerpAPIClient()

# Search for places
results = client.search_google_maps(
    query="CIH Bank Casablanca",
    ll="@33.58,-7.60,12z"  # Center coordinates
)

# Get reviews for a place
reviews = client.get_reviews(
    place_id="ChIJ...",
    sort_by="newestFirst"
)

# Get place details
details = client.get_place_details("ChIJ...")
```

All methods include:
- Automatic retry with exponential backoff (5 attempts)
- Rate limit handling (429 errors)
- Authentication error detection (401)
- Configurable delays between calls

**B. CSV Utilities**

Auto-detect column names with fallbacks:

```python
from review_analyzer.utils import read_agencies_csv

# Handles: place_id, Place_ID, placeId, _place_id
df = read_agencies_csv(
    path="agencies.csv",
    place_id_col=None,  # Auto-detect
    business_col=None,  # Auto-detect
)
```

**C. CheckpointManager**

Resumable operations:

```python
from review_analyzer.utils import CheckpointManager

checkpoint = CheckpointManager("my_operation.json")

# Save progress
checkpoint.save({"last_index": 100, "processed_ids": [...]})

# Resume
state = checkpoint.load()
if state:
    start_from = state["last_index"]

# Clear when done
checkpoint.clear()
```

**D. Custom Exceptions**

```python
from review_analyzer.utils import (
    SerpAPIError,      # Base exception
    RateLimitError,    # 429 errors
    AuthenticationError  # 401 errors
)
```

---

### 3. discover.py - Place Discovery

**Purpose:** Find business locations using multi-strategy search.

**Search Strategies (in order):**
1. Google Maps with center coordinates
2. Google Maps without center (broader)
3. Google Local fallback

**Key Function:**

```python
from review_analyzer.discover import discover_places

results_df = discover_places(
    businesses=["CIH Bank", "Attijariwafa Bank"],
    cities=["Casablanca", "Rabat"],
    output_path="discovered.csv",
    use_cache=True  # Use place_id cache
)
```

**Features:**
- Automatic deduplication by place_id
- Canonical place_id resolution (ChIJ format)
- Cache to avoid duplicate API calls
- Progress tracking with tqdm

---

### 4. collect.py - Review Collection

**Purpose:** Fetch reviews for discovered places.

**Key Function:**

```python
from review_analyzer.collect import collect_reviews

df = collect_reviews(
    input_path="discovered.csv",
    output_path="reviews.csv",
    output_mode="csv",  # or json, json-per-city, json-per-business
    checkpoint=True
)
```

**Output Modes:**
- `csv`: Single consolidated CSV
- `json`: Single JSON file
- `json-per-city`: Separate file per city
- `json-per-business`: Separate file per business

**Features:**
- Automatic pagination (fetches all reviews)
- Checkpoint/resume for long runs
- Failed place_ids saved to separate file

---

### 5. transformers/ - Data Transformation

**Purpose:** Clean and enrich raw reviews before classification.

**A. normalize_reviews.py**

Parse French relative dates and clean fields:

```python
from review_analyzer.transformers import normalize_reviews_df

df_clean = normalize_reviews_df(df_raw)

# Parses: "il y a 2 mois" -> datetime
# Adds: created_at (datetime), month (period)
# Cleans: rating (1-5 int), text (stripped)
```

**B. geocode.py**

Assign Moroccan regions:

```python
from review_analyzer.transformers import add_region, add_region_by_city

# Using coordinates (precise)
df = add_region(df, regions_path="data/00_config/cities/regions.geojson")

# Using city name (fallback)
df = add_region_by_city(df, city_col="city")
```

Morocco's 12 regions are mapped:
- Casablanca-Settat
- Rabat-Sale-Kenitra
- Marrakech-Safi
- Fes-Meknes
- etc.

**C. aggregates.py**

Pre-compute rollups for dashboards:

```python
from review_analyzer.transformers import build_aggregates

files = build_aggregates(df_classified, output_dir=Path("aggregates/"))

# Creates:
# - by_business.parquet
# - by_city.parquet
# - by_business_city.parquet
# - by_business_month.parquet (temporal)
# - by_region.parquet (if region column exists)
```

---

### 6. classify.py - AI Classification

**Purpose:** Classify reviews using OpenAI GPT-4.

**Key Function:**

```python
from review_analyzer.classify import classify_reviews

df_classified = classify_reviews(
    input_path="reviews.csv",
    output_path="classified.csv",
    wide_format=True,  # One column per category
    checkpoint=True
)
```

**Classification Output:**

| Column | Description |
|--------|-------------|
| `sentiment` | Positif / Neutre / Negatif |
| `categories_json` | JSON array of {label, confidence} |
| `language` | Detected language (French, Arabic, etc.) |
| `rationale` | AI explanation for classification |
| `[category_name]` | Binary (0/1) if wide_format=True |

**17 Categories:**

*Positive (7):*
- Accueil chaleureux et personnel attentionne
- Service client reactif et a l'ecoute
- Conseil personnalise et professionnalisme
- Efficacite et rapidite de traitement
- Accessibilite et proximite des services
- Satisfaction sans details specifiques
- Experience digitale et services en ligne

*Negative (7):*
- Attente interminable et lenteur en agence
- Service client injoignable ou non reactif
- Reclamations ignorees ou mal suivies
- Incidents techniques et erreurs recurrentes
- Frais bancaires juges abusifs
- Insatisfaction sans details specifiques
- Manque de consideration / attitude peu professionnelle

*Other (3):*
- Hors-sujet ou contenu non pertinent (Neutral)
- Autre (positif)
- Autre (negatif)

---

### 7. database/ - DuckDB Storage

**Purpose:** Persistent storage with pre-built analytics.

**A. DatabaseManager**

```python
from review_analyzer.database import DatabaseManager

with DatabaseManager() as db:
    # Insert data
    db.insert_places(df_places)
    db.insert_reviews(df_reviews)
    db.insert_classifications(df_classified)
    
    # Query
    df = db.get_reviews_df(include_classifications=True)
    
    # Export
    db.export_to_csv("export.csv")
```

**B. AnalyticsQueries**

Pre-built analytical queries:

```python
from review_analyzer.database import AnalyticsQueries, DatabaseManager

with DatabaseManager() as db:
    analytics = AnalyticsQueries(db)
    
    # Overview stats
    stats = analytics.get_overview_stats()
    # Returns: total_reviews, total_businesses, avg_rating, sentiment_distribution
    
    # Business ranking
    ranking = analytics.get_business_ranking(
        metric="positive_rate",  # or avg_rating, review_count
        top_n=10
    )
    
    # Temporal trends
    trends = analytics.get_temporal_trends(
        granularity="month",
        business="CIH Bank"
    )
    
    # Top issues
    issues = analytics.get_top_issues(business="CIH Bank", top_n=5)
    
    # Competitive position
    position = analytics.get_competitive_position("CIH Bank")
```

**Database Schema:**

| Table | Description |
|-------|-------------|
| `places` | Unique business locations |
| `reviews` | Customer reviews (deduplicated) |
| `classifications` | Sentiment + categories per review |
| `review_categories` | Wide-format category flags |
| `processing_runs` | Audit trail |

---

### 8. app.py - Streamlit Dashboard

**Purpose:** Interactive web UI for analysis.

**5 Views:**

| View | Description |
|------|-------------|
| Business View | Single business deep-dive: KPIs, categories, sentiment |
| Competitor View | Side-by-side comparison, rankings |
| Regional View | Geographic distribution across Morocco |
| Temporal View | Trends over time, seasonality |
| Map View | Interactive map with Folium |

**Key Features:**
- BCG color theme with dark/light mode toggle
- Real-time pipeline execution from sidebar
- Filter by business, city, date range
- Export to CSV/Excel

**Running:**
```bash
streamlit run app.py
```

---

## Data Folder Architecture

```
data/
├── 00_config/           # Static configuration (version controlled)
│   ├── cities/
│   │   ├── aliases.json         # City name normalization
│   │   ├── coordinates.ndjson   # OSM city data
│   │   └── regions.geojson      # Morocco region polygons
│   └── templates/
│       └── banks_template.csv   # Input template
│
├── 01_raw/              # Immutable raw data (never edit)
│   ├── discovery/YYYYMMDD/      # SerpAPI discovery outputs
│   └── reviews/GoogleMaps/YYYYMMDD/  # Raw review JSONs
│
├── 02_interim/          # Processed intermediate data (recomputable)
│   ├── discovery/cache/         # Place ID cache
│   ├── collection/              # Merged reviews
│   │   ├── bank_reviews.csv
│   │   └── checkpoints/
│   └── transform/               # Normalized data
│       └── reviews_normalized.csv
│
├── 03_processed/        # Final outputs (versioned)
│   ├── classification/
│   │   ├── YYYYMMDD/
│   │   └── latest/              # Symlink to most recent
│   └── transform/
│       └── by_*.parquet         # Pre-computed aggregates
│
├── 04_analysis/         # Notebook exports
│   ├── business_view/           # Business charts
│   ├── competitor_view/         # Comparison charts
│   ├── eda/                     # EDA outputs
│   └── reports/
│
├── 99_archive/          # Deprecated/old files
│
└── review_analyzer.duckdb       # DuckDB database file
```

**Key Rules:**
1. Never edit files in `01_raw/` (append-only)
2. `02_interim/` can be regenerated from `01_raw/`
3. Date-stamped folders use YYYYMMDD format
4. `latest/` symlinks point to most recent run

---

## Configuration System

### Environment Variables (.env)

**Required:**
```bash
SERPAPI_API_KEY=your_serpapi_key
OPENAI_API_KEY=your_openai_key
```

### config.py Settings

**API Settings:**
```python
SERPAPI_CONFIG = {
    "api_key": os.getenv("SERPAPI_API_KEY"),
    "hl": "fr",           # Language
    "gl": "ma",           # Morocco
    "delay_seconds": 1.2, # Between requests
    "max_retries": 5,
    "timeout": 60,
}

OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4.1-mini",
    "temperature": 0,
    "max_retries": 5,
}
```

**Processing Settings:**
```python
PROCESSING_CONFIG = {
    "batch_size": 100,           # Reviews per batch
    "pause_seconds": 0.5,        # Between batches
    "checkpoint_interval": 10,   # Save every N items
}

CONF_THRESHOLD = 0.55  # Min confidence for category assignment
```

---

## Common Development Tasks

### Adding a New Business Type

The pipeline is already generalized. Just use:

```bash
python -m src.review_analyzer.main pipeline \
    --businesses "Sofitel" "Four Seasons" \
    --cities "Marrakech" "Casablanca" \
    --output-mode csv
```

### Adding a New City

1. Add coordinates to `config.py`:
```python
DEFAULT_MAP_CENTERS = {
    "YourCity": "@latitude,longitude,12z",
}
```

2. Or add to `data/00_config/cities/aliases.json`

### Adding a New Classification Category

1. Edit `config.py`:
```python
CATEGORIES = [
    # ... existing categories
    "Your New Category (description)",
]
```

2. Update the classification prompt in `classify.py` if needed

### Adding a New Analytics Query

1. Add method to `database/queries.py`:
```python
class AnalyticsQueries:
    def get_my_new_analysis(self, filters=None):
        query = """
            SELECT ...
            FROM reviews r
            JOIN classifications c ON r.id = c.review_id
            WHERE ...
        """
        return self.db.execute(query).fetchdf()
```

### Adding a New Dashboard View

1. Add page function to `app.py`:
```python
def page_my_view():
    st.header("My New View")
    df = st.session_state.data
    # ... build charts
```

2. Add to tabs in `main()`:
```python
tabs = st.tabs([
    "Business View",
    "My New View",  # Add here
    # ...
])
```

---

## Code Patterns

### Pattern 1: Checkpoint/Resume

Used in collect.py and classify.py for long operations:

```python
from review_analyzer.utils import CheckpointManager

def long_operation(items, output_path):
    checkpoint = CheckpointManager(f"{output_path}.checkpoint.json")
    
    # Load previous state
    state = checkpoint.load() or {"processed": [], "last_idx": 0}
    processed = state["processed"]
    start_idx = state["last_idx"]
    
    for i, item in enumerate(items[start_idx:], start=start_idx):
        result = process_item(item)
        processed.append(result)
        
        # Save checkpoint every N items
        if i % 10 == 0:
            checkpoint.save({"processed": processed, "last_idx": i})
    
    # Clear checkpoint on success
    checkpoint.clear()
    return processed
```

### Pattern 2: Retry with Exponential Backoff

Used in utils.py SerpAPIClient:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True
)
def api_call_with_retry(url, params):
    response = requests.get(url, params=params)
    
    if response.status_code == 429:
        raise RateLimitError("Rate limit exceeded")
    if response.status_code == 401:
        raise AuthenticationError("Invalid API key")
    
    response.raise_for_status()
    return response.json()
```

### Pattern 3: Column Auto-Detection

Used in utils.py for flexible CSV reading:

```python
def find_column(df, candidates, required=True):
    """Find column from list of possible names."""
    for col in candidates:
        # Exact match
        if col in df.columns:
            return col
        # Case-insensitive match
        for df_col in df.columns:
            if df_col.lower() == col.lower():
                return df_col
    
    if required:
        raise ValueError(f"Could not find column. Tried: {candidates}")
    return None

# Usage
place_id_col = find_column(df, ["place_id", "Place_ID", "placeId", "_place_id"])
```

### Pattern 4: Timestamped Output Paths

Used throughout for versioned outputs:

```python
from datetime import datetime

def get_timestamped_path(base_dir, filename=None):
    """Create date-stamped subdirectory."""
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = base_dir / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename:
        return output_dir / filename
    return output_dir
```

---

## Debugging Guide

### Check API Keys

```python
from review_analyzer import config

print(f"SerpAPI Key: {'Set' if config.SERPAPI_KEY else 'MISSING'}")
print(f"OpenAI Key: {'Set' if config.OPENAI_API_KEY else 'MISSING'}")
```

### View Logs

```bash
# Main pipeline log
tail -f logs/pipeline.log

# Check for errors
grep -i error logs/pipeline.log
```

### Inspect Checkpoints

```bash
# See checkpoint state
cat data/02_interim/collection/checkpoints/*.json

# Reset a stuck operation
rm data/02_interim/collection/checkpoints/*.json
```

### Debug Classification

```python
# Run classification on single review for debugging
from review_analyzer.classify import classify_single_review

result = classify_single_review(
    text="Service excellent, personnel tres aimable",
    rating=5
)
print(result)
```

### Database Queries

```bash
# Direct SQL access
python db_cli.py sql "SELECT COUNT(*) FROM reviews"

# Full stats
python db_cli.py stats
```

---

## Testing

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_config.py        # Configuration tests
├── test_utils.py         # Utility function tests
├── test_discover.py      # Discovery module tests
├── test_collect.py       # Collection tests
├── test_transformers.py  # Transform tests
├── test_classify.py      # Classification tests
└── test_database.py      # Database tests
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_utils.py -v

# With coverage
pytest tests/ -v --cov=src/review_analyzer --cov-report=html

# Only fast unit tests
pytest tests/ -v -m unit
```

### Test Markers

- `@pytest.mark.unit`: Fast tests with mocked APIs
- `@pytest.mark.integration`: Tests with real file I/O
- `@pytest.mark.slow`: Performance tests

---

## Deployment

### Docker

```bash
# Build
docker build -t review-analyzer:latest .

# Run
docker-compose up

# Development mode (live reload)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Volume Mounts

| Host | Container | Purpose |
|------|-----------|---------|
| `./data` | `/app/data` | Data persistence |
| `./logs` | `/app/logs` | Log access |
| `./.env` | `/app/.env` | API keys |

### Production Considerations

1. **API Rate Limits:** Respect SerpAPI and OpenAI rate limits
2. **Data Size:** DuckDB handles 100K+ reviews efficiently
3. **Memory:** Classification batches use ~500MB for 100 reviews
4. **Disk:** Allow 500MB for data folder growth

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| `SERPAPI_API_KEY not found` | Check .env file exists and is properly formatted |
| `RateLimitError` | Wait 1 minute, or increase `delay_seconds` in config |
| Discovery returns 0 results | Verify business name matches Google Maps exactly |
| Classification stuck | Check checkpoint file, re-run (auto-resumes) |
| DuckDB locked | Close other connections, restart app |
| Memory error | Reduce `batch_size` in config |
| Date parsing fails | Check for unusual date formats in raw data |

---

## Contact & Resources

- **Main README:** See `README.md` for quick start
- **Data Architecture:** See `data/README.md`
- **Notebooks:** See `notebooks/` for step-by-step examples

For questions, check the logs first, then the checkpoint files, then escalate.
