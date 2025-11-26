# Review Analyzer

A production-ready pipeline for discovering, collecting, transforming, and classifying Google Maps reviews. Originally built for Moroccan banks, now generalized to support **any business type** (restaurants, hotels, insurance, etc.).

## Features

| Stage | Description |
|-------|-------------|
| **Discovery** | Multi-strategy search for business branches via SerpAPI |
| **Collection** | Paginated review collection with checkpoint/resume |
| **Transform** | Date parsing, geocoding (Moroccan regions), aggregation |
| **Classification** | GPT-4 powered sentiment & category analysis (17 categories) |
| **Database** | DuckDB storage with pre-built analytics queries |
| **Dashboard** | Streamlit UI with 5 analytical views |

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your SERPAPI_API_KEY and OPENAI_API_KEY
```

### 2. Launch the Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) - the UI guides you through the entire pipeline.

### 3. Or Use the CLI

```bash
# Full pipeline
python -m src.review_analyzer.main pipeline \
    --businesses "Attijariwafa Bank" "CIH Bank" \
    --cities "Casablanca" "Rabat" \
    --output-mode csv \
    --wide-format

# Individual steps
python -m src.review_analyzer.main discover --businesses "CIH Bank" --cities "Marrakech"
python -m src.review_analyzer.main collect --input agencies.csv --mode csv
python -m src.review_analyzer.main classify --input reviews.csv --wide-format
```

## Project Structure

```
review_analyzer/
├── src/review_analyzer/
│   ├── config.py            # Centralized configuration
│   ├── discover.py          # Step 1: Place discovery
│   ├── collect.py           # Step 2: Review collection
│   ├── transformers/        # Step 3: Data transformation
│   │   ├── normalize_reviews.py   # Date parsing, cleaning
│   │   ├── geocode.py             # Region assignment
│   │   └── aggregates.py          # Pre-computed rollups
│   ├── classify.py          # Step 4: AI classification
│   ├── database/            # Step 5: DuckDB storage
│   │   ├── manager.py       # CRUD operations
│   │   ├── queries.py       # Analytics queries
│   │   └── migrate.py       # CSV → DB migration
│   ├── utils.py             # Shared utilities
│   └── main.py              # CLI orchestrator
├── data/                    # See data/README.md
│   ├── 00_config/           # Static configs (cities, templates)
│   ├── 01_raw/              # Append-only raw data
│   ├── 02_interim/          # Normalized/deduped data
│   ├── 03_processed/        # Final outputs
│   ├── 04_analysis/         # EDA exports
│   └── 99_archive/          # Deprecated files
├── notebooks/               # Jupyter notebooks for each step
├── app.py                   # Streamlit dashboard
├── db_cli.py                # Database CLI tool
└── tests/                   # Test suite
```

## Dashboard Views

| View | Description |
|------|-------------|
| **Business View** | KPIs, sentiment distribution, category analysis for a single business |
| **Competitor View** | Side-by-side comparison, rankings, market positioning |
| **Regional View** | Geographic distribution across Morocco's 12 regions |
| **Temporal View** | Trend analysis, monthly evolution, seasonality |
| **Map View** | Interactive map with location ratings |

## Classification Categories

**7 Positive:**
- Accueil chaleureux et personnel attentionné
- Service client réactif et à l'écoute
- Conseil personnalisé et professionnalisme
- Efficacité et rapidité de traitement
- Accessibilité et proximité des services
- Satisfaction sans détails spécifiques
- Expérience digitale et services en ligne

**7 Negative:**
- Attente interminable et lenteur en agence
- Service client injoignable ou non réactif
- Réclamations ignorées ou mal suivies
- Incidents techniques et erreurs récurrentes
- Frais jugés abusifs ou non justifiés
- Insatisfaction sans détails spécifiques
- Manque de considération / attitude peu professionnelle

**3 Other:** Neutre (hors-sujet), Autre positif, Autre négatif

## Database CLI

```bash
# Initialize database
python db_cli.py init

# Import existing CSV data
python db_cli.py migrate

# View statistics
python db_cli.py stats

# Run analytics queries
python db_cli.py query business          # By business
python db_cli.py query ranking           # Rankings
python db_cli.py query issues --top 5    # Top issues
python db_cli.py query trends            # Temporal trends

# Custom SQL
python db_cli.py sql "SELECT business, AVG(rating) FROM reviews GROUP BY business"

# Export
python db_cli.py export --output export.csv
```

## Docker

```bash
# Build and run
docker-compose up --build

# Development mode (with live reload)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Configuration

Key settings in `.env`:

```bash
SERPAPI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

Advanced settings in `src/review_analyzer/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `SERPAPI_CONFIG.delay_seconds` | 1.2 | Delay between API calls |
| `OPENAI_CONFIG.model` | gpt-4.1-mini | OpenAI model |
| `PROCESSING_CONFIG.batch_size` | 100 | Reviews per batch |
| `CONF_THRESHOLD` | 0.55 | Category confidence threshold |

## Testing

```bash
pytest tests/ -v                    # All tests
pytest tests/ -v -m unit            # Fast unit tests only
pytest tests/ -v --cov=src          # With coverage
```

## Requirements

- Python 3.8+
- [SerpAPI](https://serpapi.com) account (discovery + collection)
- [OpenAI](https://openai.com) API key (classification)
- DuckDB (auto-installed via requirements.txt)

## Documentation

| Document | Description |
|----------|-------------|
| [`data/README.md`](data/README.md) | Data folder architecture |
| [`markdowns/HANDOVER.md`](markdowns/HANDOVER.md) | Complete technical handover |

## License

[Add your license here]
