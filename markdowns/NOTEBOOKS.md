# Jupyter Notebooks Guide

Interactive tutorials for each step of the Review Analyzer pipeline.

## Available Notebooks

| Notebook | Purpose | Duration |
|----------|---------|----------|
| `discover_placeids.ipynb` | Find Google Maps place IDs | 5-15 min |
| `collect_reviews.ipynb` | Collect reviews for places | 10-20 min |
| `transform_reviews.ipynb` | Parse dates, add regions | 5-10 min |
| `classify_reviews.ipynb` | AI classification with GPT-4 | 15-30 min |
| `EDA.ipynb` | Exploratory data analysis | 10-20 min |

---

## Getting Started

### Prerequisites

```bash
# Setup environment
cd review_analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with SERPAPI_API_KEY and OPENAI_API_KEY

# Launch Jupyter
jupyter notebook notebooks/
```

### Recommended Order

1. **discover_placeids.ipynb** - Start with single business/city test
2. **collect_reviews.ipynb** - Collect reviews for discovered places
3. **transform_reviews.ipynb** - Parse dates, assign regions
4. **classify_reviews.ipynb** - Run AI classification
5. **EDA.ipynb** - Analyze results

---

## Notebook Details

### 1. discover_placeids.ipynb

**What you'll learn:**
- Initialize DiscoveryEngine
- Single vs multi-city/business discovery
- Deduplication and canonical place IDs
- Data validation and export

**Key tests:**
- Single business, single city
- Multiple businesses, single city
- Single business, multiple cities
- Export for next step

### 2. collect_reviews.ipynb

**What you'll learn:**
- Initialize ReviewCollector
- Output modes: CSV, JSON, json-per-city, json-per-business
- Checkpoint and resume functionality
- Review validation

**Key tests:**
- Single place test
- Multiple places (5-10)
- Different output modes
- Checkpoint demonstration

### 3. transform_reviews.ipynb

**What you'll learn:**
- Parse French relative dates ("il y a 2 mois")
- Normalize fields (rating, text)
- Assign Moroccan regions by coordinates or city name
- Build pre-computed aggregates

**Key tests:**
- Date parsing examples
- Region assignment
- Aggregate generation

### 4. classify_reviews.ipynb

**What you'll learn:**
- Initialize ReviewClassifier
- Single vs batch classification
- Wide format (17 binary columns) vs JSON format
- Confidence threshold filtering

**Key tests:**
- View all 17 categories
- Classify single review
- Batch classification
- Wide format output

### 5. EDA.ipynb

**What you'll learn:**
- Load classified data
- Sentiment distribution analysis
- Category frequency analysis
- Business/city comparisons
- Temporal trends

---

## Output Formats

### Discovery Output

```csv
place_id,business,city,title,address,lat,lng,rating,reviews_count
ChIJ...,CIH Bank,Casablanca,CIH Bank Maarif,123 Rue...,33.58,-7.60,3.8,245
```

### Collection Output

```csv
place_id,business,city,rating,date,text,reviewer_name
ChIJ...,CIH Bank,Casablanca,5,il y a 2 mois,Excellent service,Ahmed M.
```

### Transform Output (adds)

```csv
...,created_at,month,region
...,2024-09-15,2024-09-01,Casablanca-Settat
```

### Classification Output (adds)

```csv
...,sentiment,categories_json,cat_accueil,cat_service,...
...,Positif,[{...}],1,0,...
```

---

## Customization

### Changing Parameters

**Discovery:**
```python
test_businesses = ["Your Business Name"]
test_cities = ["Your City"]
```

**Collection:**
```python
output_mode = "csv"  # or "json", "json-per-city", "json-per-business"
```

**Classification:**
```python
wide_format = True  # Creates 17 category columns
confidence_threshold = 0.55  # Minimum confidence
```

---

## Performance Guidelines

| Operation | Sample Size | Expected Duration |
|-----------|-------------|-------------------|
| Discover 1 city | 10-20 places | 30-60 seconds |
| Discover 5 cities | 50-100 places | 3-5 minutes |
| Collect 10 places | 50-100 reviews | 2-3 minutes |
| Collect 100 places | 500-1000 reviews | 15-30 minutes |
| Classify 10 reviews | - | 30-60 seconds |
| Classify 100 reviews | - | 5-10 minutes |
| Classify 1000 reviews | - | 50-100 minutes |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Run first setup cell (adds path to sys.path) |
| FileNotFoundError | Run previous notebook's export cell first |
| API Error: Invalid key | Check .env file; restart Jupyter kernel |
| No results found | Verify business/city names match Google Maps |
| Notebook seems stuck | Long operations are normal; check logs/ for progress |

---

## Tips

### For Beginners

1. Read markdown cells (explanations)
2. Run cells in order (top to bottom)
3. Check outputs after each cell
4. Start with simple tests (single items)
5. Use Shift+Enter to run cells

### For Experienced Developers

1. Skip to production tests
2. Modify `debug=True/False` for verbosity
3. Adjust confidence thresholds
4. Integrate with your own analysis

---

## Learning Path

| Level | Focus | Time |
|-------|-------|------|
| Beginner | Run basic tests (1-3) in each notebook | 2-3 hours |
| Intermediate | Medium tests, experiment with parameters | 1-2 hours |
| Advanced | Production tests, customize for your needs | 1 hour |

---

## Related Documentation

- [HANDOVER.md](HANDOVER.md) - Technical details about each module
- [README.md](../README.md) - Main project documentation
- [TESTING.md](TESTING.md) - Test suite documentation

