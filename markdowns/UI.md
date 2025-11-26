# Streamlit Dashboard Guide

User-friendly web interface for the Review Analyzer pipeline.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your SERPAPI_API_KEY and OPENAI_API_KEY

# Launch
streamlit run app.py
```

Opens automatically at `http://localhost:8501`

---

## Dashboard Views

The application provides 5 analytical views:

| View | Purpose |
|------|---------|
| **Business View** | Single business deep-dive: KPIs, sentiment, categories |
| **Competitor View** | Side-by-side comparison, rankings, market share |
| **Regional View** | Geographic distribution across Morocco's 12 regions |
| **Temporal View** | Trend analysis, monthly evolution, seasonality |
| **Map View** | Interactive Folium map with location ratings |

---

## Sidebar Controls

### Data Source

**Option 1: Load Existing Data**
- Toggle "Use latest classified data" to load from `data/03_processed/classification/latest/`
- Click "Load Data" to load into session

**Option 2: Upload CSV**
- Upload your own classified reviews CSV file
- Must have columns: rating, sentiment (or categories_json)

**Option 3: Run Pipeline**
- Enter business type (e.g., "Banque", "Restaurant", "Hotel")
- Add business names one by one
- Select cities (optional)
- Click "Run Pipeline" to execute discovery, collection, and classification

### Filters

Once data is loaded:
- Filter by business (multiselect)
- Filter by city (multiselect)
- Results update in all views

### Dark Mode

Toggle dark/light theme in sidebar.

---

## Usage Examples

### Example 1: Analyze Existing Data

1. Toggle "Use latest classified data"
2. Click "Load Data"
3. Navigate between views using tabs
4. Use filters to focus on specific businesses/cities

### Example 2: Run New Analysis

1. Set business type: "Banque"
2. Add businesses: "CIH Bank", "Attijariwafa Bank"
3. Select cities: "Casablanca", "Rabat"
4. Click "Run Pipeline"
5. Wait for completion (progress shown in sidebar)
6. Explore results in views

### Example 3: Compare Competitors

1. Load data
2. Go to "Competitor View"
3. View leaderboard by rating, positive rate, or review count
4. Compare sentiment distributions across businesses
5. Identify top performers by city

---

## View Details

### Business View

- **KPI Cards**: Total reviews, average rating, positive rate, negative rate
- **Sentiment Distribution**: Pie chart showing Positive/Neutral/Negative split
- **Rating Distribution**: Bar chart of 1-5 star ratings
- **Top Categories**: Most common positive and negative feedback themes
- **Sample Reviews**: Example reviews for each sentiment

### Competitor View

- **Leaderboard**: Rank businesses by selected metric
- **Comparison Chart**: Side-by-side bar charts
- **Heatmap**: Business vs City positive rate matrix
- **Winners by City**: Best-performing business in each city

### Regional View

- **Regional Distribution**: Reviews per region
- **Regional Performance**: Average rating by region
- **Best/Worst Regions**: Highlight top and bottom performers
- **Regional Sentiment**: Sentiment breakdown by region

### Temporal View

- **Monthly Trends**: Line chart of reviews over time
- **Rating Evolution**: Average rating trend
- **Sentiment Shifts**: How sentiment changes over time
- **Seasonality**: Identify peak periods

### Map View

- **Interactive Map**: Folium map centered on Morocco
- **Color-Coded Markers**: Green (4+), Orange (3-4), Red (<3) by rating
- **Popups**: Click markers for business name, city, rating

---

## Configuration

### Adding New Businesses

In sidebar, add business names one by one. For persistent changes, edit `app.py`:

```python
BUSINESS_PRESETS = {
    "Banque": ["CIH Bank", "Attijariwafa Bank", ...],
    "Restaurant": ["McDonald's", "Pizza Hut", ...],
    # Add your preset here
}
```

### Adding New Cities

Add to `src/review_analyzer/config.py`:

```python
DEFAULT_MAP_CENTERS = {
    "YourCity": "@latitude,longitude,12z",
}
```

### Customizing Colors

BCG color scheme is defined at top of `app.py`:

```python
BCG_COLORS = {
    "green": "#0B6E4F",
    "dark_green": "#003D32",
    # Modify as needed
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "API keys not configured" | Check `.env` file exists with valid keys |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| Charts not showing | Install plotly: `pip install plotly>=5.17.0` |
| Map not loading | Install folium: `pip install folium streamlit-folium` |
| Pipeline freezes | Normal for large datasets; check logs/ for progress |

### Performance Tips

- Start small: 1 business, 1 city first
- Use CSV mode (faster than JSON)
- Large classifications (1000+ reviews) may take 30+ minutes
- Pipeline uses checkpoints - safe to interrupt and resume

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `R` | Rerun app |
| `C` | Clear cache |
| `Esc` | Close dialogs |

---

## Deployment

### Local Development

```bash
streamlit run app.py
```

### Docker

```bash
docker-compose up --build
# Access at http://localhost:8501
```

### Cloud Deployment

**Streamlit Cloud:**
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repository
4. Add API keys as secrets
5. Deploy

---

## Related Documentation

- [HANDOVER.md](HANDOVER.md) - Technical details about app.py structure
- [README.md](../README.md) - Main project documentation

