"""
Review Analyzer - Streamlit Dashboard
Multi-business review analysis platform with BCG styling
Supports: Business View, Competitor View, Regional View, Temporal View, Map View
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ============================================================================
# CONFIGURATION & THEME
# ============================================================================

# BCG Color Palette
BCG_COLORS = {
    "green": "#0B6E4F",
    "dark_green": "#003D32",
    "light_green": "#A7C957",
    "teal": "#1F7A8C",
    "gray": "#5C7C89",
    "light_gray": "#E8ECEF",
    "red": "#D62728",
    "orange": "#FF7F0E",
    "yellow": "#FFD700",
    "white": "#FFFFFF",
    "text": "#20322F",
    # Dark mode colors
    "dark_bg": "#1a2f38",
    "dark_card": "#243b47",
    "dark_text": "#E8ECEF",
}

RATING_COLORS = {
    1: BCG_COLORS["red"],
    2: BCG_COLORS["orange"],
    3: BCG_COLORS["yellow"],
    4: BCG_COLORS["light_green"],
    5: BCG_COLORS["green"],
}

SENTIMENT_COLORS = {
    "Positif": BCG_COLORS["green"],
    "Négatif": BCG_COLORS["red"],
    "Neutre": BCG_COLORS["gray"],
}

BCG_COLORWAY = [
    BCG_COLORS["green"], BCG_COLORS["teal"], BCG_COLORS["light_green"],
    BCG_COLORS["dark_green"], BCG_COLORS["gray"], BCG_COLORS["orange"]
]

# Category short names for display
CATEGORY_SHORT_NAMES = {
    "Accueil chaleureux et personnel attentionné": "Accueil chaleureux",
    "Service client réactif et à l'écoute": "Service réactif",
    "Conseil personnalisé et professionnalisme des équipes": "Conseil personnalisé",
    "Efficacité et rapidité de traitement": "Efficacité/rapidité",
    "Accessibilité et proximité des services": "Accessibilité",
    "Satisfaction sans détails spécifiques": "Satisfaction générique",
    "Expérience digitale et services en ligne pratiques": "Digital",
    "Attente interminable et lenteur en agence": "Attente / Lenteur",
    "Service client injoignable ou non réactif": "Service injoignable",
    "Réclamations ignorées ou mal suivies": "Réclamations ignorées",
    "Incidents techniques et erreurs récurrentes": "Incidents techniques",
    "Frais bancaires jugés abusifs ou non justifiés": "Frais Abusifs",
    "Insatisfaction sans détails spécifiques": "Insatisfaction générique",
    "Manque de considération ou attitude peu professionnelle": "Attitude négative",
    "Autre (positif)": "Autre (+)",
    "Autre (négatif)": "Autre (-)",
    "Hors-sujet ou contenu non pertinent": "Hors-sujet",
}

# Business presets by type
BUSINESS_PRESETS = {
    "Banque": [
        "Attijariwafa Bank", "BMCE Bank of Africa", "Banque Populaire",
        "CIH Bank", "Crédit Agricole", "Crédit du Maroc", "BMCI",
        "CFG Bank", "Al Barid Bank", "Société Générale"
    ],
    "Assurance": [
        "Wafa Assurance", "RMA", "Saham Assurance", "AXA Assurance",
        "Allianz", "Atlanta", "MAMDA", "Sanad"
    ],
    "Hotel": [
        "Sofitel", "Four Seasons", "Marriott", "Hilton", "Radisson",
        "Ibis", "Novotel", "Hyatt", "Movenpick", "Royal Mansour"
    ],
    "Restaurant": [
        "McDonald's", "KFC", "Burger King", "Pizza Hut", "Subway",
        "Starbucks", "Paul", "La Sqala", "Rick's Café"
    ],
}


# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="Review Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_theme_css(dark_mode: bool = False) -> str:
    """Generate CSS for light or dark mode."""
    if dark_mode:
        bg_color = BCG_COLORS["dark_bg"]
        card_bg = BCG_COLORS["dark_card"]
        text_color = BCG_COLORS["dark_text"]
        sidebar_bg = "#0d1f26"
        border_color = "#3a5a6a"
    else:
        bg_color = BCG_COLORS["white"]
        card_bg = BCG_COLORS["white"]
        text_color = BCG_COLORS["text"]
        sidebar_bg = BCG_COLORS["dark_green"]
        border_color = BCG_COLORS["light_gray"]
    
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global */
    html, body, [class*="css"] {{
        font-family: 'Inter', 'Trebuchet MS', sans-serif;
        color: {text_color};
    }}
    
    .stApp {{
        background-color: {bg_color};
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
    }}
    
    [data-testid="stSidebar"] * {{
        color: {BCG_COLORS["white"]} !important;
    }}
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stTextInput label {{
        color: {BCG_COLORS["light_gray"]} !important;
    }}
    
    /* Headers */
    .main-header {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {BCG_COLORS["green"]};
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid {BCG_COLORS["green"]};
    }}

    .section-header {{
        font-size: 1.4rem;
        font-weight: 600;
        color: {BCG_COLORS["green"] if not dark_mode else BCG_COLORS["light_green"]};
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }}
    
    .sub-header {{
        font-size: 1.1rem;
        font-weight: 500;
        color: {BCG_COLORS["teal"]};
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }}
    
    /* Cards */
    .metric-card {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {BCG_COLORS["green"]};
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        color: {BCG_COLORS["gray"]};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metric-delta {{
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }}
    
    .delta-positive {{ color: {BCG_COLORS["green"]}; }}
    .delta-negative {{ color: {BCG_COLORS["red"]}; }}
    
    /* Info boxes */
    .info-box {{
        padding: 1rem;
        border-radius: 4px;
        background-color: {"#1e3a4a" if dark_mode else "#E3F2FD"};
        border-left: 4px solid {BCG_COLORS["teal"]};
        color: {text_color};
        margin: 1rem 0;
    }}
    
    .success-box {{
        padding: 1rem;
        border-radius: 4px;
        background-color: {"#1e3a2a" if dark_mode else "#E8F5E9"};
        border-left: 4px solid {BCG_COLORS["green"]};
        color: {text_color};
    }}
    
    .warning-box {{
        padding: 1rem;
        border-radius: 4px;
        background-color: {"#3a3a1a" if dark_mode else "#FFF3E0"};
        border-left: 4px solid {BCG_COLORS["orange"]};
        color: {text_color};
    }}
    
    /* Category badges */
    .category-badge {{
        display: inline-block;
        padding: 4px 12px;
        margin: 2px;
        border-radius: 16px;
        font-size: 0.85rem;
    }}
    
    .cat-positive {{
        background-color: {"#1e3a2a" if dark_mode else "#E8F5E9"};
        color: {BCG_COLORS["green"]};
        border: 1px solid {BCG_COLORS["green"]};
    }}
    
    .cat-negative {{
        background-color: {"#3a1e1e" if dark_mode else "#FFEBEE"};
        color: {BCG_COLORS["red"]};
        border: 1px solid {BCG_COLORS["red"]};
    }}
    
    .cat-neutral {{
        background-color: {"#2a2a3a" if dark_mode else "#ECEFF1"};
        color: {BCG_COLORS["gray"]};
        border: 1px solid {BCG_COLORS["gray"]};
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {BCG_COLORS["green"]};
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: background-color 0.3s;
    }}

    .stButton > button:hover {{
        background-color: {BCG_COLORS["dark_green"]};
        color: white;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        font-weight: 500;
        color: {BCG_COLORS["gray"]};
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {BCG_COLORS["green"]};
        color: white !important;
    }}
    
    /* Divider */
    .bcg-divider {{
        height: 2px;
        background: linear-gradient(to right, {BCG_COLORS["green"]}, {BCG_COLORS["teal"]}, {BCG_COLORS["light_green"]});
        margin: 1.5rem 0;
        border: none;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Progress bar */
    .stProgress > div > div > div > div {{
        background-color: {BCG_COLORS["green"]};
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        font-weight: 600;
        color: {BCG_COLORS["dark_green"] if not dark_mode else BCG_COLORS["light_green"]};
    }}
    
    /* DataFrames */
    .dataframe {{
        font-size: 0.9rem;
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {BCG_COLORS["green"]};
        font-weight: 700;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {BCG_COLORS["gray"]};
        font-weight: 500;
    }}
</style>
    """


def get_plotly_layout(dark_mode: bool = False) -> dict:
    """Get BCG-styled Plotly layout settings."""
    if dark_mode:
        bg_color = BCG_COLORS["dark_bg"]
        text_color = BCG_COLORS["dark_text"]
        grid_color = "#3a5a6a"
    else:
        bg_color = BCG_COLORS["white"]
        text_color = BCG_COLORS["text"]
        grid_color = BCG_COLORS["light_gray"]
    
    return dict(
        font=dict(family="Inter, Trebuchet MS, sans-serif", size=12, color=text_color),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        colorway=BCG_COLORWAY,
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            tickfont=dict(family="Inter", size=11)
        ),
        yaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            tickfont=dict(family="Inter", size=11)
        ),
    )


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "dark_mode": False,
        "data_loaded": False,
        "df": None,
        "business_type": "",
        "selected_business": None,
        "selected_competitors": [],
        "competitor_list": [],  # For building list one by one
        "selected_cities": [],
        "discovery_complete": False,
        "collection_complete": False,
        "classification_complete": False,
        "new_business_input": "",  # Temporary input for adding new business
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def get_project_paths() -> Dict[str, Path]:
    """Get all project paths."""
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    return {
        "root": project_root,
        "data": data_dir,
        "config": data_dir / "00_config",
        "raw": data_dir / "01_raw",
        "interim": data_dir / "02_interim",
        "processed": data_dir / "03_processed",
        "analysis": data_dir / "04_analysis",
        "classified_latest": data_dir / "03_processed" / "classification" / "latest",
    }


def load_cities_from_osm() -> List[str]:
    """Load city names from OSM NDJSON file."""
    paths = get_project_paths()
    osm_file = paths["config"] / "cities" / "coordinates.ndjson"
    
    cities = []
    if osm_file.exists():
        try:
            with open(osm_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        # Extract French name if available
                        name = entry.get("name", "")
                        other_names = entry.get("other_names", {})
                        fr_name = other_names.get("name:fr", name.split(" ")[0] if " " in name else name)
                        if fr_name:
                            cities.append(fr_name)
        except Exception as e:
            st.warning(f"Could not load cities: {e}")
    
    # Add major cities if not loaded
    if not cities:
        cities = [
            "Casablanca", "Rabat", "Marrakech", "Fès", "Tanger", "Agadir",
            "Meknès", "Oujda", "Kénitra", "Tétouan", "Safi", "El Jadida"
        ]
    
    return sorted(set(cities))


def find_latest_classified_file() -> Optional[Path]:
    """Find the most recent classified CSV file."""
    paths = get_project_paths()
    
    # Search locations in priority order
    search_dirs = [
        paths["classified_latest"],
        paths["processed"] / "classification",
        paths["processed"],
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            csv_files = list(search_dir.glob("*.csv"))
            if csv_files:
                # Sort by modification time, newest first
                csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return csv_files[0]
    
    return None


def load_classified_data(file_path: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Load classified reviews data with auto-detection."""
    if file_path is None:
        file_path = find_latest_classified_file()
    
    if file_path is None or not file_path.exists():
        return None, {"error": "No classified data found"}
    
    try:
        # Try different encodings
        for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        else:
            return None, {"error": "Failed to decode CSV"}
        
        # Detect columns
        col_mapping = detect_columns(df)
        
        # Preprocess
        df = preprocess_data(df, col_mapping)
        
        return df, {
            "file": file_path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "col_mapping": col_mapping,
        }
    
    except Exception as e:
        return None, {"error": str(e)}


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Auto-detect column names."""
    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None
    
    return {
        "business": find_col(["_business", "_bank", "business", "bank", "brand"]),
        "city": find_col(["_city", "city", "ville", "commune"]),
        "date": find_col(["month", "date", "created_at", "time"]),
        "rating": find_col(["rating", "note", "stars", "score"]),
        "text": find_col(["text", "review_text", "content", "snippet"]),
        "sentiment": find_col(["sentiment"]),
        "place_id": find_col(["_place_id", "place_id", "agency_id"]),
    }


def preprocess_data(df: pd.DataFrame, col_mapping: Dict) -> pd.DataFrame:
    """Preprocess dataframe for analysis."""
    df = df.copy()
    
    # Parse dates
    if col_mapping["date"] and col_mapping["date"] in df.columns:
        df[col_mapping["date"]] = pd.to_datetime(df[col_mapping["date"]], errors="coerce", dayfirst=True)
        df["_month"] = df[col_mapping["date"]].dt.to_period("M").dt.to_timestamp()
        df["_year"] = df[col_mapping["date"]].dt.year
    
    # Ensure rating is numeric
    if col_mapping["rating"] and col_mapping["rating"] in df.columns:
        df[col_mapping["rating"]] = pd.to_numeric(df[col_mapping["rating"]], errors="coerce")
    
    # Normalize business names
    if col_mapping["business"] and col_mapping["business"] in df.columns:
        df[col_mapping["business"]] = df[col_mapping["business"]].astype(str).str.strip()
    
    return df


def get_category_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Detect and categorize category columns."""
    try:
        from review_analyzer import config
        categories = list(getattr(config, "CATEGORIES", []))
    except:
        categories = []
    
    if categories:
        pos_cats = [c for c in categories[:7] if c in df.columns]
        neg_cats = [c for c in categories[7:14] if c in df.columns]
        other_cats = [c for c in categories[14:] if c in df.columns]
    else:
        # Heuristic detection
        pos_keywords = ["accueil", "réactif", "conseil", "efficac", "accessib", "satisfaction", "digital"]
        neg_keywords = ["attente", "injoignable", "réclamation", "incident", "frais", "insatisfaction", "manque"]
        
        pos_cats = []
        neg_cats = []
        other_cats = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in pos_keywords):
                pos_cats.append(col)
            elif any(kw in col_lower for kw in neg_keywords):
                neg_cats.append(col)
            elif "autre" in col_lower or "hors" in col_lower:
                other_cats.append(col)
    
    return pos_cats, neg_cats, other_cats


def shorten_category(cat: str) -> str:
    """Get short display name for category."""
    for full, short in CATEGORY_SHORT_NAMES.items():
        if full.lower() in cat.lower() or cat.lower() in full.lower():
            return short
    # Fallback: first words before parenthesis
    short = cat.split("(")[0].strip()
    return short[:30] + "..." if len(short) > 30 else short


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_kpi_metrics(df: pd.DataFrame, col_mapping: Dict, dark_mode: bool = False) -> None:
    """Create KPI metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    
    with col2:
        if col_mapping["sentiment"] and col_mapping["sentiment"] in df.columns:
            pos_pct = (df[col_mapping["sentiment"]] == "Positif").mean() * 100
            st.metric("Positive", f"{pos_pct:.1f}%", delta=f"{pos_pct - 50:.1f}pp vs 50%")
        else:
            st.metric("Positive", "N/A")
    
    with col3:
        if col_mapping["sentiment"] and col_mapping["sentiment"] in df.columns:
            neg_pct = (df[col_mapping["sentiment"]] == "Négatif").mean() * 100
            st.metric("Negative", f"{neg_pct:.1f}%", delta=f"{neg_pct - 50:.1f}pp vs 50%", delta_color="inverse")
        else:
            st.metric("Negative", "N/A")
    
    with col4:
        if col_mapping["rating"] and col_mapping["rating"] in df.columns:
            avg_rating = df[col_mapping["rating"]].mean()
            st.metric("Avg Rating", f"{avg_rating:.2f}")
        else:
            st.metric("Avg Rating", "N/A")


def create_rating_bar_chart(df: pd.DataFrame, col_mapping: Dict, dark_mode: bool = False) -> go.Figure:
    """Create horizontal bar chart of average rating by business."""
    if not col_mapping["business"] or not col_mapping["rating"]:
        return None
    
    avg_ratings = df.groupby(col_mapping["business"])[col_mapping["rating"]].mean().sort_values()
    avg_ratings = avg_ratings.reset_index()
    avg_ratings.columns = ["Business", "Avg Rating"]
    
    fig = px.bar(
        avg_ratings,
        x="Avg Rating",
        y="Business",
        orientation="h",
        title="<b>Average Rating by Business</b>",
        color="Avg Rating",
        color_continuous_scale=[[0, BCG_COLORS["red"]], [0.5, BCG_COLORS["yellow"]], [1, BCG_COLORS["green"]]],
        range_color=[1, 5],
    )
    
    fig.update_layout(**get_plotly_layout(dark_mode))
    fig.update_layout(
        height=max(400, len(avg_ratings) * 50),
        xaxis_title="Average Rating (1-5)",
        yaxis_title="",
        coloraxis_showscale=False,
        xaxis=dict(range=[0, 5]),
    )
    fig.update_traces(texttemplate="%{x:.2f}", textposition="outside")
    
    return fig


def create_sentiment_pie(df: pd.DataFrame, col_mapping: Dict, dark_mode: bool = False) -> go.Figure:
    """Create sentiment distribution pie chart."""
    if not col_mapping["sentiment"] or col_mapping["sentiment"] not in df.columns:
        return None
    
    sentiment_counts = df[col_mapping["sentiment"]].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.4,
        marker=dict(colors=[SENTIMENT_COLORS.get(s, BCG_COLORS["gray"]) for s in sentiment_counts.index]),
        textinfo='label+percent',
        textfont=dict(family="Inter", size=12),
    )])
    
    fig.update_layout(**get_plotly_layout(dark_mode))
    fig.update_layout(
        title="<b>Sentiment Distribution</b>",
        showlegend=False,
        height=350,
    )
    
    return fig


def create_rating_distribution_stacked(df: pd.DataFrame, col_mapping: Dict, dark_mode: bool = False) -> go.Figure:
    """Create stacked bar chart of rating distribution by business."""
    if not col_mapping["business"] or not col_mapping["rating"]:
        return None
    
    biz_col = col_mapping["business"]
    rating_col = col_mapping["rating"]
    
    # Calculate rating distribution
    rating_dist = df.groupby([biz_col, rating_col]).size().unstack(fill_value=0)
    rating_pct = rating_dist.div(rating_dist.sum(axis=1), axis=0) * 100
    
    # Ensure columns 1-5 exist
    for r in [1, 2, 3, 4, 5]:
        if r not in rating_pct.columns:
            rating_pct[r] = 0
    rating_pct = rating_pct[[1, 2, 3, 4, 5]]
    
    # Sort by average rating
    avg_order = df.groupby(biz_col)[rating_col].mean().sort_values(ascending=False).index
    rating_pct = rating_pct.reindex(avg_order)
    
    fig = go.Figure()
    
    for rating in [1, 2, 3, 4, 5]:
        fig.add_trace(go.Bar(
            name=f"{rating} stars",
            y=rating_pct.index,
            x=rating_pct[rating],
            orientation="h",
            marker_color=RATING_COLORS[rating],
            text=[f"{v:.0f}%" if v >= 5 else "" for v in rating_pct[rating]],
            textposition="inside",
            textfont=dict(color="white", size=10),
        ))
    
    fig.update_layout(**get_plotly_layout(dark_mode))
    fig.update_layout(
        barmode="stack",
        title="<b>Rating Distribution by Business</b>",
        xaxis_title="Percentage (%)",
        yaxis_title="",
        height=max(400, len(rating_pct) * 60),
        xaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    
    return fig


def create_category_heatmap(
    df: pd.DataFrame, 
    col_mapping: Dict, 
    category_cols: List[str],
    title: str,
    is_negative: bool = True,
    dark_mode: bool = False
) -> go.Figure:
    """Create BCG-styled category heatmap with rankings."""
    if not col_mapping["business"] or not category_cols:
        return None
    
    biz_col = col_mapping["business"]
    businesses = sorted(df[biz_col].dropna().unique())
    n_businesses = len(businesses)
    n_categories = len(category_cols)
    
    # Calculate percentages
    pct_matrix = np.zeros((n_categories, n_businesses))
    
    for j, biz in enumerate(businesses):
        sub = df[df[biz_col] == biz]
        n_reviews = len(sub)
        for i, cat in enumerate(category_cols):
            if cat in sub.columns and n_reviews > 0:
                pct = (sub[cat].fillna(0).astype(int).sum() / n_reviews * 100)
                pct_matrix[i, j] = round(pct, 1)
    
    # Calculate ranks
    rank_matrix = np.zeros_like(pct_matrix, dtype=int)
    for i in range(n_categories):
        row = pct_matrix[i, :]
        if is_negative:
            ranks = pd.Series(row).rank(ascending=True, method="min").astype(int).values
        else:
            ranks = pd.Series(row).rank(ascending=False, method="min").astype(int).values
        rank_matrix[i, :] = ranks
    
    # Normalize for colorscale
    rank_normalized = (rank_matrix - 1) / max(1, (n_businesses - 1))
    
    # Create annotations
    annot_text = []
    for i in range(n_categories):
        row_text = []
        for j in range(n_businesses):
            rank = rank_matrix[i, j]
            pct = pct_matrix[i, j]
            row_text.append(f"<b>{rank}</b> ({pct:.1f}%)")
        annot_text.append(row_text)
    
    # Colorscales
    if is_negative:
        colorscale = [[0, "#FFF5F0"], [0.5, "#FB6A4A"], [1, "#67000D"]]
        title_color = BCG_COLORS["red"]
    else:
        colorscale = [[0, "#00441B"], [0.5, "#74C476"], [1, "#F7FCF5"]]
        title_color = BCG_COLORS["green"]
    
    short_cat_names = [shorten_category(c) for c in category_cols]
    
    fig = go.Figure(data=go.Heatmap(
        z=rank_normalized,
        x=businesses,
        y=short_cat_names,
        colorscale=colorscale,
        showscale=False,
        xgap=2,
        ygap=2,
        text=annot_text,
        texttemplate="%{text}",
        textfont=dict(size=9, color="white" if dark_mode else "black"),
        hovertemplate="Business: %{x}<br>Category: %{y}<br>Rank: %{z:.0f}<extra></extra>",
    ))
    
    bg_color = BCG_COLORS["dark_bg"] if dark_mode else BCG_COLORS["white"]
    text_color = BCG_COLORS["dark_text"] if dark_mode else BCG_COLORS["text"]
    
    fig.update_layout(
        title=dict(
            text=f"<span style='color:{title_color}; font-weight:bold'>{title}</span>",
            x=0.02,
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(family="Inter", size=12, color=text_color),
        height=max(400, n_categories * 50 + 100),
        margin=dict(l=200, r=50, t=80, b=100),
        xaxis=dict(side="bottom", tickangle=-45),
        yaxis=dict(autorange="reversed"),
    )
    
    return fig


def create_temporal_chart(df: pd.DataFrame, col_mapping: Dict, dark_mode: bool = False) -> go.Figure:
    """Create temporal trend chart."""
    if "_month" not in df.columns or not col_mapping["rating"]:
        return None
    
    rating_col = col_mapping["rating"]
    
    monthly = df.groupby("_month").agg(
        avg_rating=(rating_col, "mean"),
        n_reviews=(rating_col, "size"),
    ).reset_index().sort_values("_month")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=monthly["_month"],
            y=monthly["n_reviews"],
            name="Volume",
            marker_color=BCG_COLORS["teal"],
            opacity=0.7,
        ),
        secondary_y=False,
    )
    
    # Rating line
    fig.add_trace(
        go.Scatter(
            x=monthly["_month"],
            y=monthly["avg_rating"],
            name="Avg Rating",
            mode="lines+markers",
            line=dict(color=BCG_COLORS["light_green"], width=3),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )
    
    fig.update_layout(**get_plotly_layout(dark_mode))
    fig.update_layout(
        title="<b>Review Volume & Rating Over Time</b>",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    
    fig.update_yaxes(title_text="Number of Reviews", secondary_y=False)
    fig.update_yaxes(title_text="Average Rating", range=[1, 5], secondary_y=True)
    
    return fig


def create_city_performance(df: pd.DataFrame, col_mapping: Dict, dark_mode: bool = False) -> go.Figure:
    """Create city performance chart."""
    if not col_mapping["city"] or not col_mapping["rating"]:
        return None
    
    city_col = col_mapping["city"]
    rating_col = col_mapping["rating"]
    
    city_stats = df.groupby(city_col).agg(
        avg_rating=(rating_col, "mean"),
        n_reviews=(rating_col, "size")
    ).reset_index()
    
    # Filter to cities with at least 10 reviews
    city_stats = city_stats[city_stats["n_reviews"] >= 10]
    city_stats = city_stats.sort_values("avg_rating", ascending=True).tail(20)
    
    fig = px.bar(
        city_stats,
        x="avg_rating",
        y=city_col,
        orientation="h",
        title="<b>Average Rating by City</b><br><sup>Top 20 cities with ≥10 reviews</sup>",
        color="avg_rating",
        color_continuous_scale=[[0, BCG_COLORS["red"]], [0.5, BCG_COLORS["yellow"]], [1, BCG_COLORS["green"]]],
        range_color=[1, 5],
    )
    
    fig.update_layout(**get_plotly_layout(dark_mode))
    fig.update_layout(
        height=max(400, len(city_stats) * 35),
        xaxis_title="Average Rating",
        yaxis_title="",
        coloraxis_showscale=False,
        xaxis=dict(range=[0, 5.5]),
    )
    fig.update_traces(texttemplate="%{x:.2f}", textposition="outside")
    
    return fig


def create_competitor_radar(
    df: pd.DataFrame, 
    col_mapping: Dict, 
    businesses: List[str],
    dark_mode: bool = False
) -> go.Figure:
    """Create radar chart comparing businesses."""
    if not col_mapping["business"] or not col_mapping["rating"] or not col_mapping["sentiment"]:
        return None
    
    biz_col = col_mapping["business"]
    rating_col = col_mapping["rating"]
    sentiment_col = col_mapping["sentiment"]
    
    categories = ["Avg Rating", "% Positive", "% Negative", "Volume (normalized)"]
    
    fig = go.Figure()
    
    max_volume = df[biz_col].value_counts().max()
    
    for biz in businesses:
        if biz not in df[biz_col].values:
            continue
        
        sub = df[df[biz_col] == biz]
        
        avg_rating = sub[rating_col].mean() / 5 * 100  # Normalize to 0-100
        pct_pos = (sub[sentiment_col] == "Positif").mean() * 100
        pct_neg = 100 - (sub[sentiment_col] == "Négatif").mean() * 100  # Invert so higher is better
        volume_norm = len(sub) / max_volume * 100
        
        fig.add_trace(go.Scatterpolar(
            r=[avg_rating, pct_pos, pct_neg, volume_norm],
            theta=categories,
            name=biz,
            fill="toself",
            opacity=0.6,
        ))
    
    fig.update_layout(**get_plotly_layout(dark_mode))
    fig.update_layout(
        title="<b>Competitor Comparison Radar</b>",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor=BCG_COLORS["dark_bg"] if dark_mode else BCG_COLORS["white"],
        ),
        showlegend=True,
        height=500,
    )
    
    return fig


# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def render_sidebar():
    """Render the sidebar with all controls."""
    st.sidebar.markdown("## Review Analyzer")
    st.sidebar.markdown('<div class="bcg-divider"></div>', unsafe_allow_html=True)
    
    # Dark mode toggle
    st.session_state.dark_mode = st.sidebar.toggle("Dark Mode", value=st.session_state.dark_mode)
    
    st.sidebar.markdown("---")
    
    # Data source selection
    st.sidebar.markdown("### Data Source")
    data_source = st.sidebar.radio(
        "Choose data source",
        ["Load Existing Data", "Run New Pipeline"],
        label_visibility="collapsed"
    )
    
    if data_source == "Load Existing Data":
        # File uploader or use existing
        use_existing = st.sidebar.checkbox("Use latest classified data", value=True)
        
        if use_existing:
            if st.sidebar.button("Load Data", use_container_width=True):
                with st.spinner("Loading data..."):
                    df, info = load_classified_data()
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.session_state.data_info = info
                        st.sidebar.success(f"Loaded {info['rows']:,} reviews")
                    else:
                        st.sidebar.error(f"{info.get('error', 'Unknown error')}")
        else:
            uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                col_mapping = detect_columns(df)
                df = preprocess_data(df, col_mapping)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.data_info = {"file": uploaded_file.name, "rows": len(df), "col_mapping": col_mapping}
                st.sidebar.success(f"Loaded {len(df):,} reviews")
    
    else:
        # Pipeline execution mode
        st.sidebar.markdown("### Pipeline Settings")
        
        # Business Type - Free text input with suggestions
        st.sidebar.markdown("**Business Type**")
        st.sidebar.caption("e.g., Brunch, Café, Banque, Restaurant, Hotel...")
        
        # Quick presets as buttons
        preset_cols = st.sidebar.columns(4)
        preset_types = ["Banque", "Restaurant", "Hotel", "Café"]
        for i, preset in enumerate(preset_types):
            if preset_cols[i].button(preset, key=f"preset_{preset}", use_container_width=True):
                st.session_state.business_type = preset
        
        # Free text input for custom type
        business_type = st.sidebar.text_input(
            "Or type your own",
            value=st.session_state.business_type,
            placeholder="e.g., Brunch, Spa, Coiffeur...",
            key="business_type_input"
        )
        st.session_state.business_type = business_type
        
        st.sidebar.markdown("---")
        
        # Businesses to analyze - Add one by one
        st.sidebar.markdown("**Businesses to Analyze**")
        st.sidebar.caption("Add business names one by one")
        
        # Input for new business
        col_input, col_add = st.sidebar.columns([3, 1])
        with col_input:
            new_business = st.text_input(
                "Business name",
                value="",
                placeholder="e.g., Café Clock, Basmane...",
                key="new_business_input",
                label_visibility="collapsed"
            )
        with col_add:
            if st.button("+", key="add_business", use_container_width=True):
                if new_business.strip() and new_business.strip() not in st.session_state.competitor_list:
                    st.session_state.competitor_list.append(new_business.strip())
                    st.rerun()
        
        # Show current list with remove buttons
        if st.session_state.competitor_list:
            st.sidebar.markdown("**Your list:**")
            for i, biz in enumerate(st.session_state.competitor_list):
                col_name, col_del = st.sidebar.columns([4, 1])
                col_name.markdown(f"• {biz}")
                if col_del.button("x", key=f"del_{i}", use_container_width=True):
                    st.session_state.competitor_list.pop(i)
                    st.rerun()
        else:
            st.sidebar.info("Add at least one business name above")
        
        # Quick add from presets if type matches
        if business_type in BUSINESS_PRESETS:
            with st.sidebar.expander(f"Suggested {business_type} businesses", expanded=False):
                for biz in BUSINESS_PRESETS[business_type]:
                    col_sug, col_add_sug = st.columns([3, 1])
                    col_sug.markdown(f"• {biz}")
                    if col_add_sug.button("+", key=f"sug_{biz}"):
                        if biz not in st.session_state.competitor_list:
                            st.session_state.competitor_list.append(biz)
                            st.rerun()
        
        competitors = st.session_state.competitor_list
        st.session_state.selected_competitors = competitors
        
        st.sidebar.markdown("---")
        
        # Cities
        st.sidebar.markdown("**Cities (optional)**")
        cities_list = load_cities_from_osm()
        selected_cities = st.sidebar.multiselect(
            "Select cities",
            cities_list,
            default=[],
            placeholder="All Morocco (leave empty)",
            label_visibility="collapsed"
        )
        st.session_state.selected_cities = selected_cities
        
        # City quick add
        quick_cities = st.sidebar.columns(3)
        for i, city in enumerate(["Casablanca", "Rabat", "Marrakech"]):
            if quick_cities[i].button(city, key=f"city_{city}", use_container_width=True):
                if city not in selected_cities:
                    selected_cities.append(city)
                    st.session_state.selected_cities = selected_cities
                    st.rerun()
        
        st.sidebar.markdown("---")
        
        # Summary before running
        if competitors and business_type:
            st.sidebar.success(f"""
            **Ready to analyze:**
            - Type: {business_type}
            - {len(competitors)} businesses
            - {len(selected_cities) if selected_cities else 'All'} cities
            """)
        
        # Run pipeline button
        can_run = bool(competitors) and bool(business_type.strip())
        if st.sidebar.button(
            "Run Pipeline", 
            type="primary", 
            use_container_width=True,
            disabled=not can_run
        ):
            if can_run:
                run_pipeline(business_type, competitors, selected_cities)
            else:
                st.sidebar.error("Please enter a business type and at least one business name")
        
        # Clear all button
        if competitors:
            if st.sidebar.button("Clear All", use_container_width=True):
                st.session_state.competitor_list = []
                st.session_state.business_type = ""
                st.rerun()
    
    st.sidebar.markdown("---")
    
    # Filters (only if data loaded)
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        col_mapping = st.session_state.data_info.get("col_mapping", detect_columns(df))
        
        st.sidebar.markdown("### Filters")
        
        # Business filter
        if col_mapping["business"] and col_mapping["business"] in df.columns:
            businesses = sorted(df[col_mapping["business"]].dropna().unique())
            selected_biz = st.sidebar.multiselect("Filter Businesses", businesses)
            if selected_biz:
                st.session_state.selected_competitors = selected_biz
        
        # City filter
        if col_mapping["city"] and col_mapping["city"] in df.columns:
            cities = sorted(df[col_mapping["city"]].dropna().unique())
            selected_cities = st.sidebar.multiselect("Filter Cities", cities)
            if selected_cities:
                st.session_state.selected_cities = selected_cities
    
    st.sidebar.markdown("---")
    
    # Categories panel
    with st.sidebar.expander("Classification Categories", expanded=False):
        st.markdown("**Positive Categories:**")
        for cat in list(CATEGORY_SHORT_NAMES.items())[:7]:
            st.markdown(f"[+] {cat[1]}")
        
        st.markdown("**Negative Categories:**")
        for cat in list(CATEGORY_SHORT_NAMES.items())[7:14]:
            st.markdown(f"[-] {cat[1]}")
        
        st.markdown("**Other:**")
        for cat in list(CATEGORY_SHORT_NAMES.items())[14:]:
            st.markdown(f"[o] {cat[1]}")
        
        st.info("To customize categories, please contact the development team.")


def run_pipeline(business_type: str, competitors: List[str], cities: List[str]):
    """Run the full discovery -> collection -> classification pipeline."""
    st.sidebar.warning("Pipeline running... This may take several minutes.")
    
    progress_bar = st.sidebar.progress(0)
    status = st.sidebar.empty()
    
    try:
        from review_analyzer.discover import DiscoveryEngine
        from review_analyzer.collect import ReviewCollector
        from review_analyzer.classify import ReviewClassifier
        from review_analyzer import config
        
        # Step 1: Discovery
        status.text("Step 1/3: Discovering places...")
        progress_bar.progress(10)
        
        engine = DiscoveryEngine(debug=False)
        discovery_output = config.get_output_path("discovery", f"agencies_{datetime.now().strftime('%Y%m%d')}.csv")
        
        discovery_df = engine.discover_branches(
            businesses=competitors,
            cities=cities,
            business_type=business_type.lower(),
            output_path=discovery_output,
        )
        
        progress_bar.progress(33)
        status.text(f"Found {len(discovery_df)} places")
        
        # Step 2: Collection
        status.text("Step 2/3: Collecting reviews...")
        progress_bar.progress(40)
        
        collector = ReviewCollector(debug=False)
        collection_output = config.get_output_path("collection", f"reviews_{datetime.now().strftime('%Y%m%d')}.csv")
        
        stats = collector.collect_reviews(
            input_file=discovery_output,
            output_mode="csv",
            output_path=collection_output,
        )
        
        progress_bar.progress(66)
        status.text(f"Collected {stats['total_reviews']} reviews")
        
        # Step 3: Classification
        status.text("Step 3/3: Classifying reviews...")
        progress_bar.progress(70)
        
        classifier = ReviewClassifier(debug=False)
        df = pd.read_csv(collection_output)
        df = classifier.classify_batch(df)
        df = classifier.convert_to_wide_format(df)
        
        classification_output = config.get_output_path("classification", f"reviews_classified_{datetime.now().strftime('%Y%m%d')}.csv")
        df.to_csv(classification_output, index=False)
        
        progress_bar.progress(100)
        status.text("Pipeline complete!")
        
        # Load the results
        col_mapping = detect_columns(df)
        df = preprocess_data(df, col_mapping)
        st.session_state.df = df
        st.session_state.data_loaded = True
        st.session_state.data_info = {"file": classification_output.name, "rows": len(df), "col_mapping": col_mapping}
        
        st.sidebar.success(f"Pipeline complete! {len(df)} reviews classified.")
        
    except ImportError as e:
        st.sidebar.error(f"Missing module: {e}")
        st.sidebar.info("Please ensure API keys are configured in .env file")
    except Exception as e:
        st.sidebar.error(f"Pipeline error: {e}")
        progress_bar.empty()


def page_business_view():
    """Render Business View page."""
    st.markdown('<div class="main-header">Business Performance Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.info("Please load data from the sidebar to view analytics.")
        return
    
    df = st.session_state.df.copy()
    col_mapping = st.session_state.data_info.get("col_mapping", detect_columns(df))
    dark_mode = st.session_state.dark_mode
    
    # Apply filters
    if st.session_state.selected_competitors and col_mapping["business"]:
        df = df[df[col_mapping["business"]].isin(st.session_state.selected_competitors)]
    if st.session_state.selected_cities and col_mapping["city"]:
        df = df[df[col_mapping["city"]].isin(st.session_state.selected_cities)]
    
    if df.empty:
        st.warning("No data matches the current filters.")
        return
    
    # KPI Metrics
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    create_kpi_metrics(df, col_mapping, dark_mode)
    
    st.markdown('<div class="bcg-divider"></div>', unsafe_allow_html=True)
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_sentiment_pie(df, col_mapping, dark_mode)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_rating_bar_chart(df, col_mapping, dark_mode)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="bcg-divider"></div>', unsafe_allow_html=True)
    
    # Rating distribution
    st.markdown('<div class="section-header">Rating Distribution</div>', unsafe_allow_html=True)
    fig = create_rating_distribution_stacked(df, col_mapping, dark_mode)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="bcg-divider"></div>', unsafe_allow_html=True)
    
    # Category Analysis
    st.markdown('<div class="section-header">Category Analysis</div>', unsafe_allow_html=True)
    
    pos_cats, neg_cats, other_cats = get_category_columns(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if neg_cats:
            fig = create_category_heatmap(df, col_mapping, neg_cats, "Negative Factors", is_negative=True, dark_mode=dark_mode)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if pos_cats:
            fig = create_category_heatmap(df, col_mapping, pos_cats, "Positive Factors", is_negative=False, dark_mode=dark_mode)
            if fig:
                st.plotly_chart(fig, use_container_width=True)


def page_competitor_view():
    """Render Competitor View page."""
    st.markdown('<div class="main-header">Competitor Comparison</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.info("Please load data from the sidebar to view analytics.")
        return
    
    df = st.session_state.df.copy()
    col_mapping = st.session_state.data_info.get("col_mapping", detect_columns(df))
    dark_mode = st.session_state.dark_mode
    
    if not col_mapping["business"]:
        st.error("No business column detected in data.")
        return
    
    biz_col = col_mapping["business"]
    businesses = sorted(df[biz_col].dropna().unique())
    
    # Competitor selection
    st.markdown('<div class="section-header">Select Competitors to Compare</div>', unsafe_allow_html=True)
    
    selected = st.multiselect(
        "Choose businesses",
        businesses,
        default=businesses[:min(5, len(businesses))],
        label_visibility="collapsed"
    )
    
    if not selected:
        st.warning("Please select at least one business to compare.")
        return
    
    df_filtered = df[df[biz_col].isin(selected)]
    
    st.markdown('<div class="bcg-divider"></div>', unsafe_allow_html=True)
    
    # Comparison table
    st.markdown('<div class="section-header">Performance Leaderboard</div>', unsafe_allow_html=True)
    
    leaderboard = []
    for biz in selected:
        sub = df_filtered[df_filtered[biz_col] == biz]
        n_reviews = len(sub)
        avg_rating = sub[col_mapping["rating"]].mean() if col_mapping["rating"] else None
        pct_pos = (sub[col_mapping["sentiment"]] == "Positif").mean() * 100 if col_mapping["sentiment"] else None
        pct_neg = (sub[col_mapping["sentiment"]] == "Négatif").mean() * 100 if col_mapping["sentiment"] else None
        
        leaderboard.append({
            "Business": biz,
            "Reviews": n_reviews,
            "Avg Rating": f"{avg_rating:.2f}" if avg_rating else "N/A",
            "% Positive": f"{pct_pos:.1f}%" if pct_pos else "N/A",
            "% Negative": f"{pct_neg:.1f}%" if pct_neg else "N/A",
        })
    
    leaderboard_df = pd.DataFrame(leaderboard)
    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)
    
    st.markdown('<div class="bcg-divider"></div>', unsafe_allow_html=True)
    
    # Radar chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">Radar Comparison</div>', unsafe_allow_html=True)
        fig = create_competitor_radar(df_filtered, col_mapping, selected, dark_mode)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">Rating Comparison</div>', unsafe_allow_html=True)
        fig = create_rating_bar_chart(df_filtered, col_mapping, dark_mode)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="bcg-divider"></div>', unsafe_allow_html=True)
    
    # Category comparison heatmaps
    st.markdown('<div class="section-header">Category Performance Comparison</div>', unsafe_allow_html=True)
    
    pos_cats, neg_cats, _ = get_category_columns(df_filtered)
    
    tab1, tab2 = st.tabs(["Pain Points", "Strengths"])
    
    with tab1:
        if neg_cats:
            fig = create_category_heatmap(df_filtered, col_mapping, neg_cats, "Negative Categories by Competitor", is_negative=True, dark_mode=dark_mode)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if pos_cats:
            fig = create_category_heatmap(df_filtered, col_mapping, pos_cats, "Positive Categories by Competitor", is_negative=False, dark_mode=dark_mode)
            if fig:
                st.plotly_chart(fig, use_container_width=True)


def page_regional_view():
    """Render Regional/City View page."""
    st.markdown('<div class="main-header">Regional Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.info("Please load data from the sidebar to view analytics.")
        return
    
    df = st.session_state.df.copy()
    col_mapping = st.session_state.data_info.get("col_mapping", detect_columns(df))
    dark_mode = st.session_state.dark_mode
    
    if not col_mapping["city"]:
        st.warning("No city column detected in data. Regional analysis unavailable.")
        return
    
    city_col = col_mapping["city"]
    
    # City performance chart
    st.markdown('<div class="section-header">City Performance Rankings</div>', unsafe_allow_html=True)
    fig = create_city_performance(df, col_mapping, dark_mode)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="bcg-divider"></div>', unsafe_allow_html=True)
    
    # City KPIs table
    st.markdown('<div class="section-header">City Statistics</div>', unsafe_allow_html=True)
    
    city_stats = []
    for city in df[city_col].dropna().unique():
        sub = df[df[city_col] == city]
        n_reviews = len(sub)
        avg_rating = sub[col_mapping["rating"]].mean() if col_mapping["rating"] else None
        n_businesses = sub[col_mapping["business"]].nunique() if col_mapping["business"] else None
        pct_pos = (sub[col_mapping["sentiment"]] == "Positif").mean() * 100 if col_mapping["sentiment"] else None
        
        city_stats.append({
            "City": city,
            "Reviews": n_reviews,
            "Businesses": n_businesses or "-",
            "Avg Rating": f"{avg_rating:.2f}" if avg_rating else "-",
            "% Positive": f"{pct_pos:.1f}%" if pct_pos else "-",
        })
    
    city_df = pd.DataFrame(city_stats).sort_values("Reviews", ascending=False)
    st.dataframe(city_df.head(30), use_container_width=True, hide_index=True)
    
    st.markdown('<div class="bcg-divider"></div>', unsafe_allow_html=True)
    
    # Business × City matrix
    if col_mapping["business"] and col_mapping["rating"]:
        st.markdown('<div class="section-header">Business × City Performance Matrix</div>', unsafe_allow_html=True)
        
        biz_col = col_mapping["business"]
        rating_col = col_mapping["rating"]
        
        top_cities = df[city_col].value_counts().head(15).index.tolist()
        businesses = sorted(df[biz_col].dropna().unique())
        
        matrix_data = []
        for biz in businesses:
            row = {"Business": biz}
            for city in top_cities:
                sub = df[(df[biz_col] == biz) & (df[city_col] == city)]
                row[city] = round(sub[rating_col].mean(), 2) if len(sub) >= 3 else None
            matrix_data.append(row)
        
        matrix_df = pd.DataFrame(matrix_data).set_index("Business")
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix_df.values,
            x=matrix_df.columns.tolist(),
            y=matrix_df.index.tolist(),
            colorscale=[[0, BCG_COLORS["red"]], [0.5, BCG_COLORS["yellow"]], [1, BCG_COLORS["green"]]],
            zmin=1, zmax=5,
            text=[[f"{v:.1f}" if pd.notna(v) else "-" for v in row] for row in matrix_df.values],
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="Business: %{y}<br>City: %{x}<br>Rating: %{z:.2f}<extra></extra>",
            colorbar=dict(title="Avg Rating"),
        ))
        
        fig.update_layout(**get_plotly_layout(dark_mode))
        fig.update_layout(
            height=max(400, len(businesses) * 45),
            xaxis=dict(tickangle=-45),
        )
        
        st.plotly_chart(fig, use_container_width=True)


def page_temporal_view():
    """Render Temporal View page."""
    st.markdown('<div class="main-header">Temporal Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.info("Please load data from the sidebar to view analytics.")
        return
    
    df = st.session_state.df.copy()
    col_mapping = st.session_state.data_info.get("col_mapping", detect_columns(df))
    dark_mode = st.session_state.dark_mode
    
    if "_month" not in df.columns:
        st.warning("No date column detected. Temporal analysis unavailable.")
        return
    
    # Apply filters
    if st.session_state.selected_competitors and col_mapping["business"]:
        df = df[df[col_mapping["business"]].isin(st.session_state.selected_competitors)]
    
    # Overall trend
    st.markdown('<div class="section-header">Overall Trend</div>', unsafe_allow_html=True)
    fig = create_temporal_chart(df, col_mapping, dark_mode)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="bcg-divider"></div>', unsafe_allow_html=True)
    
    # Rating by business over time
    if col_mapping["business"] and col_mapping["rating"]:
        st.markdown('<div class="section-header">Rating Trends by Business</div>', unsafe_allow_html=True)
        
        biz_col = col_mapping["business"]
        rating_col = col_mapping["rating"]
        
        monthly_by_biz = df.groupby(["_month", biz_col]).agg(
            avg_rating=(rating_col, "mean"),
            n_reviews=(rating_col, "size"),
        ).reset_index()
        
        fig = px.line(
            monthly_by_biz,
            x="_month",
            y="avg_rating",
            color=biz_col,
            title="<b>Rating Trends by Business</b>",
            markers=True,
        )
        
        fig.update_layout(**get_plotly_layout(dark_mode))
        fig.update_layout(
            height=500,
            yaxis=dict(range=[1, 5]),
            xaxis_title="Month",
            yaxis_title="Average Rating",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="bcg-divider"></div>', unsafe_allow_html=True)
    
    # Sentiment over time
    if col_mapping["sentiment"] and col_mapping["sentiment"] in df.columns:
        st.markdown('<div class="section-header">Sentiment Evolution</div>', unsafe_allow_html=True)
        
        sentiment_monthly = df.groupby(["_month", col_mapping["sentiment"]]).size().unstack(fill_value=0)
        sentiment_monthly_pct = sentiment_monthly.div(sentiment_monthly.sum(axis=1), axis=0) * 100
        sentiment_monthly_pct = sentiment_monthly_pct.reset_index()
        
        sentiment_long = sentiment_monthly_pct.melt(id_vars="_month", var_name="Sentiment", value_name="Percentage")
        
        fig = px.area(
            sentiment_long,
            x="_month",
            y="Percentage",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            title="<b>Sentiment Distribution Over Time</b>",
        )
        
        fig.update_layout(**get_plotly_layout(dark_mode))
        fig.update_layout(
            height=400,
            yaxis=dict(range=[0, 100]),
            xaxis_title="Month",
            yaxis_title="Percentage (%)",
        )
        
        st.plotly_chart(fig, use_container_width=True)


def page_map_view():
    """Render Map View page with Folium."""
    st.markdown('<div class="main-header">Geographic Map View</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.info("Please load data from the sidebar to view analytics.")
        return
    
    df = st.session_state.df.copy()
    col_mapping = st.session_state.data_info.get("col_mapping", detect_columns(df))
    
    # Check for coordinate columns
    lat_col = None
    lng_col = None
    for col in df.columns:
        col_lower = col.lower()
        if "lat" in col_lower:
            lat_col = col
        elif "lng" in col_lower or "lon" in col_lower:
            lng_col = col
    
    try:
        import folium
        from streamlit_folium import st_folium
        HAS_FOLIUM = True
    except ImportError:
        HAS_FOLIUM = False
    
    if not HAS_FOLIUM:
        st.warning("Folium package not installed. Run: `pip install folium streamlit-folium`")
        st.info("Showing city-level analysis instead.")
        
        # Fallback to city analysis
        if col_mapping["city"]:
            fig = create_city_performance(df, col_mapping, st.session_state.dark_mode)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        return
    
    if not lat_col or not lng_col:
        st.warning("No coordinate columns (lat/lng) found in data.")
        st.info("To enable map view, ensure your data includes latitude and longitude columns.")
        
        # Show city-level as fallback
        if col_mapping["city"]:
            st.markdown('<div class="section-header">City-Level Analysis</div>', unsafe_allow_html=True)
            fig = create_city_performance(df, col_mapping, st.session_state.dark_mode)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        return
    
    # Filter valid coordinates
    df_map = df.dropna(subset=[lat_col, lng_col]).copy()
    
    if df_map.empty:
        st.warning("No valid coordinates found in data.")
        return
    
    st.markdown('<div class="section-header">Interactive Map</div>', unsafe_allow_html=True)
    
    # Create map centered on Morocco
    center_lat = df_map[lat_col].mean()
    center_lng = df_map[lng_col].mean()
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=6, tiles="CartoDB positron")
    
    # Add markers
    for _, row in df_map.iterrows():
        lat = row[lat_col]
        lng = row[lng_col]
        rating = row[col_mapping["rating"]] if col_mapping["rating"] else None
        business = row[col_mapping["business"]] if col_mapping["business"] else "Unknown"
        city = row[col_mapping["city"]] if col_mapping["city"] else "Unknown"
        
        # Color based on rating
        if rating:
            if rating >= 4:
                color = "green"
            elif rating >= 3:
                color = "orange"
            else:
                color = "red"
        else:
            color = "gray"
        
        popup_text = f"<b>{business}</b><br>City: {city}<br>Rating: {rating if rating else 'N/A'}"
        
        folium.CircleMarker(
            location=[lat, lng],
            radius=8,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_text, max_width=200),
        ).add_to(m)
    
    # Display map
    st_folium(m, width=1200, height=600)
    
    st.markdown('<div class="bcg-divider"></div>', unsafe_allow_html=True)
    
    # Legend
    st.markdown("""
**Map Legend:**
- Green: Rating >= 4.0
- Orange: Rating 3.0 - 3.9
- Red: Rating < 3.0
    """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    # Apply theme CSS
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Main content with tabs
    tabs = st.tabs([
        "Business View",
        "Competitor View", 
        "Regional View",
        "Temporal View",
        "Map View"
    ])
    
    with tabs[0]:
        page_business_view()
    
    with tabs[1]:
        page_competitor_view()
    
    with tabs[2]:
        page_regional_view()
    
    with tabs[3]:
        page_temporal_view()
    
    with tabs[4]:
        page_map_view()


if __name__ == "__main__":
    main()
