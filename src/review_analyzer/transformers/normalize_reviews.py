# src/review_analyzer/transformers/normalize_reviews.py
# date parsing, cleaning
from __future__ import annotations
import re
import unicodedata
from datetime import datetime, timedelta, timezone
import pandas as pd

# Regex pattern for French relative dates
# Uses \s to match any whitespace (including non-breaking spaces after normalization)
# Supports both numeric ("2 ans") and word ("un an", "une semaine") quantities
_FR_REL_RX = re.compile(
    r"(?i)^(modifi[eé]\s+)?il\s+y\s+a\s+(un|une|\d+)\s+(semaine|semaines|jour|jours|mois|an|ans|heure|heures|minute|minutes)$"
)

# Africa/Casablanca is UTC+1 most of the year; if you want full tz, add 'pytz' or 'zoneinfo'
CASABLANCA_TZ = timezone(timedelta(hours=1))


def _normalize_whitespace(text: str) -> str:
    """Normalize all whitespace characters (including non-breaking spaces) to regular spaces."""
    if not text:
        return text
    # Replace non-breaking spaces and other unicode whitespace with regular space
    # \xa0 is the most common non-breaking space from Google
    text = text.replace('\xa0', ' ')
    text = text.replace('\u00a0', ' ')  # Same as \xa0
    text = text.replace('\u202f', ' ')  # Narrow no-break space
    # Normalize unicode and collapse multiple spaces
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_relative_french_date(text: str, now: datetime | None = None) -> datetime | None:
    """
    Parse French relative timestamps like:
      - "il y a 2 ans"
      - "il y a un an" (word form)
      - "Modifié il y a 11 mois"
      - "il y a 3 semaines"
      - "il y a 5 jours"
      - "il y a 12 heures"
      - "il y a une semaine"
    
    Handles non-breaking spaces (\\xa0) commonly found in Google data.
    Returns a naive UTC datetime.
    """
    if not text:
        return None
    
    # Normalize whitespace (critical for Google data with non-breaking spaces)
    text = _normalize_whitespace(text)
    
    m = _FR_REL_RX.search(text)
    if not m:
        return None
    
    # Handle both numeric and word quantities
    qty_str = m.group(2).lower()
    if qty_str in ('un', 'une'):
        qty = 1
    else:
        qty = int(qty_str)
    
    unit = m.group(3).lower()

    now = now or datetime.now(timezone.utc)
    # Treat 'mois' as 30 days approximate; for calendar-accurate, use dateutil.relativedelta
    if unit.startswith("minute"):
        dt = now - timedelta(minutes=qty)
    elif unit.startswith("heure"):
        dt = now - timedelta(hours=qty)
    elif unit.startswith("jour"):
        dt = now - timedelta(days=qty)
    elif unit.startswith("semaine"):
        dt = now - timedelta(weeks=qty)
    elif unit == "mois":
        dt = now - timedelta(days=30 * qty)
    elif unit in ("an", "ans"):
        dt = now - timedelta(days=365 * qty)
    else:
        return None

    # normalize to UTC naive
    return dt.astimezone(timezone.utc).replace(tzinfo=None)

def normalize_reviews_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize raw review fields:
      - compute 'created_at' (UTC naive) from French relative 'date' text
      - standardize rating as int (1-5)
      - ensure lat/lng numeric
      - clean text fields
      - derive month for temporal analysis
    
    Args:
        df: Raw reviews DataFrame
        
    Returns:
        Normalized DataFrame with standardized fields
        
    Note:
        - Handles missing values gracefully
        - Preserves original columns
        - Adds 'created_at' and 'month' if date parsing succeeds
    """
    if df is None or df.empty:
        return df
    
    out = df.copy()

    # Rating → int (clip to 1..5)
    if "rating" in out.columns:
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
        out["rating"] = out["rating"].round().clip(lower=1, upper=5)
        out["rating"] = out["rating"].astype("Int64")  # Nullable integer
    elif "review_rating" in out.columns:
        # Handle alternative column name
        out["rating"] = pd.to_numeric(out["review_rating"], errors="coerce")
        out["rating"] = out["rating"].round().clip(lower=1, upper=5)
        out["rating"] = out["rating"].astype("Int64")

    # Lat/lng → floats
    for col in ("lat", "lng", "latitude", "longitude"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Created_at from 'date' string (relative French format)
    # Support alternative column names: date, review_date
    date_col = None
    for col in ["date", "review_date"]:
        if col in out.columns:
            date_col = col
            break
    
    if date_col and "created_at" not in out.columns:
        out["created_at"] = out[date_col].apply(
            lambda x: parse_relative_french_date(x) if pd.notna(x) else None
        )
    
    # Text normalization - handle both 'text' and 'review_snippet'
    for text_col in ["text", "review_snippet", "review_text"]:
        if text_col in out.columns:
            out[text_col] = out[text_col].fillna("").astype(str).str.strip()

    # Derive month for fast grouping (if created_at exists)
    if "created_at" in out.columns:
        # Ensure created_at is datetime
        out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
        
        # Add month column for temporal analysis
        valid_dates = out["created_at"].notna()
        if valid_dates.any():
            out.loc[valid_dates, "month"] = (
                out.loc[valid_dates, "created_at"]
                .dt.to_period("M")
                .dt.to_timestamp()
            )
    
    # Normalize city names for consistency
    if "city" in out.columns:
        out["city_normalized"] = (
            out["city"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("ascii")
        )

    return out
