"""
Shared utilities for Review Analyzer Pipeline
Contains reusable functions for API calls, file operations, and data processing
"""
import json
import time
import csv
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from review_analyzer.discover import _CANONICAL_RX
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import requests
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# =========================
# Custom Exceptions
# =========================
class SerpAPIError(Exception):
    """Base exception for SerpAPI errors"""
    pass


class RateLimitError(SerpAPIError):
    """Raised when API rate limit is hit"""
    pass


class AuthenticationError(SerpAPIError):
    """Raised when API authentication fails"""
    pass


class InvalidPlaceIDError(ValueError):
    """Raised when place_id format is invalid"""
    pass


# =========================
# SerpAPI Client
# =========================
class SerpAPIClient:
    """
    Robust client for SerpAPI with automatic retry logic
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = None,
        debug: bool = False
    ):
        """
        Initialize SerpAPI client
        
        Args:
            api_key: SerpAPI key (uses config if None)
            max_retries: Max retry attempts (uses config if None)
            debug: Enable debug logging
        """
        self.api_key = api_key or config.SERPAPI_CONFIG["api_key"]
        self.max_retries = max_retries or config.SERPAPI_CONFIG["max_retries"]
        self.debug = debug
        
        if not self.api_key:
            raise AuthenticationError("SerpAPI key is required")
    
    @retry(
        reraise=True,
        stop=stop_after_attempt(config.SERPAPI_CONFIG["max_retries"]),
        wait=wait_exponential(
            multiplier=1,
            min=config.SERPAPI_CONFIG["retry_min_wait"],
            max=config.SERPAPI_CONFIG["retry_max_wait"]
        ),
        retry=retry_if_exception_type(RateLimitError),
    )
    def request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make SerpAPI request with retry logic
        
        Args:
            params: Request parameters
            
        Returns:
            API response as dictionary
            
        Raises:
            AuthenticationError: If 401 error
            RateLimitError: If 429 error (will retry)
            SerpAPIError: For other errors
        """
        params["api_key"] = self.api_key
        
        try:
            response = requests.get(
                "https://serpapi.com/search.json",
                params=params,
                timeout=config.SERPAPI_CONFIG["timeout"]
            )
            
            # Try to parse JSON
            try:
                data = response.json()
            except Exception:
                data = {}
            
            # Handle specific status codes
            if response.status_code == 401:
                error_msg = data.get("error", "Unauthorized")
                raise AuthenticationError(f"401: {error_msg}")
            
            if response.status_code == 429:
                logger.warning("Rate limit hit, will retry...")
                raise RateLimitError("429: Rate limit exceeded")
            
            if response.status_code >= 400:
                error_msg = data.get("error", response.text)
                raise SerpAPIError(
                    f"{response.status_code} error: {error_msg}"
                )
            
            # Check for API-level errors
            if "error" in data:
                logger.warning(f"API error: {data['error']}")
            
            if self.debug:
                logger.debug(f"SerpAPI request successful: {params.get('engine')}")
            
            return data
            
        except (RateLimitError, AuthenticationError):
            raise
        except Exception as e:
            logger.error(f"SerpAPI request failed: {e}")
            raise SerpAPIError(f"Request failed: {e}")
    
    def search_google_maps(self, query: str, ll: str = None, next_page_token: str = None) -> dict:
        """
        Google Maps Search via SerpApi (MAPS-ONLY)
        Returns JSON with 'local_results' (primary) or 'places' (fallback).
        """
        params = {
            "engine": "google_maps",
            "type": "search",
            "q": query,
            "hl": config.SERPAPI_CONFIG.get("hl", "fr"),
            "gl": config.SERPAPI_CONFIG.get("gl", "ma"),
            "api_key": self.api_key,
        }
        # IMPORTANT: do NOT force google_domain unless you know it's supported.
        # If you want to force Morocco ccTLD, use 'google.co.ma' (NOT google.com.ma).
        gd = config.SERPAPI_CONFIG.get("google_domain")
        if gd:
            params["google_domain"] = gd

        if ll:
            params["ll"] = ll
        if next_page_token:
            params["next_page_token"] = next_page_token

        r = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    
    def search_google_local(
        self,
        query: str,
        location: str,
        start: Optional[int] = None,
        hl: str = None,
        gl: str = None
    ) -> Dict[str, Any]:
        """
        Search Google Local
        
        Args:
            query: Search query
            location: Location string (e.g., "Rabat, Morocco")
            start: Pagination start
            hl: Language
            gl: Geo-location
            
        Returns:
            API response
        """
        params = {
            "engine": "google_local",
            "q": query,
            "location": location,
            "hl": hl or config.SERPAPI_CONFIG["hl"],
            "gl": gl or config.SERPAPI_CONFIG["gl"],
            "num": 20,
        }
        
        if start is not None:
            params["start"] = start
        
        return self.request(params)
    
    def get_place_details(self, place_id: str = None, data_id: str = None) -> str | None:
        """
        Resolve canonical place_id (ChIJ...) from either a place_id or a data_id
        using SerpApi's google_maps_place engine.
        """
        if not (place_id or data_id):
            return None

        params = {
            "engine": "google_maps_place",
            "hl": config.SERPAPI_CONFIG.get("hl", "fr"),
            "api_key": self.api_key,
        }
        if data_id:
            params["data_id"] = data_id
        else:
            params["place_id"] = place_id

        r = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        pr = (data or {}).get("place_results") or {}
        pid = pr.get("place_id")
        return pid if pid and _CANONICAL_RX.match(pid) else None
    
    def get_reviews(
        self,
        place_id: str,
        hl: str = None,
        sort_by: Optional[str] = None,
        no_cache: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all reviews for a place (handles pagination automatically)
        
        Args:
            place_id: Google Maps place_id
            hl: Language
            sort_by: Sort order ('most_relevant', 'newest')
            no_cache: Bypass cache
            
        Returns:
            List of review dictionaries
        """
        all_reviews = []
        next_page_token = None
        page = 0
        
        while True:
            page += 1
            params = {
                "engine": "google_maps_reviews",
                "place_id": place_id,
                "hl": hl or config.SERPAPI_CONFIG["hl"],
            }
            
            if sort_by:
                params["sort_by"] = sort_by
            if no_cache:
                params["no_cache"] = "true"
            if next_page_token:
                params["next_page_token"] = next_page_token
            
            try:
                data = self.request(params)
                
                if self.debug:
                    review_count = len(data.get("reviews", []))
                    logger.debug(
                        f"Page {page} for {place_id}: {review_count} reviews"
                    )
                
                # Check for API errors
                if "error" in data:
                    logger.warning(
                        f"API error on page {page} for {place_id}: "
                        f"{data['error']}"
                    )
                    break
                
                # Extract reviews
                for item in data.get("reviews", []):
                    user = item.get("user") or {}
                    all_reviews.append({
                        "page": page,
                        "name": user.get("name"),
                        "link": user.get("link"),
                        "thumbnail": user.get("thumbnail"),
                        "local_guide": user.get("local_guide"),
                        "rating": item.get("rating"),
                        "date": item.get("date"),
                        "snippet": item.get("snippet") or item.get("content"),
                        "images": item.get("images"),
                    })
                
                # Check for next page
                pagination = data.get("serpapi_pagination", {})
                next_page_token = pagination.get("next_page_token")
                
                if not next_page_token:
                    break
                
                # Rate limiting
                time.sleep(config.SERPAPI_CONFIG["delay_seconds"])
                
            except Exception as e:
                logger.error(f"Failed to get reviews for {place_id}: {e}")
                break
        
        return all_reviews


# =========================
# CSV Utilities
# =========================
def sniff_csv_format(file_path: str) -> Tuple[str, str]:
    """
    Detect CSV delimiter and encoding
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (delimiter, encoding)
    """
    # Check for BOM
    encoding = "utf-8"
    with open(file_path, "rb") as f:
        head = f.read(2048)
        if head.startswith(b"\xef\xbb\xbf"):
            encoding = "utf-8-sig"
    
    # Detect delimiter
    with open(file_path, "r", encoding=encoding, newline="") as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter, encoding


def normalize_column_names(columns: List[str]) -> Dict[str, str]:
    """
    Create mapping of original to normalized column names
    
    Args:
        columns: List of column names
        
    Returns:
        Dictionary mapping original -> normalized (lowercase, stripped)
    """
    return {col: col.strip().lower() for col in columns}


def detect_column(
    df: pd.DataFrame,
    possible_names: List[str],
    required: bool = False
) -> Optional[str]:
    """
    Detect column from list of possible names (case-insensitive)
    
    Args:
        df: DataFrame to search
        possible_names: List of possible column names
        required: Raise error if not found
        
    Returns:
        Detected column name or None
        
    Raises:
        ValueError: If required=True and column not found
    """
    # Normalize all column names
    normalized_cols = {col.lower().strip(): col for col in df.columns}
    
    # Try each possible name
    for name in possible_names:
        normalized_name = name.lower().strip().replace(" ", "").replace("_", "")
        
        for norm_col, original_col in normalized_cols.items():
            norm_col_clean = norm_col.replace(" ", "").replace("_", "")
            if normalized_name == norm_col_clean:
                return original_col
    
    if required:
        raise ValueError(
            f"Required column not found. Tried: {possible_names}. "
            f"Available columns: {list(df.columns)}"
        )
    
    return None


def read_agencies_csv(
    file_path: str,
    place_id_col: Optional[str] = None,
    city_col: Optional[str] = None,
    business_col: Optional[str] = None,
    filter_cities: Optional[List[str]] = None,
    filter_businesses: Optional[List[str]] = None,
    # Deprecated parameters (backward compatibility)
    bank_col: Optional[str] = None,
    filter_banks: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Read locations CSV with automatic column detection and filtering
    
    Args:
        file_path: Path to CSV file
        place_id_col: Place ID column name (auto-detected if None)
        city_col: City column name (auto-detected if None)
        business_col: Business column name (auto-detected if None)
        filter_cities: List of cities to include
        filter_businesses: List of businesses to include
        bank_col: [DEPRECATED] Use business_col instead
        filter_banks: [DEPRECATED] Use filter_businesses instead
        
    Returns:
        DataFrame with standardized columns
    """
    # Handle deprecated parameters
    import warnings
    if bank_col is not None and business_col is None:
        warnings.warn(
            "bank_col is deprecated, use business_col instead",
            DeprecationWarning,
            stacklevel=2
        )
        business_col = bank_col
    
    if filter_banks is not None and filter_businesses is None:
        warnings.warn(
            "filter_banks is deprecated, use filter_businesses instead",
            DeprecationWarning,
            stacklevel=2
        )
        filter_businesses = filter_banks
    # Detect format
    delimiter, encoding = sniff_csv_format(file_path)
    logger.debug(f"CSV format: delimiter='{delimiter}', encoding='{encoding}'")
    
    # Read CSV
    df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
    
    # Detect place_id column
    if place_id_col is None:
        place_id_col = detect_column(
            df,
            ["place_id", "placeid", "google_place_id", "data_id", "place id"],
            required=True
        )
    
    logger.info(f"Using place_id column: '{place_id_col}'")
    
    # Detect city column (optional)
    if city_col is None:
        city_col = detect_column(
            df,
            ["city", "ville", "perf_city", "location", "lieu"]
        )
        if city_col:
            logger.info(f"Using city column: '{city_col}'")
    
    # Detect business column (optional)
    if business_col is None:
        business_col = detect_column(
            df,
            ["business", "business_name", "business name",
             "bank", "bank_group", "banque", "bank name", "bank_name",
             "company", "organization", "brand"]
        )
        if business_col:
            logger.info(f"Using business column: '{business_col}'")
    
    # Add standardized columns
    df["_place_id"] = df[place_id_col].fillna("").astype(str).str.strip()
    
    if city_col:
        df["_city"] = df[city_col].fillna("").astype(str).str.strip()
    else:
        df["_city"] = ""
    
    if business_col:
        df["_business"] = df[business_col].fillna("").astype(str).str.strip()
    else:
        df["_business"] = ""
    
    # Filter by place_id (must have value)
    df = df[df["_place_id"] != ""].copy()
    
    # Filter by cities
    if filter_cities and city_col:
        filter_cities_lower = [c.lower() for c in filter_cities]
        df = df[df["_city"].str.lower().isin(filter_cities_lower)].copy()
        logger.info(f"Filtered to cities: {filter_cities}")
    
    # Filter by businesses
    if filter_businesses and business_col:
        filter_businesses_lower = [b.lower() for b in filter_businesses]
        df = df[df["_business"].str.lower().isin(
            filter_businesses_lower
        )].copy()
        logger.info(f"Filtered to businesses: {filter_businesses}")
    
    logger.info(f"Loaded {len(df)} locations from CSV")
    
    return df


# =========================
# JSON Utilities
# =========================
def read_json(file_path: Path) -> Dict[str, Any]:
    """
    Read JSON file safely
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary (empty if file doesn't exist or is invalid)
    """
    if not file_path.exists():
        return {}
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read JSON {file_path}: {e}")
        return {}


def write_json(file_path: Path, data: Dict[str, Any], atomic: bool = True):
    """
    Write JSON file safely
    
    Args:
        file_path: Path to output file
        data: Data to write
        atomic: Use atomic write (write to temp, then rename)
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if atomic:
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path.replace(file_path)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# Validation Utilities
# =========================
def validate_reviews(reviews: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validate review data structure
    
    Args:
        reviews: List of review dictionaries
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(reviews, list):
        return False, "Reviews must be a list"
    
    for i, review in enumerate(reviews):
        if not isinstance(review, dict):
            return False, f"Review {i} is not a dictionary"
        
        # Check for required fields
        if "snippet" not in review:
            return False, f"Review {i} missing 'snippet' field"
        
        # Validate rating if present
        if "rating" in review:
            rating = review["rating"]
            if rating is not None:
                if not (config.MIN_RATING <= rating <= config.MAX_RATING):
                    return (
                        False,
                        f"Review {i} has invalid rating: {rating}"
                    )
    
    return True, ""


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate distance between two coordinates in meters
    
    Args:
        lat1, lon1: First coordinate
        lat2, lon2: Second coordinate
        
    Returns:
        Distance in meters
    """
    import math
    
    R = 6371000.0  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dlon / 2) ** 2
    )
    
    return 2 * R * math.asin(math.sqrt(a))


# =========================
# Checkpoint Management
# =========================
class CheckpointManager:
    """
    Manages checkpoints for long-running operations
    """
    
    def __init__(self, checkpoint_path: Path):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    def save(self, data: Dict[str, Any]):
        """Save checkpoint"""
        write_json(self.checkpoint_path, data, atomic=True)
        logger.debug(f"Checkpoint saved: {self.checkpoint_path}")
    
    def load(self) -> Dict[str, Any]:
        """Load checkpoint"""
        return read_json(self.checkpoint_path)
    
    def clear(self):
        """Clear checkpoint"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.debug(f"Checkpoint cleared: {self.checkpoint_path}")


# =========================
# Progress Tracking
# =========================
def create_progress_bar(total: int, desc: str = "Processing"):
    """
    Create progress bar (using tqdm if available)
    
    Args:
        total: Total items
        desc: Description
        
    Returns:
        Progress bar or None if tqdm not installed
    """
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc)
    except ImportError:
        logger.warning("tqdm not installed, progress bar disabled")
        return None
