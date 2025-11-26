"""
Transformers Module - Data normalization and enrichment

This module provides data transformation utilities for the review pipeline:
- normalize_reviews: Parse dates, clean fields, standardize formats
- geocode: Assign Moroccan regions using GeoJSON polygons
- aggregates: Pre-compute rollups for analysis and dashboards

Pipeline Stage: Between collection and classification
Input:  Raw reviews from 02_interim/collection/
Output: Normalized reviews in 02_interim/transform/
"""

from .normalize_reviews import (
    normalize_reviews_df,
    parse_relative_french_date,
    CASABLANCA_TZ,
)

from .geocode import (
    add_region,
    add_region_by_city,
    load_regions_geojson,
    build_spatial_index,
    assign_region_row,
    CITY_REGION_MAPPING,
)

from .aggregates import (
    build_aggregates,
)

__all__ = [
    # Normalization
    "normalize_reviews_df",
    "parse_relative_french_date",
    "CASABLANCA_TZ",
    # Geocoding
    "add_region",
    "add_region_by_city",
    "load_regions_geojson",
    "build_spatial_index",
    "assign_region_row",
    "CITY_REGION_MAPPING",
    # Aggregation
    "build_aggregates",
]
