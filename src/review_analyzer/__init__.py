"""
Review Analyzer - Google Maps Review Collection and Classification Pipeline

A production-ready pipeline for:
1. Discovering bank branches via Google Maps
2. Collecting customer reviews
3. Classifying reviews with OpenAI
4. Storing and analyzing data with DuckDB

Usage:
    from review_analyzer import config, utils
    from review_analyzer.discover import DiscoveryEngine
    from review_analyzer.collect import ReviewCollector
    from review_analyzer.classify import ReviewClassifier
    from review_analyzer.database import DatabaseManager, AnalyticsQueries
"""

__version__ = "2.0.0"
__author__ = "Myriam Rahali"

# Make key modules available at package level
from . import config
from . import utils

# Database module (lazy import to avoid duckdb dependency for basic usage)
def get_database_manager(*args, **kwargs):
    """Get a DatabaseManager instance (lazy import)"""
    from .database import DatabaseManager
    return DatabaseManager(*args, **kwargs)

def get_analytics_queries(db_manager):
    """Get an AnalyticsQueries instance (lazy import)"""
    from .database import AnalyticsQueries
    return AnalyticsQueries(db_manager)

__all__ = [
    "config", 
    "utils", 
    "__version__",
    "get_database_manager",
    "get_analytics_queries"
]
