"""
Pre-built analytics queries for Review Analyzer
Provides common analytical operations using DuckDB's efficient aggregations
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import pandas as pd

from .models import CATEGORY_COLUMN_MAP

logger = logging.getLogger(__name__)


class AnalyticsQueries:
    """
    Collection of analytical queries for review data
    Uses DuckDB's columnar engine for fast aggregations
    """
    
    def __init__(self, db_manager):
        """
        Initialize with database manager
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
        self.conn = db_manager.connection
    
    # =========================
    # Overview Statistics
    # =========================
    def get_overview_stats(self) -> Dict[str, Any]:
        """Get high-level statistics about the dataset"""
        sql = """
        SELECT
            COUNT(DISTINCT p.place_id) AS total_places,
            COUNT(DISTINCT p.business) AS total_businesses,
            COUNT(DISTINCT p.city) AS total_cities,
            COUNT(*) AS total_reviews,
            AVG(r.rating) AS avg_rating,
            MIN(r.review_date_parsed) AS earliest_review,
            MAX(r.review_date_parsed) AS latest_review
        FROM reviews r
        LEFT JOIN places p ON r.place_id = p.place_id
        """
        
        result = self.conn.execute(sql).fetchdf()
        
        if result.empty:
            return {}
        
        row = result.iloc[0]
        
        # Get sentiment distribution
        sentiment_sql = """
        SELECT 
            sentiment,
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
        FROM classifications
        GROUP BY sentiment
        """
        sentiment_df = self.conn.execute(sentiment_sql).fetchdf()
        
        return {
            "total_places": int(row["total_places"]),
            "total_businesses": int(row["total_businesses"]),
            "total_cities": int(row["total_cities"]),
            "total_reviews": int(row["total_reviews"]),
            "avg_rating": float(row["avg_rating"]) if row["avg_rating"] else 0,
            "earliest_review": row["earliest_review"],
            "latest_review": row["latest_review"],
            "sentiment_distribution": sentiment_df.to_dict("records")
        }
    
    # =========================
    # Business Analytics
    # =========================
    def get_business_stats(
        self,
        business: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get statistics per business
        
        Args:
            business: Optional filter for specific business
            
        Returns:
            DataFrame with business statistics
        """
        where = f"WHERE r.business = '{business}'" if business else ""
        
        sql = f"""
        SELECT 
            r.business,
            COUNT(*) AS total_reviews,
            COUNT(DISTINCT r.place_id) AS branch_count,
            COUNT(DISTINCT r.city) AS city_count,
            ROUND(AVG(r.rating), 2) AS avg_rating,
            COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) AS positive_count,
            COUNT(CASE WHEN c.sentiment = 'Négatif' THEN 1 END) AS negative_count,
            COUNT(CASE WHEN c.sentiment = 'Neutre' THEN 1 END) AS neutral_count,
            ROUND(COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS positive_rate,
            ROUND(COUNT(CASE WHEN c.sentiment = 'Négatif' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS negative_rate
        FROM reviews r
        LEFT JOIN classifications c ON r.id = c.review_id
        {where}
        GROUP BY r.business
        ORDER BY total_reviews DESC
        """
        
        return self.conn.execute(sql).fetchdf()
    
    def get_business_ranking(
        self,
        metric: str = "avg_rating",
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Rank businesses by specified metric
        
        Args:
            metric: Ranking metric (avg_rating, positive_rate, total_reviews)
            top_n: Number of top businesses to return
            
        Returns:
            DataFrame with ranked businesses
        """
        valid_metrics = {
            "avg_rating": "ROUND(AVG(r.rating), 2)",
            "positive_rate": "ROUND(COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2)",
            "total_reviews": "COUNT(*)"
        }
        
        if metric not in valid_metrics:
            metric = "avg_rating"
        
        sql = f"""
        SELECT 
            r.business,
            {valid_metrics[metric]} AS {metric},
            COUNT(*) AS review_count
        FROM reviews r
        LEFT JOIN classifications c ON r.id = c.review_id
        GROUP BY r.business
        HAVING COUNT(*) >= 10
        ORDER BY {metric} DESC
        LIMIT {top_n}
        """
        
        return self.conn.execute(sql).fetchdf()
    
    # =========================
    # City Analytics
    # =========================
    def get_city_stats(
        self,
        city: Optional[str] = None
    ) -> pd.DataFrame:
        """Get statistics per city"""
        where = f"WHERE r.city = '{city}'" if city else ""
        
        sql = f"""
        SELECT 
            r.city,
            COUNT(*) AS total_reviews,
            COUNT(DISTINCT r.place_id) AS branch_count,
            COUNT(DISTINCT r.business) AS business_count,
            ROUND(AVG(r.rating), 2) AS avg_rating,
            COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) AS positive_count,
            COUNT(CASE WHEN c.sentiment = 'Négatif' THEN 1 END) AS negative_count,
            ROUND(COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS positive_rate
        FROM reviews r
        LEFT JOIN classifications c ON r.id = c.review_id
        {where}
        GROUP BY r.city
        ORDER BY total_reviews DESC
        """
        
        return self.conn.execute(sql).fetchdf()
    
    def get_best_in_city(
        self,
        city: str,
        metric: str = "avg_rating"
    ) -> pd.DataFrame:
        """Get best performing business in a city"""
        sql = f"""
        SELECT 
            r.business,
            r.city,
            ROUND(AVG(r.rating), 2) AS avg_rating,
            COUNT(*) AS review_count,
            ROUND(COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS positive_rate
        FROM reviews r
        LEFT JOIN classifications c ON r.id = c.review_id
        WHERE r.city = ?
        GROUP BY r.business, r.city
        HAVING COUNT(*) >= 5
        ORDER BY {metric} DESC
        LIMIT 1
        """
        
        return self.conn.execute(sql, [city]).fetchdf()
    
    def get_city_winners(self) -> pd.DataFrame:
        """Get best business in each city by rating"""
        sql = """
        WITH city_rankings AS (
            SELECT 
                r.city,
                r.business,
                ROUND(AVG(r.rating), 2) AS avg_rating,
                COUNT(*) AS review_count,
                ROW_NUMBER() OVER (PARTITION BY r.city ORDER BY AVG(r.rating) DESC) AS rank
            FROM reviews r
            GROUP BY r.city, r.business
            HAVING COUNT(*) >= 5
        )
        SELECT city, business, avg_rating, review_count
        FROM city_rankings
        WHERE rank = 1
        ORDER BY city
        """
        
        return self.conn.execute(sql).fetchdf()
    
    # =========================
    # Temporal Analytics
    # =========================
    def get_temporal_trends(
        self,
        granularity: str = "month",
        business: Optional[str] = None,
        city: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get temporal trends for reviews
        
        Args:
            granularity: Time granularity (day, week, month, quarter, year)
            business: Optional business filter
            city: Optional city filter
            
        Returns:
            DataFrame with temporal statistics
        """
        trunc_map = {
            "day": "day",
            "week": "week", 
            "month": "month",
            "quarter": "quarter",
            "year": "year"
        }
        
        trunc = trunc_map.get(granularity, "month")
        
        conditions = ["r.review_date_parsed IS NOT NULL"]
        if business:
            conditions.append(f"r.business = '{business}'")
        if city:
            conditions.append(f"r.city = '{city}'")
        
        where = " AND ".join(conditions)
        
        sql = f"""
        SELECT 
            DATE_TRUNC('{trunc}', r.review_date_parsed) AS period,
            COUNT(*) AS review_count,
            ROUND(AVG(r.rating), 2) AS avg_rating,
            COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) AS positive_count,
            COUNT(CASE WHEN c.sentiment = 'Négatif' THEN 1 END) AS negative_count,
            ROUND(COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS positive_rate
        FROM reviews r
        LEFT JOIN classifications c ON r.id = c.review_id
        WHERE {where}
        GROUP BY DATE_TRUNC('{trunc}', r.review_date_parsed)
        ORDER BY period
        """
        
        return self.conn.execute(sql).fetchdf()
    
    def get_recent_reviews(
        self,
        days: int = 30,
        business: Optional[str] = None,
        sentiment: Optional[str] = None
    ) -> pd.DataFrame:
        """Get reviews from the last N days"""
        conditions = [f"r.review_date_parsed >= CURRENT_DATE - INTERVAL '{days}' DAY"]
        
        if business:
            conditions.append(f"r.business = '{business}'")
        if sentiment:
            conditions.append(f"c.sentiment = '{sentiment}'")
        
        where = " AND ".join(conditions)
        
        sql = f"""
        SELECT 
            r.id,
            r.business,
            r.city,
            r.text,
            r.rating,
            r.review_date_parsed,
            c.sentiment,
            c.rationale
        FROM reviews r
        LEFT JOIN classifications c ON r.id = c.review_id
        WHERE {where}
        ORDER BY r.review_date_parsed DESC
        """
        
        return self.conn.execute(sql).fetchdf()
    
    # =========================
    # Category Analytics
    # =========================
    def get_category_distribution(
        self,
        business: Optional[str] = None,
        city: Optional[str] = None,
        category_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get distribution of categories
        
        Args:
            business: Optional business filter
            city: Optional city filter
            category_type: Filter by type (positive, negative, neutral)
            
        Returns:
            DataFrame with category counts and percentages
        """
        conditions = []
        if business:
            conditions.append(f"r.business = '{business}'")
        if city:
            conditions.append(f"r.city = '{city}'")
        
        where = " AND ".join(conditions) if conditions else "1=1"
        
        # We need to unnest the categories JSON
        sql = f"""
        WITH category_data AS (
            SELECT 
                r.id AS review_id,
                r.business,
                r.city,
                UNNEST(
                    CASE 
                        WHEN c.categories_json IS NOT NULL AND c.categories_json != '[]'
                        THEN from_json(c.categories_json, '["json"]')
                        ELSE []
                    END
                ) AS category_obj
            FROM reviews r
            LEFT JOIN classifications c ON r.id = c.review_id
            WHERE {where}
        )
        SELECT 
            category_obj->>'label' AS category,
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
        FROM category_data
        WHERE category_obj->>'label' IS NOT NULL
        GROUP BY category_obj->>'label'
        ORDER BY count DESC
        """
        
        try:
            return self.conn.execute(sql).fetchdf()
        except Exception as e:
            # Fallback: simple category count from review_categories table
            logger.warning(f"JSON unnest failed, using fallback: {e}")
            return self._get_category_distribution_fallback(business, city)
    
    def _get_category_distribution_fallback(
        self,
        business: Optional[str] = None,
        city: Optional[str] = None
    ) -> pd.DataFrame:
        """Fallback category distribution using review_categories table"""
        conditions = []
        if business:
            conditions.append(f"r.business = '{business}'")
        if city:
            conditions.append(f"r.city = '{city}'")
        
        where = " AND ".join(conditions) if conditions else "1=1"
        
        # Build UNION query for each category column
        category_queries = []
        for col, full_name in CATEGORY_COLUMN_MAP.items():
            # Escape single quotes in category names
            escaped_name = full_name.replace("'", "''")
            category_queries.append(f"""
                SELECT '{escaped_name}' AS category, SUM(rc.{col}) AS count
                FROM review_categories rc
                JOIN reviews r ON rc.review_id = r.id
                WHERE {where}
            """)
        
        sql = f"""
        WITH category_counts AS (
            {' UNION ALL '.join(category_queries)}
        )
        SELECT 
            category,
            count,
            ROUND(count * 100.0 / NULLIF(SUM(count) OVER (), 0), 2) AS percentage
        FROM category_counts
        WHERE count > 0
        ORDER BY count DESC
        """
        
        return self.conn.execute(sql).fetchdf()
    
    def get_category_by_business(
        self,
        category_type: str = "negative"
    ) -> pd.DataFrame:
        """
        Get category breakdown by business
        
        Args:
            category_type: "positive" or "negative"
            
        Returns:
            DataFrame with businesses as rows, categories as columns
        """
        # Select relevant category columns
        if category_type == "positive":
            cols = [c for c in CATEGORY_COLUMN_MAP.keys() 
                   if any(x in c for x in ["accueil", "service_reactif", "conseil", "efficacite", "accessibilite", "satisfaction", "digitale", "autre_positif"])]
        else:
            cols = [c for c in CATEGORY_COLUMN_MAP.keys() 
                   if any(x in c for x in ["attente", "injoignable", "reclamations", "incidents", "frais", "insatisfaction", "manque", "autre_negatif"])]
        
        col_sums = ", ".join([f"SUM(rc.{c}) AS {c}" for c in cols])
        
        sql = f"""
        SELECT 
            r.business,
            COUNT(*) AS total_reviews,
            {col_sums}
        FROM reviews r
        LEFT JOIN review_categories rc ON r.id = rc.review_id
        GROUP BY r.business
        ORDER BY r.business
        """
        
        return self.conn.execute(sql).fetchdf()
    
    def get_top_issues(
        self,
        business: Optional[str] = None,
        top_n: int = 5
    ) -> pd.DataFrame:
        """Get most common negative categories"""
        conditions = []
        if business:
            conditions.append(f"r.business = '{business}'")
        
        where = " AND ".join(conditions) if conditions else "1=1"
        
        # Negative category columns
        neg_cols = [c for c in CATEGORY_COLUMN_MAP.keys() 
                   if any(x in c for x in ["attente", "injoignable", "reclamations", "incidents", "frais", "insatisfaction", "manque", "autre_negatif"])]
        
        category_queries = []
        for col in neg_cols:
            # Escape single quotes in category names
            escaped_name = CATEGORY_COLUMN_MAP[col].replace("'", "''")
            category_queries.append(f"""
                SELECT '{escaped_name}' AS issue, SUM(rc.{col}) AS count
                FROM review_categories rc
                JOIN reviews r ON rc.review_id = r.id
                WHERE {where}
            """)
        
        sql = f"""
        WITH issue_counts AS (
            {' UNION ALL '.join(category_queries)}
        )
        SELECT issue, count
        FROM issue_counts
        WHERE count > 0
        ORDER BY count DESC
        LIMIT {top_n}
        """
        
        return self.conn.execute(sql).fetchdf()
    
    # =========================
    # Comparison Analytics
    # =========================
    def compare_businesses(
        self,
        businesses: List[str]
    ) -> pd.DataFrame:
        """Compare multiple businesses side by side"""
        placeholders = ", ".join([f"'{b}'" for b in businesses])
        
        sql = f"""
        SELECT 
            r.business,
            COUNT(*) AS total_reviews,
            ROUND(AVG(r.rating), 2) AS avg_rating,
            ROUND(COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS positive_rate,
            ROUND(COUNT(CASE WHEN c.sentiment = 'Négatif' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS negative_rate,
            COUNT(DISTINCT r.place_id) AS branch_count,
            COUNT(DISTINCT r.city) AS city_coverage
        FROM reviews r
        LEFT JOIN classifications c ON r.id = c.review_id
        WHERE r.business IN ({placeholders})
        GROUP BY r.business
        ORDER BY avg_rating DESC
        """
        
        return self.conn.execute(sql).fetchdf()
    
    def get_competitive_position(
        self,
        business: str
    ) -> Dict[str, Any]:
        """Get a business's competitive position vs market"""
        # Get business stats
        biz_stats = self.get_business_stats(business)
        if biz_stats.empty:
            return {}
        
        biz = biz_stats.iloc[0]
        
        # Get market averages
        market_sql = """
        SELECT 
            ROUND(AVG(r.rating), 2) AS market_avg_rating,
            ROUND(COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS market_positive_rate
        FROM reviews r
        LEFT JOIN classifications c ON r.id = c.review_id
        """
        
        market = self.conn.execute(market_sql).fetchdf().iloc[0]
        
        # Calculate rank
        rank_sql = f"""
        WITH business_ranks AS (
            SELECT 
                business,
                ROUND(AVG(rating), 2) AS avg_rating,
                RANK() OVER (ORDER BY AVG(rating) DESC) AS rating_rank
            FROM reviews
            GROUP BY business
            HAVING COUNT(*) >= 10
        )
        SELECT rating_rank, (SELECT COUNT(DISTINCT business) FROM reviews) AS total_businesses
        FROM business_ranks
        WHERE business = '{business}'
        """
        
        rank_result = self.conn.execute(rank_sql).fetchdf()
        
        return {
            "business": business,
            "avg_rating": float(biz["avg_rating"]),
            "positive_rate": float(biz["positive_rate"]),
            "market_avg_rating": float(market["market_avg_rating"]),
            "market_positive_rate": float(market["market_positive_rate"]),
            "rating_vs_market": float(biz["avg_rating"]) - float(market["market_avg_rating"]),
            "positive_vs_market": float(biz["positive_rate"]) - float(market["market_positive_rate"]),
            "rating_rank": int(rank_result.iloc[0]["rating_rank"]) if not rank_result.empty else None,
            "total_businesses": int(rank_result.iloc[0]["total_businesses"]) if not rank_result.empty else None
        }
    
    # =========================
    # Export Queries
    # =========================
    def export_business_report(
        self,
        business: str
    ) -> Dict[str, pd.DataFrame]:
        """Generate a complete report for a business"""
        return {
            "overview": self.get_business_stats(business),
            "by_city": self.get_city_stats(),  # Filtered later
            "trends": self.get_temporal_trends(business=business),
            "top_issues": self.get_top_issues(business=business),
            "competitive_position": pd.DataFrame([self.get_competitive_position(business)]),
            "recent_negative": self.get_recent_reviews(days=90, business=business, sentiment="Négatif")
        }

