"""
Database Manager for Review Analyzer
Handles DuckDB connections, CRUD operations, and data management
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager

import duckdb
import pandas as pd

from .models import (
    Place, Review, Classification, ProcessingRun,
    SCHEMA_SQL, CATEGORY_TO_COLUMN_MAP, CATEGORY_COLUMN_MAP,
    ProcessingStatus
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages DuckDB database for Review Analyzer
    
    Features:
    - Schema management with migrations
    - CRUD operations for places, reviews, classifications
    - Bulk import/export with Pandas
    - Processing run tracking
    - Checkpoint management for pipeline recovery
    """
    
    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        read_only: bool = False
    ):
        """
        Initialize database manager
        
        Args:
            db_path: Path to DuckDB file. If None, uses in-memory database.
                     Default location is data/review_analyzer.duckdb
            read_only: Open database in read-only mode
        """
        if db_path is None:
            # Default path relative to project
            from .. import config
            db_path = config.DATA_DIR / "review_analyzer.duckdb"
        
        self.db_path = Path(db_path) if db_path else None
        self.read_only = read_only
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        
        # Ensure directory exists
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DatabaseManager initialized: {self.db_path or 'in-memory'}")
    
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection"""
        if self._connection is None:
            if self.db_path:
                self._connection = duckdb.connect(
                    str(self.db_path),
                    read_only=self.read_only
                )
            else:
                self._connection = duckdb.connect(":memory:")
        return self._connection
    
    def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.debug("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @contextmanager
    def transaction(self):
        """Context manager for transactions"""
        try:
            yield self.connection
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise e
    
    # =========================
    # Schema Management
    # =========================
    def initialize_schema(self):
        """Create database schema if not exists"""
        logger.info("Initializing database schema...")
        
        # Split schema into individual statements
        statements = [s.strip() for s in SCHEMA_SQL.split(";") if s.strip()]
        
        for stmt in statements:
            try:
                self.connection.execute(stmt)
            except Exception as e:
                # Some statements might fail if already exists
                logger.debug(f"Schema statement skipped: {e}")
        
        self.connection.commit()
        logger.info("Database schema initialized")
    
    def get_table_stats(self) -> Dict[str, int]:
        """Get row counts for all tables"""
        tables = ["places", "reviews", "classifications", "processing_runs", "review_categories"]
        stats = {}
        
        for table in tables:
            try:
                result = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats[table] = result[0] if result else 0
            except Exception:
                stats[table] = 0
        
        return stats
    
    # =========================
    # Place Operations
    # =========================
    def upsert_place(self, place: Place) -> str:
        """Insert or update a place"""
        sql = """
        INSERT INTO places (
            place_id, business, business_type, name, address, city, region,
            zipcode, latitude, longitude, phone, website, google_rating,
            google_reviews_count, data_id, resolve_status, cluster_name,
            potential, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (place_id) DO UPDATE SET
            business = EXCLUDED.business,
            name = EXCLUDED.name,
            address = EXCLUDED.address,
            city = EXCLUDED.city,
            region = EXCLUDED.region,
            google_rating = EXCLUDED.google_rating,
            google_reviews_count = EXCLUDED.google_reviews_count,
            cluster_name = EXCLUDED.cluster_name,
            potential = EXCLUDED.potential,
            updated_at = CURRENT_TIMESTAMP
        """
        
        self.connection.execute(sql, [
            place.place_id, place.business, place.business_type, place.name,
            place.address, place.city, place.region, place.zipcode,
            place.latitude, place.longitude, place.phone, place.website,
            place.google_rating, place.google_reviews_count, place.data_id,
            place.resolve_status, place.cluster_name, place.potential,
            place.created_at, place.updated_at
        ])
        self.connection.commit()
        
        return place.place_id
    
    def upsert_places_bulk(self, df: pd.DataFrame) -> int:
        """Bulk upsert places from DataFrame"""
        if df.empty:
            return 0
        
        # Rename columns to match schema
        col_map = {
            "canonical_place_id": "place_id",
            "_place_id": "place_id",
            "_business": "business",
            "_city": "city",
            "title": "name",
            "lat": "latitude",
            "lng": "longitude",
            "reviews_count": "google_reviews_count",
        }
        
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        # Ensure required columns
        if "place_id" not in df.columns:
            logger.error("No place_id column found")
            return 0
        
        # Add timestamps
        now = datetime.now()
        df["created_at"] = now
        df["updated_at"] = now
        
        # Select only valid columns
        valid_cols = [
            "place_id", "business", "business_type", "name", "address", "city",
            "region", "zipcode", "latitude", "longitude", "phone", "website",
            "google_rating", "google_reviews_count", "data_id", "resolve_status",
            "cluster_name", "potential", "created_at", "updated_at"
        ]
        
        df_insert = df[[c for c in valid_cols if c in df.columns]].copy()
        
        # Use DuckDB's efficient bulk insert
        self.connection.execute("BEGIN TRANSACTION")
        try:
            # Register DataFrame
            self.connection.register("df_places", df_insert)
            
            # Get actual columns in the DataFrame
            insert_cols = list(df_insert.columns)
            cols_str = ", ".join(insert_cols)
            
            # Insert with conflict handling - ignore duplicates
            self.connection.execute(f"""
                INSERT OR IGNORE INTO places ({cols_str})
                SELECT {cols_str} FROM df_places
            """)
            
            self.connection.commit()
            logger.info(f"Upserted {len(df_insert)} places")
            return len(df_insert)
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Bulk place upsert failed: {e}")
            raise
    
    def get_place(self, place_id: str) -> Optional[Dict[str, Any]]:
        """Get a place by ID"""
        result = self.connection.execute(
            "SELECT * FROM places WHERE place_id = ?",
            [place_id]
        ).fetchdf()
        
        return result.to_dict("records")[0] if not result.empty else None
    
    def get_places_df(
        self,
        business: Optional[str] = None,
        city: Optional[str] = None,
        region: Optional[str] = None
    ) -> pd.DataFrame:
        """Get places as DataFrame with optional filters"""
        conditions = []
        params = []
        
        if business:
            conditions.append("business = ?")
            params.append(business)
        if city:
            conditions.append("city = ?")
            params.append(city)
        if region:
            conditions.append("region = ?")
            params.append(region)
        
        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM places WHERE {where}"
        
        return self.connection.execute(sql, params).fetchdf()
    
    # =========================
    # Review Operations
    # =========================
    def insert_review(self, review: Review) -> int:
        """Insert a new review, returns review ID"""
        sql = """
        INSERT INTO reviews (
            id, place_id, author, rating, text, review_date,
            review_date_parsed, language, source, business, city, agency_name, created_at
        ) VALUES (
            nextval('seq_reviews_id'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT DO NOTHING
        RETURNING id
        """
        
        result = self.connection.execute(sql, [
            review.place_id, review.author, review.rating, review.text,
            review.review_date, review.review_date_parsed, review.language,
            review.source, review.business, review.city, review.agency_name,
            review.created_at
        ]).fetchone()
        
        self.connection.commit()
        return result[0] if result else None
    
    def insert_reviews_bulk(self, df: pd.DataFrame) -> int:
        """Bulk insert reviews from DataFrame"""
        if df.empty:
            return 0
        
        # Column mapping
        col_map = {
            "_place_id": "place_id",
            "_business": "business",
            "_city": "city",
            "review_snippet": "text",
            "review_rating": "rating",
            "review_date": "review_date",
            "name": "author",
            "review_name": "author",
            "google_maps_name": "agency_name",
        }
        
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        # Parse dates if possible
        if "review_date" in df.columns and "review_date_parsed" not in df.columns:
            df["review_date_parsed"] = None  # Will be parsed later
        
        # Add defaults
        df["source"] = df.get("source", "google_maps")
        df["created_at"] = datetime.now()
        
        # Select valid columns
        valid_cols = [
            "place_id", "author", "rating", "text", "review_date",
            "review_date_parsed", "language", "source", "business",
            "city", "agency_name", "created_at"
        ]
        
        df_insert = df[[c for c in valid_cols if c in df.columns]].copy()
        
        # Drop duplicates
        df_insert = df_insert.drop_duplicates(subset=["place_id", "author", "text"])
        
        self.connection.execute("BEGIN TRANSACTION")
        try:
            # Generate IDs
            start_id = self.connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM reviews"
            ).fetchone()[0] + 1
            
            df_insert["id"] = range(start_id, start_id + len(df_insert))
            
            self.connection.register("df_reviews", df_insert)
            
            # Get actual columns
            insert_cols = list(df_insert.columns)
            cols_str = ", ".join(insert_cols)
            
            self.connection.execute(f"""
                INSERT INTO reviews ({cols_str})
                SELECT {cols_str} FROM df_reviews
                ON CONFLICT DO NOTHING
            """)
            
            self.connection.commit()
            logger.info(f"Inserted {len(df_insert)} reviews")
            return len(df_insert)
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Bulk review insert failed: {e}")
            raise
    
    def get_reviews_df(
        self,
        place_id: Optional[str] = None,
        business: Optional[str] = None,
        city: Optional[str] = None,
        include_classifications: bool = True,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get reviews as DataFrame with optional filters"""
        if include_classifications:
            base = "SELECT * FROM v_reviews_full"
        else:
            base = "SELECT * FROM reviews"
        
        conditions = []
        params = []
        
        if place_id:
            conditions.append("place_id = ?")
            params.append(place_id)
        if business:
            conditions.append("business = ?")
            params.append(business)
        if city:
            conditions.append("city = ?")
            params.append(city)
        
        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"{base} WHERE {where}"
        
        if limit:
            sql += f" LIMIT {limit}"
        
        return self.connection.execute(sql, params).fetchdf()
    
    def get_unclassified_reviews(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get reviews that haven't been classified yet"""
        sql = """
        SELECT r.* 
        FROM reviews r
        LEFT JOIN classifications c ON r.id = c.review_id
        WHERE c.id IS NULL
        """
        
        if limit:
            sql += f" LIMIT {limit}"
        
        return self.connection.execute(sql).fetchdf()
    
    # =========================
    # Classification Operations
    # =========================
    def upsert_classification(self, classification: Classification) -> int:
        """Insert or update a classification"""
        sql = """
        INSERT INTO classifications (
            id, review_id, sentiment, categories_json, rationale,
            model_version, processing_run_id, confidence_threshold, created_at
        ) VALUES (
            nextval('seq_classifications_id'), ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT (review_id) DO UPDATE SET
            sentiment = EXCLUDED.sentiment,
            categories_json = EXCLUDED.categories_json,
            rationale = EXCLUDED.rationale,
            model_version = EXCLUDED.model_version,
            processing_run_id = EXCLUDED.processing_run_id,
            created_at = CURRENT_TIMESTAMP
        RETURNING id
        """
        
        result = self.connection.execute(sql, [
            classification.review_id, classification.sentiment,
            classification.categories_json, classification.rationale,
            classification.model_version, classification.processing_run_id,
            classification.confidence_threshold, classification.created_at
        ]).fetchone()
        
        self.connection.commit()
        return result[0] if result else None
    
    def upsert_classifications_bulk(
        self,
        df: pd.DataFrame,
        model_version: str = "gpt-4.1-mini",
        run_id: Optional[int] = None
    ) -> int:
        """Bulk upsert classifications from DataFrame"""
        if df.empty:
            return 0
        
        # Map review_id if needed
        if "review_id" not in df.columns and "id" in df.columns:
            df["review_id"] = df["id"]
        
        # Ensure required columns
        required = ["review_id", "sentiment"]
        if not all(c in df.columns for c in required):
            logger.error(f"Missing required columns: {required}")
            return 0
        
        df["model_version"] = model_version
        df["processing_run_id"] = run_id
        df["confidence_threshold"] = 0.55
        df["created_at"] = datetime.now()
        
        # Handle categories_json
        if "categories_json" not in df.columns:
            df["categories_json"] = "[]"
        
        valid_cols = [
            "review_id", "sentiment", "categories_json", "rationale",
            "model_version", "processing_run_id", "confidence_threshold", "created_at"
        ]
        
        df_insert = df[[c for c in valid_cols if c in df.columns]].copy()
        
        self.connection.execute("BEGIN TRANSACTION")
        try:
            # Generate IDs
            start_id = self.connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM classifications"
            ).fetchone()[0] + 1
            
            df_insert["id"] = range(start_id, start_id + len(df_insert))
            
            self.connection.register("df_class", df_insert)
            
            # Delete existing classifications for these reviews
            review_ids = df_insert["review_id"].tolist()
            if review_ids:
                placeholders = ",".join(["?" for _ in review_ids])
                self.connection.execute(
                    f"DELETE FROM classifications WHERE review_id IN ({placeholders})",
                    review_ids
                )
            
            # Get actual columns
            insert_cols = list(df_insert.columns)
            cols_str = ", ".join(insert_cols)
            
            self.connection.execute(f"""
                INSERT INTO classifications ({cols_str})
                SELECT {cols_str} FROM df_class
            """)
            
            self.connection.commit()
            logger.info(f"Upserted {len(df_insert)} classifications")
            return len(df_insert)
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Bulk classification upsert failed: {e}")
            raise
    
    def update_category_flags(self, review_id: int, categories_json: str):
        """Update review_categories table from categories JSON"""
        try:
            categories = json.loads(categories_json) if categories_json else []
        except json.JSONDecodeError:
            categories = []
        
        # Build flag dict
        flags = {col: 0 for col in CATEGORY_COLUMN_MAP.keys()}
        
        for cat in categories:
            label = cat.get("label", "")
            col_name = CATEGORY_TO_COLUMN_MAP.get(label)
            if col_name:
                flags[col_name] = 1
        
        # Upsert into review_categories
        cols = ", ".join(flags.keys())
        placeholders = ", ".join(["?"] * len(flags))
        values = list(flags.values())
        
        sql = f"""
        INSERT INTO review_categories (review_id, {cols})
        VALUES (?, {placeholders})
        ON CONFLICT (review_id) DO UPDATE SET
            {', '.join(f'{c} = EXCLUDED.{c}' for c in flags.keys())}
        """
        
        self.connection.execute(sql, [review_id] + values)
        self.connection.commit()
    
    # =========================
    # Processing Run Operations
    # =========================
    def start_processing_run(
        self,
        run_type: str,
        config_dict: Optional[Dict] = None,
        total_items: int = 0
    ) -> int:
        """Start a new processing run, returns run ID"""
        sql = """
        INSERT INTO processing_runs (
            id, run_type, status, started_at, config_json, total_items
        ) VALUES (
            nextval('seq_runs_id'), ?, ?, ?, ?, ?
        )
        RETURNING id
        """
        
        result = self.connection.execute(sql, [
            run_type,
            ProcessingStatus.IN_PROGRESS.value,
            datetime.now(),
            json.dumps(config_dict or {}),
            total_items
        ]).fetchone()
        
        self.connection.commit()
        run_id = result[0] if result else None
        logger.info(f"Started processing run {run_id}: {run_type}")
        return run_id
    
    def update_processing_run(
        self,
        run_id: int,
        status: Optional[str] = None,
        processed_items: Optional[int] = None,
        failed_items: Optional[int] = None,
        stats_dict: Optional[Dict] = None,
        error_message: Optional[str] = None
    ):
        """Update a processing run"""
        updates = []
        params = []
        
        if status:
            updates.append("status = ?")
            params.append(status)
            if status in [ProcessingStatus.COMPLETED.value, ProcessingStatus.FAILED.value]:
                updates.append("completed_at = ?")
                params.append(datetime.now())
        
        if processed_items is not None:
            updates.append("processed_items = ?")
            params.append(processed_items)
        
        if failed_items is not None:
            updates.append("failed_items = ?")
            params.append(failed_items)
        
        if stats_dict:
            updates.append("stats_json = ?")
            params.append(json.dumps(stats_dict))
        
        if error_message:
            updates.append("error_message = ?")
            params.append(error_message)
        
        if not updates:
            return
        
        sql = f"UPDATE processing_runs SET {', '.join(updates)} WHERE id = ?"
        params.append(run_id)
        
        self.connection.execute(sql, params)
        self.connection.commit()
    
    def get_last_run(self, run_type: str) -> Optional[Dict[str, Any]]:
        """Get the most recent processing run of a type"""
        result = self.connection.execute("""
            SELECT * FROM processing_runs 
            WHERE run_type = ?
            ORDER BY started_at DESC
            LIMIT 1
        """, [run_type]).fetchdf()
        
        return result.to_dict("records")[0] if not result.empty else None
    
    # =========================
    # Export Operations
    # =========================
    def export_to_csv(
        self,
        output_path: Path,
        include_categories_wide: bool = True
    ):
        """Export full dataset to CSV in wide format"""
        sql = "SELECT * FROM v_reviews_full"
        df = self.connection.execute(sql).fetchdf()
        
        if include_categories_wide and not df.empty:
            # Expand categories to wide format
            for _, row in df.iterrows():
                try:
                    cats = json.loads(row.get("categories_json", "[]") or "[]")
                    for cat in cats:
                        label = cat.get("label", "")
                        if label in CATEGORY_TO_COLUMN_MAP:
                            df.loc[df["review_id"] == row["review_id"], label] = 1
                except Exception:
                    pass
            
            # Fill NaN with 0 for category columns
            for cat in CATEGORY_TO_COLUMN_MAP.keys():
                if cat in df.columns:
                    df[cat] = df[cat].fillna(0).astype(int)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} rows to {output_path}")
        return len(df)
    
    def export_to_parquet(self, output_path: Path):
        """Export full dataset to Parquet"""
        sql = "SELECT * FROM v_reviews_full"
        df = self.connection.execute(sql).fetchdf()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Exported {len(df)} rows to {output_path}")
        return len(df)
    
    # =========================
    # Import Operations
    # =========================
    def import_from_csv(
        self,
        csv_path: Path,
        data_type: str = "classified_reviews"
    ) -> Dict[str, int]:
        """
        Import data from CSV file
        
        Args:
            csv_path: Path to CSV file
            data_type: Type of data (places, reviews, classified_reviews)
            
        Returns:
            Dict with import counts
        """
        logger.info(f"Importing {data_type} from {csv_path}")
        
        df = pd.read_csv(csv_path)
        stats = {"rows_read": len(df)}
        
        if data_type == "places":
            stats["places_imported"] = self.upsert_places_bulk(df)
            
        elif data_type == "reviews":
            # Need place_ids first
            place_ids = df["place_id"].unique() if "place_id" in df.columns else []
            stats["reviews_imported"] = self.insert_reviews_bulk(df)
            
        elif data_type == "classified_reviews":
            # Import full classified dataset
            stats.update(self._import_classified_reviews(df))
        
        logger.info(f"Import complete: {stats}")
        return stats
    
    def _import_classified_reviews(self, df: pd.DataFrame) -> Dict[str, int]:
        """Import classified reviews with all related data"""
        stats = {"places": 0, "reviews": 0, "classifications": 0}
        
        # 1. Extract and insert unique places
        place_cols = [
            "place_id", "bank", "google_maps_name", "city", "region",
            "latitude", "longitude", "zipcode", "cluster_name_grouped", "potential"
        ]
        
        # Map column names
        place_df = df[[c for c in place_cols if c in df.columns]].copy()
        place_df = place_df.rename(columns={
            "bank": "business",
            "google_maps_name": "name",
            "cluster_name_grouped": "cluster_name",
        })
        place_df = place_df.drop_duplicates(subset=["place_id"])
        
        if not place_df.empty:
            stats["places"] = self.upsert_places_bulk(place_df)
        
        # 2. Insert reviews
        review_df = df.copy()
        review_df = review_df.rename(columns={
            "review_text": "text",
            "review_name": "author",
            "review_date": "review_date",
            "bank": "business",
            "google_maps_name": "agency_name",
        })
        
        # Add review IDs
        if "id" not in review_df.columns:
            review_df["id"] = range(1, len(review_df) + 1)
        
        stats["reviews"] = self.insert_reviews_bulk(review_df)
        
        # 3. Insert classifications
        class_df = df[["sentiment", "rationale", "language"]].copy()
        class_df["review_id"] = review_df["id"]
        
        # Handle categories
        if "categories_with_confidence" in df.columns:
            class_df["categories_json"] = df["categories_with_confidence"]
        elif "categories_json" in df.columns:
            class_df["categories_json"] = df["categories_json"]
        else:
            class_df["categories_json"] = "[]"
        
        stats["classifications"] = self.upsert_classifications_bulk(class_df)
        
        return stats

