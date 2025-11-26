"""
Migration script for importing existing CSV data into DuckDB
Handles the initial data migration from file-based storage to database
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from .manager import DatabaseManager
from .models import CATEGORY_TO_COLUMN_MAP, ProcessingStatus

logger = logging.getLogger(__name__)


class DataMigration:
    """
    Handles migration of existing CSV/Parquet data to DuckDB database
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize migration handler
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
    
    def migrate_classified_reviews(
        self,
        csv_path: Path,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Migrate the main classified reviews CSV to database
        
        This is the primary migration function for the existing dataset
        
        Args:
            csv_path: Path to bank_reviews_classified.csv
            batch_size: Number of rows to process at once
            
        Returns:
            Migration statistics
        """
        logger.info(f"Starting migration from: {csv_path}")
        
        # Start a processing run
        run_id = self.db.start_processing_run(
            run_type="migration",
            config_dict={"source": str(csv_path), "batch_size": batch_size}
        )
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            total_rows = len(df)
            
            logger.info(f"Read {total_rows} rows from CSV")
            
            # Update total items (we already started the run with 0)
            self.db.connection.execute(
                "UPDATE processing_runs SET total_items = ? WHERE id = ?",
                [total_rows, run_id]
            )
            self.db.connection.commit()
            
            stats = {
                "total_rows": total_rows,
                "places_migrated": 0,
                "reviews_migrated": 0,
                "classifications_migrated": 0,
                "categories_migrated": 0,
                "errors": []
            }
            
            # 1. Extract and migrate unique places
            logger.info("Migrating places...")
            places_migrated = self._migrate_places(df)
            stats["places_migrated"] = places_migrated
            
            # 2. Migrate reviews with generated IDs
            logger.info("Migrating reviews...")
            review_id_map = self._migrate_reviews(df, batch_size)
            stats["reviews_migrated"] = len(review_id_map)
            
            # 3. Migrate classifications
            logger.info("Migrating classifications...")
            classifications_migrated = self._migrate_classifications(df, review_id_map, run_id)
            stats["classifications_migrated"] = classifications_migrated
            
            # 4. Migrate category flags to wide table
            logger.info("Migrating category flags...")
            categories_migrated = self._migrate_category_flags(df, review_id_map)
            stats["categories_migrated"] = categories_migrated
            
            # Update run status
            self.db.update_processing_run(
                run_id,
                status=ProcessingStatus.COMPLETED.value,
                processed_items=total_rows,
                stats_dict=stats
            )
            
            logger.info(f"Migration complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self.db.update_processing_run(
                run_id,
                status=ProcessingStatus.FAILED.value,
                error_message=str(e)
            )
            raise
    
    def _migrate_places(self, df: pd.DataFrame) -> int:
        """Extract and migrate unique places"""
        # Column mapping for places
        place_cols = {
            "place_id": "place_id",
            "bank": "business",
            "google_maps_name": "name",
            "city": "city",
            "region": "region",
            "latitude": "latitude",
            "longitude": "longitude",
            "zipcode": "zipcode",
            "cluster_name_grouped": "cluster_name",
            "potential": "potential"
        }
        
        # Filter columns that exist
        existing_cols = {k: v for k, v in place_cols.items() if k in df.columns}
        
        # Extract and deduplicate
        places_df = df[list(existing_cols.keys())].copy()
        places_df = places_df.rename(columns=existing_cols)
        places_df = places_df.drop_duplicates(subset=["place_id"])
        
        # Remove rows without place_id
        places_df = places_df[places_df["place_id"].notna() & (places_df["place_id"] != "")]
        
        if places_df.empty:
            return 0
        
        # Add business_type if not present
        if "business_type" not in places_df.columns:
            places_df["business_type"] = "bank"
        
        # Add timestamps
        now = datetime.now()
        places_df["created_at"] = now
        places_df["updated_at"] = now
        
        # Bulk insert
        return self.db.upsert_places_bulk(places_df)
    
    def _migrate_reviews(
        self,
        df: pd.DataFrame,
        batch_size: int = 1000
    ) -> Dict[int, int]:
        """
        Migrate reviews and return mapping of original index to new review ID
        
        Returns:
            Dict mapping original DataFrame index to database review ID
        """
        # Column mapping
        review_cols = {
            "place_id": "place_id",
            "review_text": "text",
            "review_name": "author",
            "rating": "rating",
            "review_date": "review_date",
            "bank": "business",
            "city": "city",
            "google_maps_name": "agency_name",
            "language": "language",
            "created_at": "created_at",
            "month": "review_date_parsed"
        }
        
        existing_cols = {k: v for k, v in review_cols.items() if k in df.columns}
        
        reviews_df = df[list(existing_cols.keys())].copy()
        reviews_df = reviews_df.rename(columns=existing_cols)
        
        # Handle missing data
        reviews_df["text"] = reviews_df["text"].fillna("")
        reviews_df["author"] = reviews_df.get("author", "").fillna("Anonymous")
        reviews_df["rating"] = reviews_df["rating"].fillna(0).astype(int)
        
        # Generate sequential IDs
        start_id = self.db.connection.execute(
            "SELECT COALESCE(MAX(id), 0) FROM reviews"
        ).fetchone()[0] + 1
        
        reviews_df["id"] = range(start_id, start_id + len(reviews_df))
        
        # Store original index mapping
        review_id_map = dict(zip(df.index, reviews_df["id"]))
        
        # Add source and timestamp
        reviews_df["source"] = "google_maps"
        if "created_at" not in reviews_df.columns:
            reviews_df["created_at"] = datetime.now()
        
        # Select final columns
        final_cols = [
            "id", "place_id", "author", "rating", "text", "review_date",
            "review_date_parsed", "language", "source", "business", "city",
            "agency_name", "created_at"
        ]
        
        reviews_df = reviews_df[[c for c in final_cols if c in reviews_df.columns]]
        
        # Bulk insert in batches with explicit column names
        insert_cols = list(reviews_df.columns)
        cols_str = ", ".join(insert_cols)
        
        total_inserted = 0
        for i in range(0, len(reviews_df), batch_size):
            batch = reviews_df.iloc[i:i + batch_size]
            
            self.db.connection.register("df_batch", batch)
            self.db.connection.execute(f"""
                INSERT OR IGNORE INTO reviews ({cols_str})
                SELECT {cols_str} FROM df_batch
            """)
            self.db.connection.commit()
            
            total_inserted += len(batch)
            logger.debug(f"Migrated reviews: {total_inserted}/{len(reviews_df)}")
        
        return review_id_map
    
    def _migrate_classifications(
        self,
        df: pd.DataFrame,
        review_id_map: Dict[int, int],
        run_id: int
    ) -> int:
        """Migrate classification data"""
        # Get the actual review IDs that exist in the database
        existing_review_ids = set(
            self.db.connection.execute("SELECT id FROM reviews").fetchdf()["id"].tolist()
        )
        
        # Filter the mapping to only include reviews that were actually inserted
        valid_review_map = {k: v for k, v in review_id_map.items() if v in existing_review_ids}
        
        logger.debug(f"Review ID map: {len(review_id_map)} total, {len(valid_review_map)} valid")
        
        # Column mapping
        class_cols = {
            "sentiment": "sentiment",
            "rationale": "rationale",
            "language": "language",
            "categories_with_confidence": "categories_json"
        }
        
        existing_cols = {k: v for k, v in class_cols.items() if k in df.columns}
        
        class_df = df[list(existing_cols.keys())].copy()
        class_df = class_df.rename(columns=existing_cols)
        
        # Map review IDs - only for valid reviews
        class_df["review_id"] = df.index.map(lambda x: valid_review_map.get(x))
        class_df = class_df[class_df["review_id"].notna()]
        class_df["review_id"] = class_df["review_id"].astype(int)
        
        # Handle categories JSON
        if "categories_json" not in class_df.columns:
            class_df["categories_json"] = "[]"
        
        # Add metadata
        class_df["model_version"] = "gpt-4.1-mini"
        class_df["processing_run_id"] = run_id
        class_df["confidence_threshold"] = 0.55
        class_df["created_at"] = datetime.now()
        
        # Generate IDs
        start_id = self.db.connection.execute(
            "SELECT COALESCE(MAX(id), 0) FROM classifications"
        ).fetchone()[0] + 1
        
        class_df["id"] = range(start_id, start_id + len(class_df))
        
        # Select final columns - must match table schema order
        final_cols = [
            "id", "review_id", "sentiment", "categories_json", "rationale",
            "model_version", "processing_run_id", "confidence_threshold", "created_at"
        ]
        
        class_df = class_df[[c for c in final_cols if c in class_df.columns]]
        
        # Bulk insert with explicit columns
        self.db.connection.register("df_class", class_df)
        insert_cols = list(class_df.columns)
        cols_str = ", ".join(insert_cols)
        
        self.db.connection.execute(f"""
            INSERT INTO classifications ({cols_str})
            SELECT {cols_str} FROM df_class
        """)
        self.db.connection.commit()
        
        return len(class_df)
    
    def _migrate_category_flags(
        self,
        df: pd.DataFrame,
        review_id_map: Dict[int, int]
    ) -> int:
        """Migrate category binary flags to review_categories table"""
        # Get the actual review IDs that exist in the database
        existing_review_ids = set(
            self.db.connection.execute("SELECT id FROM reviews").fetchdf()["id"].tolist()
        )
        
        # Filter the mapping to only include reviews that were actually inserted
        valid_review_map = {k: v for k, v in review_id_map.items() if v in existing_review_ids}
        
        # Find category columns in the source DataFrame
        category_cols = []
        for full_name, short_col in CATEGORY_TO_COLUMN_MAP.items():
            if full_name in df.columns:
                category_cols.append((full_name, short_col))
        
        if not category_cols:
            logger.warning("No category columns found in source data")
            return 0
        
        # Build category flags DataFrame
        flags_data = []
        
        for idx, row in df.iterrows():
            review_id = valid_review_map.get(idx)
            if review_id is None:
                continue
            
            flags = {"review_id": review_id}
            for full_name, short_col in category_cols:
                val = row.get(full_name, 0)
                flags[short_col] = int(val) if pd.notna(val) else 0
            
            flags_data.append(flags)
        
        if not flags_data:
            return 0
        
        flags_df = pd.DataFrame(flags_data)
        
        # Ensure all category columns exist
        for _, short_col in CATEGORY_TO_COLUMN_MAP.items():
            if short_col not in flags_df.columns:
                flags_df[short_col] = 0
        
        # Bulk insert with explicit columns
        self.db.connection.register("df_flags", flags_df)
        insert_cols = list(flags_df.columns)
        cols_str = ", ".join(insert_cols)
        
        self.db.connection.execute(f"""
            INSERT OR IGNORE INTO review_categories ({cols_str})
            SELECT {cols_str} FROM df_flags
        """)
        self.db.connection.commit()
        
        return len(flags_df)
    
    def verify_migration(self) -> Dict[str, Any]:
        """Verify migration was successful"""
        stats = self.db.get_table_stats()
        
        # Check data integrity
        integrity_checks = {}
        
        # Check reviews have valid place_ids
        orphan_reviews = self.db.connection.execute("""
            SELECT COUNT(*) FROM reviews r
            LEFT JOIN places p ON r.place_id = p.place_id
            WHERE p.place_id IS NULL
        """).fetchone()[0]
        integrity_checks["orphan_reviews"] = orphan_reviews
        
        # Check classifications have valid review_ids
        orphan_classifications = self.db.connection.execute("""
            SELECT COUNT(*) FROM classifications c
            LEFT JOIN reviews r ON c.review_id = r.id
            WHERE r.id IS NULL
        """).fetchone()[0]
        integrity_checks["orphan_classifications"] = orphan_classifications
        
        # Check sentiment distribution
        sentiment_dist = self.db.connection.execute("""
            SELECT sentiment, COUNT(*) AS count
            FROM classifications
            GROUP BY sentiment
        """).fetchdf()
        
        return {
            "table_stats": stats,
            "integrity_checks": integrity_checks,
            "sentiment_distribution": sentiment_dist.to_dict("records") if not sentiment_dist.empty else []
        }


def run_migration(
    csv_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    verify: bool = True
) -> Dict[str, Any]:
    """
    Run the full migration process
    
    Args:
        csv_path: Path to classified reviews CSV (uses default if None)
        db_path: Path to DuckDB file (uses default if None)
        verify: Whether to run verification after migration
        
    Returns:
        Migration results and verification stats
    """
    from .. import config
    
    # Default paths
    if csv_path is None:
        csv_path = config.PROC_CLASSIFIED_LATEST / "bank_reviews_classified.csv"
    
    if db_path is None:
        db_path = config.DATA_DIR / "review_analyzer.duckdb"
    
    logger.info(f"Migration: {csv_path} -> {db_path}")
    
    # Initialize database
    with DatabaseManager(db_path) as db:
        # Create schema
        db.initialize_schema()
        
        # Run migration
        migration = DataMigration(db)
        stats = migration.migrate_classified_reviews(csv_path)
        
        # Verify
        if verify:
            verification = migration.verify_migration()
            stats["verification"] = verification
    
    return stats


# =========================
# CLI Interface
# =========================
def main():
    """CLI entry point for migration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate CSV data to DuckDB")
    parser.add_argument(
        "--input",
        type=Path,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Output DuckDB file path"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run migration
    results = run_migration(
        csv_path=args.input,
        db_path=args.db,
        verify=not args.no_verify
    )
    
    print("\n✅ Migration Complete!")
    print(f"   Places: {results['places_migrated']}")
    print(f"   Reviews: {results['reviews_migrated']}")
    print(f"   Classifications: {results['classifications_migrated']}")
    
    if "verification" in results:
        v = results["verification"]
        print(f"\n📊 Verification:")
        print(f"   Table Stats: {v['table_stats']}")
        print(f"   Integrity: {v['integrity_checks']}")


if __name__ == "__main__":
    main()

