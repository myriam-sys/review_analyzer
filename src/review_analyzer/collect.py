"""
Step 2: Review Collection
Collect reviews for discovered places with multiple output formats
"""
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import pandas as pd

from . import config, utils

logger = logging.getLogger(__name__)


class ReviewCollector:
    """
    Collects reviews for places with multiple output modes
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize review collector
        
        Args:
            api_key: SerpAPI key (uses config if None)
            debug: Enable debug logging
        """
        self.client = utils.SerpAPIClient(api_key=api_key, debug=debug)
        self.debug = debug
    
    def collect_reviews(
        self,
        input_file: Path,
        output_mode: str,
        output_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        checkpoint_file: Optional[Path] = None,
        filter_cities: Optional[List[str]] = None,
        filter_businesses: Optional[List[str]] = None,
        place_id_col: Optional[str] = None,
        city_col: Optional[str] = None,
        business_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Collect reviews from locations CSV
        
        Args:
            input_file: Path to locations CSV with place_ids
            output_mode: Output format
                (json, json-per-city, json-per-business, csv)
            output_path: Output file path (for json/csv modes)
            output_dir: Output directory (for json-per-* modes)
            checkpoint_file: Optional checkpoint file for resume capability
            filter_cities: Optional city filter
            filter_businesses: Optional business filter
            place_id_col: Place ID column name (auto-detects canonical/place_id)
            city_col: City column name
            business_col: Business column name
            
        Returns:
            Dictionary with collection statistics
        """
        logger.info(f"Starting review collection from: {input_file}")
        
        # Validate output mode
        valid_modes = [
            config.OutputMode.JSON_PER_CITY,
            config.OutputMode.JSON_PER_BUSINESS,
            config.OutputMode.SINGLE_JSON,
            config.OutputMode.CSV,
        ]
        if output_mode not in valid_modes:
            raise ValueError(
                f"Invalid output mode: {output_mode}. "
                f"Must be one of: {valid_modes}"
            )
        
        # Read locations
        df = utils.read_agencies_csv(
            str(input_file),
            place_id_col=place_id_col,
            city_col=city_col,
            business_col=business_col,
            filter_cities=filter_cities,
            filter_businesses=filter_businesses
        )
        
        if df.empty:
            logger.warning("No locations found after filtering")
            return {"status": "no_data", "locations_processed": 0}
        
        logger.info(f"Collecting reviews for {len(df)} locations")
        
        # Collect reviews
        reviews_data = self._collect_all_reviews(df, checkpoint_file)
        
        # Save in requested format
        if output_mode == config.OutputMode.SINGLE_JSON:
            self._save_single_json(reviews_data, output_path)
        
        elif output_mode == config.OutputMode.JSON_PER_CITY:
            self._save_json_per_city(reviews_data, df, output_dir)
        
        elif output_mode == config.OutputMode.JSON_PER_BUSINESS:
            self._save_json_per_business(reviews_data, df, output_dir)
        
        elif output_mode == config.OutputMode.CSV:
            self._save_as_csv(reviews_data, df, output_path)
        
        # Statistics
        total_reviews = sum(len(r) for r in reviews_data.values())
        stats = {
            "status": "success",
            "total_places": len(reviews_data),
            "agencies_processed": len(reviews_data),  # backward compat
            "total_reviews": total_reviews,
            "output_mode": output_mode
        }
        
        logger.info(
            f"Collection complete: {stats['total_places']} places, "
            f"{stats['total_reviews']} reviews"
        )
        
        return stats
    
    def _collect_all_reviews(
        self,
        df: pd.DataFrame,
        checkpoint_file: Optional[Path] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect reviews for all place_ids
        
        Args:
            df: DataFrame with _place_id column
            checkpoint_file: Optional checkpoint file path
            
        Returns:
            Dictionary mapping place_id -> reviews list
        """
        reviews_data = {}
        failed_ids = []
        
        # Setup checkpoint
        if checkpoint_file is None:
            checkpoint_file = config.LOGS_DIR / "collection_checkpoint.json"
        checkpoint_mgr = utils.CheckpointManager(checkpoint_file)
        checkpoint = checkpoint_mgr.load()
        reviews_data = checkpoint.get("reviews_data", {})
        
        progress = utils.create_progress_bar(
            len(df),
            desc="Collecting reviews"
        )
        
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            place_id = row["_place_id"]
            
            # Skip if already collected
            if place_id in reviews_data:
                if progress:
                    progress.update(1)
                continue
            
            try:
                reviews = self.client.get_reviews(place_id)
                reviews_data[place_id] = reviews
                
                if self.debug:
                    logger.debug(
                        f"Collected {len(reviews)} reviews for {place_id}"
                    )
                
                # Checkpoint every N agencies
                if idx % config.PROCESSING_CONFIG["checkpoint_interval"] == 0:
                    checkpoint_mgr.save({"reviews_data": reviews_data})
                
                time.sleep(config.SERPAPI_CONFIG["delay_seconds"])
                
            except Exception as e:
                logger.error(f"Failed to collect reviews for {place_id}: {e}")
                failed_ids.append(place_id)
                reviews_data[place_id] = []
            
            if progress:
                progress.update(1)
        
        if progress:
            progress.close()
        
        # Save failed IDs
        if failed_ids:
            failed_path = config.OUTPUT_DIR / "failed_place_ids.csv"
            pd.DataFrame({"place_id": failed_ids}).to_csv(
                failed_path,
                index=False
            )
            logger.warning(f"Failed IDs saved to: {failed_path}")
        
        # Clear checkpoint
        checkpoint_mgr.clear()
        
        return reviews_data
    
    def _save_single_json(
        self,
        reviews_data: Dict[str, List[Dict]],
        output_path: Path
    ):
        """Save all reviews to single JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        utils.write_json(output_path, reviews_data)
        logger.info(f"Saved reviews to: {output_path}")
    
    def _save_json_per_city(
        self,
        reviews_data: Dict[str, List[Dict]],
        df: pd.DataFrame,
        output_dir: Path
    ):
        """Save reviews grouped by city"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by city
        city_groups = defaultdict(dict)
        for place_id, reviews in reviews_data.items():
            # Find city for this place_id
            row = df[df["_place_id"] == place_id]
            if row.empty:
                continue
            city = row.iloc[0]["_city"]
            city_groups[city][place_id] = reviews
        
        # Save each city
        for city, city_data in city_groups.items():
            safe_city = city.lower().replace(" ", "-")
            output_path = output_dir / f"reviews_{safe_city}.json"
            utils.write_json(output_path, city_data)
            logger.info(f"Saved {city}: {output_path}")
    
    def _save_json_per_business(
        self,
        reviews_data: Dict[str, List[Dict]],
        df: pd.DataFrame,
        output_dir: Path
    ):
        """Save reviews grouped by business"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by business
        business_groups = defaultdict(dict)
        for place_id, reviews in reviews_data.items():
            row = df[df["_place_id"] == place_id]
            if row.empty:
                continue
            business = row.iloc[0]["_business"]
            business_groups[business][place_id] = reviews
        
        # Save each business
        for business, business_data in business_groups.items():
            safe_business = business.lower().replace(" ", "-")
            output_path = output_dir / f"reviews_business_{safe_business}.json"
            utils.write_json(output_path, business_data)
            logger.info(f"Saved {business}: {output_path}")
    
    def _save_as_csv(
        self,
        reviews_data: Dict[str, List[Dict]],
        df: pd.DataFrame,
        output_path: Path
    ):
        """Save reviews as flattened CSV"""
        rows = []
        
        for place_id, reviews in reviews_data.items():
            # Get metadata
            row_data = df[df["_place_id"] == place_id]
            if row_data.empty:
                continue
            
            # Support both old and new column names
            city = row_data.iloc[0].get("_city", "")
            business = (
                row_data.iloc[0].get("_business") or 
                row_data.iloc[0].get("_bank", "")
            )
            title = row_data.iloc[0].get("title", "")
            
            # Preserve geographic and metadata from discovery
            lat = row_data.iloc[0].get("lat") or row_data.iloc[0].get("latitude")
            lng = row_data.iloc[0].get("lng") or row_data.iloc[0].get("longitude")
            address = row_data.iloc[0].get("address", "")
            place_rating = row_data.iloc[0].get("place_rating") or row_data.iloc[0].get("reviews_rating")
            reviews_count = row_data.iloc[0].get("reviews_count")
            
            for review in reviews:
                rows.append({
                    "_place_id": place_id,
                    "_city": city,
                    "_business": business,
                    "title": title,
                    "address": address,
                    "lat": lat,
                    "lng": lng,
                    "place_rating": place_rating,
                    "reviews_count": reviews_count,
                    "date": review.get("date"),
                    "rating": review.get("rating"),
                    "text": review.get("snippet"),
                    "author": review.get("name"),
                })
        
        output_df = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        logger.info(f"Saved CSV: {output_path}")


# =========================
# CLI Interface
# =========================
def main():
    """CLI entry point for review collection"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect Google Maps reviews"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input CSV with place_ids"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "json",
            "json-per-city",
            "json-per-bank",
            "csv"
        ],
        help="Output mode"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (for json/csv modes)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (for json-per-* modes)"
    )
    parser.add_argument(
        "--filter-cities",
        help="Comma-separated cities to include"
    )
    parser.add_argument(
        "--filter-banks",
        help="Comma-separated banks to include"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Parse filters
    filter_cities = None
    if args.filter_cities:
        filter_cities = [c.strip() for c in args.filter_cities.split(",")]
    
    filter_banks = None
    if args.filter_banks:
        filter_banks = [b.strip() for b in args.filter_banks.split(",")]
    
    # Run collection
    collector = ReviewCollector(debug=args.debug)
    stats = collector.collect_reviews(
        input_file=args.input,
        output_mode=args.mode,
        output_path=args.output,
        output_dir=args.output_dir,
        filter_cities=filter_cities,
        filter_banks=filter_banks
    )
    
    print(f"\n✅ Collection complete!")
    print(f"   Agencies: {stats['agencies_processed']}")
    print(f"   Reviews: {stats['total_reviews']}")


if __name__ == "__main__":
    main()
