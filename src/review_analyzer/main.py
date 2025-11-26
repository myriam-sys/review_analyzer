"""
Pipeline Orchestrator - Main CLI
Coordinates discover → collect → classify workflow
"""
import sys
import logging
from pathlib import Path
from typing import Optional, List
import argparse

from . import config
from . import geo
from .discover import DiscoveryEngine
from .collect import ReviewCollector
from .classify import ReviewClassifier

logger = logging.getLogger(__name__)


# =========================
# Pipeline Runner
# =========================
class Pipeline:
    """Orchestrates the full review analysis pipeline"""
    
    def __init__(self, debug: bool = False):
        """
        Initialize pipeline
        
        Args:
            debug: Enable debug logging
        """
        self.debug = debug
        self.discovery = DiscoveryEngine(debug=debug)
        self.collector = ReviewCollector(debug=debug)
        self.classifier = ReviewClassifier(debug=debug)
    
    def run_full_pipeline(
        self,
        businesses: List[str],
        cities: List[str],
        business_type: Optional[str] = None,
        output_mode: str = "csv",
        skip_discovery: bool = False,
        discovery_input: Optional[Path] = None,
        skip_collection: bool = False,
        collection_input: Optional[Path] = None,
        skip_classification: bool = False,
        wide_format: bool = True
    ) -> dict:
        """
        Run complete pipeline: discover → collect → transform → classify
        
        Args:
            businesses: Business names to process
            cities: City names to process
            business_type: Type of business (e.g., 'bank', 'hotel')
            output_mode: Collection output format
            skip_discovery: Skip place discovery
            discovery_input: Use existing place IDs file
            skip_collection: Skip review collection
            collection_input: Use existing reviews file
            skip_classification: Skip classification
            wide_format: Convert categories to wide format
            
        Returns:
            Statistics dictionary with discovery, collection, transform, 
            and classification stats
        """
        stats = {}
        
        # ====================
        # Step 1: Discovery
        # ====================
        if not skip_discovery:
            logger.info("=" * 60)
            logger.info("STEP 1: Place Discovery")
            logger.info("=" * 60)
            
            discovery_output = config.get_output_path(
                "discovery",
                prefix="agencies",
                extension="csv"
            )
            
            map_centers = None
            cities = [geo.resolve_city_name(c) for c in cities]
            
            discovery_stats = self.discovery.discover_branches(
                businesses=businesses,
                cities=cities,
                business_type=business_type,
                map_centers=map_centers,
                brand_filter=None,
                output_path=discovery_output
            )
            
            stats["discovery"] = discovery_stats
            collection_input_file = discovery_output
        else:
            if discovery_input:
                collection_input_file = discovery_input
                logger.info(f"Using existing discovery file: {discovery_input}")
            else:
                # Use default file
                default_file = config.DEFAULT_DATA_DIR / "agencies_default.csv"
                if not default_file.exists():
                    raise FileNotFoundError(
                        f"No discovery input provided and default not found: "
                        f"{default_file}"
                    )
                collection_input_file = default_file
                logger.info(f"Using default file: {default_file}")
        
        # ====================
        # Step 2: Collection
        # ====================
        if not skip_collection:
            logger.info("=" * 60)
            logger.info("STEP 2: Review Collection")
            logger.info("=" * 60)
            
            if output_mode == "csv":
                collection_output = config.get_output_path(
                    "collection",
                    prefix="reviews",
                    extension="csv"
                )
                collection_stats = self.collector.collect_reviews(
                    input_file=collection_input_file,
                    output_mode=output_mode,
                    output_path=collection_output
                )
            else:
                collection_output_dir = config.OUTPUT_DIR / "reviews_json"
                collection_stats = self.collector.collect_reviews(
                    input_file=collection_input_file,
                    output_mode=output_mode,
                    output_dir=collection_output_dir
                )
                # For classification, need CSV
                collection_output = config.OUTPUT_DIR / "reviews_consolidated.csv"
            
            stats["collection"] = collection_stats
            classification_input_file = collection_output
        else:
            if collection_input:
                classification_input_file = collection_input
                logger.info(f"Using existing collection file: {collection_input}")
            else:
                raise ValueError(
                    "Collection skipped but no --collection-input provided"
                )
        
        # ====================
        # Step 3: Transform
        # ====================
        logger.info("=" * 60)
        logger.info("STEP 3: Transform (Normalize & Enrich)")
        logger.info("=" * 60)
        
        import pandas as pd
        from .transformers.normalize_reviews import normalize_reviews_df
        from .transformers.geocode import add_region
        
        # Load collection output
        df = pd.read_csv(classification_input_file)
        logger.info(f"Loaded {len(df)} reviews")
        
        # Normalize
        df = normalize_reviews_df(df)
        logger.info("✓ Normalized fields")
        
        # Add regions
        regions_path = config.REGIONS_FILE
        if regions_path.exists():
            try:
                df = add_region(df, regions_path)
                logger.info("✓ Added regions")
            except Exception as e:
                logger.warning(f"Region assignment failed: {e}")
        else:
            logger.warning(f"Regions file not found: {regions_path}")
        
        # Save normalized reviews to interim/transform
        transform_output = config.INTERIM_TRANSFORM_DIR / "reviews_normalized.parquet"
        transform_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(transform_output, index=False)
        
        stats["transform"] = {
            "reviews_normalized": len(df),
            "output_file": str(transform_output)
        }
        
        # ====================
        # Step 4: Classification
        # ====================
        if not skip_classification:
            logger.info("=" * 60)
            logger.info("STEP 4: Review Classification")
            logger.info("=" * 60)
            
            classification_output = config.get_output_path(
                "classification",
                prefix="reviews_classified",
                extension="csv"
            )
            
            # Use normalized data from transform step
            
            df = self.classifier.classify_batch(
                df,
                review_col="review_snippet",
                rating_col="review_rating"
            )
            
            if wide_format:
                df = self.classifier.convert_to_wide_format(df)
            
            df.to_csv(classification_output, index=False)
            
            stats["classification"] = {
                "reviews_classified": len(df),
                "output_file": str(classification_output)
            }
            
            logger.info(f"Classified reviews saved: {classification_output}")
        
        return stats


# =========================
# CLI Commands
# =========================
def cmd_discover(args):
    """Run discovery only"""
    engine = DiscoveryEngine(debug=args.debug)
    
    # Let engine load centers internally; normalize cities via alias index
    map_centers = None
    args.cities = [geo.resolve_city_name(c) for c in args.cities]
    
    output_path = args.output or config.get_output_path(
        "discovery",
        prefix="agencies",
        extension="csv"
    )
    
    stats = engine.discover_branches(
        businesses=args.businesses,
        cities=args.cities,
        business_type=getattr(args, 'business_type', None),
        map_centers=map_centers,
        brand_filter=args.brand_filter,
        output_path=output_path
    )
    
    print("\n✅ Discovery complete!")
    print(f"   Places found: {stats['total_places']}")
    print(f"   Output: {output_path}")


def cmd_collect(args):
    """Run collection only"""
    collector = ReviewCollector(debug=args.debug)
    
    output_path = args.output
    output_dir = args.output_dir
    
    if args.mode == "csv" and not output_path:
        output_path = config.get_output_path(
            "collection",
            prefix="reviews",
            extension="csv"
        )
    elif args.mode != "csv" and not output_dir:
        output_dir = config.OUTPUT_DIR / "reviews_json"
    
    stats = collector.collect_reviews(
        input_file=args.input,
        output_mode=args.mode,
        output_path=output_path,
        output_dir=output_dir
    )
    
    print(f"\n✅ Collection complete!")
    print(f"   Agencies: {stats['agencies_processed']}")
    print(f"   Reviews: {stats['total_reviews']}")


def cmd_transform(args):
    """Run transform step: normalize fields and add region"""
    import pandas as pd
    from .transformers.normalize_reviews import normalize_reviews_df
    from .transformers.geocode import add_region
    
    # Load interim reviews (output of collect)
    reviews_path = config.INTERIM_COLLECTION_DIR / "reviews.parquet"
    if not reviews_path.exists():
        # Fallback to CSV if parquet doesn't exist
        reviews_path = config.INTERIM_COLLECTION_DIR / "reviews.csv"
        if not reviews_path.exists():
            print(f"❌ Missing {reviews_path}")
            print("   Run 'collect' first to generate review data")
            sys.exit(1)
    
    print(f"📥 Loading reviews from: {reviews_path}")
    if reviews_path.suffix == ".parquet":
        df = pd.read_parquet(reviews_path)
    else:
        df = pd.read_csv(reviews_path)
    
    print(f"   Loaded {len(df)} reviews")
    
    # Normalize fields (dates, numbers, etc.)
    print("🔄 Normalizing review fields...")
    df = normalize_reviews_df(df)
    
    # Add region
    regions_path = args.regions
    if regions_path.exists():
        print(f"🗺️  Adding regions from: {regions_path}")
        try:
            df = add_region(df, regions_path)
            print(f"   ✓ Region assignment complete")
        except Exception as e:
            print(f"   ⚠️  Region assignment failed: {e}")
            print(f"   Continuing without regions...")
    else:
        print(f"⚠️  Regions file not found: {regions_path}")
        print("   Skipping region assignment")
        print("   To enable: place regions.geojson in 00_config/cities/")
    
    # Save to interim/transform
    output_path = config.INTERIM_TRANSFORM_DIR / "reviews_normalized.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"\n✅ Transform complete!")
    print(f"   Normalized reviews: {len(df)}")
    print(f"   Output: {output_path}")


def cmd_classify(args):
    """Run classification only"""
    import pandas as pd
    
    classifier = ReviewClassifier(debug=args.debug)
    
    df = pd.read_csv(args.input)
    
    df = classifier.classify_batch(
        df,
        review_col=args.review_col,
        rating_col=args.rating_col
    )
    
    if args.wide_format:
        df = classifier.convert_to_wide_format(df)
    
    output_path = args.output or config.get_output_path(
        "classification",
        prefix="reviews_classified",
        extension="csv"
    )
    
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Classification complete!")
    print(f"   Reviews: {len(df)}")
    print(f"   Output: {output_path}")


def cmd_pipeline(args):
    """Run full pipeline"""
    pipeline = Pipeline(debug=args.debug)
    
    stats = pipeline.run_full_pipeline(
        businesses=args.businesses,
        cities=args.cities,
        business_type=getattr(args, 'business_type', None),
        output_mode=args.output_mode,
        skip_discovery=args.skip_discovery,
        discovery_input=args.discovery_input,
        skip_collection=args.skip_collection,
        collection_input=args.collection_input,
        skip_classification=args.skip_classification,
        wide_format=args.wide_format
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    if "discovery" in stats:
        print(f"\n📍 Discovery: {stats['discovery']['total_places']} places")
    
    if "collection" in stats:
        print(f"📝 Collection: {stats['collection']['total_reviews']} reviews")
    
    if "transform" in stats:
        print(f"🔄 Transform: {stats['transform']['reviews_normalized']} normalized")
        print(f"   Output: {stats['transform']['output_file']}")
    
    if "classification" in stats:
        print(f"🤖 Classification: {stats['classification']['reviews_classified']} reviews")
        print(f"   Output: {stats['classification']['output_file']}")


# =========================
# Main CLI
# =========================
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Review Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (discover → collect → transform → classify)
  python -m review_analyzer.main pipeline \\
      --businesses "Attijariwafa Bank" \\
      --business-type "bank" --cities "Casablanca"
  
  # Discovery only
  python -m review_analyzer.main discover \\
      --businesses "Hilton" "Marriott" --cities "Marrakech"
  
  # Collection only
  python -m review_analyzer.main collect --input locations.csv --mode csv
  
  # Transform only (normalize + enrich)
  python -m review_analyzer.main transform --regions regions.geojson
  
  # Classification only
  python -m review_analyzer.main classify --input reviews.csv
        """
    )
    
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ===== DISCOVER =====
    discover_parser = subparsers.add_parser("discover", help="Discover places")
    discover_parser.add_argument("--businesses", nargs="+", required=True,
                                  help="Business names")
    discover_parser.add_argument("--business-type",
                                  help="Business type (bank, hotel, etc.)")
    discover_parser.add_argument("--cities", nargs="+", required=True)
    discover_parser.add_argument("--brand-filter", help="Regex brand filter")
    discover_parser.add_argument("--output", type=Path, help="Output CSV")
    discover_parser.set_defaults(func=cmd_discover)
    
    # ===== COLLECT =====
    collect_parser = subparsers.add_parser("collect", help="Collect reviews")
    collect_parser.add_argument("--input", type=Path, required=True)
    collect_parser.add_argument(
        "--mode", required=True,
        choices=["json", "json-per-city", "json-per-business", "csv"]
    )
    collect_parser.add_argument("--output", type=Path,
                                 help="Output file (csv/json)")
    collect_parser.add_argument("--output-dir", type=Path,
                                 help="Output dir (json-per-*)")
    collect_parser.set_defaults(func=cmd_collect)
    
    # ===== TRANSFORM =====
    transform_parser = subparsers.add_parser("transform",
                                              help="Normalize and enrich reviews")
    transform_parser.add_argument("--regions", type=Path,
                                   default=config.REGIONS_FILE,
                                   help="Path to regions GeoJSON file")
    transform_parser.set_defaults(func=cmd_transform)
    
    # ===== CLASSIFY =====
    classify_parser = subparsers.add_parser("classify", help="Classify reviews")
    classify_parser.add_argument("--input", type=Path, required=True)
    classify_parser.add_argument("--output", type=Path, help="Output CSV")
    classify_parser.add_argument("--review-col", default="review_snippet")
    classify_parser.add_argument("--rating-col", default="review_rating")
    classify_parser.add_argument("--wide-format", action="store_true")
    classify_parser.set_defaults(func=cmd_classify)
    
    # ===== PIPELINE =====
    pipeline_parser = subparsers.add_parser("pipeline",
                                             help="Run full pipeline")
    pipeline_parser.add_argument("--businesses", nargs="+", required=True)
    pipeline_parser.add_argument("--business-type",
                                  help="Business type")
    pipeline_parser.add_argument("--cities", nargs="+", required=True)
    pipeline_parser.add_argument(
        "--output-mode", default="csv",
        choices=["json", "json-per-city", "json-per-business", "csv"]
    )
    pipeline_parser.add_argument("--skip-discovery", action="store_true")
    pipeline_parser.add_argument("--discovery-input", type=Path)
    pipeline_parser.add_argument("--skip-collection", action="store_true")
    pipeline_parser.add_argument("--collection-input", type=Path)
    pipeline_parser.add_argument("--skip-classification", action="store_true")
    pipeline_parser.add_argument("--wide-format", action="store_true",
                                  default=True)
    pipeline_parser.set_defaults(func=cmd_pipeline)
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run command
    args.func(args)


if __name__ == "__main__":
    main()

