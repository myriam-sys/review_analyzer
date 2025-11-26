#!/usr/bin/env python3
"""
Database CLI for Review Analyzer
Commands for initializing, migrating, and querying the DuckDB database
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from review_analyzer import config
from review_analyzer.database import DatabaseManager, AnalyticsQueries
from review_analyzer.database.migrate import DataMigration, run_migration


def cmd_init(args):
    """Initialize database schema"""
    db_path = Path(args.db) if args.db else config.DATA_DIR / "review_analyzer.duckdb"
    
    print(f"🗄️  Initializing database: {db_path}")
    
    with DatabaseManager(db_path) as db:
        db.initialize_schema()
        stats = db.get_table_stats()
    
    print("✅ Database initialized!")
    print(f"   Tables created: places, reviews, classifications, processing_runs, review_categories")
    print(f"   Current stats: {stats}")


def cmd_migrate(args):
    """Migrate CSV data to database"""
    csv_path = Path(args.input) if args.input else config.PROC_CLASSIFIED_LATEST / "bank_reviews_classified.csv"
    db_path = Path(args.db) if args.db else config.DATA_DIR / "review_analyzer.duckdb"
    
    if not csv_path.exists():
        print(f"❌ Input file not found: {csv_path}")
        return 1
    
    print(f"📦 Migrating data:")
    print(f"   From: {csv_path}")
    print(f"   To: {db_path}")
    
    results = run_migration(
        csv_path=csv_path,
        db_path=db_path,
        verify=not args.no_verify
    )
    
    print("\n✅ Migration Complete!")
    print(f"   Places: {results['places_migrated']:,}")
    print(f"   Reviews: {results['reviews_migrated']:,}")
    print(f"   Classifications: {results['classifications_migrated']:,}")
    
    if "verification" in results:
        v = results["verification"]
        print(f"\n📊 Verification:")
        for table, count in v["table_stats"].items():
            print(f"   {table}: {count:,}")
        print(f"   Orphan reviews: {v['integrity_checks']['orphan_reviews']}")
        print(f"   Orphan classifications: {v['integrity_checks']['orphan_classifications']}")
    
    return 0


def cmd_stats(args):
    """Show database statistics"""
    db_path = Path(args.db) if args.db else config.DATA_DIR / "review_analyzer.duckdb"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("   Run 'db_cli.py init' first")
        return 1
    
    with DatabaseManager(db_path, read_only=True) as db:
        analytics = AnalyticsQueries(db)
        
        # Table stats
        table_stats = db.get_table_stats()
        print("📊 Table Statistics:")
        for table, count in table_stats.items():
            print(f"   {table}: {count:,}")
        
        # Overview stats
        overview = analytics.get_overview_stats()
        if overview:
            print(f"\n📈 Overview:")
            print(f"   Total Reviews: {overview['total_reviews']:,}")
            print(f"   Total Businesses: {overview['total_businesses']}")
            print(f"   Total Cities: {overview['total_cities']}")
            print(f"   Average Rating: {overview['avg_rating']:.2f}")
            
            print(f"\n😊 Sentiment Distribution:")
            for s in overview.get("sentiment_distribution", []):
                print(f"   {s['sentiment']}: {s['count']:,} ({s['percentage']}%)")
    
    return 0


def cmd_query(args):
    """Run analytical queries"""
    db_path = Path(args.db) if args.db else config.DATA_DIR / "review_analyzer.duckdb"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1
    
    with DatabaseManager(db_path, read_only=True) as db:
        analytics = AnalyticsQueries(db)
        
        if args.query == "business":
            df = analytics.get_business_stats(args.filter)
            print("\n📊 Business Statistics:")
            print(df.to_string(index=False))
            
        elif args.query == "city":
            df = analytics.get_city_stats(args.filter)
            print("\n🏙️  City Statistics:")
            print(df.to_string(index=False))
            
        elif args.query == "ranking":
            df = analytics.get_business_ranking(metric=args.metric or "avg_rating", top_n=args.top or 10)
            print(f"\n🏆 Business Ranking (by {args.metric or 'avg_rating'}):")
            print(df.to_string(index=False))
            
        elif args.query == "trends":
            df = analytics.get_temporal_trends(
                granularity=args.granularity or "month",
                business=args.filter
            )
            print("\n📈 Temporal Trends:")
            print(df.to_string(index=False))
            
        elif args.query == "issues":
            df = analytics.get_top_issues(business=args.filter, top_n=args.top or 10)
            print("\n⚠️  Top Issues:")
            print(df.to_string(index=False))
            
        elif args.query == "winners":
            df = analytics.get_city_winners()
            print("\n🥇 Best Business per City:")
            print(df.to_string(index=False))
            
        elif args.query == "compare":
            if not args.businesses:
                print("❌ --businesses required for compare query")
                return 1
            businesses = [b.strip() for b in args.businesses.split(",")]
            df = analytics.compare_businesses(businesses)
            print(f"\n⚖️  Business Comparison:")
            print(df.to_string(index=False))
        
        else:
            print(f"❌ Unknown query type: {args.query}")
            return 1
    
    return 0


def cmd_export(args):
    """Export database to CSV or Parquet"""
    db_path = Path(args.db) if args.db else config.DATA_DIR / "review_analyzer.duckdb"
    output_path = Path(args.output)
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1
    
    with DatabaseManager(db_path, read_only=True) as db:
        if output_path.suffix == ".parquet":
            count = db.export_to_parquet(output_path)
        else:
            count = db.export_to_csv(output_path, include_categories_wide=not args.no_wide)
        
        print(f"✅ Exported {count:,} rows to {output_path}")
    
    return 0


def cmd_sql(args):
    """Run raw SQL query"""
    db_path = Path(args.db) if args.db else config.DATA_DIR / "review_analyzer.duckdb"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1
    
    with DatabaseManager(db_path, read_only=True) as db:
        try:
            result = db.connection.execute(args.sql).fetchdf()
            print(result.to_string(index=False))
        except Exception as e:
            print(f"❌ SQL Error: {e}")
            return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Review Analyzer Database CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize database
  python db_cli.py init
  
  # Migrate existing CSV data
  python db_cli.py migrate
  
  # Show statistics
  python db_cli.py stats
  
  # Run queries
  python db_cli.py query business
  python db_cli.py query ranking --metric positive_rate --top 5
  python db_cli.py query trends --filter "Attijariwafa Bank" --granularity month
  python db_cli.py query issues --filter "CIH Bank"
  python db_cli.py query compare --businesses "Attijariwafa Bank,BMCE Bank,CIH Bank"
  
  # Export data
  python db_cli.py export --output output.csv
  python db_cli.py export --output output.parquet
  
  # Raw SQL
  python db_cli.py sql "SELECT business, COUNT(*) FROM reviews GROUP BY business"
        """
    )
    
    parser.add_argument("--db", help="Database path (default: data/review_analyzer.duckdb)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize database schema")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate CSV data to database")
    migrate_parser.add_argument("--input", help="Input CSV file")
    migrate_parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Run analytical queries")
    query_parser.add_argument("query", choices=["business", "city", "ranking", "trends", "issues", "winners", "compare"])
    query_parser.add_argument("--filter", help="Filter by business or city name")
    query_parser.add_argument("--metric", help="Ranking metric (avg_rating, positive_rate, total_reviews)")
    query_parser.add_argument("--top", type=int, help="Number of top results")
    query_parser.add_argument("--granularity", choices=["day", "week", "month", "quarter", "year"])
    query_parser.add_argument("--businesses", help="Comma-separated business names for comparison")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export database to file")
    export_parser.add_argument("--output", required=True, help="Output file path (.csv or .parquet)")
    export_parser.add_argument("--no-wide", action="store_true", help="Don't include wide format categories")
    
    # SQL command
    sql_parser = subparsers.add_parser("sql", help="Run raw SQL query")
    sql_parser.add_argument("sql", help="SQL query to execute")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to command
    commands = {
        "init": cmd_init,
        "migrate": cmd_migrate,
        "stats": cmd_stats,
        "query": cmd_query,
        "export": cmd_export,
        "sql": cmd_sql,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)

