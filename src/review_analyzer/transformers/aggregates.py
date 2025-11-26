"""
Aggregates Module - Pre-compute rollups for analysis

Creates aggregated views of classified reviews for:
- Dashboard visualization
- Performance tracking
- Trend analysis
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
from .. import config


def build_aggregates(
    df_labeled: pd.DataFrame,
    output_dir: Optional[Path] = None,
    date_suffix: bool = True
) -> dict:
    """
    Build and save aggregated views of classified reviews
    
    Args:
        df_labeled: DataFrame with classified reviews (must have rating,
                    sentiment/category columns)
        output_dir: Output directory (default: config.PROCESSED_DIR/aggregates)
        date_suffix: If True, creates YYYYMMDD subfolder
        
    Returns:
        Dictionary with paths to saved aggregate files
        
    Raises:
        ValueError: If required columns are missing
        
    Note:
        Creates aggregates by:
        - business_id
        - city
        - business_id + city
        - business_id + month (temporal)
        - region (if available)
    """
    if df_labeled is None or df_labeled.empty:
        raise ValueError("Cannot build aggregates from empty DataFrame")
    
    # Validate required columns
    required = ["rating"]
    missing = [col for col in required if col not in df_labeled.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Set up output directory
    if output_dir is None:
        if date_suffix:
            output_dir = config.get_timestamped_path(
                config.PROC_CLASSIFIED_DIR / "aggregates"
            )
        else:
            output_dir = config.PROC_CLASSIFIED_DIR / "aggregates"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare sentiment columns (handle different naming conventions)
    df = df_labeled.copy()
    
    # Detect sentiment column
    sentiment_col = None
    for col in ["label_sentiment", "sentiment", "category_sentiment"]:
        if col in df.columns:
            sentiment_col = col
            break
    
    # Create standardized sentiment column
    if sentiment_col:
        df["sentiment"] = df[sentiment_col]
    else:
        # Try to infer from category names
        if "category" in df.columns:
            df["sentiment"] = df["category"].apply(
                lambda x: "positive" if "positif" in str(x).lower()
                else ("negative" if "négatif" in str(x).lower() else "neutral")
            )
        else:
            df["sentiment"] = "unknown"
    
    saved_files = {}
    
    # Standardize business column name (handle alternatives: bank, business, business_id)
    business_col = None
    for col in ["business_id", "bank", "business", "business_name"]:
        if col in df.columns:
            business_col = col
            break
    
    if business_col and business_col != "business_id":
        df["business_id"] = df[business_col]
        print(f"ℹ️  Using '{business_col}' as business identifier")
    
    def safe_agg_by(cols: List[str], filename: str):
        """Aggregate with error handling"""
        # Check if all grouping columns exist
        available_cols = [c for c in cols if c in df.columns]
        if not available_cols:
            print(f"⚠️  Skipping {filename}: missing columns {cols}")
            return None
        
        try:
            g = df.groupby(available_cols, dropna=False)
            
            agg_dict = {
                "review_count": g.size(),
                "avg_rating": g["rating"].mean(),
            }
            
            # Add sentiment shares if available
            if "sentiment" in df.columns:
                agg_dict.update({
                    "positive_share": g["sentiment"].apply(
                        lambda x: (x.str.lower().str.contains("pos")).mean()
                    ),
                    "negative_share": g["sentiment"].apply(
                        lambda x: (x.str.lower().str.contains("neg")).mean()
                    ),
                    "neutral_share": g["sentiment"].apply(
                        lambda x: (x.str.lower().str.contains("neut")).mean()
                    ),
                })
            
            result = pd.DataFrame(agg_dict).reset_index()
            output_path = output_dir / filename
            result.to_parquet(output_path, index=False)
            saved_files[filename] = output_path
            print(f"✓ Saved {filename}: {len(result)} rows")
            return output_path
            
        except Exception as e:
            print(f"⚠️  Error creating {filename}: {e}")
            return None
    
    # Core aggregations
    safe_agg_by(["business_id"], "by_business.parquet")
    safe_agg_by(["city"], "by_city.parquet")
    safe_agg_by(["business_id", "city"], "by_business_city.parquet")
    
    # Regional aggregation (if region column exists)
    if "region" in df.columns:
        safe_agg_by(["region"], "by_region.parquet")
        safe_agg_by(["business_id", "region"], "by_business_region.parquet")
    
    # Temporal aggregation (if created_at exists)
    if "created_at" in df.columns or "month" in df.columns:
        if "month" not in df.columns and "created_at" in df.columns:
            df["month"] = (
                pd.to_datetime(df["created_at"], errors="coerce")
                .dt.to_period("M")
                .dt.to_timestamp()
            )
        
        if "month" in df.columns:
            safe_agg_by(["business_id", "month"], "by_business_month.parquet")
            safe_agg_by(["city", "month"], "by_city_month.parquet")
    
    print(f"\n✅ Aggregates saved to {output_dir}")
    print(f"   Files created: {len(saved_files)}")
    
    return saved_files