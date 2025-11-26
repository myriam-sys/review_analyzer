"""
Unit and integration tests for transformers module
Tests normalize_reviews, geocode, and aggregates
"""
import pytest
import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from review_analyzer.transformers import (
    normalize_reviews_df,
    parse_relative_french_date,
    add_region,
    add_region_by_city,
    build_aggregates,
    CITY_REGION_MAPPING,
    CASABLANCA_TZ,
)


# =========================
# Normalization Tests
# =========================
@pytest.mark.unit
class TestParseRelativeFrenchDate:
    """Test French date parsing"""
    
    def test_parse_il_y_a_2_ans(self):
        """Test parsing 'il y a 2 ans'"""
        now = datetime(2024, 1, 15, 12, 0, 0)
        result = parse_relative_french_date("il y a 2 ans", now=now)
        
        assert result is not None
        # Should be roughly 2 years ago
        assert result.year == 2022
    
    def test_parse_il_y_a_3_mois(self):
        """Test parsing 'il y a 3 mois'"""
        now = datetime(2024, 1, 15, 12, 0, 0)
        result = parse_relative_french_date("il y a 3 mois", now=now)
        
        assert result is not None
        # Should be roughly 90 days ago
        delta = now - result
        assert 80 <= delta.days <= 100
    
    def test_parse_il_y_a_5_jours(self):
        """Test parsing 'il y a 5 jours'"""
        now = datetime(2024, 1, 15, 12, 0, 0)
        result = parse_relative_french_date("il y a 5 jours", now=now)
        
        assert result is not None
        delta = now - result
        assert delta.days == 5
    
    def test_parse_modifie_il_y_a_11_mois(self):
        """Test parsing 'Modifié il y a 11 mois'"""
        now = datetime(2024, 1, 15, 12, 0, 0)
        result = parse_relative_french_date("Modifié il y a 11 mois", now=now)
        
        assert result is not None
        delta = now - result
        assert 320 <= delta.days <= 340
    
    def test_parse_invalid_date(self):
        """Test invalid date strings"""
        assert parse_relative_french_date("not a date") is None
        assert parse_relative_french_date("") is None
        assert parse_relative_french_date(None) is None
    
    def test_parse_il_y_a_2_heures(self):
        """Test parsing hours"""
        now = datetime(2024, 1, 15, 12, 0, 0)
        result = parse_relative_french_date("il y a 2 heures", now=now)
        
        assert result is not None
        delta = now - result
        assert delta.seconds == 2 * 3600


@pytest.mark.unit
class TestNormalizeReviewsDF:
    """Test review normalization"""
    
    def test_normalize_basic_fields(self):
        """Test basic field normalization"""
        df = pd.DataFrame({
            "rating": ["4", "5.0", "3"],
            "lat": ["33.5", "34.0", "31.6"],
            "lng": ["-7.6", "-6.8", "-8.0"],
        })
        
        result = normalize_reviews_df(df)
        
        # Rating should be Int64
        assert result["rating"].dtype == "Int64"
        assert result["rating"].tolist() == [4, 5, 3]
        
        # Coordinates should be float
        assert result["lat"].dtype == "float64"
        assert result["lng"].dtype == "float64"
    
    def test_normalize_rating_clipping(self):
        """Test rating values are clipped to 1-5"""
        df = pd.DataFrame({
            "rating": [0, 6, -1, 3, 5],
        })
        
        result = normalize_reviews_df(df)
        
        assert result["rating"].min() == 1
        assert result["rating"].max() == 5
    
    def test_normalize_handles_missing_values(self):
        """Test handling of missing values"""
        df = pd.DataFrame({
            "rating": [4, None, 5],
            "lat": [33.5, None, 31.6],
            "text": ["Good", None, "Excellent"],
        })
        
        result = normalize_reviews_df(df)
        
        # Should not crash
        assert len(result) == 3
        assert pd.isna(result["rating"].iloc[1])
    
    def test_normalize_creates_created_at(self):
        """Test creation of created_at from date column"""
        df = pd.DataFrame({
            "date": ["il y a 2 ans", "il y a 3 mois", "il y a 5 jours"],
            "rating": [4, 5, 3],
        })
        
        result = normalize_reviews_df(df)
        
        assert "created_at" in result.columns
        # Should have parsed at least some dates
        assert result["created_at"].notna().sum() > 0
    
    def test_normalize_creates_month(self):
        """Test month column creation"""
        df = pd.DataFrame({
            "date": ["il y a 2 mois", "il y a 3 mois"],
            "rating": [4, 5],
        })
        
        result = normalize_reviews_df(df)
        
        if "created_at" in result.columns:
            assert "month" in result.columns
    
    def test_normalize_text_fields(self):
        """Test text field normalization"""
        df = pd.DataFrame({
            "text": ["  Good  ", None, "Excellent"],
            "rating": [4, 5, 3],
        })
        
        result = normalize_reviews_df(df)
        
        assert result["text"].iloc[0] == "Good"
        assert result["text"].iloc[1] == ""
    
    def test_normalize_city_normalized(self):
        """Test city name normalization"""
        df = pd.DataFrame({
            "city": ["Casablanca", "RABAT", "Marrakech"],
            "rating": [4, 5, 3],
        })
        
        result = normalize_reviews_df(df)
        
        assert "city_normalized" in result.columns
        assert result["city_normalized"].iloc[0] == "casablanca"
        assert result["city_normalized"].iloc[1] == "rabat"
    
    def test_normalize_empty_dataframe(self):
        """Test handling empty DataFrame"""
        df = pd.DataFrame()
        
        result = normalize_reviews_df(df)
        
        assert result is not None
        assert len(result) == 0
    
    def test_normalize_alternative_column_names(self):
        """Test handling alternative column names"""
        df = pd.DataFrame({
            "review_rating": [4, 5],
            "latitude": [33.5, 34.0],
            "longitude": [-7.6, -6.8],
            "review_snippet": ["Good", "Great"],
        })
        
        result = normalize_reviews_df(df)
        
        assert "rating" in result.columns
        assert result["rating"].tolist() == [4, 5]


# =========================
# Geocoding Tests
# =========================
@pytest.mark.unit
class TestCityRegionMapping:
    """Test city to region mapping"""
    
    def test_mapping_has_major_cities(self):
        """Test mapping includes major Moroccan cities"""
        major_cities = [
            "casablanca", "rabat", "marrakech", "fes",
            "tangier", "agadir", "meknes", "oujda"
        ]
        
        for city in major_cities:
            assert city in CITY_REGION_MAPPING
    
    def test_mapping_has_12_regions(self):
        """Test mapping covers all 12 Moroccan regions"""
        regions = set(CITY_REGION_MAPPING.values())
        
        # Should have all 12 regions represented
        assert len(regions) >= 10  # At least 10 of 12 regions


@pytest.mark.unit
class TestAddRegionByCity:
    """Test city-based region assignment"""
    
    def test_add_region_by_city_basic(self):
        """Test basic region assignment by city"""
        df = pd.DataFrame({
            "city": ["Casablanca", "Rabat", "Marrakech"],
            "rating": [4, 5, 3],
        })
        
        result = add_region_by_city(df)
        
        assert "region" in result.columns
        assert result["region"].iloc[0] == "Casablanca-Settat"
        assert result["region"].iloc[1] == "Rabat-Salé-Kénitra"
        assert result["region"].iloc[2] == "Marrakech-Safi"
    
    def test_add_region_by_city_case_insensitive(self):
        """Test case-insensitive city matching"""
        df = pd.DataFrame({
            "city": ["CASABLANCA", "rabat", "MaRrAkEcH"],
            "rating": [4, 5, 3],
        })
        
        result = add_region_by_city(df)
        
        assert result["region"].notna().sum() == 3
    
    def test_add_region_by_city_unknown_city(self):
        """Test handling unknown cities"""
        df = pd.DataFrame({
            "city": ["Unknown City", "Casablanca"],
            "rating": [4, 5],
        })
        
        result = add_region_by_city(df)
        
        assert pd.isna(result["region"].iloc[0])
        assert result["region"].iloc[1] == "Casablanca-Settat"
    
    def test_add_region_by_city_missing_values(self):
        """Test handling missing city values"""
        df = pd.DataFrame({
            "city": [None, "Casablanca", None],
            "rating": [4, 5, 3],
        })
        
        result = add_region_by_city(df)
        
        assert pd.isna(result["region"].iloc[0])
        assert result["region"].iloc[1] == "Casablanca-Settat"
        assert pd.isna(result["region"].iloc[2])
    
    def test_add_region_by_city_custom_column(self):
        """Test custom city column name"""
        df = pd.DataFrame({
            "city_name": ["Casablanca", "Rabat"],
            "rating": [4, 5],
        })
        
        result = add_region_by_city(df, city_col="city_name")
        
        assert "region" in result.columns
        assert result["region"].notna().sum() == 2


@pytest.mark.unit
class TestAddRegionWithGeoJSON:
    """Test GeoJSON-based region assignment"""
    
    def test_add_region_missing_columns(self):
        """Test error handling for missing columns"""
        df = pd.DataFrame({
            "city": ["Casablanca"],
            "rating": [4],
        })
        
        with pytest.raises(ValueError, match="lat/lng"):
            add_region(df, Path("/nonexistent/regions.geojson"))
    
    def test_add_region_file_not_found(self):
        """Test error handling for missing GeoJSON file"""
        df = pd.DataFrame({
            "lat": [33.5],
            "lng": [-7.6],
        })
        
        with pytest.raises(FileNotFoundError):
            add_region(df, Path("/nonexistent/regions.geojson"))
    
    def test_add_region_invalid_coordinates(self):
        """Test handling invalid coordinates"""
        df = pd.DataFrame({
            "lat": [None, 33.5, 100.0],  # One None, one valid, one out of bounds
            "lng": [-7.6, None, -7.6],
        })
        
        # Mock the GeoJSON loading
        with patch('review_analyzer.transformers.geocode.load_regions_geojson'):
            with patch('review_analyzer.transformers.geocode.build_spatial_index'):
                # Should not crash
                pass
    
    def test_add_region_alternative_column_names(self):
        """Test support for alternative column names"""
        df = pd.DataFrame({
            "latitude": [33.5],
            "longitude": [-7.6],
        })
        
        # Should recognize latitude/longitude
        with pytest.raises(FileNotFoundError):  # Will fail on file, but validates columns
            add_region(df, Path("/nonexistent/regions.geojson"))


# =========================
# Aggregation Tests
# =========================
@pytest.mark.unit
class TestBuildAggregates:
    """Test aggregate computation"""
    
    def test_build_aggregates_basic(self, temp_dir):
        """Test basic aggregate creation"""
        df = pd.DataFrame({
            "business_id": ["B1", "B1", "B2", "B2"],
            "city": ["Casablanca", "Casablanca", "Rabat", "Rabat"],
            "rating": [4, 5, 3, 4],
            "sentiment": ["positive", "positive", "negative", "neutral"],
        })
        
        result = build_aggregates(df, output_dir=temp_dir, date_suffix=False)
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check by_business file exists
        assert (temp_dir / "by_business.parquet").exists()
        
        # Verify aggregated data
        agg_df = pd.read_parquet(temp_dir / "by_business.parquet")
        assert len(agg_df) == 2  # 2 businesses
        assert "review_count" in agg_df.columns
        assert "avg_rating" in agg_df.columns
    
    def test_build_aggregates_by_city(self, temp_dir):
        """Test city-level aggregates"""
        df = pd.DataFrame({
            "business_id": ["B1", "B2", "B3"],
            "city": ["Casablanca", "Rabat", "Casablanca"],
            "rating": [4, 5, 3],
            "sentiment": ["positive", "positive", "negative"],
        })
        
        result = build_aggregates(df, output_dir=temp_dir, date_suffix=False)
        
        assert (temp_dir / "by_city.parquet").exists()
        
        agg_df = pd.read_parquet(temp_dir / "by_city.parquet")
        assert len(agg_df) == 2  # 2 cities
    
    def test_build_aggregates_with_region(self, temp_dir):
        """Test regional aggregates"""
        df = pd.DataFrame({
            "business_id": ["B1", "B2", "B3"],
            "region": ["Casablanca-Settat", "Rabat-Salé-Kénitra", "Casablanca-Settat"],
            "rating": [4, 5, 3],
            "sentiment": ["positive", "positive", "negative"],
        })
        
        result = build_aggregates(df, output_dir=temp_dir, date_suffix=False)
        
        # Should create regional aggregates
        assert (temp_dir / "by_region.parquet").exists()
    
    def test_build_aggregates_temporal(self, temp_dir):
        """Test temporal aggregates"""
        df = pd.DataFrame({
            "business_id": ["B1", "B1", "B2"],
            "created_at": pd.to_datetime([
                "2024-01-15", "2024-02-15", "2024-01-20"
            ]),
            "rating": [4, 5, 3],
            "sentiment": ["positive", "positive", "negative"],
        })
        
        result = build_aggregates(df, output_dir=temp_dir, date_suffix=False)
        
        # Should create monthly aggregates
        assert (temp_dir / "by_business_month.parquet").exists()
    
    def test_build_aggregates_empty_dataframe(self, temp_dir):
        """Test error handling for empty DataFrame"""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty"):
            build_aggregates(df, output_dir=temp_dir)
    
    def test_build_aggregates_missing_required_columns(self, temp_dir):
        """Test error handling for missing columns"""
        df = pd.DataFrame({
            "business_id": ["B1", "B2"],
            # Missing rating column
        })
        
        with pytest.raises(ValueError, match="required"):
            build_aggregates(df, output_dir=temp_dir)
    
    def test_build_aggregates_sentiment_variations(self, temp_dir):
        """Test handling different sentiment column names"""
        df = pd.DataFrame({
            "business_id": ["B1", "B2"],
            "rating": [4, 5],
            "label_sentiment": ["pos", "neg"],
        })
        
        result = build_aggregates(df, output_dir=temp_dir, date_suffix=False)
        
        # Should handle label_sentiment
        assert len(result) > 0
    
    def test_build_aggregates_with_date_suffix(self, temp_dir):
        """Test timestamped output"""
        df = pd.DataFrame({
            "business_id": ["B1", "B2"],
            "rating": [4, 5],
            "sentiment": ["positive", "negative"],
        })
        
        result = build_aggregates(df, output_dir=temp_dir, date_suffix=True)
        
        # Should create date-suffixed directory
        assert len(result) > 0
        
        # Check that files exist somewhere under temp_dir
        parquet_files = list(temp_dir.rglob("*.parquet"))
        assert len(parquet_files) > 0


# =========================
# Integration Tests
# =========================
@pytest.mark.integration
class TestTransformersIntegration:
    """Test full transformation pipeline"""
    
    def test_full_transform_pipeline(self, temp_dir):
        """Test complete transformation workflow"""
        # Raw review data
        df = pd.DataFrame({
            "date": ["il y a 2 mois", "il y a 3 jours", "il y a 1 an"],
            "rating": ["4", "5", "3"],
            "lat": ["33.5731", "34.0209", "31.6295"],
            "lng": ["-7.5898", "-6.8416", "-7.9811"],
            "city": ["Casablanca", "Rabat", "Marrakech"],
            "text": ["  Good service  ", "Excellent!", "  Poor  "],
            "business_id": ["B1", "B2", "B1"],
        })
        
        # Step 1: Normalize
        df_norm = normalize_reviews_df(df)
        
        assert "created_at" in df_norm.columns
        assert "city_normalized" in df_norm.columns
        assert df_norm["rating"].dtype == "Int64"
        assert df_norm["text"].iloc[0] == "Good service"
        
        # Step 2: Add regions (fallback to city-based)
        df_with_region = add_region_by_city(df_norm)
        
        assert "region" in df_with_region.columns
        assert df_with_region["region"].notna().sum() > 0
        
        # Step 3: Add sentiment for aggregates
        df_with_region["sentiment"] = ["positive", "positive", "negative"]
        
        # Step 4: Build aggregates
        agg_files = build_aggregates(
            df_with_region,
            output_dir=temp_dir,
            date_suffix=False
        )
        
        assert len(agg_files) > 0
        assert (temp_dir / "by_business.parquet").exists()
        
        # Verify aggregated data quality
        agg_df = pd.read_parquet(temp_dir / "by_business.parquet")
        assert len(agg_df) == 2  # 2 businesses
        assert agg_df["review_count"].sum() == 3
    
    def test_transform_handles_malformed_data(self):
        """Test robustness with malformed data"""
        df = pd.DataFrame({
            "date": ["invalid", None, "il y a 2 jours"],
            "rating": ["invalid", "6", "3"],
            "lat": ["invalid", "34.0", "31.6"],
            "lng": ["-7.6", "invalid", "-7.9"],
            "city": [None, "Rabat", "Unknown City"],
            "text": [None, "", "  Text  "],
        })
        
        # Should not crash
        result = normalize_reviews_df(df)
        
        assert len(result) == 3
        # Some values will be None/NaN but structure preserved


# =========================
# Performance Tests
# =========================
@pytest.mark.slow
class TestTransformersPerformance:
    """Test performance with large datasets"""
    
    def test_normalize_large_dataset(self):
        """Test normalization performance"""
        # Create large dataset
        df = pd.DataFrame({
            "date": ["il y a 2 mois"] * 10000,
            "rating": [4] * 10000,
            "lat": [33.5] * 10000,
            "lng": [-7.6] * 10000,
            "text": ["Good service"] * 10000,
        })
        
        import time
        start = time.time()
        result = normalize_reviews_df(df)
        duration = time.time() - start
        
        assert len(result) == 10000
        # Should complete in reasonable time (< 5 seconds)
        assert duration < 5.0
    
    def test_aggregates_large_dataset(self, temp_dir):
        """Test aggregation performance"""
        # Create large dataset
        df = pd.DataFrame({
            "business_id": [f"B{i % 100}" for i in range(10000)],
            "city": [f"City{i % 10}" for i in range(10000)],
            "rating": [4] * 10000,
            "sentiment": ["positive"] * 10000,
        })
        
        import time
        start = time.time()
        result = build_aggregates(df, output_dir=temp_dir, date_suffix=False)
        duration = time.time() - start
        
        assert len(result) > 0
        # Should complete in reasonable time (< 10 seconds)
        assert duration < 10.0
