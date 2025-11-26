"""
Unit and integration tests for discover module
Tests DiscoveryEngine with mocked API calls and real data structures
"""
import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from review_analyzer.discover import DiscoveryEngine


# =========================
# DiscoveryEngine Tests
# =========================
@pytest.mark.unit
class TestDiscoveryEngineInit:
    """Test DiscoveryEngine initialization"""
    
    def test_engine_initialization(self, mock_env_vars):
        """Test engine can be initialized"""
        engine = DiscoveryEngine(debug=False)
        assert engine is not None
        assert hasattr(engine, 'client')
    
    def test_engine_debug_mode(self, mock_env_vars):
        """Test engine in debug mode"""
        engine = DiscoveryEngine(debug=True)
        assert engine.debug is True


@pytest.mark.unit
class TestDiscoverySingleCity:
    """Test discovery for single city/bank combination"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_discover_single_bank_city(
        self, mock_client_class, temp_dir, mock_serpapi_response
    ):
        """Test discovery for one bank in one city"""
        # Setup mock
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_response
        mock_client_class.return_value = mock_client
        
        engine = DiscoveryEngine(debug=False)
        output_path = temp_dir / "results.csv"
        
        stats = engine.discover_branches(
            banks=["Attijariwafa Bank"],
            cities=["Casablanca"],
            map_centers={"Casablanca": "@33.5,-7.6,12z"},
            brand_filter=None,
            output_path=output_path
        )
        
        assert stats["total_places"] > 0
        assert output_path.exists()
        
        # Verify CSV structure
        df = pd.read_csv(output_path)
        assert "_place_id" in df.columns
        assert "_bank" in df.columns
        assert "_city" in df.columns
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_discover_with_brand_filter(
        self, mock_client_class, temp_dir, mock_serpapi_response
    ):
        """Test discovery with brand name filtering"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_response
        mock_client_class.return_value = mock_client
        
        engine = DiscoveryEngine(debug=False)
        output_path = temp_dir / "filtered.csv"
        
        stats = engine.discover_branches(
            banks=["Attijariwafa Bank"],
            cities=["Casablanca"],
            map_centers={"Casablanca": "@33.5,-7.6,12z"},
            brand_filter="Attijariwafa",
            output_path=output_path
        )
        
        assert stats is not None
        assert output_path.exists()


@pytest.mark.unit
class TestDiscoveryMultipleCities:
    """Test discovery across multiple cities"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_discover_multiple_cities(
        self, mock_client_class, temp_dir, mock_serpapi_response
    ):
        """Test discovery for one bank in multiple cities"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_response
        mock_client_class.return_value = mock_client
        
        engine = DiscoveryEngine(debug=False)
        output_path = temp_dir / "multi_city.csv"
        
        cities = ["Casablanca", "Rabat", "Marrakech"]
        map_centers = {
            city: f"@33.{i},-7.{i},12z"
            for i, city in enumerate(cities)
        }
        
        stats = engine.discover_branches(
            banks=["Attijariwafa Bank"],
            cities=cities,
            map_centers=map_centers,
            brand_filter=None,
            output_path=output_path
        )
        
        assert stats["total_places"] > 0
        
        df = pd.read_csv(output_path)
        # Should have results from multiple cities
        assert df["_city"].nunique() > 0
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_discover_multiple_banks(
        self, mock_client_class, temp_dir, mock_serpapi_response
    ):
        """Test discovery for multiple banks in one city"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_response
        mock_client_class.return_value = mock_client
        
        engine = DiscoveryEngine(debug=False)
        output_path = temp_dir / "multi_bank.csv"
        
        banks = ["Attijariwafa Bank", "BMCE Bank", "CIH Bank"]
        
        stats = engine.discover_branches(
            banks=banks,
            cities=["Casablanca"],
            map_centers={"Casablanca": "@33.5,-7.6,12z"},
            brand_filter=None,
            output_path=output_path
        )
        
        assert stats["total_places"] > 0
        
        df = pd.read_csv(output_path)
        # Should have results for multiple banks
        assert df["_bank"].nunique() > 0


@pytest.mark.unit
class TestDiscoveryDeduplication:
    """Test deduplication logic"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_duplicate_places_removed(
        self, mock_client_class, temp_dir
    ):
        """Test that duplicate place_ids are removed"""
        # Mock response with duplicates
        mock_response = {
            "local_results": [
                {
                    "place_id": "ChIJSAME",
                    "title": "Bank Branch 1",
                    "address": "Address 1",
                },
                {
                    "place_id": "ChIJSAME",  # Duplicate
                    "title": "Bank Branch 1 (duplicate)",
                    "address": "Address 1",
                },
                {
                    "place_id": "ChIJDIFF",
                    "title": "Bank Branch 2",
                    "address": "Address 2",
                },
            ]
        }
        
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        engine = DiscoveryEngine(debug=False)
        output_path = temp_dir / "dedup.csv"
        
        stats = engine.discover_branches(
            banks=["Test Bank"],
            cities=["Casablanca"],
            map_centers={"Casablanca": "@33.5,-7.6,12z"},
            brand_filter=None,
            output_path=output_path
        )
        
        df = pd.read_csv(output_path)
        
        # Should have only 2 unique places, not 3
        assert len(df) == 2
        assert df["_place_id"].nunique() == 2


@pytest.mark.unit
class TestDiscoveryErrorHandling:
    """Test error handling in discovery"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_handles_empty_results(
        self, mock_client_class, temp_dir
    ):
        """Test handling when no results found"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = {"local_results": []}
        mock_client_class.return_value = mock_client
        
        engine = DiscoveryEngine(debug=False)
        output_path = temp_dir / "empty.csv"
        
        stats = engine.discover_branches(
            banks=["NonexistentBank"],
            cities=["Casablanca"],
            map_centers={"Casablanca": "@33.5,-7.6,12z"},
            brand_filter=None,
            output_path=output_path
        )
        
        assert stats["total_places"] == 0
        
        # File should still be created (empty)
        assert output_path.exists()
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_handles_api_errors(
        self, mock_client_class, temp_dir
    ):
        """Test handling API errors gracefully"""
        mock_client = MagicMock()
        mock_client.search_google_maps.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        engine = DiscoveryEngine(debug=False)
        output_path = temp_dir / "error.csv"
        
        # Should handle error, not crash
        with pytest.raises(Exception):
            engine.discover_branches(
                banks=["Test Bank"],
                cities=["Casablanca"],
                map_centers={"Casablanca": "@33.5,-7.6,12z"},
                brand_filter=None,
                output_path=output_path
            )


@pytest.mark.unit
class TestDiscoveryOutputFormat:
    """Test output format and data structure"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_output_has_required_columns(
        self, mock_client_class, temp_dir, mock_serpapi_response
    ):
        """Test that output CSV has all required columns"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_response
        mock_client_class.return_value = mock_client
        
        engine = DiscoveryEngine(debug=False)
        output_path = temp_dir / "output.csv"
        
        engine.discover_branches(
            banks=["Attijariwafa Bank"],
            cities=["Casablanca"],
            map_centers={"Casablanca": "@33.5,-7.6,12z"},
            brand_filter=None,
            output_path=output_path
        )
        
        df = pd.read_csv(output_path)
        
        required_columns = ["_place_id", "_bank", "_city"]
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_output_data_types(
        self, mock_client_class, temp_dir, mock_serpapi_response
    ):
        """Test that output columns have correct data types"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_response
        mock_client_class.return_value = mock_client
        
        engine = DiscoveryEngine(debug=False)
        output_path = temp_dir / "types.csv"
        
        engine.discover_branches(
            banks=["Attijariwafa Bank"],
            cities=["Casablanca"],
            map_centers={"Casablanca": "@33.5,-7.6,12z"},
            brand_filter=None,
            output_path=output_path
        )
        
        df = pd.read_csv(output_path)
        
        # Check types
        assert df["_place_id"].dtype == object  # String
        assert df["_bank"].dtype == object  # String
        assert df["_city"].dtype == object  # String


# =========================
# Integration Tests
# =========================
@pytest.mark.integration
class TestDiscoveryIntegration:
    """Integration tests with real-like workflows"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_full_discovery_workflow(
        self, mock_client_class, temp_dir, mock_serpapi_response
    ):
        """Test complete discovery workflow"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_response
        mock_client_class.return_value = mock_client
        
        engine = DiscoveryEngine(debug=False)
        output_path = temp_dir / "full_workflow.csv"
        
        # Run discovery
        stats = engine.discover_branches(
            banks=["Attijariwafa Bank", "BMCE Bank"],
            cities=["Casablanca", "Rabat"],
            map_centers={
                "Casablanca": "@33.5,-7.6,12z",
                "Rabat": "@34.0,-6.8,12z"
            },
            brand_filter=None,
            output_path=output_path
        )
        
        # Verify stats
        assert "total_places" in stats
        assert stats["total_places"] > 0
        
        # Verify file
        assert output_path.exists()
        df = pd.read_csv(output_path)
        
        # Verify data quality
        assert len(df) > 0
        assert df["_place_id"].notna().all()
        assert df["_bank"].notna().all()
        assert df["_city"].notna().all()


# =========================
# Performance Tests
# =========================
@pytest.mark.slow
class TestDiscoveryPerformance:
    """Test discovery performance with large datasets"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_performance_many_cities(
        self, mock_client_class, temp_dir, mock_serpapi_response
    ):
        """Test performance with many cities"""
        import time
        
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_response
        mock_client_class.return_value = mock_client
        
        engine = DiscoveryEngine(debug=False)
        output_path = temp_dir / "perf_cities.csv"
        
        # 10 cities
        cities = [f"City_{i}" for i in range(10)]
        map_centers = {
            city: f"@{33+i*0.1},-{7+i*0.1},12z"
            for i, city in enumerate(cities)
        }
        
        start = time.time()
        stats = engine.discover_branches(
            banks=["Test Bank"],
            cities=cities,
            map_centers=map_centers,
            brand_filter=None,
            output_path=output_path
        )
        duration = time.time() - start
        
        # Should complete reasonably fast (mocked, so no real delays)
        assert duration < 5.0
        assert stats["total_places"] >= 0
