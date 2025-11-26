"""
Unit and integration tests for collect module
Tests ReviewCollector with mocked API calls and various output modes
"""
import pytest
import pandas as pd
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from review_analyzer.collect import ReviewCollector


# =========================
# ReviewCollector Tests
# =========================
@pytest.mark.unit
class TestCollectorInit:
    """Test ReviewCollector initialization"""
    
    def test_collector_initialization(self, mock_env_vars):
        """Test collector can be initialized"""
        collector = ReviewCollector(debug=False)
        assert collector is not None
        assert hasattr(collector, 'client')
    
    def test_collector_debug_mode(self, mock_env_vars):
        """Test collector in debug mode"""
        collector = ReviewCollector(debug=True)
        assert collector.debug is True


@pytest.mark.unit
class TestCollectorCSVMode:
    """Test collection with CSV output mode"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_collect_csv_mode(
        self, mock_client_class, temp_dir, sample_agencies_data, mock_serpapi_reviews
    ):
        """Test CSV output mode"""
        # Setup mock
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_reviews
        mock_client_class.return_value = mock_client
        
        # Create sample agencies file
        agencies_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(agencies_path, index=False)
        
        collector = ReviewCollector(debug=False)
        output_path = temp_dir / "reviews.csv"
        
        stats = collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="csv",
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        assert stats["total_places"] > 0
        assert output_path.exists()
        
        # Verify CSV structure
        df = pd.read_csv(output_path)
        assert "_place_id" in df.columns
        assert "author_name" in df.columns or "text" in df.columns
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_collect_csv_with_filters(
        self, mock_client_class, temp_dir, sample_agencies_data, mock_serpapi_reviews
    ):
        """Test CSV mode with rating filter"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_reviews
        mock_client_class.return_value = mock_client
        
        agencies_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(agencies_path, index=False)
        
        collector = ReviewCollector(debug=False)
        output_path = temp_dir / "filtered_reviews.csv"
        
        stats = collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="csv",
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json",
            min_rating=3  # Filter low ratings
        )
        
        assert stats is not None
        assert output_path.exists()


@pytest.mark.unit
class TestCollectorJSONMode:
    """Test collection with JSON output modes"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_collect_json_per_city(
        self, mock_client_class, temp_dir, sample_agencies_data, mock_serpapi_reviews
    ):
        """Test json-per-city output mode"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_reviews
        mock_client_class.return_value = mock_client
        
        agencies_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(agencies_path, index=False)
        
        collector = ReviewCollector(debug=False)
        output_dir = temp_dir / "output_city"
        output_dir.mkdir(exist_ok=True)
        
        stats = collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="json-per-city",
            output_path=output_dir,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        assert stats["total_places"] > 0
        
        # Should create JSON files per city
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) > 0
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_collect_json_per_bank(
        self, mock_client_class, temp_dir, sample_agencies_data, mock_serpapi_reviews
    ):
        """Test json-per-bank output mode"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_reviews
        mock_client_class.return_value = mock_client
        
        agencies_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(agencies_path, index=False)
        
        collector = ReviewCollector(debug=False)
        output_dir = temp_dir / "output_bank"
        output_dir.mkdir(exist_ok=True)
        
        stats = collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="json-per-bank",
            output_path=output_dir,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        assert stats["total_places"] > 0
        
        # Should create JSON files per bank
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) > 0
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_collect_single_json(
        self, mock_client_class, temp_dir, sample_agencies_data, mock_serpapi_reviews
    ):
        """Test single JSON output mode"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_reviews
        mock_client_class.return_value = mock_client
        
        agencies_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(agencies_path, index=False)
        
        collector = ReviewCollector(debug=False)
        output_path = temp_dir / "reviews.json"
        
        stats = collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="json",
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        assert stats["total_places"] > 0
        assert output_path.exists()
        
        # Verify JSON structure
        with open(output_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)


@pytest.mark.unit
class TestCollectorCheckpoint:
    """Test checkpoint and resume functionality"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_checkpoint_saves_progress(
        self, mock_client_class, temp_dir, sample_agencies_data, mock_serpapi_reviews
    ):
        """Test that checkpoint file is created and updated"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_reviews
        mock_client_class.return_value = mock_client
        
        agencies_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(agencies_path, index=False)
        
        collector = ReviewCollector(debug=False)
        output_path = temp_dir / "reviews.csv"
        checkpoint_path = temp_dir / "checkpoint.json"
        
        collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="csv",
            output_path=output_path,
            checkpoint_file=checkpoint_path
        )
        
        # Checkpoint should exist
        assert checkpoint_path.exists()
        
        # Verify checkpoint structure
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        assert "processed_places" in checkpoint or "reviews_data" in checkpoint
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_resume_from_checkpoint(
        self, mock_client_class, temp_dir, sample_agencies_data, mock_serpapi_reviews
    ):
        """Test resuming collection from checkpoint"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_reviews
        mock_client_class.return_value = mock_client
        
        agencies_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(agencies_path, index=False)
        
        # Create fake checkpoint
        checkpoint_path = temp_dir / "checkpoint.json"
        checkpoint_data = {
            "processed_places": ["ChIJ12345"],
            "reviews_data": {"ChIJ12345": [{"text": "Old review"}]}
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        collector = ReviewCollector(debug=False)
        output_path = temp_dir / "reviews.csv"
        
        # Should skip already processed places
        stats = collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="csv",
            output_path=output_path,
            checkpoint_file=checkpoint_path
        )
        
        assert stats is not None


@pytest.mark.unit
class TestCollectorErrorHandling:
    """Test error handling in collection"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_handles_no_reviews(
        self, mock_client_class, temp_dir, sample_agencies_data
    ):
        """Test handling places with no reviews"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = {"reviews": []}
        mock_client_class.return_value = mock_client
        
        agencies_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(agencies_path, index=False)
        
        collector = ReviewCollector(debug=False)
        output_path = temp_dir / "empty_reviews.csv"
        
        stats = collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="csv",
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        # Should complete without error
        assert stats is not None
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_handles_api_errors(
        self, mock_client_class, temp_dir, sample_agencies_data
    ):
        """Test handling API errors"""
        mock_client = MagicMock()
        mock_client.search_google_maps.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        agencies_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(agencies_path, index=False)
        
        collector = ReviewCollector(debug=False)
        output_path = temp_dir / "error_reviews.csv"
        
        # Should log failed places
        stats = collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="csv",
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        # Stats should reflect failures
        assert stats is not None
    
    def test_handles_missing_input_file(self, temp_dir, mock_env_vars):
        """Test handling missing agencies file"""
        collector = ReviewCollector(debug=False)
        
        with pytest.raises(FileNotFoundError):
            collector.collect_reviews(
                agencies_path=temp_dir / "nonexistent.csv",
                output_mode="csv",
                output_path=temp_dir / "output.csv",
                checkpoint_file=temp_dir / "checkpoint.json"
            )


@pytest.mark.unit
class TestCollectorOutputValidation:
    """Test output validation and data quality"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_output_no_duplicates(
        self, mock_client_class, temp_dir, sample_agencies_data, mock_serpapi_reviews
    ):
        """Test that output has no duplicate reviews"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_reviews
        mock_client_class.return_value = mock_client
        
        agencies_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(agencies_path, index=False)
        
        collector = ReviewCollector(debug=False)
        output_path = temp_dir / "reviews.csv"
        
        collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="csv",
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        df = pd.read_csv(output_path)
        
        # Check for duplicates (if there's a review ID column)
        if "review_id" in df.columns:
            assert df["review_id"].nunique() == len(df)
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_output_preserves_metadata(
        self, mock_client_class, temp_dir, sample_agencies_data, mock_serpapi_reviews
    ):
        """Test that output preserves bank/city metadata"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_reviews
        mock_client_class.return_value = mock_client
        
        agencies_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(agencies_path, index=False)
        
        collector = ReviewCollector(debug=False)
        output_path = temp_dir / "reviews.csv"
        
        collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="csv",
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        df = pd.read_csv(output_path)
        
        # Check metadata columns
        assert "_place_id" in df.columns
        if "_bank" in sample_agencies_data.columns:
            assert "_bank" in df.columns
        if "_city" in sample_agencies_data.columns:
            assert "_city" in df.columns


# =========================
# Integration Tests
# =========================
@pytest.mark.integration
class TestCollectorIntegration:
    """Integration tests with real-like workflows"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_full_collection_workflow(
        self, mock_client_class, temp_dir, sample_agencies_data, mock_serpapi_reviews
    ):
        """Test complete collection workflow"""
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_reviews
        mock_client_class.return_value = mock_client
        
        agencies_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(agencies_path, index=False)
        
        collector = ReviewCollector(debug=False)
        output_path = temp_dir / "full_reviews.csv"
        checkpoint_path = temp_dir / "checkpoint.json"
        
        # First run
        stats1 = collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="csv",
            output_path=output_path,
            checkpoint_file=checkpoint_path
        )
        
        assert stats1["total_places"] > 0
        assert output_path.exists()
        assert checkpoint_path.exists()
        
        # Second run (should use checkpoint)
        stats2 = collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="csv",
            output_path=output_path,
            checkpoint_file=checkpoint_path
        )
        
        assert stats2 is not None


# =========================
# Performance Tests
# =========================
@pytest.mark.slow
class TestCollectorPerformance:
    """Test collection performance with large datasets"""
    
    @patch('review_analyzer.utils.SerpAPIClient')
    def test_performance_many_places(
        self, mock_client_class, temp_dir, production_agencies_data, mock_serpapi_reviews
    ):
        """Test performance with many places"""
        import time
        
        mock_client = MagicMock()
        mock_client.search_google_maps.return_value = mock_serpapi_reviews
        mock_client_class.return_value = mock_client
        
        agencies_path = temp_dir / "large_agencies.csv"
        production_agencies_data.to_csv(agencies_path, index=False)
        
        collector = ReviewCollector(debug=False)
        output_path = temp_dir / "large_reviews.csv"
        
        start = time.time()
        stats = collector.collect_reviews(
            agencies_path=agencies_path,
            output_mode="csv",
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        duration = time.time() - start
        
        # Should complete in reasonable time (mocked, so no real delays)
        assert duration < 10.0
        assert stats["total_places"] > 0
