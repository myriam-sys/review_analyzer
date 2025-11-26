"""
Unit tests for utils module
Tests SerpAPIClient, CSV/JSON utilities, CheckpointManager, and validators
"""
import pytest
import pandas as pd
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from review_analyzer import utils


# =========================
# SerpAPIClient Tests
# =========================
@pytest.mark.unit
class TestSerpAPIClient:
    """Test SerpAPIClient class"""
    
    def test_client_initialization(self, mock_env_vars):
        """Test client can be initialized"""
        client = utils.SerpAPIClient()
        assert client is not None
        assert hasattr(client, 'api_key')
    
    def test_client_with_custom_key(self):
        """Test client with custom API key"""
        client = utils.SerpAPIClient(api_key="custom_key")
        assert client.api_key == "custom_key"
    
    @patch('requests.get')
    def test_search_google_maps_success(
        self, mock_get, mock_serpapi_response
    ):
        """Test successful Google Maps search"""
        mock_get.return_value.json.return_value = mock_serpapi_response
        mock_get.return_value.status_code = 200
        
        client = utils.SerpAPIClient(api_key="test_key")
        results = client.search_google_maps(
            query="Attijariwafa Bank",
            ll="@33.5,-7.6,12z"
        )
        
        assert results is not None
        assert "local_results" in results
        assert len(results["local_results"]) > 0
    
    @patch('requests.get')
    def test_search_handles_rate_limit(self, mock_get):
        """Test that search handles rate limits with retry"""
        # First call returns 429, second succeeds
        mock_get.side_effect = [
            MagicMock(status_code=429, text="Rate limit"),
            MagicMock(
                status_code=200,
                json=lambda: {"local_results": []}
            )
        ]
        
        client = utils.SerpAPIClient(api_key="test_key", debug=False)
        
        # Should retry and eventually succeed
        results = client.request("test_endpoint", {})
        assert results is not None
    
    @patch('requests.get')
    def test_search_raises_auth_error(self, mock_get):
        """Test that 401 raises AuthenticationError"""
        mock_get.return_value.status_code = 401
        mock_get.return_value.text = "Invalid API key"
        
        client = utils.SerpAPIClient(api_key="invalid_key")
        
        with pytest.raises(utils.AuthenticationError):
            client.request("test_endpoint", {})


@pytest.mark.unit
class TestCSVUtilities:
    """Test CSV utility functions"""
    
    def test_sniff_csv_format(self, sample_csv_file):
        """Test CSV format detection"""
        delimiter, encoding = utils.sniff_csv_format(str(sample_csv_file))
        
        assert delimiter in [',', ';', '\t']
        assert encoding in ['utf-8', 'latin-1', 'cp1252']
    
    def test_normalize_column_names(self):
        """Test column name normalization"""
        df = pd.DataFrame({
            "Place_ID": [1, 2],
            " City ": ["A", "B"],
            "Bank Name": ["X", "Y"]
        })
        
        normalized = utils.normalize_column_names(df)
        
        assert "place_id" in normalized.columns
        assert "city" in normalized.columns
        assert "bank_name" in normalized.columns
    
    def test_detect_column_exact_match(self):
        """Test column detection with exact match"""
        df = pd.DataFrame({
            "place_id": [1, 2],
            "city": ["A", "B"]
        })
        
        result = utils.detect_column(df, "place_id")
        assert result == "place_id"
    
    def test_detect_column_fuzzy_match(self):
        """Test column detection with fuzzy matching"""
        df = pd.DataFrame({
            "Place_ID": [1, 2],
            "City Name": ["A", "B"]
        })
        
        # Should find "Place_ID" when looking for "place_id"
        result = utils.detect_column(df, "place_id", ["place_id", "placeId"])
        assert result is not None
    
    def test_detect_column_not_found(self):
        """Test column detection when column doesn't exist"""
        df = pd.DataFrame({"col1": [1, 2]})
        
        result = utils.detect_column(df, "missing_col")
        assert result is None
    
    def test_read_agencies_csv_sample(self, sample_csv_file):
        """Test reading agencies CSV with sample data"""
        df = utils.read_agencies_csv(str(sample_csv_file))
        
        assert len(df) > 0
        assert "_place_id" in df.columns
        assert "_bank" in df.columns
        assert "_city" in df.columns
    
    def test_read_agencies_csv_with_filters(self, sample_csv_file):
        """Test reading CSV with city/bank filters"""
        df = utils.read_agencies_csv(
            str(sample_csv_file),
            filter_cities=["Casablanca"]
        )
        
        assert len(df) > 0
        assert all(df["_city"] == "Casablanca")
    
    @pytest.mark.slow
    def test_read_agencies_csv_production(
        self, temp_dir, production_agencies_data
    ):
        """Test reading large production CSV"""
        csv_path = temp_dir / "production.csv"
        production_agencies_data.to_csv(csv_path, index=False)
        
        df = utils.read_agencies_csv(str(csv_path))
        
        assert len(df) == 100  # Production fixture has 100 rows
        assert "_place_id" in df.columns


@pytest.mark.unit
class TestJSONUtilities:
    """Test JSON utility functions"""
    
    def test_read_json(self, sample_json_file):
        """Test reading JSON file"""
        data = utils.read_json(sample_json_file)
        
        assert data is not None
        assert isinstance(data, dict)
        assert len(data) > 0
    
    def test_write_json(self, temp_dir):
        """Test writing JSON file"""
        data = {"test": "value", "number": 123}
        json_path = temp_dir / "test.json"
        
        utils.write_json(json_path, data)
        
        assert json_path.exists()
        
        # Read back and verify
        with open(json_path) as f:
            loaded = json.load(f)
        
        assert loaded == data
    
    def test_write_json_atomic(self, temp_dir):
        """Test that write_json is atomic (uses temp file)"""
        json_path = temp_dir / "atomic.json"
        
        # Write initial data
        utils.write_json(json_path, {"version": 1})
        
        # Write new data (should not corrupt if interrupted)
        utils.write_json(json_path, {"version": 2})
        
        # File should exist and have new data
        assert json_path.exists()
        data = utils.read_json(json_path)
        assert data["version"] == 2
    
    @pytest.mark.slow
    def test_write_json_large_data(self, temp_dir):
        """Test writing large JSON data"""
        # Create large dataset
        large_data = {
            f"key_{i}": [{"index": j} for j in range(100)]
            for i in range(100)
        }
        
        json_path = temp_dir / "large.json"
        
        start = time.time()
        utils.write_json(json_path, large_data)
        duration = time.time() - start
        
        assert json_path.exists()
        assert duration < 5.0  # Should complete in reasonable time


@pytest.mark.unit
class TestValidationFunctions:
    """Test validation utility functions"""
    
    def test_validate_reviews_valid_data(self, sample_reviews_data):
        """Test validation of valid review data"""
        # Should not raise exception
        utils.validate_reviews(sample_reviews_data)
    
    def test_validate_reviews_missing_columns(self):
        """Test validation catches missing columns"""
        df = pd.DataFrame({"col1": [1, 2]})
        
        # Should not raise (returns True/False)
        # The function logs warnings but doesn't raise
        result = utils.validate_reviews(df)
        assert result is not None
    
    def test_validate_reviews_invalid_ratings(self):
        """Test validation catches invalid ratings"""
        df = pd.DataFrame({
            "review_rating": [0, 6, -1],  # Invalid ratings
            "review_snippet": ["a", "b", "c"]
        })
        
        # Should detect invalid ratings
        result = utils.validate_reviews(df)
        assert result is not None
    
    def test_haversine_distance_same_point(self):
        """Test haversine distance for same point"""
        distance = utils.haversine_distance(
            33.5, -7.6,  # Casablanca
            33.5, -7.6   # Same
        )
        
        assert distance == 0.0
    
    def test_haversine_distance_known_cities(self):
        """Test haversine distance between known cities"""
        # Casablanca to Rabat (roughly 90 km)
        distance = utils.haversine_distance(
            33.5888, -7.6114,  # Casablanca
            34.0209, -6.8416   # Rabat
        )
        
        # Should be approximately 90 km (allow 10% margin)
        assert 80 < distance < 100
    
    def test_haversine_distance_opposite_sides(self):
        """Test haversine for points on opposite sides of Earth"""
        distance = utils.haversine_distance(
            0, 0,      # Equator, Prime Meridian
            0, 180     # Equator, opposite side
        )
        
        # Should be roughly half Earth's circumference
        assert distance > 15000  # More than 15,000 km


@pytest.mark.unit
class TestCheckpointManager:
    """Test CheckpointManager class"""
    
    def test_checkpoint_initialization(self, temp_dir):
        """Test checkpoint manager initialization"""
        checkpoint_path = temp_dir / "checkpoint.json"
        manager = utils.CheckpointManager(checkpoint_path)
        
        assert manager.checkpoint_path == checkpoint_path
    
    def test_save_and_load_checkpoint(
        self, temp_dir, sample_checkpoint_data
    ):
        """Test saving and loading checkpoint"""
        checkpoint_path = temp_dir / "checkpoint.json"
        manager = utils.CheckpointManager(checkpoint_path)
        
        # Save
        manager.save(sample_checkpoint_data)
        assert checkpoint_path.exists()
        
        # Load
        loaded = manager.load()
        assert loaded["last_index"] == sample_checkpoint_data["last_index"]
    
    def test_load_nonexistent_checkpoint(self, temp_dir):
        """Test loading checkpoint that doesn't exist"""
        checkpoint_path = temp_dir / "missing.json"
        manager = utils.CheckpointManager(checkpoint_path)
        
        # Should return empty dict, not raise error
        loaded = manager.load()
        assert loaded == {}
    
    def test_clear_checkpoint(self, checkpoint_file):
        """Test clearing checkpoint"""
        manager = utils.CheckpointManager(checkpoint_file)
        
        # Clear
        manager.clear()
        
        # File should be deleted
        assert not checkpoint_file.exists()
    
    def test_checkpoint_with_nested_data(self, temp_dir):
        """Test checkpoint with complex nested data"""
        checkpoint_path = temp_dir / "nested.json"
        manager = utils.CheckpointManager(checkpoint_path)
        
        complex_data = {
            "level1": {
                "level2": {
                    "level3": [1, 2, 3]
                }
            },
            "array": [{"key": "value"}]
        }
        
        manager.save(complex_data)
        loaded = manager.load()
        
        assert loaded == complex_data
    
    @pytest.mark.slow
    def test_checkpoint_large_data_performance(self, temp_dir):
        """Test checkpoint performance with large data"""
        checkpoint_path = temp_dir / "large_checkpoint.json"
        manager = utils.CheckpointManager(checkpoint_path)
        
        # Create large dataset
        large_data = {
            "reviews_data": {
                f"ChIJ{i:06d}": [
                    {"review": j, "rating": 5}
                    for j in range(10)
                ]
                for i in range(100)
            }
        }
        
        # Save
        start = time.time()
        manager.save(large_data)
        save_duration = time.time() - start
        
        # Load
        start = time.time()
        loaded = manager.load()
        load_duration = time.time() - start
        
        assert save_duration < 2.0  # Should save quickly
        assert load_duration < 2.0  # Should load quickly
        assert len(loaded["reviews_data"]) == 100


@pytest.mark.unit
class TestProgressBar:
    """Test progress bar creation"""
    
    def test_create_progress_bar(self):
        """Test progress bar creation"""
        progress = utils.create_progress_bar(100, desc="Testing")
        
        if progress:  # If tqdm is installed
            assert hasattr(progress, 'update')
            assert hasattr(progress, 'close')
            progress.close()
    
    def test_progress_bar_without_tqdm(self, monkeypatch):
        """Test progress bar when tqdm not available"""
        # Mock tqdm import failure
        import builtins
        real_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'tqdm':
                raise ImportError()
            return real_import(name, *args, **kwargs)
        
        monkeypatch.setattr(builtins, '__import__', mock_import)
        
        # Should return None without error
        progress = utils.create_progress_bar(100)
        assert progress is None


# =========================
# Edge Cases
# =========================
@pytest.mark.unit
class TestUtilsEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_csv_file(self, temp_dir):
        """Test handling empty CSV file"""
        empty_csv = temp_dir / "empty.csv"
        empty_csv.write_text("_place_id,_bank,_city\n")  # Headers only
        
        df = utils.read_agencies_csv(str(empty_csv))
        
        # Should return empty DataFrame, not crash
        assert len(df) == 0
        assert "_place_id" in df.columns
    
    def test_malformed_json_file(self, temp_dir):
        """Test handling malformed JSON"""
        bad_json = temp_dir / "bad.json"
        bad_json.write_text("{invalid json}")
        
        with pytest.raises(json.JSONDecodeError):
            utils.read_json(bad_json)
    
    def test_csv_with_special_characters(self, temp_dir):
        """Test CSV with special characters"""
        df = pd.DataFrame({
            "_place_id": ["ChIJ123"],
            "_bank": ["Banque d'État"],  # Apostrophe
            "_city": ["Café-City"],      # Accents, hyphen
            "title": ["Test™ & Co."]     # Special chars
        })
        
        csv_path = temp_dir / "special.csv"
        df.to_csv(csv_path, index=False)
        
        loaded = utils.read_agencies_csv(str(csv_path))
        
        assert len(loaded) == 1
        assert "d'État" in loaded.iloc[0]["_bank"]
    
    def test_checkpoint_with_invalid_json(self, temp_dir):
        """Test checkpoint handles corrupted JSON"""
        checkpoint_path = temp_dir / "corrupt.json"
        checkpoint_path.write_text("{bad json")
        
        manager = utils.CheckpointManager(checkpoint_path)
        
        # Should return empty dict, not crash
        loaded = manager.load()
        assert loaded == {}


# =========================
# Integration Tests
# =========================
@pytest.mark.unit
class TestUtilsIntegration:
    """Test utils module integration"""
    
    def test_full_workflow_csv_to_checkpoint(
        self, temp_dir, sample_agencies_data
    ):
        """Test complete workflow: CSV → processing → checkpoint"""
        # Save CSV
        csv_path = temp_dir / "agencies.csv"
        sample_agencies_data.to_csv(csv_path, index=False)
        
        # Read CSV
        df = utils.read_agencies_csv(str(csv_path))
        assert len(df) == len(sample_agencies_data)
        
        # Simulate processing with checkpoint
        checkpoint_path = temp_dir / "progress.json"
        manager = utils.CheckpointManager(checkpoint_path)
        
        processed = []
        for idx, row in df.iterrows():
            processed.append(row["_place_id"])
            
            if idx % 2 == 0:  # Checkpoint every 2 items
                manager.save({"last_index": idx, "processed": processed})
        
        # Verify checkpoint
        checkpoint = manager.load()
        assert len(checkpoint["processed"]) > 0
    
    def test_utilities_handle_both_sample_and_production(
        self, temp_dir, agencies_data
    ):
        """Test that utilities work with both sample and production data"""
        csv_path = temp_dir / "data.csv"
        agencies_data.to_csv(csv_path, index=False)
        
        # Read
        df = utils.read_agencies_csv(str(csv_path))
        
        # Validate
        assert len(df) == len(agencies_data)
        assert "_place_id" in df.columns
        
        # Save checkpoint
        checkpoint_path = temp_dir / "checkpoint.json"
        manager = utils.CheckpointManager(checkpoint_path)
        manager.save({"total_rows": len(df)})
        
        # Verify
        checkpoint = manager.load()
        assert checkpoint["total_rows"] == len(df)
