"""
Unit tests for config module
Tests configuration loading, validation, and helper functions
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from review_analyzer import config


# =========================
# Configuration Tests
# =========================
@pytest.mark.unit
class TestConfigLoading:
    """Test configuration loading and environment variables"""
    
    def test_api_keys_loaded_from_env(self, mock_env_vars):
        """Test that API keys are loaded from environment"""
        # Reload config to pick up mock env vars
        import importlib
        importlib.reload(config)
        
        assert config.SERPAPI_API_KEY == "test_serpapi_key_123"
        assert config.OPENAI_API_KEY == "test_openai_key_456"
    
    def test_missing_api_keys_raise_error(self, monkeypatch):
        """Test that missing API keys raise ValueError"""
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="SERPAPI_API_KEY"):
            import importlib
            importlib.reload(config)
    
    def test_project_root_path(self):
        """Test that PROJECT_ROOT is correctly set"""
        assert config.PROJECT_ROOT.exists()
        assert config.PROJECT_ROOT.is_dir()
    
    def test_data_directories_defined(self):
        """Test that all data directories are defined"""
        assert hasattr(config, "DATA_DIR")
        assert hasattr(config, "INPUT_DIR")
        assert hasattr(config, "OUTPUT_DIR")
        assert hasattr(config, "DEFAULT_DATA_DIR")
        assert hasattr(config, "LOGS_DIR")


@pytest.mark.unit
class TestConfigCategories:
    """Test category definitions"""
    
    def test_categories_count(self):
        """Test that we have exactly 17 categories"""
        assert len(config.CATEGORIES) == 17
    
    def test_categories_are_strings(self):
        """Test that all categories are non-empty strings"""
        for category in config.CATEGORIES:
            assert isinstance(category, str)
            assert len(category) > 0
    
    def test_positive_categories_exist(self):
        """Test that positive categories are defined"""
        positive_keywords = [
            "chaleureux", "réactif", "professionnalisme",
            "efficacité", "accessibilité", "satisfaction", "digitale"
        ]
        
        for keyword in positive_keywords:
            assert any(
                keyword.lower() in cat.lower()
                for cat in config.CATEGORIES
            ), f"Missing positive category with '{keyword}'"
    
    def test_negative_categories_exist(self):
        """Test that negative categories are defined"""
        negative_keywords = [
            "attente", "injoignable", "réclamations",
            "incidents", "frais", "insatisfaction", "considération"
        ]
        
        for keyword in negative_keywords:
            assert any(
                keyword.lower() in cat.lower()
                for cat in config.CATEGORIES
            ), f"Missing negative category with '{keyword}'"
    
    def test_neutral_categories_exist(self):
        """Test that neutral/other categories exist"""
        neutral_keywords = ["hors-sujet", "autre"]
        
        for keyword in neutral_keywords:
            assert any(
                keyword.lower() in cat.lower()
                for cat in config.CATEGORIES
            ), f"Missing neutral/other category with '{keyword}'"


@pytest.mark.unit
class TestConfigSettings:
    """Test configuration settings"""
    
    def test_serpapi_config(self):
        """Test SerpAPI configuration"""
        assert "api_key" in config.SERPAPI_CONFIG
        assert "delay_seconds" in config.SERPAPI_CONFIG
        assert "timeout" in config.SERPAPI_CONFIG
        
        assert config.SERPAPI_CONFIG["delay_seconds"] >= 0
        assert config.SERPAPI_CONFIG["timeout"] > 0
    
    def test_openai_config(self):
        """Test OpenAI configuration"""
        assert "api_key" in config.OPENAI_CONFIG
        assert "model" in config.OPENAI_CONFIG
        
        assert len(config.OPENAI_CONFIG["model"]) > 0
    
    def test_processing_config(self):
        """Test processing configuration"""
        assert "batch_size" in config.PROCESSING_CONFIG
        assert "checkpoint_interval" in config.PROCESSING_CONFIG
        
        assert config.PROCESSING_CONFIG["batch_size"] > 0
        assert config.PROCESSING_CONFIG["checkpoint_interval"] > 0
    
    def test_confidence_threshold(self):
        """Test confidence threshold is valid"""
        assert 0 <= config.CONF_THRESHOLD <= 1
    
    def test_validation_settings(self):
        """Test validation settings"""
        assert hasattr(config, "VALID_PLACE_ID_PATTERN")
        assert hasattr(config, "MIN_REVIEWS_PER_PLACE")
        assert hasattr(config, "MAX_RATING")
        assert hasattr(config, "MIN_RATING")
        
        assert config.MAX_RATING == 5
        assert config.MIN_RATING == 1


@pytest.mark.unit
class TestMapCenters:
    """Test default map centers"""
    
    def test_map_centers_defined(self):
        """Test that map centers are defined"""
        assert hasattr(config, "DEFAULT_MAP_CENTERS")
        assert isinstance(config.DEFAULT_MAP_CENTERS, dict)
        assert len(config.DEFAULT_MAP_CENTERS) > 0
    
    def test_major_cities_included(self):
        """Test that major Moroccan cities are included"""
        major_cities = [
            "Casablanca", "Rabat", "Marrakech",
            "Fes", "Tangier", "Agadir"
        ]
        
        for city in major_cities:
            assert city in config.DEFAULT_MAP_CENTERS, \
                f"Missing map center for {city}"
    
    def test_map_center_format(self):
        """Test that map centers have correct format"""
        for city, center in config.DEFAULT_MAP_CENTERS.items():
            assert isinstance(center, str)
            assert center.startswith("@")
            assert "," in center
            # Format: @lat,lon,zoom
            parts = center[1:].split(",")
            assert len(parts) == 3


@pytest.mark.unit
class TestOutputMode:
    """Test OutputMode class"""
    
    def test_output_modes_defined(self):
        """Test that all output modes are defined"""
        assert hasattr(config.OutputMode, "JSON_PER_CITY")
        assert hasattr(config.OutputMode, "JSON_PER_BANK")
        assert hasattr(config.OutputMode, "SINGLE_JSON")
        assert hasattr(config.OutputMode, "CSV")
    
    def test_output_mode_values(self):
        """Test output mode values"""
        assert config.OutputMode.JSON_PER_CITY == "json-per-city"
        assert config.OutputMode.JSON_PER_BANK == "json-per-bank"
        assert config.OutputMode.SINGLE_JSON == "json"
        assert config.OutputMode.CSV == "csv"


# =========================
# Helper Function Tests
# =========================
@pytest.mark.unit
class TestGetOutputPath:
    """Test get_output_path helper function"""
    
    def test_returns_path_object(self):
        """Test that function returns a Path object"""
        result = config.get_output_path("test", prefix="data")
        assert isinstance(result, Path)
    
    def test_includes_timestamp(self):
        """Test that path includes timestamp"""
        result = config.get_output_path("test", prefix="data")
        # Should have format: data_YYYYMMDD_HHMMSS
        assert "_" in result.name
    
    def test_respects_extension(self):
        """Test that extension is applied correctly"""
        result = config.get_output_path("test", extension="csv")
        assert result.suffix == ".csv"
        
        result = config.get_output_path("test", extension="json")
        assert result.suffix == ".json"
    
    def test_custom_output_dir(self, temp_dir):
        """Test custom output directory"""
        result = config.get_output_path(
            "test",
            output_dir=temp_dir
        )
        assert result.parent == temp_dir


@pytest.mark.unit
class TestValidatePlaceId:
    """Test validate_place_id function"""
    
    def test_valid_place_id(self):
        """Test validation of correct place_id format"""
        valid_ids = [
            "ChIJABC123xyz",
            "ChIJ_test-123",
            "ChIJabcdefghijklmnopqrstuvwxyz",
        ]
        
        for place_id in valid_ids:
            assert config.validate_place_id(place_id) is True
    
    def test_invalid_place_id(self):
        """Test rejection of incorrect place_id formats"""
        invalid_ids = [
            "",
            "ChIJ",  # Too short
            "invalid",
            "ChI",  # Wrong prefix
            "ChIJ@invalid",  # Invalid characters
            None,
        ]
        
        for place_id in invalid_ids:
            assert config.validate_place_id(place_id) is False
    
    def test_place_id_with_special_chars(self):
        """Test place_ids with allowed special characters"""
        # Underscore and hyphen are allowed
        assert config.validate_place_id("ChIJtest_123-abc") is True
        
        # Other special chars not allowed
        assert config.validate_place_id("ChIJtest@123") is False
        assert config.validate_place_id("ChIJtest#123") is False


# =========================
# Integration Tests
# =========================
@pytest.mark.unit
class TestConfigIntegration:
    """Test configuration integration"""
    
    def test_all_paths_under_project_root(self):
        """Test that all paths are under PROJECT_ROOT"""
        paths_to_check = [
            config.DATA_DIR,
            config.INPUT_DIR,
            config.OUTPUT_DIR,
            config.LOGS_DIR,
        ]
        
        for path in paths_to_check:
            # Convert to string for comparison
            assert str(config.PROJECT_ROOT) in str(path)
    
    def test_config_can_be_imported_multiple_times(self):
        """Test that config can be safely imported multiple times"""
        import importlib
        
        # First import
        from review_analyzer import config as config1
        
        # Reload
        importlib.reload(config)
        
        # Second import
        from review_analyzer import config as config2
        
        # Should have same values
        assert config1.CONF_THRESHOLD == config2.CONF_THRESHOLD
    
    def test_logging_config_structure(self):
        """Test logging configuration structure"""
        assert "version" in config.LOG_CONFIG
        assert "handlers" in config.LOG_CONFIG
        assert "formatters" in config.LOG_CONFIG
        
        assert "console" in config.LOG_CONFIG["handlers"]
        assert "file" in config.LOG_CONFIG["handlers"]


# =========================
# Edge Cases
# =========================
@pytest.mark.unit
class TestConfigEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_category_list_not_allowed(self):
        """Test that CATEGORIES is not empty"""
        assert len(config.CATEGORIES) > 0
    
    def test_duplicate_categories(self):
        """Test that there are no duplicate categories"""
        assert len(config.CATEGORIES) == len(set(config.CATEGORIES))
    
    def test_confidence_threshold_boundary(self):
        """Test confidence threshold is within valid range"""
        assert 0 <= config.CONF_THRESHOLD <= 1
    
    def test_delay_seconds_non_negative(self):
        """Test that API delays are non-negative"""
        assert config.SERPAPI_CONFIG["delay_seconds"] >= 0
        assert config.OPENAI_CONFIG.get("delay_seconds", 0) >= 0


# =========================
# Performance Tests
# =========================
@pytest.mark.unit
class TestConfigPerformance:
    """Test configuration loading performance"""
    
    def test_config_import_is_fast(self):
        """Test that config loads quickly"""
        import time
        import importlib
        
        start = time.time()
        importlib.reload(config)
        duration = time.time() - start
        
        # Should load in less than 1 second
        assert duration < 1.0
    
    def test_repeated_access_is_fast(self):
        """Test that accessing config values is fast"""
        import time
        
        start = time.time()
        for _ in range(1000):
            _ = config.CATEGORIES
            _ = config.CONF_THRESHOLD
            _ = config.DEFAULT_MAP_CENTERS
        duration = time.time() - start
        
        # 1000 accesses should be very fast
        assert duration < 0.1
