"""
Pytest configuration and shared fixtures
"""
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import os
import json
from unittest.mock import MagicMock


# =========================
# Pytest Configuration
# =========================
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no API calls)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may call APIs)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (large datasets)"
    )


# =========================
# Directory Fixtures
# =========================
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def test_data_dir(temp_dir):
    """Create test data directory structure"""
    dirs = {
        "input": temp_dir / "data" / "input",
        "output": temp_dir / "data" / "output",
        "logs": temp_dir / "logs",
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs


# =========================
# Sample Data Fixtures
# =========================
@pytest.fixture
def sample_agencies_data():
    """Small sample of agency data for quick tests"""
    return pd.DataFrame({
        "_place_id": ["ChIJABC123", "ChIJDEF456", "ChIJGHI789"],
        "_bank": ["Attijariwafa Bank", "BMCE Bank", "Attijariwafa Bank"],
        "_city": ["Casablanca", "Rabat", "Casablanca"],
        "title": ["Branch 1", "Branch 2", "Branch 3"],
        "address": ["Address 1", "Address 2", "Address 3"],
        "rating": [4.2, 3.8, 4.5],
        "reviews_count": [150, 80, 200],
    })


@pytest.fixture
def sample_reviews_data():
    """Small sample of review data for quick tests"""
    return pd.DataFrame({
        "place_id": ["ChIJABC123", "ChIJABC123", "ChIJDEF456"],
        "city": ["Casablanca", "Casablanca", "Rabat"],
        "bank": ["Attijariwafa Bank", "Attijariwafa Bank", "BMCE Bank"],
        "review_date": ["2024-01-15", "2024-01-20", "2024-02-01"],
        "review_rating": [5, 4, 2],
        "review_snippet": [
            "Excellent service, très professionnel",
            "Bon accueil mais un peu d'attente",
            "Service très lent, personnel peu aimable"
        ],
        "reviewer_name": ["Ahmed M.", "Fatima K.", "Hassan B."],
    })


@pytest.fixture
def sample_classified_data(sample_reviews_data):
    """Sample data with classifications"""
    df = sample_reviews_data.copy()
    df["sentiment"] = ["Positif", "Positif", "Négatif"]
    df["categories_json"] = [
        '[{"label":"Service client réactif et à l\'écoute","confidence":0.85}]',
        '[{"label":"Attente interminable et lenteur en agence","confidence":0.65}]',
        '[{"label":"Manque de considération ou attitude peu professionnelle","confidence":0.90}]',
    ]
    df["language"] = ["French", "French", "French"]
    df["rationale"] = ["Positive feedback", "Mixed review", "Negative feedback"]
    return df


@pytest.fixture
def production_agencies_data():
    """Larger dataset simulating production scale"""
    banks = ["Attijariwafa Bank", "BMCE Bank", "CIH Bank", "Banque Populaire"]
    cities = ["Casablanca", "Rabat", "Marrakech", "Fes"]
    
    data = []
    for i in range(100):  # 100 agencies
        data.append({
            "_place_id": f"ChIJProd{i:04d}",
            "_bank": banks[i % len(banks)],
            "_city": cities[i % len(cities)],
            "title": f"Branch {i}",
            "address": f"Address {i}",
            "rating": 3.0 + (i % 20) * 0.1,
            "reviews_count": 50 + (i % 50) * 10,
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def production_reviews_data():
    """Larger dataset simulating production scale"""
    place_ids = [f"ChIJProd{i:04d}" for i in range(20)]
    sentiments = ["Positif", "Négatif", "Neutre"]
    
    data = []
    for i in range(500):  # 500 reviews
        data.append({
            "place_id": place_ids[i % len(place_ids)],
            "city": "Casablanca" if i % 2 == 0 else "Rabat",
            "bank": "Attijariwafa Bank" if i % 3 == 0 else "BMCE Bank",
            "review_date": f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}",
            "review_rating": 1 + (i % 5),
            "review_snippet": f"Review text {i}",
            "reviewer_name": f"User {i}",
        })
    
    return pd.DataFrame(data)


# =========================
# API Mock Fixtures
# =========================
@pytest.fixture
def mock_serpapi_response():
    """Mock successful SerpAPI response"""
    return {
        "local_results": [
            {
                "place_id": "ChIJABC123",
                "title": "Attijariwafa Bank - Maarif",
                "address": "123 Rue Test, Casablanca",
                "rating": 4.2,
                "reviews": 150,
                "phone": "+212 5 22 XX XX XX",
            },
            {
                "place_id": "ChIJDEF456",
                "title": "BMCE Bank - Centre Ville",
                "address": "456 Rue Test, Rabat",
                "rating": 3.8,
                "reviews": 80,
            },
        ]
    }


@pytest.fixture
def mock_serpapi_reviews():
    """Mock SerpAPI reviews response"""
    return {
        "reviews": [
            {
                "date": "2024-01-15",
                "rating": 5,
                "snippet": "Excellent service",
                "user": {"name": "Ahmed M."},
            },
            {
                "date": "2024-01-20",
                "rating": 4,
                "snippet": "Bon service",
                "user": {"name": "Fatima K."},
            },
        ]
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI classification response"""
    return {
        "sentiment": "Positif",
        "categories": [
            {
                "label": "Service client réactif et à l'écoute",
                "confidence": 0.85
            }
        ],
        "language": "French",
        "rationale": "The review expresses satisfaction with service quality"
    }


@pytest.fixture
def mock_serpapi_client(monkeypatch, mock_serpapi_response, mock_serpapi_reviews):
    """Mock SerpAPIClient for testing"""
    from review_analyzer import utils
    
    mock_client = MagicMock()
    mock_client.search_google_maps.return_value = mock_serpapi_response
    mock_client.get_reviews.return_value = mock_serpapi_reviews["reviews"]
    
    # Patch the class
    monkeypatch.setattr(utils, "SerpAPIClient", lambda **kwargs: mock_client)
    
    return mock_client


@pytest.fixture
def mock_openai_client(monkeypatch, mock_openai_response):
    """Mock OpenAI client for testing"""
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from review_analyzer import classify
    
    mock_classifier = MagicMock()
    mock_classifier.classify_review.return_value = mock_openai_response
    
    # Patch the class
    monkeypatch.setattr(
        classify,
        "ReviewClassifier",
        lambda **kwargs: mock_classifier
    )
    
    return mock_classifier


# =========================
# Environment Fixtures
# =========================
@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv("SERPAPI_API_KEY", "test_serpapi_key_123")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key_456")


@pytest.fixture
def sample_csv_file(temp_dir, sample_agencies_data):
    """Create a sample CSV file"""
    csv_path = temp_dir / "sample_agencies.csv"
    sample_agencies_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_reviews_csv(temp_dir, sample_reviews_data):
    """Create a sample reviews CSV file"""
    csv_path = temp_dir / "sample_reviews.csv"
    sample_reviews_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_json_file(temp_dir):
    """Create a sample JSON file"""
    import json
    
    json_path = temp_dir / "sample_data.json"
    data = {
        "ChIJABC123": [
            {"date": "2024-01-15", "rating": 5, "snippet": "Great"},
            {"date": "2024-01-20", "rating": 4, "snippet": "Good"},
        ],
        "ChIJDEF456": [
            {"date": "2024-02-01", "rating": 2, "snippet": "Poor"},
        ],
    }
    
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    return json_path


# =========================
# Checkpoint Fixtures
# =========================
@pytest.fixture
def sample_checkpoint_data():
    """Sample checkpoint data"""
    return {
        "last_index": 50,
        "reviews_data": {
            "ChIJABC123": [{"date": "2024-01-15", "rating": 5}]
        },
        "timestamp": "2024-01-20T10:30:00"
    }


@pytest.fixture
def checkpoint_file(temp_dir, sample_checkpoint_data):
    """Create a checkpoint file"""
    import json
    
    checkpoint_path = temp_dir / "checkpoint.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(sample_checkpoint_data, f)
    
    return checkpoint_path


# =========================
# Parametrized Fixtures
# =========================
@pytest.fixture(params=["sample", "production"])
def agencies_data(request, sample_agencies_data, production_agencies_data):
    """Parametrized fixture for both sample and production data"""
    if request.param == "sample":
        return sample_agencies_data
    else:
        return production_agencies_data


@pytest.fixture(params=["sample", "production"])
def reviews_data(request, sample_reviews_data, production_reviews_data):
    """Parametrized fixture for both sample and production reviews"""
    if request.param == "sample":
        return sample_reviews_data
    else:
        return production_reviews_data


# =========================
# Transformers Fixtures
# =========================
@pytest.fixture
def sample_raw_reviews():
    """Raw reviews before normalization (for transform tests)"""
    return pd.DataFrame({
        "date": ["il y a 2 mois", "il y a 3 jours", "il y a 1 an", "il y a 5 heures"],
        "rating": ["4", "5.0", "3", "2"],
        "lat": ["33.5731", "34.0209", "31.6295", "35.7595"],
        "lng": ["-7.5898", "-6.8416", "-7.9811", "-5.8340"],
        "city": ["Casablanca", "Rabat", "Marrakech", "Tanger"],
        "text": ["  Bon service  ", "Excellent!", "  Lent  ", None],
        "business_id": ["B1", "B2", "B1", "B3"],
        "place_id": ["ChIJABC", "ChIJDEF", "ChIJABC", "ChIJGHI"],
    })


@pytest.fixture
def sample_normalized_reviews():
    """Normalized reviews (after transform step 1)"""
    return pd.DataFrame({
        "created_at": pd.to_datetime([
            "2024-01-15", "2024-03-10", "2023-03-15", "2024-03-14"
        ]),
        "rating": pd.array([4, 5, 3, 2], dtype="Int64"),
        "lat": [33.5731, 34.0209, 31.6295, 35.7595],
        "lng": [-7.5898, -6.8416, -7.9811, -5.8340],
        "city": ["Casablanca", "Rabat", "Marrakech", "Tanger"],
        "city_normalized": ["casablanca", "rabat", "marrakech", "tanger"],
        "text": ["Bon service", "Excellent!", "Lent", ""],
        "business_id": ["B1", "B2", "B1", "B3"],
        "place_id": ["ChIJABC", "ChIJDEF", "ChIJABC", "ChIJGHI"],
        "month": pd.to_datetime(["2024-01-01", "2024-03-01", "2023-03-01", "2024-03-01"]),
    })


@pytest.fixture
def sample_reviews_with_region(sample_normalized_reviews):
    """Reviews with region column added"""
    df = sample_normalized_reviews.copy()
    df["region"] = [
        "Casablanca-Settat",
        "Rabat-Salé-Kénitra",
        "Marrakech-Safi",
        "Tanger-Tétouan-Al Hoceïma"
    ]
    return df


@pytest.fixture
def sample_classified_reviews_for_aggregates(sample_reviews_with_region):
    """Classified reviews ready for aggregation"""
    df = sample_reviews_with_region.copy()
    df["sentiment"] = ["positive", "positive", "negative", "negative"]
    df["category"] = [
        "Service client réactif et à l'écoute",
        "Accueil chaleureux et personnel attentionné",
        "Attente interminable et lenteur en agence",
        "Service client injoignable ou non réactif"
    ]
    df["confidence"] = [0.85, 0.90, 0.75, 0.80]
    return df


@pytest.fixture
def mock_regions_geojson(temp_dir):
    """Create a mock regions GeoJSON file"""
    regions_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Casablanca-Settat"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-8.0, 32.5], [-7.0, 32.5],
                        [-7.0, 34.0], [-8.0, 34.0], [-8.0, 32.5]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Rabat-Salé-Kénitra"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-7.0, 33.5], [-6.0, 33.5],
                        [-6.0, 35.0], [-7.0, 35.0], [-7.0, 33.5]
                    ]]
                }
            }
        ]
    }
    
    regions_path = temp_dir / "regions.geojson"
    with open(regions_path, 'w') as f:
        json.dump(regions_data, f)
    
    return regions_path
    """Parametrized fixture for both sample and production data"""
    if request.param == "sample":
        return sample_reviews_data
    else:
        return production_reviews_data
