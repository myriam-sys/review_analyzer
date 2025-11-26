# Test Suite Documentation

Comprehensive unit and integration tests for the Review Analyzer pipeline.

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures, mock APIs
├── test_config.py        # Config module (40+ tests)
├── test_utils.py         # Utils module (50+ tests)
├── test_discover.py      # Discovery engine (35+ tests)
├── test_collect.py       # Collection module (40+ tests)
└── test_classify.py      # Classification module (45+ tests)
```

**Total: ~210 tests, ~2,900 lines**

---

## Running Tests

### Quick Commands

```bash
# During development (fast, <10s)
pytest tests/ -v -m unit

# Before committing (skip slow tests, ~30s)
pytest tests/ -v -m "not slow"

# Full validation (all tests, ~90s)
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src/review_analyzer --cov-report=html

# Specific module
pytest tests/test_config.py -v
pytest tests/test_utils.py -v
pytest tests/test_discover.py -v
pytest tests/test_collect.py -v
pytest tests/test_classify.py -v

# Specific test class or function
pytest tests/test_config.py::TestConfigCategories -v
pytest tests/test_utils.py::TestSerpAPIClient::test_search_handles_rate_limit -v
```

### Using run_tests.py Helper

```bash
python run_tests.py unit          # Fast unit tests
python run_tests.py fast          # Unit + integration (no slow)
python run_tests.py coverage      # All with coverage
python run_tests.py module utils  # Specific module
```

---

## Test Markers

| Marker | Purpose | Speed | API Calls |
|--------|---------|-------|-----------|
| `@pytest.mark.unit` | Unit tests | <1s each | Mocked |
| `@pytest.mark.integration` | Multi-component tests | 1-5s each | Mocked |
| `@pytest.mark.slow` | Production-scale tests | 5-30s each | Mocked |

---

## Test Coverage by Module

### test_config.py (40+ tests)

Tests configuration loading, categories, settings, and path management.

Key validations:
- Exactly 17 categories defined
- Positive/negative keywords present
- Confidence threshold: 0-1 range
- Map centers for 13 Moroccan cities
- Config import completes in <1 second

### test_utils.py (50+ tests)

Tests SerpAPIClient, CSV/JSON utilities, validation functions, and checkpoints.

Key validations:
- SerpAPI handles rate limiting (429 -> retry -> 200)
- CSV reading handles special characters (accents, apostrophes)
- Haversine distance: Casablanca-Rabat ~ 90km
- Checkpoint save/load with 1000 places in <2s
- JSON write is atomic (temp file -> rename)

### test_discover.py (35+ tests)

Tests discovery engine, multi-city search, deduplication, and output format.

Key validations:
- Discovers branches for single bank/city
- Handles brand name filtering
- Deduplicates places from multiple queries
- Output has required columns: place_id, business, city
- Handles empty results and API errors gracefully

### test_collect.py (40+ tests)

Tests collection modes (CSV, JSON), checkpoints, and error handling.

Key validations:
- Collects reviews in CSV mode
- Supports json-per-city and json-per-business modes
- Checkpoint allows resuming from interruption
- Preserves business/city metadata
- Handles places with no reviews

### test_classify.py (45+ tests)

Tests classification, wide format, confidence thresholds, and checkpoints.

Key validations:
- Classifies single review text
- Batch classification with progress
- Wide format with 17 binary category columns
- Filters low confidence predictions (0.55 threshold)
- Preserves original review columns

---

## Test Data

### Sample Data (Fast Tests)

- Agencies: 3 rows
- Reviews: 3 per agency
- Purpose: Fast unit tests, rapid feedback

### Production Data (Slow Tests)

- Agencies: 100 rows
- Reviews: 500 rows
- Purpose: Performance validation, integration testing

### Parametrized Fixtures

Tests automatically run on both scales:

```python
@pytest.fixture(params=["sample", "production"])
def agencies_data(request, sample_agencies_data, production_agencies_data):
    if request.param == "sample":
        return sample_agencies_data
    return production_agencies_data
```

---

## Writing New Tests

### Template

```python
@pytest.mark.unit
class TestMyNewFeature:
    """Test my new feature"""
    
    def test_basic_functionality(self, temp_dir):
        """Test basic functionality"""
        # Arrange
        input_data = "test input"
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result == expected_output
    
    def test_handles_edge_case(self):
        """Test edge case handling"""
        pass
```

### Using Mocked APIs

```python
@patch('review_analyzer.utils.SerpAPIClient')
def test_with_mocked_api(self, mock_client_class, mock_serpapi_response):
    mock_client = MagicMock()
    mock_client.search_google_maps.return_value = mock_serpapi_response
    mock_client_class.return_value = mock_client
    
    result = my_function_that_calls_api()
    
    assert result is not None
```

---

## CI/CD Integration

```bash
# Fast tests for pull requests
pytest tests/ -v -m "unit" --cov=src/review_analyzer

# All tests for main branch
pytest tests/ -v --cov=src/review_analyzer --cov-report=xml

# With coverage threshold (fail if <80%)
pytest tests/ -v --cov=src/review_analyzer --cov-fail-under=80
```

---

## Performance Targets

| Test Type | Target | Actual |
|-----------|--------|--------|
| Unit tests (all) | <10 seconds | ~5-8 seconds |
| Integration tests | <30 seconds | ~15-25 seconds |
| Slow tests | <60 seconds | ~30-50 seconds |
| Full suite | <120 seconds | ~60-90 seconds |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Run from project root: `cd review_analyzer && pytest tests/` |
| Missing API keys | Tests use mocks; check conftest.py has mock_env_vars fixture |
| Slow tests take too long | Skip with `pytest -m "not slow"` |
| Coverage not generated | Install: `pip install pytest-cov` |

---

## Best Practices

1. Write tests first (TDD) when adding features
2. Run unit tests frequently during development
3. Run all tests before committing
4. Mock external APIs for reliability
5. Use descriptive test names
6. Test edge cases (empty input, special characters)
7. Aim for >80% coverage on critical paths
8. Keep tests independent

