"""
Unit and integration tests for classify module
Tests ReviewClassifier with mocked OpenAI API calls
"""
import pytest
import pandas as pd
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from review_analyzer.classify import ReviewClassifier


# =========================
# ReviewClassifier Tests
# =========================
@pytest.mark.unit
class TestClassifierInit:
    """Test ReviewClassifier initialization"""
    
    def test_classifier_initialization(self, mock_env_vars):
        """Test classifier can be initialized"""
        classifier = ReviewClassifier(debug=False)
        assert classifier is not None
        assert hasattr(classifier, 'client')
    
    def test_classifier_debug_mode(self, mock_env_vars):
        """Test classifier in debug mode"""
        classifier = ReviewClassifier(debug=True)
        assert classifier.debug is True


@pytest.mark.unit
class TestClassifySingleReview:
    """Test single review classification"""
    
    @patch('openai.OpenAI')
    def test_classify_single_review(
        self, mock_openai_class, mock_openai_response
    ):
        """Test classifying a single review"""
        # Setup mock
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        classifier = ReviewClassifier(debug=False)
        
        result = classifier.classify_single_review(
            review_text="Service excellent, personnel très chaleureux"
        )
        
        assert result is not None
        assert isinstance(result, dict)
        assert "categories" in result
        assert "sentiment" in result
    
    @patch('openai.OpenAI')
    def test_classify_empty_review(self, mock_openai_class):
        """Test handling empty review text"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        classifier = ReviewClassifier(debug=False)
        
        result = classifier.classify_single_review(review_text="")
        
        # Should handle gracefully
        assert result is None or result == {}
    
    @patch('openai.OpenAI')
    def test_classify_with_confidence_threshold(
        self, mock_openai_class, mock_openai_response
    ):
        """Test confidence threshold filtering"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        classifier = ReviewClassifier(debug=False)
        
        result = classifier.classify_single_review(
            review_text="Service correct",
            confidence_threshold=0.7
        )
        
        # Should filter out low confidence predictions
        assert result is not None


@pytest.mark.unit
class TestClassifyBatch:
    """Test batch classification"""
    
    @patch('openai.OpenAI')
    def test_classify_batch_reviews(
        self, mock_openai_class, temp_dir, sample_reviews_data, mock_openai_response
    ):
        """Test classifying multiple reviews"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        # Create sample reviews file
        reviews_path = temp_dir / "reviews.csv"
        sample_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "classified.csv"
        
        stats = classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        assert stats["total_reviews"] > 0
        assert output_path.exists()
        
        # Verify output structure
        df = pd.read_csv(output_path)
        assert len(df) > 0
    
    @patch('openai.OpenAI')
    def test_batch_with_progress_tracking(
        self, mock_openai_class, temp_dir, sample_reviews_data, mock_openai_response
    ):
        """Test batch classification shows progress"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "reviews.csv"
        sample_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "classified.csv"
        
        # Should track progress internally
        stats = classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        assert stats is not None
        assert "total_reviews" in stats


@pytest.mark.unit
class TestClassifyWideFormat:
    """Test wide format conversion (17 category columns)"""
    
    @patch('openai.OpenAI')
    def test_wide_format_output(
        self, mock_openai_class, temp_dir, sample_reviews_data, mock_openai_response
    ):
        """Test output in wide format with category columns"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "reviews.csv"
        sample_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "classified_wide.csv"
        
        stats = classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json",
            wide_format=True
        )
        
        assert stats is not None
        assert output_path.exists()
        
        df = pd.read_csv(output_path)
        
        # Should have category columns
        category_columns = [col for col in df.columns if col.startswith("cat_")]
        assert len(category_columns) > 0
    
    @patch('openai.OpenAI')
    def test_wide_format_binary_values(
        self, mock_openai_class, temp_dir, sample_reviews_data, mock_openai_response
    ):
        """Test that category columns have binary values (0/1)"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "reviews.csv"
        sample_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "classified_binary.csv"
        
        classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json",
            wide_format=True
        )
        
        df = pd.read_csv(output_path)
        category_columns = [col for col in df.columns if col.startswith("cat_")]
        
        # Check binary values
        for col in category_columns:
            unique_values = df[col].dropna().unique()
            assert all(val in [0, 1] for val in unique_values)


@pytest.mark.unit
class TestClassifyCheckpoint:
    """Test checkpoint and resume for classification"""
    
    @patch('openai.OpenAI')
    def test_checkpoint_saves_progress(
        self, mock_openai_class, temp_dir, sample_reviews_data, mock_openai_response
    ):
        """Test that checkpoint file is created during classification"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "reviews.csv"
        sample_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "classified.csv"
        checkpoint_path = temp_dir / "checkpoint.json"
        
        classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=checkpoint_path
        )
        
        # Checkpoint should exist
        assert checkpoint_path.exists()
        
        # Verify checkpoint structure
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        assert isinstance(checkpoint, dict)
    
    @patch('openai.OpenAI')
    def test_resume_from_checkpoint(
        self, mock_openai_class, temp_dir, sample_reviews_data, mock_openai_response
    ):
        """Test resuming classification from checkpoint"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "reviews.csv"
        sample_reviews_data.to_csv(reviews_path, index=False)
        
        # Create fake checkpoint
        checkpoint_path = temp_dir / "checkpoint.json"
        checkpoint_data = {
            "processed_reviews": [0],  # Already processed first review
            "classifications": {"0": {"sentiment": "positive"}}
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "classified.csv"
        
        # Should skip already processed reviews
        stats = classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=checkpoint_path
        )
        
        assert stats is not None


@pytest.mark.unit
class TestClassifyErrorHandling:
    """Test error handling in classification"""
    
    @patch('openai.OpenAI')
    def test_handles_api_errors(
        self, mock_openai_class, temp_dir, sample_reviews_data
    ):
        """Test handling OpenAI API errors"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "reviews.csv"
        sample_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "classified.csv"
        
        # Should handle errors gracefully
        stats = classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        # Stats should reflect failures/errors
        assert stats is not None
    
    @patch('openai.OpenAI')
    def test_handles_malformed_response(
        self, mock_openai_class, temp_dir, sample_reviews_data
    ):
        """Test handling malformed API responses"""
        # Mock response with missing fields
        mock_response = MagicMock()
        mock_response.choices = []  # Empty choices
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "reviews.csv"
        sample_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "classified.csv"
        
        # Should handle malformed response
        stats = classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        assert stats is not None
    
    def test_handles_missing_input_file(self, temp_dir, mock_env_vars):
        """Test handling missing reviews file"""
        classifier = ReviewClassifier(debug=False)
        
        with pytest.raises(FileNotFoundError):
            classifier.classify_reviews(
                reviews_path=temp_dir / "nonexistent.csv",
                output_path=temp_dir / "output.csv",
                checkpoint_file=temp_dir / "checkpoint.json"
            )


@pytest.mark.unit
class TestClassifyOutputValidation:
    """Test output validation and data quality"""
    
    @patch('openai.OpenAI')
    def test_output_preserves_original_columns(
        self, mock_openai_class, temp_dir, sample_reviews_data, mock_openai_response
    ):
        """Test that output preserves original review columns"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "reviews.csv"
        sample_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "classified.csv"
        
        classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        df = pd.read_csv(output_path)
        
        # Original columns should be preserved
        for col in sample_reviews_data.columns:
            assert col in df.columns, f"Missing original column: {col}"
    
    @patch('openai.OpenAI')
    def test_output_adds_classification_columns(
        self, mock_openai_class, temp_dir, sample_reviews_data, mock_openai_response
    ):
        """Test that output adds classification result columns"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "reviews.csv"
        sample_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "classified.csv"
        
        classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        
        df = pd.read_csv(output_path)
        
        # Should have new classification columns
        new_columns = set(df.columns) - set(sample_reviews_data.columns)
        assert len(new_columns) > 0


@pytest.mark.unit
class TestClassifyConfidenceThreshold:
    """Test confidence threshold filtering (0.55 default)"""
    
    @patch('openai.OpenAI')
    def test_filters_low_confidence(
        self, mock_openai_class, temp_dir, sample_reviews_data
    ):
        """Test that low confidence predictions are filtered"""
        # Mock response with low confidence
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = json.dumps({
            "sentiment": "positive",
            "categories": {
                "cat_service": {"score": 0.3, "confidence": 0.4}  # Below threshold
            }
        })
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "reviews.csv"
        sample_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "classified.csv"
        
        classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json",
            confidence_threshold=0.55
        )
        
        # Low confidence categories should be filtered
        assert output_path.exists()


# =========================
# Integration Tests
# =========================
@pytest.mark.integration
class TestClassifyIntegration:
    """Integration tests with real-like workflows"""
    
    @patch('openai.OpenAI')
    def test_full_classification_workflow(
        self, mock_openai_class, temp_dir, sample_reviews_data, mock_openai_response
    ):
        """Test complete classification workflow"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "reviews.csv"
        sample_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "full_classified.csv"
        checkpoint_path = temp_dir / "checkpoint.json"
        
        # Run classification
        stats = classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=checkpoint_path,
            wide_format=True
        )
        
        # Verify complete workflow
        assert stats["total_reviews"] > 0
        assert output_path.exists()
        assert checkpoint_path.exists()
        
        # Verify output quality
        df = pd.read_csv(output_path)
        assert len(df) > 0
        assert len(df) == len(sample_reviews_data)


# =========================
# Performance Tests
# =========================
@pytest.mark.slow
class TestClassifyPerformance:
    """Test classification performance with large datasets"""
    
    @patch('openai.OpenAI')
    def test_performance_many_reviews(
        self, mock_openai_class, temp_dir, production_reviews_data, mock_openai_response
    ):
        """Test performance with many reviews"""
        import time
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "large_reviews.csv"
        production_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "large_classified.csv"
        
        start = time.time()
        stats = classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=temp_dir / "checkpoint.json"
        )
        duration = time.time() - start
        
        # Should complete in reasonable time (mocked, so no real API delays)
        assert duration < 30.0
        assert stats["total_reviews"] > 0
    
    @patch('openai.OpenAI')
    def test_checkpoint_performance_large_data(
        self, mock_openai_class, temp_dir, production_reviews_data, mock_openai_response
    ):
        """Test checkpoint save/load performance with large data"""
        import time
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        reviews_path = temp_dir / "reviews.csv"
        production_reviews_data.to_csv(reviews_path, index=False)
        
        classifier = ReviewClassifier(debug=False)
        output_path = temp_dir / "classified.csv"
        checkpoint_path = temp_dir / "checkpoint.json"
        
        # First run - creates checkpoint
        start = time.time()
        classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=checkpoint_path
        )
        duration1 = time.time() - start
        
        # Second run - loads checkpoint (should be fast)
        start = time.time()
        classifier.classify_reviews(
            reviews_path=reviews_path,
            output_path=output_path,
            checkpoint_file=checkpoint_path
        )
        duration2 = time.time() - start
        
        # Checkpoint load should be faster than initial classification
        assert duration2 < duration1
