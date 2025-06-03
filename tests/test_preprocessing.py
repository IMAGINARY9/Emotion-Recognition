"""
Unit tests for emotion recognition preprocessing.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing import EmotionPreprocessor, EmotionDataProcessor

class TestEmotionPreprocessor(unittest.TestCase):
    """Test emotion text preprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = EmotionPreprocessor(
            lowercase=True,
            remove_urls=True,
            remove_mentions=True,
            remove_hashtags=False,
            expand_contractions=True,
            emoji_handling="convert"
        )
    
    def test_basic_preprocessing(self):
        """Test basic text preprocessing."""
        text = "Hello World! This is a TEST."
        result = self.preprocessor.preprocess(text)
        
        # Should be lowercase
        self.assertEqual(result, "hello world! this is a test.")
    
    def test_url_removal(self):
        """Test URL removal."""
        text = "Check this out https://example.com and http://test.org"
        result = self.preprocessor.preprocess(text)
        
        self.assertNotIn("https://example.com", result)
        self.assertNotIn("http://test.org", result)
    
    def test_mention_removal(self):
        """Test @mention removal."""
        text = "Hey @user123 and @another_user, how are you?"
        result = self.preprocessor.preprocess(text)
        
        self.assertNotIn("@user123", result)
        self.assertNotIn("@another_user", result)
    
    def test_hashtag_preservation(self):
        """Test hashtag preservation when remove_hashtags=False."""
        text = "This is #awesome and #great!"
        result = self.preprocessor.preprocess(text)
        
        # Should keep hashtags but clean them
        self.assertIn("awesome", result)
        self.assertIn("great", result)
    
    def test_hashtag_removal(self):
        """Test hashtag removal when remove_hashtags=True."""
        preprocessor = EmotionPreprocessor(remove_hashtags=True)
        text = "This is #awesome and #great!"
        result = preprocessor.preprocess(text)
        
        self.assertNotIn("#awesome", result)
        self.assertNotIn("#great", result)
    
    def test_contraction_expansion(self):
        """Test contraction expansion."""
        text = "I don't think it's working. We can't do this."
        result = self.preprocessor.preprocess(text)
        
        self.assertIn("do not", result)
        self.assertIn("it is", result)
        self.assertIn("cannot", result)
    
    def test_emoji_conversion(self):
        """Test emoji to text conversion."""
        text = "I'm so happy 😊 and excited 🎉!"
        result = self.preprocessor.preprocess(text)
        
        # Should contain text representations
        self.assertIn("smiling", result.lower())
    
    def test_emoji_removal(self):
        """Test emoji removal."""
        preprocessor = EmotionPreprocessor(emoji_handling="remove")
        text = "I'm so happy 😊 and excited 🎉!"
        result = preprocessor.preprocess(text)
        
        # Should not contain emojis
        self.assertNotIn("😊", result)
        self.assertNotIn("🎉", result)
    
    def test_emoji_keeping(self):
        """Test emoji keeping."""
        preprocessor = EmotionPreprocessor(emoji_handling="keep")
        text = "I'm so happy 😊!"
        result = preprocessor.preprocess(text)
        
        # Should still contain emojis
        self.assertIn("😊", result)
    
    def test_extra_whitespace_removal(self):
        """Test extra whitespace removal."""
        text = "This   has    too     many      spaces"
        result = self.preprocessor.preprocess(text)
        
        # Should have single spaces
        self.assertNotIn("  ", result)
        self.assertEqual(result, "this has too many spaces")
    
    def test_stopword_removal(self):
        """Test stopword removal."""
        preprocessor = EmotionPreprocessor(
            remove_stopwords=True,
            lowercase=True
        )
        text = "The quick brown fox jumps over the lazy dog"
        result = preprocessor.preprocess(text)
        
        # Common stopwords should be removed
        self.assertNotIn("the", result)
        self.assertNotIn("over", result)
        # Content words should remain
        self.assertIn("quick", result)
        self.assertIn("brown", result)
        self.assertIn("fox", result)
    
    def test_batch_preprocessing(self):
        """Test batch text preprocessing."""
        texts = [
            "Hello @user! Check https://example.com 😊",
            "I don't like this #bad content",
            "This is GREAT and #awesome!!!"
        ]
        
        results = self.preprocessor.preprocess_batch(texts)
        
        self.assertEqual(len(results), len(texts))
        
        # Check that all preprocessing was applied
        for result in results:
            self.assertNotIn("@", result)
            self.assertNotIn("http", result)
            self.assertEqual(result, result.lower())

class TestEmotionDataProcessor(unittest.TestCase):
    """Test emotion data processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = EmotionPreprocessor()
        
        # Mock label encoder
        self.label_encoder = Mock()
        self.label_encoder.transform.return_value = np.array([0, 1, 2])
        
        self.data_processor = EmotionDataProcessor(
            preprocessor=self.preprocessor,
            label_encoder=self.label_encoder,
            max_length=128
        )
    
    def test_dataframe_processing(self):
        """Test DataFrame processing."""
        df = pd.DataFrame({
            'text': ['I love this!', 'This is sad', 'I am angry'],
            'label': ['love', 'sadness', 'anger']
        })
        
        processed_df = self.data_processor.process_dataframe(
            df, 'text', 'label'
        )
        
        self.assertIn('processed_text', processed_df.columns)
        self.assertIn('encoded_label', processed_df.columns)
        self.assertEqual(len(processed_df), len(df))
    
    @patch('src.models.EmotionDataset')
    def test_dataset_creation(self, mock_dataset):
        """Test dataset creation."""
        df = pd.DataFrame({
            'text': ['I love this!', 'This is sad'],
            'label': ['love', 'sadness']
        })
        
        dataset = self.data_processor.create_dataset(
            df, 'text', 'label', model_type='distilbert'
        )
        
        # Should call EmotionDataset constructor
        mock_dataset.assert_called_once()
    
    def test_text_length_filtering(self):
        """Test filtering texts by length."""
        texts = [
            "Short",
            "This is a medium length text for testing",
            "Very " * 100 + "long text"  # Very long text
        ]
        
        filtered_texts = self.data_processor._filter_by_length(
            texts, min_length=5, max_length=50
        )
        
        # Only medium text should remain
        self.assertEqual(len(filtered_texts), 1)
        self.assertIn("medium length", filtered_texts[0])

if __name__ == '__main__':
    # Run tests
    unittest.main()
