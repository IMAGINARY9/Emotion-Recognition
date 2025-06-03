"""
Unit tests for emotion recognition data utilities.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_utils import EmotionDataLoader, setup_data_directories, download_sample_data

class TestEmotionDataLoader(unittest.TestCase):
    """Test emotion data loader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_loader = EmotionDataLoader(self.temp_dir)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'text': [
                'I love this so much!',
                'This makes me really angry',
                'I feel so sad today',
                'What a wonderful surprise!',
                'I am scared of heights',
                'This brings me joy'
            ],
            'label': ['love', 'anger', 'sadness', 'surprise', 'fear', 'joy']
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_emotion_labels(self):
        """Test emotion labels initialization."""
        expected_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.assertEqual(self.data_loader.emotion_labels, expected_labels)
    
    def test_csv_loading(self):
        """Test CSV file loading."""
        # Save sample data
        csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.sample_data.to_csv(csv_path, index=False)
        
        # Load data
        loaded_df = self.data_loader.load_csv_data(csv_path)
        
        self.assertEqual(len(loaded_df), len(self.sample_data))
        self.assertListEqual(list(loaded_df.columns), list(self.sample_data.columns))
    
    def test_data_validation(self):
        """Test data validation."""
        # Test with valid data
        validated_df = self.data_loader.validate_data(self.sample_data)
        self.assertEqual(len(validated_df), len(self.sample_data))
        
        # Test with invalid labels
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'label'] = 'invalid_emotion'
        
        validated_df = self.data_loader.validate_data(invalid_data)
        self.assertEqual(len(validated_df), len(self.sample_data) - 1)
        
        # Test with empty text
        empty_text_data = self.sample_data.copy()
        empty_text_data.loc[0, 'text'] = ''
        
        validated_df = self.data_loader.validate_data(empty_text_data)
        self.assertEqual(len(validated_df), len(self.sample_data) - 1)
    
    def test_label_encoding(self):
        """Test label encoding and decoding."""
        labels = ['joy', 'anger', 'sadness']
        
        # Encode labels
        encoded = self.data_loader.encode_labels(labels)
        self.assertEqual(len(encoded), len(labels))
        self.assertTrue(all(isinstance(x, np.integer) for x in encoded))
        
        # Decode labels
        decoded = self.data_loader.decode_labels(encoded)
        self.assertEqual(decoded, labels)
    
    def test_data_splitting(self):
        """Test data splitting functionality."""
        train_df, val_df, test_df = self.data_loader.split_data(
            self.sample_data,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            random_state=42
        )
        
        total_samples = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_samples, len(self.sample_data))
        
        # Check approximate proportions
        self.assertAlmostEqual(len(train_df) / total_samples, 0.6, delta=0.1)
        self.assertAlmostEqual(len(val_df) / total_samples, 0.2, delta=0.1)
        self.assertAlmostEqual(len(test_df) / total_samples, 0.2, delta=0.1)
    
    def test_save_splits(self):
        """Test saving data splits."""
        train_df, val_df, test_df = self.data_loader.split_data(self.sample_data)
        
        file_paths = self.data_loader.save_splits(
            train_df, val_df, test_df, self.temp_dir
        )
        
        # Check that files were created
        for split_name, file_path in file_paths.items():
            self.assertTrue(os.path.exists(file_path))
            
            # Load and verify
            loaded_df = pd.read_csv(file_path)
            self.assertIn('text', loaded_df.columns)
            self.assertIn('label', loaded_df.columns)
    
    def test_load_splits(self):
        """Test loading existing data splits."""
        # First save some splits
        train_df, val_df, test_df = self.data_loader.split_data(self.sample_data)
        self.data_loader.save_splits(train_df, val_df, test_df, self.temp_dir)
        
        # Then load them
        splits = self.data_loader.load_splits(self.temp_dir)
        
        self.assertIn('train', splits)
        self.assertIn('validation', splits)
        self.assertIn('test', splits)
        
        # Verify data integrity
        total_loaded = len(splits['train']) + len(splits['validation']) + len(splits['test'])
        self.assertEqual(total_loaded, len(self.sample_data))
    
    def test_class_weights_calculation(self):
        """Test class weights calculation."""
        # Create imbalanced data
        imbalanced_labels = ['joy'] * 10 + ['anger'] * 2 + ['sadness'] * 1
        
        weights = self.data_loader.get_class_weights(imbalanced_labels)
        
        # Joy should have lower weight (more frequent)
        # Sadness should have higher weight (less frequent)
        joy_idx = self.data_loader.emotion_labels.index('joy')
        sadness_idx = self.data_loader.emotion_labels.index('sadness')
        
        self.assertLess(weights[joy_idx], weights[sadness_idx])

class TestDataUtilities(unittest.TestCase):
    """Test data utility functions."""
    
    def test_setup_data_directories(self):
        """Test data directory setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directories = setup_data_directories(temp_dir)
            
            expected_dirs = ['raw', 'processed', 'splits', 'external']
            
            for dir_name in expected_dirs:
                self.assertIn(dir_name, directories)
                self.assertTrue(os.path.exists(directories[dir_name]))
    
    def test_download_sample_data(self):
        """Test sample data generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, 'raw')
            
            file_path = download_sample_data(output_dir)
            
            # Check that file was created
            self.assertTrue(os.path.exists(file_path))
            
            # Load and validate sample data
            df = pd.read_csv(file_path)
            self.assertIn('text', df.columns)
            self.assertIn('label', df.columns)
            self.assertGreater(len(df), 0)
            
            # Check that all emotion labels are present
            emotion_labels = set(df['label'].unique())
            expected_labels = {'sadness', 'joy', 'love', 'anger', 'fear', 'surprise'}
            self.assertTrue(expected_labels.issubset(emotion_labels))

if __name__ == '__main__':
    # Run tests
    unittest.main()
