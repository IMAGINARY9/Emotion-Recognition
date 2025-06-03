"""
Unit tests for emotion recognition models.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import (
    DistilBERTEmotionModel,
    TwitterRoBERTaEmotionModel, 
    BiLSTMEmotionModel,
    EnsembleEmotionModel,
    EmotionDataset,
    EmotionPredictor,
    create_model
)

class TestEmotionModels(unittest.TestCase):
    """Test emotion recognition models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 6
        self.vocab_size = 1000
        self.embedding_dim = 128
        self.hidden_dim = 64
        self.max_length = 32
        self.batch_size = 4
        self.device = 'cpu'
    
    def test_distilbert_model_creation(self):
        """Test DistilBERT model creation."""
        model = DistilBERTEmotionModel(
            num_classes=self.num_classes,
            dropout_rate=0.3
        )
        
        self.assertIsInstance(model, DistilBERTEmotionModel)
        self.assertEqual(model.num_classes, self.num_classes)
    
    def test_distilbert_forward_pass(self):
        """Test DistilBERT forward pass."""
        model = DistilBERTEmotionModel(num_classes=self.num_classes)
        
        # Create dummy input
        input_ids = torch.randint(0, 1000, (self.batch_size, self.max_length))
        attention_mask = torch.ones(self.batch_size, self.max_length)
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Forward pass
        outputs = model(inputs)
        
        self.assertEqual(outputs.shape, (self.batch_size, self.num_classes))
    
    def test_bilstm_model_creation(self):
        """Test BiLSTM model creation."""
        model = BiLSTMEmotionModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes
        )
        
        self.assertIsInstance(model, BiLSTMEmotionModel)
        self.assertEqual(model.num_classes, self.num_classes)
        self.assertEqual(model.vocab_size, self.vocab_size)
    
    def test_bilstm_forward_pass(self):
        """Test BiLSTM forward pass."""
        model = BiLSTMEmotionModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes
        )
        
        # Create dummy input
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_length))
        
        inputs = {'input_ids': input_ids}
        
        # Forward pass
        outputs = model(inputs)
        
        self.assertEqual(outputs.shape, (self.batch_size, self.num_classes))
    
    def test_ensemble_model_creation(self):
        """Test ensemble model creation."""
        # Create individual models
        distilbert = DistilBERTEmotionModel(num_classes=self.num_classes)
        bilstm = BiLSTMEmotionModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes
        )
        
        models = [distilbert, bilstm]
        weights = [0.6, 0.4]
        
        ensemble = EnsembleEmotionModel(
            models=models,
            weights=weights,
            num_classes=self.num_classes
        )
        
        self.assertIsInstance(ensemble, EnsembleEmotionModel)
        self.assertEqual(len(ensemble.models), 2)
        self.assertEqual(ensemble.weights, weights)
    
    def test_create_model_factory(self):
        """Test model factory function."""
        # Test DistilBERT creation
        model = create_model(
            model_type="distilbert",
            num_classes=self.num_classes
        )
        self.assertIsInstance(model, DistilBERTEmotionModel)
        
        # Test BiLSTM creation
        model = create_model(
            model_type="bilstm",
            num_classes=self.num_classes,
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        self.assertIsInstance(model, BiLSTMEmotionModel)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            create_model(model_type="invalid", num_classes=self.num_classes)

class TestEmotionDataset(unittest.TestCase):
    """Test emotion dataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.texts = ["I love this!", "This is sad", "I'm angry"]
        self.labels = [2, 0, 3]  # love, sadness, anger
        self.max_length = 32
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = EmotionDataset(
            texts=self.texts,
            labels=self.labels,
            max_length=self.max_length
        )
        
        self.assertEqual(len(dataset), len(self.texts))
    
    def test_dataset_getitem(self):
        """Test dataset item access."""
        dataset = EmotionDataset(
            texts=self.texts,
            labels=self.labels,
            max_length=self.max_length
        )
        
        inputs, label = dataset[0]
        
        self.assertIn('input_ids', inputs)
        self.assertIn('attention_mask', inputs)
        self.assertEqual(label, self.labels[0])

class TestEmotionPredictor(unittest.TestCase):
    """Test emotion predictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        self.device = 'cpu'
    
    @patch('src.models.DistilBERTEmotionModel')
    @patch('src.preprocessing.EmotionPreprocessor')
    def test_predictor_creation(self, mock_preprocessor, mock_model):
        """Test predictor creation."""
        predictor = EmotionPredictor(
            model=mock_model,
            preprocessor=mock_preprocessor,
            emotion_labels=self.emotion_labels,
            device=self.device
        )
        
        self.assertEqual(predictor.emotion_labels, self.emotion_labels)
        self.assertEqual(predictor.device, self.device)
    
    @patch('src.models.DistilBERTEmotionModel')
    @patch('src.preprocessing.EmotionPreprocessor')
    def test_single_prediction(self, mock_preprocessor, mock_model):
        """Test single text prediction."""
        # Mock preprocessor
        mock_preprocessor.preprocess.return_value = "processed text"
        
        # Mock model
        mock_model.eval.return_value = None
        mock_model.return_value = torch.tensor([[0.1, 0.8, 0.05, 0.02, 0.02, 0.01]])
        
        predictor = EmotionPredictor(
            model=mock_model,
            preprocessor=mock_preprocessor,
            emotion_labels=self.emotion_labels,
            device=self.device
        )
        
        # Mock tokenizer
        with patch.object(predictor, '_tokenize') as mock_tokenize:
            mock_tokenize.return_value = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            result = predictor.predict("test text")
            
            self.assertEqual(result, "joy")  # Index 1 has highest probability

if __name__ == '__main__':
    # Run tests
    unittest.main()
