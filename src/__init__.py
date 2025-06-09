"""
Emotion Recognition Project

A comprehensive emotion recognition system for text classification using
transformer models and traditional ML approaches.

This package provides tools for:
- Text preprocessing specifically for emotion recognition
- Multiple model architectures (DistilBERT, Twitter RoBERTa, BiLSTM, Ensemble)
- Training and evaluation pipelines
- Configuration management
- Data utilities for emotion datasets

Author: Generated for Emotion Recognition Project
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Emotion Recognition Project"

# Core modules
from .models import (
    EmotionDataset,
    DistilBERTEmotionModel,
    TwitterRoBERTaEmotionModel,
    BiLSTMEmotionModel,
    EnsembleEmotionModel,
    EmotionPredictor,
    create_model
)

from .preprocessing import (
    EmotionPreprocessor,
    EmotionDataProcessor
)

from .training import EmotionTrainer

from .evaluation import EmotionEvaluator

from utils.data_utils import (
    EmotionDataLoader,
    setup_data_directories,
    download_sample_data
)

from .config import (
    ConfigManager,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    PreprocessingConfig,
    EvaluationConfig,
    PathsConfig,
    ExperimentConfig,
    setup_logging,
    get_device_config
)

# Package metadata
SUPPORTED_MODELS = [
    "distilbert",
    "twitter_roberta", 
    "bilstm",
    "ensemble"
]

EMOTION_LABELS = [
    "sadness",
    "joy", 
    "love",
    "anger",
    "fear",
    "surprise"
]

# Default configurations
DEFAULT_CONFIG = {
    "model": {
        "type": "distilbert",
        "num_classes": 6,
        "dropout_rate": 0.3,
        "max_length": 128
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "warmup_steps": 500,
        "weight_decay": 0.01
    },
    "data": {
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1
    },
    "preprocessing": {
        "lowercase": True,
        "remove_urls": True,
        "emoji_handling": "convert"
    }
}

def get_version():
    """Get package version."""
    return __version__

def get_supported_models():
    """Get list of supported model types."""
    return SUPPORTED_MODELS.copy()

def get_emotion_labels():
    """Get list of emotion labels."""
    return EMOTION_LABELS.copy()

def quick_predict(text, model_path, device='auto'):
    """
    Quick prediction function for single text.
    
    Args:
        text: Input text to classify
        model_path: Path to trained model
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        Predicted emotion label
    """
    import torch
    from pathlib import Path
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Load model and config
    model_path = Path(model_path)
    config_manager = ConfigManager()
    
    if (model_path / "config.yaml").exists():
        config = config_manager.load_config(model_path / "config.yaml")
    else:
        raise FileNotFoundError("Configuration file not found in model directory")
    
    # Load model
    model = create_model(
        model_type=config.model.type,
        num_classes=config.model.num_classes,
        model_name=config.model.get('model_name'),
        **config.model
    )
    
    # Load checkpoint
    checkpoint_path = model_path / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_path / "final_model.pt"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Create preprocessor and predictor
    preprocessor = EmotionPreprocessor(
        **config.preprocessing
    )
    
    predictor = EmotionPredictor(
        model=model,
        preprocessor=preprocessor,
        emotion_labels=config.emotions,
        device=device,
        max_length=config.model.max_length,
        model_type=config.model.type,
        model_name=config.model.get('model_name')
    )
    
    # Make prediction
    result = predictor.predict(text)
    
    return result

# Import error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def check_dependencies():
    """
    Check if required dependencies are available.
    
    Returns:
        Dictionary with dependency status
    """
    deps = {
        'torch': TORCH_AVAILABLE,
        'transformers': TRANSFORMERS_AVAILABLE
    }
    
    try:
        import sklearn
        deps['sklearn'] = True
    except ImportError:
        deps['sklearn'] = False
    
    try:
        import pandas
        deps['pandas'] = True
    except ImportError:
        deps['pandas'] = False
    
    try:
        import numpy
        deps['numpy'] = True  
    except ImportError:
        deps['numpy'] = False
    
    return deps

# Logging setup
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Package-level constants
MAX_SEQUENCE_LENGTH = 512
MIN_SEQUENCE_LENGTH = 10
SUPPORTED_LANGUAGES = ['en']  # Currently only English
CACHE_DIR = '.emotion_recognition_cache'

__all__ = [
    # Core classes
    'EmotionDataset',
    'DistilBERTEmotionModel', 
    'TwitterRoBERTaEmotionModel',
    'BiLSTMEmotionModel',
    'EnsembleEmotionModel',
    'EmotionPredictor',
    'EmotionPreprocessor',
    'EmotionDataProcessor',
    'EmotionTrainer',
    'EmotionEvaluator',
    'EmotionDataLoader',
    'ConfigManager',
    
    # Factory functions
    'create_model',
    
    # Utility functions
    'setup_data_directories',
    'download_sample_data',
    'setup_logging',
    'get_device_config',
    'quick_predict',
    'get_version',
    'get_supported_models',
    'get_emotion_labels',
    'check_dependencies',
    
    # Configuration classes
    'ModelConfig',
    'TrainingConfig', 
    'DataConfig',
    'PreprocessingConfig',
    'EvaluationConfig',
    'PathsConfig',
    'ExperimentConfig',
    
    # Constants
    'SUPPORTED_MODELS',
    'EMOTION_LABELS',
    'DEFAULT_CONFIG',
    'MAX_SEQUENCE_LENGTH',
    'MIN_SEQUENCE_LENGTH',
    'SUPPORTED_LANGUAGES',
    'CACHE_DIR'
]
