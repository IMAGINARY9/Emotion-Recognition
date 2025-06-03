"""
Unit tests initialization for emotion recognition project.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Test configuration
TEST_CONFIG = {
    'test_data_size': 100,
    'test_batch_size': 4,
    'test_max_length': 32,
    'test_num_classes': 6,
    'test_vocab_size': 1000,
    'test_device': 'cpu'
}

# Emotion labels for testing
TEST_EMOTION_LABELS = [
    "sadness",
    "joy", 
    "love",
    "anger",
    "fear",
    "surprise"
]

# Sample texts for testing
SAMPLE_TEST_TEXTS = [
    ("I love this amazing day!", "joy"),
    ("This makes me really angry", "anger"),
    ("I'm scared of what might happen", "fear"),
    ("That was such a beautiful surprise", "surprise"),
    ("I love spending time with my family", "love"),
    ("I feel really sad about this news", "sadness"),
]

def get_test_config():
    """Get test configuration."""
    return TEST_CONFIG.copy()

def get_test_emotion_labels():
    """Get test emotion labels."""
    return TEST_EMOTION_LABELS.copy()

def get_sample_test_texts():
    """Get sample test texts."""
    return SAMPLE_TEST_TEXTS.copy()
