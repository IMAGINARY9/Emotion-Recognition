"""
Vocabulary utilities for BiLSTM models.

This module provides improved vocabulary building and management
for BiLSTM models with better handling of text preprocessing and
vocabulary optimization.
"""

import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import pandas as pd
import numpy as np
from preprocessing import EmotionPreprocessor

class BiLSTMVocabularyBuilder:
    """
    Advanced vocabulary builder for BiLSTM models with emotion-aware optimization.
    """
    
    def __init__(self, 
                 min_freq: int = 2,
                 max_vocab_size: int = 15000,
                 special_tokens: List[str] = None,
                 preserve_emotion_words: bool = True,
                 preprocessor: EmotionPreprocessor = None):
        """
        Initialize the vocabulary builder.
        
        Args:
            min_freq: Minimum frequency for including words
            max_vocab_size: Maximum vocabulary size
            special_tokens: Special tokens to include
            preserve_emotion_words: Whether to preserve emotion-related words regardless of frequency
            preprocessor: Text preprocessor for consistent tokenization
        """
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<URL>', '<USER>']
        self.preserve_emotion_words = preserve_emotion_words
        self.preprocessor = preprocessor or EmotionPreprocessor(
            handle_emojis="convert",
            expand_contractions=True,
            remove_stopwords=False,
            normalize_case=True,
            handle_social_media=True
        )
        
        # Emotion-related words to preserve
        self.emotion_keywords = {
            'positive': [
                'happy', 'joy', 'love', 'excited', 'wonderful', 'amazing', 'great', 'awesome',
                'fantastic', 'excellent', 'brilliant', 'beautiful', 'good', 'nice', 'pleasant',
                'delighted', 'thrilled', 'elated', 'cheerful', 'optimistic', 'grateful',
                'satisfied', 'content', 'pleased', 'glad', 'blissful', 'ecstatic'
            ],
            'negative': [
                'sad', 'angry', 'fear', 'hate', 'terrible', 'awful', 'bad', 'horrible',
                'disgusting', 'annoying', 'frustrated', 'disappointed', 'depressed',
                'worried', 'anxious', 'scared', 'furious', 'enraged', 'devastated',
                'heartbroken', 'miserable', 'upset', 'distressed', 'troubled', 'concerned'
            ],
            'intensity': [
                'very', 'extremely', 'really', 'totally', 'completely', 'absolutely',
                'incredibly', 'tremendously', 'enormously', 'utterly', 'quite',
                'rather', 'fairly', 'somewhat', 'slightly', 'barely', 'hardly'
            ],
            'negation': [
                'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither',
                'nor', 'dont', "don't", 'cant', "can't", 'wont', "won't", 'isnt', "isn't"
            ]
        }
        
        # Flatten emotion keywords
        self.important_words = set()
        for category in self.emotion_keywords.values():
            self.important_words.update(category)
            
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text using the preprocessor for consistent vocabulary building.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Apply preprocessing
        processed_text = self.preprocessor.preprocess_text(text)
        
        if not processed_text or len(processed_text.strip()) == 0:
            return []
            
        # Simple tokenization (split by whitespace)
        tokens = processed_text.split()
        
        # Additional cleaning for vocabulary
        cleaned_tokens = []
        for token in tokens:
            # Keep tokens that are:
            # 1. Alphabetic
            # 2. Special tokens
            # 3. Alphanumeric with some punctuation (like contractions)
            if (token.isalpha() or 
                token in self.special_tokens or
                re.match(r"^[a-zA-Z0-9]+['\-_]?[a-zA-Z0-9]*$", token)):
                cleaned_tokens.append(token.lower())
                
        return cleaned_tokens
    
    def build_vocabulary(self, 
                        texts: List[str], 
                        labels: Optional[List[int]] = None) -> Dict[str, int]:
        """
        Build vocabulary from training texts with emotion-aware optimization.
        
        Args:
            texts: List of training texts
            labels: Optional list of labels for emotion-specific vocabulary building
            
        Returns:
            Dictionary mapping words to indices
        """
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Count word frequencies
        word_counts = Counter()
        emotion_word_counts = defaultdict(Counter)  # Track words by emotion class
        
        for i, text in enumerate(texts):
            tokens = self.tokenize_text(text)
            word_counts.update(tokens)
            
            # Track emotion-specific word usage
            if labels is not None and i < len(labels):
                emotion_word_counts[labels[i]].update(tokens)
        
        print(f"Found {len(word_counts)} unique tokens before filtering")
        
        # Start with special tokens
        vocab = {}
        for i, token in enumerate(self.special_tokens):
            vocab[token] = i
        
        # Filter words by frequency and importance
        filtered_words = []
        
        for word, count in word_counts.items():
            # Skip special tokens (already added)
            if word in self.special_tokens:
                continue
                
            # Include if:
            # 1. Meets minimum frequency requirement
            # 2. Is an important emotion word (regardless of frequency)
            # 3. Appears in multiple emotion classes (versatile word)
            include_word = False
            
            if count >= self.min_freq:
                include_word = True
            elif self.preserve_emotion_words and word in self.important_words:
                include_word = True
            elif labels is not None:
                # Include if word appears in multiple emotion classes
                emotion_classes = sum(1 for emotion_counts in emotion_word_counts.values() 
                                    if emotion_counts[word] > 0)
                if emotion_classes >= 2:
                    include_word = True
            
            if include_word:
                filtered_words.append((word, count))
        
        print(f"Filtered to {len(filtered_words)} words meeting criteria")
        
        # Sort by frequency (descending) but prioritize important words
        def word_priority(word_count_pair):
            word, count = word_count_pair
            # Higher priority for emotion words
            priority_boost = 1000000 if word in self.important_words else 0
            return count + priority_boost
        
        filtered_words.sort(key=word_priority, reverse=True)
        
        # Add words to vocabulary up to max size
        max_regular_words = self.max_vocab_size - len(self.special_tokens)
        for word, count in filtered_words[:max_regular_words]:
            vocab[word] = len(vocab)
        
        print(f"Final vocabulary size: {len(vocab)}")
        print(f"Coverage: {self._calculate_coverage(vocab, texts):.2%}")
        
        return vocab
    
    def _calculate_coverage(self, vocab: Dict[str, int], texts: List[str]) -> float:
        """Calculate vocabulary coverage on the training texts."""
        total_tokens = 0
        covered_tokens = 0
        
        for text in texts[:1000]:  # Sample for speed
            tokens = self.tokenize_text(text)
            total_tokens += len(tokens)
            covered_tokens += sum(1 for token in tokens if token in vocab)
        
        return covered_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def analyze_vocabulary(self, vocab: Dict[str, int], texts: List[str]) -> Dict:
        """
        Analyze vocabulary characteristics and provide statistics.
        
        Args:
            vocab: Built vocabulary
            texts: Training texts
            
        Returns:
            Dictionary with vocabulary statistics
        """
        # Count total tokens in texts
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize_text(text))
        
        token_counts = Counter(all_tokens)
        
        # Calculate statistics
        vocab_words = set(vocab.keys()) - set(self.special_tokens)
        oov_words = set(token_counts.keys()) - set(vocab.keys())
        
        # Frequency distribution
        in_vocab_freq = sum(count for word, count in token_counts.items() if word in vocab)
        total_freq = sum(token_counts.values())
        
        # Emotion word coverage
        emotion_words_in_vocab = len(vocab_words & self.important_words)
        total_emotion_words = len(self.important_words)
        
        stats = {
            'vocab_size': len(vocab),
            'special_tokens': len(self.special_tokens),
            'regular_words': len(vocab_words),
            'unique_tokens_in_data': len(token_counts),
            'oov_words': len(oov_words),
            'frequency_coverage': in_vocab_freq / total_freq if total_freq > 0 else 0,
            'emotion_word_coverage': emotion_words_in_vocab / total_emotion_words,
            'most_frequent_oov': token_counts.most_common() if oov_words else [],
            'vocab_distribution': {
                'low_freq': sum(1 for word in vocab_words if token_counts[word] < 5),
                'med_freq': sum(1 for word in vocab_words if 5 <= token_counts[word] < 50),
                'high_freq': sum(1 for word in vocab_words if token_counts[word] >= 50),
            }
        }
        
        return stats
    
    def save_vocabulary(self, vocab: Dict[str, int], filepath: str):
        """Save vocabulary to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
        
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> Dict[str, int]:
        """Load vocabulary from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        print(f"Vocabulary loaded from {filepath} (size: {len(vocab)})")
        return vocab
    
    def text_to_ids(self, text: str, vocab: Dict[str, int], max_length: int = 128) -> List[int]:
        """
        Convert text to list of token IDs.
        
        Args:
            text: Input text
            vocab: Vocabulary dictionary
            max_length: Maximum sequence length
            
        Returns:
            List of token IDs with padding
        """
        tokens = self.tokenize_text(text)
        
        # Convert to IDs
        ids = []
        unk_id = vocab.get('<UNK>', 1)
        for token in tokens[:max_length]:
            ids.append(vocab.get(token, unk_id))
        
        # Pad to max_length
        pad_id = vocab.get('<PAD>', 0)
        while len(ids) < max_length:
            ids.append(pad_id)
            
        return ids
    
    def batch_texts_to_ids(self, 
                          texts: List[str], 
                          vocab: Dict[str, int], 
                          max_length: int = 128) -> np.ndarray:
        """
        Convert batch of texts to numpy array of token IDs.
        
        Args:
            texts: List of input texts
            vocab: Vocabulary dictionary  
            max_length: Maximum sequence length
            
        Returns:
            Numpy array of shape (batch_size, max_length)
        """
        batch_ids = []
        for text in texts:
            ids = self.text_to_ids(text, vocab, max_length)
            batch_ids.append(ids)
        
        return np.array(batch_ids, dtype=np.int64)


def create_bilstm_vocabulary(train_texts: List[str], 
                           train_labels: Optional[List[int]] = None,
                           config: Optional[Dict] = None) -> Dict[str, int]:
    """
    Convenient function to create BiLSTM vocabulary from training data.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels (optional)
        config: Configuration dictionary with vocab settings
        
    Returns:
        Vocabulary dictionary
    """
    # Extract vocabulary settings from config
    if config is None:
        config = {}
    
    vocab_config = config.get('preprocessing', {})
    min_freq = vocab_config.get('min_freq', 2)
    max_vocab_size = vocab_config.get('max_vocab_size', 15000)
    
    # Create preprocessor based on config
    preproc_config = config.get('preprocessing', {})
    preprocessor = EmotionPreprocessor(
        handle_emojis=preproc_config.get('emoji_handling', 'convert'),
        expand_contractions=preproc_config.get('expand_contractions', True),
        remove_stopwords=preproc_config.get('remove_stopwords', False),
        normalize_case=preproc_config.get('lowercase', True),
        handle_social_media=True
    )
    
    # Build vocabulary
    builder = BiLSTMVocabularyBuilder(
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        preprocessor=preprocessor
    )
    
    vocab = builder.build_vocabulary(train_texts, train_labels)
    
    # Print statistics
    stats = builder.analyze_vocabulary(vocab, train_texts)
    print("\nVocabulary Statistics:")
    print(f"  Vocabulary size: {stats['vocab_size']}")
    print(f"  Regular words: {stats['regular_words']}")
    print(f"  Frequency coverage: {stats['frequency_coverage']:.2%}")
    print(f"  Emotion word coverage: {stats['emotion_word_coverage']:.2%}")
    print(f"  OOV words: {stats['oov_words']}")
    
    return vocab
