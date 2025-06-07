"""
Enhanced BiLSTM Dataset with improved tokenization and vocabulary handling.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import numpy as np
from vocab_utils import BiLSTMVocabularyBuilder


class EnhancedBiLSTMDataset(Dataset):
    """Enhanced dataset for BiLSTM with better preprocessing and vocabulary handling."""
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int],
                 vocab: Optional[Dict[str, int]] = None,
                 max_length: int = 128,
                 vocab_builder: Optional[BiLSTMVocabularyBuilder] = None):
        """
        Initialize the enhanced BiLSTM dataset.
        
        Args:
            texts: List of input texts
            labels: List of labels
            vocab: Pre-built vocabulary (if None, will build from texts)
            max_length: Maximum sequence length
            vocab_builder: Vocabulary builder instance
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.vocab = vocab
        self.vocab_builder = vocab_builder or BiLSTMVocabularyBuilder()
        
        # Build vocabulary if not provided
        if self.vocab is None:
            print("Building vocabulary for BiLSTM dataset...")
            self.vocab = self.vocab_builder.build_vocabulary(texts, labels)
        
        # Convert texts to token IDs
        print("Converting texts to token IDs...")
        self.token_ids = []
        for text in texts:
            ids = self.vocab_builder.text_to_ids(text, self.vocab, max_length)
            self.token_ids.append(ids)
        
        print(f"Dataset ready: {len(texts)} samples, vocab size: {len(self.vocab)}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.token_ids[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'text': self.texts[idx]  # Keep original text for debugging
        }
    
    def get_vocab(self):
        """Return the vocabulary."""
        return self.vocab
    
    def get_collate_fn(self):
        """Return a collate function for DataLoader."""
        def collate_fn(batch):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            labels = torch.stack([item['label'] for item in batch])
            texts = [item['text'] for item in batch]
            
            return {
                'input_ids': input_ids,
                'label': labels,
                'texts': texts
            }
        return collate_fn


class JointEnsembleDataset(Dataset):
    """Dataset for joint ensemble training with multiple tokenizers."""
    
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': self.labels[idx]
        }
