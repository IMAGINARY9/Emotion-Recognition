"""
Data utilities for emotion recognition project.
Handles data loading, validation, and management.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class EmotionDataLoader:
    """Handles loading and managing emotion datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.emotion_labels)
        
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} samples from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame, text_col: str = 'text', 
                     label_col: str = 'label') -> pd.DataFrame:
        """
        Validate and clean the dataset.
        
        Args:
            df: Input DataFrame
            text_col: Name of text column
            label_col: Name of label column
            
        Returns:
            Validated DataFrame
        """
        logger.info("Validating dataset...")
        
        # Check required columns
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in data")
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in data")
        
        # Remove empty text
        initial_len = len(df)
        df = df.dropna(subset=[text_col, label_col])
        df = df[df[text_col].str.strip() != '']
        
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} empty/null samples")
        
        # Validate labels
        valid_labels = set(self.emotion_labels)
        invalid_labels = set(df[label_col].unique()) - valid_labels
        
        if invalid_labels:
            logger.warning(f"Found invalid labels: {invalid_labels}")
            df = df[df[label_col].isin(valid_labels)]
            logger.info(f"Filtered to {len(df)} samples with valid labels")
        
        # Check class distribution
        class_counts = df[label_col].value_counts()
        logger.info("Class distribution:")
        for label, count in class_counts.items():
            logger.info(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
        
        return df.reset_index(drop=True)
    
    def encode_labels(self, labels: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Encode text labels to integers.
        
        Args:
            labels: Text labels to encode
            
        Returns:
            Encoded labels as numpy array
        """
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels: Union[List[int], np.ndarray, torch.Tensor]) -> List[str]:
        """
        Decode integer labels to text.
        
        Args:
            encoded_labels: Encoded labels to decode
            
        Returns:
            List of text labels
        """
        if isinstance(encoded_labels, torch.Tensor):
            encoded_labels = encoded_labels.cpu().numpy()
        return self.label_encoder.inverse_transform(encoded_labels).tolist()
    
    def split_data(self, df: pd.DataFrame, 
                   train_size: float = 0.8, 
                   val_size: float = 0.1,
                   test_size: float = 0.1,
                   stratify: bool = True,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            train_size: Proportion for training set
            val_size: Proportion for validation set
            test_size: Proportion for test set
            stratify: Whether to stratify by label
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Validate split proportions
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("Split proportions must sum to 1.0")
        
        # First split: train vs (val + test)
        stratify_col = df['label'] if stratify else None
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_size + test_size),
            stratify=stratify_col,
            random_state=random_state
        )
        
        # Second split: val vs test
        if val_size > 0 and test_size > 0:
            val_prop = val_size / (val_size + test_size)
            stratify_col = temp_df['label'] if stratify else None
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - val_prop),
                stratify=stratify_col,
                random_state=random_state
            )
        elif val_size > 0:
            val_df = temp_df
            test_df = pd.DataFrame()
        else:
            val_df = pd.DataFrame()
            test_df = temp_df
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, 
                   val_df: pd.DataFrame, 
                   test_df: pd.DataFrame,
                   output_dir: str = None) -> Dict[str, str]:
        """
        Save data splits to CSV files.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame  
            test_df: Test DataFrame
            output_dir: Output directory (defaults to self.data_dir)
            
        Returns:
            Dictionary mapping split names to file paths
        """
        if output_dir is None:
            output_dir = self.data_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        
        if not train_df.empty:
            train_path = output_dir / "train.csv"
            train_df.to_csv(train_path, index=False)
            file_paths['train'] = str(train_path)
            logger.info(f"Saved training data: {train_path}")
        
        if not val_df.empty:
            val_path = output_dir / "validation.csv"
            val_df.to_csv(val_path, index=False)
            file_paths['validation'] = str(val_path)
            logger.info(f"Saved validation data: {val_path}")
        
        if not test_df.empty:
            test_path = output_dir / "test.csv"
            test_df.to_csv(test_path, index=False)
            file_paths['test'] = str(test_path)
            logger.info(f"Saved test data: {test_path}")
        
        return file_paths
    
    def load_splits(self, data_dir: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load existing data splits.
        
        Args:
            data_dir: Directory containing split files
            
        Returns:
            Dictionary mapping split names to DataFrames
        """
        if data_dir is None:
            data_dir = self.data_dir
        
        data_dir = Path(data_dir)
        splits = {}
        
        # Try to load each split
        for split_name in ['train', 'validation', 'test']:
            file_path = data_dir / f"{split_name}.csv"
            if file_path.exists():
                splits[split_name] = self.load_csv_data(str(file_path))
            else:
                logger.warning(f"Split file not found: {file_path}")
        
        return splits
    
    def create_data_loaders(self, datasets: Dict[str, Dataset], 
                           batch_sizes: Dict[str, int] = None,
                           shuffle: Dict[str, bool] = None,
                           num_workers: int = 0) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders from datasets.
        
        Args:
            datasets: Dictionary mapping split names to datasets
            batch_sizes: Batch sizes for each split
            shuffle: Whether to shuffle each split
            num_workers: Number of worker processes
            
        Returns:
            Dictionary mapping split names to DataLoaders
        """
        if batch_sizes is None:
            batch_sizes = {'train': 32, 'validation': 64, 'test': 64}
        
        if shuffle is None:
            shuffle = {'train': True, 'validation': False, 'test': False}
        
        data_loaders = {}
        
        for split_name, dataset in datasets.items():
            data_loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_sizes.get(split_name, 32),
                shuffle=shuffle.get(split_name, False),
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            logger.info(f"Created {split_name} DataLoader: "
                       f"batch_size={batch_sizes.get(split_name, 32)}, "
                       f"shuffle={shuffle.get(split_name, False)}")
        
        return data_loaders
    
    def get_class_weights(self, labels: Union[List[str], pd.Series]) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data.
        
        Args:
            labels: Training labels
            
        Returns:
            Tensor of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.array(self.emotion_labels)
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=labels
        )
        
        logger.info("Class weights:")
        for label, weight in zip(self.emotion_labels, weights):
            logger.info(f"  {label}: {weight:.3f}")
        
        return torch.FloatTensor(weights)
    
    def save_metadata(self, metadata: Dict, output_path: str = None):
        """
        Save dataset metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            output_path: Output file path
        """
        if output_path is None:
            output_path = self.data_dir / "metadata.json"
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata: {output_path}")
    
    def load_metadata(self, input_path: str = None) -> Dict:
        """
        Load dataset metadata from JSON file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Metadata dictionary
        """
        if input_path is None:
            input_path = self.data_dir / "metadata.json"
        
        if not os.path.exists(input_path):
            logger.warning(f"Metadata file not found: {input_path}")
            return {}
        
        with open(input_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded metadata: {input_path}")
        return metadata

def setup_data_directories(base_dir: str = "data") -> Dict[str, str]:
    """
    Set up data directory structure.
    
    Args:
        base_dir: Base data directory
        
    Returns:
        Dictionary mapping directory names to paths
    """
    base_path = Path(base_dir)
    
    directories = {
        'raw': base_path / 'raw',
        'processed': base_path / 'processed', 
        'splits': base_path / 'splits',
        'external': base_path / 'external'
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return {name: str(path) for name, path in directories.items()}

def download_sample_data(output_dir: str = "data/raw") -> str:
    """
    Create sample emotion dataset for testing.
    
    Args:
        output_dir: Output directory
        
    Returns:
        Path to created sample file
    """
    # Sample emotion data
    sample_data = [
        ("I feel so happy today!", "joy"),
        ("This makes me really angry", "anger"),
        ("I'm scared of what might happen", "fear"),
        ("That was such a beautiful surprise", "surprise"),
        ("I love spending time with my family", "love"),
        ("I feel really sad about this news", "sadness"),
        ("What an amazing day this has been!", "joy"),
        ("I can't believe this happened to me", "anger"),
        ("I'm worried about the future", "fear"),
        ("Wow, I never expected this!", "surprise"),
        ("I absolutely adore this song", "love"),
        ("This situation makes me feel down", "sadness"),
        ("The sunshine makes me feel wonderful", "joy"),
        ("This is so frustrating and annoying", "anger"),
        ("I have anxiety about tomorrow", "fear"),
        ("What a pleasant unexpected gift", "surprise"),
        ("I cherish these precious moments", "love"),
        ("I feel empty and melancholic", "sadness")
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_data, columns=['text', 'label'])
    
    # Add more samples by duplicating with variations
    additional_samples = []
    for text, label in sample_data:
        # Create variations
        variations = [
            text.lower(),
            text.upper(),
            text + " Really!",
            text.replace("I", "We"),
        ]
        for variation in variations:
            additional_samples.append((variation, label))
    
    additional_df = pd.DataFrame(additional_samples, columns=['text', 'label'])
    df = pd.concat([df, additional_df], ignore_index=True)
    
    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "sample_emotions.csv"
    
    df.to_csv(file_path, index=False)
    logger.info(f"Created sample dataset with {len(df)} samples: {file_path}")
    
    return str(file_path)
