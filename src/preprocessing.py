"""
Text preprocessing for emotion recognition.

This module provides comprehensive text preprocessing capabilities
specifically designed for social media text and emotion analysis,
including emoji handling, social media artifacts, and text normalization.
"""

import re
import string
import emoji
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import contractions
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class EmotionPreprocessor:
    """
    Comprehensive text preprocessor for emotion recognition from social media text.
    """
    
    def __init__(self, 
                 handle_emojis: str = "convert",  # "convert", "remove", "keep"
                 expand_contractions: bool = True,
                 remove_stopwords: bool = False,
                 lemmatize: bool = False,
                 normalize_case: bool = True,
                 handle_social_media: bool = True,
                 min_length: int = 3,
                 max_length: int = 512):
        """
        Initialize the preprocessor with specified options.
        
        Args:
            handle_emojis: How to handle emojis ("convert", "remove", "keep")
            expand_contractions: Whether to expand contractions
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            normalize_case: Whether to normalize case
            handle_social_media: Whether to handle social media artifacts
            min_length: Minimum text length after preprocessing
            max_length: Maximum text length after preprocessing
        """
        self.handle_emojis = handle_emojis
        self.expand_contractions = expand_contractions
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.normalize_case = normalize_case
        self.handle_social_media = handle_social_media
        self.min_length = min_length
        self.max_length = max_length
        
        # Initialize NLTK components
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
            
        # Social media patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.repeated_char_pattern = re.compile(r'(.)\1{2,}')
        
    def preprocess_text(self, text: str) -> str:
        """
        Apply all preprocessing steps to a single text.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
            
        # Basic cleaning
        text = self._basic_cleaning(text)
        
        # Handle social media artifacts
        if self.handle_social_media:
            text = self._handle_social_media_artifacts(text)
            
        # Handle emojis
        if self.handle_emojis == "convert":
            text = self._convert_emojis(text)
        elif self.handle_emojis == "remove":
            text = self._remove_emojis(text)
            
        # Expand contractions
        if self.expand_contractions:
            text = self._expand_contractions(text)
            
        # Normalize case
        if self.normalize_case:
            text = text.lower()
            
        # Tokenize and clean
        tokens = self._tokenize_and_clean(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self._remove_stopwords(tokens)
            
        # Lemmatize
        if self.lemmatize:
            tokens = self._lemmatize_tokens(tokens)
            
        # Join tokens back
        text = ' '.join(tokens)
        
        # Final validation
        if len(text) < self.min_length:
            return ""
            
        if len(text) > self.max_length:
            text = text[:self.max_length]
            
        return text.strip()
    
    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove null bytes
        text = text.replace('\x00', '')
        return text.strip()
    
    def _handle_social_media_artifacts(self, text: str) -> str:
        """Handle social media specific elements."""
        # Replace URLs with token
        text = self.url_pattern.sub(' <URL> ', text)
        
        # Replace mentions with token
        text = self.mention_pattern.sub(' <USER> ', text)
        
        # Extract hashtag text (remove # but keep the word)
        text = self.hashtag_pattern.sub(r' \1 ', text)
        
        # Handle repeated characters (e.g., "sooooo" -> "so")
        text = self.repeated_char_pattern.sub(r'\1\1', text)
        
        return text
    
    def _convert_emojis(self, text: str) -> str:
        """Convert emojis to text descriptions."""
        try:
            # Convert emojis to text descriptions
            text = emoji.demojize(text, delimiters=(" <", "> "))
            # Clean up the emoji descriptions
            text = re.sub(r'<[^>]+>', lambda m: m.group(0).replace('_', ' '), text)
        except Exception:
            # If emoji processing fails, just remove emojis
            text = self._remove_emojis(text)
        return text
    
    def _remove_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        try:
            return emoji.replace_emoji(text, replace='')
        except Exception:
            # Fallback: use regex to remove common emoji patterns
            emoji_pattern = re.compile("["
                                     u"\U0001F600-\U0001F64F"  # emoticons
                                     u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                     u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                     u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                     u"\U00002702-\U000027B0"
                                     u"\U000024C2-\U0001F251"
                                     "]+", flags=re.UNICODE)
            return emoji_pattern.sub(r'', text)
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text."""
        try:
            return contractions.fix(text)
        except Exception:
            # Basic contraction expansion as fallback
            contractions_dict = {
                "ain't": "am not", "aren't": "are not", "can't": "cannot",
                "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                "haven't": "have not", "he'd": "he would", "he'll": "he will",
                "he's": "he is", "i'd": "i would", "i'll": "i will",
                "i'm": "i am", "i've": "i have", "isn't": "is not",
                "it'd": "it would", "it'll": "it will", "it's": "it is",
                "let's": "let us", "shouldn't": "should not", "that's": "that is",
                "there's": "there is", "they'd": "they would", "they'll": "they will",
                "they're": "they are", "they've": "they have", "we'd": "we would",
                "we're": "we are", "we've": "we have", "weren't": "were not",
                "what's": "what is", "where's": "where is", "who's": "who is",
                "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                "you'll": "you will", "you're": "you are", "you've": "you have"
            }
            
            for contraction, expansion in contractions_dict.items():
                text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
            return text
    
    def _tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize text and clean tokens."""
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback tokenization
            tokens = text.split()
            
        # Clean tokens
        cleaned_tokens = []
        for token in tokens:
            # Remove punctuation and keep only alphabetic tokens (with some exceptions)
            if token.isalpha() or token in ['<URL>', '<USER>'] or token.startswith('<') and token.endswith('>'):
                cleaned_tokens.append(token)
            elif re.match(r"^[a-zA-Z]+['\"]?[a-zA-Z]*$", token):
                # Keep words with apostrophes
                cleaned_tokens.append(token)
                
        return cleaned_tokens
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens."""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str, 
                          label_column: str, dataset_name: str = "") -> pd.DataFrame:
        """
        Preprocess an entire dataset.
        
        Args:
            df: Input dataframe
            text_column: Name of the text column
            label_column: Name of the label column
            dataset_name: Name of the dataset for logging
            
        Returns:
            Preprocessed dataframe
        """
        print(f"📚 Processing {dataset_name} dataset...")
        print(f"   Original size: {len(df)} rows")
        
        # Create a copy of the dataframe
        processed_df = df.copy()
        
        # Preprocess texts
        processed_texts = []
        for text in df[text_column]:
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
        
        processed_df['clean_text'] = processed_texts
        
        # Remove empty texts
        original_size = len(processed_df)
        processed_df = processed_df[processed_df['clean_text'].str.len() >= self.min_length]
        removed_count = original_size - len(processed_df)
        
        if removed_count > 0:
            print(f"   Removed {removed_count} texts (too short after preprocessing)")
        
        print(f"   Final size: {len(processed_df)} rows")
        
        return processed_df
    
    def get_emotion_statistics(self, df: pd.DataFrame, label_column: str, 
                              emotion_names: List[str]) -> Dict:
        """
        Get statistics about emotion distribution in the dataset.
        
        Args:
            df: Input dataframe
            label_column: Name of the label column
            emotion_names: List of emotion names
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Emotion distribution
        emotion_counts = df[label_column].value_counts().sort_index()
        stats['emotion_distribution'] = {}
        
        for idx, count in emotion_counts.items():
            if idx < len(emotion_names):
                emotion_name = emotion_names[idx]
                percentage = (count / len(df)) * 100
                stats['emotion_distribution'][emotion_name] = {
                    'count': count,
                    'percentage': percentage
                }
        
        # Text length statistics
        text_lengths = df['clean_text'].str.len()
        stats['text_length'] = {
            'mean': text_lengths.mean(),
            'median': text_lengths.median(),
            'min': text_lengths.min(),
            'max': text_lengths.max(),
            'std': text_lengths.std()
        }
        
        # Word count statistics
        word_counts = df['clean_text'].str.split().str.len()
        stats['word_count'] = {
            'mean': word_counts.mean(),
            'median': word_counts.median(),
            'min': word_counts.min(),
            'max': word_counts.max(),
            'std': word_counts.std()
        }
        
        return stats
    
    def show_preprocessing_examples(self, df: pd.DataFrame, text_column: str, 
                                   emotion_names: List[str], num_examples: int = 3):
        """
        Show examples of preprocessing results.
        
        Args:
            df: Processed dataframe
            text_column: Name of original text column
            emotion_names: List of emotion names
            num_examples: Number of examples to show
        """
        print(f"\n💡 Preprocessing Examples:\n")
        
        for i in range(min(num_examples, len(df))):
            row = df.iloc[i]
            emotion_idx = row['label'] if 'label' in row else row.get('emotion', 0)
            emotion_name = emotion_names[emotion_idx] if emotion_idx < len(emotion_names) else f"Emotion {emotion_idx}"
            
            original_text = row[text_column][:100] + "..." if len(row[text_column]) > 100 else row[text_column]
            cleaned_text = row['clean_text'][:100] + "..." if len(row['clean_text']) > 100 else row['clean_text']
            
            print(f"   Example {i+1}:")
            print(f"   Original:  {original_text}")
            print(f"   Cleaned:   {cleaned_text}")
            print(f"   Emotion:   {emotion_name}")
            print()

class EmotionDataProcessor:
    """
    High-level data processor for emotion recognition datasets.
    """
    
    def __init__(self, preprocessor: EmotionPreprocessor = None):
        self.preprocessor = preprocessor or EmotionPreprocessor()
        self.emotion_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    
    def process_emotion_dataset(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                               test_df: pd.DataFrame, text_column: str = 'text', 
                               label_column: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process emotion dataset splits.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe  
            test_df: Test dataframe
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Tuple of processed dataframes
        """
        # Process each split
        train_processed = self.preprocessor.preprocess_dataset(
            train_df, text_column, label_column, "Training"
        )
        
        val_processed = self.preprocessor.preprocess_dataset(
            val_df, text_column, label_column, "Validation"
        )
        
        test_processed = self.preprocessor.preprocess_dataset(
            test_df, text_column, label_column, "Test"
        )
        
        # Show statistics
        print("\n📊 Dataset Statistics:")
        for name, df in [("Training", train_processed), ("Validation", val_processed), ("Test", test_processed)]:
            stats = self.preprocessor.get_emotion_statistics(df, label_column, self.emotion_names)
            print(f"\n{name} Set:")
            for emotion, info in stats['emotion_distribution'].items():
                print(f"   {emotion.capitalize()}: {info['count']} ({info['percentage']:.1f}%)")
        
        # Show preprocessing examples
        self.preprocessor.show_preprocessing_examples(
            train_processed, text_column, self.emotion_names
        )
        
        return train_processed, val_processed, test_processed
    
    def prepare_for_training(self, df: pd.DataFrame, text_column: str = 'clean_text', 
                           label_column: str = 'label') -> Tuple[List[str], List[int]]:
        """
        Prepare processed dataframe for model training.
        
        Args:
            df: Processed dataframe
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Tuple of texts and labels
        """
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        return texts, labels
