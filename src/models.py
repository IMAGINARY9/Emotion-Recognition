"""
Emotion recognition models.

This module provides various neural network models for emotion recognition
from social media text, including transformer-based models and traditional
neural networks with word embeddings.

Based on the Twitter emotion classification task with 6 emotions:
- sadness, joy, love, anger, fear, surprise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.nn.utils.rnn import pad_sequence

class EmotionDataset(Dataset):
    """Dataset class for emotion recognition."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer=None, max_length: int = 128, vocab=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = vocab or {}
        
    def __len__(self):
        return len(self.texts)
    
    def text_to_ids(self, text: str) -> List[int]:
        """Convert text to list of token IDs for traditional models."""
        tokens = text.lower().split()
        return [self.vocab.get(token, 0) for token in tokens[:self.max_length]]
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.tokenizer:
            # For transformer models
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # For traditional models, return text and label
            return {
                'input_ids': torch.tensor(self.text_to_ids(text), dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            }
    
    def get_collate_fn(self):
        """Return a collate function that pads input_ids for traditional models."""
        if self.tokenizer:
            return None  # Use default collate for transformers
        def collate_fn(batch):
            # Pad input_ids to max_length
            input_ids = [item['input_ids'] for item in batch]
            labels = [item['label'] for item in batch]
            # Pad sequences
            padded = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=0
            )
            # Truncate or pad to max_length
            if padded.size(1) > self.max_length:
                padded = padded[:, :self.max_length]
            elif padded.size(1) < self.max_length:
                pad_width = self.max_length - padded.size(1)
                padded = torch.nn.functional.pad(padded, (0, pad_width), value=0)
            labels = torch.stack(labels)
            return {'input_ids': padded, 'label': labels}
        return collate_fn

class DistilBERTEmotionModel(nn.Module):
    """
    DistilBERT-based model for emotion recognition.
    
    This model uses a pre-trained DistilBERT model and adds
    additional layers for emotion classification (6 classes).
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 num_labels: int = 6, dropout: float = 0.1, freeze_base: bool = False):
        super().__init__()
        
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.distilbert = AutoModel.from_pretrained(model_name)
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        # Additional layers for fine-tuning
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
            
        Returns:
            Dictionary containing logits and optionally loss
        """
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        result = {'logits': logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result['loss'] = loss
            
        return result

class TwitterRoBERTaEmotionModel(nn.Module):
    """
    Twitter RoBERTa-based model for emotion recognition.
    
    This model uses a pre-trained Twitter RoBERTa model optimized
    for social media text and emotion classification.
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-emotion", 
                 num_labels: int = 6, dropout: float = 0.1, freeze_base: bool = False):
        super().__init__()
        
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        # Additional layers for fine-tuning
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        result = {'logits': logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result['loss'] = loss
            
        return result

class BiLSTMEmotionModel(nn.Module):
    """
    BiLSTM with attention mechanism for emotion recognition.
    
    This model uses word embeddings (GloVe) and bidirectional LSTM
    with attention for emotion classification.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300, hidden_dim: int = 128,
                 num_layers: int = 2, num_labels: int = 6, dropout: float = 0.3,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_labels = num_labels
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            
        # BiLSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        
    def attention_mechanism(self, lstm_output, final_state):
        """Apply attention mechanism to LSTM outputs."""
        # lstm_output: (batch_size, seq_len, hidden_dim * 2)
        # final_state: (batch_size, hidden_dim * 2)
        
        # Calculate attention weights
        attention_weights = torch.tanh(self.attention(lstm_output))  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights.squeeze(2), dim=1)  # (batch_size, seq_len)
        
        # Apply attention weights
        attended_output = torch.sum(lstm_output * attention_weights.unsqueeze(2), dim=1)
        return attended_output
        
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_output, (hidden, cell) = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Concatenate final hidden states from both directions
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch_size, hidden_dim * 2)
        
        # Apply attention
        attended_output = self.attention_mechanism(lstm_output, final_hidden)
        
        # Classification
        output = self.dropout(attended_output)
        logits = self.classifier(output)
        
        result = {'logits': logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result['loss'] = loss
            
        return result

class EnsembleEmotionModel(nn.Module):
    """
    Ensemble model combining multiple approaches for emotion recognition.
    
    Combines transformer models with traditional models and lexicon-based approaches.
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.num_labels = models[0].num_labels if hasattr(models[0], 'num_labels') else 6
        
    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble."""
        logits_list = []
        
        for model in self.models:
            try:
                outputs = model(**inputs)
                logits_list.append(outputs['logits'])
            except Exception as e:
                # Skip models that can't handle the input format
                continue
                
        if not logits_list:
            raise RuntimeError("No models in ensemble could process the inputs")
            
        # Weighted average of logits
        weighted_logits = sum(w * logits for w, logits in zip(self.weights, logits_list))
        
        result = {'logits': weighted_logits}
        
        if 'labels' in inputs and inputs['labels'] is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(weighted_logits, inputs['labels'])
            result['loss'] = loss
            
        return result

class EmotionPredictor:
    """
    High-level predictor class for emotion recognition.
    
    Provides easy-to-use interface for emotion prediction with preprocessing.
    """
    
    def __init__(self, model: nn.Module, tokenizer, preprocessor=None, 
                 label_names: List[str] = None, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.device = device
        self.label_names = label_names or ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, texts: Union[str, List[str]], return_probabilities: bool = False) -> Union[Dict, List[Dict]]:
        """
        Predict emotions for input text(s).
        
        Args:
            texts: Input text or list of texts
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Prediction results with emotions and optionally probabilities
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
            
        # Preprocess texts if preprocessor is available
        if self.preprocessor:
            texts = [self.preprocessor.preprocess_text(text) for text in texts]
            
        results = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Predict
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                probabilities = F.softmax(logits, dim=-1)
                
                # Get prediction
                predicted_class = torch.argmax(logits, dim=-1).item()
                predicted_emotion = self.label_names[predicted_class]
                confidence = probabilities[0, predicted_class].item()
                
                result = {
                    'text': text,
                    'emotion': predicted_emotion,
                    'confidence': confidence
                }
                
                if return_probabilities:
                    prob_dict = {
                        emotion: prob.item() 
                        for emotion, prob in zip(self.label_names, probabilities[0])
                    }
                    result['probabilities'] = prob_dict
                    
                results.append(result)
        
        return results[0] if is_single else results

def create_model(model_type: str, model_config: Dict) -> nn.Module:
    """
    Factory function to create emotion recognition models.
    
    Args:
        model_type: Type of model ('distilbert', 'twitter-roberta', 'bilstm', 'ensemble')
        model_config: Configuration dictionary for the model
        
    Returns:
        Initialized model
    """
    if model_type == 'distilbert':
        return DistilBERTEmotionModel(**model_config)
    elif model_type == 'twitter-roberta':
        return TwitterRoBERTaEmotionModel(**model_config)
    elif model_type == 'bilstm':
        return BiLSTMEmotionModel(**model_config)
    elif model_type == 'ensemble':
        # For ensemble, model_config should contain 'models' and optionally 'weights'
        return EnsembleEmotionModel(**model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
