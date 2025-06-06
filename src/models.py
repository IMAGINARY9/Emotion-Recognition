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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.nn.utils.rnn import pad_sequence
import itertools

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
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 freeze_embeddings: bool = False,
                 loss_type: str = 'cross_entropy',
                 gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.loss_type = loss_type
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = not freeze_embeddings
        # Dropout after embedding
        self.embedding_dropout = nn.Dropout(dropout)
        # BiLSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # BatchNorm after attention
        self.batchnorm = nn.BatchNorm1d(hidden_dim * 2)
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        # Loss
        if self.loss_type == 'focal':
            from src.losses import FocalLoss
            self.loss_fct = FocalLoss(gamma=self.gamma, label_smoothing=self.label_smoothing)
        elif self.loss_type == 'cross_entropy':
            self.loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
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
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        # LSTM
        lstm_output, (hidden, cell) = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        # Concatenate final hidden states from both directions
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch_size, hidden_dim * 2)
        # Apply attention
        attended_output = self.attention_mechanism(lstm_output, final_hidden)
        # BatchNorm
        normed_output = self.batchnorm(attended_output)
        # Classification
        output = self.dropout(normed_output)
        logits = self.classifier(output)
        result = {'logits': logits}
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            result['loss'] = loss
        return result

class EnsembleEmotionModel(nn.Module):
    """
    Ensemble model combining multiple approaches for emotion recognition.
    
    Combines transformer models with traditional models and lexicon-based approaches.
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None, label_names: Optional[List[str]] = None):
        super().__init__()
        self.models = nn.ModuleList([m for m in models if m is not None])
        if weights is not None:
            if len(weights) != len(models):
                print("[Ensemble Warning] Number of weights does not match number of models. Using equal weights.")
                self.weights = [1.0 / len(self.models)] * len(self.models)
            else:
                total = sum(weights)
                self.weights = [float(w) / total for w in weights]
        else:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        # Add label names for debug output
        self.label_names = label_names or ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    def _idx_to_label(self, idx_tensor):
        # idx_tensor: torch.Tensor of indices
        if isinstance(idx_tensor, torch.Tensor):
            return [self.label_names[i] if 0 <= i < len(self.label_names) else str(i) for i in idx_tensor.cpu().tolist()]
        return str(idx_tensor)

    def forward(self, debug_predictions: bool = False, **inputs) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble.
        Args:
            debug_predictions: If True, print [Ensemble Debug] prediction/logit info for each model and ensemble output.
        """
        logits_list = []
        successful_indices = []
        for i, model in enumerate(self.models):
            try:
                model_name = type(model).__name__.lower()
                if 'distilbert' in model_name:
                    for key in ['distilbert', 'distilbert-emotion', 'distilbert_emotion']:
                        if key in inputs:
                            sub_inputs = inputs[key]
                            outputs = model(**sub_inputs, labels=inputs['labels'])
                            break
                    else:
                        raise ValueError(f"No matching input for model {model_name}. Available keys: {list(inputs.keys())}")
                elif 'roberta' in model_name or 'twitterroberta' in model_name:
                    for key in ['twitter-roberta', 'twitter_roberta', 'roberta', 'roberta-emotion', 'roberta_emotion']:
                        if key in inputs:
                            sub_inputs = inputs[key]
                            outputs = model(**sub_inputs, labels=inputs['labels'])
                            break
                    else:
                        raise ValueError(f"No matching input for model {model_name}. Available keys: {list(inputs.keys())}")
                elif 'bilstm' in model_name:
                    if 'bilstm' in inputs:
                        sub_inputs = inputs['bilstm']
                        outputs = model(**sub_inputs, labels=inputs['labels'])
                    else:
                        raise ValueError(f"No matching input for model {model_name}. Available keys: {list(inputs.keys())}")
                else:
                    raise ValueError(f"No matching input for model {model_name}. Available keys: {list(inputs.keys())}")
                logits_list.append(outputs['logits'])
                successful_indices.append(i)
            except Exception as e:
                print(f"[Ensemble Warning] Model {i} failed: {e}")
                continue
        if not logits_list:
            raise RuntimeError("No models in ensemble could process the inputs")
        successful_weights = [self.weights[i] for i in successful_indices]
        total_weight = sum(successful_weights)
        normalized_weights = [w / total_weight for w in successful_weights]
        # Debug: print individual model predictions and logits (with mapped labels)
        if debug_predictions:
            with torch.no_grad():
                for i, (logits, weight) in enumerate(zip(logits_list, normalized_weights)):
                    preds = torch.argmax(logits, dim=-1)
                    sample_logits = logits[0] if logits.shape[0] > 0 else logits
                    pred_labels = self._idx_to_label(preds[:5])
                    print(f"[Ensemble Debug] Model {successful_indices[i]} predictions: {preds[:5]} ({pred_labels}), logits[0]: {sample_logits}, weight: {weight:.3f}")
        weighted_logits = torch.zeros_like(logits_list[0])
        for weight, logits in zip(normalized_weights, logits_list):
            weighted_logits += weight * logits
        result = {'logits': weighted_logits}
        if 'labels' in inputs and inputs['labels'] is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(weighted_logits, inputs['labels'])
            result['loss'] = loss
        # Debug: check if all predictions are for a single class (with mapped label)
        if debug_predictions:
            with torch.no_grad():
                preds = torch.argmax(weighted_logits, dim=-1)
                unique = torch.unique(preds)
                sample_weighted_logits = weighted_logits[0] if weighted_logits.shape[0] > 0 else weighted_logits
                pred_labels = self._idx_to_label(preds[:5])
                unique_labels = self._idx_to_label(unique)
                print(f"[Ensemble Debug] Final weighted logits[0]: {sample_weighted_logits}")
                print(f"[Ensemble Debug] Final ensemble predictions: {preds[:5]} ({pred_labels}), unique classes: {unique} ({unique_labels})")
                if unique.numel() == 1:
                    print(f"[Ensemble Warning] All ensemble predictions are for class {unique.item()} ({unique_labels[0]}) in this batch.")
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
    
    def predict(self, texts: Union[str, List[str]], return_probabilities: bool = False, debug_predictions: bool = False) -> Union[Dict, List[Dict]]:
        """
        Predict emotions for input text(s).
        
        Args:
            texts: Input text or list of texts
            return_probabilities: Whether to return class probabilities
            debug_predictions: Whether to print ensemble debug info (if supported)
            
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
        # --- ENSEMBLE-aware prediction batching ---
        if hasattr(self.model, 'models') and isinstance(self.model, nn.Module):
            # Always build tokenizers dict with both keys
            tokenizers = {
                'distilbert': None,
                'twitter-roberta': None
            }
            if hasattr(self.tokenizer, 'name_or_path'):
                if 'distilbert' in self.tokenizer.name_or_path.lower():
                    tokenizers['distilbert'] = self.tokenizer
                elif 'roberta' in self.tokenizer.name_or_path.lower():
                    tokenizers['twitter-roberta'] = self.tokenizer
                else:
                    # Fallback: assign to both
                    tokenizers['distilbert'] = self.tokenizer
                    tokenizers['twitter-roberta'] = self.tokenizer
            # Try to get vocab for BiLSTM if present
            vocab = getattr(self.model, 'vocab', None)
            try:
                from src.models import get_joint_ensemble_collate_fn
            except ImportError:
                from .models import get_joint_ensemble_collate_fn
            collate_fn = get_joint_ensemble_collate_fn(tokenizers, max_length=128, vocab=vocab)
            batch = [{'text': t, 'label': 0} for t in texts]  # dummy label
            batch_dict = collate_fn(batch)
            # Remove 'labels' key for prediction
            batch_dict = {k: v for k, v in batch_dict.items() if k != 'labels'}
            # Remove submodel keys where value is None
            batch_dict = {k: (None if v is None else {kk: vv.to(self.device) for kk, vv in v.items()}) for k, v in batch_dict.items() if v is not None}
            with torch.no_grad():
                if 'debug_predictions' in self.model.forward.__code__.co_varnames:
                    output = self.model(**batch_dict, debug_predictions=debug_predictions)
                else:
                    output = self.model(**batch_dict)
                logits = output['logits']
                probabilities = F.softmax(logits, dim=-1)
                for i, text in enumerate(texts):
                    predicted_class = torch.argmax(logits[i]).item()
                    predicted_emotion = self.label_names[predicted_class]
                    confidence = probabilities[i, predicted_class].item()
                    result = {
                        'text': text,
                        'emotion': predicted_emotion,
                        'confidence': confidence
                    }
                    if return_probabilities:
                        prob_dict = {
                            emotion: prob.item() 
                            for emotion, prob in zip(self.label_names, probabilities[i])
                        }
                        result['probabilities'] = prob_dict
                    results.append(result)
            return results[0] if is_single else results
        # --- Single model prediction ---
        with torch.no_grad():
            for text in texts:
                output = self.model(text=[text], debug_predictions=debug_predictions) if 'debug_predictions' in self.model.forward.__code__.co_varnames else self.model(text=[text])
                logits = output['logits']
                probabilities = F.softmax(logits, dim=-1)
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

def move_to_device(batch, device):
    """
    Recursively move all tensors in a batch (possibly nested dicts) to the specified device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch

def calibrate_ensemble_weights(ensemble_model, val_loader, device='cpu', metric='f1_macro', step=0.1, verbose=True):
    """
    Calibrate ensemble weights using grid search on a validation set.
    Args:
        ensemble_model: EnsembleEmotionModel instance
        val_loader: DataLoader for validation set
        device: Device to run evaluation
        metric: Metric to optimize ('f1_macro' or 'accuracy')
        step: Step size for grid search (e.g., 0.1)
        verbose: Print progress
    Returns:
        best_weights: List of calibrated weights
        best_score: Best metric score
    """
    n_models = len(ensemble_model.models)
    # Generate all possible weight combinations that sum to 1
    grid = [w for w in itertools.product(
        *( [ [i*step for i in range(int(1/step)+1)] ] * n_models ) )
        if abs(sum(w)-1.0) < 1e-6 and all(x > 0 for x in w)
    ]
    best_score = -1
    best_weights = None
    y_true_all, y_pred_all = [], []
    for weights in grid:
        ensemble_model.weights = list(weights)
        y_true, y_pred = [], []
        for batch in val_loader:
            # Move all tensors in batch to device, handle nested dicts for joint ensemble
            batch_on_device = move_to_device(batch, device)
            # Remove 'labels' from inputs, keep for y_true
            inputs = {k: v for k, v in batch_on_device.items() if k != 'labels'}
            labels = batch_on_device['labels']
            with torch.no_grad():
                logits = ensemble_model(**inputs)['logits']
                preds = torch.argmax(logits, dim=-1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
        if metric == 'f1_macro':
            score = f1_score(y_true, y_pred, average='macro')
        else:
            score = accuracy_score(y_true, y_pred)
        if verbose:
            print(f"[Calibration] Weights: {weights}, {metric}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_weights = list(weights)
    if verbose:
        print(f"[Calibration] Best weights: {best_weights}, best {metric}: {best_score:.4f}")
    ensemble_model.weights = best_weights
    return best_weights, best_score

class JointEnsembleDataset(Dataset):
    """
    Dataset for joint ensemble training. Returns raw text and label for each sample.
    Tokenization is handled in the collate_fn for each submodel.
    """
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}

def get_joint_ensemble_collate_fn(tokenizers, max_length=128, vocab=None):
    """
    Returns a collate_fn that tokenizes a batch for each submodel.
    tokenizers: dict with keys 'distilbert', 'twitter-roberta', etc.
    vocab: for bilstm, a vocab dict.
    """
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        batch_dict = {'labels': labels}
        # DistilBERT
        if 'distilbert' in tokenizers:
            enc = tokenizers['distilbert'](
                texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt'
            )
            batch_dict['distilbert'] = {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']}
        else:
            # Always provide the key for ensemble robustness
            batch_dict['distilbert'] = None
        # Twitter-RoBERTa
        if 'twitter-roberta' in tokenizers:
            enc = tokenizers['twitter-roberta'](
                texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt'
            )
            batch_dict['twitter-roberta'] = {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']}
        else:
            batch_dict['twitter-roberta'] = None
        # BiLSTM
        if vocab is not None:
            input_ids = []
            for text in texts:
                tokens = text.split()
                ids = [vocab.get(tok, vocab.get('<UNK>', 1)) for tok in tokens][:max_length]
                ids += [vocab.get('<PAD>', 0)] * (max_length - len(ids))
                input_ids.append(ids)
            batch_dict['bilstm'] = {'input_ids': torch.tensor(input_ids, dtype=torch.long)}
        else:
            batch_dict['bilstm'] = None

        return batch_dict
    return collate_fn
