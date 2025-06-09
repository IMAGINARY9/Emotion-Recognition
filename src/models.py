"""
Emotion recognition models.

This module provides various neural network models for emotion recognition
from social media text, including transformer-based models and traditional
neural networks with word embeddings.

Based on the Twitter emotion classification task with 6 emotions:
- sadness, joy, love, anger, fear, surprise
"""

from torch.utils.data import Dataset
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union # Added Union
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import itertools
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score # Added classification_report

class EmotionDataset(Dataset):
    """Dataset class for emotion recognition."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer=None, max_length: int = 128, vocab=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = vocab or {}
        
        # Build vocabulary if not provided and no tokenizer
        if not self.tokenizer and not vocab:
            self.build_vocab()
        
    def __len__(self):
        return len(self.texts)
    
    @property
    def vocab_size(self):
        """Return vocabulary size."""
        return len(self.vocab) if self.vocab else 0
    
    def build_vocab(self, min_freq: int = 2, max_vocab_size: int = 15000):
        """Build vocabulary from texts with frequency filtering."""
        from collections import Counter
        
        # Count token frequencies
        token_counts = Counter()
        for text in self.texts:
            tokens = text.lower().split()
            token_counts.update(tokens)
        
        # Start with special tokens
        vocab = {'<PAD>': 0, '<UNK>': 1}
        vocab_idx = 2
        
        # Add tokens that meet minimum frequency
        for token, count in token_counts.most_common():
            if count >= min_freq and len(vocab) < max_vocab_size:
                vocab[token] = vocab_idx
                vocab_idx += 1
            elif len(vocab) >= max_vocab_size:
                break
        
        self.vocab = vocab
        print(f"Built vocabulary with {len(vocab)} tokens (min_freq={min_freq})")
    
    def text_to_ids(self, text: str) -> List[int]:
        """Convert text to list of token IDs for traditional models."""
        tokens = text.lower().split()
        return [self.vocab.get(token, 1) for token in tokens[:self.max_length]]  # Use 1 for <UNK>
    
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
    Enhanced BiLSTM with multiple attention mechanisms and improved architecture
    for emotion recognition.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300, hidden_dim: int = 256,
                 num_layers: int = 3, num_labels: int = 6, dropout: float = 0.3,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 freeze_embeddings: bool = False,
                 bidirectional: bool = True, 
                 use_attention: bool = False, 
                 padding_idx: int = 0, 
                 loss_type: str = 'focal',
                 gamma: float = 2.0,
                 label_smoothing: float = 0.1,
                 vocab: Optional[dict] = None):
        if vocab_size is None or embedding_dim is None:
            raise ValueError("vocab_size and embedding_dim must not be None. Check your config and vocab loading.")
        if not isinstance(vocab_size, int) or not isinstance(embedding_dim, int):
            raise ValueError(f"vocab_size and embedding_dim must be integers, got {type(vocab_size)}, {type(embedding_dim)}")
        self.padding_idx = padding_idx
        super().__init__()
        self.vocab = vocab 
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.lstm_output_dim = lstm_output_dim  # Save for reference if needed
        self.num_labels = num_labels
        self.dropout_rate = dropout
        if self.use_attention:
            self.attention = SelfAttention(lstm_output_dim, dropout=dropout)
        self.fc = EnhancedClassifier(lstm_output_dim, num_labels, dropout=dropout)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding.weight' in name and self.embedding.weight.requires_grad:
                nn.init.xavier_uniform_(param)
            elif 'lstm.weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'lstm.weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'lstm.bias' in name:
                nn.init.constant_(param, 0.0)
                # Initialize forget gate bias to 1 for LSTM
                # LSTM bias is init as (b_ii|b_if|b_ig|b_io)
                # Hidden_dim is the size of each of these parts
                if hasattr(param, 'data'): # Check if param is a tensor
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.)

        # EnhancedClassifier might have its own _init_weights
        if hasattr(self.fc, '_init_weights') and callable(getattr(self.fc, '_init_weights')):
            self.fc._init_weights()
        elif isinstance(self.fc, nn.Linear):
            nn.init.xavier_uniform_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # input_ids: (batch_size, seq_len)
        # Create mask for padding tokens before embedding, if needed by attention or other parts
        # Mask is True for non-padded tokens, False for padded ones.
        attention_mask = (input_ids != self.padding_idx).float() # (batch_size, seq_len)

        embedded = self.dropout(self.embedding(input_ids))
        # embedded: (batch_size, seq_len, embedding_dim)

        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)

        if self.use_attention:
            # SelfAttention expects (batch_size, seq_len, features) and mask (batch_size, seq_len)
            # Assuming SelfAttention returns (batch_size, features_after_attention)
            processed_output = self.attention(lstm_out, attention_mask) 
        else:
            # If not using attention, use the final hidden state(s)
            if self.bidirectional:
                # Concatenate the final forward and backward hidden states from the last layer
                # hidden is (num_layers * 2, batch, hidden_dim)
                # Forward final: hidden[-2, :, :], Backward final: hidden[-1, :, :]
                processed_output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            else:
                # hidden is (num_layers * 1, batch, hidden_dim)
                processed_output = hidden[-1,:,:]
            processed_output = self.dropout(processed_output)
            # processed_output: (batch_size, hidden_dim * num_directions) or (batch_size, hidden_dim)

        logits = self.fc(processed_output)
        # logits: (batch_size, num_labels)

        loss = None
        # Loss calculation is typically handled by the trainer
        # If labels are provided and loss needs to be computed here (e.g. for FocalLoss which might be part of the model):
        # if labels is not None:
        #     if self.loss_type == 'focal': # Example, actual loss comp might be more complex
        #         loss_fct = FocalLoss(gamma=self.gamma, label_smoothing=self.label_smoothing) # Assuming FocalLoss is defined
        #         loss = loss_fct(logits, labels)
        #     else:
        #         loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        #         loss = loss_fct(logits, labels)

        return {"logits": logits, "loss": loss} 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class SelfAttention(nn.Module):
    """Improved self-attention mechanism for BiLSTM."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1, num_heads: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization for residual connection
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, lstm_output: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # lstm_output: (batch_size, seq_len, hidden_dim)
        # mask: (batch_size, seq_len), 1 for valid tokens, 0 for padding
        
        batch_size, seq_len, hidden_dim = lstm_output.shape
        
        # Compute Q, K, V
        Q = self.query(lstm_output)  # (batch_size, seq_len, hidden_dim)
        K = self.key(lstm_output)
        V = self.value(lstm_output)
        
        # Reshape for multi-head attention
        if self.num_heads > 1:
            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            # Now: (batch_size, num_heads, seq_len, head_dim)
        else:
            # Single head case
            Q = Q.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_dim)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)
        
        # Compute attention scores
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=Q.device))
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # attention_scores: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            # mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # context: (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        if self.num_heads > 1:
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        else:
            context = context.squeeze(1)
        
        # Output projection
        output = self.out_proj(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + lstm_output)
        
        # Global pooling with mask awareness
        if mask is not None:
            # Mask out padding tokens
            mask_expanded = mask.unsqueeze(-1).float()
            masked_output = output * mask_expanded
            
            # Compute weighted average
            sum_output = torch.sum(masked_output, dim=1)
            count_non_pad = torch.sum(mask_expanded, dim=1).clamp(min=1e-8)
            pooled_output = sum_output / count_non_pad
        else:
            # Simple mean pooling
            pooled_output = torch.mean(output, dim=1)
        
        return pooled_output

class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.main_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(input_dim // 2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, num_labels)
        )
        self.aux_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 3),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 3, num_labels)
        )
        self.confidence_estimator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
    def _init_weights(self):
        for layer in self.main_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
        for layer in self.aux_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
        for layer in self.confidence_estimator:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_logits = self.main_classifier(x)
        aux_logits = self.aux_classifier(x)
        logits = (main_logits + aux_logits) / 2
        return logits

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
        # Only lowercase top-level keys, do NOT replace underscores with dashes
        normalized_inputs = {k.lower(): v for k, v in inputs.items()}
        for i, model in enumerate(self.models):
            try:
                model_name = type(model).__name__.lower().replace('_', '-')
                # Map model_name to input key
                if 'distilbert' in model_name:
                    key = 'distilbert'
                elif 'roberta' in model_name or 'twitterroberta' in model_name or 'twitter-roberta' in model_name:
                    key = 'twitter-roberta'
                elif 'bilstm' in model_name:
                    key = 'bilstm'
                else:
                    raise ValueError(f"Unknown submodel type: {model_name}")
                if key in normalized_inputs and normalized_inputs[key] is not None:
                    sub_inputs = normalized_inputs[key]
                    outputs = model(**sub_inputs, labels=normalized_inputs.get('labels'))
                else:
                    raise ValueError(f"No matching input for model {model_name}. Available keys: {list(normalized_inputs.keys())}")
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
        if 'labels' in normalized_inputs and normalized_inputs['labels'] is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(weighted_logits, normalized_inputs['labels'])
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
            # If only one tokenizer is available, use it for both keys
            if self.tokenizer:
                if hasattr(self.tokenizer, 'name_or_path'):
                    if 'distilbert' in self.tokenizer.name_or_path.lower():
                        tokenizers['distilbert'] = self.tokenizer
                        tokenizers['twitter-roberta'] = self.tokenizer  # fallback
                    elif 'roberta' in self.tokenizer.name_or_path.lower():
                        tokenizers['twitter-roberta'] = self.tokenizer
                        tokenizers['distilbert'] = self.tokenizer  # fallback
                    else:
                        tokenizers['distilbert'] = self.tokenizer
                        tokenizers['twitter-roberta'] = self.tokenizer
                else:
                    tokenizers['distilbert'] = self.tokenizer
                    tokenizers['twitter-roberta'] = self.tokenizer
            vocab = getattr(self.model, 'vocab', None)
            try:
                from src.models import get_joint_ensemble_collate_fn
            except ImportError:
                from .models import get_joint_ensemble_collate_fn
            collate_fn = get_joint_ensemble_collate_fn(tokenizers, max_length=128, vocab=vocab)
            batch = [{'text': t, 'label': 0} for t in texts]  # dummy label
            batch_dict = collate_fn(batch)
            # Remove submodel keys where value is None
            batch_dict = {
                k: (
                    None if v is None else
                    {kk: vv.to(self.device) for kk, vv in v.items()} if isinstance(v, dict)
                    else v.to(self.device)
                )
                for k, v in batch_dict.items() if v is not None
            }
            # Do NOT remove 'labels' key; submodels expect it
            with torch.no_grad():
                if 'debug_predictions' in self.model.forward.__code__.co_varnames:
                    output = self.model(**batch_dict, debug_predictions=debug_predictions)
                else:
                    output = self.model(**batch_dict)
                logits = output['logits']
                probabilities = F.softmax(logits, dim=-1)
                # --- ENSEMBLE EXPLANATIONS ---
                # Collect submodel votes and explanations if possible
                ensemble_votes = {}
                ensemble_submodel_explanations = {}
                if hasattr(self.model, 'models'):
                    for i, submodel in enumerate(self.model.models):
                        submodel_name = type(submodel).__name__
                        # Try to get submodel input
                        sub_input = None
                        if 'distilbert' in submodel_name.lower() and 'distilbert' in batch_dict:
                            sub_input = batch_dict['distilbert']
                        elif ('roberta' in submodel_name.lower() or 'twitterroberta' in submodel_name.lower()) and 'twitter-roberta' in batch_dict:
                            sub_input = batch_dict['twitter-roberta']
                        elif 'bilstm' in submodel_name.lower() and 'bilstm' in batch_dict:
                            sub_input = batch_dict['bilstm']
                        if sub_input is not None:
                            sub_out = submodel(**sub_input)
                            sub_logits = sub_out['logits']
                            sub_probs = F.softmax(sub_logits, dim=-1)
                            sub_pred = torch.argmax(sub_logits, dim=-1).item()
                            sub_label = self.label_names[sub_pred]
                            ensemble_votes[submodel_name] = sub_label
                            # --- Submodel explanations ---
                            sub_expl = {}
                            # Word importances
                            if 'bilstm' in submodel_name.lower() and hasattr(submodel, 'use_attention') and submodel.use_attention:
                                # Try to extract attention weights
                                if hasattr(submodel, 'attention') and hasattr(submodel.attention, 'last_attention_weights'):
                                    attn = submodel.attention.last_attention_weights
                                    if attn is not None:
                                        tokens = batch[0]['text'].split()
                                        importances = attn[0].cpu().numpy().tolist()[:len(tokens)]
                                        sub_expl['word_importances'] = {'words': tokens, 'importances': importances}
                            # For transformers, try to extract attention weights
                            elif ('distilbert' in submodel_name.lower() or 'roberta' in submodel_name.lower()):
                                # Try to extract attention weights if available
                                attentions = None
                                tokens = None
                                if hasattr(submodel, 'distilbert') and hasattr(submodel.distilbert, 'config'):
                                    submodel.distilbert.config.output_attentions = True
                                if hasattr(submodel, 'roberta') and hasattr(submodel.roberta, 'config'):
                                    submodel.roberta.config.output_attentions = True
                                # Run with output_attentions=True
                                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                                    encoding = self.tokenizer(batch[0]['text'], return_tensors='pt', truncation=True)
                                    encoding = {k: v.to(self.device) for k, v in encoding.items()}
                                    if 'distilbert' in submodel_name.lower() and hasattr(submodel, 'distilbert'):
                                        out = submodel.distilbert(**encoding, output_attentions=True)
                                    elif 'roberta' in submodel_name.lower() and hasattr(submodel, 'roberta'):
                                        out = submodel.roberta(**encoding, output_attentions=True)
                                    else:
                                        out = None
                                    if out is not None and hasattr(out, 'attentions') and out.attentions is not None:
                                        attentions = out.attentions  # tuple: (num_layers, batch, num_heads, seq_len, seq_len)
                                        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
                                # Compute word importances from attentions
                                if attentions is not None and tokens is not None:
                                    # Average over all layers and heads, take attention from [CLS] (token 0) to each token
                                    attn = torch.stack(attentions)  # (num_layers, batch, num_heads, seq_len, seq_len)
                                    attn = attn[:, 0, :, 0, :]  # (num_layers, num_heads, seq_len)
                                    attn = attn.mean(dim=0).mean(dim=0)  # (seq_len,)
                                    importances = attn.cpu().numpy().tolist()
                                    # Remove special tokens if needed
                                    if tokens[0] in ['[CLS]', '<s>']:
                                        tokens = tokens[1:]
                                        importances = importances[1:]
                                    if tokens and tokens[-1] in ['[SEP]', '</s>']:
                                        tokens = tokens[:-1]
                                        importances = importances[:-1]
                                    sub_expl['word_importances'] = {'words': tokens, 'importances': importances}
                                else:
                                    # Fallback: uniform importances
                                    if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                                        tokens = self.tokenizer.tokenize(batch[0]['text'])
                                    else:
                                        tokens = batch[0]['text'].split()
                                    importances = [1.0 / len(tokens)] * len(tokens) if tokens else []
                                    sub_expl['word_importances'] = {'words': tokens, 'importances': importances}
                            # Confidence progression (simulate for BiLSTM)
                            if 'bilstm' in submodel_name.lower():
                                tokens = batch[0]['text'].split()
                                confidences = []
                                for j in range(1, len(tokens)+1):
                                    ids = [submodel.vocab.get(tok, submodel.vocab.get('<UNK>', 1)) for tok in tokens[:j]]
                                    ids += [submodel.vocab.get('<PAD>', 0)] * (submodel.embedding.num_embeddings - len(ids))
                                    ids_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
                                    out = submodel(input_ids=ids_tensor)
                                    probs = F.softmax(out['logits'], dim=-1)
                                    conf = probs[0].max().item()
                                    confidences.append(conf)
                                sub_expl['confidence_progression'] = {'tokens': tokens, 'confidences': confidences}
                            # Save submodel probabilities for ensemble plot
                            sub_expl['probabilities'] = {label: sub_probs[0, idx].item() for idx, label in enumerate(self.label_names)}
                            ensemble_submodel_explanations[submodel_name] = sub_expl
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
                    if ensemble_votes:
                        result['ensemble_votes'] = ensemble_votes
                    if ensemble_submodel_explanations:
                        result['ensemble_submodel_explanations'] = ensemble_submodel_explanations
                    results.append(result)
            return results[0] if is_single else results
        # --- Single model prediction ---
        with torch.no_grad():
            for text in texts:
                # --- BiLSTM path ---
                if 'bilstm' in self.model.__class__.__name__.lower():
                    # Tokenize to input_ids using vocab
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None and hasattr(self.tokenizer, 'encode'):
                        input_ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long, device=self.device)
                    else:
                        vocab = getattr(self.model, 'vocab', None)
                        if vocab is None:
                            raise ValueError("BiLSTM model requires a vocab for tokenization.")
                        tokens = text.split()
                        input_ids = torch.tensor([[vocab.get(token, vocab.get('<unk>', 0)) for token in tokens]], dtype=torch.long, device=self.device)
                    output = self.model(input_ids=input_ids)
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
                    tokens = text.split()
                    # --- BiLSTM attention word importances ---
                    if hasattr(self.model, 'use_attention') and self.model.use_attention and hasattr(self.model, 'attention') and hasattr(self.model.attention, 'last_attention_weights'):
                        attn = self.model.attention.last_attention_weights
                        if attn is not None:
                            importances = attn[0].cpu().numpy().tolist()[:len(tokens)]
                            result['word_importances'] = {'words': tokens, 'importances': importances}
                    else:
                        # Fallback: uniform importances
                        importances = [1.0 / len(tokens)] * len(tokens) if tokens else []
                        result['word_importances'] = {'words': tokens, 'importances': importances}
                    # --- BiLSTM confidence progression ---
                    confidences = []
                    for j in range(1, len(tokens)+1):
                        ids = [self.model.vocab.get(tok, self.model.vocab.get('<UNK>', 1)) for tok in tokens[:j]]
                        ids += [self.model.vocab.get('<PAD>', 0)] * (self.model.embedding.num_embeddings - len(ids))
                        ids_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
                        out = self.model(input_ids=ids_tensor)
                        probs = F.softmax(out['logits'], dim=-1)
                        conf = probs[0].max().item()
                        confidences.append(conf)
                    result['confidence_progression'] = {'tokens': tokens, 'confidences': confidences}
                    results.append(result)
                else:
                    # Transformer path (DistilBERT, RoBERTa, etc.)
                    # Try to extract attention weights if possible
                    # (requires output_attentions=True in config, not always available)
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
                    # --- Transformer attention word importances (placeholder: uniform) ---
                    tokens = text.split()
                    importances = [1.0 / len(tokens)] * len(tokens) if tokens else []
                    result['word_importances'] = {'words': tokens, 'importances': importances}
                    # Confidence progression for transformers is not directly available; skip for now
                    results.append(result)
        return results[0] if is_single else results

def create_model(model_type, model_config=None, vocab=None, **kwargs):
    """
    Factory function to create emotion recognition models.
    
    Args:
        model_type: Type of model ('distilbert', 'twitter-roberta', 'bilstm', 'ensemble')
        model_config: Configuration dictionary for the model
        
    Returns:
        Initialized model
    """
    # Remove 'type' and 'weight' keys if present to avoid passing them to model constructors
    if model_config:
        model_config = dict(model_config)  # Make a copy to avoid side effects
        model_config.pop('type', None)
        model_config.pop('weight', None)
        
    if model_type.lower() == "bilstm":
        vocab_size = None
        embedding_dim = None
        hidden_dim = None
        num_layers = 1
        num_labels = 6
        dropout = 0.3
        bidirectional = True
        use_attention = False
        pretrained_embeddings = None
        freeze_embeddings = False
        padding_idx = 0
        loss_type = 'focal'
        gamma = 2.0
        label_smoothing = 0.1
        # Extract from config if present
        if model_config is not None:
            # Try to get from model_config, fallback to main config if needed
            vocab_size = getattr(model_config, 'vocab_size', None) or (model_config.get('vocab_size', None) if isinstance(model_config, dict) else None)
            embedding_dim = getattr(model_config, 'embedding_dim', None) or (model_config.get('embedding_dim', None) if isinstance(model_config, dict) else None)
            if embedding_dim is None and 'embedding_dim' in kwargs:
                embedding_dim = kwargs['embedding_dim']
            hidden_dim = getattr(model_config, 'hidden_dim', None) or (model_config.get('hidden_dim', None) if isinstance(model_config, dict) else None)
            num_layers = getattr(model_config, 'num_layers', num_layers) or (model_config.get('num_layers', num_layers) if isinstance(model_config, dict) else num_layers)
            num_labels = getattr(model_config, 'num_labels', None) or model_config.get('num_labels', None)
            if num_labels is None:
                num_labels = getattr(model_config, 'num_classes', None) or model_config.get('num_classes', 6)
            dropout = getattr(model_config, 'dropout', None) or model_config.get('dropout', None) or getattr(model_config, 'dropout_rate', None) or model_config.get('dropout_rate', 0.3)
            bidirectional = getattr(model_config, 'bidirectional', bidirectional) or model_config.get('bidirectional', bidirectional)
            use_attention = getattr(model_config, 'attention', use_attention) or model_config.get('attention', use_attention)
            pretrained_embeddings = getattr(model_config, 'pretrained_embeddings', None) or model_config.get('pretrained_embeddings', None)
            freeze_embeddings = getattr(model_config, 'freeze', freeze_embeddings) or model_config.get('freeze', freeze_embeddings)
            padding_idx = getattr(model_config, 'padding_idx', padding_idx) or model_config.get('padding_idx', padding_idx)
            loss_type = getattr(model_config, 'loss_type', loss_type) or model_config.get('loss_type', loss_type)
            gamma = getattr(model_config, 'gamma', gamma) or model_config.get('gamma', gamma)
            label_smoothing = getattr(model_config, 'label_smoothing', label_smoothing) or model_config.get('label_smoothing', label_smoothing)
        if vocab_size is None and vocab is not None:
            vocab_size = len(vocab)
        if embedding_dim is None:
            embedding_dim = 300  # fallback default for BiLSTM
        if vocab_size is None:
            raise ValueError("vocab_size must be specified in the config or inferred from vocab for BiLSTM model.")
        return BiLSTMEmotionModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_labels=num_labels,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
            padding_idx=padding_idx,
            loss_type=loss_type,
            gamma=gamma,
            label_smoothing=label_smoothing,
            vocab=vocab,
            **kwargs
        )
    elif model_type == "distilbert":
        return DistilBERTEmotionModel(**model_config)
    elif model_type == "twitter-roberta":
        return TwitterRoBERTaEmotionModel(**model_config)
    elif model_type == "ensemble":
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
        if tokenizers and 'distilbert' in tokenizers and tokenizers['distilbert'] is not None:
            enc = tokenizers['distilbert'](
                texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt'
            )
            batch_dict['distilbert'] = {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']}
        else:
            batch_dict['distilbert'] = None
        # Twitter-RoBERTa
        if tokenizers and 'twitter-roberta' in tokenizers and tokenizers['twitter-roberta'] is not None:
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
