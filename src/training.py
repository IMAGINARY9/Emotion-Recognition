"""
Training utilities for emotion recognition models.

This module provides training loops, optimization strategies, and
model management for emotion recognition models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import time
from datetime import datetime

from models import EmotionDataset, create_model
from preprocessing import EmotionPreprocessor
from src.losses import FocalLoss
from enhanced_datasets import EnhancedBiLSTMDataset, JointEnsembleDataset
from utils.vocab_utils import BiLSTMVocabularyBuilder

class EmotionTrainer:
    """
    Trainer class for emotion recognition models.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 tokenizer = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 100,
                 gradient_clipping: float = 1.0,
                 save_dir: str = 'models',
                 emotion_names: List[str] = None,
                 class_weights: torch.Tensor = None,
                 loss_type: str = 'cross_entropy',
                 gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        """
        Initialize the trainer.
        
        Args:
            model: The emotion recognition model
            tokenizer: Tokenizer for text processing
            device: Device to use for training
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps for learning rate scheduling
            gradient_clipping: Maximum gradient norm for clipping
            save_dir: Directory to save models
            emotion_names: List of emotion names
            class_weights: Tensor of class weights for imbalanced data (computed from training set)
            loss_type: Loss function type ('cross_entropy' or 'focal')
            gamma: Focal loss gamma
            label_smoothing: Label smoothing for loss
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.gradient_clipping = gradient_clipping
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = None # Initialize scaler for AMP

        self.emotion_names = emotion_names or ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': []
        }
        
        # Setup logging
        self.setup_logging()
        
        # Loss function configuration
        self.loss_type = loss_type
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def setup_logging(self):
        """Setup logging for training (dedicated file handler per trainer instance)."""
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        # Unique log file per trainer instance
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = getattr(self.model, 'name', self.model.__class__.__name__)
        log_file = log_dir / f"training_{model_name}_{timestamp}.log"
        self.logger = logging.getLogger(f"EmotionTrainer.{id(self)}")
        self.logger.setLevel(logging.INFO)
        # Remove existing handlers to avoid duplicate logs
        self.logger.handlers = []
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        # Also add a stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        # Prevent log propagation to root logger
        self.logger.propagate = False
        self.logger.info(f"Logging started for EmotionTrainer. Log file: {log_file}")
    
    def create_data_loaders(self, 
                           train_texts: List[str], 
                           train_labels: List[int],
                           val_texts: List[str], 
                           val_labels: List[int],
                           batch_size: int = 32,
                           max_length: int = 128,
                           use_enhanced_bilstm: bool = False,
                           vocab_builder: Optional[BiLSTMVocabularyBuilder] = None,
                           vocab_save_path: Optional[str] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for training and validation.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            batch_size: Batch size
            max_length: Maximum sequence length
            use_enhanced_bilstm: Whether to use enhanced BiLSTM dataset
            vocab_builder: Vocabulary builder for BiLSTM
            vocab_save_path: Path to save vocabulary
            
        Returns:
            Tuple of training and validation data loaders
        """
        if use_enhanced_bilstm:
            self.logger.info("Using Enhanced BiLSTM Dataset with vocabulary builder")
            
            # Initialize vocabulary builder if not provided
            if vocab_builder is None:
                vocab_builder = BiLSTMVocabularyBuilder(
                    min_freq=2,
                    max_vocab_size=15000,
                    preserve_emotion_words=True
                )
            
            # Create training dataset (builds vocabulary)
            train_dataset = EnhancedBiLSTMDataset(
                texts=train_texts,
                labels=train_labels,
                vocab=None,  # Will build vocabulary
                max_length=max_length,
                vocab_builder=vocab_builder
            )
            
            # Get vocabulary from training dataset
            vocab = train_dataset.get_vocab()
            
            # Save vocabulary if path provided
            if vocab_save_path:
                vocab_builder.save_vocabulary(vocab, vocab_save_path)
                self.logger.info(f"Vocabulary saved to {vocab_save_path}")
            
            # Create validation dataset with same vocabulary
            val_dataset = EnhancedBiLSTMDataset(
                texts=val_texts,
                labels=val_labels,
                vocab=vocab,  # Use training vocabulary
                max_length=max_length,
                vocab_builder=vocab_builder
            )
            
            # Analyze vocabulary
            stats = vocab_builder.analyze_vocabulary(vocab, train_texts)
            self.logger.info(f"Vocabulary statistics: {stats}")
            
            # Store vocabulary for later use
            self.vocab = vocab
            self.vocab_builder = vocab_builder
            
        else:
            # Use traditional dataset
            train_dataset = EmotionDataset(train_texts, train_labels, self.tokenizer, max_length)
            val_dataset = EmotionDataset(val_texts, val_labels, self.tokenizer, max_length)
        
        # Use custom collate_fn if available (for BiLSTM/traditional models)
        train_collate_fn = getattr(train_dataset, 'get_collate_fn', lambda: None)()
        val_collate_fn = getattr(val_dataset, 'get_collate_fn', lambda: None)()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=train_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=val_collate_fn)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler=None, debug_predictions: bool = False, use_amp: bool = False) -> Tuple[float, float, float, Dict]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            use_amp: Whether to use Automatic Mixed Precision
            
        Returns:
            Tuple of average loss, accuracy, F1 score, and prediction distribution
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        if use_amp and self.scaler is None and self.device == 'cuda': # Initialize scaler only if using AMP and on CUDA
            self.scaler = torch.cuda.amp.GradScaler()

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            # Determine if it's a joint ensemble batch
            is_joint_ensemble_batch = any(k in batch for k in ['distilbert', 'twitter-roberta', 'bilstm'])

            if is_joint_ensemble_batch:
                for k in batch: # Move all tensor data to device
                    if isinstance(batch[k], dict):
                        for subk in batch[k]:
                            if isinstance(batch[k][subk], torch.Tensor):
                                batch[k][subk] = batch[k][subk].to(self.device)
                    elif isinstance(batch[k], torch.Tensor):
                         batch[k] = batch[k].to(self.device)
                labels = batch['labels'].to(self.device) # Ensure labels are on device
            else: # Single model batch
                labels = batch['label'].to(self.device)
                if self.tokenizer: # For HuggingFace models
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                else: # For models not using HuggingFace tokenizers (e.g. some BiLSTMs)
                    input_ids = batch['input_ids'].to(self.device)
              # Autocast for mixed precision
            with torch.autocast(device_type='cuda', enabled=(use_amp and self.device == 'cuda')):
                if is_joint_ensemble_batch:
                    # Only pass submodel keys and 'labels' to the ensemble model
                    model_inputs = {k: v for k, v in batch.items() if k in ['distilbert', 'twitter-roberta', 'bilstm', 'labels'] and v is not None}
                    if hasattr(self.model, 'is_ensemble') or 'Ensemble' in self.model.__class__.__name__:
                        outputs = self.model(**model_inputs, debug_predictions=debug_predictions)
                    else:
                        outputs = self.model(**model_inputs)
                else: # Single model
                    if self.tokenizer:
                        if hasattr(self.model, 'is_ensemble') or 'Ensemble' in self.model.__class__.__name__:
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, debug_predictions=debug_predictions)
                        else:
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    else: # Non-HF model (e.g. BiLSTM)
                        if hasattr(self.model, 'is_ensemble') or 'Ensemble' in self.model.__class__.__name__:
                            outputs = self.model(input_ids=input_ids, labels=labels, debug_predictions=debug_predictions)
                        else:
                            outputs = self.model(input_ids=input_ids, labels=labels)

                loss = outputs.get('loss') 
                logits = outputs['logits']

                if loss is None and labels is not None:
                    if self.loss_type == 'cross_entropy':
                        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, 
                                                      label_smoothing=self.label_smoothing if hasattr(self, 'label_smoothing') else 0.0)
                        loss = loss_fct(logits, labels)
                    elif self.loss_type == 'focal':
                        gamma = getattr(self, 'gamma', 2.0) 
                        label_smoothing = getattr(self, 'label_smoothing', 0.0)
                        loss_fct = FocalLoss(gamma=gamma, weight=self.class_weights, 
                                             label_smoothing=label_smoothing)
                        loss = loss_fct(logits, labels)
                    elif hasattr(self.model, 'loss_fct'):
                        loss = self.model.loss_fct(logits, labels)
                    else:
                        self.logger.error("Loss is None and could not be computed in trainer. Ensure model's forward returns loss or trainer's loss_type is correctly configured.")
                        # loss = torch.tensor(0.0, device=self.device, requires_grad=True) # Avoid crash

            # Backward pass
            if loss is not None:
                if use_amp and self.device == 'cuda':
                    self.scaler.scale(loss).backward()
                    if self.gradient_clipping > 0:
                        self.scaler.unscale_(optimizer) 
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else: # No AMP or not on CUDA
                    loss.backward()
                    if self.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    optimizer.step()
            
                if scheduler:
                    scheduler.step()
            
                total_loss += loss.item()
            else:
                if labels is not None: # Only warn if labels were present, implying loss should have been computed
                    self.logger.warning(f"Loss was None at batch {batch_idx}, backward pass skipped. Check model output and loss configuration.")
            
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Log prediction distribution
        unique, counts = np.unique(all_predictions, return_counts=True)
        pred_dist = dict(zip(unique, counts))
        self.logger.info(f"Prediction distribution (train): {pred_dist}")
        if len(pred_dist) == 1:
            self.logger.warning(f"All predictions in this epoch are for class {unique[0]}. Model may be collapsing.")
        
        return avg_loss, accuracy, f1, pred_dist

    def validate(self, val_loader: DataLoader, debug_predictions: bool = False, use_amp: bool = False) -> Tuple[float, float, float, Dict]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            use_amp: Whether to use Automatic Mixed Precision
            
        Returns:
            Tuple of average loss, accuracy, F1 score, and detailed metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                is_joint_ensemble_batch = any(k in batch for k in ['distilbert', 'twitter-roberta', 'bilstm'])

                if is_joint_ensemble_batch:
                    for k in batch: # Move all tensor data to device
                        if isinstance(batch[k], dict):
                            for subk in batch[k]:
                                if isinstance(batch[k][subk], torch.Tensor):
                                    batch[k][subk] = batch[k][subk].to(self.device)
                        elif isinstance(batch[k], torch.Tensor):
                            batch[k] = batch[k].to(self.device)
                    labels = batch['labels'].to(self.device) # Ensure labels are on device
                else: # Single model batch
                    labels = batch['label'].to(self.device)
                    if self.tokenizer: # For HuggingFace models
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                    else: # For models not using HuggingFace tokenizers
                        input_ids = batch['input_ids'].to(self.device)

                with torch.autocast(device_type='cuda', enabled=(use_amp and self.device == 'cuda')):
                    if is_joint_ensemble_batch:
                        model_inputs = {k: v for k, v in batch.items() if k != 'labels'}
                        if hasattr(self.model, 'is_ensemble') or 'Ensemble' in self.model.__class__.__name__:
                            outputs = self.model(**model_inputs, labels=labels, debug_predictions=debug_predictions)
                        else:
                            outputs = self.model(**model_inputs, labels=labels)
                    else: # Single model
                        if self.tokenizer:
                            if hasattr(self.model, 'is_ensemble') or 'Ensemble' in self.model.__class__.__name__:
                                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, debug_predictions=debug_predictions)
                            else:
                                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        else: # Non-HF model
                            if hasattr(self.model, 'is_ensemble') or 'Ensemble' in self.model.__class__.__name__:
                                outputs = self.model(input_ids=input_ids, labels=labels, debug_predictions=debug_predictions)
                            else:
                                outputs = self.model(input_ids=input_ids, labels=labels)
                    
                    logits = outputs['logits']
                    loss = outputs.get('loss')

                    if loss is None and labels is not None:
                        if self.loss_type == 'cross_entropy':
                            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, 
                                                          label_smoothing=self.label_smoothing if hasattr(self, 'label_smoothing') else 0.0)
                            loss = loss_fct(logits, labels)
                        elif self.loss_type == 'focal':
                            gamma = getattr(self, 'gamma', 2.0)
                            label_smoothing = getattr(self, 'label_smoothing', 0.0)
                            loss_fct = FocalLoss(gamma=gamma, weight=self.class_weights, 
                                                 label_smoothing=label_smoothing)
                            loss = loss_fct(logits, labels)
                        elif hasattr(self.model, 'loss_fct'):
                            loss = self.model.loss_fct(logits, labels)
                        else:
                            self.logger.error("Validation loss is None and could not be computed. Check model output and loss configuration.")
                            # loss = torch.tensor(0.0, device=self.device) # Avoid crash
                
                if loss is not None:
                    total_loss += loss.item()
                else:
                    if labels is not None: # Only warn if labels were present
                        self.logger.warning(f"Validation loss was None at batch {batch_idx}. Check model output and loss configuration.")

                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # Log prediction distribution
        unique, counts = np.unique(all_predictions, return_counts=True)
        pred_dist = dict(zip(unique, counts))
        self.logger.info(f"Prediction distribution (val): {pred_dist}")
        if len(pred_dist) == 1:
            self.logger.warning(f"All validation predictions are for class {unique[0]}. Model may be collapsing.")
        
        # Detailed classification report
        detailed_metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': classification_report(
                all_labels, all_predictions, 
                target_names=self.emotion_names,
                output_dict=True,
                zero_division=0
            ),
            'prediction_distribution': pred_dist
        }
        
        return avg_loss, accuracy, f1, detailed_metrics
    
    def train(self, 
              train_texts: List[str], 
              train_labels: List[int], 
              val_texts: List[str], 
              val_labels: List[int], 
              num_epochs: int, 
              batch_size: int, 
              max_length: int = 128,
              use_enhanced_bilstm: bool = False,
              vocab_builder: Optional[BiLSTMVocabularyBuilder] = None,
              vocab_save_path: Optional[str] = None,
              debug_predictions: bool = False,
              use_amp: bool = False): 
        """
        Train the model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            num_epochs: Number of epochs
            batch_size: Batch size
            max_length: Maximum sequence length
            use_enhanced_bilstm: Whether to use enhanced BiLSTM dataset
            vocab_builder: Vocabulary builder for BiLSTM
            vocab_save_path: Path to save vocabulary
            debug_predictions: Whether to log debug predictions
            use_amp: Whether to use Automatic Mixed Precision
        """
        self.logger.info(f"Starting training on {self.device} with learning rate {self.learning_rate}")
        
        # Determine actual use_amp based on device
        actual_use_amp = use_amp and (self.device == 'cuda')
        if self.device == 'cuda':
            self.logger.info(f"Automatic Mixed Precision (AMP) enabled: {actual_use_amp}")
        else:
            self.logger.info(f"Automatic Mixed Precision (AMP) enabled: False (AMP is CUDA only or was disabled in config)")


        train_loader, val_loader = self.create_data_loaders(
            train_texts, train_labels, val_texts, val_labels, batch_size, max_length, use_enhanced_bilstm, vocab_builder, vocab_save_path
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )        
        best_val_f1 = 0
        patience_counter = 0
        collapse_recovery_attempts = 0
        max_recovery_attempts = 3
        
        for epoch in range(num_epochs):
            start_time = time.time()
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss, train_acc, train_f1, _ = self.train_epoch(train_loader, optimizer, scheduler, debug_predictions=debug_predictions, use_amp=actual_use_amp)
            val_loss, val_acc, val_f1, _ = self.validate(val_loader, debug_predictions=debug_predictions, use_amp=actual_use_amp)
            end_time = time.time()
            
            # Detect model collapse on training and validation
            train_all_labels = []
            train_all_predictions = []
            for batch in train_loader:
                if self.tokenizer:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    labels = batch['label'].to(self.device)
                    with torch.no_grad():
                        if attention_mask is not None:
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        else:
                            outputs = self.model(input_ids=input_ids)
                        predictions = torch.argmax(outputs['logits'], dim=-1)
                        train_all_predictions.extend(predictions.cpu().numpy())
                        train_all_labels.extend(labels.cpu().numpy())
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids)
                        predictions = torch.argmax(outputs['logits'], dim=-1)
                        train_all_predictions.extend(predictions.cpu().numpy())
                        train_all_labels.extend(labels.cpu().numpy())
            
            # Check for collapse on training data
            train_collapse_info = self.detect_model_collapse(train_all_predictions, train_all_labels)
            
            # Get validation predictions for collapse detection
            val_all_labels = []
            val_all_predictions = []
            for batch in val_loader:
                if self.tokenizer:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    labels = batch['label'].to(self.device)
                    with torch.no_grad():
                        if attention_mask is not None:
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        else:
                            outputs = self.model(input_ids=input_ids)
                        predictions = torch.argmax(outputs['logits'], dim=-1)
                        val_all_predictions.extend(predictions.cpu().numpy())
                        val_all_labels.extend(labels.cpu().numpy())
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids)
                        predictions = torch.argmax(outputs['logits'], dim=-1)
                        val_all_predictions.extend(predictions.cpu().numpy())
                        val_all_labels.extend(labels.cpu().numpy())
            
            val_collapse_info = self.detect_model_collapse(val_all_predictions, val_all_labels)
            
            # Log collapse detection results
            if train_collapse_info['collapse_detected']:
                self.logger.warning(f"Model collapse detected in training! Active classes: {train_collapse_info['active_classes']}/{train_collapse_info['total_classes']}")
                self.logger.warning(f"Dominant class: {self.emotion_names[train_collapse_info['dominant_class']]} ({train_collapse_info['dominant_class_proportion']:.2%})")
                
            if val_collapse_info['collapse_detected']:
                self.logger.warning(f"Model collapse detected in validation! Active classes: {val_collapse_info['active_classes']}/{val_collapse_info['total_classes']}")
                
                # Apply recovery if collapse is severe and we haven't reached max attempts
                if (val_collapse_info['active_classes'] <= 2 and 
                    collapse_recovery_attempts < max_recovery_attempts):
                    
                    recovery_type = ['lr_reset', 'lr_boost', 'weight_noise'][collapse_recovery_attempts % 3]
                    self.apply_collapse_recovery(optimizer, recovery_type)
                    collapse_recovery_attempts += 1
                    
                    # Reset patience counter for recovery attempt
                    patience_counter = max(0, patience_counter - 2)
                    
                    self.logger.info(f"Collapse recovery attempt {collapse_recovery_attempts}/{max_recovery_attempts}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                if save_best:
                    self.save_model("best_model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        return self.history
    
    def save_model(self, model_name: str = None):
        """
        Save the trained model.
        
        Args:
            model_name: Name for the saved model or full path to save file
        """
        from pathlib import Path
        import os
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"emotion_model_{timestamp}"
            model_path = self.save_dir / f"{model_name}.pt"
        else:
            # If model_name looks like a path (endswith .pt or contains os.sep), treat as path
            if model_name.endswith('.pt') or os.sep in model_name or '/' in model_name:
                model_path = Path(model_name)
            else:
                model_path = self.save_dir / f"{model_name}.pt"
        # Ensure parent directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Save model state dict and metadata
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': getattr(self.model, 'config', None),
            'emotion_names': self.emotion_names,
            'training_history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(save_dict, model_path)
        # Save tokenizer if available
        if self.tokenizer:
            tokenizer_path = model_path.parent / f"{model_path.stem}_tokenizer"
            self.tokenizer.save_pretrained(tokenizer_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.history = checkpoint['training_history']
        
        if 'emotion_names' in checkpoint:
            self.emotion_names = checkpoint['emotion_names']
        
        self.logger.info(f"Model loaded from {model_path}")
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', marker='s')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_accuracy'], label='Train Accuracy', marker='o')
        axes[0, 1].plot(self.history['val_accuracy'], label='Validation Accuracy', marker='s')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.history['train_f1'], label='Train F1', marker='o')
        axes[1, 0].plot(self.history['val_f1'], label='Validation F1', marker='s')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        axes[1, 1].text(0.5, 0.5, 'Training Completed', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=16)
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def detect_model_collapse(self, predictions: List[int], labels: List[int], 
                             threshold: float = 0.05) -> Dict[str, any]:
        """
        Detect if the model is collapsing (predicting only one or few classes).
        
        Args:
            predictions: Model predictions
            labels: True labels
            threshold: Minimum proportion for a class to be considered active
            
        Returns:
            Dictionary with collapse detection results
        """
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        
        total_predictions = len(predictions)
        pred_proportions = pred_counts / total_predictions
        
        # Check for collapse
        active_classes = np.sum(pred_proportions >= threshold)
        total_classes = len(self.emotion_names)
        
        collapse_detected = active_classes <= max(1, total_classes * 0.3)  # Less than 30% of classes active
        
        # Calculate per-class metrics
        class_metrics = {}
        for i, emotion in enumerate(self.emotion_names):
            class_preds = np.sum(np.array(predictions) == i)
            class_labels = np.sum(np.array(labels) == i)
            
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            
            if class_preds > 0:
                precision = np.sum((np.array(predictions) == i) & (np.array(labels) == i)) / class_preds
            if class_labels > 0:
                recall = np.sum((np.array(predictions) == i) & (np.array(labels) == i)) / class_labels
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            class_metrics[emotion] = {
                'predictions': int(class_preds),
                'true_labels': int(class_labels),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
        
        return {
            'collapse_detected': collapse_detected,
            'active_classes': int(active_classes),
            'total_classes': int(total_classes),
            'dominant_class': int(unique_preds[np.argmax(pred_counts)]),
            'dominant_class_proportion': float(np.max(pred_proportions)),
            'class_metrics': class_metrics,
            'prediction_distribution': dict(zip(unique_preds.astype(int), pred_counts.astype(int)))
        }
    
    def apply_collapse_recovery(self, optimizer, recovery_type: str = 'lr_reset'):
        """
        Apply recovery strategies when model collapse is detected.
        
        Args:
            optimizer: Current optimizer
            recovery_type: Type of recovery strategy
        """
        self.logger.warning(f"Applying collapse recovery strategy: {recovery_type}")
        
        if recovery_type == 'lr_reset':
            # Reset learning rate to initial value
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        elif recovery_type == 'lr_boost':
            # Temporarily increase learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 2.0, self.learning_rate * 5)
                
        elif recovery_type == 'weight_noise':
            # Add small noise to model weights
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * 0.01
                        param.add_(noise)
        
        self.logger.info(f"Recovery strategy '{recovery_type}' applied")

def train_model_from_config(config_path: str, data_path: str, save_dir: str, debug_predictions: bool = False): # Removed use_amp from signature
    """
    Train a model based on a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.
        data_path: Path to the CSV data file.
        save_dir: Directory to save the trained model and logs.
        debug_predictions: Whether to log debug predictions.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get AMP flag from config, default to False if not specified
    use_amp_from_config = config['training'].get('mixed_precision', False)
    device_from_config = config['training'].get('device', 'cpu')


    # Setup logging for the main training script
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_file = log_dir / f"main_training_script_{timestamp}.log"
    
    # Configure root logger for this script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler() # Also log to console
        ]
    )
    logger = logging.getLogger(__name__) # Logger for this script
    logger.info(f"Main training script logging started. Log file: {main_log_file}")
    
    if device_from_config == 'cuda':
        logger.info(f"Configured to use Automatic Mixed Precision: {use_amp_from_config}")
    else:
        logger.info(f"Configured to use Automatic Mixed Precision: False (AMP is CUDA only or device is not CUDA)")


    # Load data
    logger.info(f"Loading data from {data_path}")
    # ...existing code...

    # Initialize trainer
    trainer = EmotionTrainer(
        model=model,
        tokenizer=tokenizer, # Pass tokenizer if it exists
        device=config['training']['device'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        gradient_clipping=config['training']['gradient_clip_norm'],
        save_dir=save_dir,
        emotion_names=config['data']['emotion_names'],
        class_weights=class_weights_tensor,
        loss_type=config['training'].get('loss_type', 'cross_entropy'), # Default to cross_entropy
        gamma=config['training'].get('focal_loss_gamma', 2.0), # Default gamma for focal loss
        label_smoothing=config['training'].get('label_smoothing', 0.0) # Default label smoothing
    )

    # Train model
    logger.info("Starting model training...")
    trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        max_length=config['data']['max_length'],
        use_enhanced_bilstm=use_enhanced_bilstm,
        vocab_builder=vocab_builder if use_enhanced_bilstm else None,
        vocab_save_path=vocab_save_path if use_enhanced_bilstm else None,
        debug_predictions=debug_predictions,
        use_amp=use_amp_from_config # Pass AMP flag from config
    )
def train_ensemble_from_config(config_path: str, data_path: str, save_dir: str, debug_predictions: bool = False): # Removed use_amp from signature
    """
    Train an ensemble model based on a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file for the ensemble.
        data_path: Path to the CSV data file.
        save_dir: Directory to save the trained ensemble model and logs.
        debug_predictions: Whether to log debug predictions.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get AMP flag from config, default to False if not specified
    use_amp_from_config = config['training'].get('mixed_precision', False)
    device_from_config = config['training'].get('device', 'cpu')

    # Setup logging for the main training script
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_file = log_dir / f"ensemble_training_script_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Ensemble training script logging started. Log file: {main_log_file}")
    if device_from_config == 'cuda':
        logger.info(f"Configured to use Automatic Mixed Precision: {use_amp_from_config}")
    else:
        logger.info(f"Configured to use Automatic Mixed Precision: False (AMP is CUDA only or device is not CUDA)")


    # Load data
    logger.info(f"Loading data from {data_path}")
    # ...existing code...

    # Initialize trainer
    trainer = EmotionTrainer(
        model=ensemble_model,
        device=config['training']['device'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        gradient_clipping=config['training']['gradient_clip_norm'],
        save_dir=save_dir,
        emotion_names=config['data']['emotion_names'],
        class_weights=class_weights_tensor,
        loss_type=config['training'].get('loss_type', 'cross_entropy'),
        gamma=config['training'].get('focal_loss_gamma', 2.0),
        label_smoothing=config['training'].get('label_smoothing', 0.0)
    )

    # Train ensemble model
    logger.info("Starting ensemble model training...")
    trainer.train(
        train_texts=train_texts, 
        train_labels=train_labels, 
        val_texts=val_texts, 
        val_labels=val_labels, 
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'], 
        max_length=config['data']['max_length'], 
        debug_predictions=debug_predictions,
        use_amp=use_amp_from_config # Pass AMP flag from config
    )

if __name__ == '__main__':
    import argparse
    import yaml # Ensure yaml is imported

    parser = argparse.ArgumentParser(description="Train an emotion recognition model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument('--save_dir', type=str, default='models/saved_models', help="Directory to save trained models and logs.")
    parser.add_argument('--ensemble', action='store_true', help="Train an ensemble model.")
    parser.add_argument('--debug_predictions', action='store_true', help="Log debug predictions during training and validation.")
    # Removed --use_amp argument, as it's now controlled by the config file's 'mixed_precision' flag.

    args = parser.parse_args()

    # Create save directory if it doesn't exist
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.ensemble:
        # Pass debug_predictions; use_amp is now handled internally based on config
        train_ensemble_from_config(args.config, args.data_path, args.save_dir, args.debug_predictions)
    else:
        # Pass debug_predictions; use_amp is now handled internally based on config
        train_model_from_config(args.config, args.data_path, args.save_dir, args.debug_predictions)
