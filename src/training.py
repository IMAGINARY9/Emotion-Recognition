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
                 class_weights: torch.Tensor = None):
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
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.gradient_clipping = gradient_clipping
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.emotion_names = emotion_names or ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.class_weights = class_weights
        
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
        
    def setup_logging(self):
        """Setup logging for training."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_data_loaders(self, 
                           train_texts: List[str], 
                           train_labels: List[int],
                           val_texts: List[str], 
                           val_labels: List[int],
                           batch_size: int = 32,
                           max_length: int = 128) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for training and validation.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            batch_size: Batch size
            max_length: Maximum sequence length
            
        Returns:
            Tuple of training and validation data loaders
        """
        # Create datasets
        train_dataset = EmotionDataset(train_texts, train_labels, self.tokenizer, max_length)
        val_dataset = EmotionDataset(val_texts, val_labels, self.tokenizer, max_length)
        # Use custom collate_fn if available (for BiLSTM/traditional models)
        train_collate_fn = getattr(train_dataset, 'get_collate_fn', lambda: None)()
        val_collate_fn = getattr(val_dataset, 'get_collate_fn', lambda: None)()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=train_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=val_collate_fn)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler=None, debug_predictions: bool = False) -> Tuple[float, float, float, Dict]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Tuple of average loss, accuracy, F1 score, and prediction distribution
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Joint ensemble: batch is a dict with submodel keys
            if any(k in batch for k in ['distilbert', 'twitter-roberta', 'bilstm']):
                for k in batch:
                    if isinstance(batch[k], dict):
                        for subk in batch[k]:
                            batch[k][subk] = batch[k][subk].to(self.device)
                labels = batch['labels'].to(self.device)
                # Only pass debug_predictions if model is ensemble
                if hasattr(self.model, 'is_ensemble') or 'Ensemble' in self.model.__class__.__name__:
                    outputs = self.model(**batch, debug_predictions=debug_predictions)
                else:
                    outputs = self.model(**batch)
            else:
                # Single model
                if self.tokenizer:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    if hasattr(self.model, 'is_ensemble') or 'Ensemble' in self.model.__class__.__name__:
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, debug_predictions=debug_predictions)
                    else:
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    if hasattr(self.model, 'is_ensemble') or 'Ensemble' in self.model.__class__.__name__:
                        outputs = self.model(input_ids=input_ids, labels=labels, debug_predictions=debug_predictions)
                    else:
                        outputs = self.model(input_ids=input_ids, labels=labels)
            
            # Use class weights if available
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(outputs['logits'], labels)
            else:
                loss = outputs['loss']
            logits = outputs['logits']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
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

    def validate(self, val_loader: DataLoader, debug_predictions: bool = False) -> Tuple[float, float, float, Dict]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of average loss, accuracy, F1 score, and detailed metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if any(k in batch for k in ['distilbert', 'twitter-roberta', 'bilstm']):
                    for k in batch:
                        if isinstance(batch[k], dict):
                            for subk in batch[k]:
                                batch[k][subk] = batch[k][subk].to(self.device)
                    labels = batch['labels'].to(self.device)
                    if hasattr(self.model, 'is_ensemble') or 'Ensemble' in self.model.__class__.__name__:
                        outputs = self.model(**batch, debug_predictions=debug_predictions)
                    else:
                        outputs = self.model(**batch)
                else:
                    if self.tokenizer:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        if hasattr(self.model, 'is_ensemble') or 'Ensemble' in self.model.__class__.__name__:
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, debug_predictions=debug_predictions)
                        else:
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    else:
                        input_ids = batch['input_ids'].to(self.device)
                        labels = batch['label'].to(self.device)
                        if hasattr(self.model, 'is_ensemble') or 'Ensemble' in self.model.__class__.__name__:
                            outputs = self.model(input_ids=input_ids, labels=labels, debug_predictions=debug_predictions)
                        else:
                            outputs = self.model(input_ids=input_ids, labels=labels)
                # Use class weights if available
                if self.class_weights is not None:
                    loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                    loss = loss_fct(outputs['logits'], labels)
                else:
                    loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        
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
                output_dict=True
            ),
            'prediction_distribution': pred_dist
        }
        
        return avg_loss, accuracy, f1, detailed_metrics
    
    def train(self, 
              train_texts: List[str] = None, 
              train_labels: List[int] = None,
              val_texts: List[str] = None, 
              val_labels: List[int] = None,
              num_epochs: int = 5,
              batch_size: int = 32,
              max_length: int = 128,
              patience: int = 3,
              save_best: bool = True,
              train_loader=None,
              val_loader=None,
              debug_predictions: bool = False) -> Dict:
        """
        Train the emotion recognition model.
        If train_loader/val_loader are provided, use them directly (for joint ensemble training).
        Otherwise, create DataLoaders from texts/labels (for individual training).
        """
        self.logger.info(f"Starting training with {num_epochs} epochs")
        if train_loader is None or val_loader is None:
            self.logger.info("Creating DataLoaders from texts/labels (individual training mode)")
            train_loader, val_loader = self.create_data_loaders(
                train_texts, train_labels, val_texts, val_labels, batch_size, max_length
            )
        else:
            self.logger.info("Using provided DataLoaders (joint ensemble mode)")
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
        for epoch in range(num_epochs):
            start_time = time.time()
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            # Training
            train_loss, train_acc, train_f1, train_pred_dist = self.train_epoch(train_loader, optimizer, scheduler, debug_predictions=debug_predictions)
            # Validation
            val_loss, val_acc, val_f1, detailed_metrics = self.validate(val_loader, debug_predictions=debug_predictions)
            val_pred_dist = detailed_metrics.get('prediction_distribution', {})
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
