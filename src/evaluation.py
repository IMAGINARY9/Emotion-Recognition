"""
Evaluation utilities for emotion recognition models.

This module provides comprehensive evaluation capabilities including
metrics calculation, visualization, error analysis, and model comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path
from datetime import datetime
import logging

from models import EmotionDataset

class EmotionEvaluator:
    """
    Comprehensive evaluator for emotion recognition models.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 tokenizer=None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 emotion_names: List[str] = None,
                 save_dir: str = None):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained emotion recognition model
            tokenizer: Tokenizer for text processing
            device: Device to use for evaluation
            emotion_names: List of emotion names
            save_dir: Directory to save evaluation reports
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.emotion_names = emotion_names or ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        model_name = getattr(model, 'name', model.__class__.__name__)
        # Determine default save_dir based on model's directory if not provided
        if save_dir is None:
            # Try to get model directory from model attribute or state_dict path
            model_dir = None
            if hasattr(model, 'model_dir') and model.model_dir:
                model_dir = Path(model.model_dir)
            elif hasattr(model, 'save_dir') and model.save_dir:
                model_dir = Path(model.save_dir)
            elif hasattr(model, 'state_dict_path') and model.state_dict_path:
                model_dir = Path(model.state_dict_path).parent
            elif hasattr(model, 'name_or_path'):
                model_dir = Path(model.name_or_path)
            # Fallback: use model.name or class name
            if model_dir and model_dir.exists():
                save_dir = Path('reports') / model_dir.name
            else:
                save_dir = Path('reports') / model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.model.eval()
    
    def evaluate_model(self, 
                      test_texts: List[str], 
                      test_labels: List[int],
                      batch_size: int = 32,
                      max_length: int = 128,
                      debug_predictions: bool = False) -> Dict:
        """
        Comprehensive evaluation of the model.
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
            debug_predictions: Whether to enable debug predictions (if supported by the model)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        self.logger.info(f"Evaluating model on {len(test_texts)} test samples")
        
        # Get predictions
        predictions, probabilities, all_labels = self._get_predictions(
            test_texts, test_labels, batch_size, max_length, debug_predictions=debug_predictions
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, predictions, probabilities)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, predictions)
        metrics['confusion_matrix'] = cm
        
        # Classification report
        class_report = classification_report(
            all_labels, predictions,
            target_names=self.emotion_names,
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = class_report
        
        # Error analysis
        error_analysis = self._perform_error_analysis(
            test_texts, all_labels, predictions, probabilities
        )
        metrics['error_analysis'] = error_analysis
        
        self.logger.info(f"Evaluation completed. Overall accuracy: {metrics['accuracy']:.4f}")

        # --- Always generate and save reports and plots ---
        # Save JSON report and human-readable summary
        model_name = getattr(self.model, 'name', self.model.__class__.__name__)
        report_path = self.generate_evaluation_report(metrics, test_texts, test_labels, model_name=model_name)
        self.logger.info(f"Saved JSON evaluation report to: {report_path}")
        # Save confusion matrix plot
        cm_path = self.save_dir / f"confusion_matrix_{model_name}.png"
        self.plot_confusion_matrix(metrics['confusion_matrix'], save_path=str(cm_path))
        self.logger.info(f"Saved confusion matrix plot to: {cm_path}")
        # Save per-class metrics plot
        pcm_path = self.save_dir / f"per_class_metrics_{model_name}.png"
        self.plot_per_class_metrics(metrics, save_path=str(pcm_path))
        self.logger.info(f"Saved per-class metrics plot to: {pcm_path}")
        # Save t-SNE embeddings plot
        tsne_path = self.save_dir / f"embeddings_tsne_{model_name}.png"
        self.visualize_embeddings(test_texts, test_labels, save_path=str(tsne_path))
        self.logger.info(f"Saved t-SNE embeddings plot to: {tsne_path}")
        # Optionally: add more plots as needed
        
        return metrics
    
    def _get_predictions(self, 
                        texts: List[str], 
                        labels: List[int],
                        batch_size: int = 32,
                        max_length: int = 128,
                        debug_predictions: bool = False) -> Tuple[List[int], List[List[float]], List[int]]:
        """
        Get model predictions for the given texts.
        
        Returns:
            Tuple of predictions, probabilities, and labels
        """
        # Create dataset and dataloader
        dataset = EmotionDataset(texts, labels, self.tokenizer, max_length)
        collate_fn = getattr(dataset, 'get_collate_fn', lambda: None)()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels_batch = batch['label'].to(self.device)
                # Only pass attention_mask if model expects it (transformers)
                if self.tokenizer and ('distilbert' in self.model.__class__.__name__.lower() or 'roberta' in self.model.__class__.__name__.lower()):
                    attention_mask = batch['attention_mask'].to(self.device)
                    if hasattr(self.model, 'forward') and 'debug_predictions' in self.model.forward.__code__.co_varnames:
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, debug_predictions=debug_predictions)
                    else:
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    if hasattr(self.model, 'forward') and 'debug_predictions' in self.model.forward.__code__.co_varnames:
                        outputs = self.model(input_ids=input_ids, debug_predictions=debug_predictions)
                    else:
                        outputs = self.model(input_ids=input_ids)
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
        
        return all_predictions, all_probabilities, all_labels
    
    def _calculate_metrics(self, 
                          labels: List[int], 
                          predictions: List[int], 
                          probabilities: List[List[float]]) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['precision'] = precision_score(labels, predictions, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(labels, predictions, average='weighted')
        metrics['f1'] = f1_score(labels, predictions, average='weighted')
        
        # Per-class metrics
        metrics['per_class_precision'] = precision_score(labels, predictions, average=None, zero_division=0)
        metrics['per_class_recall'] = recall_score(labels, predictions, average=None)
        metrics['per_class_f1'] = f1_score(labels, predictions, average=None)
        
        # Macro and micro averages
        metrics['macro_f1'] = f1_score(labels, predictions, average='macro')
        metrics['micro_f1'] = f1_score(labels, predictions, average='micro')
        
        # Calculate AUC for multiclass (one-vs-rest)
        try:
            probabilities_array = np.array(probabilities)
            metrics['auc_ovr'] = roc_auc_score(labels, probabilities_array, multi_class='ovr')
            metrics['auc_ovo'] = roc_auc_score(labels, probabilities_array, multi_class='ovo')
        except Exception as e:
            self.logger.warning(f"Could not calculate AUC: {e}")
            metrics['auc_ovr'] = None
            metrics['auc_ovo'] = None
        
        return metrics
    
    def _perform_error_analysis(self, 
                               texts: List[str], 
                               labels: List[int], 
                               predictions: List[int], 
                               probabilities: List[List[float]]) -> Dict:
        """Perform detailed error analysis."""
        error_analysis = {}
        
        # Find misclassified examples
        misclassified_indices = [i for i, (true, pred) in enumerate(zip(labels, predictions)) if true != pred]
        
        error_analysis['num_misclassified'] = len(misclassified_indices)
        error_analysis['misclassification_rate'] = len(misclassified_indices) / len(labels)
        
        # Analyze confidence of misclassified examples
        if misclassified_indices:
            misclassified_confidences = [max(probabilities[i]) for i in misclassified_indices]
            error_analysis['avg_misclassified_confidence'] = np.mean(misclassified_confidences)
            error_analysis['misclassified_confidence_std'] = np.std(misclassified_confidences)
        
        # Most confident wrong predictions
        wrong_confidences = [(i, max(probabilities[i])) for i in misclassified_indices]
        wrong_confidences.sort(key=lambda x: x[1], reverse=True)
        
        error_analysis['most_confident_errors'] = []
        for i, confidence in wrong_confidences[:10]:  # Top 10
            error_analysis['most_confident_errors'].append({
                'text': texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                'true_emotion': self.emotion_names[int(labels[i])],
                'predicted_emotion': self.emotion_names[int(predictions[i])],
                'confidence': confidence
            })
        
        # Confusion patterns
        cm = confusion_matrix(labels, predictions)
        confusion_patterns = {}
        for i in range(len(self.emotion_names)):
            for j in range(len(self.emotion_names)):
                if i != j and cm[i, j] > 0:
                    pattern = f"{self.emotion_names[i]} -> {self.emotion_names[j]}"
                    confusion_patterns[pattern] = cm[i, j]
        
        # Sort by frequency
        error_analysis['confusion_patterns'] = dict(
            sorted(confusion_patterns.items(), key=lambda x: x[1], reverse=True)
        )
        
        return error_analysis
    
    def plot_confusion_matrix(self, 
                             confusion_matrix: np.ndarray, 
                             save_path: str = None,
                             normalize: bool = True) -> None:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            save_path: Path to save the plot
            normalize: Whether to normalize the matrix
        """
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            title = 'Confusion Matrix'
            fmt = '.2f'
        else:
            cm = confusion_matrix
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, 
                   annot=True, 
                   fmt=fmt, 
                   cmap='Blues',
                   xticklabels=self.emotion_names,
                   yticklabels=self.emotion_names)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_per_class_metrics(self, metrics: Dict, save_path: str = None) -> None:
        """Plot per-class performance metrics."""
        per_class_metrics = {
            'Precision': metrics['per_class_precision'],
            'Recall': metrics['per_class_recall'],
            'F1-Score': metrics['per_class_f1']
        }
        
        x = np.arange(len(self.emotion_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (metric_name, values) in enumerate(per_class_metrics.items()):
            ax.bar(x + i * width, values, width, label=metric_name, alpha=0.8)
        
        ax.set_xlabel('Emotions', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.emotion_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (metric_name, values) in enumerate(per_class_metrics.items()):
            for j, v in enumerate(values):
                ax.text(j + i * width, v + 0.01, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_embeddings(self, 
                           texts: List[str], 
                           labels: List[int],
                           sample_size: int = 1000,
                           save_path: str = None) -> None:
        """
        Visualize text embeddings using t-SNE.
        
        Args:
            texts: Input texts
            labels: True labels
            sample_size: Number of samples to visualize
            save_path: Path to save the plot
        """
        self.logger.info("Generating embedding visualization...")
        # Sample data if needed
        if len(texts) > sample_size:
            import numpy as np
            idx = np.random.choice(len(texts), sample_size, replace=False)
            texts = [texts[i] for i in idx]
            labels = [labels[i] for i in idx]
        # Extract embeddings
        embeddings = self._extract_embeddings(texts)
        if embeddings.size == 0:
            self.logger.warning("No embeddings extracted; skipping embedding visualization.")
            return
        
        # Apply t-SNE
        scaler = MinMaxScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings_scaled)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.emotion_names)))
        
        # Plot each emotion separately
        for i, emotion in enumerate(self.emotion_names):
            emotion_indices = [j for j, label in enumerate(labels) if label == i]
            if emotion_indices:
                emotion_embeddings = embeddings_2d[emotion_indices]
                axes[i].scatter(emotion_embeddings[:, 0], emotion_embeddings[:, 1], 
                              c=[colors[i]], alpha=0.6, s=20)
            axes[i].set_title(f'{emotion.capitalize()}', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('t-SNE Visualization of Text Embeddings by Emotion', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def _extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings from the model."""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                if self.tokenizer:
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=128,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    
                    # Get hidden states (assuming transformer model)
                    if hasattr(self.model, 'distilbert'):
                        outputs = self.model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
                    elif hasattr(self.model, 'roberta'):
                        outputs = self.model.roberta(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        # Fallback to model forward without labels
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        if 'logits' in outputs:
                            # Use logits as embeddings for visualization
                            embeddings.append(outputs['logits'].cpu().numpy().flatten())
                            continue
                    
                    # Use [CLS] token embedding
                    cls_embedding = outputs.last_hidden_state[:, 0]
                    embeddings.append(cls_embedding.cpu().numpy().flatten())
        
        return np.array(embeddings)
    
    def generate_evaluation_report(self, 
                                  metrics: Dict, 
                                  test_texts: List[str], 
                                  test_labels: List[int],
                                  model_name: str = "EmotionModel") -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics: Evaluation metrics dictionary
            test_texts: Test texts
            test_labels: Test labels
            model_name: Name of the model
            
        Returns:
            Path to the generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.save_dir / f"evaluation_report_{model_name}_{timestamp}.json"
        
        # Prepare report data
        report_data = {
            'model_name': model_name,
            'evaluation_timestamp': timestamp,
            'dataset_size': len(test_texts),
            'emotion_names': list(self.emotion_names),
            'metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'macro_f1': float(metrics['macro_f1']),
                'micro_f1': float(metrics['micro_f1']),
                'auc_ovr': float(metrics['auc_ovr']) if metrics['auc_ovr'] is not None else None,
                'auc_ovo': float(metrics['auc_ovo']) if metrics['auc_ovo'] is not None else None
            },
            'per_class_metrics': {
                str(emotion): {
                    'precision': float(metrics['per_class_precision'][i]),
                    'recall': float(metrics['per_class_recall'][i]),
                    'f1': float(metrics['per_class_f1'][i])
                }
                for i, emotion in enumerate(list(self.emotion_names))
            },
            'error_analysis': metrics['error_analysis'],
            'classification_report': metrics['classification_report']
        }
        
        def to_serializable(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            elif isinstance(obj, np.generic):
                return obj.item()
            else:
                return obj
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(to_serializable(report_data), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Evaluation report saved to {report_path}")
        
        # Also create a human-readable summary
        summary_path = self.save_dir / f"evaluation_summary_{model_name}_{timestamp}.txt"
        self._create_text_summary(report_data, summary_path)
        
        return str(report_path)
    
    def _create_text_summary(self, report_data: Dict, summary_path: Path):
        """Create a human-readable summary of the evaluation."""
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Emotion Recognition Model Evaluation Report\n")
            f.write(f"=" * 50 + "\n\n")
            
            f.write(f"Model: {report_data['model_name']}\n")
            f.write(f"Evaluation Date: {report_data['evaluation_timestamp']}\n")
            f.write(f"Dataset Size: {report_data['dataset_size']} samples\n\n")
            
            f.write(f"Overall Performance:\n")
            f.write(f"  Accuracy:  {report_data['metrics']['accuracy']:.4f}\n")
            f.write(f"  Precision: {report_data['metrics']['precision']:.4f}\n")
            f.write(f"  Recall:    {report_data['metrics']['recall']:.4f}\n")
            f.write(f"  F1-Score:  {report_data['metrics']['f1']:.4f}\n\n")
            
            f.write(f"Per-Class Performance:\n")
            for emotion, metrics in report_data['per_class_metrics'].items():
                f.write(f"  {emotion.capitalize()}:\n")
                f.write(f"    Precision: {metrics['precision']:.4f}\n")
                f.write(f"    Recall:    {metrics['recall']:.4f}\n")
                f.write(f"    F1-Score:  {metrics['f1']:.4f}\n")
            
            f.write(f"\nError Analysis:\n")
            error_analysis = report_data['error_analysis']
            f.write(f"  Misclassified: {error_analysis['num_misclassified']} samples\n")
            f.write(f"  Error Rate: {error_analysis['misclassification_rate']:.4f}\n")
            
            if 'confusion_patterns' in error_analysis:
                f.write(f"\nMost Common Confusion Patterns:\n")
                for pattern, count in list(error_analysis['confusion_patterns'].items())[:5]:
                    f.write(f"  {pattern}: {count} cases\n")
        
        self.logger.info(f"Evaluation summary saved to {summary_path}")
