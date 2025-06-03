#!/usr/bin/env python3
"""
Main training script for emotion recognition models.

This script provides a command-line interface for training emotion recognition
models with different architectures and configurations.
"""

import argparse
import os
import sys
from pathlib import Path
import logging
import torch
import pandas as pd
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import ConfigManager, setup_logging, get_device_config
from data_utils import EmotionDataLoader, setup_data_directories, download_sample_data
from preprocessing import EmotionPreprocessor, EmotionDataProcessor
from models import create_model, EmotionDataset
from training import EmotionTrainer
from evaluation import EmotionEvaluator

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train emotion recognition models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="distilbert",
        help="Model configuration to use (distilbert, twitter_roberta, bilstm, ensemble)"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to custom configuration file"
    )
    
    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data CSV file"
    )
    
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Use generated sample data for testing"
    )
    
    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    
    # Output
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for this experiment"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    # Options
    parser.add_argument(
        "--no-evaluation",
        action="store_true",
        help="Skip evaluation after training"
    )
    
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_and_prepare_data(config, data_path: str = None, use_sample: bool = False):
    """Load and prepare training data."""
    data_loader = EmotionDataLoader(config.paths.data_dir)
    
    # Setup data directories
    setup_data_directories(config.paths.data_dir)
    
    if use_sample:
        logger.info("Creating sample data for testing...")
        data_path = download_sample_data(os.path.join(config.paths.data_dir, "raw"))
    
    if not data_path:
        # Try to find existing data files
        data_dir = Path(config.paths.data_dir)
        possible_files = [
            data_dir / "raw" / "emotions.csv",
            data_dir / "raw" / "sample_emotions.csv",
            data_dir / "training.csv",
            data_dir / "train.csv"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                data_path = str(file_path)
                logger.info(f"Found data file: {data_path}")
                break
        
        if not data_path:
            raise FileNotFoundError(
                "No data file found. Please specify --data-path or use --use-sample-data"
            )
    
    # Load and validate data
    logger.info(f"Loading data from: {data_path}")
    df = data_loader.load_csv_data(data_path)
    df = data_loader.validate_data(df, config.data.text_column, config.data.label_column)
    
    # Split data
    train_df, val_df, test_df = data_loader.split_data(
        df,
        train_size=config.data.train_split,
        val_size=config.data.val_split,
        test_size=config.data.test_split
    )
    
    # Save splits
    splits_dir = os.path.join(config.paths.data_dir, "splits")
    split_paths = data_loader.save_splits(train_df, val_df, test_df, splits_dir)
    
    return data_loader, train_df, val_df, test_df

def create_datasets(config, data_loader, train_df, val_df, test_df):
    """Create PyTorch datasets."""
    # Initialize preprocessor
    preprocessor = EmotionPreprocessor(
        lowercase=config.preprocessing.lowercase,
        remove_urls=config.preprocessing.remove_urls,
        remove_mentions=config.preprocessing.remove_mentions,
        remove_hashtags=config.preprocessing.remove_hashtags,
        remove_extra_whitespace=config.preprocessing.remove_extra_whitespace,
        remove_stopwords=config.preprocessing.remove_stopwords,
        expand_contractions=config.preprocessing.expand_contractions,
        emoji_handling=config.preprocessing.emoji_handling
    )
    
    # Create data processor
    data_processor = EmotionDataProcessor(
        preprocessor=preprocessor,
        label_encoder=data_loader.label_encoder,
        max_length=config.data.max_length
    )
    
    # Create datasets
    datasets = {}
    
    if not train_df.empty:
        datasets['train'] = data_processor.create_dataset(
            train_df, config.data.text_column, config.data.label_column,
            model_type=config.model.type, model_name=config.model.get('model_name')
        )
    
    if not val_df.empty:
        datasets['validation'] = data_processor.create_dataset(
            val_df, config.data.text_column, config.data.label_column,
            model_type=config.model.type, model_name=config.model.get('model_name')
        )
    
    if not test_df.empty:
        datasets['test'] = data_processor.create_dataset(
            test_df, config.data.text_column, config.data.label_column,
            model_type=config.model.type, model_name=config.model.get('model_name')
        )
    
    return datasets, data_processor

def train_model(config, datasets, device: str, output_dir: str):
    """Train the emotion recognition model."""
    # Create model
    model = create_model(
        model_type=config.model.type,
        num_classes=config.model.num_classes,
        model_name=config.model.get('model_name'),
        dropout_rate=config.model.dropout_rate,
        max_length=config.model.max_length,
        **config.model
    )
    
    # Calculate class weights for imbalanced data
    if 'train' in datasets:
        train_labels = [item[1] for item in datasets['train']]
        # Convert back to text labels for class weight calculation
        from data_utils import EmotionDataLoader
        temp_loader = EmotionDataLoader()
        text_labels = temp_loader.decode_labels(train_labels)
        class_weights = temp_loader.get_class_weights(text_labels)
        class_weights = class_weights.to(device)
    else:
        class_weights = None
    
    # Create trainer
    trainer = EmotionTrainer(
        model=model,
        device=device,
        output_dir=output_dir,
        class_weights=class_weights
    )
    
    # Create data loaders
    from data_utils import EmotionDataLoader
    data_loader = EmotionDataLoader()
    data_loaders = data_loader.create_data_loaders(
        datasets,
        batch_sizes=config.data.batch_sizes,
        num_workers=config.data.get('num_workers', 0)
    )
    
    # Training configuration
    training_config = {
        'learning_rate': config.training.learning_rate,
        'num_epochs': config.training.num_epochs,
        'warmup_steps': config.training.warmup_steps,
        'weight_decay': config.training.weight_decay,
        'gradient_clip_norm': config.training.gradient_clip_norm,
        'accumulation_steps': config.training.get('accumulation_steps', 1)
    }
    
    # Early stopping configuration
    early_stopping_config = None
    if config.training.get('early_stopping', {}).get('enabled', False):
        early_stopping_config = config.training.early_stopping
    
    # Train model
    logger.info("Starting training...")
    training_history = trainer.train(
        train_loader=data_loaders['train'],
        val_loader=data_loaders.get('validation'),
        **training_config,
        early_stopping_config=early_stopping_config
    )
    
    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"Training completed. Model saved to: {output_dir}")
    
    return trainer, training_history

def evaluate_model(config, trainer, datasets, device: str, output_dir: str):
    """Evaluate the trained model."""
    if config.get('no_evaluation', False):
        logger.info("Skipping evaluation as requested")
        return None
    
    logger.info("Starting evaluation...")
    
    # Create evaluator
    evaluator = EmotionEvaluator(
        model=trainer.model,
        device=device,
        emotion_labels=config.emotions
    )
    
    # Evaluate on test set if available
    if 'test' in datasets:
        from data_utils import EmotionDataLoader
        data_loader = EmotionDataLoader()
        test_loader = data_loader.create_data_loaders(
            {'test': datasets['test']},
            batch_sizes={'test': config.data.batch_sizes.get('test', 64)}
        )['test']
        
        # Run evaluation
        results = evaluator.evaluate(test_loader)
        
        # Generate report
        report_path = os.path.join(output_dir, "evaluation_report.json")
        evaluator.generate_report(results, report_path)
        
        # Create visualizations
        if config.evaluation.get('plot_confusion_matrix', True):
            evaluator.plot_confusion_matrix(results['confusion_matrix'], output_dir)
        
        if config.evaluation.get('plot_embeddings', True):
            evaluator.plot_embeddings(test_loader, output_dir)
        
        logger.info(f"Evaluation completed. Results saved to: {output_dir}")
        
        return results
    else:
        logger.warning("No test set available for evaluation")
        return None

def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config_manager = ConfigManager()
    
    if args.config_file:
        config = config_manager.load_config(args.config_file)
    else:
        config = config_manager.load_model_config(args.config)
    
    # Apply command line overrides
    overrides = {}
    if args.epochs:
        overrides['training.num_epochs'] = args.epochs
    if args.batch_size:
        overrides['training.batch_size'] = args.batch_size
        overrides['data.batch_sizes.train'] = args.batch_size
    if args.learning_rate:
        overrides['training.learning_rate'] = args.learning_rate
    if args.force_cpu:
        overrides['device.force_cpu'] = True
    if args.output_dir:
        overrides['paths.output_dir'] = args.output_dir
    
    if overrides:
        config = config_manager.merge_configs(config, overrides)
    
    # Validate configuration
    config_manager.validate_config(config)
    
    # Setup logging
    if args.verbose:
        config.logging.level = "DEBUG"
    setup_logging(config)
    
    # Get device
    device = get_device_config(config)
    logger.info(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"{config.model.type}_{timestamp}"
    output_dir = Path(config.paths.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration
    config_path = output_dir / "config.yaml"
    config_manager.save_config(config, config_path)
    
    try:
        # Load and prepare data
        data_loader, train_df, val_df, test_df = load_and_prepare_data(
            config, args.data_path, args.use_sample_data
        )
        
        # Create datasets
        datasets, data_processor = create_datasets(
            config, data_loader, train_df, val_df, test_df
        )
        
        # Train model
        trainer, training_history = train_model(
            config, datasets, device, str(output_dir)
        )
        
        # Evaluate model
        if not args.no_evaluation:
            evaluation_results = evaluate_model(
                config, trainer, datasets, device, str(output_dir)
            )
        
        logger.info("Training pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Experiment: {experiment_name}")
        print(f"Model: {config.model.type}")
        print(f"Device: {device}")
        print(f"Epochs: {config.training.num_epochs}")
        print(f"Final Training Loss: {training_history['train_loss'][-1]:.4f}")
        if 'val_loss' in training_history:
            print(f"Final Validation Loss: {training_history['val_loss'][-1]:.4f}")
        print(f"Output Directory: {output_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
