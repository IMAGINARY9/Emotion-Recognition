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

from src.config import ConfigManager, setup_logging, get_device_config
from src.data_utils import EmotionDataLoader, setup_data_directories, download_sample_data
from src.preprocessing import EmotionPreprocessor, EmotionDataProcessor
from src.models import create_model, EmotionDataset
from src.training import EmotionTrainer
from src.evaluation import EmotionEvaluator

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

    # If data_path is not provided, try to find a default in data/raw or data/splits
    if not data_path:
        # Prefer split files if they exist
        splits_dir = Path(config.paths.data_dir) / "splits"
        split_files = [
            splits_dir / "training.csv",
            splits_dir / "train.csv"
        ]
        for file_path in split_files:
            if file_path.exists():
                data_path = str(file_path)
                logger.info(f"Found split data file: {data_path}")
                break

    # If still not found, try raw data
    if not data_path:
        raw_dir = Path(config.paths.data_dir) / "raw"
        raw_files = [
            raw_dir / "emotions.csv",
            raw_dir / "sample_emotions.csv"
        ]
        for file_path in raw_files:
            if file_path.exists():
                data_path = str(file_path)
                logger.info(f"Found raw data file: {data_path}")
                break

    if use_sample and not data_path:
        logger.info("Creating sample data for testing...")
        data_path = download_sample_data(os.path.join(config.paths.data_dir, "raw"))

    if not data_path:
        raise FileNotFoundError(
            "No data file found. Please specify --data-path or ensure data/raw or data/splits contains a CSV."
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
        handle_emojis=config.preprocessing.get('emoji_handling', 'convert'),
        expand_contractions=config.preprocessing.get('expand_contractions', True),
        remove_stopwords=config.preprocessing.get('remove_stopwords', False),
        lemmatize=config.preprocessing.get('lemmatize', False),
        normalize_case=config.preprocessing.get('lowercase', True),
        handle_social_media=True,  # Always handle social media artifacts
        min_length=3,
        max_length=config.data.get('max_length', 128)
    )
    
    # Create data processor
    data_processor = EmotionDataProcessor(
        preprocessor=preprocessor
    )

    # Preprocess all splits
    processed_train, processed_val, processed_test = data_processor.process_emotion_dataset(
        train_df, val_df, test_df, text_column=config.data.text_column, label_column=config.data.label_column
    )

    # Prepare for training and create datasets
    datasets = {}
    # Use label encoder for integer labels
    label_encoder = data_loader.label_encoder
    if not processed_train.empty:
        train_texts, train_labels = data_processor.prepare_for_training(
            processed_train, text_column='clean_text', label_column=config.data.label_column
        )
        train_labels = label_encoder.transform(train_labels)
        datasets['train'] = EmotionDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=None,  # Set tokenizer if needed for transformers
            max_length=config.data.get('max_length', 128)
        )
    if not processed_val.empty:
        val_texts, val_labels = data_processor.prepare_for_training(
            processed_val, text_column='clean_text', label_column=config.data.label_column
        )
        val_labels = label_encoder.transform(val_labels)
        datasets['validation'] = EmotionDataset(
            texts=val_texts,
            labels=val_labels,
            tokenizer=None,
            max_length=config.data.get('max_length', 128)
        )
    if not processed_test.empty:
        test_texts, test_labels = data_processor.prepare_for_training(
            processed_test, text_column='clean_text', label_column=config.data.label_column
        )
        test_labels = label_encoder.transform(test_labels)
        datasets['test'] = EmotionDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=None,
            max_length=config.data.get('max_length', 128)
        )
    return datasets, data_processor

def train_model(config, datasets, device: str, output_dir: str):
    """Train the emotion recognition model."""
    # Create model
    # Prepare model config for ensemble
    if config.model.type == 'ensemble':
        # Instantiate each submodel
        submodels = []
        config_manager = ConfigManager()
        for submodel_cfg in config.model.models:
            submodel_type = submodel_cfg['type'].replace('_', '-')  # Normalize type for create_model
            # Remove 'type', 'weight', 'max_length', and 'dropout_rate' from submodel config for instantiation
            submodel_cfg_clean = {k: v for k, v in submodel_cfg.items() if k not in ('type', 'weight', 'max_length', 'dropout_rate')}
            # Use defaults from top-level config if not present in submodel config
            if 'num_classes' not in submodel_cfg_clean and hasattr(config.model, 'num_classes'):
                submodel_cfg_clean['num_classes'] = config.model.num_classes
            if 'dropout_rate' not in submodel_cfg_clean and hasattr(config.model, 'dropout_rate'):
                submodel_cfg_clean['dropout_rate'] = config.model.dropout_rate
            # Map num_classes to num_labels for model constructors
            if 'num_classes' in submodel_cfg_clean:
                submodel_cfg_clean['num_labels'] = submodel_cfg_clean.pop('num_classes')
            # Map dropout_rate to dropout for model constructors
            if 'dropout_rate' in submodel_cfg_clean:
                submodel_cfg_clean['dropout'] = submodel_cfg_clean.pop('dropout_rate')
            # Inject BiLSTM required params if missing, using bilstm config file as fallback
            if submodel_type == 'bilstm':
                # Load bilstm config file
                bilstm_config = config_manager.load_config('configs/bilstm_config.yaml')
                bilstm_model_cfg = bilstm_config['model']
                for param in ['vocab_size', 'embedding_dim', 'hidden_dim', 'num_layers']:
                    if param not in submodel_cfg_clean and param in bilstm_model_cfg:
                        submodel_cfg_clean[param] = bilstm_model_cfg[param]
                # num_labels for BiLSTM
                if 'num_labels' not in submodel_cfg_clean:
                    if 'num_classes' in bilstm_model_cfg:
                        submodel_cfg_clean['num_labels'] = bilstm_model_cfg['num_classes']
                    elif hasattr(config.model, 'num_classes'):
                        submodel_cfg_clean['num_labels'] = config.model.num_classes
                # dropout for BiLSTM
                if 'dropout' not in submodel_cfg_clean:
                    if 'dropout_rate' in bilstm_model_cfg:
                        submodel_cfg_clean['dropout'] = bilstm_model_cfg['dropout_rate']
                    elif hasattr(config.model, 'dropout_rate'):
                        submodel_cfg_clean['dropout'] = config.model.dropout_rate
            submodels.append(create_model(
                model_type=submodel_type,
                model_config=submodel_cfg_clean
            ))
        weights = [m.get('weight', 1.0/len(submodels)) for m in config.model.models]
        model = create_model(
            model_type='ensemble',
            model_config={'models': submodels, 'weights': weights}
        )
    else:
        model = create_model(
            model_type=config.model.type,
            model_config=dict(config.model)
        )
    
    # Calculate class weights for imbalanced data
    if 'train' in datasets:
        train_labels = [item['label'].item() if hasattr(item['label'], 'item') else item['label'] for item in datasets['train']]
        # Convert back to text labels for class weight calculation
        from src.data_utils import EmotionDataLoader
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
        save_dir=output_dir,
        class_weights=class_weights,
    )
    
    # Create data loaders
    from src.data_utils import EmotionDataLoader
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
    # For traditional models, pass texts and labels
    train_texts = datasets['train'].texts
    train_labels = datasets['train'].labels
    val_texts = datasets['validation'].texts if 'validation' in datasets else []
    val_labels = datasets['validation'].labels if 'validation' in datasets else []
    training_history = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        num_epochs=config.training.num_epochs,
        batch_size=config.data.batch_sizes.get('train', 32),
        max_length=config.data.get('max_length', 128),
        patience=config.training.get('early_stopping', {}).get('patience', 3),
        save_best=True
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
        emotion_names=config.emotions
    )
    
    # Evaluate on test set if available
    if 'test' in datasets:
        test_texts = datasets['test'].texts
        test_labels = datasets['test'].labels
        # Run evaluation using evaluate_model
        results = evaluator.evaluate_model(
            test_texts,
            test_labels,
            batch_size=config.data.batch_sizes.get('test', 64),
            max_length=config.data.get('max_length', 128)
        )

        # Generate report
        report_path = evaluator.generate_evaluation_report(
            results, test_texts, test_labels, model_name=config.model.type
        )

        # Create visualizations
        if config.evaluation.get('plot_confusion_matrix', True):
            evaluator.plot_confusion_matrix(results['confusion_matrix'], save_path=os.path.join(output_dir, 'confusion_matrix.png'))

        if config.evaluation.get('plot_embeddings', True):
            evaluator.visualize_embeddings(test_texts, test_labels, save_path=os.path.join(output_dir, 'embeddings_tsne.png'))

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
