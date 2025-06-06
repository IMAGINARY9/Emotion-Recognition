#!/usr/bin/env python3
"""
Main training script for emotion recognition models.

This script provides a command-line interface for training emotion recognition
models with different architectures and configurations.
"""

import torch
import pandas as pd

import argparse
from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime
from collections import Counter
import os
import json
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import ConfigManager, setup_logging, get_device_config
from src.data_utils import EmotionDataLoader, setup_data_directories, download_sample_data
from src.preprocessing import EmotionPreprocessor, EmotionDataProcessor
from src.models import create_model, EmotionDataset
from src.training import EmotionTrainer
from src.evaluation import EmotionEvaluator
from utils.model_loading import filter_model_config

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments (refactored for brevity)."""
    parser = argparse.ArgumentParser(
        description="Train emotion recognition models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Config and data
    parser.add_argument("-c", "--config", type=str, default="distilbert",
                        help="Model config (distilbert, twitter_roberta, bilstm, ensemble)")
    parser.add_argument("--config-file", type=str, help="Path to custom config file")
    parser.add_argument("--data-path", type=str, help="Path to training data CSV")
    parser.add_argument("--use-sample-data", action="store_true", help="Use generated sample data")

    # Training overrides
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")

    # Output and options
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--no-evaluation", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug-predictions", action="store_true", help="Show debug predictions/logits")

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

def create_datasets(config, data_loader, train_df, val_df, test_df, models_dir=None, upsample_minority=True):
    """Create PyTorch datasets."""
    from src.models import JointEnsembleDataset
    preprocessor = EmotionPreprocessor(
        handle_emojis=config.preprocessing.get('emoji_handling', 'convert'),
        expand_contractions=config.preprocessing.get('expand_contractions', True),
        remove_stopwords=config.preprocessing.get('remove_stopwords', False),
        lemmatize=config.preprocessing.get('lemmatize', False),
        normalize_case=config.preprocessing.get('lowercase', True),
        handle_social_media=True,
        min_length=3,
        max_length=config.data.get('max_length', 128)
    )
    data_processor = EmotionDataProcessor(preprocessor=preprocessor)
    processed_train, processed_val, processed_test = data_processor.process_emotion_dataset(
        train_df, val_df, test_df, text_column=config.data.text_column, label_column=config.data.label_column
    )
    # --- Optional: Upsample minority classes in train set ---
    if upsample_minority and not processed_train.empty:
        from sklearn.utils import resample
        label_col = config.data.label_column
        class_counts = processed_train[label_col].value_counts()
        max_count = class_counts.max()
        dfs = []
        for label in class_counts.index:
            df_label = processed_train[processed_train[label_col] == label]
            if len(df_label) < max_count:
                df_label_upsampled = resample(df_label, replace=True, n_samples=max_count, random_state=42)
                dfs.append(df_label_upsampled)
            else:
                dfs.append(df_label)
        processed_train = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Upsampled training set to {len(processed_train)} samples (per-class: {max_count})")
    # --- BiLSTM-specific: Build vocab and tokenizer ---
    use_bilstm = any(m['type'] == 'bilstm' for m in getattr(config.model, 'models', [])) or getattr(config.model, 'type', None) == 'bilstm'
    bilstm_tokenizer = None
    vocab = None
    if use_bilstm:
        from collections import Counter
        train_texts = processed_train['clean_text'].tolist() if not processed_train.empty else []
        tokens = [word for text in train_texts for word in text.split()]
        vocab_counter = Counter(tokens)
        vocab = {word: idx+2 for idx, (word, _) in enumerate(vocab_counter.most_common())}  # +2 for PAD/UNK
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        def bilstm_tokenizer_fn(text):
            return [vocab.get(tok, vocab.get('<UNK>', 1)) for tok in text.split()]
        bilstm_tokenizer = bilstm_tokenizer_fn
        # --- Save BiLSTM vocab in models_dir if provided ---
        if models_dir:
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            vocab_path = Path(models_dir) / "bilstm_vocab.json"
            with open(vocab_path, "w", encoding="utf-8") as f:
                import json
                json.dump(vocab, f)
            logger.info(f"Saved BiLSTM vocab to {vocab_path}")
    # Joint training: use JointEnsembleDataset
    if getattr(config.training, 'strategy', None) == 'joint':
        datasets = {}
        label_encoder = data_loader.label_encoder
        if not processed_train.empty:
            train_texts, train_labels = data_processor.prepare_for_training(
                processed_train, text_column='clean_text', label_column=config.data.label_column
            )
            train_labels = label_encoder.transform(train_labels)
            datasets['train'] = JointEnsembleDataset(train_texts, train_labels)
        if not processed_val.empty:
            val_texts, val_labels = data_processor.prepare_for_training(
                processed_val, text_column='clean_text', label_column=config.data.label_column
            )
            val_labels = label_encoder.transform(val_labels)
            datasets['validation'] = JointEnsembleDataset(val_texts, val_labels)
        if not processed_test.empty:
            test_texts, test_labels = data_processor.prepare_for_training(
                processed_test, text_column='clean_text', label_column=config.data.label_column
            )
            test_labels = label_encoder.transform(test_labels)
            datasets['test'] = JointEnsembleDataset(test_texts, test_labels)
        return datasets, data_processor, vocab
    # Prepare for training and create datasets
    datasets = {}
    label_encoder = data_loader.label_encoder
    if not processed_train.empty:
        train_texts, train_labels = data_processor.prepare_for_training(
            processed_train, text_column='clean_text', label_column=config.data.label_column
        )
        train_labels = label_encoder.transform(train_labels)
        datasets['train'] = EmotionDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=bilstm_tokenizer if use_bilstm else None,
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
            tokenizer=bilstm_tokenizer if use_bilstm else None,
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
            tokenizer=bilstm_tokenizer if use_bilstm else None,
            max_length=config.data.get('max_length', 128)
        )
    return datasets, data_processor, vocab

def train_model(config, datasets, device: str, output_dir: str, vocab=None, data_loader=None, debug_predictions: bool = False):
    """Train the emotion recognition model."""
    from src.models import get_joint_ensemble_collate_fn
    # Create model
    # Prepare model config for ensemble
    if config.model.type == 'ensemble':
        strategy = getattr(config.training, 'strategy', 'joint')
        config_manager = ConfigManager()
        submodels = []
        submodel_paths = []
        for i, submodel_cfg in enumerate(config.model.models):
            submodel_type = submodel_cfg['type'].replace('_', '-')
            submodel_cfg_clean = {k: v for k, v in submodel_cfg.items() if k not in ('type', 'weight', 'max_length', 'dropout_rate')}
            # Map num_classes to num_labels for all submodels
            if 'num_classes' in submodel_cfg_clean:
                submodel_cfg_clean['num_labels'] = submodel_cfg_clean.pop('num_classes')
            if 'num_classes' not in submodel_cfg_clean and hasattr(config.model, 'num_classes'):
                submodel_cfg_clean['num_labels'] = config.model.num_classes
            if 'dropout_rate' not in submodel_cfg_clean and hasattr(config.model, 'dropout_rate'):
                submodel_cfg_clean['dropout'] = config.model.dropout_rate
            # Add vocab_size and vocab for BiLSTM
            if submodel_type == 'bilstm':
                if vocab is None:
                    raise ValueError('vocab must be built for BiLSTM submodel')
                submodel_cfg_clean['vocab_size'] = len(vocab)
                if 'vocab' in submodel_cfg_clean:
                    del submodel_cfg_clean['vocab']
            # Individual training for ensemble
            if strategy == 'individual':
                print(f"[Ensemble] Training submodel {submodel_type} individually...")
                submodel = create_model(model_type=submodel_type, model_config=submodel_cfg_clean)
                trainer_tokenizer = None
                if submodel_type in ['distilbert', 'twitter-roberta', 'twitter_roberta', 'roberta']:
                    from transformers import AutoTokenizer
                    model_name = submodel_cfg.get('model_name', None) or submodel_cfg_clean.get('model_name', None)
                    if not model_name:
                        # Fallback to default names
                        if submodel_type == 'distilbert':
                            model_name = 'distilbert-base-uncased'
                        elif submodel_type in ['twitter-roberta', 'twitter_roberta', 'roberta']:
                            model_name = 'cardiffnlp/twitter-roberta-base-emotion'
                    trainer_tokenizer = AutoTokenizer.from_pretrained(model_name)
                trainer = EmotionTrainer(model=submodel, tokenizer=trainer_tokenizer, device=device, save_dir=output_dir)
                train_texts = datasets['train'].texts
                train_labels = datasets['train'].labels
                val_texts = datasets['validation'].texts if 'validation' in datasets else []
                val_labels = datasets['validation'].labels if 'validation' in datasets else []
                trainer.train(
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
                submodel_name = f"{submodel_type}_best_model"
                trainer.save_model(submodel_name)
                submodel_path = os.path.join(output_dir, f"{submodel_name}.pt")
                submodel.load_state_dict(torch.load(submodel_path, weights_only=False)['model_state_dict'])
                submodels.append(submodel)
                submodel_paths.append(submodel_path)
            else:
                submodels.append(create_model(model_type=submodel_type, model_config=submodel_cfg_clean))
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
        train_labels = datasets['train'].labels
        num_classes = len(set(train_labels))
        class_counts = Counter(train_labels)
        total = sum(class_counts.values())
        weights = [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)]
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)
    else:
        class_weights = None
    # Provide tokenizer for transformer/ensemble models
    trainer_tokenizer = None
    if hasattr(model, 'tokenizer'):
        trainer_tokenizer = model.tokenizer
    elif hasattr(model, 'models') and hasattr(model.models[0], 'tokenizer'):
        trainer_tokenizer = model.models[0].tokenizer
    else:
        from transformers import AutoTokenizer
        if hasattr(model, 'distilbert'):
            trainer_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        elif hasattr(model, 'roberta'):
            trainer_tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-emotion')
    trainer = EmotionTrainer(
        model=model,
        tokenizer=trainer_tokenizer,
        device=device,
        save_dir=output_dir,
        class_weights=class_weights,
    )
    # Create data loaders
    from src.data_utils import EmotionDataLoader
    # Joint strategy: use joint collate_fn and tokenizers
    if getattr(config.training, 'strategy', None) == 'joint':
        from transformers import AutoTokenizer
        tokenizers = {
            'distilbert': AutoTokenizer.from_pretrained('distilbert-base-uncased'),
            'twitter-roberta': AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-emotion')
        }
        if vocab is not None:
            tokenizers['bilstm'] = None
        collate_fn = get_joint_ensemble_collate_fn(tokenizers, max_length=config.data.get('max_length', 128), vocab=vocab)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(datasets['train'], batch_size=config.data.batch_sizes.get('train', 16), shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(datasets['validation'], batch_size=config.data.batch_sizes.get('validation', 32), shuffle=False, collate_fn=collate_fn)
        data_loaders = {'train': train_loader, 'validation': val_loader}
    else:
        if data_loader is None:
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
    early_stopping_config = None
    if config.training.get('early_stopping', {}).get('enabled', False):
        early_stopping_config = config.training.early_stopping
    logger.info("Starting training...")
    if getattr(config.training, 'strategy', None) == 'joint':
        training_history = trainer.train(
            num_epochs=config.training.num_epochs,
            batch_size=config.data.batch_sizes.get('train', 16),
            max_length=config.data.get('max_length', 128),
            patience=config.training.get('early_stopping', {}).get('patience', 3),
            save_best=True,
            train_loader=train_loader,
            val_loader=val_loader,
            debug_predictions=debug_predictions
        )
    else:
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
            save_best=True,
            debug_predictions=debug_predictions
        )
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training completed. Model saved to: {output_dir}")
    return trainer, training_history

def evaluate_model(config, trainer, datasets, device: str, output_dir: str, debug_predictions: bool = False):
    """Evaluate the trained model."""
    if config.get('no_evaluation', False):
        logger.info("Skipping evaluation as requested")
        return None
    
    logger.info("Starting evaluation...")
    
    # Create evaluator
    evaluator = EmotionEvaluator(
        model=trainer.model,
        device=device,
        emotion_names=config.emotions,
        save_dir=output_dir
    )
    
    # Evaluate on test set if available
    if 'test' in datasets:
        test_texts = datasets['test'].texts
        test_labels = datasets['test'].labels
        results = evaluator.evaluate_model(
            test_texts,
            test_labels,
            batch_size=config.data.batch_sizes.get('test', 64),
            max_length=config.data.get('max_length', 128),
            debug_predictions=debug_predictions
        )

        logger.info(f"Evaluation completed. Results saved to: {output_dir}")
        return results
    else:
        logger.warning("No test set available for evaluation")
        return None

def train_and_save_individual_model(submodel_type, submodel_cfg, datasets, device, output_dir, vocab=None, debug_predictions=False):
    from transformers import AutoTokenizer
    from src.training import EmotionTrainer
    from src.models import BiLSTMEmotionModel, DistilBERTEmotionModel, TwitterRoBERTaEmotionModel
    # Map model type to class
    model_class_map = {
        'bilstm': BiLSTMEmotionModel,
        'distilbert': DistilBERTEmotionModel,
        'twitter-roberta': TwitterRoBERTaEmotionModel,
        'twitter_roberta': TwitterRoBERTaEmotionModel,
        'roberta': TwitterRoBERTaEmotionModel
    }
    model_class = model_class_map.get(submodel_type, None)
    if model_class is None:
        raise ValueError(f"Unknown model type: {submodel_type}")
    # Prepare model config (universal filtering)
    model_config = dict(submodel_cfg)
    if submodel_type == 'bilstm' and vocab is not None:
        model_config['vocab_size'] = len(vocab)
        model_config['vocab'] = vocab
    model_config = filter_model_config(model_class, model_config)
    model = create_model(model_type=submodel_type, model_config=model_config)
    # Tokenizer
    tokenizer = None
    if submodel_type == 'distilbert':
        tokenizer = AutoTokenizer.from_pretrained(model_config.get('model_name', 'distilbert-base-uncased'))
    elif submodel_type in ['twitter-roberta', 'twitter_roberta', 'roberta']:
        tokenizer = AutoTokenizer.from_pretrained(model_config.get('model_name', 'cardiffnlp/twitter-roberta-base-emotion'))
    # Trainer
    trainer = EmotionTrainer(model=model, tokenizer=tokenizer, device=device, save_dir=output_dir)
    train_texts = datasets['train'].texts
    train_labels = datasets['train'].labels
    val_texts = datasets['validation'].texts if 'validation' in datasets else []
    val_labels = datasets['validation'].labels if 'validation' in datasets else []
    trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        num_epochs=submodel_cfg.get('num_epochs', 5),
        batch_size=submodel_cfg.get('batch_size', 32),
        max_length=submodel_cfg.get('max_length', 128),
        patience=submodel_cfg.get('early_stopping', {}).get('patience', 3),
        save_best=True
    )
    # Save model checkpoint
    model_name = f"{submodel_type}_best_model"
    trainer.save_model(model_name)
    # Save tokenizer if applicable
    if tokenizer:
        tokenizer_dir = Path(output_dir) / f"{model_name}_tokenizer"
        tokenizer.save_pretrained(tokenizer_dir)
    # Save vocab if BiLSTM
    if submodel_type == 'bilstm' and vocab is not None:
        vocab_path = Path(output_dir) / "bilstm_vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f)
    # Save training history
    history_path = Path(output_dir) / f"{model_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.history, f, indent=2)
    return model, trainer

def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)
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
    config_manager.validate_config(config)
    if args.verbose:
        config.logging.level = "DEBUG"
    setup_logging(config)
    device = get_device_config(config)
    logger.info(f"Using device: {device}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"{config.model.type}_{timestamp}"
    # --- Output directory structure ---
    models_dir = Path("models") / experiment_name
    reports_dir = Path("reports") / experiment_name
    logs_dir = Path("logs")
    for d in [models_dir, reports_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Model dir: {models_dir}")
    logger.info(f"Reports dir: {reports_dir}")
    logger.info(f"Logs dir: {logs_dir}")
    config_path = models_dir / "config.yaml"
    config_manager.save_config(config, config_path)
    try:
        # --- Data Preparation ---
        data_loader, train_df, val_df, test_df = load_and_prepare_data(config, args.data_path, args.use_sample_data)
        datasets, data_processor, vocab = create_datasets(config, data_loader, train_df, val_df, test_df, models_dir=models_dir)
        # --- Training ---
        if getattr(config.model, 'type', None) == 'ensemble':
            strategy = getattr(config.training, 'strategy', 'joint')
            submodels = []
            submodel_trainers = []
            submodel_dirs = []
            if strategy == 'individual':
                # Train each submodel separately and save assets
                for i, sub_cfg in enumerate(config.model.models):
                    sub_type = sub_cfg['type'].replace('_', '-')
                    sub_dir = models_dir / f"{sub_type}_submodel"
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"[Ensemble-Individual] Training submodel: {sub_type} (dir: {sub_dir})")
                    model, trainer = train_and_save_individual_model(sub_type, sub_cfg, datasets, device, sub_dir, vocab=vocab, debug_predictions=args.debug_predictions)
                    submodels.append(model)
                    submodel_trainers.append(trainer)
                    submodel_dirs.append(sub_dir)
                # Save ensemble config and submodel references
                ensemble_config = dict(config.model)
                ensemble_config['submodel_dirs'] = [str(d) for d in submodel_dirs]
                with open(models_dir / "ensemble_config.json", "w", encoding="utf-8") as f:
                    json.dump(ensemble_config, f, indent=2)
                config_manager.save_config(config, models_dir / "config.yaml")
                logger.info(f"Ensemble (individual) training complete. All submodels saved.")
            else:
                # Joint training: train ensemble as a single model
                from src.models import get_joint_ensemble_collate_fn
                from torch.utils.data import DataLoader
                from transformers import AutoTokenizer
                tokenizers = {
                    'distilbert': AutoTokenizer.from_pretrained('distilbert-base-uncased'),
                    'twitter-roberta': AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-emotion')
                }
                collate_fn = get_joint_ensemble_collate_fn(tokenizers, max_length=config.data.get('max_length', 128), vocab=vocab)
                train_loader = DataLoader(datasets['train'], batch_size=config.data.batch_sizes.get('train', 16), shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(datasets['validation'], batch_size=config.data.batch_sizes.get('validation', 32), shuffle=False, collate_fn=collate_fn)
                submodels = []
                for sub_cfg in config.model.models:
                    sub_type = sub_cfg['type'].replace('_', '-')
                    submodel_cfg = dict(sub_cfg)
                    if sub_type == 'bilstm' and vocab is not None:
                        submodel_cfg['vocab_size'] = len(vocab)
                        submodel_cfg['vocab'] = vocab
                    submodel = create_model(model_type=sub_type, model_config=submodel_cfg)
                    submodels.append(submodel)
                weights = [m.get('weight', 1.0/len(submodels)) for m in config.model.models]
                ensemble_model = create_model(model_type='ensemble', model_config={'models': submodels, 'weights': weights})
                from src.training import EmotionTrainer
                trainer = EmotionTrainer(model=ensemble_model, device=device, save_dir=str(models_dir))
                training_history = trainer.train(
                    num_epochs=config.training.num_epochs,
                    batch_size=config.data.batch_sizes.get('train', 16),
                    max_length=config.data.get('max_length', 128),
                    patience=config.training.get('early_stopping', {}).get('patience', 3),
                    save_best=True,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    debug_predictions=args.debug_predictions
                )
                trainer.save_model("ensemble_best_model")
                for sub_cfg in config.model.models:
                    sub_type = sub_cfg['type'].replace('_', '-')
                    if sub_type == 'distilbert':
                        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
                        tokenizer.save_pretrained(models_dir / 'distilbert_best_model_tokenizer')
                    elif sub_type in ['twitter-roberta', 'twitter_roberta', 'roberta']:
                        tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-emotion')
                        tokenizer.save_pretrained(models_dir / 'twitter-roberta_best_model_tokenizer')
                if vocab is not None:
                    vocab_path = models_dir / "bilstm_vocab.json"
                    with open(vocab_path, "w", encoding="utf-8") as f:
                        json.dump(vocab, f)
                with open(outputs_dir / "training_history.json", "w", encoding="utf-8") as f:
                    json.dump(training_history, f, indent=2)
                logger.info(f"Ensemble (joint) training complete. Model and assets saved.")
        else:
            # Single model training
            sub_type = config.model.type.replace('_', '-')
            logger.info(f"Training single model: {sub_type}")
            model, trainer = train_and_save_individual_model(sub_type, dict(config.model), datasets, device, models_dir, vocab=vocab, debug_predictions=args.debug_predictions)
        # --- Evaluation ---
        if not args.no_evaluation:
            # Save evaluation reports in reports_dir
            evaluate_model(config, trainer if 'trainer' in locals() else submodel_trainers[0], datasets, device, str(reports_dir), debug_predictions=args.debug_predictions)
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
