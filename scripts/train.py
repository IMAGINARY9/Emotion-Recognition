#!/usr/bin/env python3
"""
Main training script for emotion recognition models.

This script provides a command-line interface for training emotion recognition
models with different architectures and configurations.
"""

import torch
from torch.utils.data import DataLoader # Explicit import for DataLoader
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
from utils.data_utils import EmotionDataLoader, setup_data_directories, download_sample_data
from src.preprocessing import EmotionPreprocessor, EmotionDataProcessor
from src.models import (
    create_model,
    EmotionDataset,
    JointEnsembleDataset, 
    get_joint_ensemble_collate_fn
)
from src.training import EmotionTrainer
from src.evaluation import EmotionEvaluator
from utils.model_loading import filter_model_config
from utils.training_utils import load_and_prepare_data, create_datasets, create_datasets

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
    parser.add_argument("--data-path", "--data_path", type=str, help="Path to training data CSV")
    parser.add_argument("--use-sample-data", "--use_sample_data", action="store_true", help="Use generated sample data")

    # Training overrides
    parser.add_argument("--epochs", type=int, help="Number of training epochs (overrides config)")
    parser.add_argument("--batch-size", "--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", "--learning_rate", type=float, help="Learning rate")

    # Output and options
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--no-evaluation", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--validate", action="store_true", help="Run validation after training")
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
        logger.info(f"Upsampled training set to {len(processed_train)} samples (per-class: {max_count})")    # --- BiLSTM-specific: Build vocab and tokenizer ---
    use_bilstm = any(m['type'] == 'bilstm' for m in getattr(config.model, 'models', [])) or getattr(config.model, 'type', None) == 'bilstm'
    bilstm_tokenizer = None
    vocab = None
    if use_bilstm:
        from utils.vocab_utils import build_bilstm_vocab_and_tokenizer
        train_texts = processed_train['clean_text'].tolist() if not processed_train.empty else []
        vocab_path = None
        if models_dir:
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            vocab_path = str(Path(models_dir) / "bilstm_vocab.json")
        vocab, bilstm_tokenizer = build_bilstm_vocab_and_tokenizer(train_texts, vocab_path)
        if vocab_path:
            logger.info(f"Saved BiLSTM vocab to {vocab_path}")
    # Joint training: use JointEnsembleDataset
    if getattr(config.training, 'strategy', None) == 'joint':
        logger.info("create_datasets: Joint training strategy detected. Preparing JointEnsembleDataset.")
        if not hasattr(config.model, 'models') or not config.model.models:
            logger.error("Joint strategy selected, but config.model.models is not defined or empty.")
            raise ValueError("config.model.models must be defined for joint ensemble strategy.")

        from transformers import AutoTokenizer # Ensure AutoTokenizer is imported
        sub_models_tokenizers_and_configs = []
        for model_detail_cfg_dict in config.model.models:
            model_type = model_detail_cfg_dict['type']
            model_name = model_detail_cfg_dict.get('name', model_type) # Matches EnsembleEmotionModel's naming

            sub_tokenizer = None
            tokenizer_path = model_detail_cfg_dict.get('model_name_or_path', None)
            if not tokenizer_path: # Fallback for known types if path not specified
                if model_type == 'distilbert':
                    tokenizer_path = model_detail_cfg_dict.get('pretrained_model_name_or_path', 'distilbert-base-uncased')
                elif model_type in ['twitter-roberta', 'twitter_roberta', 'roberta']:
                    tokenizer_path = model_detail_cfg_dict.get('pretrained_model_name_or_path', 'cardiffnlp/twitter-roberta-base-emotion')
                # Add other model types and their default tokenizer paths as needed

            if tokenizer_path:
                try:
                    sub_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                    logger.info(f"Loaded tokenizer for sub-model '{model_name}' ({model_type}) from {tokenizer_path}")
                except Exception as e:
                    logger.error(f"Failed to load tokenizer for sub-model '{model_name}' ({model_type}) from {tokenizer_path}: {e}", exc_info=True)
                    raise
            elif model_type == 'bilstm':
                if bilstm_tokenizer is None: # Assuming bilstm_tokenizer is prepared if BiLSTM is a sub-model type
                    logger.error("BiLSTM sub-model specified for joint ensemble, but BiLSTM tokenizer (vocab-based) is not available.")
                    raise ValueError("BiLSTM tokenizer required for BiLSTM sub-model in joint ensemble.")
                sub_tokenizer = bilstm_tokenizer # This needs to be handled by JointEnsembleDataset if not an AutoTokenizer
                logger.info(f"Using pre-built BiLSTM tokenizer for sub-model '{model_name}' ({model_type})")
            
            sub_max_length = model_detail_cfg_dict.get('max_length', config.data.get('max_length', 128))

            if sub_tokenizer:
                sub_models_tokenizers_and_configs.append({
                    'name': model_name.replace('_', '-'),  # Normalize to hyphen
                    'tokenizer': sub_tokenizer,
                    'max_length': sub_max_length,
                    'type': model_type.replace('_', '-')   # Normalize to hyphen
                })
            else:
                logger.warning(f"No tokenizer could be loaded or prepared for sub-model {model_name} ({model_type}) from {tokenizer_path}. "
                               "This may lead to issues in joint ensemble training. "
                               "Ensure the tokenizer is correctly specified in the model config.")

        if not sub_models_tokenizers_and_configs:
            logger.error("No sub-models with tokenizers could be prepared for JointEnsembleDataset.")
            raise ValueError("Failed to prepare sub-models for JointEnsembleDataset.")

        label_encoder = data_loader.label_encoder
        datasets = {}
        if not processed_train.empty:
            datasets['train'] = JointEnsembleDataset(
                texts=processed_train[config.data.text_column].tolist(),
                labels=label_encoder.transform(processed_train[config.data.label_column]).tolist()
            )
            logger.info(f"Created JointEnsembleDataset for train with {len(datasets['train'])} samples.")
        if not processed_val.empty:
            datasets['validation'] = JointEnsembleDataset(
                texts=processed_val[config.data.text_column].tolist(),
                labels=label_encoder.transform(processed_val[config.data.label_column]).tolist()
            )
            logger.info(f"Created JointEnsembleDataset for validation with {len(datasets['validation'])} samples.")
        if not processed_test.empty:
            datasets['test'] = JointEnsembleDataset(
                texts=processed_test[config.data.text_column].tolist(),
                labels=label_encoder.transform(processed_test[config.data.label_column]).tolist()
            )
            logger.info(f"Created JointEnsembleDataset for test with {len(datasets['test'])} samples.")
        return datasets, data_processor, vocab, sub_models_tokenizers_and_configs # Also return sub_models_tokenizers_and_configs

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
    from src.models import get_joint_ensemble_collate_fn, JointEnsembleDataset
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
        class_weights=class_weights,    )
    # Create data loaders
    from utils.data_utils import EmotionDataLoader
    # Joint strategy: use joint collate_fn and tokenizers
    if getattr(config.training, 'strategy', None) == 'joint':
        # --- ENFORCE CORRECT DATASET AND COLLATE_FN FOR JOINT ENSEMBLE ---
        assert isinstance(datasets['train'], JointEnsembleDataset), (
            "For joint ensemble, 'train' dataset must be JointEnsembleDataset. "
            "Check dataset creation logic.")
        assert isinstance(datasets['validation'], JointEnsembleDataset), (
            "For joint ensemble, 'validation' dataset must be JointEnsembleDataset. "
            "Check dataset creation logic.")
        # Use tokenizers from sub_models_tokenizers_and_configs (from create_datasets)
        tokenizers = {d['name']: d['tokenizer'] for d in sub_models_tokenizers_and_configs}
        if vocab is not None:
            tokenizers['bilstm'] = None
        collate_fn = get_joint_ensemble_collate_fn(tokenizers, max_length=config.data.get('max_length', 128), vocab=vocab)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(datasets['train'], batch_size=config.data.batch_sizes.get('train', 16), shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(datasets['validation'], batch_size=config.data.batch_sizes.get('validation', 32), shuffle=False, collate_fn=collate_fn)
        # --- VALIDATE BATCH STRUCTURE ---
        sample_batch = next(iter(train_loader))
        required_keys = set(tokenizers.keys()) | {'labels'}
        if not required_keys.issubset(set(sample_batch.keys())):
            raise RuntimeError(f"Joint ensemble batch keys mismatch: expected {required_keys}, got {set(sample_batch.keys())}")
        data_loaders = {'train': train_loader, 'validation': val_loader}
    else:        
        if data_loader is None:
            from utils.data_utils import EmotionDataLoader
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
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.training.num_epochs,
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
        logger.info("Skipping evaluation as per configuration.")
        return {} # Return empty dict or suitable default

    logger.info("Starting evaluation...")
    
    evaluator = EmotionEvaluator(
        model=trainer.model,
        device=device,
        emotion_names=config.emotions,
        save_dir=output_dir
    )

    is_joint_ensemble = hasattr(trainer.model, 'strategy') and getattr(trainer.model, 'strategy', None) == 'joint'
    logger.info(f"Is joint ensemble for evaluation: {is_joint_ensemble}")

    results = {}
    if is_joint_ensemble:
        logger.info("Evaluating joint ensemble using JointEnsembleDataset and custom DataLoader.")
        if 'test' not in datasets or not datasets['test']:
            logger.warning("Test dataset not available or empty for joint ensemble. Skipping evaluation.")
            return {}
            
        # Corrected assert statement to be on one line or ensure backslash is properly handled
        assert isinstance(datasets['test'], JointEnsembleDataset), "For joint ensemble evaluation, datasets['test'] must be a JointEnsembleDataset."

        sub_model_descriptors = getattr(trainer.model, 'sub_model_descriptors_for_collate', None)
        vocab_for_collate = getattr(trainer.model, 'vocab_for_collate', None)

        if sub_model_descriptors is None:
            logger.error("JointEnsembleModel does not have 'sub_model_descriptors_for_collate'. Cannot create test DataLoader.")
            # Potentially raise an error or return an empty dict to signify failure
            return {}

        # Fix: build tokenizers dict from sub_model_descriptors
        tokenizers = {d['name']: d['tokenizer'] for d in sub_model_descriptors}
        collate_fn_joint = get_joint_ensemble_collate_fn(
            tokenizers,
            max_length=config.data.get('max_length', 128),
            vocab=vocab_for_collate
        )
        test_loader = DataLoader(
            datasets['test'], 
            batch_size=config.data.batch_sizes.get('test', 32),
            shuffle=False,
            collate_fn=collate_fn_joint
        )
        logger.info(f"Created test_loader for joint ensemble with {len(test_loader.dataset)} samples.")
        results = evaluator.evaluate_model_from_dataloader(
            test_loader,
            debug_predictions=debug_predictions
        )
    else: # Standard model evaluation
        logger.info("Evaluating standard model.")
        if 'test' in datasets and datasets['test']:
            # For standard models, datasets['test'] should be an EmotionDataset (or similar)
            # which should have .texts and .labels attributes after create_datasets
            try:
                test_texts = datasets['test'].texts
                test_labels = datasets['test'].labels
                
                if not test_texts or not test_labels: # Check if lists are empty
                    logger.warning("Test dataset's texts or labels are empty. Skipping evaluation.")
                    return {}

                # The tokenizer for standard models should be available on the trainer
                # or handled by the evaluator internally.
                # If evaluator.evaluate_model needs a tokenizer, it should be passed.
                # trainer_tokenizer = getattr(trainer, 'tokenizer', None)
                # if not trainer_tokenizer:
                # logger.warning("Trainer does not have a tokenizer. Standard evaluation might fail if tokenizer is required by evaluator.")
                
                logger.info(f"Evaluating standard model on {len(test_texts)} test samples.")
                results = evaluator.evaluate_model(
                    test_texts,
                    test_labels,
                    # tokenizer=trainer_tokenizer, # Pass if evaluator.evaluate_model requires it
                    batch_size=config.data.batch_sizes.get('test', 32),
                    max_length=config.data.get('max_length', 128),
                    debug_predictions=debug_predictions
                )
            except AttributeError as e:
                logger.error(f"Test dataset for standard model is missing .texts or .labels attributes: {e}. Skipping evaluation.")
                return {}
            except Exception as e:
                logger.error(f"An unexpected error occurred during standard model evaluation setup: {e}")
                return {}
        else:
            logger.warning("Test dataset not available or empty for standard model. Skipping evaluation.")
            return {}
            
    logger.info("Evaluation finished.")
    return results

def train_and_save_individual_model(submodel_type, submodel_cfg, datasets, device, output_dir, config, vocab=None, debug_predictions=False):
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
    
    # Use config epochs instead of submodel_cfg epochs
    num_epochs = config.training.num_epochs
    logger.info(f"Training {submodel_type} for {num_epochs} epochs")
    
    trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        num_epochs=num_epochs,
        batch_size=config.data.batch_sizes.get('train', 32),
        max_length=config.data.get('max_length', 128),
        patience=config.training.get('early_stopping', {}).get('patience', 3),
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

def apply_arg_overrides(config, args):
    """Apply command-line argument overrides to the config object."""
    # Training
    if getattr(args, 'epochs', None) is not None:
        config.training.num_epochs = args.epochs
        logger.info(f"[OVERRIDE] Set config.training.num_epochs = {args.epochs}")
    if getattr(args, 'batch_size', None) is not None:
        config.training.batch_size = args.batch_size
        if hasattr(config.data, 'batch_sizes') and isinstance(config.data.batch_sizes, dict):
            config.data.batch_sizes['train'] = args.batch_size
        logger.info(f"[OVERRIDE] Set config.training.batch_size = {args.batch_size}")
    if getattr(args, 'learning_rate', None) is not None:
        config.training.learning_rate = args.learning_rate
        logger.info(f"[OVERRIDE] Set config.training.learning_rate = {args.learning_rate}")
    if getattr(args, 'force_cpu', False):
        config.training.device = 'cpu'
        logger.info(f"[OVERRIDE] Set config.training.device = 'cpu'")
    # Data
    if getattr(args, 'data_path', None) is not None:
        config.paths.data_path = args.data_path
        logger.info(f"[OVERRIDE] Set config.paths.data_path = {args.data_path}")
    if getattr(args, 'use_sample_data', False):
        config.data.use_sample_data = True
        logger.info(f"[OVERRIDE] Set config.data.use_sample_data = True")
    # Output
    if getattr(args, 'output_dir', None) is not None:
        config.paths.output_dir = args.output_dir
        logger.info(f"[OVERRIDE] Set config.paths.output_dir = {args.output_dir}")
    if getattr(args, 'experiment_name', None) is not None:
        config.experiment_name = args.experiment_name
        logger.info(f"[OVERRIDE] Set config.experiment_name = {args.experiment_name}")
    # Logging
    if getattr(args, 'verbose', False):
        config.logging.level = "DEBUG"
        logger.info(f"[OVERRIDE] Set config.logging.level = DEBUG")
    if getattr(args, 'seed', None) is not None:
        config.seed = args.seed
        logger.info(f"[OVERRIDE] Set config.seed = {args.seed}")
    # Add more overrides as needed
    return config

def needs_ensemble_calibration(config):
    """Check if ensemble weights are uniform or missing, indicating need for calibration."""
    if getattr(config.model, 'type', None) != 'ensemble':
        return False
    weights = getattr(config.model, 'weights', None)
    if not weights or len(set(weights)) == 1:
        return True
    return False

def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)
    
    config_manager = ConfigManager()
    
    # Load configuration - prioritize --config-file over -c/--config
    if args.config_file:
        config = config_manager.load_config(args.config_file)
    else:
        config = config_manager.load_model_config(args.config)
    
    # Apply command line overrides in a single, clear step
    config = apply_arg_overrides(config, args)

    # Forcefully set num_epochs if provided (guaranteed override)
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
        logger.info(f"[FORCE] Set config.training.num_epochs to {args.epochs} after overrides")
    
    # Validate configuration
    config_manager.validate_config(config)
    # Set up logging
    setup_logging(config)
    # Log final epoch count for verification
    logger.info(f"Final training configuration - epochs: {config.training.num_epochs}")
    # --- CUDA check logic ---
    import torch
    requested_device = getattr(config.training, 'device', 'auto')
    if requested_device == 'cuda':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
    elif requested_device == 'cpu':
        device = 'cpu'
    else:  # auto or unspecified
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    # Overwrite config.training.device with resolved device
    config.training.device = device
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = getattr(config, 'experiment_name', None) or args.experiment_name or f"{config.model.type}_{timestamp}"
    # --- Output directory structure ---
    models_dir = Path("models") / experiment_name
    reports_dir = Path("reports") / experiment_name
    outputs_dir = models_dir / "outputs"
    logs_dir = Path("logs")
    for d in [models_dir, reports_dir, outputs_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Model dir: {models_dir}")
    logger.info(f"Reports dir: {reports_dir}")
    logger.info(f"Outputs dir: {outputs_dir}")
    logger.info(f"Logs dir: {logs_dir}")
    config_path = models_dir / "config.yaml"
    config_manager.save_config(config, config_path)
    try:
        # --- Data Preparation ---
        data_loader, train_df, val_df, test_df = load_and_prepare_data(config, args.data_path, args.use_sample_data)
        if getattr(config.training, 'strategy', None) == 'joint':
            datasets, data_processor, vocab, sub_models_tokenizers_and_configs = create_datasets(config, data_loader, train_df, val_df, test_df, models_dir=models_dir)
        else:
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
                    model, trainer = train_and_save_individual_model(sub_type, sub_cfg, datasets, device, sub_dir, config, vocab=vocab, debug_predictions=args.debug_predictions)
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
                # Build tokenizers dict from sub_models_tokenizers_and_configs
                tokenizers = {d['name']: d['tokenizer'] for d in sub_models_tokenizers_and_configs}
                collate_fn = get_joint_ensemble_collate_fn(tokenizers, max_length=config.data.get('max_length', 128), vocab=vocab)
                train_loader = DataLoader(datasets['train'], batch_size=config.data.batch_sizes.get('train', 16), shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(datasets['validation'], batch_size=config.data.batch_sizes.get('validation', 32), shuffle=False, collate_fn=collate_fn)
                submodels = []
                # Only include submodels for which we have tokenizers/configs
                valid_model_names = {d['name'] for d in sub_models_tokenizers_and_configs}
                for sub_cfg in config.model.models:
                    sub_type = sub_cfg['type'].replace('_', '-')                    
                    sub_name = sub_cfg.get('name', sub_type)
                    if sub_name in valid_model_names:
                        submodel_cfg = dict(sub_cfg)
                        if sub_type == 'bilstm' and vocab is not None:
                            submodel_cfg['vocab_size'] = len(vocab)
                            submodel_cfg['vocab'] = vocab
                        submodel = create_model(model_type=sub_type, model_config=submodel_cfg)
                        submodels.append(submodel)
                    else:
                        logger.warning(f"Skipping submodel {sub_name} as no tokenizer/config was prepared for it.")
                weights = [m.get('weight', 1.0/len(submodels)) for m in config.model.models if m.get('name', m['type'].replace('_', '-')) in valid_model_names]
                ensemble_model = create_model(model_type='ensemble', model_config={'models': submodels, 'weights': weights})
                from src.training import EmotionTrainer
                trainer = EmotionTrainer(model=ensemble_model, device=device, save_dir=str(models_dir))
                
                # For joint ensemble training, we need to use the special DataLoaders with joint collate function
                # Create custom training loop using the joint DataLoaders
                from torch.optim import AdamW
                from transformers.optimization import get_linear_schedule_with_warmup
                
                optimizer = AdamW(ensemble_model.parameters(), lr=float(config.training.learning_rate), weight_decay=float(config.training.weight_decay))
                total_steps = len(train_loader) * config.training.num_epochs
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=config.training.warmup_steps,
                    num_training_steps=total_steps
                )
                
                # Use the trainer's train_epoch and validate methods with our joint DataLoaders
                training_history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [], 'train_f1': [], 'val_f1': []}
                best_val_f1 = 0
                patience_counter = 0
                patience = config.training.get('early_stopping', {}).get('patience', 3)
                num_epochs = config.training.num_epochs
                logger.info(f"[DEBUG] Joint ensemble training will run for num_epochs: {num_epochs} (from config.training.num_epochs)")
                for epoch in range(num_epochs):
                    logger.info(f"Joint Ensemble Epoch {epoch + 1}/{num_epochs}")
                    # Training
                    train_loss, train_acc, train_f1, _ = trainer.train_epoch(train_loader, optimizer, scheduler, debug_predictions=args.debug_predictions)
                    # Validation
                    val_loss, val_acc, val_f1, _ = trainer.validate(val_loader, debug_predictions=args.debug_predictions)
                    
                    # Update history
                    training_history['train_loss'].append(train_loss)
                    training_history['val_loss'].append(val_loss)
                    training_history['train_accuracy'].append(train_acc)
                    training_history['val_accuracy'].append(val_acc)
                    training_history['train_f1'].append(train_f1)
                    training_history['val_f1'].append(val_f1)
                    
                    # Early stopping
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        patience_counter = 0
                        trainer.save_model("ensemble_best_model")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping triggered at epoch {epoch+1}")
                            break
                    
                    logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
                
                # Save final model if no early stopping occurred
                if patience_counter < patience:
                    trainer.save_model("ensemble_best_model")
                # Remove duplicate save_model call
                # trainer.save_model("ensemble_best_model")
                # Attach joint strategy and collate info to model for evaluation
                ensemble_model.strategy = 'joint'
                ensemble_model.sub_model_descriptors_for_collate = sub_models_tokenizers_and_configs
                ensemble_model.vocab_for_collate = vocab
                trainer.model = ensemble_model
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
            model, trainer = train_and_save_individual_model(sub_type, dict(config.model), datasets, device, models_dir, config, vocab=vocab, debug_predictions=args.debug_predictions)
        
        # --- Evaluation ---
        # Handle evaluation based on command line arguments
        should_evaluate = not args.no_evaluation or args.validate
        
        if should_evaluate:
            logger.info("Starting evaluation...")
            # Save evaluation reports in reports_dir
            evaluate_model(config, trainer if 'trainer' in locals() else submodel_trainers[0], datasets, device, str(reports_dir), debug_predictions=args.debug_predictions)
        else:
            logger.info("Skipping evaluation as requested")
            
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        raise

    # After joint ensemble training, extract and save submodels for future calibration or use
    if getattr(config.training, 'strategy', None) == 'joint' and 'trainer' in locals() and hasattr(trainer.model, 'models'):
        extract_and_save_submodels(trainer.model, models_dir)
    # After training, check if ensemble weights need calibration and apply if needed
    if getattr(config.model, 'type', None) == 'ensemble' and needs_ensemble_calibration(config):
        from utils.calibrate_ensemble_weights import calibrate_ensemble_weights
        logger.info("Calibrating ensemble weights...")
        # Assume validation set and loader are available as val_loader
        best_weights, best_score = calibrate_ensemble_weights(trainer.model, val_loader, device=device, metric='f1_macro', step=0.1, verbose=True)
        # Replace weights in config (if there more submodels than weights, fill with 0)
        for i, w in enumerate(config.model.models):
            print(f"Setting weight for submodel {w['type']} to {best_weights[i] if i < len(best_weights) else 0.0}")
            if i < len(best_weights):
                w['weight'] = float(best_weights[i])
            else:
                w['weight'] = 0.0
        # Save updated config with new weights using config_manager.save_config for robust serialization
        config_path = Path(models_dir) / 'config.yaml'
        config_manager.save_config(config, config_path)
        logger.info(f"Calibrated ensemble weights saved to config: {config_path}")


from utils.extract_submodels import extract_and_save_submodels

if __name__ == "__main__":
    main()

