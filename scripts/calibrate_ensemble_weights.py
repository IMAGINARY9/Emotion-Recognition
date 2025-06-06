#!/usr/bin/env python3
"""
Calibrate ensemble weights for an already trained ensemble model.
Loads the model and config, runs calibration on the validation set, updates config YAML, and saves new weights.
"""
import argparse
import os
import sys
from pathlib import Path
import logging
import torch
import yaml
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import ConfigManager, setup_logging, get_device_config
from src.data_utils import EmotionDataLoader, setup_data_directories, download_sample_data
from src.preprocessing import EmotionPreprocessor, EmotionDataProcessor
from src.models import create_model, calibrate_ensemble_weights, get_joint_ensemble_collate_fn

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate ensemble weights for a trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to model config.yaml file (in the experiment output directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Experiment output directory (where model and config are saved)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data CSV (if not in default location)"
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Use generated sample data for testing"
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
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    set_seed(args.seed)
    config_manager = ConfigManager()
    # Load config
    config_path = Path(args.config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = config_manager.load_config(str(config_path))
    if args.verbose:
        config.logging.level = "DEBUG"
    setup_logging(config)
    device = get_device_config(config)
    logger.info(f"Using device: {device}")
    output_dir = Path(args.output_dir)
    # --- Data Preparation ---
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    # Actually load and prepare data
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    data_loader, train_df, val_df, test_df = EmotionDataLoader(config.paths.data_dir), None, None, None
    # Use the same logic as in train.py to load and preprocess data
    from src.preprocessing import EmotionPreprocessor, EmotionDataProcessor
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
    df = data_loader.load_csv_data(args.data_path) if args.data_path else data_loader.load_default_data()
    df = data_loader.validate_data(df, config.data.text_column, config.data.label_column)
    train_df, val_df, test_df = data_loader.split_data(
        df,
        train_size=config.data.train_split,
        val_size=config.data.val_split,
        test_size=config.data.test_split
    )
    processed_train, processed_val, processed_test = data_processor.process_emotion_dataset(
        train_df, val_df, test_df, text_column=config.data.text_column, label_column=config.data.label_column
    )
    # Build vocab if needed
    use_bilstm = any(m['type'] == 'bilstm' for m in config_dict['model']['models'])
    vocab = None
    if use_bilstm:
        from collections import Counter
        train_texts = processed_train['clean_text'].tolist() if not processed_train.empty else []
        tokens = [word for text in train_texts for word in text.split()]
        vocab_counter = Counter(tokens)
        vocab = {word: idx+2 for idx, (word, _) in enumerate(vocab_counter.most_common())}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
    # Prepare validation dataset
    label_encoder = data_loader.label_encoder
    val_texts, val_labels = data_processor.prepare_for_training(
        processed_val, text_column='clean_text', label_column=config.data.label_column
    )
    val_labels = label_encoder.transform(val_labels)
    val_dataset = JointEnsembleDataset(val_texts, val_labels)
    # Build collate_fn for joint ensemble if needed
    from torch.utils.data import DataLoader
    strategy = config_dict.get('training', {}).get('strategy', None)
    if strategy == 'joint':
        from transformers import AutoTokenizer
        tokenizers = {
            'distilbert': AutoTokenizer.from_pretrained('distilbert-base-uncased'),
            'twitter-roberta': AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-emotion')
        }
        if vocab is not None:
            tokenizers['bilstm'] = None
        collate_fn = get_joint_ensemble_collate_fn(tokenizers, max_length=config.data.get('max_length', 128), vocab=vocab)
    else:
        collate_fn = getattr(val_dataset, 'get_collate_fn', lambda: None)()
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_sizes.get('validation', 32), shuffle=False, collate_fn=collate_fn)
    # --- Model Loading ---
    # Find best ensemble model checkpoint
    model_files = list(output_dir.glob('best_model*.pt'))
    if not model_files:
        model_files = list(output_dir.glob('*.pt'))
    if not model_files:
        raise FileNotFoundError(f"No model checkpoint found in {output_dir}")
    best_model_path = str(model_files[0])
    logger.info(f"Loading ensemble model from: {best_model_path}")
    # Rebuild submodels
    submodels = []
    for sub_cfg in config_dict['model']['models']:
        submodel_type = sub_cfg['type']
        submodel_cfg = dict(sub_cfg)
        if 'num_classes' in submodel_cfg:
            submodel_cfg['num_labels'] = submodel_cfg.pop('num_classes')
        if submodel_type == 'bilstm':
            submodel_cfg['vocab_size'] = len(vocab)
        submodels.append(create_model(submodel_type, submodel_cfg))
    weights = [m.get('weight', 1.0/len(submodels)) for m in config_dict['model']['models']]
    ensemble_model = create_model('ensemble', {'models': submodels, 'weights': weights})
    # Load state dict
    state = torch.load(best_model_path, map_location=device)
    if 'model_state_dict' in state:
        ensemble_model.load_state_dict(state['model_state_dict'])
    else:
        ensemble_model.load_state_dict(state)
    ensemble_model.to(device)
    # --- Calibration ---
    best_weights, best_score = calibrate_ensemble_weights(
        ensemble_model, val_loader, device=device, metric='f1_macro', step=0.1, verbose=args.verbose
    )
    logger.info(f"Best ensemble weights: {best_weights}, best macro F1: {best_score:.4f}")
    # Update config.yaml with new weights
    for i, w in enumerate(best_weights):
        config_dict['model']['models'][i]['weight'] = float(w)
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_dict, f)
    # Save weights to file
    with open(output_dir / "ensemble_weights.txt", "w") as f:
        f.write(f"Best weights: {best_weights}\nBest macro F1: {best_score:.4f}\n")
    logger.info(f"Updated config.yaml and saved new weights to ensemble_weights.txt.")

if __name__ == "__main__":
    main()
