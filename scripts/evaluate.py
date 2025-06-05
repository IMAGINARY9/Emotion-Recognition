#!/usr/bin/env python3
"""
Evaluation script for emotion recognition models.

This script provides a command-line interface for evaluating trained
emotion recognition models on test data.

"""

import argparse
import sys
import logging
from pathlib import Path
import torch
import pandas as pd
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import ConfigManager, get_device_config
from src.data_utils import EmotionDataLoader
from src.preprocessing import EmotionPreprocessor, EmotionDataProcessor
from src.models import create_model, EmotionDataset
from src.evaluation import EmotionEvaluator

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate emotion recognition models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-path", "-m", type=str, required=True, help="Path to trained model directory or checkpoint")
    parser.add_argument("--config-path", type=str, help="Path to model configuration file")
    parser.add_argument("--data-path", type=str, help="Path to test data CSV file")
    parser.add_argument("--use-validation", action="store_true", help="Use validation split instead of test split")
    parser.add_argument("--output-dir", type=str, default="evaluations", help="Output directory for evaluation results")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def load_model_and_config(model_path: str, config_path: str = None):
    """Load trained model and configuration."""
    model_path = Path(model_path)
    if config_path:
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
    elif (model_path / "config.yaml").exists():
        config_manager = ConfigManager()
        config = config_manager.load_config(model_path / "config.yaml")
    else:
        raise FileNotFoundError("Configuration file not found. Please specify --config-path")
    model_config = dict(config.model)
    model_type = model_config.pop('type').replace('_', '-')
    if 'num_classes' in model_config:
        model_config['num_labels'] = model_config.pop('num_classes')
    if model_type == 'ensemble':
        submodels, weights = [], []
        for sub_cfg in config.model.models:
            sub_cfg = dict(sub_cfg)
            weight = sub_cfg.pop('weight', 1.0/len(config.model.models))
            weights.append(weight)
            sub_type = sub_cfg.pop('type').replace('_', '-')
            if 'num_classes' not in sub_cfg:
                sub_cfg['num_classes'] = config.model.get('num_classes', 6)
            if 'num_classes' in sub_cfg:
                sub_cfg['num_labels'] = sub_cfg['num_classes']
                sub_cfg.pop('num_classes')
            for k in ['dropout_rate', 'max_length', 'bidirectional', 'attention']:
                if k in sub_cfg:
                    if k == 'dropout_rate':
                        sub_cfg['dropout'] = sub_cfg.pop(k)
                    else:
                        sub_cfg.pop(k)
            if sub_type == 'bilstm':
                vocab_path = model_path / 'bilstm_vocab.json'
                if vocab_path.exists():
                    with open(vocab_path, 'r', encoding='utf-8') as f:
                        vocab = json.load(f)
                    sub_cfg['vocab'] = vocab
                    sub_cfg['vocab_size'] = len(vocab)
                else:
                    logger.warning(f"BiLSTM vocab file not found: {vocab_path}, skipping BiLSTM submodel.")
                    continue
                if 'pretrained_embeddings' in sub_cfg:
                    if not isinstance(sub_cfg['pretrained_embeddings'], torch.Tensor):
                        sub_cfg.pop('pretrained_embeddings')
            submodel = create_model(model_type=sub_type, model_config=sub_cfg)
            if sub_type == 'distilbert':
                checkpoint_path = model_path / 'distilbert_best_model.pt'
            elif sub_type == 'twitter-roberta':
                checkpoint_path = model_path / 'twitter-roberta_best_model.pt'
            else:
                continue
            if not checkpoint_path.exists():
                logger.warning(f"Model checkpoint not found: {checkpoint_path}, skipping.")
                continue
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                submodel.load_state_dict(checkpoint['model_state_dict'])
            else:
                submodel.load_state_dict(checkpoint)
            logger.info(f"Loaded {sub_type} model from: {checkpoint_path}")
            submodels.append(submodel)
        model = create_model(model_type='ensemble', model_config={'models': submodels, 'weights': weights[:len(submodels)]})
        logger.info(f"Created ensemble model with {len(submodels)} components")
    else:
        checkpoint_path = None
        if model_path.is_file() and model_path.suffix in ['.pt', '.pth']:
            checkpoint_path = model_path
        elif (model_path / "best_model.pt").exists():
            checkpoint_path = model_path / "best_model.pt"
        elif (model_path / "final_model.pt").exists():
            checkpoint_path = model_path / "final_model.pt"
        else:
            raise FileNotFoundError(f"Model checkpoint not found in {model_path}")
        for k in ['dropout_rate', 'max_length', 'bidirectional', 'attention']:
            if k in model_config:
                if k == 'dropout_rate':
                    model_config['dropout'] = model_config.pop(k)
                else:
                    model_config.pop(k)
        if model_type == 'bilstm':
            vocab_path = model_path.parent / 'bilstm_vocab.json'
            if vocab_path.exists():
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab = json.load(f)
                model_config['vocab'] = vocab
                model_config['vocab_size'] = len(vocab)
            if 'pretrained_embeddings' in model_config:
                if not isinstance(model_config['pretrained_embeddings'], torch.Tensor):
                    model_config.pop('pretrained_embeddings')
        model = create_model(model_type=model_type, model_config=model_config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded model from: {checkpoint_path}")
    return model, config

def load_test_data(config, data_path: str = None, use_validation: bool = False):
    """Load test data for evaluation."""
    data_loader = EmotionDataLoader(config.paths.data_dir)
    if data_path:
        df = data_loader.load_csv_data(data_path)
        df = data_loader.validate_data(df, config.data.text_column, config.data.label_column)
        logger.info(f"Loaded test data from: {data_path}")
        return data_loader, df
    splits_dir = Path(config.paths.data_dir) / "splits"
    split_file = splits_dir / ("validation.csv" if use_validation else "test.csv")
    if split_file.exists():
        df = data_loader.load_csv_data(str(split_file))
        df = data_loader.validate_data(df, config.data.text_column, config.data.label_column)
        logger.info(f"Loaded test data from: {split_file}")
        return data_loader, df
    raw_dir = Path(config.paths.data_dir) / "raw"
    for fname in ["emotions.csv", "sample_emotions.csv"]:
        raw_file = raw_dir / fname
        if raw_file.exists():
            df = data_loader.load_csv_data(str(raw_file))
            df = data_loader.validate_data(df, config.data.text_column, config.data.label_column)
            logger.info(f"Loaded test data from: {raw_file}")
            return data_loader, df
    raise FileNotFoundError("No test/validation data found. Please specify --data-path or ensure data/splits or data/raw contains a CSV.")

def create_test_dataset(config, data_loader, test_df):
    """Create test dataset(s) for all models in the config."""
    model_types = [m['type'] for m in config.model.models] if hasattr(config.model, 'models') else [config.model.type]
    tokenizers, vocabs = {}, {}
    from transformers import AutoTokenizer
    for mtype in model_types:
        if mtype == 'distilbert':
            tokenizers['distilbert'] = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        elif mtype == 'twitter-roberta':
            tokenizers['twitter-roberta'] = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-emotion')
        elif mtype == 'bilstm':
            vocab_path = Path(config.paths.model_dir) / 'bilstm_vocab.json'
            if vocab_path.exists():
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocabs['bilstm'] = json.load(f)
            else:
                vocabs['bilstm'] = None
    preprocessor = EmotionPreprocessor(
        normalize_case=getattr(config.preprocessing, 'lowercase', True),
        handle_emojis=getattr(config.preprocessing, 'emoji_handling', 'convert'),
        expand_contractions=getattr(config.preprocessing, 'expand_contractions', True),
        remove_stopwords=getattr(config.preprocessing, 'remove_stopwords', False),
        lemmatize=getattr(config.preprocessing, 'lemmatize', False),
        handle_social_media=getattr(config.preprocessing, 'handle_social_media', True),
        min_length=getattr(config.preprocessing, 'min_length', 3),
        max_length=getattr(config.data, 'max_length', 128)
    )
    data_processor = EmotionDataProcessor(preprocessor=preprocessor)
    processed_test = data_processor.preprocessor.preprocess_dataset(
        test_df, config.data.text_column, config.data.label_column
    )
    test_texts, test_labels = data_processor.prepare_for_training(
        processed_test, text_column='clean_text', label_column=config.data.label_column
    )
    test_labels = data_loader.label_encoder.transform(test_labels)
    datasets = {}
    if 'distilbert' in model_types:
        datasets['distilbert'] = EmotionDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=tokenizers['distilbert'],
            max_length=getattr(config.data, 'max_length', 128)
        )
    if 'twitter-roberta' in model_types:
        datasets['twitter-roberta'] = EmotionDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=tokenizers['twitter-roberta'],
            max_length=getattr(config.data, 'max_length', 128)
        )
    if 'bilstm' in model_types:
        datasets['bilstm'] = EmotionDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=None,
            max_length=getattr(config.data, 'max_length', 128),
            vocab=vocabs['bilstm']
        )
    if len(datasets) == 1:
        return list(datasets.values())[0], data_processor
    return datasets, data_processor

def run_evaluation(model, config, test_dataset, device: str, output_dir: str, args):
    """Run comprehensive evaluation."""
    evaluator = EmotionEvaluator(
        model=model,
        device=device,
        emotion_names=config.emotions,
        save_dir=output_dir
    )
    def match_submodel_key(model_name, batch_keys):
        candidates = [model_name,
                      model_name.replace('-', '_'),
                      model_name.replace('_', '-'),
                      model_name.lower(),
                      model_name.lower().replace('-', ''),
                      model_name.lower().replace('_', ''),
                      model_name.lower().replace('emotionmodel', ''),
                      model_name.lower().replace('emotion', ''),
                      model_name.lower().replace('model', '')]
        for c in candidates:
            for k in batch_keys:
                if c == k or c in k or k in c:
                    return k
        return None
    if isinstance(test_dataset, dict):
        data_loader = EmotionDataLoader()
        loaders = {k: data_loader.create_data_loaders({'test': ds}, batch_sizes={'test': args.batch_size})['test'] for k, ds in test_dataset.items()}
        test_loader = zip(*(loaders[k] for k in test_dataset.keys()))
        logger.info("Running evaluation (ensemble, multi-input)...")
        all_predictions, all_probabilities, all_labels, all_texts = [], [], [], []
        model.eval()
        with torch.no_grad():
            for batches in test_loader:
                batch_dict = {}
                labels = None
                batch_keys = list(test_dataset.keys())
                for i, (k, batch) in enumerate(zip(batch_keys, batches)):
                    mapped_key = match_submodel_key(k, batch_keys)
                    if 'attention_mask' in batch:
                        batch_dict[mapped_key] = {
                            'input_ids': batch['input_ids'].to(device),
                            'attention_mask': batch['attention_mask'].to(device)
                        }
                    else:
                        batch_dict[mapped_key] = {
                            'input_ids': batch['input_ids'].to(device)
                        }
                    if labels is None:
                        labels = batch['label'].to(device)
                batch_dict['labels'] = labels
                outputs = model(**batch_dict)
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                batch_size = len(labels)
                all_texts.extend([f"text_{i}" for i in range(len(all_texts), len(all_texts) + batch_size)])
        results = evaluator._calculate_metrics(all_labels, all_predictions, all_probabilities)
        from sklearn.metrics import confusion_matrix, classification_report
        results['confusion_matrix'] = confusion_matrix(all_labels, all_predictions)
        results['classification_report'] = classification_report(all_labels, all_predictions, target_names=config.emotions, output_dict=True, zero_division=0)
        results['error_analysis'] = evaluator._perform_error_analysis(all_texts, all_labels, all_predictions, all_probabilities)
        model_name = getattr(model, 'name', model.__class__.__name__)
        evaluator.generate_evaluation_report(results, all_texts, all_labels, model_name=model_name)
        cm_path = Path(output_dir) / f"confusion_matrix_{model_name}.png"
        evaluator.plot_confusion_matrix(results['confusion_matrix'], save_path=str(cm_path))
        pcm_path = Path(output_dir) / f"per_class_metrics_{model_name}.png"
        evaluator.plot_per_class_metrics(results, save_path=str(pcm_path))
        tsne_path = Path(output_dir) / f"embeddings_tsne_{model_name}.png"
        evaluator.visualize_embeddings(all_texts, all_labels, save_path=str(tsne_path))
        return results
    else:
        results = evaluator.evaluate_model(
            test_dataset.texts,
            test_dataset.labels,
            batch_size=args.batch_size,
            max_length=getattr(config.data, 'max_length', 128)
        )
        model_name = getattr(model, 'name', model.__class__.__name__)
        evaluator.generate_evaluation_report(results, test_dataset.texts, test_dataset.labels, model_name=model_name)
        cm_path = Path(output_dir) / f"confusion_matrix_{model_name}.png"
        evaluator.plot_confusion_matrix(results['confusion_matrix'], save_path=str(cm_path))
        pcm_path = Path(output_dir) / f"per_class_metrics_{model_name}.png"
        evaluator.plot_per_class_metrics(results, save_path=str(pcm_path))
        tsne_path = Path(output_dir) / f"embeddings_tsne_{model_name}.png"
        evaluator.visualize_embeddings(test_dataset.texts, test_dataset.labels, save_path=str(tsne_path))
        return results

def main():
    """Main evaluation function."""
    args = parse_args()
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    try:
        model, config = load_model_and_config(args.model_path, args.config_path)
        if args.force_cpu:
            config.device.force_cpu = True
        device = get_device_config(config)
        model = model.to(device)
        logger.info(f"Using device: {device}")
        data_loader, test_df = load_test_data(config, args.data_path, args.use_validation)
        test_dataset, data_processor = create_test_dataset(config, data_loader, test_df)
        results = run_evaluation(
            model, config, test_dataset, device, args.output_dir, args
        )
        logger.info("Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
