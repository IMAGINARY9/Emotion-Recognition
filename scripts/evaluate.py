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

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import AutoTokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import ConfigManager, get_device_config
from src.data_utils import EmotionDataLoader
from src.preprocessing import EmotionPreprocessor, EmotionDataProcessor
from src.models import create_model, EmotionDataset
from src.evaluation import EmotionEvaluator
from utils.model_loading import load_model_and_assets, filter_model_config

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
    parser.add_argument(
        "--debug-predictions",
        action="store_true",
        help="Show ensemble debug predictions/logits during evaluation"
    )
    return parser.parse_args()

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
    # If ensemble, use JointEnsembleDataset and joint collate
    if hasattr(config.model, 'type') and config.model.type == 'ensemble':
        from src.models import JointEnsembleDataset, get_joint_ensemble_collate_fn
        test_texts = test_df[config.data.text_column].astype(str).tolist()
        test_labels = data_loader.label_encoder.transform(test_df[config.data.label_column].tolist())
        test_dataset = JointEnsembleDataset(test_texts, test_labels)
        # Build tokenizers for joint collate
        from transformers import AutoTokenizer
        tokenizers = {
            'distilbert': AutoTokenizer.from_pretrained('distilbert-base-uncased'),
            'twitter-roberta': AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-emotion')
        }
        # Check for BiLSTM vocab
        import os
        vocab = None
        for m in config.model.models:
            if m['type'] == 'bilstm':
                vocab_path = Path(config.paths.model_dir) / 'bilstm_vocab.json'
                if vocab_path.exists():
                    with open(vocab_path, 'r', encoding='utf-8') as f:
                        vocab = json.load(f)
                break
        collate_fn = get_joint_ensemble_collate_fn(tokenizers, max_length=getattr(config.data, 'max_length', 128), vocab=vocab)
        return test_dataset, collate_fn
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
        return list(datasets.values())[0], None
    return datasets, None

def run_evaluation(model, config, test_dataset, device: str, output_dir: str, args, collate_fn=None):
    """Run comprehensive evaluation."""
    evaluator = EmotionEvaluator(
        model=model,
        device=device,
        emotion_names=config.emotions,
        save_dir=output_dir
    )
    # If using joint ensemble, use joint DataLoader and pass batches as-is
    if hasattr(config.model, 'type') and config.model.type == 'ensemble' and isinstance(test_dataset, object) and not isinstance(test_dataset, dict):
        from torch.utils.data import DataLoader
        loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        logger.info("Running evaluation (joint ensemble, joint batch)...")
        all_predictions, all_probabilities, all_labels, all_texts = [], [], [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                labels = batch['labels'].to(device)
                # Move submodel input dicts to device, keep labels at top level
                batch_on_device = {}
                for k, v in batch.items():
                    if isinstance(v, dict):
                        batch_on_device[k] = {kk: vv.to(device) for kk, vv in v.items()}
                    elif k == 'labels':
                        batch_on_device[k] = v.to(device)
                    else:
                        batch_on_device[k] = v
                outputs = model(**batch_on_device)
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
            max_length=getattr(config.data, 'max_length', 128),
            debug_predictions=args.debug_predictions
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
    args = parse_args()
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    try:
        model, config, tokenizers, vocabs = load_model_and_assets(args.model_path, args.config_path)
        if args.force_cpu:
            config.device.force_cpu = True
        device = get_device_config(config)
        model = model.to(device)
        logger.info(f"Using device: {device}")
        data_loader, test_df = load_test_data(config, args.data_path, args.use_validation)
        test_dataset, collate_fn = create_test_dataset(config, data_loader, test_df)
        # --- Set output directory to evaluations/<model_dir_name>/ ---
        from pathlib import Path
        model_path = Path(args.model_path)
        if model_path.is_file():
            model_dir_name = model_path.parent.name
        else:
            model_dir_name = model_path.name
        output_dir = Path(args.output_dir) / model_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        run_evaluation(model, config, test_dataset, device, str(output_dir), args, collate_fn=collate_fn)
        logger.info(f"Evaluation completed successfully! Results saved to: {output_dir}")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
