#!/usr/bin/env python3
"""
Evaluation script for emotion recognition models.

This script provides a command-line interface for evaluating trained
emotion recognition models on test data.

"""

import argparse
import os
import sys
from pathlib import Path
import logging
import torch
import pandas as pd
import json
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import ConfigManager, setup_logging, get_device_config
from src.data_utils import EmotionDataLoader
from src.preprocessing import EmotionPreprocessor, EmotionDataProcessor
from src.models import create_model, EmotionDataset  # Import EmotionDataset here
from src.evaluation import EmotionEvaluator

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate emotion recognition models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        required=True,
        help="Path to trained model directory or checkpoint"
    )
    
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to model configuration file"
    )
    
    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to test data CSV file"
    )
    
    parser.add_argument(
        "--use-validation",
        action="store_true",
        help="Use validation split instead of test split"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluations",
        help="Output directory for evaluation results"
    )
    
    # Options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions to file"
    )
    
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="Perform detailed error analysis"
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
    
    return parser.parse_args()

def load_model_and_config(model_path: str, config_path: str = None):
    """Load trained model and configuration."""
    model_path = Path(model_path)
    
    # Load configuration
    if config_path:
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
    elif (model_path / "config.yaml").exists():
        config_manager = ConfigManager()
        config = config_manager.load_config(model_path / "config.yaml")
    else:
        raise FileNotFoundError("Configuration file not found. Please specify --config-path")
    
    # Load model
    checkpoint_path = None
    if model_path.is_file() and model_path.suffix in ['.pt', '.pth']:
        checkpoint_path = model_path
    elif (model_path / "best_model.pt").exists():
        checkpoint_path = model_path / "best_model.pt"
    elif (model_path / "final_model.pt").exists():
        checkpoint_path = model_path / "final_model.pt"
    else:
        raise FileNotFoundError(f"Model checkpoint not found in {model_path}")
    
    # Create model
    model_config = dict(config.model)
    model_type = model_config.pop('type')

    if model_type == 'ensemble':
        submodels = []
        for sub_cfg in model_config['models']:
            sub_cfg = dict(sub_cfg)
            sub_type = sub_cfg.pop('type')
            # Normalize submodel type (replace underscores with dashes)
            sub_type = sub_type.replace('_', '-')
            # If BiLSTM, ensure vocab_size is present
            if sub_type == 'bilstm' and 'vocab_size' not in sub_cfg:
                with open('configs/bilstm_config.yaml', 'r') as f:
                    default_bilstm_cfg = yaml.safe_load(f)
                sub_cfg['vocab_size'] = default_bilstm_cfg['model']['vocab_size']
            # Remove keys not valid for submodel
            for k in ['weight']:
                sub_cfg.pop(k, None)
            submodel = create_model(model_type=sub_type, model_config=sub_cfg)
            submodels.append(submodel)
        weights = model_config.get('weights', [1.0/len(submodels)]*len(submodels))
        model = create_model(
            model_type='ensemble',
            model_config={'models': submodels, 'weights': weights}
        )
    else:
        model = create_model(
            model_type=model_type,
            model_config=model_config
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info(f"Loaded model from: {checkpoint_path}")
    
    return model, config

def load_test_data(config, data_path: str = None, use_validation: bool = False):
    """Load test data for evaluation."""
    data_loader = EmotionDataLoader(config.paths.data_dir)

    # Prefer explicit data_path if provided
    if data_path:
        df = data_loader.load_csv_data(data_path)
        df = data_loader.validate_data(df, config.data.text_column, config.data.label_column)
        logger.info(f"Loaded test data from: {data_path}")
        return data_loader, df

    # Try to load from splits directory
    splits_dir = Path(config.paths.data_dir) / "splits"
    split_file = splits_dir / ("validation.csv" if use_validation else "test.csv")
    if split_file.exists():
        df = data_loader.load_csv_data(str(split_file))
        df = data_loader.validate_data(df, config.data.text_column, config.data.label_column)
        logger.info(f"Loaded test data from: {split_file}")
        return data_loader, df

    # Fallback: try to load from raw data
    raw_dir = Path(config.paths.data_dir) / "raw"
    for fname in ["emotions.csv", "sample_emotions.csv"]:
        raw_file = raw_dir / fname
        if raw_file.exists():
            df = data_loader.load_csv_data(str(raw_file))
            df = data_loader.validate_data(df, config.data.text_column, config.data.label_column)
            logger.info(f"Loaded test data from: {raw_file}")
            return data_loader, df

    raise FileNotFoundError(
        "No test/validation data found. Please specify --data-path or ensure data/splits or data/raw contains a CSV."
    )

def create_test_dataset(config, data_loader, test_df):
    """Create test dataset."""
    # Initialize preprocessor
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
    
    # Create data processor
    data_processor = EmotionDataProcessor(
        preprocessor=preprocessor
    )
    
    # Preprocess test data
    processed_test = data_processor.preprocessor.preprocess_dataset(
        test_df, config.data.text_column, config.data.label_column
    )
    # Prepare for evaluation
    test_texts, test_labels = data_processor.prepare_for_training(
        processed_test, text_column='clean_text', label_column=config.data.label_column
    )
    test_labels = data_loader.label_encoder.transform(test_labels)
    test_dataset = EmotionDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=None,
        max_length=getattr(config.data, 'max_length', 128)
    )
    return test_dataset, data_processor

def run_evaluation(model, config, test_dataset, device: str, output_dir: str, args):
    """Run comprehensive evaluation."""
    # Create evaluator
    evaluator = EmotionEvaluator(
        model=model,
        device=device,
        emotion_names=config.emotions
    )
    
    # Create data loader
    data_loader = EmotionDataLoader()
    test_loader = data_loader.create_data_loaders(
        {'test': test_dataset},
        batch_sizes={'test': args.batch_size}
    )['test']
    
    # Run evaluation
    logger.info("Running evaluation...")
    test_texts = test_dataset.texts
    test_labels = test_dataset.labels
    results = evaluator.evaluate_model(
        test_texts,
        test_labels,
        batch_size=args.batch_size,
        max_length=getattr(config.data, 'max_length', 128)
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score (Macro): {results['macro_f1']:.4f}")
    print(f"F1 Score (Weighted): {results['f1']:.4f}")
    print(f"Precision (Macro): {results['precision']:.4f}")
    print(f"Recall (Macro): {results['recall']:.4f}")
    
    if 'roc_auc' in results:
        print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    print("\nPer-class metrics:")
    for i, emotion in enumerate(config.emotions):
        precision = results['per_class_precision'][i]
        recall = results['per_class_recall'][i]
        f1 = results['per_class_f1'][i]
        print(f"  {emotion}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    print("="*50)
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate detailed report
    evaluator.generate_evaluation_report(
        results, test_dataset.texts, test_dataset.labels, model_name=config.model.type
    )
    
    # Save human-readable report
    text_report_path = output_dir / "evaluation_report.txt"
    with open(text_report_path, 'w') as f:
        f.write("EMOTION RECOGNITION MODEL EVALUATION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model Type: {config.model.type}\n")
        f.write(f"Test Samples: {len(test_dataset)}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"F1 Score (Macro): {results['macro_f1']:.4f}\n")
        f.write(f"F1 Score (Weighted): {results['f1']:.4f}\n")
        f.write(f"Precision (Macro): {results['precision']:.4f}\n")
        f.write(f"Recall (Macro): {results['recall']:.4f}\n")
        
        if 'roc_auc' in results:
            f.write(f"ROC AUC: {results['roc_auc']:.4f}\n")
        
        f.write("\nPER-CLASS METRICS:\n")
        for i, emotion in enumerate(config.emotions):
            precision = results['per_class_precision'][i]
            recall = results['per_class_recall'][i]
            f1 = results['per_class_f1'][i]
            f.write(f"{emotion}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}\n")
    
    # Generate visualizations
    if not args.no_plots:
        logger.info("Generating visualizations...")
        
        # Confusion matrix
        evaluator.plot_confusion_matrix(results['confusion_matrix'], str(Path(output_dir) / "confusion_matrix.png"))
        
        # Embeddings plot
        evaluator.visualize_embeddings(
            test_dataset.texts, test_dataset.labels, save_path=str(Path(output_dir) / "embeddings_tsne.png")
        )
        
        # ROC curves if available
        if 'roc_curves' in results:
            evaluator.plot_roc_curves(results['roc_curves'], str(output_dir))
    
    # Save predictions
    if args.save_predictions:
        logger.info("Saving predictions...")
        predictions_path = output_dir / "predictions.csv"
        
        # Get predictions and original texts
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_texts = []
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Get original text (this is a simplified approach)
                # In practice, you'd want to store the original texts
                batch_size = len(labels)
                all_texts.extend([f"text_{i}" for i in range(len(all_texts), len(all_texts) + batch_size)])
        
        # Convert to emotion labels
        data_loader = EmotionDataLoader()
        pred_emotions = data_loader.decode_labels(all_predictions)
        true_emotions = data_loader.decode_labels(all_labels)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'text': all_texts,
            'true_emotion': true_emotions,
            'predicted_emotion': pred_emotions,
            'correct': [p == t for p, t in zip(pred_emotions, true_emotions)]
        })
        
        # Add probability columns
        for i, emotion in enumerate(config.emotions):
            predictions_df[f'prob_{emotion}'] = [probs[i] for probs in all_probabilities]
        
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Saved predictions to: {predictions_path}")
    
    # Error analysis
    if args.error_analysis:
        logger.info("Performing error analysis...")
        error_analysis_path = output_dir / "error_analysis.json"
        error_analysis = evaluator.error_analysis(test_loader, save_path=str(error_analysis_path))
    
    logger.info(f"Evaluation completed. Results saved to: {output_dir}")
    
    return results

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load model and configuration
        model, config = load_model_and_config(args.model_path, args.config_path)
        
        # Override device if needed
        if args.force_cpu:
            config.device.force_cpu = True
        
        # Get device
        device = get_device_config(config)
        model = model.to(device)
        logger.info(f"Using device: {device}")
        
        # Load test data
        data_loader, test_df = load_test_data(config, args.data_path, args.use_validation)
        
        # Create test dataset
        test_dataset, data_processor = create_test_dataset(config, data_loader, test_df)
        
        # Run evaluation
        results = run_evaluation(
            model, config, test_dataset, device, args.output_dir, args
        )
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
