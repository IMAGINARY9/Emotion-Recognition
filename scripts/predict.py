#!/usr/bin/env python3
"""
Prediction script for emotion recognition models.

This script provides a command-line interface for making predictions
on new text data using trained emotion recognition models.
"""

import argparse
import os
import sys
from pathlib import Path
import logging
import torch
import pandas as pd
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import ConfigManager, setup_logging, get_device_config
from data_utils import EmotionDataLoader
from preprocessing import EmotionPreprocessor
from models import create_model, EmotionPredictor

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make predictions with emotion recognition models",
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
    
    # Input data
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text", "-t",
        type=str,
        help="Single text to predict"
    )
    
    group.add_argument(
        "--text-file",
        type=str,
        help="File containing texts to predict (one per line)"
    )
    
    group.add_argument(
        "--csv-file",
        type=str,
        help="CSV file with text column to predict"
    )
    
    # CSV options
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column in CSV file"
    )
    
    # Output
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        help="Output file for predictions (default: print to console)"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "csv", "txt"],
        default="json",
        help="Output format"
    )
    
    # Options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction"
    )
    
    parser.add_argument(
        "--include-probabilities",
        action="store_true",
        help="Include prediction probabilities in output"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Return top-k predictions per text"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        help="Confidence threshold for predictions"
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
    model = create_model(
        model_type=config.model.type,
        num_classes=config.model.num_classes,
        model_name=config.model.get('model_name'),
        dropout_rate=config.model.dropout_rate,
        max_length=config.model.max_length,
        **config.model
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info(f"Loaded model from: {checkpoint_path}")
    
    return model, config

def load_input_data(args):
    """Load input data for prediction."""
    texts = []
    
    if args.text:
        texts = [args.text]
        logger.info("Using single text input")
    
    elif args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} texts from file: {args.text_file}")
    
    elif args.csv_file:
        df = pd.read_csv(args.csv_file)
        if args.text_column not in df.columns:
            raise ValueError(f"Column '{args.text_column}' not found in CSV file")
        texts = df[args.text_column].dropna().tolist()
        logger.info(f"Loaded {len(texts)} texts from CSV: {args.csv_file}")
    
    if not texts:
        raise ValueError("No input texts found")
    
    return texts

def make_predictions(model, config, texts, device: str, args):
    """Make predictions on input texts."""
    # Create preprocessor
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
    
    # Create predictor
    predictor = EmotionPredictor(
        model=model,
        preprocessor=preprocessor,
        emotion_labels=config.emotions,
        device=device,
        max_length=config.model.max_length,
        model_type=config.model.type,
        model_name=config.model.get('model_name')
    )
    
    # Make predictions
    logger.info(f"Making predictions on {len(texts)} texts...")
    
    if args.top_k > 1:
        predictions = predictor.predict_batch_top_k(
            texts,
            k=args.top_k,
            batch_size=args.batch_size,
            return_probabilities=args.include_probabilities,
            threshold=args.threshold
        )
    else:
        predictions = predictor.predict_batch(
            texts,
            batch_size=args.batch_size,
            return_probabilities=args.include_probabilities,
            threshold=args.threshold
        )
    
    return predictions

def format_output(texts, predictions, args):
    """Format predictions for output."""
    results = []
    
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        if args.top_k > 1:
            # Top-k predictions
            result = {
                "id": i,
                "text": text,
                "predictions": []
            }
            
            for j, (emotion, prob) in enumerate(pred):
                pred_item = {"rank": j + 1, "emotion": emotion}
                if args.include_probabilities:
                    pred_item["probability"] = float(prob)
                result["predictions"].append(pred_item)
        
        else:
            # Single prediction
            result = {
                "id": i,
                "text": text,
                "predicted_emotion": pred["emotion"] if isinstance(pred, dict) else pred
            }
            
            if args.include_probabilities and isinstance(pred, dict):
                result["probability"] = float(pred["probability"])
        
        results.append(result)
    
    return results

def save_output(results, args):
    """Save or print results."""
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif args.output_format == "csv":
            # Flatten results for CSV
            flattened = []
            for result in results:
                if "predictions" in result:
                    # Top-k format
                    for pred in result["predictions"]:
                        row = {
                            "id": result["id"],
                            "text": result["text"],
                            "rank": pred["rank"],
                            "emotion": pred["emotion"]
                        }
                        if "probability" in pred:
                            row["probability"] = pred["probability"]
                        flattened.append(row)
                else:
                    # Single prediction format
                    flattened.append(result)
            
            df = pd.DataFrame(flattened)
            df.to_csv(output_path, index=False)
        
        elif args.output_format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"Text: {result['text']}\n")
                    
                    if "predictions" in result:
                        f.write("Top predictions:\n")
                        for pred in result["predictions"]:
                            prob_str = f" ({pred['probability']:.3f})" if "probability" in pred else ""
                            f.write(f"  {pred['rank']}. {pred['emotion']}{prob_str}\n")
                    else:
                        emotion = result["predicted_emotion"]
                        prob_str = f" (probability: {result['probability']:.3f})" if "probability" in result else ""
                        f.write(f"Predicted emotion: {emotion}{prob_str}\n")
                    
                    f.write("\n")
        
        logger.info(f"Results saved to: {output_path}")
    
    else:
        # Print to console
        print("\n" + "="*50)
        print("EMOTION PREDICTIONS")
        print("="*50)
        
        for result in results:
            print(f"\nText: {result['text']}")
            
            if "predictions" in result:
                print("Top predictions:")
                for pred in result["predictions"]:
                    prob_str = f" ({pred['probability']:.3f})" if "probability" in pred else ""
                    print(f"  {pred['rank']}. {pred['emotion']}{prob_str}")
            else:
                emotion = result["predicted_emotion"]
                prob_str = f" (probability: {result['probability']:.3f})" if "probability" in result else ""
                print(f"Predicted emotion: {emotion}{prob_str}")
        
        print("="*50)

def main():
    """Main prediction function."""
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
        
        # Load input data
        texts = load_input_data(args)
        
        # Make predictions
        predictions = make_predictions(model, config, texts, device, args)
        
        # Format and save output
        results = format_output(texts, predictions, args)
        save_output(results, args)
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
