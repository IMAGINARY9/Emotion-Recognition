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
from transformers import AutoTokenizer

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import ConfigManager, setup_logging, get_device_config
from src.preprocessing import EmotionPreprocessor
from src.models import create_model, EmotionPredictor
from utils.model_loading import load_model_and_assets, filter_model_config

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make predictions with emotion recognition models", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", "-m", type=str, required=True, help="Path to trained model directory or checkpoint")
    parser.add_argument("--config-path", type=str, help="Path to model configuration file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t", type=str, help="Single text to predict")
    group.add_argument("--text-file", type=str, help="File containing texts to predict (one per line)")
    group.add_argument("--csv-file", type=str, help="CSV file with text column to predict")
    parser.add_argument("--text-column", type=str, default="text", help="Name of text column in CSV file")
    parser.add_argument("--output-file", "-o", type=str, help="Output file for predictions (default: print to console)")
    parser.add_argument("--output-format", type=str, choices=["json", "csv", "txt"], default="json", help="Output format")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for prediction")
    parser.add_argument("--include-probabilities", action="store_true", help="Include prediction probabilities in output")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--debug-predictions",
        action="store_true",
        help="Show ensemble debug predictions/logits during prediction"
    )
    return parser.parse_args()

def load_input_texts(args):
    """Load input texts from command line arguments."""
    if args.text:
        return [args.text]
    elif args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    elif args.csv_file:
        df = pd.read_csv(args.csv_file)
        if args.text_column not in df.columns:
            raise ValueError(f"Column '{args.text_column}' not found in CSV file")
        return df[args.text_column].dropna().tolist()
    else:
        raise ValueError("No input texts found")

def make_predictions(model, config, texts, device, args, tokenizers=None, vocabs=None):
    """Make predictions on input texts."""
    preproc_kwargs = {}
    preproc_section = getattr(config, 'preprocessing', {})
    for k in ['handle_emojis', 'expand_contractions', 'remove_stopwords', 'lemmatize', 'normalize_case', 'handle_social_media', 'min_length', 'max_length']:
        if hasattr(preproc_section, k):
            preproc_kwargs[k] = getattr(preproc_section, k)
    preprocessor = EmotionPreprocessor(**preproc_kwargs)
    # For ensemble, pass tokenizers/vocabs to predictor if needed
    predictor = EmotionPredictor(
        model=model,
        tokenizer=tokenizers.get('distilbert') or tokenizers.get('twitter-roberta') if tokenizers else None,
        preprocessor=preprocessor,
        label_names=getattr(config, 'emotions', None),
        device=device
    )
    # Make predictions
    logger.info(f"Making predictions on {len(texts)} texts...")
    return predictor.predict(texts, return_probabilities=args.include_probabilities, debug_predictions=args.debug_predictions)

def save_output(results, args):
    """Save or print results."""
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif args.output_format == "csv":
            pd.DataFrame(results).to_csv(output_path, index=False)
        
        elif args.output_format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"Text: {result['text']}\n")
                    if 'emotion' in result:
                        prob_str = f" (probability: {result['confidence']:.3f})" if 'confidence' in result else ""
                        f.write(f"Predicted emotion: {result['emotion']}{prob_str}\n")
                    f.write("\n")
        
        logger.info(f"Results saved to: {output_path}")
    
    else:
        # Print to console
        print("\n" + "="*50)
        print("EMOTION PREDICTIONS")
        print("="*50)
        
        for result in results:
            print(f"\nText: {result['text']}")
            if 'emotion' in result:
                prob_str = f" (probability: {result['confidence']:.3f})" if 'confidence' in result else ""
                print(f"Predicted emotion: {result['emotion']}{prob_str}")
        
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
        model, config, tokenizers, vocabs = load_model_and_assets(args.model_path, args.config_path)
        
        # Override device if needed
        if args.force_cpu:
            config.device.force_cpu = True
        
        # Get device
        device = get_device_config(config)
        model = model.to(device)
        logger.info(f"Using device: {device}")
        
        # Load input data
        texts = load_input_texts(args)
        
        # Make predictions
        predictions = make_predictions(model, config, texts, device, args, tokenizers=tokenizers, vocabs=vocabs)
        
        # Save output
        save_output(predictions, args)
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
