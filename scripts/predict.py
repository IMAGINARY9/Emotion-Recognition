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
from src.visualization import plot_word_importances, plot_ensemble_votes, plot_confidence_progression

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
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions to return per input (adaptive: if model is uncertain, may return more even if top-k=1)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize prediction explanations and ensemble voting (saved to visualizations/<model_dir_name>/)"
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
    """Make predictions on input texts with support for top-k/adaptive output and visualization."""
    preproc_kwargs = {}
    preproc_section = getattr(config, 'preprocessing', {})
    for k in ['handle_emojis', 'expand_contractions', 'remove_stopwords', 'lemmatize', 'normalize_case', 'handle_social_media', 'min_length', 'max_length']:
        if hasattr(preproc_section, k):
            preproc_kwargs[k] = getattr(preproc_section, k)
    preprocessor = EmotionPreprocessor(**preproc_kwargs)
    # --- Do NOT pass vocab to EmotionPredictor; vocab is attached to the model if needed ---
    predictor_kwargs = dict(
        model=model,
        tokenizer=tokenizers.get('distilbert') or tokenizers.get('twitter-roberta') if tokenizers else None,
        preprocessor=preprocessor,
        label_names=getattr(config, 'emotions', None),
        device=device
    )
    predictor = EmotionPredictor(**predictor_kwargs)
    logger.info(f"Making predictions on {len(texts)} texts...")

    # --- Adaptive top-k logic ---
    results = []
    for idx, text in enumerate(texts):
        pred = predictor.predict([text], return_probabilities=True, debug_predictions=args.debug_predictions)
        # pred is a list of dicts, one per input
        pred = pred[0]
        # Get sorted probabilities
        if 'probabilities' in pred:
            probs = pred['probabilities']
            labels = list(probs.keys())
            prob_values = list(probs.values())
            sorted_indices = sorted(range(len(prob_values)), key=lambda i: prob_values[i], reverse=True)
            topk = args.top_k
            # Adaptive: if top-1 and top-2 are close, or ensemble disagrees, return both
            if topk == 1:
                if len(prob_values) > 1 and abs(prob_values[sorted_indices[0]] - prob_values[sorted_indices[1]]) < 0.10:
                    topk = 2
                # For ensemble: if available, check for disagreement (stub)
                if 'ensemble_votes' in pred and len(set(pred['ensemble_votes'].values())) > 1:
                    topk = max(topk, len(set(pred['ensemble_votes'].values())))
            top_preds = [
                {
                    'label': labels[i],
                    'probability': prob_values[i]
                } for i in sorted_indices[:topk]
            ]
            pred['predictions'] = top_preds
        results.append(pred)

        # Visualization using external module
        if args.visualize:
            try:
                from pathlib import Path
                model_path = Path(args.model_path)
                model_dir_name = model_path.parent.name if model_path.is_file() else model_path.name
                vis_dir = Path('visualizations') / model_dir_name
                vis_dir.mkdir(parents=True, exist_ok=True)
                saved_files = []
                plot_types = []

                # 1. Ensemble votes (if present)
                # Try to plot submodel probabilities if available
                submodel_probs = None
                if 'ensemble_submodel_explanations' in pred and pred['ensemble_submodel_explanations']:
                    submodel_probs = {
                        submodel: sub_expl['probabilities']
                        for submodel, sub_expl in pred['ensemble_submodel_explanations'].items()
                        if 'probabilities' in sub_expl
                    }
                if 'ensemble_votes' in pred and pred['ensemble_votes']:
                    fname = plot_ensemble_votes(pred['ensemble_votes'], text, vis_dir, idx=idx, submodel_probs=submodel_probs)
                    saved_files.append(fname)
                    plot_types.append('ensemble_votes')

                # 2. Submodel explanations (if present)
                if 'ensemble_submodel_explanations' in pred and pred['ensemble_submodel_explanations']:
                    for submodel_name, sub_expl in pred['ensemble_submodel_explanations'].items():
                        wi = sub_expl.get('word_importances')
                        if wi and len(set(wi['importances'])) > 1:
                            fname = plot_word_importances(wi['words'], wi['importances'], text, vis_dir, idx=f"{idx}_{submodel_name}")
                            saved_files.append(fname)
                            plot_types.append(f'submodel_word_importances_{submodel_name}')
                        # Confidence progression for submodel (if present)
                        cp = sub_expl.get('confidence_progression')
                        if cp and 'tokens' in cp and 'confidences' in cp and len(cp['tokens']) == len(cp['confidences']):
                            fname = plot_confidence_progression(cp['tokens'], cp['confidences'], text, vis_dir, idx=f"{idx}_{submodel_name}")
                            saved_files.append(fname)
                            plot_types.append(f'submodel_confidence_progression_{submodel_name}')

                # 3. Main model word importances (if present and not uniform/fake)
                wi = pred.get('word_importances')
                if wi and len(set(wi['importances'])) > 1:
                    fname = plot_word_importances(wi['words'], wi['importances'], text, vis_dir, idx=idx)
                    saved_files.append(fname)
                    plot_types.append('main_word_importances')

                # 4. Main model confidence progression (if present)
                cp = pred.get('confidence_progression')
                if cp and 'tokens' in cp and 'confidences' in cp and len(cp['tokens']) == len(cp['confidences']):
                    fname = plot_confidence_progression(cp['tokens'], cp['confidences'], text, vis_dir, idx=idx)
                    saved_files.append(fname)
                    plot_types.append('main_confidence_progression')

                # 5. If no meaningful explanation, create a summary txt file for the prediction
                if not saved_files:
                    summary_path = vis_dir / f'prediction_{idx}_summary.txt'
                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write(f"Text: {text}\n")
                        f.write(f"Predicted emotion: {pred.get('emotion')}\n")
                    saved_files.append(str(summary_path))
                    print(f"No plot generated for prediction {idx}, summary saved:")
                else:
                    print(f"Plots generated for prediction {idx}: {', '.join(plot_types)}")
                for fname in saved_files:
                    print(f"[Visualization saved] {fname}")
            except Exception as ve:
                logger.warning(f"Visualization failed: {ve}")
    return results

def save_output(results, args):
    """Save or print results."""
    if args.output_file:
        output_path = Path(args.output_file)
        # --- Save predictions in reports/<model_dir_name>/ if not absolute path ---
        if not output_path.is_absolute():
            # Use model dir name for subfolder
            model_path = Path(args.model_path)
            if model_path.is_file():
                model_dir_name = model_path.parent.name
            else:
                model_dir_name = model_path.name
            output_path = Path("reports") / model_dir_name / output_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif args.output_format == "csv":
            import pandas as pd
            pd.DataFrame(results).to_csv(output_path, index=False)
        elif args.output_format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(str(item) + "\n")
        print(f"Predictions saved to: {output_path}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))

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
