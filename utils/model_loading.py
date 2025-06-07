import inspect
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from src.config import ConfigManager
from src.models import create_model, BiLSTMEmotionModel, DistilBERTEmotionModel, TwitterRoBERTaEmotionModel

def filter_model_config(model_class, config_dict):
    """Return a config dict with only keys accepted by the model_class __init__."""
    sig = inspect.signature(model_class.__init__)
    valid_keys = set(sig.parameters.keys()) - {'self', 'args', 'kwargs'}
    config = dict(config_dict)
    if 'num_classes' in config and 'num_labels' in valid_keys:
        config['num_labels'] = config.pop('num_classes')
    if 'dropout_rate' in config and 'dropout' in valid_keys:
        config['dropout'] = config.pop('dropout_rate')
    # Remove pretrained_embeddings if not a tensor
    if 'pretrained_embeddings' in config:
        val = config['pretrained_embeddings']
        if not isinstance(val, torch.Tensor):
            config.pop('pretrained_embeddings')
    filtered = {k: v for k, v in config.items() if k in valid_keys}
    return filtered

def load_model_and_assets(model_path: str, config_path: str = None):
    """Load model, config, tokenizers, and vocabularies for all model types."""
    model_path = Path(model_path)
    if config_path:
        config = ConfigManager().load_config(config_path)
    elif (model_path / "config.yaml").exists():
        config = ConfigManager().load_config(model_path / "config.yaml")
    else:
        raise FileNotFoundError("Configuration file not found. Please specify --config-path")
    model_config = dict(config.model)
    model_type = model_config.pop('type').replace('_', '-')
    tokenizers = {}
    vocabs = {}
    model_class_map = {
        'bilstm': BiLSTMEmotionModel,
        'distilbert': DistilBERTEmotionModel,
        'twitter-roberta': TwitterRoBERTaEmotionModel,
        'twitter_roberta': TwitterRoBERTaEmotionModel,
        'roberta': TwitterRoBERTaEmotionModel
    }
    # --- Ensemble ---
    if model_type == 'ensemble':
        submodels, weights = [], []
        for sub_cfg in config.model.models:
            sub_cfg = dict(sub_cfg)
            weight = sub_cfg.pop('weight', 1.0/len(config.model.models))
            weights.append(weight)
            sub_type = sub_cfg.pop('type').replace('_', '-')
            model_class = model_class_map.get(sub_type)
            # Tokenizer
            if sub_type == 'distilbert':
                tokenizer_dir = model_path / 'distilbert_best_model_tokenizer'
                tokenizers['distilbert'] = AutoTokenizer.from_pretrained(str(tokenizer_dir)) if tokenizer_dir.exists() else AutoTokenizer.from_pretrained('distilbert-base-uncased')
            elif sub_type == 'twitter-roberta':
                tokenizer_dir = model_path / 'twitter-roberta_best_model_tokenizer'
                tokenizers['twitter-roberta'] = AutoTokenizer.from_pretrained(str(tokenizer_dir)) if tokenizer_dir.exists() else AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-emotion')
            # Vocab
            if sub_type == 'bilstm':
                vocab_path = model_path / 'bilstm_vocab.json'
                if vocab_path.exists():
                    with open(vocab_path, 'r', encoding='utf-8') as f:
                        vocab = json.load(f)
                    sub_cfg['vocab'] = vocab
                    sub_cfg['vocab_size'] = len(vocab)
                    vocabs['bilstm'] = vocab
                else:
                    print(f"[Ensemble Warning] BiLSTM vocab file not found: {vocab_path}. Skipping BiLSTM submodel.")
                    continue  # Skip this submodel if vocab is missing
            # Filter config for submodel
            sub_cfg_filtered = filter_model_config(model_class, sub_cfg)
            submodel = create_model(model_type=sub_type, model_config=sub_cfg_filtered, vocab=sub_cfg.get('vocab', None))
            if sub_type == 'distilbert':
                checkpoint_path = model_path / 'distilbert_best_model.pt'
            elif sub_type == 'twitter-roberta':
                checkpoint_path = model_path / 'twitter-roberta_best_model.pt'
            elif sub_type == 'bilstm':
                checkpoint_path = model_path / 'bilstm_best_model.pt'
            else:
                continue
            if not checkpoint_path.exists():
                continue
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                submodel.load_state_dict(checkpoint['model_state_dict'])
            else:
                submodel.load_state_dict(checkpoint)
            submodels.append(submodel)
        model = create_model(model_type='ensemble', model_config={'models': submodels, 'weights': weights[:len(submodels)]})
        return model, config, tokenizers, vocabs
    # --- Single model ---
    else:
        model_class = model_class_map.get(model_type)
        if model_type == 'distilbert':
            tokenizer_dir = model_path / 'distilbert_best_model_tokenizer'
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir)) if tokenizer_dir.exists() else AutoTokenizer.from_pretrained('distilbert-base-uncased')
            tokenizers['distilbert'] = tokenizer
        elif model_type == 'twitter-roberta':
            tokenizer_dir = model_path / 'twitter-roberta_best_model_tokenizer'
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir)) if tokenizer_dir.exists() else AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-emotion')
            tokenizers['twitter-roberta'] = tokenizer
        elif model_type == 'bilstm':
            # Always load vocab from model_path/bilstm_vocab.json
            vocab_path = model_path / 'bilstm_vocab.json'
            if not vocab_path.exists():
                # Also try model_path.parent for legacy support
                vocab_path = model_path.parent / 'bilstm_vocab.json'
            if vocab_path.exists():
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab = json.load(f)
                vocabs['bilstm'] = vocab
                model_config['vocab'] = vocab
                model_config['vocab_size'] = len(vocab)
            else:
                raise FileNotFoundError(f"BiLSTM vocab file not found at {vocab_path}. Cannot load model.")
        # Filter config for single model
        model_config_filtered = filter_model_config(model_class, model_config)
        # When creating the model, pass vocab if available
        model = create_model(model_type=model_type, model_config=model_config_filtered, vocab=vocab)
        checkpoint_path = None
        if model_path.is_file() and model_path.suffix in ['.pt', '.pth']:
            checkpoint_path = model_path
        elif (model_path / "best_model.pt").exists():
            checkpoint_path = model_path / "best_model.pt"
        elif (model_path / "final_model.pt").exists():
            checkpoint_path = model_path / "final_model.pt"
        else:
            raise FileNotFoundError(f"Model checkpoint not found in {model_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            # When loading model state dict, allow missing/unexpected keys
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            except RuntimeError as e:
                import logging
                logging.warning(f"Model loaded with missing/unexpected keys: {e}")
        else:
            model.load_state_dict(checkpoint)
        return model, config, tokenizers, vocabs
