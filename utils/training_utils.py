"""
Shared utilities for training, evaluation, and prediction in emotion recognition.
"""
import os
import json
from pathlib import Path
import logging
import pandas as pd
from utils.data_utils import EmotionDataLoader, setup_data_directories, download_sample_data
from src.preprocessing import EmotionPreprocessor, EmotionDataProcessor
from src.models import EmotionDataset, JointEnsembleDataset

logger = logging.getLogger(__name__)

def load_and_prepare_data(config, data_path: str = None, use_sample: bool = False):
    """
    Load and prepare training/validation/test dataframes.
    Returns: data_loader, train_df, val_df, test_df
    """
    data_loader = EmotionDataLoader(config.paths.data_dir)
    setup_data_directories(config.paths.data_dir)
    if not data_path:
        splits_dir = Path(config.paths.data_dir) / "splits"
        split_files = [splits_dir / "training.csv", splits_dir / "train.csv"]
        for file_path in split_files:
            if file_path.exists():
                data_path = str(file_path)
                logger.info(f"Found split data file: {data_path}")
                break
    if not data_path:
        raw_dir = Path(config.paths.data_dir) / "raw"
        raw_files = [raw_dir / "emotions.csv", raw_dir / "sample_emotions.csv"]
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
    logger.info(f"Loading data from: {data_path}")
    df = data_loader.load_csv_data(data_path)
    df = data_loader.validate_data(df, config.data.text_column, config.data.label_column)
    train_df, val_df, test_df = data_loader.split_data(
        df,
        train_size=config.data.train_split,
        val_size=config.data.val_split,
        test_size=config.data.test_split
    )
    splits_dir = os.path.join(config.paths.data_dir, "splits")
    data_loader.save_splits(train_df, val_df, test_df, splits_dir)
    return data_loader, train_df, val_df, test_df

def create_datasets(config, data_loader, train_df, val_df, test_df, models_dir=None, upsample_minority=True):
    """Create PyTorch datasets for individual and joint ensemble strategies."""
    from src.models import JointEnsembleDataset, EmotionDataset
    from src.preprocessing import EmotionPreprocessor, EmotionDataProcessor
    import pandas as pd
    import logging
    from pathlib import Path
    logger = logging.getLogger(__name__)

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

        from transformers import AutoTokenizer
        sub_models_tokenizers_and_configs = []
        for model_detail_cfg_dict in config.model.models:
            model_type = model_detail_cfg_dict['type']
            model_name = model_detail_cfg_dict.get('model_name', None) or model_detail_cfg_dict.get('name', model_type)
            sub_tokenizer = None
            # Always prefer model_name if present
            tokenizer_path = model_detail_cfg_dict.get('model_name', None) or \
                             model_detail_cfg_dict.get('model_name_or_path', None) or \
                             model_detail_cfg_dict.get('pretrained_model_name_or_path', None)
            if not tokenizer_path:
                if model_type in ['distilbert']:
                    tokenizer_path = 'distilbert-base-uncased'
                elif model_type in ['twitter_roberta', 'twitter-roberta', 'roberta']:
                    tokenizer_path = 'cardiffnlp/twitter-roberta-base-emotion'
            if tokenizer_path and model_type != 'bilstm':
                try:
                    from transformers import AutoTokenizer
                    sub_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                    logger.info(f"Loaded tokenizer for sub-model '{model_name}' ({model_type}) from {tokenizer_path}")
                except Exception as e:
                    logger.error(f"Failed to load tokenizer for sub-model '{model_name}' ({model_type}) from {tokenizer_path}: {e}", exc_info=True)
                    raise
            elif model_type == 'bilstm':
                if bilstm_tokenizer is None:
                    logger.error("BiLSTM sub-model specified for joint ensemble, but BiLSTM tokenizer (vocab-based) is not available.")
                    raise ValueError("BiLSTM tokenizer required for BiLSTM sub-model in joint ensemble.")
                sub_tokenizer = bilstm_tokenizer
                logger.info(f"Using pre-built BiLSTM tokenizer for sub-model '{model_name}' ({model_type})")
            sub_max_length = model_detail_cfg_dict.get('max_length', config.data.get('max_length', 128))
            if sub_tokenizer:
                sub_models_tokenizers_and_configs.append({
                    'name': model_name,
                    'tokenizer': sub_tokenizer,
                    'max_length': sub_max_length,
                    'type': model_type
                })
            else:
                logger.warning(f"No tokenizer could be loaded or prepared for sub-model {model_name} ({model_type}). It will be excluded from JointEnsembleDataset.")
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
        return datasets, data_processor, vocab, sub_models_tokenizers_and_configs
    # Prepare for training and create datasets (individual strategy)
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
