# Emotion Recognition Project

A comprehensive emotion recognition system for text classification using transformer models and traditional ML approaches. This project implements state-of-the-art emotion detection models that can classify text into 6 emotion categories: **sadness**, **joy**, **love**, **anger**, **fear**, and **surprise**.

## Features

- **Multiple Model Architectures**: DistilBERT, Twitter RoBERTa, BiLSTM, and Ensemble models
- **Social Media Optimized**: Specialized preprocessing for social media text (hashtags, mentions, emojis)
- **Comprehensive Preprocessing**: Smart text cleaning with configurable options
- **Advanced Training**: Learning rate scheduling, early stopping, gradient clipping
- **Detailed Evaluation**: Multiple metrics, confusion matrices, embedding visualizations
- **Easy-to-Use**: Command-line tools and Python API
- **Configurable**: YAML-based configuration system
- **Production Ready**: Model serialization, batch prediction, and deployment utilities

## Project Structure

```
Emotion-Recognition/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── models.py                # Model architectures
│   ├── preprocessing.py         # Text preprocessing
│   ├── training.py              # Training pipeline
│   ├── evaluation.py            # Evaluation metrics
│   ├── data_utils.py            # Data loading utilities
│   └── config.py                # Configuration management
├── configs/                      # Model configurations
│   ├── distilbert_config.yaml   # DistilBERT configuration
│   ├── twitter_roberta_config.yaml # Twitter RoBERTa configuration
│   ├── bilstm_config.yaml       # BiLSTM configuration
│   └── ensemble_config.yaml     # Ensemble configuration
├── scripts/                      # Command-line tools
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── predict.py               # Prediction script
├── data/                         # Data directory
├── models/                       # Saved models
├── outputs/                      # Training outputs
├── logs/                         # Log files
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── setup.bat                     # Windows setup script
├── setup.ps1                     # PowerShell setup script
├── activate.bat                  # Environment activation
└── README.md                     # This file
```

## Installation

### Automatic Setup (Windows)

1. **Run the setup script:**
   ```bash
   # Using batch script
   setup.bat
   
   # Or using PowerShell
   powershell -ExecutionPolicy Bypass -File setup.ps1
   ```

2. **Activate the environment:**
   ```bash
   activate.bat
   ```

### Manual Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Emotion-Recognition
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv emotion_env
   
   # Windows
   emotion_env\Scripts\activate
   
   # Linux/Mac
   source emotion_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package:**
   ```bash
   pip install -e .
   ```

## Quick Start

### 1. Train a Model

```bash
# Train DistilBERT model with sample data
python scripts/train.py --config distilbert --use-sample-data --experiment-name my_first_model

# Train with custom data
python scripts/train.py --config distilbert --data-path data/emotions.csv --epochs 5
```

### 2. Make Predictions

```bash
# Single text prediction
python scripts/predict.py -m outputs/my_first_model -t "I love this amazing day!"

# Batch prediction from file
python scripts/predict.py -m outputs/my_first_model --text-file input_texts.txt -o predictions.json

# CSV file prediction
python scripts/predict.py -m outputs/my_first_model --csv-file data.csv --text-column "text" -o results.csv
```

### 3. Evaluate Model

```bash
# Evaluate on test set
python scripts/evaluate.py -m outputs/my_first_model --output-dir evaluation_results

# Evaluate with error analysis
python scripts/evaluate.py -m outputs/my_first_model --error-analysis --save-predictions
```

## Model Types

### 1. DistilBERT (`distilbert`)
- **Best for**: General emotion recognition
- **Speed**: Fast
- **Accuracy**: High
- **Memory**: Low

### 2. Twitter RoBERTa (`twitter_roberta`)
- **Best for**: Social media text
- **Speed**: Medium
- **Accuracy**: Very High
- **Memory**: Medium

### 3. BiLSTM (`bilstm`)
- **Best for**: Resource-constrained environments
- **Speed**: Very Fast
- **Accuracy**: Medium
- **Memory**: Very Low

### 4. Ensemble (`ensemble`)
- **Best for**: Maximum accuracy
- **Speed**: Slow
- **Accuracy**: Highest
- **Memory**: High

## Configuration

Each model type has its own configuration file in the `configs/` directory. Key parameters:

```yaml
model:
  type: "distilbert"
  num_classes: 6
  dropout_rate: 0.3
  max_length: 128

training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 10
  early_stopping:
    enabled: true
    patience: 3

preprocessing:
  lowercase: true
  remove_urls: true
  emoji_handling: "convert"  # remove, convert, keep
```

## Configs Overview

The `configs/` directory contains YAML configuration files for each supported model architecture and for ensemble setups. These files define all hyperparameters, preprocessing options, and data handling settings for reproducible experiments and easy customization.

**Available Config Files:**

- `distilbert_config.yaml`: Configuration for the DistilBERT-based model. Includes transformer-specific parameters, training settings, and preprocessing suited for general text.
- `twitter_roberta_config.yaml`: Configuration for the Twitter RoBERTa model. Optimized for social media data, with lighter preprocessing and settings tailored for tweets.
- `bilstm_config.yaml`: Configuration for the BiLSTM model. Includes options for embedding, vocabulary, and more aggressive text cleaning.
- `ensemble_config.yaml`: Configuration for combining multiple models (DistilBERT, Twitter RoBERTa, BiLSTM) into an ensemble. Specifies voting strategies and references individual model configs.

**Key Sections in Each Config:**
- `model`: Model type, architecture-specific parameters, and (for ensembles) weights and strategies.
- `training`: Batch size, learning rate, epochs, scheduler, early stopping, and optimizer settings.
- `data`: Data splits, column names, batch sizes, and shuffling.
- `preprocessing`: Text cleaning options (lowercasing, emoji handling, stopword removal, etc.).
- `evaluation`: Metrics and visualization options.
- `paths`: Directories for data, models, outputs, logs, and cache.
- `logging` & `device`: Logging format/level and device selection.
- `emotions`: List of emotion labels (should match your dataset).

**How to Use:**
- Select or modify a config file in `configs/` to match your experiment needs.
- Pass the config name (without `.yaml`) to scripts, e.g., `--config distilbert`.
- For custom experiments, copy an existing config, adjust parameters, and use `--config-file my_config.yaml`.

See each YAML file in `configs/` for detailed, model-specific options and comments.

## Python API

```python
from src import EmotionPredictor, EmotionPreprocessor, create_model, quick_predict

# Quick prediction
result = quick_predict("I'm so happy today!", "outputs/my_model")
print(result)  # Output: "joy"

# Advanced usage
from src import ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_model_config("distilbert")

# Create model
model = create_model(
    model_type="distilbert",
    num_classes=6
)

# Create preprocessor
preprocessor = EmotionPreprocessor(
    lowercase=True,
    emoji_handling="convert"
)

# Create predictor
predictor = EmotionPredictor(
    model=model,
    preprocessor=preprocessor,
    emotion_labels=["sadness", "joy", "love", "anger", "fear", "surprise"]
)

# Make predictions
emotions = predictor.predict_batch([
    "I love this!",
    "This makes me angry",
    "I'm feeling sad today"
])
```

## Training Custom Models

### 1. Prepare Your Data

Your CSV file should have columns:
- `text`: The text to classify
- `label`: The emotion label (sadness, joy, love, anger, fear, surprise)

Example:
```csv
text,label
"I'm so happy today!",joy
"This is frustrating",anger
"I love spending time with family",love
```

### 2. Custom Configuration

Create a custom config file:

```yaml
# my_config.yaml
model:
  type: "distilbert"
  num_classes: 6
  max_length: 256  # Longer sequences

training:
  batch_size: 16   # Smaller batch size
  learning_rate: 1e-5  # Lower learning rate
  num_epochs: 15   # More epochs

data:
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1
```

### 3. Train with Custom Config

```bash
python scripts/train.py --config-file my_config.yaml --data-path my_data.csv
```

## Evaluation Metrics

The evaluation script provides comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Macro and weighted F1 scores
- **Precision/Recall**: Per-class and averaged
- **ROC AUC**: Area under the ROC curve
- **Confusion Matrix**: Visual confusion matrix
- **Embeddings Plot**: t-SNE visualization of learned representations

## Advanced Features

### Ensemble Models

Combine multiple models for better performance:

```bash
# Train ensemble model
python scripts/train.py --config ensemble --data-path data/emotions.csv
```

### Hyperparameter Tuning

Override config parameters from command line:

```bash
python scripts/train.py --config distilbert --learning-rate 1e-5 --batch-size 16 --epochs 20
```

### Error Analysis

Get detailed error analysis:

```bash
python scripts/evaluate.py -m outputs/my_model --error-analysis --save-predictions
```

### Batch Prediction with Probabilities

```bash
python scripts/predict.py -m outputs/my_model --csv-file data.csv --include-probabilities --top-k 3
```

## Preprocessing Options

The preprocessing pipeline supports various text cleaning options:

- **URL Removal**: Remove or keep URLs
- **Mention Handling**: Remove or keep @mentions
- **Hashtag Processing**: Remove, keep, or clean hashtags
- **Emoji Handling**: Remove, convert to text, or keep
- **Contraction Expansion**: "don't" → "do not"
- **Case Normalization**: Convert to lowercase
- **Stopword Removal**: Remove common words
- **Special Characters**: Clean or preserve

## Deployment

### Model Export

```python
# Save model for deployment
trainer.save_model("production_model")

# Load in production
model = torch.load("production_model/final_model.pt")
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "scripts/predict.py"]
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest tests/ --cov=src
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- emoji

See `requirements.txt` for complete list.

## Acknowledgments

- Hugging Face Transformers library
- Cardiff NLP for Twitter RoBERTa model
- The emotion recognition research community

