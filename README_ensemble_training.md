# Ensemble Model Training Guide

This project now supports two strategies for ensemble model training:

## 1. Joint Training (strategy: joint)
- All submodels (e.g., DistilBERT, Twitter RoBERTa, BiLSTM) are trained together as part of the ensemble.
- The optimizer updates all submodel parameters jointly.
- Use this for end-to-end ensemble learning.

**How to use:**
```powershell
python scripts/train.py --config-file configs/ensemble_config.yaml --data-path data/splits/train.csv --epochs 5 --batch-size 16 --learning-rate 1e-5
```
- In your `ensemble_config.yaml`, set:
  ```yaml
  training:
    strategy: joint
  ```

## 2. Individual Training (strategy: individual)
- Each submodel is trained separately on the same data.
- After training, the best weights for each submodel are saved.
- The ensemble is then constructed from these trained submodels for evaluation or further use.
- Use this if you want to pretrain submodels and then ensemble their predictions.

**How to use:**
```powershell
python scripts/train.py --config-file configs/ensemble_config.yaml --data-path data/splits/train.csv --epochs 5 --batch-size 16 --learning-rate 1e-5
```
- In your `ensemble_config.yaml`, set:
  ```yaml
  training:
    strategy: individual
  ```

## Notes
- The `strategy` key in your config controls the behavior.
- Model checkpoints for each submodel are saved in the output directory when using `individual`.
- For both strategies, the ensemble model is evaluated after training.
- If you want to use pretrained submodels, place their checkpoints in the output directory with names like `distilbert_best_model.pt`, `twitter_roberta_best_model.pt`, etc.

## Example Config Snippet
```yaml
model:
  type: "ensemble"
  models:
    - type: "distilbert"
      model_name: "distilbert-base-uncased"
      weight: 0.4
    - type: "twitter_roberta"
      model_name: "cardiffnlp/twitter-roberta-base-emotion"
      weight: 0.4
    - type: "bilstm"
      weight: 0.2
  num_classes: 6
  voting_strategy: "soft"
training:
  strategy: joint  # or 'individual'
  batch_size: 16
  learning_rate: 1e-5
  num_epochs: 5
```

## Troubleshooting
- If your ensemble collapses to a single class, ensure your submodels are being trained (not random weights) and the correct strategy is set.
- For `individual`, check that each submodel's checkpoint is saved and loaded correctly.
- For `joint`, all submodels should be updated during training.
