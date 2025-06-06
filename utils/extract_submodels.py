import torch
from pathlib import Path
import sys
import yaml
import json

# Adjust these paths as needed
ensemble_dir = Path("models/ensemble_20250605_234139")  # or your actual output dir
ensemble_ckpt = ensemble_dir / "best_model.pt"  # or "final_model.pt" or similar

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.models import create_model

# Load config
def save_config(config, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

with open(ensemble_dir / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Build ensemble model, skipping submodels that can't be extracted
submodels = []
submodel_names = []
submodel_cfgs = []
weights = []
missing = []
for i, sub_cfg in enumerate(config['model']['models']):
    submodel_type = sub_cfg['type']
    orig_type = submodel_type
    # Map config type to model type expected by create_model
    if submodel_type == 'twitter_roberta':
        submodel_type = 'twitter-roberta'
    # Remove 'type' and 'weight' keys for model constructor
    submodel_cfg = {k: v for k, v in sub_cfg.items() if k not in ('type', 'weight')}
    if 'num_classes' in submodel_cfg:
        submodel_cfg['num_labels'] = submodel_cfg.pop('num_classes')
    can_extract = True
    if submodel_type == 'bilstm':
        vocab_path = ensemble_dir / 'bilstm_vocab.json'
        if not vocab_path.exists():
            print(f"WARNING: BiLSTM vocab file not found: {vocab_path}. Skipping BiLSTM submodel.")
            can_extract = False
        else:
            with open(vocab_path, 'r', encoding='utf-8') as vf:
                vocab = json.load(vf)
            submodel_cfg['vocab_size'] = len(vocab)
    if can_extract:
        try:
            submodels.append(create_model(submodel_type, submodel_cfg))
            submodel_names.append(orig_type if orig_type != 'twitter_roberta' else 'twitter-roberta')
            submodel_cfgs.append(sub_cfg)
            weights.append(sub_cfg.get('weight', 1.0))
        except Exception as e:
            print(f"WARNING: Could not create submodel {orig_type}: {e}. Skipping.")
            can_extract = False
    if not can_extract:
        missing.append(i)

if not submodels:
    raise RuntimeError("No submodels could be extracted from the ensemble. Aborting.")

# Load ensemble checkpoint
ckpt = torch.load(ensemble_ckpt, map_location='cpu')
if 'model_state_dict' in ckpt:
    model_state_dict = ckpt['model_state_dict']
else:
    model_state_dict = ckpt

# Save each submodel, only if its state_dict loads successfully
final_submodels = []
final_names = []
final_cfgs = []
final_weights = []
for idx, (name, submodel, cfg, weight) in enumerate(zip(submodel_names, submodels, submodel_cfgs, weights)):
    # Try to load state_dict for this submodel
    key_prefix = f"models.{idx}."
    # Extract sub-state_dict for this submodel
    sub_state_dict = {k[len(key_prefix):]: v for k, v in model_state_dict.items() if k.startswith(key_prefix)}
    try:
        submodel.load_state_dict(sub_state_dict, strict=True)
        out_path = ensemble_dir / f'{name}_best_model.pt'
        torch.save({'model_state_dict': submodel.state_dict()}, out_path)
        print(f"Saved {name} submodel to {out_path}")
        final_submodels.append(submodel)
        final_names.append(name)
        final_cfgs.append(cfg)
        final_weights.append(weight)
    except Exception as e:
        print(f"WARNING: Could not load state_dict for submodel {name} (index {idx}): {e}. Skipping.")

if not final_submodels:
    raise RuntimeError("No submodels could be extracted from the ensemble. Aborting.")

# Adjust weights to sum to 1
if final_weights:
    total = sum(final_weights)
    final_weights = [w/total for w in final_weights]

# Update config: remove missing submodels and adjust weights
if len(final_submodels) != len(submodels):
    print(f"Updating config.yaml: removing submodels that could not be extracted and adjusting weights.")
    new_models = []
    for i, cfg in enumerate(final_cfgs):
        cfg = dict(cfg)
        cfg['weight'] = float(final_weights[i])
        new_models.append(cfg)
    config['model']['models'] = new_models
    save_config(config, ensemble_dir / "config.yaml")
    print(f"Updated config.yaml with {len(new_models)} submodels and normalized weights.")