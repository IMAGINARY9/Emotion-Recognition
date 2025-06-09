"""
DEPRECATED: This module contains utility functions that are no longer used.
Consider removing this file in a future cleanup.
"""
import os
import torch
from pathlib import Path
import warnings

def extract_and_save_submodels(ensemble_model, output_dir):
    """
    DEPRECATED: This function is no longer used and may be removed in future versions.
    
    Extracts submodels from a trained ensemble and saves them individually.
    Args:
        ensemble_model: The trained ensemble model (should have .models attribute)
        output_dir: Directory to save submodels
    """
    warnings.warn(
        "extract_and_save_submodels is deprecated and may be removed in future versions.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if not hasattr(ensemble_model, 'models'):
        raise ValueError("Provided model is not an ensemble or does not have submodels.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, submodel in enumerate(ensemble_model.models):
        submodel_path = output_dir / f"submodel_{idx}_{type(submodel).__name__}.pt"
        torch.save(submodel.state_dict(), submodel_path)
        print(f"Saved submodel {idx} to {submodel_path}")
