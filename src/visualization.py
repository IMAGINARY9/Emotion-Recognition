import matplotlib.pyplot as plt
from pathlib import Path

def plot_word_importances(words, importances, text, vis_dir, idx=None):
    """Plot and save word importances for a given text input."""
    plt.figure(figsize=(max(6, len(words)*0.7), 2.5))
    bars = plt.bar(words, importances, color='skyblue', edgecolor='black')
    plt.title(f"Word importances for: {text[:40]}...", fontsize=11)
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    fname = f"word_importances_{idx if idx is not None else ''}.png"
    plt.savefig(Path(vis_dir) / fname)
    plt.close()
    return str(Path(vis_dir) / fname)

def plot_ensemble_votes(votes, text, vis_dir, idx=None, submodel_probs=None):
    """Plot and save ensemble voting results for a given text input.
    If submodel_probs is provided, plot grouped bars of probabilities for each label and submodel.
    """
    import numpy as np
    from collections import Counter
    plt.figure(figsize=(8, 3.5))
    if submodel_probs:
        # submodel_probs: dict of submodel_name -> {label: prob, ...}
        labels = list(next(iter(submodel_probs.values())).keys())
        submodels = list(submodel_probs.keys())
        x = np.arange(len(labels))
        width = 0.8 / len(submodels)
        for i, submodel in enumerate(submodels):
            probs = [submodel_probs[submodel][label] for label in labels]
            plt.bar(x + i * width, probs, width, label=submodel)
        plt.xticks(x + width * (len(submodels)-1)/2, labels, rotation=45, ha='right')
        plt.ylabel('Probability')
        plt.title(f"Ensemble submodel probabilities for: {text[:40]}...", fontsize=11)
        plt.legend()
    else:
        # Fallback: plot label counts
        label_counts = Counter(votes.values())
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        plt.bar(labels, counts, color='orange', edgecolor='black')
        plt.ylabel('Votes')
        plt.xlabel('Label')
        plt.title(f"Ensemble votes for: {text[:40]}...", fontsize=11)
    plt.tight_layout()
    fname = f"ensemble_votes_{idx if idx is not None else ''}.png"
    plt.savefig(Path(vis_dir) / fname)
    plt.close()
    return str(Path(vis_dir) / fname)

def plot_confidence_progression(tokens, confidences, text, vis_dir, idx=None):
    """Plot and save model confidence progression as it processes each token."""
    plt.figure(figsize=(max(6, len(tokens)*0.7), 2.5))
    plt.plot(range(1, len(tokens)+1), confidences, marker='o', color='purple')
    plt.xticks(range(1, len(tokens)+1), tokens, rotation=45, ha='right', fontsize=9)
    plt.title(f"Model confidence progression for: {text[:40]}...", fontsize=11)
    plt.ylabel('Confidence (max prob)')
    plt.xlabel('Token position')
    plt.tight_layout()
    fname = f"confidence_progression_{idx if idx is not None else ''}.png"
    plt.savefig(Path(vis_dir) / fname)
    plt.close()
    return str(Path(vis_dir) / fname)
