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

def plot_ensemble_votes(votes, text, vis_dir, idx=None):
    """Plot and save ensemble voting results for a given text input."""
    labels = list(votes.keys())
    counts = list(votes.values())
    plt.figure(figsize=(max(4, len(labels)*0.7), 2.5))
    bars = plt.bar(labels, counts, color='orange', edgecolor='black')
    plt.title(f"Ensemble votes for: {text[:40]}...", fontsize=11)
    plt.ylabel('Votes')
    plt.tight_layout()
    fname = f"ensemble_votes_{idx if idx is not None else ''}.png"
    plt.savefig(Path(vis_dir) / fname)
    plt.close()
    return str(Path(vis_dir) / fname)
