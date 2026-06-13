import json
import matplotlib.pyplot as plt
import os
import sys

def plot_history_from_json(json_path, save_path, title_prefix):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        h = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(h["loss"], label="Train Loss", color="#2196F3")
    if "val_loss" in h:
        axes[0].plot(h["val_loss"], label="Val Loss", color="#FF5722")
    axes[0].set_title(f"{title_prefix} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(h["accuracy"], label="Train Acc", color="#4CAF50")
    if "val_accuracy" in h:
        axes[1].plot(h["val_accuracy"], label="Val Acc", color="#FF9800")
    axes[1].set_title(f"{title_prefix} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_hpo_dir = os.path.join(base_dir, 'results', 'hpo')
    
    hpo_1d_json = os.path.join(results_hpo_dir, '1dcnn_hpo_history.json')
    hpo_2d_json = os.path.join(results_hpo_dir, '2dcnn_hpo_history.json')
    
    plot_history_from_json(hpo_1d_json, os.path.join(results_hpo_dir, '1dcnn_hpo_training_curves.png'), '1D-CNN HPO')
    plot_history_from_json(hpo_2d_json, os.path.join(results_hpo_dir, '2dcnn_hpo_training_curves.png'), '2D-CNN HPO')

