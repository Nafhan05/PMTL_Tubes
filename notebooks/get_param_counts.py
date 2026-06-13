"""
Get parameter count for all models and generate:
1. Updated final summary table (with Precision, Recall, both baselines)
2. Computational cost table (parameter count, inference latency, training time)
"""
import json
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Try to load models and get param counts
param_counts = {}

try:
    import src.gpu_setup
    import tensorflow as tf
    
    models_dir = os.path.join(BASE_DIR, "models")
    
    for name, fname in [("LSTM", "best_lstm.keras"), 
                         ("1D-CNN Baseline", "best_1dcnn.keras"),
                         ("2D-CNN Baseline", "best_2dcnn.keras"),
                         ("1D-CNN HPO", "best_1dcnn_hpo.keras"),
                         ("2D-CNN HPO", "best_2dcnn_hpo.keras")]:
        fpath = os.path.join(models_dir, fname)
        if os.path.exists(fpath):
            model = tf.keras.models.load_model(fpath)
            total = model.count_params()
            trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
            param_counts[name] = {"total": total, "trainable": trainable}
            print(f"{name}: {total:,} params ({trainable:,} trainable)")
            del model
        else:
            print(f"{name}: FILE NOT FOUND ({fpath})")
except Exception as e:
    print(f"Error loading models: {e}")

# Also check training epochs from history
for name, fpath in [("1D-CNN Baseline", "results/1dcnn_history.json"),
                     ("LSTM", "results/lstm_history.json"),
                     ("2D-CNN Baseline", "results/2dcnn_history.json"),
                     ("1D-CNN HPO", "results/hpo/1dcnn_hpo_history.json"),
                     ("2D-CNN HPO", "results/hpo/2dcnn_hpo_history.json")]:
    full = os.path.join(BASE_DIR, fpath)
    if os.path.exists(full):
        with open(full) as f:
            h = json.load(f)
        epochs = len(h.get("loss", []))
        print(f"{name}: {epochs} epochs trained")

# Save param counts for visualization script
out = os.path.join(BASE_DIR, "results", "hpo", "param_counts.json")
with open(out, "w") as f:
    json.dump(param_counts, f, indent=2)
print(f"\nSaved param counts to {out}")
