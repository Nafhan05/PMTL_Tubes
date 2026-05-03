"""
3_evaluate_models.py — Evaluasi & Benchmarking Model Deteksi Wireless Jamming.

Script ini:
1. Load semua model terlatih dari folder models/
2. Evaluasi pada test set (dihasilkan oleh JammingDataGenerator)
3. Hitung metrik: Accuracy, Precision, Recall, F1-Score
4. Generate & simpan Confusion Matrix
5. Benchmark inference latency
6. Cetak tabel perbandingan: 1D-CNN vs LSTM

Usage:
    python src/3_evaluate_models.py
    python src/3_evaluate_models.py --max-samples 5000
"""

import argparse
import os
import sys
import json

# Tambahkan parent directory ke path agar bisa import modul lain
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import src.gpu_setup  # Harus sebelum import tensorflow agar GPU DLL ditemukan
import tensorflow as tf
from tensorflow import keras

from src.data_loader import (
    JammingDataGenerator,
    create_train_val_test_split,
    HDF5_FILE,
    NUM_CLASSES,
)
from src.utils import (
    compute_metrics,
    plot_confusion_matrix,
    print_classification_report,
    plot_classification_matrix,
    measure_latency,
    CLASS_NAMES,
    RESULTS_DIR,
)

# --------------------------------------------------------------------------- #
#                              KONFIGURASI                                     #
# --------------------------------------------------------------------------- #
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

MODEL_CONFIGS = {
    "1D-CNN": {
        "path": os.path.join(MODELS_DIR, "best_1dcnn.keras"),
        "seq_len": 256,  # v2: downsampled dari 1024
    },
    "LSTM": {
        "path": os.path.join(MODELS_DIR, "best_lstm.keras"),
        "seq_len": 256,  # Sesuaikan jika training menggunakan seq_len berbeda
    },
}


# --------------------------------------------------------------------------- #
#                     EVALUASI SATU MODEL                                      #
# --------------------------------------------------------------------------- #
def evaluate_model(
    model: keras.Model,
    model_name: str,
    test_gen,
    seq_len: int = 1024,
) -> dict:
    """
    Evaluasi satu model pada test set.

    Returns:
        Dictionary berisi semua metrik dan latency info.
    """
    print(f"\n{'='*50}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*50}")

    # --- Collect predictions ---
    all_y_true = []
    all_y_pred = []

    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]

        # Downsample jika model membutuhkan sequence lebih pendek
        if seq_len < X_batch.shape[1]:
            indices = np.linspace(0, X_batch.shape[1] - 1, seq_len, dtype=int)
            X_batch = X_batch[:, indices, :]

        preds = model.predict(X_batch, verbose=0)
        y_pred = np.argmax(preds, axis=-1)

        all_y_true.extend(y_batch)
        all_y_pred.extend(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # --- Metrik ---
    metrics = compute_metrics(all_y_true, all_y_pred)
    print(f"\n[METRICS] {model_name}:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision_macro']:.4f} (macro)")
    print(f"   Recall:    {metrics['recall_macro']:.4f} (macro)")
    print(f"   F1-Score:  {metrics['f1_macro']:.4f} (macro)")

    print(f"\n[REPORT] Classification Report:")
    print_classification_report(all_y_true, all_y_pred)

    # --- Confusion Matrix ---
    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name.lower().replace('-', '_')}.png")
    plot_confusion_matrix(
        all_y_true,
        all_y_pred,
        title=f"Confusion Matrix — {model_name}",
        save_path=cm_path,
    )
    print(f"   Confusion matrix disimpan: {cm_path}")

    # --- Classification Report Matrix (Precision, Recall, F1) ---
    cr_path = os.path.join(RESULTS_DIR, f"classification_report_{model_name.lower().replace('-', '_')}.png")
    plot_classification_matrix(
        metrics=metrics,
        model_name=model_name,
        save_path=cr_path,
    )
    print(f"   Classification report matrix disimpan: {cr_path}")

    # --- Latency ---
    sample_input = np.random.randn(1, seq_len, 2).astype(np.float32)
    latency = measure_latency(model, sample_input, n_runs=50)
    print(f"\n[LATENCY] Inference Latency ({model_name}):")
    print(f"   Avg: {latency['avg_ms']:.2f} ms | Min: {latency['min_ms']:.2f} ms | Max: {latency['max_ms']:.2f} ms")

    metrics["latency"] = latency
    return metrics


# --------------------------------------------------------------------------- #
#                          MAIN PIPELINE                                       #
# --------------------------------------------------------------------------- #
def main(args):
    print("=" * 60)
    print("Model Evaluation & Benchmarking")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Test split ---
    _, _, test_idx = create_train_val_test_split(
        hdf5_path=HDF5_FILE,
        max_samples=args.max_samples,
    )

    test_gen = JammingDataGenerator(
        indices=test_idx,
        batch_size=args.batch_size,
        shuffle=False,
        seed=456,
    )
    print(f"Test batches: {len(test_gen)}")

    # --- Evaluate each model ---
    all_results = {}
    for name, config in MODEL_CONFIGS.items():
        model_path = config["path"]
        if not os.path.exists(model_path):
            print(f"\n[WARN] Model {name} tidak ditemukan: {model_path} -- SKIP")
            continue

        model = keras.models.load_model(model_path)
        results = evaluate_model(model, name, test_gen, seq_len=config["seq_len"])
        all_results[name] = results

    # --- Tabel Perbandingan ---
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  TABEL PERBANDINGAN MODEL")
        print(f"{'='*60}")
        header = f"{'Model':<12} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Latency':>10}"
        print(header)
        print("-" * len(header))
        for name, m in all_results.items():
            lat = m["latency"]["avg_ms"]
            print(f"{name:<12} {m['accuracy']:>8.4f} {m['precision_macro']:>8.4f} {m['recall_macro']:>8.4f} {m['f1_macro']:>8.4f} {lat:>8.2f}ms")

    # --- Save results ---
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    # Konversi numpy types ke Python native untuk JSON serialization
    serializable = {}
    for name, m in all_results.items():
        serializable[name] = {
            k: (v if not isinstance(v, np.ndarray) else v.tolist())
            for k, v in m.items()
        }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[SAVED] Hasil evaluasi disimpan ke: {results_path}")
    print("\n[OK] Evaluasi selesai!")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
