"""
3_evaluate_models.py — Evaluasi & Benchmarking Model Deteksi Wireless Jamming (Terpadu).

Script ini:
1. Load semua model terlatih dari folder models/ (LSTM, 1D-CNN baseline/HPO, 2D-CNN baseline/HPO)
2. Evaluasi pada test set (dihasilkan oleh JammingDataGenerator)
3. Melakukan temporal downsampling untuk 1D-model dan konversi STFT spektrogram untuk 2D-model secara dinamis
4. Hitung metrik: Accuracy, Precision, Recall, F1-Score
5. Generate & simpan Confusion Matrix serta Classification Report Matrix
6. Benchmark inference latency
7. Cetak tabel perbandingan komprehensif (baseline vs HPO)

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
    "LSTM": {
        "path": os.path.join(MODELS_DIR, "best_lstm.keras"),
        "type": "1d",
        "seq_len": 256,
    },
    "1D-CNN Baseline": {
        "path": os.path.join(MODELS_DIR, "best_1dcnn.keras"),
        "type": "1d",
        "seq_len": 256,
    },
    "1D-CNN HPO": {
        "path": os.path.join(MODELS_DIR, "best_1dcnn_hpo.keras"),
        "type": "1d",
        "seq_len": 256,
    },
    "2D-CNN Baseline": {
        "path": os.path.join(MODELS_DIR, "best_2dcnn.keras"),
        "type": "2d",
        "nperseg": 64,
        "hop_length": 16,
    },
    "2D-CNN HPO": {
        "path": os.path.join(MODELS_DIR, "best_2dcnn_hpo.keras"),
        "type": "2d",
        "nperseg": 64,  # Nilai default, akan diupdate dinamis dari hasil HPO
        "hop_length": 16,
    },
}


# --------------------------------------------------------------------------- #
#                     KONVERSI SINYAL → SPEKTROGRAM                            #
# --------------------------------------------------------------------------- #
def batch_to_spectrogram(X_batch: np.ndarray, nperseg: int = 64, hop_length: int = 16) -> np.ndarray:
    """
    Konversi batch sinyal I/Q ke spektrogram 2D (log-magnitude), ternormalisasi.
    """
    complex_signals = X_batch[:, :, 0] + 1j * X_batch[:, :, 1]
    frame_starts = np.arange(0, 1024 - nperseg + 1, hop_length)
    frame_indices = frame_starts[:, None] + np.arange(nperseg)
    window = np.hanning(nperseg).astype(np.float32)
    frames = complex_signals[:, frame_indices] * window[None, None, :]
    spec = np.fft.fft(frames, axis=2)
    mag = np.abs(spec).transpose(0, 2, 1)
    log_mag = np.log10(mag + 1e-10)
    for i in range(len(log_mag)):
        vmin, vmax = log_mag[i].min(), log_mag[i].max()
        log_mag[i] = (log_mag[i] - vmin) / (vmax - vmin + 1e-10)
    return log_mag[..., np.newaxis].astype(np.float32)


# --------------------------------------------------------------------------- #
#                     EVALUASI SATU MODEL                                      #
# --------------------------------------------------------------------------- #
def evaluate_model(
    model: keras.Model,
    model_name: str,
    test_gen,
    model_type: str = "1d",
    seq_len: int = 256,
    nperseg: int = 64,
    hop_length: int = 16,
) -> dict:
    """
    Evaluasi satu model pada test set secara dinamis.

    Returns:
        Dictionary berisi semua metrik dan latency info.
    """
    print(f"\n{'='*50}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*50}")

    all_y_true = []
    all_y_pred = []

    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]

        if model_type == "2d":
            # Konversi sinyal I/Q ke spektrogram 2D
            X_batch = batch_to_spectrogram(X_batch, nperseg, hop_length)
        else:
            # Downsample jika model membutuhkan sequence 1D lebih pendek
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
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{safe_name}.png")
    plot_confusion_matrix(
        all_y_true,
        all_y_pred,
        title=f"Confusion Matrix — {model_name}",
        save_path=cm_path,
    )
    print(f"   Confusion matrix disimpan: {cm_path}")

    # --- Classification Report Matrix (Precision, Recall, F1) ---
    cr_path = os.path.join(RESULTS_DIR, f"classification_report_{safe_name}.png")
    plot_classification_matrix(
        metrics=metrics,
        model_name=model_name,
        save_path=cr_path,
    )
    print(f"   Classification report matrix disimpan: {cr_path}")

    # --- Latency ---
    if model_type == "2d":
        tf_count = len(np.arange(0, 1024 - nperseg + 1, hop_length))
        sample_input = np.random.randn(1, nperseg, tf_count, 1).astype(np.float32)
    else:
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
    print("Unified Model Evaluation & Benchmarking")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Memuat Konfigurasi HPO 2D-CNN secara Dinamis ---
    hpo_2d_json = os.path.join(RESULTS_DIR, "hpo_best_2dcnn.json")
    if os.path.exists(hpo_2d_json):
        try:
            with open(hpo_2d_json) as f:
                hpo_cfg = json.load(f)
            MODEL_CONFIGS["2D-CNN HPO"]["nperseg"] = hpo_cfg.get("stft_nperseg", 64)
            MODEL_CONFIGS["2D-CNN HPO"]["hop_length"] = hpo_cfg.get("stft_hop_length", 16)
            print(f"[INFO] Konfigurasi HPO 2D-CNN dimuat: nperseg={MODEL_CONFIGS['2D-CNN HPO']['nperseg']}, hop={MODEL_CONFIGS['2D-CNN HPO']['hop_length']}")
        except Exception as e:
            print(f"[WARN] Gagal membaca hpo_best_2dcnn.json ({e}). Menggunakan default.")

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

        print(f"\n[INFO] Loading model: {name} dari {model_path}")
        model = keras.models.load_model(model_path)
        
        results = evaluate_model(
            model=model,
            model_name=name,
            test_gen=test_gen,
            model_type=config["type"],
            seq_len=config.get("seq_len", 256),
            nperseg=config.get("nperseg", 64),
            hop_length=config.get("hop_length", 16),
        )
        all_results[name] = results

    # --- Tabel Perbandingan ---
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print("  TABEL PERBANDINGAN MODEL DETEKSI WIRELESS JAMMING")
        print(f"{'='*80}")
        header = f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Latency':>12}"
        print(header)
        print("-" * len(header))
        for name, m in all_results.items():
            lat = m["latency"]["avg_ms"]
            print(f"{name:<20} {m['accuracy']:>10.4f} {m['precision_macro']:>10.4f} {m['recall_macro']:>10.4f} {m['f1_macro']:>10.4f} {lat:>10.2f} ms")
        print("="*80)

    # --- Save results ---
    results_path = os.path.join(RESULTS_DIR, "evaluation_results_all.json")
    serializable = {}
    for name, m in all_results.items():
        serializable[name] = {
            k: (v if not isinstance(v, np.ndarray) else v.tolist())
            for k, v in m.items()
        }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[SAVED] Hasil evaluasi lengkap disimpan ke: {results_path}")
    print("\n[OK] Evaluasi terpadu selesai!")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Model Evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

