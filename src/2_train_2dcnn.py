"""
2_train_2dcnn.py — Training 2D-CNN dengan Spektrogram untuk Deteksi Wireless Jamming.

Pendekatan:
    1. Sinyal I/Q (1024, 2) → Complex signal (I + jQ)
    2. STFT → Spektrogram 2D (log-magnitude)
    3. 2D-CNN (image classification) → Prediksi kelas

Usage:
    python src/2_train_2dcnn.py
    python src/2_train_2dcnn.py --max-samples 1000000
    python src/2_train_2dcnn.py --epochs 1 --dry-run
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import src.gpu_setup
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.data_loader import JammingDataGenerator, create_train_val_test_split, HDF5_FILE, NUM_CLASSES
from src.utils import plot_training_history, compute_metrics, plot_confusion_matrix, plot_classification_matrix, print_classification_report, measure_latency, CLASS_NAMES, RESULTS_DIR

# --------------------------------------------------------------------------- #
#                              KONFIGURASI                                     #
# --------------------------------------------------------------------------- #
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_NAME = "best_2dcnn"

# STFT Parameters
NPERSEG = 64      # Panjang window FFT
HOP_LENGTH = 16   # Jarak antar window (overlap = NPERSEG - HOP = 48)
# Hasil: spectrogram shape = (64, 61) → (freq_bins, time_frames)


# --------------------------------------------------------------------------- #
#                     KONVERSI SINYAL → SPEKTROGRAM                            #
# --------------------------------------------------------------------------- #
def batch_to_spectrogram(X_batch: np.ndarray) -> np.ndarray:
    """
    Konversi batch sinyal I/Q ke batch spektrogram.

    Args:
        X_batch: shape (batch, 1024, 2) — sinyal I/Q

    Returns:
        spectrograms: shape (batch, 64, 61, 1) — log-magnitude spectrogram
    """
    # Gabungkan I + jQ menjadi sinyal kompleks
    complex_signals = X_batch[:, :, 0] + 1j * X_batch[:, :, 1]  # (batch, 1024)

    # Buat indeks frame untuk STFT
    frame_starts = np.arange(0, 1024 - NPERSEG + 1, HOP_LENGTH)  # (61,)
    frame_indices = frame_starts[:, None] + np.arange(NPERSEG)     # (61, 64)

    # Hanning window
    window = np.hanning(NPERSEG).astype(np.float32)

    # Ekstrak semua frame sekaligus (vectorized)
    frames = complex_signals[:, frame_indices]  # (batch, 61, 64)
    frames = frames * window[None, None, :]     # Apply window

    # FFT per frame
    spec = np.fft.fft(frames, axis=2)  # (batch, 61, 64)

    # Magnitude → transpose ke (batch, freq, time)
    mag = np.abs(spec).transpose(0, 2, 1)  # (batch, 64, 61)

    # Log scale (kompresi dynamic range)
    log_mag = np.log10(mag + 1e-10)

    # Normalize per sample ke [0, 1]
    for i in range(len(log_mag)):
        vmin, vmax = log_mag[i].min(), log_mag[i].max()
        log_mag[i] = (log_mag[i] - vmin) / (vmax - vmin + 1e-10)

    # Tambah channel dimension → (batch, 64, 61, 1)
    return log_mag[..., np.newaxis].astype(np.float32)


# --------------------------------------------------------------------------- #
#                  GENERATOR DENGAN KONVERSI SPEKTROGRAM                       #
# --------------------------------------------------------------------------- #
class SpectrogramGenerator(tf.keras.utils.Sequence):
    """
    Wrapper yang mengambil output dari JammingDataGenerator
    dan mengkonversi sinyal I/Q ke spektrogram 2D.
    """

    def __init__(self, base_generator: JammingDataGenerator):
        self.base = base_generator

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        X, y = self.base[idx]
        X_spec = batch_to_spectrogram(X)
        return X_spec, y

    def on_epoch_end(self):
        self.base.on_epoch_end()


# --------------------------------------------------------------------------- #
#                           ARSITEKTUR MODEL                                   #
# --------------------------------------------------------------------------- #
def build_2dcnn(input_shape: tuple = (64, 61, 1), num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Bangun model 2D-CNN untuk klasifikasi spektrogram.

    Architecture:
        Input (64, 61, 1)  ← log-magnitude spectrogram
        → Conv2D(32, 3×3, L2) → BN → ReLU → MaxPool(2×2)     → (32, 30, 32)
        → Conv2D(64, 3×3, L2) → BN → ReLU → MaxPool(2×2)     → (16, 15, 64)
        → Conv2D(128, 3×3, L2) → BN → ReLU → GlobalAvgPool   → (128,)
        → Dense(64, L2) → Dropout(0.5)
        → Dense(3, softmax)
    """
    l2 = regularizers.l2(1e-4)
    inputs = keras.Input(shape=input_shape, name="spectrogram_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.1)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.15)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Head
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="2D_CNN_Spectrogram_Detector")
    return model


# --------------------------------------------------------------------------- #
#                           TRAINING PIPELINE                                  #
# --------------------------------------------------------------------------- #
def train(args):
    """Pipeline training utama."""
    print("=" * 60)
    print("2D-CNN (Spectrogram) Training — Wireless Jamming Detection")
    print("=" * 60)

    # --- Data Split ---
    train_idx, val_idx, test_idx = create_train_val_test_split(
        hdf5_path=HDF5_FILE,
        max_samples=args.max_samples,
    )

    # --- Spectrogram Generators ---
    base_train = JammingDataGenerator(indices=train_idx, batch_size=args.batch_size, shuffle=True)
    base_val = JammingDataGenerator(indices=val_idx, batch_size=args.batch_size, shuffle=False, seed=123)

    train_gen = SpectrogramGenerator(base_train)
    val_gen = SpectrogramGenerator(base_val)

    # Preview satu batch untuk verifikasi shape
    X_sample, y_sample = train_gen[0]
    print(f"\n[INFO] Spectrogram shape per sample: {X_sample.shape[1:]}")
    print(f"[INFO] Batch shape: {X_sample.shape}")
    print(f"[INFO] Value range: [{X_sample.min():.4f}, {X_sample.max():.4f}]")
    print(f"Train batches: {len(train_gen)}, Val batches: {len(val_gen)}")

    # --- Build Model ---
    model = build_2dcnn(input_shape=X_sample.shape[1:])
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # --- Callbacks ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{MODEL_NAME}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # --- Training ---
    print(f"\n>> Memulai training ({args.epochs} epoch)...\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Save History & Plot ---
    history_path = os.path.join(RESULTS_DIR, "2dcnn_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"\n[INFO] Training history disimpan ke: {history_path}")

    plot_training_history(
        history,
        model_name="2D-CNN (Spectrogram)",
        save_path=os.path.join(RESULTS_DIR, "2dcnn_training_curves.png"),
    )

    # --- Evaluasi pada Test Set ---
    print("\n" + "=" * 60)
    print("  Evaluasi 2D-CNN pada Test Set")
    print("=" * 60)

    test_gen_base = JammingDataGenerator(indices=test_idx, batch_size=args.batch_size, shuffle=False, seed=456)
    test_gen = SpectrogramGenerator(test_gen_base)

    all_y_true, all_y_pred = [], []
    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]
        preds = model.predict(X_batch, verbose=0)
        y_pred = np.argmax(preds, axis=-1)
        all_y_true.extend(y_batch)
        all_y_pred.extend(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    metrics = compute_metrics(all_y_true, all_y_pred)
    print(f"\n[METRICS] 2D-CNN:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision_macro']:.4f} (macro)")
    print(f"   Recall:    {metrics['recall_macro']:.4f} (macro)")
    print(f"   F1-Score:  {metrics['f1_macro']:.4f} (macro)")

    print(f"\n[REPORT] Classification Report:")
    print_classification_report(all_y_true, all_y_pred)

    # Confusion Matrix
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_2d_cnn.png")
    plot_confusion_matrix(all_y_true, all_y_pred, title="Confusion Matrix — 2D-CNN", save_path=cm_path)
    print(f"   Confusion matrix: {cm_path}")

    # Classification Report Matrix
    cr_path = os.path.join(RESULTS_DIR, "classification_report_2dcnn.png")
    plot_classification_matrix(metrics=metrics, model_name="2D-CNN", save_path=cr_path)
    print(f"   Classification report matrix: {cr_path}")

    # Latency (menggunakan input spectrogram)
    sample_spec = np.random.randn(1, *X_sample.shape[1:]).astype(np.float32)
    latency = measure_latency(model, sample_spec, n_runs=50)
    print(f"\n[LATENCY] Inference (2D-CNN):")
    print(f"   Avg: {latency['avg_ms']:.2f} ms | Min: {latency['min_ms']:.2f} ms | Max: {latency['max_ms']:.2f} ms")

    metrics["latency"] = latency

    # Save results
    results_path = os.path.join(RESULTS_DIR, "evaluation_results_2dcnn.json")
    serializable = {
        "2D-CNN": {
            k: (v if not isinstance(v, np.ndarray) else v.tolist())
            for k, v in metrics.items()
        }
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[SAVED] Hasil evaluasi: {results_path}")
    print("\n[OK] Training & Evaluasi 2D-CNN selesai!")


# --------------------------------------------------------------------------- #
#                              ARGUMENT PARSER                                 #
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train 2D-CNN (Spectrogram) for Wireless Jamming Detection")
    parser.add_argument("--epochs", type=int, default=30, help="Jumlah epoch (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 0.0003)")
    parser.add_argument("--max-samples", type=int, default=2555904, help="Batasi jumlah sample (default: 2.5M)")
    parser.add_argument("--dry-run", action="store_true", help="Quick test mode (1 epoch, 1000 samples)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        args.epochs = 1
        args.max_samples = args.max_samples or 1000
        print("[DRY RUN MODE] epochs=1, max_samples=1000")
    train(args)
