"""
2_train_1dcnn.py — Pipeline Training Model 1D-CNN untuk Deteksi Wireless Jamming.

Arsitektur: Beberapa blok Conv1D → BatchNorm → ReLU → MaxPool1D,
diikuti GlobalAveragePooling1D → Dense → Softmax(3).

Usage:
    python src/2_train_1dcnn.py
    python src/2_train_1dcnn.py --epochs 1 --dry-run     # Quick test
    python src/2_train_1dcnn.py --max-samples 10000       # Subset kecil
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
from tensorflow.keras import layers, regularizers

from src.data_loader import JammingDataGenerator, create_train_val_test_split, HDF5_FILE, NUM_CLASSES
from src.utils import plot_training_history, RESULTS_DIR

# --------------------------------------------------------------------------- #
#                              KONFIGURASI                                     #
# --------------------------------------------------------------------------- #
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_NAME = "best_1dcnn"
INPUT_SHAPE = (1024, 2)


# --------------------------------------------------------------------------- #
#                           ARSITEKTUR MODEL                                   #
# --------------------------------------------------------------------------- #
def build_1dcnn(input_shape: tuple = INPUT_SHAPE, num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Bangun model 1D-CNN untuk klasifikasi sinyal I/Q.

    Architecture (v2 — anti-overfitting):
        Input (1024, 2)
        → [Conv1D(64, L2) → BN → ReLU → MaxPool → SpatialDropout] × 1
        → [Conv1D(128, L2) → BN → ReLU → MaxPool → SpatialDropout] × 1
        → [Conv1D(128, L2) → BN → ReLU → MaxPool → SpatialDropout] × 1
        → [Conv1D(128, L2) → BN → ReLU → MaxPool → SpatialDropout] × 1
        → GlobalAveragePooling1D
        → Dense(64, L2) → Dropout(0.5)
        → Dense(3, softmax)

    Perubahan dari v1:
        - L2 regularization (1e-4) pada semua Conv1D dan Dense
        - Filter dikurangi: 64→128→128→128 (dari 64→128→256→256)
        - Dense head dikecilkan: 64 (dari 128)
        - Default lr diturunkan ke 3e-4 (di parse_args)
    """
    l2 = regularizers.l2(1e-4)
    inputs = keras.Input(shape=input_shape, name="iq_input")

    # Block 1
    x = layers.Conv1D(64, kernel_size=7, padding="same", kernel_regularizer=l2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.SpatialDropout1D(0.1)(x)

    # Block 2
    x = layers.Conv1D(128, kernel_size=5, padding="same", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.SpatialDropout1D(0.15)(x)

    # Block 3
    x = layers.Conv1D(128, kernel_size=3, padding="same", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.SpatialDropout1D(0.2)(x)

    # Block 4
    x = layers.Conv1D(128, kernel_size=3, padding="same", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.SpatialDropout1D(0.2)(x)

    # Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="1D_CNN_Jammer_Detector_v2")
    return model


# --------------------------------------------------------------------------- #
#                           TRAINING PIPELINE                                  #
# --------------------------------------------------------------------------- #
def train(args):
    """Pipeline training utama."""
    print("=" * 60)
    print("1D-CNN Training Pipeline — Wireless Jamming Detection")
    print("=" * 60)

    # --- Data Split ---
    train_idx, val_idx, test_idx = create_train_val_test_split(
        hdf5_path=HDF5_FILE,
        max_samples=args.max_samples,
    )

    # --- Data Generators ---
    train_gen = JammingDataGenerator(indices=train_idx, batch_size=args.batch_size, shuffle=True)
    val_gen = JammingDataGenerator(indices=val_idx, batch_size=args.batch_size, shuffle=False, seed=123)

    print(f"\nTrain batches: {len(train_gen)}, Val batches: {len(val_gen)}")

    # --- Build Model ---
    model = build_1dcnn()
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
    history_path = os.path.join(RESULTS_DIR, "1dcnn_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"\n[INFO] Training history disimpan ke: {history_path}")

    plot_training_history(
        history,
        model_name="1D-CNN",
        save_path=os.path.join(RESULTS_DIR, "1dcnn_training_curves.png"),
    )

    print("\n[OK] Training selesai!")
    print(f"   Model terbaik: {os.path.join(MODELS_DIR, MODEL_NAME + '.keras')}")


# --------------------------------------------------------------------------- #
#                              ARGUMENT PARSER                                 #
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train 1D-CNN for Wireless Jamming Detection")
    parser.add_argument("--epochs", type=int, default=30, help="Jumlah epoch (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 0.0003)")
    parser.add_argument("--max-samples", type=int, default=None, help="Batasi jumlah sample (untuk debug)")
    parser.add_argument("--dry-run", action="store_true", help="Quick test mode (1 epoch, 1000 samples)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        args.epochs = 1
        args.max_samples = args.max_samples or 1000
        print("[DRY RUN MODE] epochs=1, max_samples=1000")
    train(args)
