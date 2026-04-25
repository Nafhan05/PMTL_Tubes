"""
2_train_lstm.py — Pipeline Training Model LSTM untuk Deteksi Wireless Jamming.

Arsitektur: LSTM → Dense → Softmax(3).
Sebagai baseline perbandingan terhadap 1D-CNN.

Usage:
    python src/2_train_lstm.py
    python src/2_train_lstm.py --epochs 1 --dry-run
    python src/2_train_lstm.py --seq-len 256                   # Downsample ke 256
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
from tensorflow.keras import layers

from src.data_loader import JammingDataGenerator, create_train_val_test_split, HDF5_FILE, NUM_CLASSES
from src.utils import plot_training_history, RESULTS_DIR

# --------------------------------------------------------------------------- #
#                              KONFIGURASI                                     #
# --------------------------------------------------------------------------- #
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_NAME = "best_lstm"


# --------------------------------------------------------------------------- #
#                           ARSITEKTUR MODEL                                   #
# --------------------------------------------------------------------------- #
def build_lstm(seq_len: int = 1024, n_features: int = 2, num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Bangun model LSTM untuk klasifikasi sinyal I/Q.

    Architecture:
        Input (seq_len, 2)
        → LSTM(128, return_sequences=True)
        → Dropout(0.3)
        → LSTM(64)
        → Dropout(0.3)
        → Dense(64, relu)
        → Dense(3, softmax)
    """
    inputs = keras.Input(shape=(seq_len, n_features), name="iq_input")

    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.4)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LSTM_Jammer_Detector")
    return model


# --------------------------------------------------------------------------- #
#                   WRAPPER GENERATOR DENGAN DOWNSAMPLING                      #
# --------------------------------------------------------------------------- #
class DownsampledGenerator(tf.keras.utils.Sequence):
    """
    Wrapper yang mengambil output dari JammingDataGenerator
    dan melakukan temporal downsampling (mengurangi panjang sequence).

    Ini berguna untuk mempercepat training LSTM yang lambat pada sequence panjang.
    """

    def __init__(self, base_generator: JammingDataGenerator, target_len: int = 256):
        self.base = base_generator
        self.target_len = target_len

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        X, y = self.base[idx]
        orig_len = X.shape[1]
        if self.target_len < orig_len:
            # Uniform downsampling
            indices = np.linspace(0, orig_len - 1, self.target_len, dtype=int)
            X = X[:, indices, :]
        return X, y

    def on_epoch_end(self):
        self.base.on_epoch_end()


# --------------------------------------------------------------------------- #
#                           TRAINING PIPELINE                                  #
# --------------------------------------------------------------------------- #
def train(args):
    """Pipeline training utama."""
    print("=" * 60)
    print("LSTM Training Pipeline — Wireless Jamming Detection")
    print("=" * 60)

    # --- Data Split ---
    train_idx, val_idx, test_idx = create_train_val_test_split(
        hdf5_path=HDF5_FILE,
        max_samples=args.max_samples,
    )

    # --- Data Generators ---
    base_train_gen = JammingDataGenerator(indices=train_idx, batch_size=args.batch_size, shuffle=True)
    base_val_gen = JammingDataGenerator(indices=val_idx, batch_size=args.batch_size, shuffle=False, seed=123)

    if args.seq_len < 1024:
        print(f"\n[INFO] Downsampling sequence: 1024 -> {args.seq_len}")
        train_gen = DownsampledGenerator(base_train_gen, target_len=args.seq_len)
        val_gen = DownsampledGenerator(base_val_gen, target_len=args.seq_len)
    else:
        train_gen = base_train_gen
        val_gen = base_val_gen

    print(f"Train batches: {len(train_gen)}, Val batches: {len(val_gen)}")

    # --- Build Model ---
    model = build_lstm(seq_len=args.seq_len)
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
    print(f"\n>> Memulai training LSTM ({args.epochs} epoch)...\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Save History & Plot ---
    history_path = os.path.join(RESULTS_DIR, "lstm_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"\n[INFO] Training history disimpan ke: {history_path}")

    plot_training_history(
        history,
        model_name="LSTM",
        save_path=os.path.join(RESULTS_DIR, "lstm_training_curves.png"),
    )

    print("\n[OK] Training LSTM selesai!")
    print(f"   Model terbaik: {os.path.join(MODELS_DIR, MODEL_NAME + '.keras')}")


# --------------------------------------------------------------------------- #
#                              ARGUMENT PARSER                                 #
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM for Wireless Jamming Detection")
    parser.add_argument("--epochs", type=int, default=30, help="Jumlah epoch (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--seq-len", type=int, default=256, help="Target sequence length setelah downsampling (default: 256)")
    parser.add_argument("--max-samples", type=int, default=None, help="Batasi jumlah sample (untuk debug)")
    parser.add_argument("--dry-run", action="store_true", help="Quick test mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        args.epochs = 1
        args.max_samples = args.max_samples or 1000
        print("[DRY RUN MODE] epochs=1, max_samples=1000")
    train(args)
