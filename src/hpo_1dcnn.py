"""
hpo_1dcnn.py — Hyperparameter Optimization untuk 1D-CNN menggunakan Keras Tuner.

Metode: Bayesian Optimization
Target: Mengatasi overfitting pada 1D-CNN (baseline: 75.5% accuracy)

Usage:
    run_gpu.bat python src/hpo_1dcnn.py                        # Full HPO (30 trials)
    run_gpu.bat python src/hpo_1dcnn.py --max-trials 5 --dry-run  # Quick test
    run_gpu.bat python src/hpo_1dcnn.py --max-samples 500000   # Subset data
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

import keras_tuner as kt

from src.data_loader import JammingDataGenerator, create_train_val_test_split, HDF5_FILE, NUM_CLASSES

# --------------------------------------------------------------------------- #
#                              KONFIGURASI                                     #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HPO_DIR = os.path.join(PROJECT_ROOT, "hpo_results", "1dcnn")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# --------------------------------------------------------------------------- #
#                   WRAPPER GENERATOR DENGAN DOWNSAMPLING                      #
# --------------------------------------------------------------------------- #
class DownsampledGenerator(tf.keras.utils.Sequence):
    """Downsampling temporal untuk mengurangi panjang sequence."""

    def __init__(self, base_generator: JammingDataGenerator, target_len: int = 256):
        self.base = base_generator
        self.target_len = target_len

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        X, y = self.base[idx]
        orig_len = X.shape[1]
        if self.target_len < orig_len:
            indices = np.linspace(0, orig_len - 1, self.target_len, dtype=int)
            X = X[:, indices, :]
        return X, y

    def on_epoch_end(self):
        self.base.on_epoch_end()


# --------------------------------------------------------------------------- #
#             MODEL BUILDER UNTUK KERAS TUNER (SEARCH SPACE)                   #
# --------------------------------------------------------------------------- #
def build_model(hp):
    """
    Bangun model 1D-CNN dengan hyperparameter yang di-tuning.

    Search space:
        - num_blocks: 2-4 blok konvolusi
        - filters_block_N: 32-256 filter per blok
        - kernel_size: 3, 5, atau 7
        - spatial_dropout: 0.05-0.4
        - l2_weight: 1e-5, 1e-4, atau 1e-3
        - dense_units: 32, 64, atau 128
        - dense_dropout: 0.3-0.6
        - learning_rate: 1e-4 sampai 1e-2 (log scale)
    """
    seq_len = 256
    num_classes = NUM_CLASSES

    # --- Hyperparameters ---
    num_blocks = hp.Int("num_blocks", min_value=2, max_value=4, step=1)
    kernel_size = hp.Choice("kernel_size", values=[3, 5, 7])
    l2_weight = hp.Choice("l2_weight", values=[1e-5, 1e-4, 1e-3])
    dense_units = hp.Choice("dense_units", values=[32, 64, 128])
    dense_dropout = hp.Float("dense_dropout", min_value=0.3, max_value=0.6, step=0.1)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    l2 = regularizers.l2(l2_weight)
    inputs = keras.Input(shape=(seq_len, 2), name="iq_input")
    x = inputs

    # --- Conv Blocks (jumlah dinamis) ---
    for i in range(num_blocks):
        filters = hp.Int(f"filters_block_{i}", min_value=32, max_value=256, step=32)
        spatial_drop = hp.Float(f"spatial_dropout_{i}", min_value=0.05, max_value=0.4, step=0.05)

        x = layers.Conv1D(filters, kernel_size=kernel_size, padding="same", kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.SpatialDropout1D(spatial_drop)(x)

    # --- Head ---
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# --------------------------------------------------------------------------- #
#                     BUILD MODEL DARI JSON CONFIG                             #
# --------------------------------------------------------------------------- #
def build_model_from_config(config: dict) -> keras.Model:
    """
    Bangun model 1D-CNN dari dictionary config (hasil HPO).
    Digunakan untuk retrain-only mode tanpa Keras Tuner.
    """
    seq_len = 256
    num_classes = NUM_CLASSES

    num_blocks = config["num_blocks"]
    kernel_size = config["kernel_size"]
    l2_weight = config["l2_weight"]
    dense_units = config["dense_units"]
    dense_dropout = config["dense_dropout"]
    learning_rate = config["learning_rate"]

    l2 = regularizers.l2(l2_weight)
    inputs = keras.Input(shape=(seq_len, 2), name="iq_input")
    x = inputs

    for i in range(num_blocks):
        filters = config[f"filters_block_{i}"]
        spatial_drop = config[f"spatial_dropout_{i}"]

        x = layers.Conv1D(filters, kernel_size=kernel_size, padding="same", kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.SpatialDropout1D(spatial_drop)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="1D_CNN_HPO_Best")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# --------------------------------------------------------------------------- #
#                           HPO PIPELINE                                       #
# --------------------------------------------------------------------------- #
def run_hpo(args):
    """Pipeline HPO utama."""
    print("=" * 60)
    print("  HPO 1D-CNN — Bayesian Optimization")
    print("=" * 60)

    # --- Data Split ---
    train_idx, val_idx, test_idx = create_train_val_test_split(
        hdf5_path=HDF5_FILE,
        max_samples=args.max_samples,
    )

    # --- Data Generators ---
    base_train = JammingDataGenerator(indices=train_idx, batch_size=args.batch_size, shuffle=True)
    base_val = JammingDataGenerator(indices=val_idx, batch_size=args.batch_size, shuffle=False, seed=123)

    train_gen = DownsampledGenerator(base_train, target_len=256)
    val_gen = DownsampledGenerator(base_val, target_len=256)

    print(f"\n[INFO] Train batches: {len(train_gen)}, Val batches: {len(val_gen)}")

    # ================================================================== #
    #  MODE: RETRAIN-ONLY (skip search, langsung pakai best HP dari JSON)
    # ================================================================== #
    if args.retrain_only:
        hp_path = os.path.join(RESULTS_DIR, "hpo_best_1dcnn.json")
        if not os.path.exists(hp_path):
            print(f"\n[ERROR] File {hp_path} tidak ditemukan!")
            print("        Jalankan HPO search dulu sebelum retrain-only.")
            sys.exit(1)

        with open(hp_path, "r") as f:
            best_config = json.load(f)

        print("\n[MODE] RETRAIN-ONLY — Skip search, langsung retrain.")
        print("\n[BEST HYPERPARAMETERS dari file]")
        for k, v in sorted(best_config.items()):
            print(f"   {k}: {v}")

        # Build model dari config
        model = build_model_from_config(best_config)
        model.summary()

        retrain_callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, "best_1dcnn_hpo.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=7,
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

        print(f"\n>> Memulai Re-training ({args.retrain_epochs} epoch)...\n")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.retrain_epochs,
            callbacks=retrain_callbacks,
            verbose=1,
        )

        # Save history
        os.makedirs(RESULTS_DIR, exist_ok=True)
        history_path = os.path.join(RESULTS_DIR, "1dcnn_hpo_history.json")
        with open(history_path, "w") as f:
            json.dump(
                {k: [float(v) for v in vals] for k, vals in history.history.items()},
                f, indent=2,
            )
        print(f"\n[SAVED] Training history: {history_path}")
        print(f"[SAVED] Best model: {os.path.join(MODELS_DIR, 'best_1dcnn_hpo.keras')}")
        print("\n[OK] Re-training 1D-CNN selesai!")
        return

    # ================================================================== #
    #  MODE: HPO SEARCH (Bayesian Optimization)
    # ================================================================== #
    print(f"[INFO] Max trials: {args.max_trials}")
    print(f"[INFO] Epochs per trial: {args.epochs_per_trial}")

    os.makedirs(HPO_DIR, exist_ok=True)

    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=args.max_trials,
        num_initial_points=min(5, args.max_trials),
        directory=HPO_DIR,
        project_name="bayesian_search",
        overwrite=args.overwrite,
    )

    print("\n[INFO] Search space summary:")
    tuner.search_space_summary()

    trial_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    print(f"\n>> Memulai Bayesian Optimization ({args.max_trials} trials)...\n")
    tuner.search(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs_per_trial,
        callbacks=trial_callbacks,
        verbose=1,
    )

    # --- Hasil ---
    print("\n" + "=" * 60)
    print("  HASIL HPO 1D-CNN")
    print("=" * 60)

    tuner.results_summary(num_trials=5)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n[BEST HYPERPARAMETERS]")
    best_config = best_hps.values
    for k, v in sorted(best_config.items()):
        print(f"   {k}: {v}")

    # Save best hyperparameters ke JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    hp_path = os.path.join(RESULTS_DIR, "hpo_best_1dcnn.json")
    with open(hp_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n[SAVED] Best hyperparameters: {hp_path}")

    # --- Opsional: Retrain langsung setelah HPO ---
    if args.retrain:
        print("\n" + "=" * 60)
        print("  RE-TRAINING DENGAN BEST HYPERPARAMETERS")
        print("=" * 60)

        model = build_model_from_config(best_config)
        model.summary()

        retrain_callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, "best_1dcnn_hpo.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=7,
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

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.retrain_epochs,
            callbacks=retrain_callbacks,
            verbose=1,
        )

        history_path = os.path.join(RESULTS_DIR, "1dcnn_hpo_history.json")
        with open(history_path, "w") as f:
            json.dump(
                {k: [float(v) for v in vals] for k, vals in history.history.items()},
                f, indent=2,
            )
        print(f"\n[SAVED] Training history: {history_path}")
        print(f"[SAVED] Best model: {os.path.join(MODELS_DIR, 'best_1dcnn_hpo.keras')}")

    print("\n[OK] HPO 1D-CNN selesai!")


# --------------------------------------------------------------------------- #
#                              ARGUMENT PARSER                                 #
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="HPO for 1D-CNN — Wireless Jamming Detection")
    parser.add_argument("--max-trials", type=int, default=30, help="Jumlah trial Bayesian (default: 30)")
    parser.add_argument("--epochs-per-trial", type=int, default=15, help="Max epoch per trial (default: 15)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--max-samples", type=int, default=500000,
                        help="Jumlah sample untuk HPO (default: 500K)")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain model terbaik langsung setelah HPO search selesai")
    parser.add_argument("--retrain-only", action="store_true",
                        help="SKIP search, langsung retrain dari hpo_best_1dcnn.json")
    parser.add_argument("--retrain-epochs", type=int, default=40,
                        help="Jumlah epoch untuk retrain (default: 40)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite hasil HPO sebelumnya")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test (2 trials, 2 epochs, 10K samples)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        args.max_trials = 2
        args.epochs_per_trial = 2
        args.max_samples = 10000
        print("[DRY RUN MODE] max_trials=2, epochs=2, samples=10K")
    run_hpo(args)
