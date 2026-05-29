"""
hpo_2dcnn.py — Hyperparameter Optimization untuk 2D-CNN (Spektrogram).

Metode: Bayesian Optimization via Keras Tuner
Target: Mengatasi underfitting pada 2D-CNN (baseline: 83.3% accuracy)

Usage:
    run_gpu.bat python src/hpo_2dcnn.py                              # Full HPO
    run_gpu.bat python src/hpo_2dcnn.py --dry-run                    # Quick test
    run_gpu.bat python src/hpo_2dcnn.py --retrain-only --retrain-epochs 40  # Retrain saja
"""

import argparse, os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import src.gpu_setup
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import keras_tuner as kt

from src.data_loader import JammingDataGenerator, create_train_val_test_split, HDF5_FILE, NUM_CLASSES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HPO_DIR = os.path.join(PROJECT_ROOT, "hpo_results", "2dcnn")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

_CURRENT_STFT_CONFIG = {"nperseg": 64, "hop_length": 16, "shape": (64, 61, 1)}

STFT_CONFIGS = {
    "64_16":  {"nperseg": 64,  "hop_length": 16, "shape": (64, 61, 1)},
    "64_8":   {"nperseg": 64,  "hop_length": 8,  "shape": (64, 121, 1)},
    "128_16": {"nperseg": 128, "hop_length": 16, "shape": (128, 57, 1)},
    "128_32": {"nperseg": 128, "hop_length": 32, "shape": (128, 29, 1)},
    "32_8":   {"nperseg": 32,  "hop_length": 8,  "shape": (32, 125, 1)},
}


# --------------------------------------------------------------------------- #
#                     KONVERSI SINYAL → SPEKTROGRAM                            #
# --------------------------------------------------------------------------- #
def batch_to_spectrogram(X_batch, nperseg=64, hop_length=16):
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


class HPOSpectrogramGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_generator, nperseg=64, hop_length=16):
        self.base = base_generator
        self.nperseg = nperseg
        self.hop_length = hop_length

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        X, y = self.base[idx]
        return batch_to_spectrogram(X, self.nperseg, self.hop_length), y

    def on_epoch_end(self):
        self.base.on_epoch_end()


# --------------------------------------------------------------------------- #
#             MODEL BUILDER — KERAS TUNER SEARCH SPACE                         #
# --------------------------------------------------------------------------- #
def build_model(hp):
    input_shape = _CURRENT_STFT_CONFIG.get("shape", (64, 61, 1))
    num_blocks = hp.Int("num_blocks", 3, 5, step=1)
    kernel_size = hp.Choice("kernel_size", [3, 5])
    use_l2 = hp.Boolean("use_l2")
    l2_weight = hp.Choice("l2_weight", [1e-5, 1e-4, 1e-3]) if use_l2 else 0
    dense_units = hp.Choice("dense_units", [64, 128, 256])
    dense_dropout = hp.Float("dense_dropout", 0.2, 0.6, step=0.1)
    use_global_avg = hp.Boolean("use_global_avg")
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    l2 = regularizers.l2(l2_weight) if use_l2 else None
    inputs = keras.Input(shape=input_shape, name="spectrogram_input")
    x = inputs

    for i in range(num_blocks):
        filters = hp.Int(f"filters_block_{i}", 32, 256, step=32)
        sd = hp.Float(f"spatial_dropout_{i}", 0.0, 0.3, step=0.05)
        x = layers.Conv2D(filters, (kernel_size, kernel_size), padding="same", kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        if sd > 0:
            x = layers.SpatialDropout2D(sd)(x)

    x = layers.GlobalAveragePooling2D()(x) if use_global_avg else layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------- #
#                     BUILD MODEL DARI JSON CONFIG                             #
# --------------------------------------------------------------------------- #
def build_model_from_config(config):
    input_shape = _CURRENT_STFT_CONFIG.get("shape", (64, 61, 1))
    use_l2 = config.get("use_l2", True)
    l2_weight = config.get("l2_weight", 1e-4) if use_l2 else 0
    l2 = regularizers.l2(l2_weight) if use_l2 else None

    inputs = keras.Input(shape=input_shape, name="spectrogram_input")
    x = inputs
    for i in range(config["num_blocks"]):
        filters = config[f"filters_block_{i}"]
        sd = config.get(f"spatial_dropout_{i}", 0.0)
        x = layers.Conv2D(filters, (config["kernel_size"], config["kernel_size"]),
                          padding="same", kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        if sd > 0:
            x = layers.SpatialDropout2D(sd)(x)

    if config.get("use_global_avg", True):
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.Flatten()(x)

    x = layers.Dense(config["dense_units"], activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(config["dense_dropout"])(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="2D_CNN_HPO_Best")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------- #
#                           HPO PIPELINE                                       #
# --------------------------------------------------------------------------- #
def run_hpo(args):
    print("=" * 60)
    print("  HPO 2D-CNN (Spectrogram) — Bayesian Optimization")
    print("=" * 60)

    # STFT config
    stft_key = f"{args.nperseg}_{args.hop_length}"
    if stft_key in STFT_CONFIGS:
        stft_cfg = STFT_CONFIGS[stft_key]
    else:
        tf_count = len(np.arange(0, 1024 - args.nperseg + 1, args.hop_length))
        stft_cfg = {"nperseg": args.nperseg, "hop_length": args.hop_length,
                     "shape": (args.nperseg, tf_count, 1)}

    global _CURRENT_STFT_CONFIG
    _CURRENT_STFT_CONFIG = stft_cfg
    print(f"\n[INFO] STFT: nperseg={args.nperseg}, hop={args.hop_length}, shape={stft_cfg['shape']}")

    # Data
    train_idx, val_idx, _ = create_train_val_test_split(hdf5_path=HDF5_FILE, max_samples=args.max_samples)
    base_train = JammingDataGenerator(indices=train_idx, batch_size=args.batch_size, shuffle=True)
    base_val = JammingDataGenerator(indices=val_idx, batch_size=args.batch_size, shuffle=False, seed=123)
    train_gen = HPOSpectrogramGenerator(base_train, args.nperseg, args.hop_length)
    val_gen = HPOSpectrogramGenerator(base_val, args.nperseg, args.hop_length)
    print(f"[INFO] Train: {len(train_gen)} batches, Val: {len(val_gen)} batches")

    # ==================== RETRAIN-ONLY MODE ==================== #
    if args.retrain_only:
        hp_path = os.path.join(RESULTS_DIR, "hpo_best_2dcnn.json")
        if not os.path.exists(hp_path):
            print(f"[ERROR] {hp_path} tidak ditemukan! Jalankan HPO search dulu.")
            sys.exit(1)

        with open(hp_path) as f:
            best_config = json.load(f)

        print("\n[MODE] RETRAIN-ONLY — Skip search, langsung retrain.")
        for k, v in sorted(best_config.items()):
            print(f"   {k}: {v}")

        model = build_model_from_config(best_config)
        model.summary()

        cbs = [
            keras.callbacks.ModelCheckpoint(os.path.join(MODELS_DIR, "best_2dcnn_hpo.keras"),
                                           monitor="val_accuracy", save_best_only=True, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ]

        history = model.fit(train_gen, validation_data=val_gen,
                            epochs=args.retrain_epochs, callbacks=cbs, verbose=1)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "2dcnn_hpo_history.json"), "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
        print("\n[OK] Re-training 2D-CNN selesai!")
        return

    # ==================== HPO SEARCH MODE ==================== #
    print(f"[INFO] Max trials: {args.max_trials}")
    os.makedirs(HPO_DIR, exist_ok=True)

    tuner = kt.BayesianOptimization(
        build_model, objective="val_accuracy", max_trials=args.max_trials,
        num_initial_points=min(5, args.max_trials),
        directory=HPO_DIR, project_name=f"bayesian_stft_{stft_key}", overwrite=args.overwrite,
    )
    tuner.search_space_summary()

    trial_cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0),
    ]

    print(f"\n>> Memulai search ({args.max_trials} trials)...\n")
    tuner.search(train_gen, validation_data=val_gen, epochs=args.epochs_per_trial,
                 callbacks=trial_cbs, verbose=1)

    tuner.results_summary(num_trials=5)
    best_config = tuner.get_best_hyperparameters(1)[0].values
    best_config["stft_nperseg"] = args.nperseg
    best_config["stft_hop_length"] = args.hop_length

    print("\n[BEST HP]")
    for k, v in sorted(best_config.items()):
        print(f"   {k}: {v}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "hpo_best_2dcnn.json"), "w") as f:
        json.dump(best_config, f, indent=2)

    print("\n[OK] HPO 2D-CNN selesai!")


def parse_args():
    p = argparse.ArgumentParser(description="HPO 2D-CNN")
    p.add_argument("--max-trials", type=int, default=30)
    p.add_argument("--epochs-per-trial", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=500000)
    p.add_argument("--nperseg", type=int, default=64)
    p.add_argument("--hop-length", type=int, default=16)
    p.add_argument("--retrain-only", action="store_true", help="SKIP search, langsung retrain dari JSON")
    p.add_argument("--retrain-epochs", type=int, default=40)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        args.max_trials = 2
        args.epochs_per_trial = 2
        args.max_samples = 10000
    run_hpo(args)
