"""
1_data_generator.py — Pipeline Injeksi Jamming Sintetis & Keras Data Generator.

Modul ini berisi:
- Fungsi injeksi CW Jamming (gelombang sinus konstan)
- Fungsi injeksi Barrage Jamming (Gaussian noise wideband)
- JammingDataGenerator: Keras Sequence yang membaca HDF5 secara lazy
  dan meng-inject jamming on-the-fly per batch.
"""

import numpy as np
import h5py
import tensorflow as tf
import os

# --------------------------------------------------------------------------- #
#                              KONFIGURASI                                     #
# --------------------------------------------------------------------------- #
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
HDF5_FILE = os.path.join(DATA_DIR, "GOLD_XYZ_OSC.0001_1024.hdf5")

# Rentang Signal-to-Jamming Ratio (dB). Semakin rendah SJR, semakin kuat jamming.
SJR_MIN_DB = -10.0
SJR_MAX_DB = 10.0

# Label kelas
LABEL_NORMAL = 0
LABEL_CW = 1
LABEL_BARRAGE = 2
NUM_CLASSES = 3


# --------------------------------------------------------------------------- #
#                         FUNGSI INJEKSI JAMMING                               #
# --------------------------------------------------------------------------- #
def inject_cw_jamming(signal: np.ndarray, sjr_db: float) -> np.ndarray:
    """
    Injeksi Continuous Wave (CW) Jamming ke sinyal I/Q.

    CW Jamming = gelombang sinusoidal konstan pada frekuensi acak.

    Args:
        signal: Sinyal I/Q asli, shape (N, 2).
        sjr_db: Signal-to-Jamming Ratio dalam dB.

    Returns:
        Sinyal terkontaminasi CW Jamming, shape (N, 2).
    """
    n_samples = signal.shape[0]

    # Hitung power sinyal asli
    signal_power = np.mean(signal ** 2)

    # Hitung power jamming dari SJR
    sjr_linear = 10 ** (sjr_db / 10)
    jam_power = signal_power / sjr_linear

    # Generate CW: sinusoidal dengan frekuensi acak & fase acak
    freq = np.random.uniform(0.01, 0.5)  # normalized frequency
    phase = np.random.uniform(0, 2 * np.pi)
    t = np.arange(n_samples)

    cw_i = np.cos(2 * np.pi * freq * t + phase)
    cw_q = np.sin(2 * np.pi * freq * t + phase)
    cw = np.stack([cw_i, cw_q], axis=-1)

    # Skala CW ke power yang diinginkan
    cw_current_power = np.mean(cw ** 2)
    if cw_current_power > 0:
        scale = np.sqrt(jam_power / cw_current_power)
        cw = cw * scale

    return signal + cw


def inject_barrage_jamming(signal: np.ndarray, sjr_db: float) -> np.ndarray:
    """
    Injeksi Barrage Jamming ke sinyal I/Q.

    Barrage Jamming = Gaussian noise wideband yang menutupi seluruh bandwidth.

    Args:
        signal: Sinyal I/Q asli, shape (N, 2).
        sjr_db: Signal-to-Jamming Ratio dalam dB.

    Returns:
        Sinyal terkontaminasi Barrage Jamming, shape (N, 2).
    """
    # Hitung power sinyal asli
    signal_power = np.mean(signal ** 2)

    # Hitung power jamming dari SJR
    sjr_linear = 10 ** (sjr_db / 10)
    jam_power = signal_power / sjr_linear

    # Generate Gaussian noise
    noise = np.random.randn(*signal.shape)

    # Skala noise ke power yang diinginkan
    noise_power = np.mean(noise ** 2)
    if noise_power > 0:
        scale = np.sqrt(jam_power / noise_power)
        noise = noise * scale

    return signal + noise


# --------------------------------------------------------------------------- #
#                        KERAS DATA GENERATOR                                  #
# --------------------------------------------------------------------------- #
class JammingDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence generator untuk dataset jamming sintetis.

    Membaca frame I/Q dari HDF5 secara lazy (batch-by-batch),
    kemudian meng-inject CW atau Barrage jamming secara acak on-the-fly.
    Setiap batch terdiri dari campuran sinyal Normal, CW, dan Barrage.

    Args:
        hdf5_path: Path ke file HDF5.
        indices: Array indeks frame yang akan digunakan (untuk train/val/test split).
        batch_size: Ukuran batch.
        sjr_range: Tuple (min_sjr_db, max_sjr_db).
        normal_ratio: Proporsi sample Normal dalam setiap batch (default 1/3).
        shuffle: Acak urutan indeks setiap epoch.
        seed: Jika diberikan, gunakan seed tetap untuk injeksi jamming.
               Penting untuk validation/test agar konsisten antar epoch.
    """

    def __init__(
        self,
        hdf5_path: str = HDF5_FILE,
        indices: np.ndarray | None = None,
        batch_size: int = 64,
        sjr_range: tuple = (SJR_MIN_DB, SJR_MAX_DB),
        normal_ratio: float = 1 / 3,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.sjr_range = sjr_range
        self.normal_ratio = normal_ratio
        self.shuffle = shuffle
        self.seed = seed

        # Buka HDF5 untuk mendapatkan jumlah total frame
        with h5py.File(self.hdf5_path, "r") as f:
            # Dataset DeepSig: key 'X' berisi sinyal (N, 1024, 2)
            self.total_frames = f["X"].shape[0]

        if indices is not None:
            self.indices = indices.copy()
        else:
            self.indices = np.arange(self.total_frames)

        self.on_epoch_end()

    def __len__(self) -> int:
        """Jumlah batch per epoch."""
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx: int):
        """
        Ambil satu batch data.

        Returns:
            (X_batch, y_batch) — X shape (B, 1024, 2), y shape (B,) integer labels.
        """
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        actual_batch_size = len(batch_indices)

        # Baca sinyal dari HDF5
        with h5py.File(self.hdf5_path, "r") as f:
            # Sort indices untuk HDF5 fancy indexing (harus terurut)
            sorted_idx = np.sort(batch_indices)
            X_batch = f["X"][sorted_idx]  # (B, 1024, 2)

        # Gunakan RNG deterministik untuk val/test, atau random untuk training
        if self.seed is not None:
            rng = np.random.RandomState(self.seed + idx)
        else:
            rng = np.random.RandomState()  # truly random

        # Tentukan label untuk setiap sample dalam batch
        y_batch = np.zeros(actual_batch_size, dtype=np.int32)
        normal_count = int(actual_batch_size * self.normal_ratio)
        jam_count = actual_batch_size - normal_count
        cw_count = jam_count // 2
        barrage_count = jam_count - cw_count

        # Acak urutan assignment
        assignment = rng.permutation(actual_batch_size)

        # Inject CW Jamming
        for i in assignment[:cw_count]:
            sjr = rng.uniform(*self.sjr_range)
            X_batch[i] = inject_cw_jamming(X_batch[i], sjr)
            y_batch[i] = LABEL_CW

        # Inject Barrage Jamming
        for i in assignment[cw_count : cw_count + barrage_count]:
            sjr = rng.uniform(*self.sjr_range)
            X_batch[i] = inject_barrage_jamming(X_batch[i], sjr)
            y_batch[i] = LABEL_BARRAGE

        # Sisanya tetap Normal (label 0, sudah default)

        return X_batch.astype(np.float32), y_batch

    def on_epoch_end(self):
        """Dipanggil di akhir setiap epoch — acak indeks jika shuffle=True."""
        if self.shuffle:
            np.random.shuffle(self.indices)


# --------------------------------------------------------------------------- #
#                          FUNGSI PEMBANTU SPLIT                               #
# --------------------------------------------------------------------------- #
def create_train_val_test_split(
    hdf5_path: str = HDF5_FILE,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Buat split train/val/test berdasarkan indeks.

    Args:
        hdf5_path: Path ke file HDF5.
        train_ratio, val_ratio, test_ratio: Proporsi split.
        max_samples: Jika diberikan, batasi jumlah total sample (untuk debugging).
        seed: Random seed untuk reprodusibilitas.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    with h5py.File(hdf5_path, "r") as f:
        total = f["X"].shape[0]

    if max_samples is not None:
        total = min(total, max_samples)

    rng = np.random.RandomState(seed)
    all_indices = rng.permutation(total)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train : n_train + n_val]
    test_idx = all_indices[n_train + n_val :]

    print(f"[Split] Total: {total} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
#                              QUICK TEST                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 60)
    print("Data Generator — Quick Smoke Test")
    print("=" * 60)

    if not os.path.exists(HDF5_FILE):
        print(f"\n[ERROR] Dataset tidak ditemukan: {HDF5_FILE}")
        print("Silakan download GOLD_XYZ_OSC.0001_1024.hdf5 dan letakkan di folder data/")
    else:
        train_idx, val_idx, test_idx = create_train_val_test_split(max_samples=1000)
        gen = JammingDataGenerator(indices=train_idx, batch_size=32)
        X, y = gen[0]
        print(f"\nBatch shape: X={X.shape}, y={y.shape}")
        print(f"Label distribution: {np.bincount(y, minlength=NUM_CLASSES)}")
        print(f"  Normal: {(y == 0).sum()}, CW: {(y == 1).sum()}, Barrage: {(y == 2).sum()}")
        print("\n[OK] Data Generator berfungsi dengan baik!")
