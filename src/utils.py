"""
utils.py — Kumpulan fungsi pembantu untuk proyek Deteksi Wireless Jamming.
Berisi fungsi visualisasi sinyal, metrik evaluasi, dan benchmarking latency.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import time
import os

# --------------------------------------------------------------------------- #
#                          KONSTANTA & LABEL                                   #
# --------------------------------------------------------------------------- #
CLASS_NAMES = ["Normal", "CW_Jamming", "Barrage_Jamming"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


# --------------------------------------------------------------------------- #
#                        VISUALISASI SINYAL                                    #
# --------------------------------------------------------------------------- #
def plot_time_domain(signal: np.ndarray, title: str = "Time Domain", save_path: str | None = None):
    """
    Plot sinyal I/Q di domain waktu.

    Args:
        signal: Array shape (N, 2) — kolom 0 = In-phase, kolom 1 = Quadrature.
        title: Judul plot.
        save_path: Jika diberikan, simpan gambar ke path ini.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(signal[:, 0], linewidth=0.5, color="#2196F3")
    axes[0].set_ylabel("In-phase (I)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(signal[:, 1], linewidth=0.5, color="#FF5722")
    axes[1].set_ylabel("Quadrature (Q)")
    axes[1].set_xlabel("Sample Index")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_constellation(signal: np.ndarray, title: str = "Constellation Diagram", save_path: str | None = None):
    """
    Plot diagram konstelasi I vs Q.

    Args:
        signal: Array shape (N, 2).
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(signal[:, 0], signal[:, 1], s=1, alpha=0.4, color="#9C27B0")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_psd(signal: np.ndarray, fs: float = 1.0, title: str = "Power Spectral Density", save_path: str | None = None):
    """
    Plot Power Spectral Density dari sinyal I/Q (menggunakan komponen kompleks I + jQ).

    Args:
        signal: Array shape (N, 2).
        fs: Sampling frequency (default 1.0 — normalized).
    """
    complex_signal = signal[:, 0] + 1j * signal[:, 1]
    plt.figure(figsize=(10, 4))
    plt.psd(complex_signal, NFFT=256, Fs=fs, color="#4CAF50")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------- #
#                          METRIK EVALUASI                                     #
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Hitung metrik klasifikasi komprehensif.

    Returns:
        Dictionary berisi accuracy, precision, recall, f1 (macro & per-kelas).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
):
    """
    Plot confusion matrix sebagai heatmap.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None):
    """Cetak classification report dari sklearn."""
    if class_names is None:
        class_names = CLASS_NAMES
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


def plot_classification_matrix(
    metrics: dict,
    model_name: str,
    class_names: list[str] | None = None,
    save_path: str | None = None,
):
    """
    Plot presisi, recall, dan f1-score per kelas sebagai heatmap.
    """
    if class_names is None:
        class_names = CLASS_NAMES
        
    metrics_names = ["Precision", "Recall", "F1-Score"]
    
    # Buat matriks 3x3 (Baris: Kelas, Kolom: Metrik)
    matrix = np.zeros((3, 3))
    matrix[:, 0] = metrics["precision_per_class"]
    matrix[:, 1] = metrics["recall_per_class"]
    matrix[:, 2] = metrics["f1_per_class"]
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        xticklabels=metrics_names,
        yticklabels=class_names,
        vmin=0.90, vmax=1.00
    )
    plt.title(f"{model_name} — Classification Report Matrix")
    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close() # Hindari plotting popup yang mengganggu CLI
    else:
        plt.show()


# --------------------------------------------------------------------------- #
#                       BENCHMARK INFERENCE LATENCY                            #
# --------------------------------------------------------------------------- #
def measure_latency(model, x_sample: np.ndarray, n_runs: int = 100) -> dict:
    """
    Ukur inference latency model.

    Args:
        model: Model Keras yang sudah di-load.
        x_sample: Satu sample input, shape (1, 1024, 2).
        n_runs: Jumlah iterasi pengukuran.

    Returns:
        Dictionary berisi avg_ms, min_ms, max_ms.
    """
    # Warm-up
    for _ in range(5):
        model.predict(x_sample, verbose=0)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(x_sample, verbose=0)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    return {
        "avg_ms": np.mean(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "std_ms": np.std(latencies),
    }


# --------------------------------------------------------------------------- #
#                          PLOTTING TRAINING HISTORY                           #
# --------------------------------------------------------------------------- #
def plot_training_history(history, model_name: str = "Model", save_path: str | None = None):
    """
    Plot training & validation loss/accuracy curves.

    Args:
        history: Keras History object atau dict dengan keys 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
        model_name: Nama model untuk judul.
    """
    h = history.history if hasattr(history, "history") else history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(h["loss"], label="Train Loss", color="#2196F3")
    axes[0].plot(h["val_loss"], label="Val Loss", color="#FF5722")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(h["accuracy"], label="Train Acc", color="#4CAF50")
    axes[1].plot(h["val_accuracy"], label="Val Acc", color="#FF9800")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
