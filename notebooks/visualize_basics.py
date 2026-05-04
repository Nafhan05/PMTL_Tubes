"""Visualisasi untuk menjawab pertanyaan dasar proyek."""
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAVE_DIR = r"C:\Users\Acer\.gemini\antigravity\brain\16a951e0-5913-41db-a5c7-3523fccfa3a2"
HDF5 = r"D:\Coding\College Coding vibes\PMTL_Tubes\data\GOLD_XYZ_OSC.0001_1024.hdf5"

# ====================================================================
# PLOT 1: Bentuk data mentah — 1 sample sebagai matriks
# ====================================================================
with h5py.File(HDF5, "r") as f:
    X_all = f["X"]
    Y_all = f["Y"]
    # Ambil 1 sample
    sample = X_all[0]  # shape (1024, 2)
    label = Y_all[0]   # one-hot
    
    # Info dataset
    total_samples = X_all.shape[0]
    seq_len = X_all.shape[1]
    channels = X_all.shape[2]
    print(f"Dataset shape: {X_all.shape}")
    print(f"Total samples: {total_samples:,}")
    print(f"Seq length: {seq_len}")
    print(f"Channels: {channels}")
    print(f"Label shape: {Y_all.shape}")
    print(f"Sample 0 label: {label}")
    print(f"Sample 0 data (first 10 rows):")
    print(sample[:10])

fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1.5, 1.5], hspace=0.4, wspace=0.4)

# A: Matriks angka mentah (hanya 20 baris pertama)
ax_matrix = fig.add_subplot(gs[:, 0])
ax_matrix.set_title("Data Mentah 1 Sample\n(20 baris pertama dari 1024)", fontsize=12, fontweight="bold")
table_data = []
for i in range(20):
    table_data.append([f"{sample[i, 0]:.6f}", f"{sample[i, 1]:.6f}"])
table_data.append(["...", "..."])
table_data.append([f"{sample[-1, 0]:.6f}", f"{sample[-1, 1]:.6f}"])

table = ax_matrix.table(
    cellText=table_data,
    colLabels=["I (In-phase)", "Q (Quadrature)"],
    rowLabels=[str(i) for i in range(20)] + ["...", "1023"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.2)
ax_matrix.axis("off")
ax_matrix.text(0.5, -0.02, "Shape: (1024, 2)\n= 1024 titik waktu × 2 komponen (I dan Q)",
              ha="center", fontsize=10, color="blue", transform=ax_matrix.transAxes)

# B: Grafik I dan Q
ax_i = fig.add_subplot(gs[0, 1:])
ax_i.plot(sample[:, 0], linewidth=0.5, color="#2196F3")
ax_i.set_title("Komponen I (In-phase) — 1024 titik", fontsize=11, fontweight="bold")
ax_i.set_ylabel("Amplitudo")
ax_i.grid(True, alpha=0.3)

ax_q = fig.add_subplot(gs[1, 1:])
ax_q.plot(sample[:, 1], linewidth=0.5, color="#FF5722")
ax_q.set_title("Komponen Q (Quadrature) — 1024 titik", fontsize=11, fontweight="bold")
ax_q.set_xlabel("Sample Index (waktu)")
ax_q.set_ylabel("Amplitudo")
ax_q.grid(True, alpha=0.3)

plt.suptitle(f"1 SAMPLE dari Dataset DeepSig (total: {total_samples:,} sample)", fontsize=14, fontweight="bold")
plt.savefig(f"{SAVE_DIR}/q1_raw_data.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nPlot 1 saved: q1_raw_data.png")

# ====================================================================
# PLOT 2: Hubungan antar sample — time series di DALAM sample
# ====================================================================
with h5py.File(HDF5, "r") as f:
    X_all = f["X"]
    samples = X_all[:3]  # 3 sample pertama

fig, axes = plt.subplots(3, 1, figsize=(14, 9))
colors = ["#2196F3", "#4CAF50", "#FF9800"]
for i in range(3):
    axes[i].plot(samples[i, :, 0], linewidth=0.5, color=colors[i])
    axes[i].set_title(f"Sample #{i} — komponen I (1024 titik)", fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Amplitudo")
    axes[i].grid(True, alpha=0.3)
    axes[i].annotate(f"← Time series ada DI DALAM sample ini (1024 titik berurutan)",
                    xy=(512, 0), fontsize=10, color=colors[i],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
axes[2].set_xlabel("Sample Index (waktu)")
plt.suptitle("Setiap SAMPLE adalah sinyal TIME SERIES tersendiri\n(antar sample TIDAK terhubung)", 
            fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/q2_samples_relation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 2 saved: q2_samples_relation.png")

# ====================================================================
# PLOT 3: Perbandingan input 3 model
# ====================================================================
np.random.seed(42)
signal_1d = np.sin(np.linspace(0, 4*np.pi, 256)) + np.random.randn(256)*0.2

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1D-CNN input
axes[0].plot(signal_1d, color="#2196F3", linewidth=1)
axes[0].set_title("Input 1D-CNN & LSTM\n(Sinyal 1D: 256 titik × 2)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Waktu (sample index)")
axes[0].set_ylabel("Amplitudo")
axes[0].grid(True, alpha=0.3)
axes[0].annotate("Bentuk: vektor/garis\nModel melihat KIRI→KANAN", xy=(60, -0.8), fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightyellow"))

# 2D-CNN input (spectrogram)
t = np.linspace(0, 1, 256)
complex_sig = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*30*t) + np.random.randn(256)*0.3
# Manual STFT
nperseg = 32
hop = 8
n_frames = (256 - nperseg) // hop + 1
window = np.hanning(nperseg)
frames = np.array([complex_sig[i*hop:i*hop+nperseg] * window for i in range(n_frames)])
spec = np.abs(np.fft.fft(frames, axis=1))
log_spec = np.log10(spec + 1e-10)

axes[1].imshow(log_spec.T, aspect="auto", cmap="viridis", origin="lower")
axes[1].set_title("Input 2D-CNN\n(Spektrogram: frekuensi × waktu)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Waktu (frame)")
axes[1].set_ylabel("Frekuensi (bin)")
axes[1].annotate("Bentuk: gambar 2D\nModel melihat POLA WARNA", xy=(2, 25), fontsize=10, color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

# Perbedaan cara melihat
axes[2].axis("off")
comparison_text = """
CARA MODEL MELIHAT DATA:

🔵 1D-CNN:
   Melihat potongan kecil sinyal
   (filter geser kiri→kanan)
   "Ada spike di sini?"

🟢 LSTM:
   Membaca sinyal berurutan
   dari awal sampai akhir
   "Bagaimana alur perubahan-nya?"

🟣 2D-CNN:
   Melihat gambar spektrogram
   secara keseluruhan
   "Pola warna apa yang muncul?"
"""
axes[2].text(0.1, 0.95, comparison_text, fontsize=13, fontfamily="monospace",
            va="top", transform=axes[2].transAxes,
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.9))

plt.suptitle("Perbandingan Input 3 Model", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/q4_model_inputs.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 3 saved: q4_model_inputs.png")

# ====================================================================
# PLOT 4: Arsitektur & cara kerja CNN vs LSTM
# ====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# CNN cara kerja
ax = axes[0]
ax.set_title("1D-CNN: Filter Geser (Sliding Window)", fontsize=13, fontweight="bold")
signal = np.sin(np.linspace(0, 2*np.pi, 50)) + np.random.randn(50)*0.1
ax.plot(signal, 'b-', linewidth=1.5)
# Highlight filter windows
for start, color in [(5, '#FF5722'), (12, '#FF9800'), (19, '#FFC107')]:
    ax.axvspan(start, start+7, alpha=0.3, color=color)
    ax.annotate(f"Filter\n(size=7)", xy=(start+3, signal[start+3]+0.3), 
               ha="center", fontsize=9, color=color,
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.set_xlabel("Waktu")
ax.set_ylabel("Amplitudo")
ax.grid(True, alpha=0.2)
ax.text(0.5, -0.15, "CNN: Melihat POTONGAN KECIL sinyal\nLalu geser ke kanan → cari pola lokal",
       ha="center", fontsize=11, transform=ax.transAxes, color="red",
       bbox=dict(boxstyle="round", facecolor="mistyrose"))

# LSTM cara kerja
ax = axes[1]
ax.set_title("LSTM: Membaca Berurutan (Sequential)", fontsize=13, fontweight="bold")
ax.plot(signal, 'g-', linewidth=1.5)
# Arrows showing sequential reading
for i in range(0, 45, 5):
    ax.annotate("", xy=(i+5, signal[min(i+5,49)]), xytext=(i, signal[i]),
               arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=2))
ax.scatter([0], [signal[0]], s=100, color="green", zorder=5, label="Mulai")
ax.scatter([49], [signal[49]], s=100, color="red", zorder=5, label="Selesai")
ax.legend(fontsize=10)
ax.set_xlabel("Waktu")
ax.set_ylabel("Amplitudo")
ax.grid(True, alpha=0.2)
ax.text(0.5, -0.15, "LSTM: Membaca dari AWAL sampai AKHIR\nMengingat konteks → paham alur perubahan",
       ha="center", fontsize=11, transform=ax.transAxes, color="green",
       bbox=dict(boxstyle="round", facecolor="honeydew"))

plt.suptitle("Perbedaan Cara 1D-CNN vs LSTM Melihat Data", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/q4_cnn_vs_lstm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 4 saved: q4_cnn_vs_lstm.png")

print("\nSemua visualisasi berhasil dibuat!")
